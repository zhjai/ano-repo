"""Exact episode sampling and narrowly supported role-streaming Trainer execution.

Trainer owns optimizer, clipping, scheduling and checkpoints. This module only
owns the rank-local episode loader and loss/backward scheduling; it does not
select an evaluation backend or establish hardware capacity.
"""
from __future__ import annotations

from contextlib import nullcontext
import math


def role_supervised_count(batch) -> int:
    """Count actual shifted labels in one isolated, batch-one conversation."""
    labels = batch["labels"]
    if labels.ndim != 2 or labels.shape[0] != 1:
        raise ValueError("Role execution requires one isolated conversation")
    count = int((labels[:, 1:] != -100).sum())
    if count < 1:
        raise ValueError("Role-local episode has no supervised tokens")
    return count


def weighted_episode_roles(episodes, scale=1.0):
    """Validate *all* masks before a forward; return (batch, count, weight)."""
    if not episodes:
        raise ValueError("Empty episode batch")
    roles = []
    for episode in episodes:
        rows = episode["rows"]
        if not rows:
            raise ValueError("Complete episode requires supervised tokens")
        counts = [role_supervised_count(batch) for batch in rows]
        denominator = sum(counts)
        roles.extend((batch, count, scale / denominator) for batch, count in zip(rows, counts))
    return roles


def role_token_loss_sum(model, batch, count=None):
    """Summed shifted CE; always forward through the supplied wrapped policy."""
    import torch
    if count is None:
        count = role_supervised_count(batch)
    native = getattr(model, "module", model)
    if getattr(getattr(native, "config", None), "model_type", None) == "qwen3_5_text":
        positions = torch.nonzero(batch["labels"][0, 1:] != -100, as_tuple=False).flatten()
        logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                       use_cache=False, logits_to_keep=positions).logits
        targets = batch["labels"][:, positions + 1]
        return torch.nn.functional.cross_entropy(logits.float().reshape(-1, logits.shape[-1]),
                                                  targets.reshape(-1), reduction="sum")
    return model(**batch, use_cache=False).loss * count


class ExactEpisodeBatchSampler:
    """Deterministic global batches split once into nonempty rank-local batches.

    A tail of at least world_size is balanced with a difference of at most one
    episode between ranks. Smaller nonempty tails fail rather than repeat data
    or leave ranks with unequal numbers of collective steps.
    """

    def __init__(self, dataset_size: int, batch_size: int, world_size: int = 1,
                 rank: int = 0, seed: int = 0, shuffle: bool = True):
        for name, value in (("dataset_size", dataset_size), ("batch_size", batch_size),
                            ("world_size", world_size)):
            if type(value) is not int or value < 1:
                raise ValueError(name + " must be a positive integer")
        if type(rank) is not int or not 0 <= rank < world_size:
            raise ValueError("Invalid sampler rank")
        self.dataset_size, self.batch_size = dataset_size, batch_size
        self.num_processes, self.rank = world_size, rank
        self.seed, self.shuffle, self.epoch = seed, shuffle, 0
        self.drop_last = False
        tail = dataset_size % (batch_size * world_size)
        if 0 < tail < world_size:
            raise ValueError("Nonempty episode tail is smaller than world_size; exact nonempty ranks required")

    @property
    def sampler(self):
        # DataLoaderShard.set_epoch follows this protocol after Trainer wraps
        # us in accelerate.SkipBatchSampler during a mid-epoch resume.
        return self

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def __len__(self):
        return math.ceil(self.dataset_size / (self.batch_size * self.num_processes))

    def __iter__(self):
        import torch
        if self.shuffle:
            generator = torch.Generator().manual_seed(self.seed + self.epoch)
            indices = torch.randperm(self.dataset_size, generator=generator).tolist()
        else:
            indices = list(range(self.dataset_size))
        width = self.batch_size * self.num_processes
        for offset in range(0, self.dataset_size, width):
            global_batch = indices[offset:offset + width]
            quotient, remainder = divmod(len(global_batch), self.num_processes)
            start = self.rank * quotient + min(self.rank, remainder)
            size = quotient + (self.rank < remainder)
            yield global_batch[start:start + size]


def validate_execution_backend(trainer, model):
    """Reject collective schedules for which role streaming is not supported."""
    import torch
    accelerator = trainer.accelerator
    kind = str(accelerator.distributed_type).split(".")[-1]
    if kind not in ("NO", "MULTI_CPU", "MULTI_GPU"):
        raise ValueError("Episode execution supports single-device or dense text DDP only; got " + kind)
    pc = getattr(accelerator, "parallelism_config", None)
    if pc is not None and (getattr(pc, "non_data_parallel_size", 1) != 1
            or any((getattr(pc, name, 1) or 1) > 1 for name in ("cp_size", "tp_size", "sp_size"))):
        raise ValueError("Context, tensor and sequence parallelism are unsupported for role streaming")
    if getattr(trainer.args, "torch_empty_cache_steps", None) is not None:
        raise ValueError("Episode execution does not implement torch_empty_cache_steps")
    if getattr(trainer.args, "fp16", False):
        raise ValueError("Episode execution does not support float16/GradScaler")
    if (getattr(trainer, "is_fsdp_enabled", False) or getattr(trainer, "is_deepspeed_enabled", False)
            or isinstance(model, torch.nn.DataParallel)):
        raise ValueError("FSDP, DeepSpeed and DataParallel are unsupported for role streaming")
    if str(getattr(trainer.args, "optim", "")).split(".")[-1].lower() in ("lomo", "adalomo"):
        raise ValueError("Role streaming requires a Trainer-owned ordinary optimizer step")
    ddp = isinstance(model, torch.nn.parallel.DistributedDataParallel)
    if accelerator.num_processes > 1 and not ddp:
        raise ValueError("Distributed role streaming must forward through the DDP wrapper")
    if ddp:
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM
        native = model.module
        # This installed exact text class has no MTP modules (checkpoint MTP
        # keys are ignored by Transformers); do not check another model's config field.
        if (type(native) is not Qwen3_5ForCausalLM
                or native.config.model_type != "qwen3_5_text"):
            raise ValueError("DDP role streaming requires the known dense qwen3_5_text model; conditional/unused parameters unsupported")
        if model.broadcast_buffers or model.find_unused_parameters or model.static_graph:
            raise ValueError("DDP requires broadcast_buffers=False, find_unused_parameters=False, static_graph=False")
        if any(not parameter.requires_grad for parameter in native.parameters()):
            raise ValueError("Dense DDP requires all parameters trainable")
    return ddp


class EpisodeExecutionMixin:
    """Compose before EpisodeObjectiveMixin and transformers.Trainer in the MRO."""

    execution_mode = "role_streaming"

    def get_train_dataloader(self):
        from accelerate.data_loader import DataLoaderShard
        if self.train_dataset is None:
            raise ValueError("Episode training requires a sized dataset")
        if getattr(self.args, "dataloader_drop_last", False):
            raise ValueError("Exact episode sampling cannot drop the final batch")
        if getattr(self.args, "auto_find_batch_size", False):
            raise ValueError("Automatic batch-size changes are unsupported for exact episodes")
        sampler = ExactEpisodeBatchSampler(len(self.train_dataset), self.args.per_device_train_batch_size,
            world_size=self.accelerator.num_processes, rank=self.accelerator.process_index,
            seed=self.args.data_seed if self.args.data_seed is not None else self.args.seed)
        options = {"num_workers": self.args.dataloader_num_workers,
                   "pin_memory": self.args.dataloader_pin_memory}
        if options["num_workers"]:
            options["persistent_workers"] = self.args.dataloader_persistent_workers
            options["prefetch_factor"] = self.args.dataloader_prefetch_factor
        loader = DataLoaderShard(self.train_dataset, device=self.accelerator.device,
            batch_sampler=sampler, collate_fn=self.data_collator, rng_types=None, **options)
        # The sampler has already assigned rank-local episodes. prepare() must
        # never install another BatchSamplerShard over this loader.
        loader._is_accelerate_prepared = True
        return loader

    def training_step(self, model, inputs, num_items_in_batch=None):
        ddp = validate_execution_backend(self, model)
        if self.execution_mode not in ("role_streaming", "reference"):
            raise ValueError("Unknown episode execution mode")
        if (num_items_in_batch is None or not math.isfinite(float(num_items_in_batch))
                or float(num_items_in_batch) <= 0):
            raise ValueError("Episode training requires the actual global accumulated episode count")
        divisor = self.accelerator.gradient_accumulation_steps
        if type(divisor) is not int or divisor < 1:
            raise ValueError("Invalid Accelerator gradient accumulation divisor")
        model.train()
        if hasattr(self.optimizer, "train") and callable(self.optimizer.train):
            self.optimizer.train()
        inputs = self._prepare_inputs(inputs)
        scale = self.accelerator.num_processes / num_items_in_batch
        roles = weighted_episode_roles(inputs["episodes"], scale)
        if self.execution_mode == "reference":
            if ddp:
                raise ValueError("Attached-graph reference training is single-device only; use role_streaming for DDP")
            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs, num_items_in_batch=num_items_in_batch)
            self.accelerator.backward(loss * divisor)
            return loss.detach()

        # Exact model-class restriction establishes a dense architecture. Hooks
        # additionally verify actual participation on *every* role backward,
        # including checkpoint recomputation; existing accumulated .grad alone
        # cannot prove that the final role touched every parameter.
        used, handles = set(), []
        expected = {name for name, parameter in model.named_parameters() if parameter.requires_grad} if ddp else set()
        if ddp:
            for name, parameter in model.named_parameters():
                handles.append(parameter.register_hook(lambda grad, name=name: used.add(name)))
        total = None
        try:
            for index, (batch, count, weight) in enumerate(roles):
                used.clear()
                synchronize = self.accelerator.sync_gradients and index == len(roles) - 1
                context = nullcontext() if synchronize else self.accelerator.no_sync(model)
                with context:
                    with self.compute_loss_context_manager():
                        loss = role_token_loss_sum(model, batch, count) * weight
                    self.accelerator.backward(loss * divisor)
                if ddp and used != expected:
                    raise RuntimeError("Dense role parameter participation failed: " + ", ".join(sorted(expected - used)))
                detached = loss.detach()
                total = detached if total is None else total + detached
                del loss, detached
        finally:
            for handle in handles:
                handle.remove()
        return total
