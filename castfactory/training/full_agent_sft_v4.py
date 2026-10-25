"""Native four-role SFT consumption and explicitly configured full-weight training.

The collector's rows remain immutable. Accepted T1 review provenance is offline
metadata; only recorded public role messages enter the model. No pilot quotas,
teacher requests, LoRA adapters, or implied approved training recipe are used.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import MISSING, asdict, dataclass
import json
import math
from pathlib import Path

from castfactory.agentic import distillation_protocol as protocol
from castfactory.agentic.full_agent_v4 import ROLES, sft_rows
from castfactory.agentic.json_rows_protocol import VERSION as JSON_ROWS_VERSION, validate_sft_row
from castfactory.agentic.compact_protocol import VERSION as COMPACT_VERSION, STEP_TARGET_VERSION
from castfactory.training.full_agent_sft_execution import (
    EpisodeExecutionMixin, ExactEpisodeBatchSampler, role_token_loss_sum, weighted_episode_roles,
)

VERSION = protocol.VERSION
COMPACT_VERSIONS = (COMPACT_VERSION, STEP_TARGET_VERSION)
SUPPORTED_VERSIONS = (VERSION, JSON_ROWS_VERSION, *COMPACT_VERSIONS)

def validate_offline_root_demonstration(row):
    raise ValueError("Private offline corpus repair provenance is not distributed; use original teacher demonstrations")

def validate_deadline_root_review(row, original=None):
    if row.get("serialization_provenance", {}).get("root_review_provenance") is not None:
        raise ValueError("Private offline review assets are not distributed")
    return original

def adapt_t1_public_policy(row):
    """Project old exported T1 capacity fields; archive and targets stay exact."""
    if row.get("role") != "T1":
        return row
    from castfactory.agentic.full_agent_v4 import public_policy, PUBLIC_POLICY_VERSION
    adapted = deepcopy(row)
    marker, end = "Compute limits for this plan:\n", "\n\nForecast evidence:"
    for message in adapted["messages"]:
        if message.get("role") != "user" or marker not in message.get("content", ""):
            continue
        before, _, tail = message["content"].partition(marker)
        policy_text, separator, after = tail.partition(end)
        if not separator:
            raise ValueError("T1 public policy must use the current natural task template")
        policy = json.loads(policy_text)
        filtered = public_policy(policy)
        message["content"] = before + marker + json.dumps(filtered, ensure_ascii=False, allow_nan=False,
                                                          separators=(",", ":")) + end + after
        provenance = adapted.setdefault("serialization_provenance", {})
        if policy != filtered or "public_policy_adapter" not in provenance:
            provenance["public_policy_adapter"] = {
                "kind": "deterministic_public_policy_projection", "version": PUBLIC_POLICY_VERSION,
                "omitted_fields": sorted(set(policy) - set(filtered))}
            if policy != filtered and not row.get("teacher_messages"):
                provenance["public_policy_adapter"]["original_student_messages"] = deepcopy(row["messages"])
        return adapted
    raise ValueError("T1 row is missing its public compute policy template")


def encode_row(row, tokenizer, max_tokens=None):
    """Supervise the native continuation, masking every public prompt token."""
    if (row.get("profile_version") not in SUPPORTED_VERSIONS or row.get("split") != "train"
            or row.get("response_format") != "student" or row.get("role") not in ROLES
            or row.get("target_source") not in ("teacher_demonstration", "offline_root_demonstration")):
        raise ValueError("Expected an explicitly supported four-role train teacher student row")
    if row["target_source"] == "offline_root_demonstration":
        validate_offline_root_demonstration(row)
    elif (row.get("serialization_provenance") or {}).get("root_fallback_provenance") is not None:
        raise ValueError("Root fallback cannot be relabelled as teacher demonstration")
    elif ((row.get("serialization_provenance") or {}).get("backend_provenance") or {}).get("backend") == "root_direct":
        raise ValueError("Root direct source cannot be relabelled as a teacher demonstration")
    if not isinstance(row.get("case_id"), str) or row.get("episode_id") != row["case_id"]:
        raise ValueError("SFT row requires matching case/episode identity")
    if row.get("source_call_id") is None:
        raise ValueError("SFT row requires its actual source call")
    if row["role"] == "T1":
        validate_deadline_root_review(row)
        review = row.get("t1_supervision") or {}
        if (review.get("kind") != "label_informed" or review.get("decision") not in ("KEEP", "REWRITE")
                or review.get("pass1_call_id") != row["source_call_id"]
                or not review.get("review_record") or row.get("derived") is not True):
            raise ValueError("T1 requires complete label-informed review provenance")
    elif row.get("derived") is not False:
        raise ValueError("Downstream roles must preserve actual teacher outputs")
    if max_tokens is not None and (type(max_tokens) is not int or max_tokens < 1):
        raise ValueError("max_tokens must be a positive explicit limit")
    if row["profile_version"] == JSON_ROWS_VERSION:
        validate_sft_row(row)
    elif row["profile_version"] in COMPACT_VERSIONS:
        from castfactory.agentic.compact_protocol import validate_sft_row as validate_compact
        validate_compact(row)
    else:
        row = adapt_t1_public_policy(row)
    if row["profile_version"] in COMPACT_VERSIONS:
        from castfactory.training.full_agent_serialization import student_training_tokens
        prompt, response = student_training_tokens(tokenizer, row["messages"], row["assistant_response"],
                                                   profile_version=row["profile_version"])
    else:
        prompt, response = protocol.student_training_tokens(tokenizer, row["messages"], row["assistant_response"])
    if not response:
        raise ValueError("No native response supervision")
    if max_tokens is not None and len(prompt) + len(response) > max_tokens:
        raise ValueError("SFT truncation forbidden: selected context limit exceeded")
    return {"profile_version": row["profile_version"], "role": row["role"], "case_id": row["case_id"], "episode_id": row["episode_id"],
            "source_call_id": row["source_call_id"],
            "row_id": row.get("row_id", row["case_id"] + ":" + row["role"] + ":" + str(row["source_call_id"])),
            "input_ids": prompt + response, "attention_mask": [1] * (len(prompt) + len(response)),
            "labels": [-100] * len(prompt) + response, "prompt_tokens": len(prompt),
            "response_tokens": len(response), "supervised_tokens": len(response),
            "serialization_provenance": deepcopy(row.get("serialization_provenance", {}))}


def _jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def prompt_adapter_summary(features):
    adapted = [row.get("serialization_provenance", {}).get("public_policy_adapter") for row in features]
    adapted = [record for record in adapted if record]
    return {"versions": dict(Counter(record["version"] for record in adapted)),
            "rows_with_capacity_projection": sum(bool(record.get("omitted_fields")) for record in adapted),
            "original_student_requests_archived": sum("original_student_messages" in record for record in adapted)}


def _complete_rows(rows):
    grouped = defaultdict(list)
    seen = set()
    for row in rows:
        key = (row.get("case_id"), row.get("role"), row.get("source_call_id"))
        if key in seen:
            raise ValueError("Duplicate role-local source call")
        seen.add(key)
        grouped[row.get("case_id")].append(row)
    if not grouped:
        raise ValueError("No completed teacher role rows")
    profiles = {row.get("profile_version") for row in rows}
    if len(profiles) != 1 or not profiles.issubset(SUPPORTED_VERSIONS):
        raise ValueError("A release must contain one explicitly supported SFT profile")
    modes = Counter()
    for case_id, calls in grouped.items():
        front = [r for r in calls if r.get("role") == "T1"]
        counts = Counter(r.get("role") for r in calls)
        if len(front) != 1 or not counts["T2"] or not counts["T3"]:
            raise ValueError("Incomplete four-role trajectory: " + str(case_id))
        mode = protocol.parse_reply(front[0]["assistant_response"], "student")["answer"].get("mode")
        if mode not in ("DIRECT", "FACTORIZE") or counts["T4"] != (1 if mode == "FACTORIZE" else 0):
            raise ValueError("T4 coverage must follow actual DIRECT/FACTORIZE plan")
        modes[mode] += 1
    return grouped, modes


def load_release(directory, tokenizer, max_tokens=None, *, profile_version=VERSION):
    """Read supplied JSONL/manifest or completed collector episodes, without writes.

    A manifest may specify row_file, profile_version, role_rows and complete.
    Counts are verified when supplied; an ongoing collector's episodes are a
    partial corpus and that fact is retained rather than called a full release.
    """
    if profile_version not in SUPPORTED_VERSIONS:
        raise ValueError("Unsupported explicit SFT profile")
    path = Path(directory)
    manifest, skipped = {}, 0
    if path.is_dir():
        summary = path / "summary.json"
        if summary.exists():
            manifest = json.loads(summary.read_text())
        if profile_version != VERSION and not manifest:
            raise ValueError("Converted release directory requires its completed materialization summary")
        row_name = {JSON_ROWS_VERSION: "agent_sft_json_rows_v1.jsonl",
                    COMPACT_VERSION: "agent_sft_compact_v1.jsonl",
                    STEP_TARGET_VERSION: "agent_sft_compact_v2.jsonl"}.get(profile_version, "agent_sft_v4.jsonl")
        if (path / row_name).exists():
            rows = _jsonl(path / row_name)
            source = str(path / row_name)
        else:
            if profile_version != VERSION:
                raise ValueError("Converted rows require their explicitly converted release")
            rows = []
            source = str(path / "episodes")
            for artifact in sorted((path / "episodes").glob("*.json")):
                record = json.loads(artifact.read_text())
                episode = record.get("episode", record)
                if episode.get("normal_completion") is not True:
                    skipped += 1
                    continue
                rows.extend(sft_rows(episode, response_format="student"))
    elif path.suffix == ".json":
        manifest = json.loads(path.read_text())
        row_file = Path(manifest["row_file"])
        row_file = row_file if row_file.is_absolute() else path.parent / row_file
        rows, source = _jsonl(row_file), str(row_file)
    else:
        rows, source = _jsonl(path), str(path)
    if manifest and manifest.get("profile_version") != profile_version:
        raise ValueError("Manifest profile differs from explicitly selected SFT profile")
    if any(row.get("profile_version") != profile_version for row in rows):
        raise ValueError("Row profile differs from explicitly selected SFT profile")
    grouped, modes = _complete_rows(rows)
    if "role_rows" in manifest and manifest["role_rows"] != len(rows):
        raise ValueError("Manifest role-row count mismatch")
    if "complete" in manifest and manifest["complete"] != len(grouped):
        raise ValueError("Manifest completed-case count mismatch")
    encoded = [encode_row(row, tokenizer, max_tokens) for row in rows]
    return encoded, {"profile_version": profile_version, "source": source, "role_rows": len(rows),
                     "complete": len(grouped), "rows_by_role": dict(Counter(r["role"] for r in rows)),
                     "cases_by_mode": dict(modes), "skipped_incomplete_episodes": skipped,
                     "max_prompt_tokens": max(row["prompt_tokens"] for row in encoded),
                     "max_response_tokens": max(row["response_tokens"] for row in encoded),
                     "max_total_tokens": max(len(row["input_ids"]) for row in encoded),
                     "student_prompt_adapters": prompt_adapter_summary(encoded),
                     "manifest": manifest, "teacher_calls": 0,
                     "collection_complete": bool(manifest and (manifest.get("collection_complete") is True
                         if profile_version != VERSION else manifest.get("failed_or_missing") == 0))}


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int
    batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    weight_decay: float
    warmup_ratio: float
    lr_scheduler_type: str
    max_grad_norm: float
    seed: int
    context_tokens: int
    dtype: str
    gradient_checkpointing: bool
    device: str
    report_to: str
    execution_mode: str = "role_streaming"

    def __post_init__(self):
        for name in ("epochs", "batch_size", "gradient_accumulation_steps", "context_tokens"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError("Invalid explicit training setting: " + name)
        for name in ("learning_rate", "max_grad_norm"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError("Invalid explicit training setting: " + name)
        if (not math.isfinite(self.weight_decay) or self.weight_decay < 0
                or not math.isfinite(self.warmup_ratio) or not 0 <= self.warmup_ratio <= 1):
            raise ValueError("Invalid weight decay/warmup configuration")
        if self.dtype not in ("float32", "bfloat16", "float16") or self.device not in ("cpu", "cuda", "npu"):
            raise ValueError("Invalid dtype/device")
        if self.device == "cpu" and self.dtype != "float32":
            raise ValueError("CPU training requires explicitly selected float32")
        if self.execution_mode not in ("role_streaming", "reference"):
            raise ValueError("Invalid episode execution mode")
        if self.dtype == "float16":
            raise ValueError("Episode execution does not support pure float16 parameters/GradScaler")


def training_plan(features, config, world_size=1):
    if type(world_size) is not int or world_size < 1:
        raise ValueError("Invalid distributed world size")
    episodes = group_episodes(features)
    profiles = {row.get("profile_version", VERSION) for row in features}
    if len(profiles) != 1 or not profiles.issubset(SUPPORTED_VERSIONS):
        raise ValueError("Training requires a single supported SFT profile")
    micro_batches = len(ExactEpisodeBatchSampler(len(episodes), config.batch_size, world_size))
    return {"parameterization": "full", "profile_version": next(iter(profiles)), "config": asdict(config),
            "world_size": world_size, "batch_unit": "complete_teacher_episode",
            "role_rows": len(features), "cases": len(episodes),
            "micro_batches_per_epoch": micro_batches,
            "planned_optimizer_steps": config.epochs * math.ceil(micro_batches / config.gradient_accumulation_steps),
            "loss_normalization": "mean_of_complete_episode_supervised_token_means",
            "execution_mode": config.execution_mode,
            "episode_sampling": "exact_rank_local_balanced_tail_no_duplicates",
            "training_recipe_source": "explicit_caller_configuration"}


def group_episodes(features):
    grouped = {}
    for row in features:
        grouped.setdefault(row["episode_id"], []).append(row)
    return [{"episode_id": identity, "rows": rows} for identity, rows in grouped.items()]


def collate_episodes(episodes):
    """Retain complete episode denominators and independent role conversations."""
    import torch
    return {"episodes": [{"rows": [{name: torch.tensor([row[name]], dtype=torch.long)
                                     for name in ("input_ids", "attention_mask", "labels")}
                                    for row in episode["rows"]]}
                         for episode in episodes]}


def episode_normalized_loss(model, episodes):
    """Mean episode CE, using each episode's full supervised-token denominator."""
    total = None
    roles = weighted_episode_roles(episodes, 1.0 / len(episodes) if episodes else 1.0)
    for batch, count, weight in roles:
        contribution = role_token_loss_sum(model, batch, count) * weight
        total = contribution if total is None else total + contribution
    return total


class EpisodeObjectiveMixin:
    """Use actual accumulated episode counts, including uneven final batches."""

    def _get_num_items_in_batch(self, batch_samples, device):
        import torch
        count = torch.tensor(sum(len(batch["episodes"]) for batch in batch_samples), device=device)
        if self.accelerator.num_processes > 1:
            count = self.accelerator.gather(count).sum()
        return count

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        episodes = inputs["episodes"]
        loss = episode_normalized_loss(model, episodes)
        if num_items_in_batch is not None:
            # DDP averages gradients across ranks: undo that average after using
            # the true all-rank, all-microbatch episode denominator.
            loss = loss * len(episodes) * self.accelerator.num_processes / num_items_in_batch
        return (loss, {"loss": loss}) if return_outputs else loss


def collate_rows(rows, pad_token_id):
    """Right pad independent conversations; padding and prompts have no loss."""
    import torch
    width = max(len(r["input_ids"]) for r in rows)
    values = {"input_ids": [], "attention_mask": [], "labels": []}
    for row in rows:
        padding = width - len(row["input_ids"])
        values["input_ids"].append(row["input_ids"] + [pad_token_id] * padding)
        values["attention_mask"].append(row["attention_mask"] + [0] * padding)
        values["labels"].append(row["labels"] + [-100] * padding)
    return {name: torch.tensor(value, dtype=torch.long) for name, value in values.items()}


def train(args, *, tokenizer=None, backend=None):
    """Reachable full-parameter backend; dry-run loads no model or optimizer."""
    import os
    config = TrainingConfig(**{name: (getattr(args, name) if field.default is MISSING
                                    else getattr(args, name, field.default))
                               for name, field in TrainingConfig.__dataclass_fields__.items()})
    if tokenizer is None:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    features, release = load_release(args.dataset, tokenizer, config.context_tokens,
                                     profile_version=getattr(args, "profile_version", VERSION))
    plan = training_plan(features, config, int(os.environ.get("WORLD_SIZE", "1")))
    report = {**plan, "dataset": release, "model": str(args.model), "out": str(args.out),
              "status": "dry_run_validated" if args.dry_run else "training_pending"}
    if args.dry_run:
        return report
    if Path(args.out).exists():
        raise ValueError("Output exists; implicit overwrite/resume forbidden")
    result = (backend or _transformers_train)(args, config, features, tokenizer)
    report.update(result)
    return report


def read_export_receipt(path, candidate=None):
    """Read concrete full-HF reload/EOS evidence for this selected export."""
    path = Path(path).resolve()
    receipt = json.loads(path.read_text())
    if receipt.get("schema_version") == "agentic-fsdp-export-v1" or receipt.get("status") == "full_parameter_fsdp_sft_export_verified":
        raise ValueError("Native FSDP receipts belong to the cluster training release, not this HF entrypoint")
    if (receipt.get("status") not in ("full_parameter_sft_completed", "full_parameter_sft_checkpoint_export_verified")
            or receipt.get("profile_version") not in SUPPORTED_VERSIONS or receipt.get("parameterization") != "full"
            or receipt.get("all_parameters_trainable") is not True):
        raise ValueError("Expected actual full-parameter v4 SFT export receipt")
    model = Path(receipt["model_path"]).resolve()
    if (not (model / "config.json").is_file() or (model / "adapter_config.json").exists()
            or not (list(model.glob("*.safetensors")) or list(model.glob("*.bin")))):
        raise ValueError("SFT receipt must name a complete full-HF model")
    probes, parity = receipt.get("probe_ids", {}), receipt.get("export_reload_max_abs_difference", {})
    if (set(probes) != set(ROLES) or set(parity) != set(ROLES)
            or any(not isinstance(probes[r], list) or not probes[r]
                   or any(type(t) is not int or t < 0 for t in probes[r]) for r in ROLES)
            or any(type(parity[r]) not in (int, float) or not math.isfinite(parity[r])
                   or not 0 <= parity[r] <= 1e-4 for r in ROLES)):
        raise ValueError("SFT export requires actual all-role reload probes/parity")
    eos = receipt.get("generation_eos", {})
    if eos:
        generation = json.loads((model / "generation_config.json").read_text())
        if (type(eos.get("eos_token_id")) is not int or eos["eos_token_id"] < 0
                or eos.get("source") != "effective_chat_tokenizer"
                or generation.get("eos_token_id") != eos["eos_token_id"]):
            raise ValueError("SFT export generation EOS differs from its native tokenizer receipt")
    elif receipt["status"] != "full_parameter_sft_completed":
        raise ValueError("Checkpoint export requires its bound native generation EOS")
    if receipt["status"] == "full_parameter_sft_checkpoint_export_verified":
        state_path = Path(receipt["checkpoint_state_path"]).resolve()
        if state_path.parent != model:
            raise ValueError("Checkpoint receipt points to another trainer state")
        state = json.loads(state_path.read_text())
        if state.get("global_step") != receipt.get("optimizer_steps") or state.get("epoch") != receipt.get("epoch"):
            raise ValueError("Checkpoint export receipt differs from its actual trainer step")
    if candidate is not None and (model != Path(candidate["model_path"]).resolve()
            or receipt.get("candidate_id") != candidate.get("candidate_id")
            or receipt.get("optimizer_steps") != candidate.get("optimizer_step")
            or receipt.get("epoch") != candidate.get("epoch")):
        raise ValueError("Candidate and SFT export receipt differ")
    return receipt


def verify_saved_export(model, destination, probes, dtype):
    """Probe a separate reload without moving live training parameter storage.

    Checkpoint verification requires space for a second inference model. It
    never offloads/replaces live parameters or changes optimizer/DDP bindings.
    """
    import torch
    from transformers import AutoModelForCausalLM
    previous_training, device = model.training, next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        expected = {role: model(input_ids=torch.tensor([ids], device=device), use_cache=False,
                                logits_to_keep=1).logits.detach().float().cpu()
                    for role, ids in probes.items()}
    reloaded = None
    try:
        reloaded = AutoModelForCausalLM.from_pretrained(destination, local_files_only=True,
                                                       dtype=dtype).to(device).eval()
        with torch.no_grad():
            differences = {role: float((reloaded(input_ids=torch.tensor([ids], device=device), use_cache=False,
                    logits_to_keep=1).logits.float().cpu() - expected[role]).abs().max())
                           for role, ids in probes.items()}
        if any(not math.isfinite(v) or v > 1e-4 for v in differences.values()):
            raise RuntimeError("Full-role export/reload parity failed")
        return differences
    finally:
        if reloaded is not None:
            del reloaded
        if device.type == "cuda":
            torch.cuda.empty_cache()
        model.train(previous_training)


def _transformers_train(args, config, features, tokenizer):
    import torch
    from transformers import AutoModelForCausalLM, Trainer, TrainerCallback, TrainingArguments
    selected_profile = features[0].get("profile_version", VERSION)
    probes = {role: next(row["input_ids"][-64:] for row in features if row["role"] == role)
              for role in ROLES if any(row["role"] == role for row in features)}
    if set(probes) != set(ROLES):
        raise ValueError("Full-role training requires natural coverage of all T1-T4 roles")
    if config.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Configured CUDA device is unavailable")
    if config.device == "npu":
        import torch_npu  # noqa: F401
        if not torch.npu.is_available():
            raise RuntimeError("Configured NPU device is unavailable")
    model = AutoModelForCausalLM.from_pretrained(args.model, local_files_only=True,
                                               dtype=getattr(torch, config.dtype))
    model.config.use_cache = False
    parameters = list(model.parameters())
    if not parameters or any(not p.requires_grad for p in parameters):
        raise ValueError("All model parameters must be trainable; adapters/frozen roles are forbidden")
    if tokenizer.pad_token_id is None:
        raise ValueError("Selected native tokenizer requires a padding token")
    from castfactory.training.full_agent_export import bind_chat_generation_eos
    generation_eos = bind_chat_generation_eos(model, tokenizer)
    options = TrainingArguments(output_dir=str(args.out), num_train_epochs=config.epochs,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate, weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio, lr_scheduler_type=config.lr_scheduler_type,
        max_grad_norm=config.max_grad_norm, seed=config.seed,
        bf16=config.dtype == "bfloat16", fp16=config.dtype == "float16", use_cpu=config.device == "cpu",
        gradient_checkpointing=config.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False}, remove_unused_columns=False,
        ddp_broadcast_buffers=False, ddp_find_unused_parameters=False,
        save_strategy="epoch", logging_strategy="steps", logging_steps=1,
        report_to=[] if config.report_to == "none" else [config.report_to])
    class CandidateManifest(TrainerCallback):
        def on_save(self, training_args, state, control, **kwargs):
            output = Path(training_args.output_dir)
            destination = output / ("checkpoint-" + str(state.global_step))
            if torch.distributed.is_initialized():
                torch.distributed.barrier()
            differences = verify_saved_export(model, destination, probes, getattr(torch, config.dtype))
            if not state.is_world_process_zero:
                if torch.distributed.is_initialized():
                    torch.distributed.barrier()
                return
            receipt_path = destination / "export_receipt.json"
            candidate_id = "sft_step_" + str(state.global_step)
            receipt = {"status": "full_parameter_sft_checkpoint_export_verified", "profile_version": selected_profile,
                "candidate_id": candidate_id, "parameterization": "full", "all_parameters_trainable": True,
                "model_path": str(destination.resolve()), "optimizer_steps": state.global_step,
                "epoch": state.epoch, "checkpoint_state_path": str((destination / "trainer_state.json").resolve()),
                "probe_ids": probes, "export_reload_max_abs_difference": differences, "generation_eos": generation_eos,
                "config": asdict(config), "teacher_calls": 0,
                "student_prompt_adapters": prompt_adapter_summary(features),
                "loss_normalization": "mean_of_complete_episode_supervised_token_means"}
            receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
            read_export_receipt(receipt_path)
            path = output / "candidates.json"
            manifest = json.loads(path.read_text()) if path.exists() else {"profile_version": selected_profile, "candidates": []}
            manifest["candidates"].append({"candidate_id": candidate_id,
                "model_path": str(destination.resolve()), "export_receipt_path": str(receipt_path.resolve()),
                "optimizer_step": state.global_step, "epoch": state.epoch, "parameterization": "full"})
            path.write_text(json.dumps(manifest, indent=2) + "\n")
            if torch.distributed.is_initialized():
                torch.distributed.barrier()
    class EpisodeTrainer(EpisodeExecutionMixin, EpisodeObjectiveMixin, Trainer):
        def __init__(self, *values, **kwargs):
            super().__init__(*values, **kwargs)
            # Our compute_loss consumes num_items_in_batch. Trainer must not
            # also divide the loss by the number of accumulation microbatches.
            self.model_accepts_loss_kwargs = True
            self.execution_mode = config.execution_mode
    trainer = EpisodeTrainer(model=model, args=options, train_dataset=group_episodes(features),
                      data_collator=collate_episodes,
                      processing_class=tokenizer, callbacks=[CandidateManifest()])
    trained = trainer.train()
    destination = Path(args.out) / "final_hf"
    trainer.save_model(str(destination))
    trainer.accelerator.wait_for_everyone()
    differences = verify_saved_export(model, destination, probes, getattr(torch, config.dtype))
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(destination)
        report = {"status": "full_parameter_sft_completed", "profile_version": selected_profile,
                  "parameterization": "full", "teacher_calls": 0,
                  "all_parameters_trainable": True, "optimizer_steps": trainer.state.global_step,
                  "model_path": str(destination.resolve()), "metrics": trained.metrics,
                  "config": asdict(config), "probe_ids": probes,
                  "generation_eos": generation_eos,
                  "export_reload_max_abs_difference": differences,
                  "role_rows": len(features), "cases": len({r["episode_id"] for r in features}),
                  "student_prompt_adapters": prompt_adapter_summary(features),
                  "batch_unit": "complete_teacher_episode",
                  "execution_mode": config.execution_mode,
                  "episode_sampling": "exact_rank_local_balanced_tail_no_duplicates",
                  "loss_normalization": "mean_of_complete_episode_supervised_token_means",
                  "candidate_manifest": str((Path(args.out) / "candidates.json").resolve()),
                  "best_selection_status": "pending_shared_validation_evaluation"}
        (Path(args.out) / "training_report.json").write_text(json.dumps(report, indent=2) + "\n")
    return {"status": "full_parameter_sft_completed", "optimizer_steps": trainer.state.global_step,
            "all_parameters_trainable": True, "model_path": str(destination.resolve())}


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--profile-version", choices=SUPPORTED_VERSIONS, required=True)
    for name in ("model", "dataset", "out"):
        p.add_argument("--" + name, type=Path, required=True)
    for name in ("epochs", "batch_size", "gradient_accumulation_steps", "seed", "context_tokens"):
        p.add_argument("--" + name.replace("_", "-"), type=int, required=True,
                       help="Complete episodes per device" if name == "batch_size" else None)
    for name in ("learning_rate", "weight_decay", "warmup_ratio", "max_grad_norm"):
        p.add_argument("--" + name.replace("_", "-"), type=float, required=True)
    p.add_argument("--lr-scheduler-type", choices=("linear", "cosine", "constant", "constant_with_warmup"), required=True)
    p.add_argument("--dtype", choices=("float32", "bfloat16", "float16"), required=True)
    p.add_argument("--device", choices=("cpu", "cuda", "npu"), required=True)
    p.add_argument("--gradient-checkpointing", choices=("on", "off"), required=True)
    p.add_argument("--report-to", required=True, help="Explicit integration name, e.g. swanlab, or none")
    p.add_argument("--execution-mode", choices=("role_streaming", "reference"), default="role_streaming",
                   help="Role-wise graph release, or the single-device attached-graph reference")
    p.add_argument("--dry-run", action="store_true")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    args.gradient_checkpointing = args.gradient_checkpointing == "on"
    print(json.dumps(train(args), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
