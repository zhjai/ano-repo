"""Episode-mean objective registered through TrainingWorker.set_loss_fn."""
from __future__ import annotations

import torch


def agentic_sft_loss(config, model_output, data, dp_group=None):
    """Return W/B times this microbatch's complete-episode weighted NLL.

    Native verl sums micro losses/backwards and FSDP averages ranks. No extra
    microbatch, token-count, or gradient-accumulation divisor belongs here.
    """
    from verl.utils import tensordict_utils as tu

    if tu.get_non_tensor_data(data, "pad_mode", "no_padding") != "no_padding":
        raise ValueError("Agentic loss requires nested no_padding transport")
    world = tu.get_non_tensor_data(data, "dp_size", None)
    count = tu.get_non_tensor_data(data, "actual_episode_batch_size", None)
    if world is None or count is None or world <= 0 or count <= 0:
        raise ValueError("Missing actual global episode denominator")
    log_probs = model_output["log_probs"]
    weights = data["episode_weights"]
    masks = data["loss_mask"]
    if not all(x.is_nested for x in (log_probs, weights, masks)):
        raise ValueError("Expected nested role log-probs, weights and masks")
    if (not torch.equal(log_probs.offsets(), weights.offsets())
            or not torch.equal(log_probs.offsets(), masks.offsets())):
        raise ValueError("Role log-prob and target boundaries differ")
    flat = log_probs.values()
    shifted_weights = torch.roll(weights.values(), -1, 0)
    shifted_mask = torch.roll(masks.values(), -1, 0).bool()
    ends = log_probs.offsets()[1:] - 1
    shifted_weights[ends] = 0
    shifted_mask[ends] = False
    selected = flat[shifted_mask].float()
    selected_weights = shifted_weights[shifted_mask].float()
    failure = 0
    loss = None
    if selected.numel():
        if (not torch.isfinite(selected).all() or not torch.isfinite(selected_weights).all()
                or not (selected_weights > 0).all()):
            failure = 1
        else:
            loss = -(selected * selected_weights).sum() * (world / count)
    else:
        # Every dummy still executes a full forward/backward. Check output before
        # constructing graph-connected zero; IEEE 0*NaN is not a valid dummy.
        if flat.numel() == 0 or not torch.isfinite(flat).all():
            failure = 2
        else:
            loss = (flat.float() * 0.0).sum()
    if loss is not None and not torch.isfinite(loss):
        failure = 3
    # Identical equalized micro slots on all ranks: coordinate BEFORE any peer
    # enters this slot's FSDP backward. This is a scalar failure flag, not custom
    # gradient synchronization. Never return a masked/repaired invalid loss.
    status = torch.tensor(failure, device=flat.device, dtype=torch.int32)
    if torch.distributed.is_initialized() and torch.distributed.get_world_size(dp_group) > 1:
        torch.distributed.all_reduce(status, op=torch.distributed.ReduceOp.MAX, group=dp_group)
    failure = int(status.item())
    if failure:
        messages = {1: "Nonfinite supervised log-prob or invalid episode weight",
                    2: "Dummy model output must be nonempty and finite",
                    3: "Nonfinite episode objective"}
        raise FloatingPointError(messages[failure] + "; all ranks stopped before backward")
    return loss, {}
