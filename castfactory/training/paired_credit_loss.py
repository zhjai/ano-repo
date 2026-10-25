"""Joint differentiable paired routing and separately clipped call credit.

Rows already carry global macro reductions. A native microbatch may contain
complete q/local-preference pairs plus generated calls; it never renormalizes.
"""
from __future__ import annotations

import math
from castfactory.training.paired_credit_objective import NATURAL_OBJECTIVE_VERSIONS

from castfactory.training.paired_credit_objective import INTEGRATED_OBJECTIVE_VERSIONS, SEMANTIC_OBJECTIVE_VERSION, NATURAL_OBJECTIVE_VERSION

from castfactory.training.paired_credit_objective import (
    INTEGRATED_OBJECTIVE_VERSION, LEGACY_OBJECTIVE_VERSION, normalize_objective_version,
)


def _padded(data, key):
    value = data[key]
    return value.to_padded_tensor(0) if value.is_nested else value


def credit_objective(new, data, *, clip=.2, objective_version=LEGACY_OBJECTIVE_VERSION):
    """Pure tensor objective, before the native DP-average compensation."""
    import torch

    objective_version = normalize_objective_version(objective_version)
    if new.ndim != 2 or not math.isfinite(clip) or clip < 0:
        raise ValueError("Invalid paired objective geometry/clip")
    old, generated, score_mask, advantages = (
        _padded(data, key).to(new.device) for key in
        ("old_log_probs", "generated_mask", "response_mask", "advantages"))
    if any(value.shape != new.shape for value in (old, generated, score_mask, advantages)):
        raise ValueError("Paired policy response tensors differ")
    if objective_version in NATURAL_OBJECTIVE_VERSIONS and "contract_mask" not in data:
        raise ValueError("Natural objective requires explicit contract mask")
    contract = _padded(data, "contract_mask").to(new.device) if "contract_mask" in data else generated
    if (contract.shape != new.shape or not torch.all((contract == 0) | (contract == 1))
            or torch.any(contract > generated)):
        raise ValueError("Contract mask must be a binary subset of generated tokens")
    if not torch.all((generated == 0) | (generated == 1)) or not torch.all((score_mask == 0) | (score_mask == 1)):
        raise ValueError("Objective masks must be binary")
    count = new.shape[0]
    def vector(key):
        result = data[key].to(new.device)
        if result.ndim != 1 or result.shape[0] != count:
            raise ValueError("Invalid paired row field: " + key)
        return result

    kind = vector("row_kind")
    padding = vector("distributed_padding").bool()
    integrated = objective_version in INTEGRATED_OBJECTIVE_VERSIONS
    if not torch.all((kind == 0) | (kind == 1) | ((kind == 2) & (objective_version == INTEGRATED_OBJECTIVE_VERSION))):
        raise ValueError("Unknown paired row kind")
    if objective_version in (SEMANTIC_OBJECTIVE_VERSION, *NATURAL_OBJECTIVE_VERSIONS) and torch.any(kind != 0):
        raise ValueError("Generated-only objective permits only generated LM rows")
    score_rows = kind != 0
    if torch.any(generated[score_rows | padding]) or torch.any(score_mask[padding]):
        raise ValueError("Score/padding entered generated objective mask")
    if torch.any(generated > score_mask):
        raise ValueError("Generated objective is outside actual response mask")
    weights = {key: vector(key).to(new.dtype) for key in ("task_weight", "contract_weight", "kl_weight")}
    weights["initial_weight"] = (vector("initial_weight").to(new.dtype).detach()
        if "initial_weight" in data else new.new_zeros(count))
    initial_advantage = (vector("initial_advantage").to(new.dtype)
        if "initial_advantage" in data else new.new_zeros(count))
    if not integrated and torch.any(weights["initial_weight"]):
        raise ValueError("INITIAL local credit requires integrated objective")
    for key, value in weights.items():
        if not torch.isfinite(value).all() or torch.any(value < 0) or torch.any(value[score_rows | padding]):
            raise ValueError("Invalid globally normalized " + key)
    if not torch.isfinite(initial_advantage).all() or torch.any(initial_advantage[score_rows | padding]):
        raise ValueError("Invalid INITIAL advantage")
    enabled = generated.bool()
    if not torch.isfinite(new[score_mask.bool()]).all() or not torch.isfinite(old[enabled]).all() or not torch.isfinite(advantages[enabled]).all():
        raise ValueError("Nonfinite live objective input")
    # Independent graph branches make component gradient evidence unambiguous.
    task_new, initial_new, contract_new, kl_new, route_new, local_new = (new * 1. for _ in range(6))
    safe_old = torch.where(enabled, old.detach(), torch.zeros_like(old))
    def surrogate(current, advantage):
        difference = torch.where(enabled, current, torch.zeros_like(current)) - safe_old
        ratio = difference.exp()
        if not torch.isfinite(ratio[enabled]).all():
            raise ValueError("Nonfinite generated policy ratio")
        value = -torch.minimum(ratio * advantage, ratio.clamp(1 - clip, 1 + clip) * advantage)
        return value * generated, ratio

    safe_advantage = torch.where(enabled, advantages.detach(), torch.zeros_like(advantages))
    task_terms, ratio = surrogate(task_new, safe_advantage)
    initial_terms, _ = surrogate(initial_new, initial_advantage.detach()[:, None])
    contract_terms, _ = surrogate(contract_new, -torch.ones_like(new))
    task_loss = (task_terms * weights["task_weight"][:, None]).sum()
    initial_loss = (initial_terms * weights["initial_weight"][:, None]).sum()
    contract_loss = (contract_terms * contract * weights["contract_weight"][:, None]).sum()
    kl_loss = new.sum() * 0.
    if torch.any(weights["kl_weight"]):
        ref = _padded(data, "ref_log_prob").to(new.device).detach()
        if ref.shape != new.shape or not torch.isfinite(ref[enabled]).all():
            raise ValueError("Invalid frozen generated reference log probabilities")
        difference = torch.where(enabled, ref - kl_new, torch.zeros_like(new)).clamp(-20, 20)
        terms = (difference.exp() - difference - 1).clamp(-10, 10)
        kl_loss = (terms * generated * weights["kl_weight"][:, None]).sum()

    route_loss, route_kl = new.sum() * 0., new.sum() * 0.
    route_count, valid_count = 0, 0
    q_indices = torch.where((kind == 1) & ~padding)[0]
    if q_indices.numel():
        keys, candidate = vector("q_pair_index"), vector("q_candidate")
        valid, scoreable = vector("route_valid"), vector("route_scoreable")
        delta, denominator = vector("route_delta"), vector("route_M")
        reference_score = vector("q_reference_score").detach()
        if torch.any(keys[q_indices] < 0):
            raise ValueError("Missing actual q pair identity")
        for key in torch.unique(keys[q_indices]):
            indices = q_indices[keys[q_indices] == key]
            if indices.numel() != 2 or sorted(candidate[indices].tolist()) != [0, 1]:
                raise ValueError("Native microbatch split or duplicated a q pair")
            indices = indices[torch.argsort(candidate[indices])]
            d, f = indices[0], indices[1]
            for name, value in (("valid", valid), ("scoreable", scoreable), ("delta", delta), ("M", denominator)):
                if not torch.equal(value[d], value[f]):
                    raise ValueError("Paired router metadata differs: " + name)
            if valid[d].item() not in (0, 1) or scoreable[d].item() not in (0, 1):
                raise ValueError("Router eligibility must be boolean")
            if valid[d] and not scoreable[d]:
                raise ValueError("Unscoreable router cannot carry task credit")
            if not scoreable[d]:
                continue
            if denominator[d] <= 0 or not torch.isfinite(denominator[d]) or not torch.isfinite(delta[d]):
                raise ValueError("Invalid planned router denominator/delta")
            if not torch.all(score_mask[indices].sum(dim=1) > 0) or not torch.isfinite(reference_score[indices]).all():
                raise ValueError("Missing actual q suffix/reference score")
            scores = (route_new[indices] * score_mask[indices]).sum(dim=1)
            logq = torch.log_softmax(scores, dim=0)
            q = logq.exp()
            logref = torch.log_softmax(reference_score[indices].to(scores.dtype), dim=0)
            divisor = denominator[d].detach()
            if objective_version == LEGACY_OBJECTIVE_VERSION:
                route_loss = route_loss - q[1] * valid[d].detach().to(q.dtype) * delta[d].detach() / divisor
            elif valid[d]:
                # Full-suffix log_softmax stays stable at wrong extreme q.
                # Half scale matches v1's data derivative at q=.5 only; it
                # does not claim equivalence under shared-model optimization.
                frozen_delta = delta[d].detach()
                route_loss = route_loss - .5 * (
                    frozen_delta.clamp_min(0) * logq[1] +
                    (-frozen_delta).clamp_min(0) * logq[0]) / divisor
            route_kl = route_kl + .001 * (q * (logq - logref)).sum() / divisor
            route_count += 1
            valid_count += int(valid[d])
    local_loss = new.sum() * 0.
    family_losses = [new.sum() * 0., new.sum() * 0.]
    family_counts = [0, 0]
    local_indices = torch.where((kind == 2) & ~padding)[0]
    if local_indices.numel():
        keys, candidate = vector("local_pair_index"), vector("local_candidate")
        family = vector("local_family")
        delta, weight = vector("local_delta"), vector("local_weight")
        reference_score = vector("local_reference_score").detach()
        if torch.any(keys[local_indices] < 0):
            raise ValueError("Missing actual local preference identity")
        for key in torch.unique(keys[local_indices]):
            indices = local_indices[keys[local_indices] == key]
            if indices.numel() != 2 or sorted(candidate[indices].tolist()) != [0, 1]:
                raise ValueError("Native microbatch split or duplicated a local preference pair")
            indices = indices[torch.argsort(candidate[indices])]
            a, b = indices[0], indices[1]
            for name, value in (("family", family), ("delta", delta), ("weight", weight)):
                if not torch.equal(value[a], value[b]):
                    raise ValueError("Local preference metadata differs: " + name)
            if family[a].item() not in (0, 1):
                raise ValueError("Unknown local preference family")
            if not torch.isfinite(delta[a]) or not torch.isfinite(weight[a]) or weight[a] < 0:
                raise ValueError("Invalid local preference delta/weight")
            lengths = score_mask[indices].sum(dim=1)
            if torch.any(lengths <= 0) or not torch.isfinite(reference_score[indices]).all():
                raise ValueError("Missing actual local response/reference score")
            means = (local_new[indices] * score_mask[indices]).sum(dim=1) / lengths
            relative = means - reference_score[indices].to(means.dtype)
            margin = relative[1] - relative[0]
            frozen_delta = delta[a].detach().to(means.dtype)
            term = weight[a].detach().to(means.dtype) * frozen_delta.abs() * torch.nn.functional.softplus(
                -frozen_delta.sign() * margin)
            local_loss = local_loss + term
            family_index = int(family[a])
            family_losses[family_index] = family_losses[family_index] + term
            family_counts[family_index] += 1
    loss = task_loss + initial_loss + contract_loss + kl_loss + route_loss + route_kl + local_loss
    if not torch.isfinite(loss):
        raise ValueError("Nonfinite paired objective")
    metrics = {"paired_credit/task": task_loss.detach(),
               "paired_credit/initial": initial_loss.detach(),
               "paired_credit/contract": contract_loss.detach(),
               "paired_credit/generated_kl": kl_loss.detach(),
               "paired_credit/route": route_loss.detach(),
               "paired_credit/route_kl": route_kl.detach(),
               "paired_credit/local_preference": local_loss.detach(),
               "paired_credit/t3_preference": family_losses[0].detach(),
               "paired_credit/t4_preference": family_losses[1].detach(),
               "paired_credit/t3_pairs": new.new_tensor(family_counts[0]),
               "paired_credit/t4_pairs": new.new_tensor(family_counts[1]),
               "paired_credit/enabled_tokens": generated.sum().detach(),
               "paired_credit/q_pairs": new.new_tensor(route_count),
               "paired_credit/valid_q_pairs": new.new_tensor(valid_count),
               "paired_credit/ratio_sum": ratio[enabled].sum().detach(),
               "paired_credit/call_rows": ((kind == 0) & ~padding).sum().detach()}
    return loss, metrics, {"pg": task_new, "initial": initial_new, "contract": contract_new,
                           "kl": kl_new, "route": route_new, "local_preference": local_new}


def paired_credit_loss(model_output, data, dp_group=None):
    """Native callback: preglobalnormalized components, one DP compensation."""
    import torch
    from verl.utils import tensordict_utils as tu
    from verl.workers.utils.padding import no_padding_2_padding

    if tu.get_non_tensor_data(data, "objective_reduction", None) != "paired_credit_v1":
        raise ValueError("Paired loss requires explicit workflow dispatch")
    new = no_padding_2_padding(model_output["log_probs"], data)
    M = tu.get_non_tensor_data(data, "M", None)
    q_rows = (data["row_kind"] == 1) & ~data["distributed_padding"].bool()
    if type(M) is not int or M <= 0 or torch.any(data["route_M"][q_rows] != M):
        raise ValueError("Router denominator differs from immutable macro M")
    objective_version = tu.get_non_tensor_data(data, "credit_objective_version", LEGACY_OBJECTIVE_VERSION)
    ready = tu.get_non_tensor_data(data, "paired_reference_ready", None)
    reference_missing = ready is not True if objective_version in INTEGRATED_OBJECTIVE_VERSIONS else ready is False
    if (reference_missing and (torch.any(data["kl_weight"] > 0)
            or torch.any(data["route_scoreable"][q_rows])
            or torch.any((data["row_kind"] == 2) & ~data["distributed_padding"].bool()))):
        raise ValueError("Frozen reference scores are not ready for paired credit")
    loss, metrics, branches = credit_objective(
        new, data, clip=tu.get_non_tensor_data(data, "clip_ratio", .2),
        objective_version=objective_version)
    dp_size = tu.get_non_tensor_data(data, "dp_size", 1)
    if type(dp_size) is not int or dp_size < 1:
        raise ValueError("Invalid native DP size")
    journal = tu.get_non_tensor_data(data, "gradient_journal", None)
    if journal and new.requires_grad:
        roles = tu.get_non_tensor_data(data, "role_names", ("T1", "T2", "T3", "T4"))
        role_indices = data["role_index"].detach().cpu().tolist()
        padding = data["distributed_padding"].detach().cpu().tolist()
        kinds = data["row_kind"].detach().cpu().tolist()
        def hook_for(objective):
            def record(gradient):
                import json
                from pathlib import Path
                values = gradient.detach().float().abs().sum(dim=1).cpu().tolist()
                with Path(journal).open("a") as handle:
                    for role, value, dummy, kind in zip(role_indices, values, padding, kinds):
                        if not dummy:
                            handle.write(json.dumps({"role": roles[role], "objective": objective,
                                "row_kind": ("generated", "q", "local_preference")[kind],
                                "objective_gradient_l1": value}) + "\n")
                return gradient
            return record
        for objective, branch in branches.items():
            branch.register_hook(hook_for(objective))
    # Diagnostics use SUM; never average already globally reduced contributions.
    # Pack by dtype/device to avoid one collective and host synchronization per
    # scalar, without casting integer counts. The loss graph is unchanged.
    groups = {}
    for key, value in metrics.items():
        groups.setdefault((value.device, value.dtype), []).append(key)
    scalars = {}
    for keys in groups.values():
        packed = torch.stack([metrics[key] for key in keys])
        if dp_group is not None and torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(packed, op=torch.distributed.ReduceOp.SUM, group=dp_group)
        scalars.update(zip(keys, map(float, packed.cpu().tolist())))
    return loss * dp_size, {key: scalars[key] for key in metrics}
