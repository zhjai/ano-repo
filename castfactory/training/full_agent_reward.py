"""Versioned episode reward arithmetic shared by rollout and evaluation.

The legacy profile is deliberately the default so historical ledgers and
evaluation artifacts remain reproducible.  New compact-v2 runs opt in to
``normalized-wape-v1``.  All arithmetic is Decimal based; the exact WAPE is
never clipped or replaced with its presentation float.
"""
from __future__ import annotations

import math
from typing import Any

from castfactory.training.full_agent_metrics import metric_context, metric_decimal, scoring_wape, serialize_metric

LEGACY_PROFILE = "legacy"
NORMALIZED_WAPE_PROFILE = "normalized-wape-v1"
BOUNDED_WAPE_PROFILE = "bounded-wape-v1"
SUPPORTED_REWARD_PROFILES = (LEGACY_PROFILE, NORMALIZED_WAPE_PROFILE, BOUNDED_WAPE_PROFILE)
REWARD_VERSION = "full-agent-reward-v2"


def _canonical(profile: str | None) -> str:
    value = LEGACY_PROFILE if profile is None else str(profile)
    if value not in SUPPORTED_REWARD_PROFILES:
        raise ValueError("Unknown reward profile: " + value)
    return value


def reward_settings(profile: str | None = LEGACY_PROFILE, cost_weight: float = .01) -> dict[str, Any]:
    """Return serializable identity metadata for an explicit reward profile."""
    profile = _canonical(profile)
    if not math.isfinite(float(cost_weight)) or float(cost_weight) < 0:
        raise ValueError("Invalid reward cost coefficient")
    wape_scale = "1" if profile == LEGACY_PROFILE else "0.10"
    result = {
        "reward_profile": profile,
        "profile": profile,
        "version": REWARD_VERSION if profile == NORMALIZED_WAPE_PROFILE else "full_role_episode_grpo_future_wape_v1",
        "reward_version": REWARD_VERSION if profile == NORMALIZED_WAPE_PROFILE else "full_role_episode_grpo_future_wape_v1",
        "wape_scale": wape_scale,
        "failure_scale": "1",
        "cost_ratio_scale": "1",
        "coefficients": {"forecast": "1", "failure": "1", "cost": str(cost_weight)},
        "scales": {"forecast": wape_scale, "failure": "1", "cost": "1"},
        "cost_weight": float(cost_weight),
        "clipping": "none",
    }

    if profile == BOUNDED_WAPE_PROFILE:
        result.update(version="full-agent-reward-bounded-v1", reward_version="full-agent-reward-bounded-v1",
            quality_transform="-4*raw_WAPE/(raw_WAPE+.10)", quality_bound="[-4,0]",
            wape_units="fraction", quality_scale=".10", quality_budget="4",
            clipping="none_monotone_saturation")
    return result


def reward_components(*, wape: Any, actual_cost: int, c_ref: int,
                      failed: bool, profile: str | None = LEGACY_PROFILE,
                      cost_weight: float = .01) -> dict[str, Any]:
    """Compute exact component values for one episode.

    ``forecast`` and ``cost`` are signed reward contributions; the companion
    ``*_penalty`` values remain positive for diagnostics and compatibility.
    """
    profile = _canonical(profile)
    settings = reward_settings(profile, cost_weight)
    if type(actual_cost) is not int or actual_cost < 0:
        raise ValueError("Invalid measured episode cost")
    if isinstance(c_ref, bool) or not isinstance(c_ref, (int, float)) or not math.isfinite(float(c_ref)) or c_ref <= 0:
        raise ValueError("Invalid measured DIRECT reference")
    exact_wape = metric_decimal(wape)
    if exact_wape < 0:
        raise ValueError("WAPE cannot be negative")
    with metric_context():
        wape_scale = metric_decimal(settings["wape_scale"])
        if profile == BOUNDED_WAPE_PROFILE:
            from castfactory.training.paired_credit_quality import quality
            from castfactory.training.paired_credit_objective import BOUNDED_NATURAL_OBJECTIVE_VERSION
            forecast_decimal = quality(exact_wape, BOUNDED_NATURAL_OBJECTIVE_VERSION)
        else:
            forecast_decimal = -exact_wape / wape_scale
        ratio_decimal = metric_decimal(actual_cost) / metric_decimal(c_ref)
        # Old artifacts used the exact binary64 coefficient.  Preserve it
        # for legacy while parsing the adopted normalized .01 exactly.
        coefficient = metric_decimal(cost_weight) if profile == LEGACY_PROFILE else metric_decimal(str(cost_weight))
        cost_penalty_decimal = coefficient * metric_decimal(actual_cost) / metric_decimal(c_ref)
        failure_decimal = metric_decimal(int(bool(failed)))
        total_decimal = forecast_decimal - cost_penalty_decimal - failure_decimal
    return {
        "reward": serialize_metric(total_decimal),
        "reward_decimal": str(total_decimal),
        "total": serialize_metric(total_decimal),
        "total_decimal": str(total_decimal),
        "forecast": serialize_metric(forecast_decimal),
        "forecast_decimal": str(forecast_decimal),
        "weighted_forecast": serialize_metric(forecast_decimal),
        "weighted_forecast_decimal": str(forecast_decimal),
        "cost": serialize_metric(cost_penalty_decimal.copy_negate()),
        "cost_penalty": serialize_metric(cost_penalty_decimal),
        "contract": serialize_metric(failure_decimal.copy_negate()),
        "contract_penalty": serialize_metric(failure_decimal),
        "actual_cost": actual_cost,
        "c_ref": c_ref,
        "cost_ratio": serialize_metric(ratio_decimal),
        "cost_weight": float(cost_weight),
        **settings,
    }


def reward_components_from_metrics(metrics: dict[str, Any], *, actual_cost: int,
                                   c_ref: int, failed: bool,
                                   profile: str | None = LEGACY_PROFILE,
                                   cost_weight: float = .01) -> dict[str, Any]:
    """Same helper using the evaluator's exact WAPE companion."""
    if metrics.get("normalization_status") != "ok":
        raise ValueError("WAPE unavailable: " + str(metrics.get("normalization_status")))
    return reward_components(wape=scoring_wape(metrics), actual_cost=actual_cost,
                             c_ref=c_ref, failed=failed, profile=profile,
                             cost_weight=cost_weight)
