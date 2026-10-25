from __future__ import annotations

import numpy as np

from castfactory.time_r1 import reward as base_reward
from castfactory.time_r1.forecast_answer_contract import parse_gefcom_forecast_answer
from castfactory.training.chat_template_compat import PROMPT_OWNED_RESPONSE_PREFIX_KEY


HARD_INVALID = -6.0
DEFAULT_NMAE_WEIGHT = 0.9
DEFAULT_METRIC_CAP = 4.0
DEFAULT_UPLIFT_WEIGHT = 0.1
DEFAULT_RELATIVE_MARGIN = 0.05
DEFAULT_RELATIVE_WIDTH = 0.20
DEFAULT_ABSOLUTE_MARGIN = 0.02
DEFAULT_ABSOLUTE_WIDTH = 0.08
DEFAULT_AUTOCORR_THRESHOLD = 0.50

EPS = base_reward.EPS

DEFAULT_REWARD_COMPONENT_KEYS = [
    "format",
    "outcome",
    "uplift_bonus",
    "scale_penalty",
    "explosion_penalty",
]
DEFAULT_REWARD_DIAGNOSTIC_KEYS = [
    "nmae",
    "nmse",
    "metric_loss",
    "history_baseline_loss",
    "no_exo_baseline_loss",
    "baseline_loss",
    "baseline_source",
    "baseline_period",
    "history_baseline_available",
    "no_exo_baseline_available",
    "improvement",
    "relative_gain",
    "relative_part",
    "absolute_part",
    "bonus_active",
]
DEFAULT_REWARD_LOGGING_KEYS = DEFAULT_REWARD_COMPONENT_KEYS + DEFAULT_REWARD_DIAGNOSTIC_KEYS


extract_ground_truth_values = base_reward.extract_ground_truth_values


def extract_qwen_results(text: str, attr: str | None = None) -> list[float]:
    del attr
    return parse_gefcom_forecast_answer(text).values


def compute_score(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    *,
    nmae_weight: float = DEFAULT_NMAE_WEIGHT,
    metric_cap: float = DEFAULT_METRIC_CAP,
    uplift_weight: float = DEFAULT_UPLIFT_WEIGHT,
    relative_margin: float = DEFAULT_RELATIVE_MARGIN,
    relative_width: float = DEFAULT_RELATIVE_WIDTH,
    absolute_margin: float = DEFAULT_ABSOLUTE_MARGIN,
    absolute_width: float = DEFAULT_ABSOLUTE_WIDTH,
    autocorr_threshold: float = DEFAULT_AUTOCORR_THRESHOLD,
):
    del data_source
    return compute_score_components(
        solution_str,
        ground_truth,
        extra_info=extra_info,
        nmae_weight=nmae_weight,
        metric_cap=metric_cap,
        uplift_weight=uplift_weight,
        relative_margin=relative_margin,
        relative_width=relative_width,
        absolute_margin=absolute_margin,
        absolute_width=absolute_width,
        autocorr_threshold=autocorr_threshold,
    )


def compute_score_components(
    solution_str: str,
    ground_truth,
    *,
    extra_info=None,
    nmae_weight: float = DEFAULT_NMAE_WEIGHT,
    metric_cap: float = DEFAULT_METRIC_CAP,
    uplift_weight: float = DEFAULT_UPLIFT_WEIGHT,
    relative_margin: float = DEFAULT_RELATIVE_MARGIN,
    relative_width: float = DEFAULT_RELATIVE_WIDTH,
    absolute_margin: float = DEFAULT_ABSOLUTE_MARGIN,
    absolute_width: float = DEFAULT_ABSOLUTE_WIDTH,
    autocorr_threshold: float = DEFAULT_AUTOCORR_THRESHOLD,
) -> dict[str, float]:
    _validate_config(
        nmae_weight=nmae_weight,
        metric_cap=metric_cap,
        uplift_weight=uplift_weight,
        relative_margin=relative_margin,
        relative_width=relative_width,
        absolute_margin=absolute_margin,
        absolute_width=absolute_width,
        autocorr_threshold=autocorr_threshold,
    )

    ground_truth_values = extract_ground_truth_values(ground_truth)
    if not ground_truth_values or not base_reward._all_finite(ground_truth_values):
        return _invalid_result()

    normalized_output = _normalize_reward_output(solution_str, extra_info)
    contract = parse_gefcom_forecast_answer(
        normalized_output,
        expected_length=len(ground_truth_values),
    )
    if not contract.valid or not base_reward._all_finite(contract.values):
        return _invalid_result()

    prediction = [float(value) for value in contract.values]
    target = [float(value) for value in ground_truth_values]
    nmae, nmse = _normalized_errors(prediction, target)
    if not np.isfinite(nmae) or not np.isfinite(nmse):
        return _invalid_result()

    metric_loss = _metric_loss(
        nmae,
        nmse,
        nmae_weight=nmae_weight,
        metric_cap=metric_cap,
    )
    history = base_reward._reward_context(
        extra_info,
        prediction_length=len(target),
    )["history"]
    history_baselines, period = _build_history_baselines(
        history,
        horizon=len(target),
        autocorr_threshold=autocorr_threshold,
    )
    history_loss = _best_history_baseline_loss(
        history_baselines,
        target,
        nmae_weight=nmae_weight,
        metric_cap=metric_cap,
    )
    no_exo_loss = _no_exo_reference_loss(
        extra_info,
        nmae_weight=nmae_weight,
        metric_cap=metric_cap,
    )
    baseline_loss, baseline_source = _strong_baseline_loss(history_loss, no_exo_loss)
    uplift = _uplift_components(
        metric_loss=metric_loss,
        baseline_loss=baseline_loss,
        uplift_weight=uplift_weight,
        relative_margin=relative_margin,
        relative_width=relative_width,
        absolute_margin=absolute_margin,
        absolute_width=absolute_width,
    )

    result = _zero_result()
    result["format"] = 0.0
    result["outcome"] = float(-metric_loss)
    result["uplift_bonus"] = float(uplift["bonus"])
    result["scale_penalty"] = float(base_reward._scale_penalty(prediction, target))
    result["explosion_penalty"] = float(base_reward._explosion_penalty(prediction, target))
    result["nmae"] = float(nmae)
    result["nmse"] = float(nmse)
    result["metric_loss"] = float(metric_loss)
    result["history_baseline_loss"] = _finite_or_zero(history_loss)
    result["no_exo_baseline_loss"] = _finite_or_zero(no_exo_loss)
    result["baseline_loss"] = _finite_or_zero(baseline_loss)
    result["baseline_source"] = float(baseline_source)
    result["baseline_period"] = float(period or 0)
    result["history_baseline_available"] = 1.0 if np.isfinite(history_loss) else 0.0
    result["no_exo_baseline_available"] = 1.0 if np.isfinite(no_exo_loss) else 0.0
    result["improvement"] = float(uplift["improvement"])
    result["relative_gain"] = float(uplift["relative_gain"])
    result["relative_part"] = float(uplift["relative_part"])
    result["absolute_part"] = float(uplift["absolute_part"])
    result["bonus_active"] = 1.0 if uplift["bonus"] > 0.0 else 0.0
    result["score"] = float(sum(result[key] for key in DEFAULT_REWARD_COMPONENT_KEYS))
    return {"score": float(result.pop("score")), **result}


def _normalize_reward_output(
    solution_str: str | None,
    extra_info=None,
) -> str:
    raw_output = "" if solution_str is None else str(solution_str)
    info = dict(extra_info or {})
    prefix = info.get(PROMPT_OWNED_RESPONSE_PREFIX_KEY, "")
    if (
        isinstance(prefix, str)
        and prefix
        and not raw_output.lstrip().lower().startswith("<think>")
    ):
        return prefix + raw_output
    return raw_output


def _validate_config(
    *,
    nmae_weight: float,
    metric_cap: float,
    uplift_weight: float,
    relative_margin: float,
    relative_width: float,
    absolute_margin: float,
    absolute_width: float,
    autocorr_threshold: float,
) -> None:
    if not 0.0 <= float(nmae_weight) <= 1.0:
        raise ValueError("nmae_weight must be in [0, 1]")
    if float(metric_cap) <= 0.0:
        raise ValueError("metric_cap must be positive")
    if float(uplift_weight) < 0.0:
        raise ValueError("uplift_weight must be non-negative")
    if float(relative_margin) < 0.0 or float(absolute_margin) < 0.0:
        raise ValueError("uplift margins must be non-negative")
    if float(relative_width) <= 0.0 or float(absolute_width) <= 0.0:
        raise ValueError("uplift widths must be positive")
    if not -1.0 <= float(autocorr_threshold) <= 1.0:
        raise ValueError("autocorr_threshold must be in [-1, 1]")


def _invalid_result() -> dict[str, float]:
    result = _zero_result()
    result["format"] = HARD_INVALID
    result["score"] = HARD_INVALID
    return {"score": float(result.pop("score")), **result}


def _zero_result() -> dict[str, float]:
    return {
        key: 0.0
        for key in DEFAULT_REWARD_COMPONENT_KEYS + DEFAULT_REWARD_DIAGNOSTIC_KEYS
    }


def _normalized_errors(prediction: list[float], target: list[float]) -> tuple[float, float]:
    pred = np.asarray(prediction, dtype=float)
    gt = np.asarray(target, dtype=float)
    scale = _target_scale(gt)
    error = pred - gt
    nmae = float(np.mean(np.abs(error)) / scale)
    nmse = float(np.mean(np.square(error)) / (scale * scale))
    return nmae, nmse


def _target_scale(target: np.ndarray) -> float:
    return max(
        float(np.std(target)),
        0.05 * float(np.mean(np.abs(target))),
        EPS,
    )


def _metric_loss(
    nmae: float,
    nmse: float,
    *,
    nmae_weight: float,
    metric_cap: float,
) -> float:
    nmae_term = min(max(float(nmae), 0.0), float(metric_cap))
    nmse_term = min(max(float(nmse), 0.0), float(metric_cap))
    return float(nmae_weight * nmae_term + (1.0 - nmae_weight) * nmse_term)


def _build_history_baselines(
    history: list[float],
    horizon: int,
    *,
    autocorr_threshold: float = DEFAULT_AUTOCORR_THRESHOLD,
) -> tuple[dict[str, list[float]], int | None]:
    values = [float(value) for value in history if np.isfinite(value)]
    baselines: dict[str, list[float]] = {}
    if horizon <= 0 or not values:
        return baselines, None

    baselines["last_value"] = [values[-1]] * horizon
    recent = values[-min(horizon, len(values)) :]
    baselines["recent_horizon_mean"] = [float(np.mean(recent))] * horizon

    period = _infer_stable_period(
        values,
        horizon=horizon,
        autocorr_threshold=autocorr_threshold,
    )
    if period is not None:
        last_period = values[-period:]
        baselines["seasonal_naive"] = [
            last_period[index % period]
            for index in range(horizon)
        ]
    return baselines, period


def _infer_stable_period(
    history: list[float],
    *,
    horizon: int,
    autocorr_threshold: float,
) -> int | None:
    values = np.asarray(history, dtype=float)
    max_period = min(max(int(horizon), 2), values.size // 3)
    if values.size < 6 or max_period < 2:
        return None

    x = np.arange(values.size, dtype=float)
    slope, intercept = np.polyfit(x, values, 1)
    detrended = values - (slope * x + intercept)
    if float(np.std(detrended)) <= EPS:
        return None

    scores = {
        period: _lag_correlation(detrended, period)
        for period in range(1, max_period + 2)
    }
    candidates = []
    for period in range(2, max_period + 1):
        previous = scores[period - 1]
        score = scores[period]
        following = scores[period + 1]
        if not all(np.isfinite(value) for value in (previous, score, following)):
            continue
        if score < autocorr_threshold or score < previous or score < following:
            continue
        candidates.append((score, period))
    if not candidates:
        return None
    best_score = max(score for score, _period in candidates)
    best_period = min(
        period
        for score, period in candidates
        if score >= best_score - EPS
    )
    return int(best_period)


def _lag_correlation(values: np.ndarray, lag: int) -> float:
    left = values[:-lag]
    right = values[lag:]
    if left.size < 2 or float(np.std(left)) <= EPS or float(np.std(right)) <= EPS:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def _best_history_baseline_loss(
    baselines: dict[str, list[float]],
    target: list[float],
    *,
    nmae_weight: float,
    metric_cap: float,
) -> float:
    losses = []
    for prediction in baselines.values():
        if len(prediction) != len(target):
            continue
        nmae, nmse = _normalized_errors(prediction, target)
        losses.append(
            _metric_loss(
                nmae,
                nmse,
                nmae_weight=nmae_weight,
                metric_cap=metric_cap,
            )
        )
    return float(min(losses)) if losses else float("nan")


def _no_exo_reference_loss(
    extra_info,
    *,
    nmae_weight: float,
    metric_cap: float,
) -> float:
    nmae = _reference_metric(extra_info, "no_exo_ref_nmae")
    nmse = _reference_metric(extra_info, "no_exo_ref_nmse")
    if nmae_weight > 0.0 and not np.isfinite(nmae):
        return float("nan")
    if nmae_weight < 1.0 and not np.isfinite(nmse):
        return float("nan")
    if not np.isfinite(nmae):
        nmae = 0.0
    if not np.isfinite(nmse):
        nmse = 0.0
    return _metric_loss(
        nmae,
        nmse,
        nmae_weight=nmae_weight,
        metric_cap=metric_cap,
    )


def _reference_metric(extra_info, key: str) -> float:
    info = dict(extra_info or {})
    references = info.get("reward_references", {})
    if not isinstance(references, dict):
        return float("nan")
    try:
        value = float(references.get(key))
    except (TypeError, ValueError):
        return float("nan")
    return value if np.isfinite(value) and value >= 0.0 else float("nan")


def _strong_baseline_loss(history_loss: float, no_exo_loss: float) -> tuple[float, int]:
    candidates = []
    if np.isfinite(history_loss):
        candidates.append((float(history_loss), 1))
    if np.isfinite(no_exo_loss):
        candidates.append((float(no_exo_loss), 2))
    if not candidates:
        return float("nan"), 0
    return min(candidates, key=lambda item: item[0])


def _uplift_components(
    *,
    metric_loss: float,
    baseline_loss: float,
    uplift_weight: float,
    relative_margin: float,
    relative_width: float,
    absolute_margin: float,
    absolute_width: float,
) -> dict[str, float]:
    if not np.isfinite(baseline_loss) or uplift_weight <= 0.0:
        return {
            "improvement": 0.0,
            "relative_gain": 0.0,
            "relative_part": 0.0,
            "absolute_part": 0.0,
            "bonus": 0.0,
        }
    improvement = float(baseline_loss - metric_loss)
    relative_gain = float(improvement / max(float(baseline_loss), 0.1))
    relative_part = _clip(
        (relative_gain - relative_margin) / relative_width,
        0.0,
        1.0,
    )
    absolute_part = _clip(
        (improvement - absolute_margin) / absolute_width,
        0.0,
        1.0,
    )
    bonus = float(uplift_weight * min(relative_part, absolute_part))
    return {
        "improvement": improvement,
        "relative_gain": relative_gain,
        "relative_part": relative_part,
        "absolute_part": absolute_part,
        "bonus": bonus,
    }


def _clip(value: float, lower: float, upper: float) -> float:
    return float(min(upper, max(lower, float(value))))


def _finite_or_zero(value: float) -> float:
    return float(value) if np.isfinite(value) else 0.0
