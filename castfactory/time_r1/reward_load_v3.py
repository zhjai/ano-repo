"""Reasoning load RewardV3: WAPE/NRMSE and strict native-think JSON answers.

Ratios are fractions, not percentages. NRMSE uses mean(abs(target)), not
target standard deviation or range. Original reward_v3 remains unchanged.
The opt-in bounded-v1 outcome profile rescales numerical quality while
leaving raw metrics, uplift thresholds and the invalid score unchanged.
"""
from __future__ import annotations

import json
import numpy as np

from castfactory.agentic.distillation_protocol import parse_reply, validate_values
from castfactory.time_r1 import reward as base_reward
from castfactory.time_r1 import reward_v3 as legacy

VERSION = "reward-load-v3-wape-nrmse-v1"
NORMALIZED_VERSION = "reward-load-v3-wape-nrmse-bounded-v1"
DEFAULT_OUTCOME_PROFILE = "raw-v1"
# A provisional raw-loss half-saturation point, not a fitted optimum.
DEFAULT_HALF_SATURATION_LOSS = 0.10
DEFAULT_OUTCOME_BOUND = 4.0
METRIC_CONTRACT = "reward_load_v3._normalized_errors"
HARD_INVALID = legacy.HARD_INVALID
DEFAULT_WAPE_WEIGHT = 0.9
DEFAULT_METRIC_CAP = legacy.DEFAULT_METRIC_CAP
DEFAULT_UPLIFT_WEIGHT = legacy.DEFAULT_UPLIFT_WEIGHT
DEFAULT_RELATIVE_MARGIN = legacy.DEFAULT_RELATIVE_MARGIN
DEFAULT_RELATIVE_WIDTH = legacy.DEFAULT_RELATIVE_WIDTH
DEFAULT_ABSOLUTE_MARGIN = legacy.DEFAULT_ABSOLUTE_MARGIN
DEFAULT_ABSOLUTE_WIDTH = legacy.DEFAULT_ABSOLUTE_WIDTH
DEFAULT_AUTOCORR_THRESHOLD = legacy.DEFAULT_AUTOCORR_THRESHOLD
DEFAULT_REWARD_COMPONENT_KEYS = list(legacy.DEFAULT_REWARD_COMPONENT_KEYS)
DEFAULT_REWARD_DIAGNOSTIC_KEYS = [
    "forecast_valid", "wape", "nrmse", "metric_loss", "history_baseline_loss", "no_exo_baseline_loss",
    "baseline_loss", "baseline_source", "baseline_period", "history_baseline_available",
    "no_exo_baseline_available", "improvement", "relative_gain", "relative_part",
    "absolute_part", "bonus_active",
]
DEFAULT_REWARD_LOGGING_KEYS = DEFAULT_REWARD_COMPONENT_KEYS + DEFAULT_REWARD_DIAGNOSTIC_KEYS
NORMALIZED_REWARD_DIAGNOSTIC_KEYS = [
    "normalization_active", "raw_outcome", "outcome_valid", "half_saturation_loss", "outcome_bound",
]
NORMALIZED_REWARD_LOGGING_KEYS = DEFAULT_REWARD_LOGGING_KEYS + NORMALIZED_REWARD_DIAGNOSTIC_KEYS
_normalize_reward_output = legacy._normalize_reward_output
_build_history_baselines = legacy._build_history_baselines
_strong_baseline_loss = legacy._strong_baseline_loss
_uplift_components = legacy._uplift_components


def extract_ground_truth_values(ground_truth):
    if isinstance(ground_truth, str):
        ground_truth = json.loads(ground_truth)
    if isinstance(ground_truth, dict):
        # VERL's standard RLVR export wraps labels; JSON answers use values.
        ground_truth = ground_truth["values"] if "values" in ground_truth else ground_truth["label"]
    if isinstance(ground_truth, np.ndarray):
        ground_truth = ground_truth.tolist()
    if isinstance(ground_truth, tuple):
        ground_truth = list(ground_truth)
    if not isinstance(ground_truth, list) or not ground_truth:
        raise ValueError("Ground truth must be a nonempty numeric array or values object")
    if all(isinstance(value, (list, tuple)) and len(value) == 1 for value in ground_truth):
        ground_truth = [value[0] for value in ground_truth]
    return validate_values(ground_truth, len(ground_truth))


def extract_qwen_results(text, attr=None, *, extra_info=None, expected_length=None):
    """Select a grammar only from explicit metadata, never from model text."""
    del attr
    from castfactory.time_r1.forecast_answer_contract import (
        ROWS_ANSWER_FORMAT, ROWS_PROTOCOL_VERSION, parse_rows_forecast_answer,
        MARKDOWN_ANSWER_FORMAT, COMPACT_PROTOCOL_VERSION, parse_markdown_forecast_answer, MARKDOWN_PROFILES,
    )
    info = dict(extra_info or {})
    metadata = info.get("metadata") or {}
    if not isinstance(metadata, dict):
        raise ValueError("Forecast metadata must be an object")
    keys = ("answer_format", "protocol_version")
    selected = {key: metadata.get(key, info.get(key)) for key in keys}
    rows_profile = any(scope.get(key) == marker for scope in (info, metadata)
                       for key, marker in zip(keys, (ROWS_ANSWER_FORMAT, ROWS_PROTOCOL_VERSION)))
    compact_profile = any(scope.get("answer_format") in MARKDOWN_PROFILES
                          or scope.get("protocol_version") in MARKDOWN_PROFILES.values()
                          for scope in (info, metadata))
    if rows_profile or compact_profile:
        expected = ((selected["answer_format"], MARKDOWN_PROFILES.get(selected["answer_format"])) if compact_profile
                    else (ROWS_ANSWER_FORMAT, ROWS_PROTOCOL_VERSION))
        if (expected[1] is None or selected != dict(zip(keys, expected))
                or any(key in info and key in metadata and info[key] != metadata[key] for key in keys)):
            raise ValueError("Explicit reward requires matching answer_format and protocol_version")
        horizon = expected_length if expected_length is not None else metadata.get("prediction_length")
        if "prediction_length" in metadata and (type(metadata["prediction_length"]) is not int
                                                  or metadata["prediction_length"] != horizon):
            raise ValueError("rowJSON reward metadata and target horizons disagree")
        parser = parse_markdown_forecast_answer if compact_profile else parse_rows_forecast_answer
        parsed = parser(text, expected_length=horizon,
                        prompt_owned_prefix=info.get("prompt_owned_response_prefix", ""),
                        **({"answer_format": selected["answer_format"]} if compact_profile else {}))
        if not parsed.valid:
            raise ValueError(parsed.error)
        return parsed.values
    answer = parse_reply(_normalize_reward_output(text, extra_info), "student")["answer"]
    if set(answer) != {"values"} or not isinstance(answer["values"], list):
        raise ValueError("Load answer requires exactly values")
    return validate_values(answer["values"], len(answer["values"]))


def _normalized_errors(prediction, target):
    """WAPE and mean-absolute-target NRMSE; a zero denominator is undefined."""
    gt = np.asarray(target, dtype=float)
    pred = np.asarray(prediction, dtype=float)
    if (gt.ndim != 1 or not gt.size or pred.shape != gt.shape
            or not np.all(np.isfinite(gt)) or not np.all(np.isfinite(pred))):
        raise ValueError("Metrics require equal-length finite numeric vectors")
    peak = float(np.max(np.abs(gt)))
    if peak == 0:
        raise ValueError("WAPE/NRMSE undefined: zero future absolute-load denominator")
    # Scaling both arrays first avoids squaring raw MW or very small denominators.
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        scaled_gt = gt / peak
        error = pred / peak - scaled_gt
        denominator = float(np.mean(np.abs(scaled_gt)))
        wape = float(np.mean(np.abs(error)) / denominator)
        error_peak = float(np.max(np.abs(error)))
        nrmse = (0.0 if error_peak == 0 else
                 float(error_peak / denominator * np.sqrt(np.mean((error / error_peak) ** 2))))
    return wape, nrmse


def _metric_loss(wape, nrmse, *, wape_weight=DEFAULT_WAPE_WEIGHT, metric_cap=DEFAULT_METRIC_CAP):
    return float(wape_weight * np.clip(wape, 0, metric_cap)
                 + (1 - wape_weight) * np.clip(nrmse, 0, metric_cap))


def _best_history_baseline_loss(baselines, target, *, wape_weight, metric_cap):
    losses = [_metric_loss(*_normalized_errors(prediction, target),
                           wape_weight=wape_weight, metric_cap=metric_cap)
              for prediction in baselines.values() if len(prediction) == len(target)]
    return min(losses) if losses else float("nan")


def _no_exo_reference_loss(extra_info, *, wape_weight, metric_cap):
    # Never reinterpret old nMAE/nMSE fields as WAPE/NRMSE.
    refs = dict(extra_info or {}).get("reward_references") or {}
    if not isinstance(refs, dict):
        return float("nan")
    if refs.get("no_exo_ref_metric_contract") != METRIC_CONTRACT:
        return float("nan")
    wape = legacy._reference_metric(extra_info, "no_exo_ref_wape")
    nrmse = legacy._reference_metric(extra_info, "no_exo_ref_nrmse")
    if (wape_weight > 0 and not np.isfinite(wape)) or (wape_weight < 1 and not np.isfinite(nrmse)):
        return float("nan")
    return _metric_loss(wape if np.isfinite(wape) else 0, nrmse if np.isfinite(nrmse) else 0,
                        wape_weight=wape_weight, metric_cap=metric_cap)


def _zero_result():
    return dict.fromkeys(DEFAULT_REWARD_LOGGING_KEYS, 0.0)


def _validate_outcome_profile(profile, half_saturation_loss, outcome_bound):
    if profile not in ("raw-v1", "bounded-v1"):
        raise ValueError("Unknown outcome_profile; expected raw-v1 or bounded-v1")
    if (isinstance(half_saturation_loss, (bool, np.bool_))
            or not np.isfinite(float(half_saturation_loss)) or float(half_saturation_loss) <= 0):
        raise ValueError("half_saturation_loss must be finite and positive")
    # Existing auxiliary penalties sum to at worst -1.0. Preserve a strict
    # gap above HARD_INVALID=-6, even with zero uplift and saturated outcome.
    if (isinstance(outcome_bound, (bool, np.bool_))
            or not np.isfinite(float(outcome_bound)) or not 0 < float(outcome_bound) < 5):
        raise ValueError("outcome_bound must be finite and in (0, 5)")


def _outcome_loss(metric_loss, *, half_saturation_loss, outcome_bound):
    """Bounded loss without a new hard plateau or an overflowing L+s sum."""
    loss = float(metric_loss)
    scale = float(half_saturation_loss)
    if loss == 0:
        return 0.0
    if loss <= scale:
        ratio = loss / scale
        return float(outcome_bound) * ratio / (1.0 + ratio)
    return float(outcome_bound) / (1.0 + scale / loss)


def _with_outcome_diagnostics(result, profile, half_saturation_loss, outcome_bound):
    if profile == "bounded-v1":
        result.update(normalization_active=1.0,
                      raw_outcome=(-result["metric_loss"] if result["forecast_valid"] else float("nan")),
                      outcome_valid=(result["outcome"] if result["forecast_valid"] else float("nan")),
                      half_saturation_loss=float(half_saturation_loss), outcome_bound=float(outcome_bound))
    return result


def _invalid_result():
    result = _zero_result()
    result["format"] = HARD_INVALID
    # Undefined forecasting errors are not perfect forecasts. The finite score
    # still penalizes failures and remains suitable for checkpoint selection.
    result.update(wape=float("nan"), nrmse=float("nan"), metric_loss=float("nan"))
    return {"score": HARD_INVALID, **result}


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    del data_source
    return compute_score_components(solution_str, ground_truth, extra_info=extra_info, **kwargs)


def compute_score_components(solution_str, ground_truth, *, extra_info=None,
                             wape_weight=DEFAULT_WAPE_WEIGHT, metric_cap=DEFAULT_METRIC_CAP,
                             uplift_weight=DEFAULT_UPLIFT_WEIGHT,
                             relative_margin=DEFAULT_RELATIVE_MARGIN, relative_width=DEFAULT_RELATIVE_WIDTH,
                             absolute_margin=DEFAULT_ABSOLUTE_MARGIN, absolute_width=DEFAULT_ABSOLUTE_WIDTH,
                             autocorr_threshold=DEFAULT_AUTOCORR_THRESHOLD,
                             outcome_profile=DEFAULT_OUTCOME_PROFILE,
                             half_saturation_loss=DEFAULT_HALF_SATURATION_LOSS,
                             outcome_bound=DEFAULT_OUTCOME_BOUND):
    _validate_outcome_profile(outcome_profile, half_saturation_loss, outcome_bound)
    config = [wape_weight, metric_cap, uplift_weight, relative_margin, relative_width,
              absolute_margin, absolute_width, autocorr_threshold]
    if not all(np.isfinite(float(value)) for value in config):
        raise ValueError("Reward parameters must be finite")
    legacy._validate_config(nmae_weight=wape_weight, metric_cap=metric_cap,
                            uplift_weight=uplift_weight, relative_margin=relative_margin,
                            relative_width=relative_width, absolute_margin=absolute_margin,
                            absolute_width=absolute_width, autocorr_threshold=autocorr_threshold)
    try:
        target = extract_ground_truth_values(ground_truth)
    except (ValueError, TypeError, KeyError, OverflowError):
        return _with_outcome_diagnostics(_invalid_result(), outcome_profile, half_saturation_loss, outcome_bound)
    # Bad data is not a model-format failure and must not receive a fabricated ratio.
    if not any(abs(value) > 0 for value in target):
        raise ValueError("WAPE/NRMSE undefined: zero future absolute-load denominator")
    try:
        prediction = extract_qwen_results(solution_str, extra_info=extra_info, expected_length=len(target))
        validate_values(prediction, len(target))
        wape, nrmse = _normalized_errors(prediction, target)
    except (ValueError, TypeError, KeyError, OverflowError):
        return _with_outcome_diagnostics(_invalid_result(), outcome_profile, half_saturation_loss, outcome_bound)
    if not np.isfinite(wape) or not np.isfinite(nrmse):
        return _with_outcome_diagnostics(_invalid_result(), outcome_profile, half_saturation_loss, outcome_bound)
    metric_loss = _metric_loss(wape, nrmse, wape_weight=wape_weight, metric_cap=metric_cap)
    history = base_reward._reward_context(extra_info, prediction_length=len(target))["history"]
    if not history:
        model_input = dict(extra_info or {}).get("model_input") or {}
        history = base_reward._extract_history_values((model_input.get("target_history") or {}).get("values"))
    baselines, period = _build_history_baselines(history, len(target), autocorr_threshold=autocorr_threshold)
    history_loss = _best_history_baseline_loss(baselines, target, wape_weight=wape_weight, metric_cap=metric_cap)
    no_exo_loss = _no_exo_reference_loss(extra_info, wape_weight=wape_weight, metric_cap=metric_cap)
    baseline_loss, baseline_source = _strong_baseline_loss(history_loss, no_exo_loss)
    uplift = _uplift_components(metric_loss=metric_loss, baseline_loss=baseline_loss,
                                uplift_weight=uplift_weight, relative_margin=relative_margin,
                                relative_width=relative_width, absolute_margin=absolute_margin,
                                absolute_width=absolute_width)
    result = _zero_result()
    outcome_loss = (_outcome_loss(metric_loss, half_saturation_loss=half_saturation_loss,
                                  outcome_bound=outcome_bound)
                    if outcome_profile == "bounded-v1" else metric_loss)
    result.update(forecast_valid=1.0, outcome=-outcome_loss, uplift_bonus=uplift["bonus"],
                  scale_penalty=float(base_reward._scale_penalty(prediction, target)),
                  explosion_penalty=float(base_reward._explosion_penalty(prediction, target)),
                  wape=wape, nrmse=nrmse, metric_loss=metric_loss,
                  history_baseline_loss=legacy._finite_or_zero(history_loss),
                  no_exo_baseline_loss=legacy._finite_or_zero(no_exo_loss),
                  baseline_loss=legacy._finite_or_zero(baseline_loss), baseline_source=float(baseline_source),
                  baseline_period=float(period or 0), history_baseline_available=float(np.isfinite(history_loss)),
                  no_exo_baseline_available=float(np.isfinite(no_exo_loss)),
                  improvement=uplift["improvement"], relative_gain=uplift["relative_gain"],
                  relative_part=uplift["relative_part"], absolute_part=uplift["absolute_part"],
                  bonus_active=float(uplift["bonus"] > 0))
    return _with_outcome_diagnostics(
        {"score": float(sum(result[key] for key in DEFAULT_REWARD_COMPONENT_KEYS)), **result},
        outcome_profile, half_saturation_loss, outcome_bound)
