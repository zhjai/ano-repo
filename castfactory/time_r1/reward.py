from __future__ import annotations

import re
import string

import numpy as np

from castfactory.time_r1.prediction_values import extract_prediction_values


THINK_BLOCK_PATTERN = re.compile(r"<think>\s*(.*?)\s*</think>", re.IGNORECASE | re.DOTALL)
ANSWER_BLOCK_PATTERN = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL)
JSON_PAYLOAD_PATTERN = re.compile(r"(\{[\s\S]*\}|\[[\s\S]*\])")
REWARD_SCALE_FLOOR_RATIO = 0.05
SKILL_TEMPERATURE = 1.0
EPS = 1e-8
CYCLE_LENGTH = 24
IGNORED_FORECAST_COLUMNS = {
    "step",
    "time",
    "timestamp",
    "hod",
    "hour",
    "dow",
    "day",
    "month",
    "doy",
    "doy_sin",
    "doy_cos",
    "target",
    "value",
}


def mean_squared_error(y_true, y_pred) -> float:
    return float(np.mean((np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)) ** 2))


def decompose(values) -> tuple[np.ndarray, np.ndarray]:
    array = np.asarray(values, dtype=float).reshape(-1)
    kernel_size = 25
    pad = (kernel_size - 1) // 2
    padded = np.concatenate(
        [
            np.repeat(array[:1], pad),
            array,
            np.repeat(array[-1:], pad),
        ]
    )
    kernel = np.ones(kernel_size, dtype=float) / kernel_size
    trend = np.convolve(padded, kernel, mode="valid")
    season = array - trend
    return season, trend


def mean_squared_error_season_trend(y_true, y_pred) -> tuple[float, float]:
    season_true, trend_true = decompose(y_true)
    season_pred, trend_pred = decompose(y_pred)
    return (
        mean_squared_error(season_true, season_pred),
        mean_squared_error(trend_true, trend_pred),
    )


def extract_answer(text: str) -> str:
    match = ANSWER_BLOCK_PATTERN.search(str(text))
    return match.group(1) if match else str(text)


def extract_qwen_results(text: str, attr: str | None = None) -> list[float]:
    payload = _extract_reward_payload(text)
    return extract_prediction_values(payload, attr=attr)


def extract_ground_truth_values(text) -> list[float]:
    if isinstance(text, (list, tuple, np.ndarray)):
        array = np.asarray(text, dtype=float)
        return [float(value) for value in array.reshape(-1)]
    return [float(value) for value in re.findall(r"(-?\d+\.\d+|-?\d+)", str(text))]


def normalize_answer(text: str) -> str:
    def remove_articles(value: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", value)

    def white_space_fix(value: str) -> str:
        return " ".join(value.split())

    def remove_punc(value: str) -> str:
        return "".join(ch for ch in value if ch not in set(string.punctuation))

    return white_space_fix(remove_articles(remove_punc(str(text).lower())))


def extract_solution_answer(solution_str: str) -> str | None:
    payload = _extract_reward_payload(solution_str).strip()
    return payload or None


def compute_format_score(solution_str: str | None) -> float:
    if solution_str is None:
        return -1.0
    text = str(solution_str)
    if not text.strip():
        return -1.0
    if THINK_BLOCK_PATTERN.search(text) is not None:
        return 0.0
    if ANSWER_BLOCK_PATTERN.search(text) is not None:
        return 0.0
    if JSON_PAYLOAD_PATTERN.search(text) is not None:
        return 0.0
    return -1.0


def reward_norm(x_list, y_list) -> tuple[list[float], list[float]]:
    x_array = np.asarray(x_list, dtype=float)
    y_array = np.asarray(y_list, dtype=float)
    if x_array.size == 0 or y_array.size == 0:
        return [], []
    mu = np.nanmean(y_array)
    gt_std = np.nanstd(y_array)
    gt_level = np.nanmean(np.abs(y_array))
    denominator = max(
        float(gt_std),
        float(REWARD_SCALE_FLOOR_RATIO * gt_level),
        1e-8,
    )
    return ((x_array - mu) / denominator).tolist(), ((y_array - mu) / denominator).tolist()


def compute_answer_score(solution_str: str, ground_truth, *, attr: str | None = None) -> float:
    ground_truth_values = extract_ground_truth_values(ground_truth)
    answer = extract_qwen_results(solution_str, attr=attr)
    if len(answer) < len(ground_truth_values):
        return 0.0
    norm_answer, norm_gt = reward_norm(answer, ground_truth_values)
    if not norm_answer or not norm_gt:
        return 0.0
    mse = mean_squared_error(norm_gt, norm_answer[: len(ground_truth_values)])
    score = (1 - 1 / (1 + np.exp(-2 * mse))) * 2
    return float(score * 0.6)


def compute_score_length(solution_str: str, ground_truth, *, attr: str | None = None) -> float:
    ground_truth_values = extract_ground_truth_values(ground_truth)
    answer = extract_qwen_results(solution_str, attr=attr)
    if len(answer) >= len(ground_truth_values):
        return 0.1
    if not ground_truth_values:
        return 0.0
    return float(0.1 * len(answer) / len(ground_truth_values))


def compute_answer_score_season_trend(solution_str: str, ground_truth, *, attr: str | None = None) -> float:
    ground_truth_values = extract_ground_truth_values(ground_truth)
    answer = extract_qwen_results(solution_str, attr=attr)
    if len(answer) < len(ground_truth_values):
        return 0.0
    norm_answer, norm_gt = reward_norm(answer, ground_truth_values)
    if not norm_answer or not norm_gt:
        return 0.0
    mse_season, mse_trend = mean_squared_error_season_trend(
        norm_gt,
        norm_answer[: len(ground_truth_values)],
    )
    score_season = (1 - 1 / (1 + np.exp(-2 * mse_season))) * 2
    score_trend = (1 - 1 / (1 + np.exp(-2 * mse_trend))) * 2
    return float(0.05 * score_season + 0.15 * score_trend)


def change_point(data) -> tuple[list[int], list[int]]:
    values = [float(value) for value in data]
    if len(values) < 3:
        return [], []
    data_new = [values[2], values[1]] + values + [values[-2], values[-1]]
    change_point_max = []
    change_point_min = []
    for index in range(len(values)):
        center = data_new[index + 2]
        if (
            center >= data_new[index]
            and center >= data_new[index + 1]
            and center >= data_new[index + 3]
            and center >= data_new[index + 4]
        ):
            change_point_max.append(index)
        if (
            center <= data_new[index]
            and center <= data_new[index + 1]
            and center <= data_new[index + 3]
            and center <= data_new[index + 4]
        ):
            change_point_min.append(index)
    return change_point_max, change_point_min


def compute_change_point_score(solution_str: str, ground_truth, *, attr: str | None = None) -> float:
    ground_truth_values = extract_ground_truth_values(ground_truth)
    answer = extract_qwen_results(solution_str, attr=attr)
    if len(answer) < 3 or len(ground_truth_values) < 3:
        return 0.0
    ground_max, ground_min = change_point(ground_truth_values)
    answer_max, answer_min = change_point(answer)
    max_score = 0.0
    min_score = 0.0
    if ground_max:
        max_score = sum(1 for item in answer_max if item in ground_max) / len(ground_max) * 0.1
    if ground_min:
        min_score = sum(1 for item in answer_min if item in ground_min) / len(ground_min) * 0.1
    return float(max_score + min_score)


DEFAULT_REWARD_COMPONENT_KEYS = [
    "format",
    "length",
    "value",
    "flat_penalty",
    "copy_history_penalty",
    "repeat24_drift_penalty",
    "scale_penalty",
    "explosion_penalty",
    "change_point",
    "season_trend",
]
DEFAULT_REWARD_DIAGNOSTIC_KEYS = [
    "covariate_gate",
    "exo_gain",
    "blind_ref_nmse",
    "exo_ref_nmse",
    "pred_nmse",
    "value_mode_gated",
]
DEFAULT_REWARD_LOGGING_KEYS = DEFAULT_REWARD_COMPONENT_KEYS + DEFAULT_REWARD_DIAGNOSTIC_KEYS


def compute_score_components(
    solution_str: str,
    ground_truth,
    *,
    attr: str | None = None,
    extra_info=None,
) -> dict[str, float]:
    format_score = compute_format_score(solution_str)
    if format_score < 0.0:
        return _empty_component_result(score=-1.0, format_score=-1.0)

    components = _compute_forecast_reward_components(
        solution_str=solution_str,
        ground_truth=ground_truth,
        attr=attr,
        extra_info=extra_info,
    )
    components["format"] = float(format_score)
    components["score"] = float(sum(components[key] for key in DEFAULT_REWARD_COMPONENT_KEYS))
    return {"score": components.pop("score"), **components}


def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    del data_source
    attr = _resolve_attr(extra_info)
    return compute_score_components(solution_str, ground_truth, attr=attr, extra_info=extra_info)


def _empty_component_result(*, score: float, format_score: float) -> dict[str, float]:
    result = {key: 0.0 for key in DEFAULT_REWARD_COMPONENT_KEYS}
    result.update({key: 0.0 for key in DEFAULT_REWARD_DIAGNOSTIC_KEYS})
    result["format"] = float(format_score)
    result["score"] = float(score)
    return result


def _compute_forecast_reward_components(
    *,
    solution_str: str,
    ground_truth,
    attr: str | None,
    extra_info,
) -> dict[str, float]:
    ground_truth_values = extract_ground_truth_values(ground_truth)
    answer = extract_qwen_results(solution_str, attr=attr)
    components = {key: 0.0 for key in DEFAULT_REWARD_COMPONENT_KEYS}
    components.update({key: 0.0 for key in DEFAULT_REWARD_DIAGNOSTIC_KEYS})
    components["length"] = _length_score(len(answer), len(ground_truth_values))
    if len(answer) < len(ground_truth_values) or not ground_truth_values:
        return components

    prediction = [float(value) for value in answer[: len(ground_truth_values)]]
    if not _all_finite(prediction):
        components["explosion_penalty"] = -0.8
        return components

    norm_prediction, norm_gt = reward_norm(prediction, ground_truth_values)
    pred_nmse = mean_squared_error(norm_gt, norm_prediction)
    components["pred_nmse"] = _finite_or_zero(pred_nmse)

    context = _reward_context(extra_info, prediction_length=len(ground_truth_values))
    history = context["history"]
    blind_baselines = _build_blind_baselines(history, len(ground_truth_values))
    blind_metrics = _baseline_metrics(blind_baselines, ground_truth_values)
    _blind_name, blind_nmse, blind_prediction = _best_baseline(blind_metrics)
    if not np.isfinite(blind_nmse):
        blind_nmse = max(float(np.var(norm_gt)), 1.0)
        blind_prediction = [float(np.mean(ground_truth_values))] * len(ground_truth_values)
    components["blind_ref_nmse"] = _finite_or_zero(blind_nmse)

    exo_nmse = _exo_ref_nmse(extra_info)
    exo_gain = _relative_gain(blind_nmse, exo_nmse)
    gate_on = exo_gain >= 0.05 and np.isfinite(exo_nmse) and np.isfinite(blind_nmse)
    components["exo_ref_nmse"] = _finite_or_zero(exo_nmse)
    components["exo_gain"] = _finite_or_zero(exo_gain)
    components["covariate_gate"] = 1.0 if gate_on else 0.0
    components["value_mode_gated"] = 1.0 if gate_on else 0.0

    if gate_on:
        components["value"] = _gated_skill_score(pred_nmse, blind_nmse, exo_nmse)
    else:
        components["value"] = _log_skill_score(pred_nmse, blind_nmse)
    components["flat_penalty"] = _flat_penalty(prediction, ground_truth_values)
    components["copy_history_penalty"] = _copy_history_penalty(
        prediction=prediction,
        blind_prediction=blind_prediction,
        exo_gain=exo_gain,
    )
    components["repeat24_drift_penalty"] = _repeat_cycle_drift_penalty(
        prediction=prediction,
        ground_truth=ground_truth_values,
        cycle=CYCLE_LENGTH,
    )
    components["scale_penalty"] = _scale_penalty(prediction, ground_truth_values)
    components["explosion_penalty"] = _explosion_penalty(prediction, ground_truth_values)

    # Keep shape rewards as small diagnostics. They must not rescue a poor value forecast.
    if components["value"] > -0.5 and components["explosion_penalty"] == 0.0:
        components["change_point"] = 0.25 * compute_change_point_score(
            solution_str,
            ground_truth,
            attr=attr,
        )
        components["season_trend"] = 0.25 * compute_answer_score_season_trend(
            solution_str,
            ground_truth,
            attr=attr,
        )
    return components


def _extract_reward_payload(text: str) -> str:
    text = str(text)
    answer = extract_answer(text).strip()
    if answer:
        return answer
    return text


def _resolve_attr(extra_info=None) -> str | None:
    info = dict(extra_info or {})
    channel_names = info.get("channel_names")
    if isinstance(channel_names, (list, tuple)) and channel_names:
        return str(channel_names[0])

    metadata = info.get("metadata")
    if isinstance(metadata, dict):
        for key in ("target_channel", "attr"):
            value = metadata.get(key)
            if value:
                return str(value)

    return None


def _length_score(answer_len: int, target_len: int) -> float:
    if target_len <= 0:
        return 0.0
    return float(0.05 * min(max(answer_len, 0), target_len) / target_len)


def _all_finite(values) -> bool:
    array = np.asarray(values, dtype=float)
    return bool(array.size and np.all(np.isfinite(array)))


def _finite_or_zero(value) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return number if np.isfinite(number) else 0.0


def _reward_context(extra_info, *, prediction_length: int) -> dict[str, list[float]]:
    info = dict(extra_info or {})
    return {
        "history": _extract_history_values(info.get("observed_values")),
    }


def _extract_history_values(observed_values) -> list[float]:
    if observed_values is None:
        return []
    array = np.asarray(observed_values, dtype=object)
    values: list[float] = []
    if array.ndim == 0:
        raw_values = [array.item()]
    elif array.ndim == 1:
        raw_values = array.tolist()
    else:
        raw_values = array[:, 0].tolist()
    for value in raw_values:
        if isinstance(value, (list, tuple, np.ndarray)):
            value = value[0] if len(value) else np.nan
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(number):
            values.append(number)
    return values


def _build_blind_baselines(history: list[float], horizon: int) -> dict[str, list[float]]:
    baselines: dict[str, list[float]] = {}
    if horizon <= 0:
        return baselines
    if history:
        baselines["last_value"] = [float(history[-1])] * horizon
    if len(history) >= CYCLE_LENGTH:
        last_cycle = [float(value) for value in history[-CYCLE_LENGTH:]]
        baselines["repeat24"] = [last_cycle[index % CYCLE_LENGTH] for index in range(horizon)]

        buckets: list[list[float]] = [[] for _ in range(CYCLE_LENGTH)]
        offset = len(history) % CYCLE_LENGTH
        for index, value in enumerate(history):
            buckets[index % CYCLE_LENGTH].append(float(value))
        cycle_mean = []
        fallback = float(np.mean(history))
        for step in range(horizon):
            phase = (offset + step) % CYCLE_LENGTH
            bucket = buckets[phase]
            cycle_mean.append(float(np.mean(bucket)) if bucket else fallback)
        baselines["cycle_mean"] = cycle_mean
    return baselines


def _baseline_metrics(
    baselines: dict[str, list[float]],
    ground_truth_values: list[float],
) -> dict[str, tuple[float, list[float]]]:
    metrics: dict[str, tuple[float, list[float]]] = {}
    for name, prediction in baselines.items():
        if len(prediction) < len(ground_truth_values):
            continue
        norm_prediction, norm_gt = reward_norm(prediction[: len(ground_truth_values)], ground_truth_values)
        if not norm_prediction or not norm_gt:
            continue
        metrics[name] = (mean_squared_error(norm_gt, norm_prediction), prediction[: len(ground_truth_values)])
    return metrics


def _best_baseline(metrics: dict[str, tuple[float, list[float]]]) -> tuple[str | None, float, list[float]]:
    if not metrics:
        return None, float("nan"), []
    name, (score, prediction) = min(metrics.items(), key=lambda item: item[1][0])
    return name, float(score), [float(value) for value in prediction]


def _exo_ref_nmse(extra_info) -> float:
    info = dict(extra_info or {})
    references = info.get("reward_references", {})
    if not isinstance(references, dict):
        return float("nan")
    value = references.get("exo_ref_nmse")
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return number if np.isfinite(number) and number >= 0.0 else float("nan")


def _relative_gain(blind_nmse: float, exo_nmse: float) -> float:
    if not np.isfinite(blind_nmse) or not np.isfinite(exo_nmse) or blind_nmse <= EPS:
        return 0.0
    return float(max(0.0, (blind_nmse - exo_nmse) / (blind_nmse + EPS)))


def _log_skill_score(pred_nmse: float, blind_nmse: float) -> float:
    if not np.isfinite(pred_nmse):
        return -0.9
    if pred_nmse <= EPS:
        return 0.9
    if not np.isfinite(blind_nmse):
        blind_nmse = 1.0
    skill = np.log((blind_nmse + EPS) / (pred_nmse + EPS))
    return float(0.9 * np.tanh(skill / SKILL_TEMPERATURE))


def _gated_skill_score(pred_nmse: float, blind_nmse: float, exo_nmse: float) -> float:
    if not np.isfinite(pred_nmse):
        return -0.9
    achievable_gain = max(float(blind_nmse - exo_nmse), EPS)
    skill = (float(blind_nmse) - float(pred_nmse)) / achievable_gain
    return float(0.9 * np.tanh(skill / SKILL_TEMPERATURE))


def _flat_penalty(prediction: list[float], ground_truth_values: list[float]) -> float:
    gt_std = float(np.std(np.asarray(ground_truth_values, dtype=float)))
    if gt_std <= EPS:
        return 0.0
    pred_std = float(np.std(np.asarray(prediction, dtype=float)))
    return float(-0.10 * max(0.0, 1.0 - pred_std / (gt_std + EPS)))


def _copy_history_penalty(
    *,
    prediction: list[float],
    blind_prediction: list[float],
    exo_gain: float,
) -> float:
    if exo_gain < 0.05 or not blind_prediction:
        return 0.0
    pred = np.asarray(prediction, dtype=float)
    blind = np.asarray(blind_prediction[: len(prediction)], dtype=float)
    scale = max(float(np.std(pred)), float(np.std(blind)), float(np.mean(np.abs(blind))), 1.0)
    similarity = float(np.mean(np.abs(pred - blind)) / (scale + EPS))
    if similarity >= 0.05:
        return 0.0
    return float(-0.10 * min(1.0, exo_gain))


def _repeat_cycle_drift_penalty(
    *,
    prediction: list[float],
    ground_truth: list[float],
    cycle: int,
) -> float:
    if cycle <= 0 or len(prediction) < cycle * 2 or len(ground_truth) < cycle * 2:
        return 0.0
    pred_blocks = _cycle_blocks(prediction, cycle)
    gt_blocks = _cycle_blocks(ground_truth, cycle)
    if len(pred_blocks) < 2 or len(gt_blocks) < 2:
        return 0.0
    pred_drift = _mean_consecutive_block_delta(pred_blocks)
    gt_drift = _mean_consecutive_block_delta(gt_blocks)
    gt_scale = max(float(np.std(np.asarray(ground_truth, dtype=float))), 1.0)
    normalized_gap = max(0.0, gt_drift - pred_drift) / (gt_scale + EPS)
    return float(-0.10 * min(1.0, normalized_gap))


def _cycle_blocks(values: list[float], cycle: int) -> list[np.ndarray]:
    full = len(values) // cycle
    return [
        np.asarray(values[index * cycle : (index + 1) * cycle], dtype=float)
        for index in range(full)
    ]


def _mean_consecutive_block_delta(blocks: list[np.ndarray]) -> float:
    if len(blocks) < 2:
        return 0.0
    deltas = [
        float(np.mean(np.abs(blocks[index + 1] - blocks[index])))
        for index in range(len(blocks) - 1)
    ]
    return float(np.mean(deltas)) if deltas else 0.0


def _scale_penalty(prediction: list[float], ground_truth_values: list[float]) -> float:
    pred = np.asarray(prediction, dtype=float)
    gt = np.asarray(ground_truth_values, dtype=float)
    gt_level = max(float(np.mean(np.abs(gt))), float(np.std(gt)), 1.0)
    pred_level = float(np.mean(np.abs(pred)))
    ratio = pred_level / (gt_level + EPS)
    if ratio <= 10.0:
        return 0.0
    return float(-0.20 * min(1.0, np.log10(ratio) / 3.0))


def _explosion_penalty(prediction: list[float], ground_truth_values: list[float]) -> float:
    pred = np.asarray(prediction, dtype=float)
    if not np.all(np.isfinite(pred)):
        return -0.8
    gt = np.asarray(ground_truth_values, dtype=float)
    gt_level = max(float(np.max(np.abs(gt))) if gt.size else 0.0, float(np.std(gt)) if gt.size else 0.0, 1.0)
    max_abs = float(np.max(np.abs(pred))) if pred.size else 0.0
    ratio = max_abs / (gt_level + EPS)
    if ratio <= 100.0:
        return 0.0
    return float(-0.80 * min(1.0, np.log10(ratio / 100.0 + 1.0)))
