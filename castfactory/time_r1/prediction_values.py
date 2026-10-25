"""Original generic numeric extraction helpers, separated from experiment orchestration.

Strict Reasoning rewards still validate their explicit answer grammar first.
"""
from __future__ import annotations
import json
import re
from json import JSONDecodeError
from typing import Iterable
NUMBER_PATTERN = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")
VALUE_KEY_PATTERN = re.compile(r"^v(\d+)$", re.IGNORECASE)

def extract_prediction_values(text: str, attr: str | None = None) -> list[float]:
    for candidate in _prediction_candidates(text):
        values = _extract_structured_prediction_values(candidate, attr)
        if values:
            return values
        values = _extract_table_prediction_values(candidate)
        if values:
            return values
    return _extract_table_prediction_values(text)


def _prediction_candidates(text: str) -> list[str]:
    candidates: list[str] = []

    answer_blocks = re.findall(r"<answer>([\s\S]*?)</answer>", text, flags=re.IGNORECASE)
    candidates.extend(block.strip() for block in answer_blocks if block.strip())

    if "<answer>" in text:
        tail = text.split("<answer>", 1)[1].strip()
        if tail:
            candidates.append(tail)

    if "</think>" in text:
        tail = text.split("</think>", 1)[1].strip()
        if tail:
            candidates.append(tail)

    fenced_blocks = re.findall(r"```(?:[A-Za-z0-9_+-]+)?\s*([\s\S]*?)```", text)
    candidates.extend(block.strip() for block in fenced_blocks if block.strip())

    stripped = text.strip()
    if stripped:
        candidates.append(stripped)

    ordered: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        ordered.append(candidate)
    return ordered


def _extract_structured_prediction_values(text: str, attr: str | None = None) -> list[float]:
    stripped = text.strip()
    if not stripped:
        return []

    for payload in _iter_json_payloads(stripped):
        values = _extract_prediction_from_payload(payload, attr)
        if values:
            return values

    field_names = [attr] if attr else []
    field_names.extend(("load", "value", "prediction"))
    for field_name in field_names:
        values = _extract_named_field_sequence(stripped, field_name)
        if values:
            return values

    values = _extract_v_key_sequence(stripped)
    if values:
        return values

    values = _extract_forecast_number_array(stripped)
    if values:
        return values

    return []


def _iter_json_payloads(text: str) -> Iterable[object]:
    decoder = json.JSONDecoder()

    try:
        yield json.loads(text)
    except JSONDecodeError:
        pass

    for match in re.finditer(r"[\[{]", text):
        try:
            payload, _ = decoder.raw_decode(text[match.start() :])
        except JSONDecodeError:
            continue
        yield payload


def _extract_prediction_from_payload(payload: object, attr: str | None = None) -> list[float]:
    if isinstance(payload, list):
        numeric_values = [_coerce_float(value) for value in payload]
        if payload and all(value is not None for value in numeric_values):
            return [float(value) for value in numeric_values if value is not None]

        if len(payload) == 1:
            values = _extract_prediction_from_payload(payload[0], attr)
            if values:
                return values

        load_values: list[float] = []
        for item in payload:
            if not isinstance(item, dict):
                load_values = []
                break
            load_value = _extract_single_value_from_mapping(item, attr)
            if load_value is None:
                load_values = []
                break
            load_values.append(load_value)
        if load_values:
            return load_values

        for item in payload:
            values = _extract_prediction_from_payload(item, attr)
            if len(values) > 1:
                return values
        return []

    if isinstance(payload, dict):
        if "forecast" in payload:
            values = _extract_prediction_from_payload(payload["forecast"], attr)
            if values:
                return values

        value_pairs: list[tuple[int, float]] = []
        for key, value in payload.items():
            match = VALUE_KEY_PATTERN.fullmatch(str(key))
            if match is None:
                continue
            numeric_value = _coerce_float(value)
            if numeric_value is None:
                continue
            value_pairs.append((int(match.group(1)), numeric_value))
        if value_pairs:
            return [value for _, value in sorted(value_pairs)]

        for value in payload.values():
            values = _extract_prediction_from_payload(value, attr)
            if len(values) > 1:
                return values

        single_value = _extract_single_value_from_mapping(payload, attr)
        if single_value is not None:
            return [single_value]
        return []

    numeric_value = _coerce_float(payload)
    if numeric_value is not None:
        return [numeric_value]
    return []


def _normalize_field_name(name: object) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def _extract_single_value_from_mapping(payload: dict, attr: str | None = None) -> float | None:
    if attr:
        target = _normalize_field_name(attr)
        for key, value in payload.items():
            if _normalize_field_name(key) == target:
                numeric_value = _coerce_float(value)
                if numeric_value is not None:
                    return numeric_value

    for key in ("load", "value", "prediction", "forecast", "target", "y", "yhat", "y_pred"):
        if key not in payload:
            continue
        numeric_value = _coerce_float(payload[key])
        if numeric_value is not None:
            return numeric_value

    filtered_values = [
        _coerce_float(value)
        for key, value in payload.items()
        if str(key).lower() not in {"timestamp", "time", "date", "datetime", "step", "index"}
    ]
    numeric_values = [value for value in filtered_values if value is not None]
    if len(numeric_values) == 1:
        return float(numeric_values[0])
    return None


def _coerce_float(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        try:
            return float(value)
        except OverflowError:
            return None
    if isinstance(value, str):
        stripped = value.strip()
        if NUMBER_PATTERN.fullmatch(stripped):
            try:
                return float(stripped)
            except (OverflowError, ValueError):
                return None
    return None


def _extract_named_field_sequence(text: str, field_name: str) -> list[float]:
    pattern = re.compile(
        rf'["\']{re.escape(field_name)}["\']\s*:\s*["\']?\s*({NUMBER_PATTERN.pattern})\s*["\']?',
        flags=re.IGNORECASE,
    )
    matches = pattern.findall(text)
    return [float(value) for value in matches]


def _extract_v_key_sequence(text: str) -> list[float]:
    pattern = re.compile(rf'["\']?v(\d+)["\']?\s*:\s*({NUMBER_PATTERN.pattern})', flags=re.IGNORECASE)
    matches = [(int(index), float(value)) for index, value in pattern.findall(text)]
    return [value for _, value in sorted(matches)]


def _extract_forecast_number_array(text: str) -> list[float]:
    match = re.search(r'"forecast"\s*:\s*\[([\s\S]*?)\]', text)
    if match is None:
        return []
    body = match.group(1)
    # Only treat forecast as a flat numeric array here. If the body contains
    # object syntax, timestamps, or keyed fields, prefer the dedicated extractors
    # above instead of mixing timestamp digits into the forecast.
    if "{" in body or ":" in body:
        return []
    return [float(value) for value in NUMBER_PATTERN.findall(body)]


def _extract_table_prediction_values(text: str) -> list[float]:
    block_match = re.search(r"```([\s\S]*?)```", text)
    if block_match is None:
        lines = [line for line in text.splitlines() if "|" in line]
    else:
        lines = block_match.group(1).strip().splitlines()

    values: list[float] = []
    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue
        if line.endswith("|"):
            line = line[:-1].rstrip()
        if "|" in line:
            cells = [cell.strip() for cell in line.strip("|").split("|")]
            candidate = cells[-1] if cells else line
        else:
            candidate = line
        matches = NUMBER_PATTERN.findall(candidate)
        if matches:
            values.append(float(matches[-1]))
    return values
