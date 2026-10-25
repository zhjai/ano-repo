"""Strict parser for the teacher/student ``t0.v1`` Context Workspace.

The parser is deliberately independent from any LLM client.  It accepts only
the issue-time, label-free JSON contract and maps it to the runtime
``ContextWorkspace``.  Action fields, forecasts, target-derived fields and
unknown keys fail closed.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Mapping, Sequence

from castfactory.agentic.contracts import ContextDriver, ContextWorkspace


class T0ValidationError(ValueError):
    """Raised when a t0.v1 response violates the workspace contract."""


_ROOT_KEYS = {"schema_version", "drivers", "global_uncertainty", "evidence_ids_used"}
_DRIVER_KEYS = {"id", "name", "evidence", "mechanism", "relevance", "impact_windows", "interactions", "uncertainty"}
_EVIDENCE_KEYS = {"evidence_id", "kind", "as_of", "quote_or_feature"}
_INTERACTION_KEYS = {"with_driver_id", "description"}
_KINDS = {"history", "covariate", "calendar", "context"}
_RELEVANCE = {"high", "medium", "low"}
_EFFECTS = {"up", "down", "mixed", "uncertain"}
_FORBIDDEN = {
    "action", "spans", "segment_count", "forecast", "forecast_values", "prediction",
    "target", "future_values", "reward", "score", "label", "gold_turning_point",
    "effect_mask", "reasoning_content",
}
_MAX_DRIVERS = 16
_MAX_EVIDENCE_PER_DRIVER = 16
_MAX_WINDOWS_PER_DRIVER = 16
_MAX_INTERACTIONS_PER_DRIVER = 16
_MAX_TEXT = 2048


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise T0ValidationError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_non_finite(value: str) -> Any:
    raise T0ValidationError(f"non-finite JSON number is not allowed: {value}")


def load_t0_json(text: str) -> Any:
    """Load exactly one strict, unfenced JSON value for the T0 contract."""

    if not isinstance(text, str) or not text.strip():
        raise T0ValidationError("teacher response must be non-empty text")
    stripped = text.strip()
    if stripped.startswith("```"):
        raise T0ValidationError("markdown fences are not allowed")
    decoder = json.JSONDecoder(
        object_pairs_hook=_reject_duplicate_keys,
        parse_constant=_reject_non_finite,
    )
    try:
        payload, end = decoder.raw_decode(stripped)
    except json.JSONDecodeError as exc:
        raise T0ValidationError("teacher response is not valid JSON") from exc
    if stripped[end:].strip():
        raise T0ValidationError("teacher response contains trailing content")
    return payload


def _bounded_text(value: Any, name: str, *, maximum: int = _MAX_TEXT) -> str:
    if not isinstance(value, str) or not value.strip():
        raise T0ValidationError(f"{name} must be a non-empty string")
    if len(value) > maximum:
        raise T0ValidationError(f"{name} exceeds {maximum} characters")
    return value


def _iso(value: Any, name: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise T0ValidationError(f"{name} must be a non-empty ISO-8601 string")
    try:
        return datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError as exc:
        raise T0ValidationError(f"{name} must be ISO-8601") from exc


def _keys(value: Mapping[str, Any], allowed: set[str], name: str) -> None:
    unknown = set(value) - allowed
    if unknown:
        raise T0ValidationError(f"{name} contains unknown keys: {sorted(map(str, unknown))}")
    forbidden = set(value) & _FORBIDDEN
    if forbidden:
        raise T0ValidationError(f"{name} contains forbidden label/action keys: {sorted(forbidden)}")


def parse_t0_workspace(
    payload: Mapping[str, Any],
    *,
    horizon: int,
    evidence_manifest: Sequence[Mapping[str, Any]],
    issue_time: Any,
) -> ContextWorkspace:
    """Validate and convert one strict t0.v1 JSON object.

    ``evidence_manifest`` is the immutable source record. Every evidence
    reference must match its declared kind and as-of timestamp exactly.
    """

    if not isinstance(payload, Mapping):
        raise T0ValidationError("t0 payload must be a JSON object")
    _keys(payload, _ROOT_KEYS, "t0 payload")
    if payload.get("schema_version") != "t0.v1":
        raise T0ValidationError("unsupported t0 schema_version")
    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon <= 0:
        raise T0ValidationError("horizon must be a positive integer")
    if not isinstance(payload.get("drivers"), list):
        raise T0ValidationError("drivers must be a list")
    _bounded_text(payload.get("global_uncertainty"), "global_uncertainty")
    if len(payload["drivers"]) > _MAX_DRIVERS:
        raise T0ValidationError(f"drivers exceeds {_MAX_DRIVERS} items")
    declared_ids = payload.get("evidence_ids_used")
    if not isinstance(declared_ids, list) or any(not isinstance(item, str) or not item for item in declared_ids):
        raise T0ValidationError("evidence_ids_used must be a list of non-empty strings")
    declared_ids = list(dict.fromkeys(declared_ids))
    if evidence_manifest is None or issue_time is None:
        raise T0ValidationError("evidence_manifest and issue_time are mandatory")
    issue_dt = _iso(issue_time, "issue_time")
    evidence_by_id: dict[str, Mapping[str, Any]] = {}
    for item in evidence_manifest:
        if not isinstance(item, Mapping):
            raise T0ValidationError("evidence_manifest entries must be objects")
        evidence_id = str(item.get("evidence_id", ""))
        kind = str(item.get("kind", ""))
        as_of = item.get("as_of")
        if not evidence_id or evidence_id in evidence_by_id:
            raise T0ValidationError("evidence_manifest ids must be non-empty and unique")
        if kind not in _KINDS:
            raise T0ValidationError("evidence_manifest kind is invalid")
        evidence_dt = _iso(as_of, "evidence_manifest.as_of")
        try:
            is_future = evidence_dt > issue_dt
        except TypeError as exc:
            raise T0ValidationError(
                "evidence_manifest.as_of and issue_time must use compatible timezones"
            ) from exc
        if is_future:
            raise T0ValidationError("evidence_manifest is not available at issue_time")
        evidence_by_id[evidence_id] = item
    allow = set(evidence_by_id)
    if not set(declared_ids).issubset(allow):
        raise T0ValidationError("evidence_ids_used contains an unknown source_id")
    drivers: list[ContextDriver] = []
    evidence_seen: list[str] = []
    raw_driver_ids: list[str] = []
    for index, raw in enumerate(payload["drivers"]):
        if not isinstance(raw, Mapping):
            raise T0ValidationError(f"drivers[{index}] must be an object")
        driver_id = _bounded_text(raw.get("id"), f"drivers[{index}].id", maximum=128)
        raw_driver_ids.append(driver_id)
    if len(set(raw_driver_ids)) != len(raw_driver_ids):
        raise T0ValidationError("driver ids must be unique")
    driver_id_set = set(raw_driver_ids)
    for index, raw in enumerate(payload["drivers"]):
        _keys(raw, _DRIVER_KEYS, f"drivers[{index}]")
        for key in ("id", "name", "mechanism", "relevance", "uncertainty"):
            _bounded_text(raw.get(key), f"drivers[{index}].{key}")
        if raw["relevance"] not in _RELEVANCE:
            raise T0ValidationError(f"drivers[{index}].relevance is invalid")
        evidence = raw.get("evidence")
        if not isinstance(evidence, list):
            raise T0ValidationError(f"drivers[{index}].evidence must be a list")
        if not evidence or len(evidence) > _MAX_EVIDENCE_PER_DRIVER:
            raise T0ValidationError(
                f"drivers[{index}].evidence must contain 1..{_MAX_EVIDENCE_PER_DRIVER} items"
            )
        evidence_ids: list[str] = []
        for e_index, evidence_raw in enumerate(evidence):
            if not isinstance(evidence_raw, Mapping):
                raise T0ValidationError(f"drivers[{index}].evidence[{e_index}] must be an object")
            _keys(evidence_raw, _EVIDENCE_KEYS, f"drivers[{index}].evidence[{e_index}]")
            for key in _EVIDENCE_KEYS:
                _bounded_text(evidence_raw.get(key), f"evidence.{key}")
            if evidence_raw["kind"] not in _KINDS:
                raise T0ValidationError(f"unsupported evidence kind: {evidence_raw['kind']}")
            evidence_id = evidence_raw["evidence_id"]
            if evidence_id not in allow:
                raise T0ValidationError(f"unknown evidence_id: {evidence_id}")
            expected = evidence_by_id[evidence_id]
            if evidence_raw["kind"] != expected["kind"]:
                raise T0ValidationError("evidence kind does not match immutable manifest")
            if evidence_raw["as_of"] != expected["as_of"]:
                raise T0ValidationError("evidence as_of does not match immutable manifest")
            evidence_dt = _iso(evidence_raw["as_of"], "evidence.as_of")
            try:
                is_future = evidence_dt > issue_dt
            except TypeError as exc:
                raise T0ValidationError("evidence.as_of and issue_time must use compatible timezones") from exc
            if is_future:
                raise T0ValidationError("evidence is not available at issue_time")
            evidence_ids.append(evidence_id)
            evidence_seen.append(evidence_id)
        windows = raw.get("impact_windows")
        if not isinstance(windows, list) or not windows or len(windows) > _MAX_WINDOWS_PER_DRIVER:
            raise T0ValidationError(
                f"drivers[{index}].impact_windows must contain 1..{_MAX_WINDOWS_PER_DRIVER} items"
            )
        parsed_windows: list[tuple[int, int]] = []
        effects: list[str] = []
        for w_index, window in enumerate(windows):
            if not isinstance(window, Mapping) or set(window) != {"start", "end", "effect"}:
                raise T0ValidationError(f"drivers[{index}].impact_windows[{w_index}] has invalid keys")
            start, end = window["start"], window["end"]
            if isinstance(start, bool) or isinstance(end, bool) or not isinstance(start, int) or not isinstance(end, int):
                raise T0ValidationError("impact window bounds must be integers")
            if not 0 <= start < end <= horizon:
                raise T0ValidationError("impact window is outside the forecast horizon")
            if window["effect"] not in _EFFECTS:
                raise T0ValidationError("impact window effect is invalid")
            parsed_windows.append((start, end))
            effects.append(window["effect"])
        interactions = raw.get("interactions")
        if not isinstance(interactions, list) or len(interactions) > _MAX_INTERACTIONS_PER_DRIVER:
            raise T0ValidationError("interactions must be a bounded list")
        for interaction in interactions:
            if not isinstance(interaction, Mapping):
                raise T0ValidationError("interactions must contain objects")
            _keys(interaction, _INTERACTION_KEYS, "interaction")
            target_id = _bounded_text(interaction.get("with_driver_id"), "interaction.with_driver_id", maximum=128)
            description = _bounded_text(interaction.get("description"), "interaction.description")
            if target_id not in driver_id_set:
                raise T0ValidationError(f"interaction references unknown driver id: {target_id}")
            if target_id == raw["id"]:
                raise T0ValidationError("a driver cannot interact with itself")
        drivers.append(
            ContextDriver(
                name=raw["name"],
                relevance=raw["relevance"],
                direction=effects[0] if len(set(effects)) == 1 else "mixed",
                mechanism=raw["mechanism"],
                impact_windows=tuple(parsed_windows),
                uncertainty=raw["uncertainty"],
                evidence=tuple(evidence_ids),
            )
        )
    if set(declared_ids) != set(evidence_seen):
        raise T0ValidationError("evidence_ids_used must exactly match referenced evidence")
    interactions = []
    for raw in payload["drivers"]:
        for interaction in raw["interactions"]:
            interactions.append(
                f"{raw['id']}->{interaction['with_driver_id']}: {interaction['description']}"
            )
    metadata = {
        "schema_version": "t0.v1",
        "global_uncertainty": payload["global_uncertainty"],
        "evidence_ids_used": tuple(declared_ids),
    }
    return ContextWorkspace(
        drivers=tuple(drivers),
        interactions=tuple(dict.fromkeys(interactions)),
        metadata=metadata,
    )


def parse_t0_json(
    text: str,
    *,
    horizon: int,
    evidence_manifest: Sequence[Mapping[str, Any]],
    issue_time: Any,
) -> ContextWorkspace:
    """Parse exactly one unfenced JSON object and validate it as t0.v1."""

    payload = load_t0_json(text)
    return parse_t0_workspace(
        payload,
        horizon=horizon,
        evidence_manifest=evidence_manifest,
        issue_time=issue_time,
    )


__all__ = ["T0ValidationError", "load_t0_json", "parse_t0_json", "parse_t0_workspace"]
