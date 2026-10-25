"""Role-independent JSON rows over the existing forecast-clean public evidence.

This is an opt-in serializer, not a new source adapter or a split selector.
Non-GEF data go through the same v2 cleaning and alias projection as the current
distillation view. GEF uses its checked saved five-temperature-summary tables;
the canonical 25 station arrays are deliberately not expanded into this view.
No supervisor field is copied into the result.
"""
from __future__ import annotations

import copy
from datetime import datetime
import json
from typing import Any

from .distillation_case import prepare_public_case, validate_public_geometry
from .forecast_input_view import NEW_VERSION, _embedded_objects, project_model_input
from .gef_final_public import _table
from . import public_event_view_v2


_GEF_SOURCES = {"GEF_original_only", "GEF_synthetic_final"}
_CHANNEL_ARRAYS = {"values", "timestamps"}
_TASK_FIELDS = (
    "dataset_id", "domain", "track", "series_id", "freq", "cutoff_time",
    "context_length", "prediction_length", "input_protocol", "research_mode",
    "origin_time", "issue_time", "target_start",
)


def _table_rows(columns: list[str], arrays: list[list], positions: range) -> dict:
    count = len(positions)
    if any(not isinstance(a, list) or len(a) != count for a in arrays):
        raise ValueError("Row table and source channel lengths disagree")
    if len(columns) != len(set(columns)):
        raise ValueError("Duplicate or reserved row column name")
    return {"columns": ["position"] + columns,
            "rows": [[p] + [a[i] for a in arrays] for i, p in enumerate(positions)]}


def _channel_metadata(channel: dict) -> dict:
    # Masks and channel-level availability remain exact, including explicit
    # null publication clocks, enabled flags, and research-only qualifications.
    return {k: copy.deepcopy(v) for k, v in channel.items() if k not in _CHANNEL_ARRAYS}


def _embedded_context(text: str) -> Any:
    """Decode only the designated household structural text carriers.

    Native notice bodies are never passed here. Surrounding text survives as
    ordered literal parts, so a JSON-looking fragment cannot erase a qualifier.
    """
    found = _embedded_objects(text)
    if not found:
        return text
    parts, offset = [], 0
    for start, end, value in found:
        if start > offset:
            parts.append(text[offset:start])
        if (isinstance(value, dict) and isinstance(value.get("timestamps"), list)
                and isinstance(value.get("columns"), dict)):
            times, channels = value["timestamps"], value["columns"]
            if set(value) == {"timestamps", "columns"} and all(
                    isinstance(a, list) and len(a) == len(times) for a in channels.values()):
                value = {"columns": ["timestamp"] + list(channels),
                         "rows": [[t] + [a[i] for a in channels.values()]
                                  for i, t in enumerate(times)]}
        parts.append(value)
        offset = end
    if offset < len(text):
        parts.append(text[offset:])
    return {"parts": parts}


def _context_and_events(value: Any, path: str, events: list[dict]) -> Any:
    """Relocate explicit public event carriers, preserving their source scope."""
    if isinstance(value, dict):
        result = {}
        for key, child in value.items():
            location = path + "." + key if path else key
            if key in {"event_records", "supplemental_notices"} and isinstance(child, list):
                for record in child:
                    events.append({"carrier": location, "content": copy.deepcopy(record)})
            elif key in {"scenario_text", "static_attribute_text"} and isinstance(child, str):
                result[key] = _embedded_context(child)
            else:
                result[key] = _context_and_events(child, location, events)
        return result
    if isinstance(value, list):
        return [_context_and_events(child, f"{path}[{i}]", events)
                for i, child in enumerate(value)]
    return copy.deepcopy(value)


def _native_json_case(row: dict) -> dict:
    clean, _ = project_model_input(row["model_input"], policy=NEW_VERSION)
    public, _ = public_event_view_v2.project_public(clean, source=row["source"])
    history = public["target_history"]
    past, future = history["timestamps"], public["future_timestamps"]
    variables = {"load": {"history": _channel_metadata(history)}}
    historical_names, historical_arrays = ["load"], [history["values"]]
    future_names, future_arrays = [], []
    for scope, clock, names, arrays in (
        ("exogenous_histories", past, historical_names, historical_arrays),
        ("future_known_exogenous", future, future_names, future_arrays),
    ):
        output_scope = "history" if scope == "exogenous_histories" else "future_known"
        for channel in public.get(scope) or []:
            name = channel["name"]
            if name in {"position", "load"} or name in names:
                raise ValueError("Duplicate or reserved source channel name")
            if channel["timestamps"] != clock:
                raise ValueError("Source channel does not share the declared target grid")
            names.append(name)
            arrays.append(channel["values"])
            variables.setdefault(name, {})[output_scope] = _channel_metadata(channel)

    events = []
    for record in public.get("temporal_text_events") or []:
        content = copy.deepcopy(record)
        # This exact ERCOT marker identifies a serialization format, not an
        # announcement revision. Other native version/revision fields survive.
        if row["source"] == "ERCOT" and content.get("version") == "ercot_public_month_v1":
            content.pop("version")
        events.append({"carrier": "temporal_text_events", "content": content})
    allocated = set(_TASK_FIELDS) | {
        "target_history", "exogenous_histories", "future_known_exogenous",
        "future_timestamps", "temporal_text_events",
    }
    context = _context_and_events({k: v for k, v in public.items() if k not in allocated}, "", events)
    # iFlex partial experimental prices explicitly share the historical axis.
    # Keep their nulls/masks/units; never infer a filled or future price channel.
    household = (context.get("source_context") or {}).get("context") or {}
    partial = household.get("partial_historical_covariates")
    if isinstance(partial, dict):
        for name, channel in partial.items():
            if name in {"position", "load"} or name in historical_names:
                raise ValueError("Duplicate partial historical channel")
            if "timestamps" in channel and channel["timestamps"] != past:
                raise ValueError("Partial historical channel clock mismatch")
            historical_names.append(name)
            historical_arrays.append(channel["values"])
            variables.setdefault(name, {})["history"] = _channel_metadata(channel)
        household.pop("partial_historical_covariates")
    return {
        "task": {k: copy.deepcopy(public[k]) for k in _TASK_FIELDS if k in public},
        "grid": {"history_timestamps": copy.deepcopy(past),
                 "future_timestamps": copy.deepcopy(future)},
        "variables": variables,
        "history": _table_rows(historical_names, historical_arrays, range(-len(past), 0)),
        "future_known": _table_rows(future_names, future_arrays, range(len(future)) if future_names else range(0)),
        "context": context,
        "events": events,
    }


def _gef_json_case(row: dict) -> dict:
    # The inherited saved-prompt/source binding validation consults private
    # labels for consistency. Labels are never serialized as public evidence.
    prepared = prepare_public_case(row)
    text, model = prepared["public_case"], row["model_input"]
    past = model["target_history"]["timestamps"]
    future = model["future_timestamps"]
    old_columns, old_rows = _table(text, "Observed window:")
    new_columns, new_rows = _table(text, "Forecast window:")

    def columns_and_arrays(columns, rows, timestamps):
        if len(rows) != len(timestamps):
            raise ValueError("GEF saved table and public grid length mismatch")
        # Saved display clocks are incomplete (no year). Bind each to the
        # existing full source timestamp, never manufacture a timestamp.
        for record, timestamp in zip(rows, timestamps):
            if record[columns.index("time")] != datetime.fromisoformat(timestamp).strftime("%m-%d %H"):
                raise ValueError("GEF saved display clock mismatch")
        names = (["load"] if "target" in columns else []) + [
            c for c in columns if c not in {"step", "time", "target"}]
        arrays = [[json.loads(record[columns.index("target" if name == "load" else name)])
                   for record in rows] for name in names]
        return names, arrays

    history_names, history_arrays = columns_and_arrays(old_columns, old_rows, past)
    future_names, future_arrays = columns_and_arrays(new_columns, new_rows, future)
    if "load" in future_names:
        raise ValueError("GEF saved forecast table exposes target values")
    preamble = text.split("Observed window:\n", 1)[0]
    # The feature descriptions are source semantics; the old step/time legend
    # is replaced by the explicit global positions and full timestamp grid.
    legend = preamble.split("Feature legend:\n", 1)[1].strip().splitlines()
    variables = {"load": {"history": {"name": model["target_history"]["name"],
                                       "unit": model["target_history"].get("unit", "unknown")}}}
    for name in future_names:
        description = next((line[2:].split(": ", 1)[1] for line in legend
                            if line.startswith("- " + name + ": ")), None)
        metadata = {"name": name}
        if description is not None:
            metadata["description"] = description
        # No Celsius/Fahrenheit claim is inferred from the hdd65 threshold.
        if name in {"temp_mean", "temp_min", "temp_max", "hdd65", "cdd65"}:
            metadata["unit"] = "temperature"
        availability = ("oracle_future_observed" if name in {
            "temp_mean", "temp_min", "temp_max", "hdd65", "cdd65"}
                        else "calendar_from_supplied_timestamp")
        variables[name] = {"history": copy.deepcopy(metadata),
                           "future_known": dict(metadata, availability=availability)}
    track = preamble.split("Track guidance:\n", 1)[1].split("Forecasting guidance:\n", 1)[0]
    guidance = preamble.split("Forecasting guidance:\n", 1)[1].split("Feature legend:\n", 1)[0]
    tail = text.split("Forecast window:\n", 1)[1].split("\n\n", 1)[1]
    tail, marker, programmes = tail.partition("\n\nAnnounced load programmes:\n")
    context = {"track_guidance": track.strip(), "forecasting_guidance": guidance.strip(),
               "covariate_qualification": tail.strip()}
    events = []
    if marker:
        rule, separator, notices = programmes.partition("\n\n")
        if not separator:
            raise ValueError("GEF announced programmes lack notice body")
        context["synthetic_programme_rule"] = rule
        # Full public prose survives, including priorities, return laws,
        # administrative updates, cancellations and supersession relations.
        events.append({"carrier": "synthetic_announced_programmes", "content": {
            "synthetic": True, "text": notices,
        }})
    return {
        "task": {k: copy.deepcopy(model[k]) for k in _TASK_FIELDS if k in model},
        "grid": {"history_timestamps": copy.deepcopy(past), "future_timestamps": copy.deepcopy(future)},
        "variables": variables,
        "history": _table_rows(history_names, history_arrays, range(-len(past), 0)),
        "future_known": _table_rows(future_names, future_arrays, range(len(future))),
        "context": context, "events": events,
    }


def build_public_json_case(raw_shared_row: dict) -> dict:
    """Build one shared JSON evidence object for any split, without labels.

    Row position -1 is the last historical target; future position 0 maps to
    grid.future_timestamps[0]. Native target names/units live in variables.load.
    Known-future arrays contain covariates only. Source-specific availability
    and clock qualifications remain in channel metadata, context and events.
    """
    validate_public_geometry(raw_shared_row)
    result = (_gef_json_case(raw_shared_row) if raw_shared_row["source"] in _GEF_SOURCES
              else _native_json_case(raw_shared_row))
    # Reject nonfinite/non-JSON values instead of silently emitting invalid JSON.
    json.dumps(result, ensure_ascii=False, allow_nan=False)
    return result


def render_public_json_case(raw_shared_row: dict) -> str:
    """Serialize the shared evidence object as JSON, without prompt wrappers."""
    return json.dumps(build_public_json_case(raw_shared_row), ensure_ascii=False,
                      separators=(",", ":"), allow_nan=False)
