"""Shared train-only distillation boundary; labels never enter prepared cases."""
from __future__ import annotations

import copy
from datetime import datetime, timedelta
from functools import lru_cache
import json
import math
from pathlib import Path
from typing import Any

from .forecast_input_view import (
    NEW_VERSION, _embedded_objects, project_model_input, render_projected_forecast_view,
)
from . import gef_final_public, public_event_view_v2

REPO = Path(__file__).resolve().parents[2]
DEFAULT_TRAIN = REPO / "datasets/load_datasets/legacy_pre_compact_v2/rl/train.jsonl"
PUBLIC_VERSION = "forecast-input-view-v2"
GEF_PUBLIC_VERSION = PUBLIC_VERSION + "+gef-evidence-only-v1"
_GEF_LEGACY_OUTPUT_BLOCK = (
    "Return exactly 96 future target values.\n"
    "You must first conduct reasoning inside <think>...</think>.\n"
    "When you have the final answer, output it inside <answer>...</answer>.\n"
    "Inside <answer>, output a markdown code block with lines in the form "
    "'index | value' from 1 to 96. Please obey the format strictly."
)
_GEF_ANNOUNCED_HEADER = "\n\nAnnounced load programmes:\n"


def _gef_evidence_only(text: str) -> str:
    """Remove only the recognized generated footer from the v4 evidence view.

    The saved original renderer ends in this exact block. Synthetic conversion
    appends the known header and public rule/notices after it. Keep every other
    byte, including qualifications before the block and all appended evidence.
    Similar instructions inside source notices are not generated footers.
    """
    original, separator, notices = text.partition(_GEF_ANNOUNCED_HEADER)
    if original.endswith(_GEF_LEGACY_OUTPUT_BLOCK):
        original = original[:-len(_GEF_LEGACY_OUTPUT_BLOCK)]
        return original + separator + notices
    return text


def _numbers(values: Any, length: int | None = None) -> list:
    if not isinstance(values, list) or (length is not None and len(values) != length):
        raise ValueError("Invalid numeric array length")
    if not values or any(isinstance(v, bool) or not isinstance(v, (int, float))
                         or not math.isfinite(v) for v in values):
        raise ValueError("Numeric arrays require finite JSON numbers")
    return copy.deepcopy(values)


def _clock(values: Any, length: int, name: str) -> list[datetime]:
    if not isinstance(values, list) or len(values) != length:
        raise ValueError(name + " length mismatch")
    try:
        times = [datetime.fromisoformat(v.replace("Z", "+00:00")) for v in values]
        if any(b - a != timedelta(hours=1) for a, b in zip(times, times[1:])):
            raise ValueError("non-hourly")
    except (TypeError, ValueError, AttributeError) as error:
        raise ValueError(name + " must be an ordered hourly timestamp grid") from error
    return times


def validate_case(row: dict) -> None:
    if row.get("split") != "train" or row.get("evaluation_only"):
        raise ValueError("Distillation accepts train cases only")
    validate_public_geometry(row)


def validate_public_geometry(row: dict) -> None:
    """Validate physical input geometry without treating eval rows as train."""
    if not isinstance(row.get("case_id"), str) or not row["case_id"]:
        raise ValueError("Missing case identity")
    model = row["model_input"]
    h = model["prediction_length"]
    if isinstance(h, bool) or h not in (24, 96):
        raise ValueError("Distillation supports H24/H96 hourly cases")
    if model.get("freq", "").lower() not in {"h", "1h", "60min"}:
        raise ValueError("Unsupported target cadence")
    history = model["target_history"]
    values = _numbers(history["values"])
    past = _clock(history["timestamps"], len(values), "History")
    future = _clock(model["future_timestamps"], h, "Future")
    try:
        if future[0] - past[-1] != timedelta(hours=1):
            raise ValueError("History/future boundary is not contiguous")
    except TypeError as error:
        raise ValueError("History/future timezone mismatch") from error
    if row.get("future_points", h) != h or model.get("context_length", len(values)) != len(values):
        raise ValueError("Declared window geometry mismatch")


def canonical_labels(row: dict) -> list:
    """Private supervisor accessor; never call from an actor message builder."""
    validate_case(row)
    label = row["supervised_label"]
    if label["timestamps"] != row["model_input"]["future_timestamps"]:
        raise ValueError("Canonical labels and public timestamps disagree")
    return _numbers(label["values"], row["model_input"]["prediction_length"])


def load_train_cases(path: str | Path = DEFAULT_TRAIN) -> list[dict]:
    result, seen = [], set()
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            canonical_labels(row)
            if row["case_id"] in seen:
                raise ValueError("Duplicate train case identity")
            seen.add(row["case_id"])
            result.append(row)
    if not result:
        raise ValueError("Empty training manifest")
    return result


@lru_cache(maxsize=1)
def _tokenizer():
    from tokenizers import Tokenizer
    project = REPO.parents[2]
    return Tokenizer.from_file(str(project.parent / "models/Qwen3-0.6B/tokenizer.json"))


def _token_count(text: str) -> int:
    return len(_tokenizer().encode(text, add_special_tokens=False).ids)


def _manifest(public: dict, basis: Any, *, gef: bool = False) -> list[dict]:
    """Citations address the SAME aliased tree rendered to the model.

    A location is a locator, not another copy or semantic summary of an event.
    Null clocks are preserved; origin is never substituted for publication.
    """
    entries, ids = [], set()
    records = {}

    def add(identifier, kind, value, location):
        identifier = str(identifier)
        if identifier in ids:
            if records[identifier] == value:
                # Exact duplicate carriers can share a citation, but revisions
                # with the same native identity must remain separately citable.
                return
            identifier = f"{identifier}@{location}"
        if identifier in ids:
            return
        ids.add(identifier)
        records[identifier] = copy.deepcopy(value)
        asof = next((value[k] for k in (
            "as_of", "publication_time", "issued_at", "sent", "available_time",
            "available_at_utc", "available_at", "published_at",
        ) if value.get(k) is not None), None)
        availability = next((value[k] for k in (
            "availability_basis", "availability", "historical_availability", "research_mode",
        ) if value.get(k) is not None), basis)
        entries.append({"evidence_id": identifier, "kind": kind, "as_of": copy.deepcopy(asof),
                        "availability_basis": copy.deepcopy(availability), "location": location})

    add("HISTORY", "history", {}, "Observed window table" if gef else "target_history")
    if gef:
        add("KNOWN_COVARIATES", "covariate", {}, "Forecast window table: supplied calendar and temperature summaries")
    else:
        for field in ("exogenous_histories", "future_known_exogenous"):
            for i, value in enumerate(public.get(field) or []):
                if isinstance(value, dict):
                    add(f"{field}[{i}]", "covariate", value, f"{field}[{i}]")

    def walk(value, path):
        if isinstance(value, dict):
            native_id = next((value[k] for k in ("event_id", "notice_id", "native_event_id", "record_id")
                              if isinstance(value.get(k), (str, int))), None)
            if native_id is not None:
                add(native_id, "context", value, path)
            elif path.startswith("temporal_text_events[") and path.count("[") == 1:
                add(path, "context", value, path)
            for key, child in value.items():
                walk(child, f"{path}.{key}" if path else key)
        elif isinstance(value, list):
            for i, child in enumerate(value):
                walk(child, f"{path}[{i}]")
        elif isinstance(value, str):
            # The cleaned source may carry event objects inside Context prose.
            # Inspect that SAME public/aliased text, never the raw private tree.
            for start, _, embedded in _embedded_objects(value):
                walk(embedded, f"{path}.embedded_json[{start}]")

    walk(public, "")
    # Household tariff/Context may have no native event ID. The evidence exists
    # nonetheless; its public carrier is citable without inventing episodes.
    for key in ("context_text", "source_context"):
        if public.get(key):
            add(key, "context", public[key] if isinstance(public[key], dict) else {}, key)
    return entries


def prepare_case(row: dict, compressed: bool = True) -> dict:
    validate_case(row)
    return _prepare_public_case(row, compressed)


def prepare_public_case(row: dict, compressed: bool = True) -> dict:
    """Render shared public evidence for the row's actual split; no teacher call.

    GEF source-binding checks remain in the existing renderer. Only its public
    projection is returned; private label/sidecar data never enter messages.
    Teacher collection must continue to use prepare_case, which is train-only.
    """
    validate_public_geometry(row)
    return _prepare_public_case(row, compressed)


def _prepare_public_case(row: dict, compressed: bool) -> dict:
    model = row["model_input"]
    basis = model.get("research_mode")
    if basis is None:
        basis = (model.get("source_context") or {}).get("historical_availability", "unknown")
    gef = row["source"] in {"GEF_original_only", "GEF_synthetic_final"}
    if gef:
        text = _gef_evidence_only(gef_final_public.render_public(row, compressed=compressed))
        visible = {"gef_synthetic_public": copy.deepcopy(model.get("gef_synthetic_public", {}))}
    else:
        projected, _ = project_model_input(model, policy=NEW_VERSION)
        if compressed:
            view = render_projected_forecast_view(projected, source=row["source"], count_units=_token_count)
            text, visible = view.text, view.public
        else:
            visible, _ = public_event_view_v2.project_public(projected, source=row["source"])
            text = json.dumps(visible, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    geometry = {"H": model["prediction_length"], "frequency": model["freq"],
                "unit": model["target_history"].get("unit", "unknown"),
                "future_timestamps": copy.deepcopy(model["future_timestamps"]),
                "history_length": len(model["target_history"]["values"]),
                "origin_time": model.get("origin_time"), "issue_time": model.get("issue_time"),
                "target_start": model.get("target_start"), "availability_basis": copy.deepcopy(basis)}
    return {"case_id": row["case_id"], "source": row["source"], "split": row["split"],
            "public_case": text, "geometry": geometry,
            "evidence_manifest": _manifest(visible, basis, gef=gef),
            "history_values": _numbers(model["target_history"]["values"]),
            "public_input_version": GEF_PUBLIC_VERSION if gef else PUBLIC_VERSION}
