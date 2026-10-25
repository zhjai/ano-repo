"""Shared numeric cases and public prompts for the final GEF load protocol.

The saved parent prompt supplies the existing five-column temperature summary.
Only historical target cells are replaced; labels and experimental identities
remain on the supervision/evaluator side of the shared case boundary.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta
import math
from typing import Any, Mapping

from castfactory.data.scaling import build_scaler


SOURCE = "GEF_synthetic_final"
PUBLIC_VERSION = "gef-final-public-v1"
PLAN_FIELDS = {"version", "law", "template_index", "notices"}
NOTICE_FIELDS = {
    "notice_id", "programme", "entity_id", "issued_at", "status_as_of_cutoff",
    "direct_start", "direct_end", "signed_fraction", "return_fraction",
    "recovery_start", "recovery_end", "priority", "authority", "supersedes",
    "text", "description", "version", "connection",
}
PRIVATE_FIELDS = {
    "label", "labels", "supervised_label", "target_future_label", "target_future",
    "evaluator_only", "controlled_synthesis", "family_id", "parent_sample_id",
    "arm", "cell", "held", "ood_axis", "split", "quality", "lineage",
    "delta", "interaction", "actual_q", "realized_energy", "Q", "H0", "Y0", "Y1",
}
MODEL_FIELDS = {
    "sample_id", "dataset_id", "domain", "track", "series_id", "freq", "cutoff_time",
    "context_length", "prediction_length", "target_history", "exogenous_histories",
    "future_known_exogenous", "future_timestamps", "temporal_text_events",
    "context_text", "static_attributes", "input_protocol", "research_mode",
    "normalization", "origin_time", "issue_time", "target_start", "schema_version",
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _dense(values: Any, field: str) -> list[float]:
    _require(isinstance(values, (list, tuple)) and len(values) == 96,
             field + ":requires_96_values")
    _require(all(not isinstance(v, bool) and math.isfinite(float(v)) and float(v) >= 0
                 for v in values), field + ":requires_finite_nonnegative_values")
    return list(values)


def _screen(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _screen(child) for key, child in value.items()
                if key not in PRIVATE_FIELDS and not key.endswith(("_path", "_paths", "_sha256"))}
    if isinstance(value, (list, tuple)):
        return [_screen(child) for child in value]
    return deepcopy(value)


def _table(prompt: str, heading: str) -> tuple[list[str], list[list[str]]]:
    marker = heading + "\n"
    _require(prompt.count(marker) == 1, "saved_prompt_table_missing_or_duplicate")
    lines = prompt.split(marker, 1)[1].split("\n\n", 1)[0].splitlines()
    columns = [part.strip() for part in lines[0].split("|")]
    rows = [[part.strip() for part in line.split("|")] for line in lines[1:]]
    _require(len(rows) == 96 and all(len(row) == len(columns) for row in rows),
             "saved_prompt_table_geometry")
    return columns, rows


def _replace_history(prompt: str, original: list[float], history: list[float]) -> str:
    columns, rows = _table(prompt, "Observed window:")
    _require("target" in columns, "saved_prompt_target_column_missing")
    index = columns.index("target")
    _require(all(abs(float(row[index]) - float(value)) <= 5.01e-7
                 for row, value in zip(rows, original)), "saved_prompt_history_mismatch")
    marker = "Observed window:\n"
    before, rest = prompt.split(marker, 1)
    table, separator, after = rest.partition("\n\n")
    lines = table.splitlines(keepends=True)
    for i, (old, new) in enumerate(zip(original, history), 1):
        if old == new:
            continue
        cells = lines[i].split("|")
        token = cells[index]
        left = token[:len(token) - len(token.lstrip())]
        right = token[len(token.rstrip()):]
        cells[index] = left + repr(float(new)) + right
        lines[i] = "|".join(cells)
    return before + marker + "".join(lines) + separator + after


def _validate_plan(plan: Mapping[str, Any], cutoff: str) -> None:
    _require(set(plan) <= PLAN_FIELDS, "private_or_unknown_public_plan_field")
    _require(isinstance(plan.get("notices"), list), "public_plan_notices_missing")
    for notice in plan["notices"]:
        _require(isinstance(notice, Mapping) and set(notice) <= NOTICE_FIELDS,
                 "private_or_unknown_notice_field")
        _require(datetime.fromisoformat(notice["issued_at"]) <= datetime.fromisoformat(cutoff),
                 "notice_issued_after_cutoff")


def _saved_payload(parent_case: Mapping[str, Any]) -> dict[str, Any]:
    """Train/val save RLVR payloads; ID-test saves a canonical prompt override."""
    raw = parent_case["sidecar"]["original_payload"]
    if "prompt" in raw:
        return deepcopy(dict(raw))
    prompt = (raw.get("metadata") or {}).get("prompt_override")
    history = parent_case["model_input"]["target_history"]["values"]
    blocks = parent_case["model_input"].get("exogenous_histories") or []
    return {
        "sample_id": raw["sample_id"], "prompt": prompt, "model_input_text": prompt,
        "label": deepcopy(raw["target_future_label"]["values"]),
        "prediction_length": 96, "channel_names": [raw["target_history"]["name"]]
            + [block["name"] for block in blocks],
        "observed_values": [[value] + [block["values"][i] for block in blocks]
                            for i, value in enumerate(history)],
        "cutoff_time": raw["cutoff_time"],
    }


def canonical_to_shared_case(
    row: Mapping[str, Any], parent_case: Mapping[str, Any], *,
    public_notices: list[str], public_rule: str, evaluator_only: Mapping[str, Any],
    preserve_original: bool = False, public_plan: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Convert one replayed row using a separately supplied original shared case.

    ``row`` supplies actual H/Y and timestamps. The generator supplies only
    announced facts in ``public_plan`` and text; its replay truth stays separate.
    Empty arms must opt in to the checked byte-identical saved prompt path.
    """
    _require(parent_case["source"] == "GEF_original_only", "requires_original_gef_parent_case")
    parent = parent_case["model_input"]
    payload = _saved_payload(parent_case)
    _require(payload["sample_id"] == parent_case["case_id"], "saved_parent_identity_mismatch")
    _require(payload["label"] == parent_case["supervised_label"]["values"], "saved_parent_label_mismatch")
    history = _dense(row["target_history"]["values"], "history")
    future = _dense(row["target_future_label"]["values"], "future")
    original = _dense(parent["target_history"]["values"], "parent_history")
    for key in ("series_id", "track", "domain", "freq", "cutoff_time",
                "exogenous_histories", "future_known_exogenous", "static_attributes"):
        _require(row.get(key) == parent.get(key), "changed_parent_public_field:" + key)
    _require(row["target_history"]["timestamps"] == parent["target_history"]["timestamps"],
             "changed_history_clock")
    label = deepcopy(row["target_future_label"])
    _require(label["timestamps"] == parent_case["supervised_label"]["timestamps"], "changed_future_clock")
    times = row["target_history"]["timestamps"] + label["timestamps"]
    _require(len(times) == 192 and all(datetime.fromisoformat(b) - datetime.fromisoformat(a)
             == timedelta(hours=1) for a, b in zip(times, times[1:])), "non_hourly_gef_clock")
    _require(isinstance(payload.get("prompt"), str) and payload["prompt"].strip(), "missing_saved_prompt")
    _require(all(isinstance(text, str) and text.strip() for text in public_notices), "invalid_public_notice")
    if preserve_original:
        _require(not public_notices and not public_rule and not public_plan, "zero_view_has_synthetic_context")
        _require(history == original and label == parent_case["supervised_label"], "zero_view_changed_history_or_label")
        # Zero arms are a cache contract: return the exact saved object.
        return deepcopy(parent_case)
    else:
        _require(bool(public_notices), "synthetic_view_requires_public_notices")
        _require(public_plan is not None, "synthetic_view_requires_structured_public_plan")
    model = {key: deepcopy(parent.get(key)) for key in MODEL_FIELDS if key in parent}
    model.update(sample_id=row["sample_id"], target_history=deepcopy(row["target_history"]),
                 future_timestamps=deepcopy(label["timestamps"]))
    # Keep the renderer's public prompt in the model envelope as the single
    # consumer-visible representation.
    # Original source_context contains administrative and alias copies. Rebuild
    # it from public fields rather than copying stale statistics or references.
    model["source_context"] = {}
    model["context_text"] = _screen(parent.get("context_text") or {})
    model["temporal_text_events"] = _screen(parent.get("temporal_text_events") or [])
    normalization = model.get("normalization") or {"method": "none", "scope": "series"}
    method = normalization.get("method", normalization.get("name", "none"))
    fitted = build_scaler(method).fit(history).params
    model["normalization"] = {"method": method, "scope": normalization.get("scope", "series")}
    model["history_scale"] = max(sum(abs(float(v)) for v in history) / 96, 1e-8)
    if method not in {"none", "identity", "no_op", "noop"}:
        model["normalization"].update(center=fitted["center"], scale=fitted["scale"])
    prompt = _replace_history(payload["prompt"], original, history)
    if public_plan is not None:
        _validate_plan(public_plan, model["cutoff_time"])
        model["gef_synthetic_public"] = deepcopy(dict(public_plan))
        prompt += "\n\nAnnounced load programmes:\n" + public_rule + "\n\n" + "\n\n".join(public_notices)
    model["public_prompt"] = prompt
    observed = deepcopy(payload["observed_values"])
    _require(len(observed) == 96, "saved_observed_values_geometry")
    for i, value in enumerate(history):
        observed[i][0] = value
    updated_payload = {
        "sample_id": row["sample_id"], "prompt": prompt, "model_input_text": prompt,
        "label": future, "prediction_length": 96, "channel_names": deepcopy(payload["channel_names"]),
        "observed_values": observed, "cutoff_time": model["cutoff_time"],
    }
    sidecar = {
        "original_payload": updated_payload, "canonical_row": deepcopy(dict(row)),
        "evaluator_only": deepcopy(dict(evaluator_only)),
        "parent_case_id": parent_case["case_id"], "saved_parent_prompt": payload["prompt"],
        "saved_parent_label": deepcopy(parent_case["supervised_label"]), "parent_history": original,
        "history_scale": max(sum(abs(float(v)) for v in history) / 96, 1e-8),
        "private_reference_policy": {"optional_uplift_enabled": False,
                                     "reason": "final_case_reference_not_recomputed"},
        "preserve_original": preserve_original,
    }
    case = {
        "schema_version": "castmind-shared-case-v1", "case_id": row["sample_id"],
        "source": "GEF_original_only" if preserve_original else SOURCE,
        "split": row.get("split", parent_case["split"]), "shared_reasoning_agent_case": True,
        "model_input_format": "castmind-shared-model-input-v1", "source_payload_format": PUBLIC_VERSION,
        "history_points": 96, "future_points": 96, "origin_semantics": parent_case["origin_semantics"],
        "model_input": model, "supervised_label": label, "sidecar": sidecar,
    }
    validate_public_case(case)
    return case


def validate_public_case(case: Mapping[str, Any]) -> None:
    """Check physical binding and the public/private boundary before rendering."""
    model, label, sidecar = case["model_input"], case["supervised_label"], case["sidecar"]
    # Existing zero arms are immutable saved cases and predate this renderer.
    if case.get("source") == "GEF_original_only" and "public_prompt" not in model:
        payload = sidecar.get("original_payload", {})
        prompt = payload.get("prompt") or (payload.get("metadata") or {}).get("prompt_override")
        _require(isinstance(prompt, str) and prompt.strip(), "missing_saved_prompt")
        _require(payload.get("sample_id") == case["case_id"], "saved_parent_identity_mismatch")
        saved_label = payload.get("target_future_label")
        _require(saved_label == label if saved_label is not None else payload.get("label") == label["values"],
                 "saved_parent_label_mismatch")
        _require(model["sample_id"] == case["case_id"] and model["future_timestamps"] == label["timestamps"],
                 "legacy_public_identity_or_clock_mismatch")
        _dense(model["target_history"]["values"], "history")
        _dense(label["values"], "future")
        return
    _require(model["sample_id"] == case["case_id"], "case_identity_mismatch")
    _require(model["future_timestamps"] == label["timestamps"], "public_label_clock_mismatch")
    history = _dense(model["target_history"]["values"], "history")
    _dense(label["values"], "future")
    # Existing zero-arm shared rows predate the final converter and retain
    # their native sidecar shape. They are trusted parent snapshots here;
    # synthetic rows below carry the stricter public prompt contract.
    if case.get("source") == "GEF_original_only" and not sidecar.get("preserve_original"):
        return
    payload = sidecar["original_payload"]
    _require(payload["label"] == label["values"], "saved_case_label_mismatch")
    _require([record[0] for record in payload["observed_values"]] == history, "observed_history_mismatch")
    _require(payload["prompt"] == model["public_prompt"] == payload["model_input_text"], "public_prompt_views_disagree")
    columns, rows = _table(model["public_prompt"], "Observed window:")
    _require(all(abs(float(record[columns.index("target")]) - float(value)) <= 5.01e-7
                 for record, value in zip(rows, history)), "prompt_history_mismatch")
    future_columns, _ = _table(model["public_prompt"], "Forecast window:")
    _require("target" not in future_columns, "public_future_target_column")
    def walk(value: Any) -> None:
        if isinstance(value, Mapping):
            _require(not PRIVATE_FIELDS.intersection(value), "private_field_in_model_input")
            for child in value.values():
                walk(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                walk(child)
    walk(model)
    if sidecar["preserve_original"]:
        _require(model["public_prompt"] == sidecar["saved_parent_prompt"]
                 and label == sidecar["saved_parent_label"] and history == sidecar["parent_history"],
                 "saved_zero_prompt_or_physical_values_changed")
    else:
        _validate_plan(model["gef_synthetic_public"], model["cutoff_time"])


def render_public(case: Mapping[str, Any], *, compressed: bool = False) -> str:
    """Return the same compressed-temperature public input for every consumer."""
    validate_public_case(case)
    model = case["model_input"]
    if "public_prompt" in model:
        return model["public_prompt"]
    payload = case["sidecar"]["original_payload"]
    return payload.get("prompt") or (payload.get("metadata") or {}).get("prompt_override")


def shared_consumer_envelope(case: Mapping[str, Any]) -> dict[str, Any]:
    """Use the existing research adapter for both native and Agent observations."""
    validate_public_case(case)
    return {"sample_id": case["case_id"], "source": case["source"],
            "research_mode": case["model_input"].get("research_mode"),
            "model_input": deepcopy(case["model_input"]),
            "supervised_label": deepcopy(case["supervised_label"]),
            "evaluator_only": deepcopy(case["sidecar"].get("evaluator_only", {}))}


def actor_public_payload(case: Mapping[str, Any]) -> dict[str, Any]:
    """Serialize the actor view without numeric labels or private arrays.

    Reasoning and Agent consumers may use the same payload. The structured
    target history is intentionally absent because ``public_prompt`` already
    contains the compressed, checked target table and future table.
    """
    validate_public_case(case)
    model = case["model_input"]
    public = {
        "sample_id": case["case_id"], "prompt": render_public(case),
        "history_points": 96, "future_points": 96,
        "future_timestamps": deepcopy(model["future_timestamps"]),
        "research_mode": model.get("research_mode"),
    }
    if "gef_synthetic_public" in model:
        public["gef_synthetic_public"] = deepcopy(model["gef_synthetic_public"])
    return public


def shared_actor_observation(case: Mapping[str, Any]) -> Any:
    """Bind the checked compressed public case to the existing Agent contract."""
    from castfactory.agentic.contracts import ForecastObservation, ProvenanceRecord

    validate_public_case(case)
    model = case["model_input"]
    prompt = render_public(case)
    columns, rows = _table(prompt, "Forecast window:")
    known = {name: [float(record[columns.index(name)]) for record in rows]
             for name in ("temp_mean", "temp_min", "temp_max", "hdd65", "cdd65")}
    context = {"public_prompt": prompt, "research_mode": model.get("research_mode"),
               "clock_basis": "source_native_naive_no_utc_claim"}
    if model.get("gef_synthetic_public"):
        context["gef_synthetic_public"] = deepcopy(model["gef_synthetic_public"])
    issue = model["cutoff_time"]
    records = [ProvenanceRecord("history:target_history", "GEFCom2014", PUBLIC_VERSION, issue),
               ProvenanceRecord("context:public_prompt", "synthetic_announced_protocol", PUBLIC_VERSION, issue)]
    records.extend(ProvenanceRecord("known_future:" + name, "offline_oracle_temperature_summary", PUBLIC_VERSION, issue)
                   for name in known)
    return ForecastObservation(
        sample_id=case["case_id"], target_history=model["target_history"]["values"],
        history_timestamps=model["target_history"]["timestamps"],
        future_timestamps=model["future_timestamps"], known_future=known, context=context,
        unit=model["target_history"].get("unit", ""), issue_time=issue, provenance=records,
    )
