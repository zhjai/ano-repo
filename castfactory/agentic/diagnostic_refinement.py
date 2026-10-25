"""Typed, source-bound T3 directives and deterministic segment-local feedback.

This module transports review hypotheses, never oracle errors or numeric edits.
It deliberately has no access to evaluator targets or sibling forecasts.
"""
from __future__ import annotations

from copy import deepcopy
import json


PROTOCOL = "diagnostic_refinement_v1"
PROFILE_VERSION = "full-agent-v3-diagnostic-refinement-v1"
CONCERNS = ("event_coverage", "event_interaction", "mechanism_consistency",
            "timing", "shape", "uncertainty")
REPORT_PATHS = {"summary": ("summary",), "self_check": ("payload", "self_check"),
                "uncertainty": ("payload", "uncertainty")}
INSTRUCTIONS = {
    "event_coverage": "Check whether the cited mechanism is adequately considered; omission from a short report is not proof of omission from the forecast.",
    "event_interaction": "Compare the two cited mechanisms, including possible offset, overlap, lag or rebound. Do not invent an established causal interaction.",
    "mechanism_consistency": "Compare the cited public evidence and mechanism hypotheses with your selected preceding report.",
    "timing": "Review event alignment, peak timing and possible lag using public evidence.",
    "shape": "Review level, peak and ramp plausibility using public evidence, without assuming a prescribed direction.",
    "uncertainty": "Reassess the specific uncertainty and whether the public evidence justifies changing the prediction.",
}


def response_schema():
    return {"action": "STOP|REFINE", "segment_id": "eligible segment for REFINE; null for STOP",
            "directive": {"concern": "one of " + "|".join(CONCERNS),
                          "review_span": "null or [start,end), integer indices inside selected segment",
                          "driver_refs": "0..2 distinct existing local driver IDs; coverage >=1; interaction exactly 2",
                          "own_report_field": "summary|self_check|uncertainty|null; mechanism_consistency requires summary or self_check"}}


def _keys(value, expected, name):
    if not isinstance(value, dict) or set(value) != set(expected):
        raise ValueError(name + " keys must be exactly " + ",".join(expected))


def _report(draft, field):
    value = draft
    for key in REPORT_PATHS[field]:
        if not isinstance(value, dict) or key not in value:
            raise ValueError("Selected own report field is absent")
        value = value[key]
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Selected own report field must be nonempty text")
    return value


def validate_directive(directive, segment, workspace, draft):
    _keys(directive, ("concern", "review_span", "driver_refs", "own_report_field"), "directive")
    concern, span = directive["concern"], directive["review_span"]
    refs, field = directive["driver_refs"], directive["own_report_field"]
    if not isinstance(concern, str) or concern not in CONCERNS:
        raise ValueError("Unknown diagnostic concern")
    if span is not None and (not isinstance(span, list) or len(span) != 2
                            or any(type(v) is not int for v in span)
                            or not segment["start"] <= span[0] < span[1] <= segment["stop"]):
        raise ValueError("review_span must be a half-open integer interval inside the selected segment")
    if (not isinstance(refs, list) or len(refs) > 2
            or any(not isinstance(r, str) or not r for r in refs) or len(set(refs)) != len(refs)):
        raise ValueError("driver_refs must contain at most two distinct local IDs")
    if (concern == "event_coverage" and not refs) or (concern == "event_interaction" and len(refs) != 2):
        raise ValueError("Concern requires registered driver references")
    drivers = {d["id"]: d for d in workspace["drivers"]}
    for ref in refs:
        if ref not in drivers:
            raise ValueError("Driver reference is not visible to the selected segment")
        if span is not None and not any(w["start"] < span[1] and w["end"] > span[0]
                                        for w in drivers[ref]["impact_windows"]):
            raise ValueError("Driver impact window does not overlap review_span")
    if field is not None and (not isinstance(field, str) or field not in REPORT_PATHS):
        raise ValueError("Unknown own_report_field")
    if concern == "mechanism_consistency" and field not in ("summary", "self_check"):
        raise ValueError("mechanism_consistency requires own summary or self_check")
    if field is not None:
        _report(draft, field)


def validate_action(payload, state, local_workspace):
    _keys(payload, ("action", "segment_id", "directive"), "T3 payload")
    if payload["action"] == "STOP":
        if payload["segment_id"] is not None or payload["directive"] is not None:
            raise ValueError("STOP requires null segment_id and directive")
        return
    sid = payload["segment_id"]
    if payload["action"] != "REFINE" or not isinstance(sid, str) or sid not in state["eligible_segments"]:
        raise ValueError("REFINE must select an eligible segment")
    segment = next((s for s in state["plan"]["spans"] if s["segment_id"] == sid), None)
    if segment is None or sid not in state["reports"]:
        raise ValueError("Selected segment or current report is absent")
    validate_directive(payload["directive"], segment, local_workspace(state["workspace"], segment),
                       state["reports"][sid])


def bind_feedback(payload, *, segment, workspace, draft, public, sample_id, depth, protocol_version):
    """Capture an owned local snapshot. No caller may supply global reports."""
    if protocol_version != PROFILE_VERSION or type(depth) is not int or depth < 0:
        raise ValueError("Diagnostic feedback protocol/depth mismatch")
    if payload.get("action") != "REFINE" or payload.get("segment_id") != segment["segment_id"]:
        raise ValueError("Diagnostic feedback segment mismatch")
    directive = payload["directive"]
    validate_directive(directive, segment, workspace, draft)
    if any(draft["payload"][key] != segment[key] for key in ("segment_id", "start", "stop")):
        raise ValueError("Diagnostic feedback draft belongs to another segment")
    manifest = {e["evidence_id"]: e for e in public["evidence_manifest"]}
    by_id = {d["id"]: d for d in workspace["drivers"]}
    drivers, evidence = [], {}
    for ref in sorted(directive["driver_refs"]):
        driver = by_id[ref]
        drivers.append(deepcopy(driver))
        for item in driver["evidence"]:
            source = manifest.get(item["evidence_id"])
            if source is None or any(source.get(k) != item.get(k) for k in ("kind", "as_of")):
                raise ValueError("Diagnostic evidence reference does not match public manifest")
            evidence[item["evidence_id"]] = deepcopy(source)
    field = directive["own_report_field"]
    content = {"concern": directive["concern"], "review_span": directive["review_span"],
               "drivers": drivers, "source_evidence": [evidence[k] for k in sorted(evidence)],
               "own_report": None if field is None else {"field": field, "text": _report(draft, field)},
               "provenance": {k: deepcopy(public[k]) for k in ("covariate_vintage", "context_origin") if k in public}}
    instruction = (INSTRUCTIONS[directive["concern"]] +
                   " This is a review hypothesis, not a verified error. Keep the prediction unchanged if appropriate. "
                   "review_span is an attention window, not an edit mask: return the complete assigned segment; "
                   "lag/rebound changes may extend within that segment, never into another segment. "
                   "T0 mechanisms and interactions are hypotheses; retain source availability qualifications.\n" +
                   json.dumps(content, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    return {"protocol": PROTOCOL, "sample_id": sample_id, "segment_id": segment["segment_id"],
            "depth": depth, "profile_version": protocol_version,
            "directive": deepcopy(directive), "feedback": instruction,
            "current_draft": deepcopy(draft)}
