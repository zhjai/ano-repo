"""Replay the exact user-authorized four-case prose display; no content creation.

Semantic requests remain original. Only root-bound explanation strings change
in the model view, with original typed state and forecasts retained verbatim.
"""
from copy import deepcopy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / "assets/private_role_state"
SOURCE = ROOT / "ROLE_STATE_PROSE.json"
REQUESTS = ROOT / "ROLE_STATE_PROSE_REQUESTS.json"
ALLOWED = {"eia930::SWPP::h96_f96::20180403T0000Z": "T4",
           "eia930::SWPP::h96_f96::20190719T0000Z": "T4",
           "eia930::SWPP::h168_f24::20181214T0000Z": "T3",
           "eia930::SWPP::h168_f24::20210101T0000Z": "T3"}


def bound_role_state_view(request, case_id, role=None):
    """Read genuine root prose and exact original request, never approve new prose."""
    role = role or request.get("role")
    allowed_roles = ("T3", "T4") if ALLOWED.get(case_id) == "T3" else ("T4",)
    if case_id not in ALLOWED or role not in allowed_roles:
        return request, None
    if not SOURCE.exists():
        return request, None
    def require(condition, message):
        if not condition:
            raise ValueError("Root role-state display: " + message)
    require(SOURCE.is_file() and REQUESTS.is_file() and not SOURCE.is_symlink() and not REQUESTS.is_symlink(),
            "canonical regular prose binding files required")
    source_bytes, request_bytes = SOURCE.read_bytes(), REQUESTS.read_bytes()
    source, packets = json.loads(source_bytes), json.loads(request_bytes)
    entry = source.get(case_id)
    if entry is None:
        return request, None
    exact = packets[case_id]
    if exact["request"] != request or exact["role"] != role:
        if role != "T4" or exact["role"] != "T3" or any(
                request.get(key) != exact["request"].get(key) for key in ("case", "geometry", "accepted_t1")):
            return request, None
    original = request["accepted_t1"]["explanation"]
    explanation = entry.get("accepted_t1_explanation_by_role", {}).get(role, entry["accepted_t1_explanation"])
    require(entry.get("role") == exact["role"] and entry.get("original_explanation") == original
            and isinstance(entry.get("accepted_t1_explanation"), str) and entry["accepted_t1_explanation"].strip()
            and entry.get("source") == "user_authorized_root_faithful_role_state_prose",
            "original role/explanation/source binding differs")
    view = deepcopy(request)
    view["accepted_t1"]["explanation"] = explanation
    reports_key = "current_segment_reports" if role == "T3" else "final_segment_reports"
    original_reports = {r["answer"]["segment_id"]: r for r in exact["request"].get("current_segment_reports", [])}
    changes = []
    for report in view.get(reports_key, []):
        sid = report.get("answer", {}).get("segment_id")
        adapted = entry.get("report_explanations", {}).get(sid)
        if adapted is None:
            continue
        if report != original_reports.get(sid):
            continue  # A genuinely revised report keeps its own prose, never the old bound view.
        require(adapted.get("original_explanation") == report["explanation"]
                and isinstance(adapted.get("explanation"), str) and adapted["explanation"].strip(),
                "original report explanation differs")
        changes.append({"segment_id": sid, "original_explanation": report["explanation"],
                        "rendered_explanation": adapted["explanation"]})
        report["explanation"] = adapted["explanation"]
    restored = deepcopy(view)
    restored["accepted_t1"]["explanation"] = original
    for report in restored.get(reports_key, []):
        for changed in changes:
            if report["answer"]["segment_id"] == changed["segment_id"]:
                report["explanation"] = changed["original_explanation"]
    require(restored == request, "a non-prose field changed")
    require(SOURCE.read_bytes() == source_bytes and REQUESTS.read_bytes() == request_bytes,
            "source changed during validation")
    return view, {"source": str(SOURCE), "case_id": case_id, "role": role,
                  "request_source": str(REQUESTS), "original_request": deepcopy(request),
                  "binding_request": exact["request"], "original_explanation": original,
                  "rendered_explanation": explanation, "report_explanation_views": changes,
                  "semantic_request_unchanged": True, "typed_and_numeric_state_unchanged": True}


def validated_row_state_view(row):
    """Require exact honest provenance before replaying a non-default display."""
    request = row["semantic_request"]
    marker = row.get("serialization_provenance", {}).get("role_state_prose_view")
    if marker is None:
        return request
    view, actual = bound_role_state_view(request, row.get("case_id"), request["role"])
    if actual is None or marker != actual:
        raise ValueError("Root role-state display provenance differs from actual bound prose")
    return view
