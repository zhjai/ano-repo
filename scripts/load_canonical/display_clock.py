"""Verified display calendars over unchanged physical source coordinates.

This registry selects a *display reference*, not a measurement timezone or an
event publication clock. Only aware instants are converted. Source timestamp
strings, numeric rows, positions and native event clocks are never mutated.
Evidence paths are repository-relative unless they refer to the raw workspace.
"""
from __future__ import annotations

import copy
from datetime import datetime, timedelta, timezone
import re
from typing import Any
from zoneinfo import ZoneInfo


CLOCK_POLICY_VERSION = "verified-reference-display-clock-v1"
_AUDIT_ROOT = "local-audit-not-distributed/"
_RAW_ROOT = "local-source-not-distributed/"

# Selection deliberately uses dataset + physical target, never event geography
# or a source display name. See the task clock_registry.md for evidence limits.
VERIFIED_CLOCKS = {
    "lcl_london": {
        "timezone": "Europe/London",
        "label": "Europe/London (GMT/BST; source readings GMT)",
        "basis": "verified_household_local_reference_calendar",
        "evidence": [
            _RAW_ROOT + "lcl/catalogue.html:98 (London households)",
            _AUDIT_ROOT + "canonical_response_sources/lcl_clock.draft.json:evidence_quote (source GMT without BST)",
            _AUDIT_ROOT + "canonical_response_sources/local_clock_protocol_evidence.json:pdf_quotes[227] (C3, page 71; GMT/BST seasonal clock)",
        ],
    },
    "pjm_ept": {
        "timezone": "America/New_York",
        "label": "PJM operator EPT (America/New_York; DST)",
        "basis": "verified_operator_reference_calendar",
        "evidence": [
            _RAW_ROOT + "pjm/audit_report.md:Load Findings (EPT follows DST)",
            _AUDIT_ROOT + "pjm_nws_fast_audit/REPORT.md:24-26 (all acquired EPT/UTC rows agree with America/New_York)",
            _RAW_ROOT + "pjm/audit_report.md:Alignment and Dataset Use (PJM and EIA-930 PJM same physical system)",
        ],
    },
    "spp_ct": {
        "timezone": "America/Chicago",
        "label": "SPP operator CT (America/Chicago; DST)",
        "basis": "verified_operator_reference_calendar",
        "evidence": [
            _RAW_ROOT + "spp/event_history_to_20261007/grid_conditions_observed_20261007T005909Z.html (East and West BAA notices explicitly CT)",
            "datasets/load_datasets/sources/spp/event_history_to_20261007/live_advisory_verification.json:publication_ct,publication_utc,clock_values_exact",
            "datasets/load_datasets/sources/spp/event_history_to_20261007/REPORT.md:New Live Advisory Incorporated (America/Chicago CT offset)",
        ],
    },
    "aemo_nem": {
        "timezone": "UTC+10:00",
        "label": "AEMO NEM market AEST (UTC+10; no DST)",
        "basis": "verified_operator_standard_reference_calendar",
        "evidence": [
            "datasets/load_datasets/canonical_candidates_v1/event_focused_aemo_20261005/public_evidence_delta.json:facts[0] (official NEM time AEST UTC+10, not local times)",
            "datasets/load_datasets/canonical_candidates_v1/event_focused_aemo_20261005/SOURCE_HANDOFF.md:38-40",
        ],
    },
}
FALLBACK_EVIDENCE = {
    "iflex_hourly": [
        _RAW_ROOT + "iflex/primary_clock_evidence/article.html (Norway and city regions)",
        _AUDIT_ROOT + "canonical_response_sources/iflex_clock.draft.json:evidence_limit (Norwegian Date/Hour mapping unresolved)",
    ],
    "entsoe": [
        "datasets/load_datasets/json_rows_v1/*.jsonl:case.task,case.context.source_context.research_context (national physical UTC targets; contextual Nord areas)",
    ],
    "gefcom2014": [
        "datasets/load_datasets/json_rows_v1/*.jsonl:GEF case.grid (supplied naive labels)",
    ],
    "ercot": [
        _RAW_ROOT + "ercot/extended_20261003/load_product.html:706:lastUpdatedDate_dt_tz (page metadata only)",
        "datasets/load_datasets/sources/ercot/manifest.extended-20261003.json:field_roles.Hour Ending (local clock without named timezone)",
    ],
}


def _public_case(case: dict) -> dict:
    """Accept either a public case or its selected-inventory envelope."""
    return case["case"] if isinstance(case.get("case"), dict) else case


def _parse_timestamp(value: str) -> datetime:
    if not isinstance(value, str):
        raise ValueError("Display timestamp must be an ISO timestamp string")
    # Python 3.9's fromisoformat does not accept terminal Z.
    try:
        return datetime.fromisoformat(value[:-1] + "+00:00" if value.endswith("Z") else value)
    except ValueError as exc:
        raise ValueError("Invalid source display timestamp: " + repr(value)) from exc


def _fallback(reason: str, evidence: list[str] | None = None) -> dict:
    return {
        "version": CLOCK_POLICY_VERSION,
        "status": "source_clock_fallback",
        "timezone": None,
        "label": "source clock retained; local/operator reference unverified",
        "basis": "source_timestamp_literal",
        "reason": reason,
        "evidence": list(evidence or []),
        "physical_instants_changed": False,
        "native_event_clocks_changed": False,
    }


def display_clock_for_case(case: dict) -> dict[str, Any]:
    """Return a verified reference-calendar policy or explicit source fallback.

    An unzoned timestamp anywhere in either target grid keeps the whole case in
    its source clock: localizing naive labels would manufacture physical time.
    Unknown dataset/target combinations cannot inherit another entity's clock.
    """
    public = _public_case(case)
    task = public.get("task") or {}
    dataset = str(task.get("dataset_id", "")).lower()
    entity = str(task.get("series_id", ""))
    key = None
    if dataset == "lcl_hourly" and entity.startswith("MAC"):
        key = "lcl_london"
    elif (dataset == "pjm" and entity == "RTO") or (dataset == "eia930" and entity == "PJM"):
        key = "pjm_ept"
    elif dataset == "eia930" and entity in {"SWPP", "SWPW"}:
        key = "spp_ct"
    elif dataset == "aemo" and entity in {"NSW1", "QLD1", "SA1", "TAS1", "VIC1"}:
        key = "aemo_nem"
    if key is None:
        reason = {
            "iflex_hourly": "UTC-Z measurements and Norwegian location are verified, but an explicit local reference-clock convention was not recovered; do not infer it from municipality.",
            "entsoe": "UTC numeric national targets do not establish a single local/operator reference clock for the selected Nordic/German targets; notice geography is not clock evidence.",
            "gefcom2014": "GEF supplied naive timestamp labels have no verified timezone; preserve source calendar.",
            "ercot": "CST6CDT source-page lastUpdated metadata alone does not establish the target/operator reference clock; leave source timestamps unchanged.",
        }.get(dataset, "No verified reference clock for this dataset/target combination.")
        return _fallback(reason, FALLBACK_EVIDENCE.get(dataset))
    policy = copy.deepcopy(VERIFIED_CLOCKS[key])
    grid = public.get("grid") or {}
    for scope in ("history_timestamps", "future_timestamps"):
        for index, value in enumerate(grid.get(scope) or []):
            if isinstance(value, str) and re.search(r"\.\d{7,}", value):
                fallback = _fallback(
                    f"Submicrosecond source precision at {scope}[{index}]; retain literal timestamps to avoid precision loss.",
                    policy["evidence"],
                )
                fallback["label"] = "source clock retained; timestamp precision preserved"
                return fallback
            parsed = _parse_timestamp(value)
            if parsed.tzinfo is None or parsed.utcoffset() is None:
                fallback = _fallback(
                    f"Naive source timestamp at {scope}[{index}]; verified reference cannot establish its physical instant.",
                    policy["evidence"],
                )
                fallback["label"] = "source clock retained; naive timestamps not localized"
                return fallback
    policy.update({
        "version": CLOCK_POLICY_VERSION,
        "status": "verified",
        "registry_key": key,
        "reason": "Reference display calendar established by the cited local evidence; source coordinates remain unchanged.",
        "physical_instants_changed": False,
        "native_event_clocks_changed": False,
    })
    return policy


def display_timestamps(case: dict, scope: str) -> dict[str, Any]:
    """Return display ISO timestamps plus policy for history or future_known.

    Numeric UTC offsets remain explicit on every converted row, distinguishing
    repeated DST hours. Fallback returns source strings verbatim, including Z,
    existing offsets, fractional seconds and naive labels. No sorting occurs.
    """
    if scope not in {"history", "future_known"}:
        raise ValueError("Display clock scope must be history or future_known")
    public = _public_case(case)
    grid_key = "history_timestamps" if scope == "history" else "future_timestamps"
    timestamps = list((public.get("grid") or {}).get(grid_key) or [])
    policy = display_clock_for_case(public)
    if policy["status"] != "verified":
        return {"timestamps": timestamps, "policy": policy, "converted": False}
    zone = (timezone(timedelta(hours=10)) if policy["timezone"] == "UTC+10:00"
            else ZoneInfo(policy["timezone"]))
    displayed = [_parse_timestamp(value).astimezone(zone).isoformat() for value in timestamps]
    return {"timestamps": displayed, "policy": policy, "converted": True}
