"""Forecast inputs without private archive logistics or admission labels.

The canonical tree is never modified. Explicit field classifications, rather
than event-prose matching, determine exclusions. A private sparse edit sidecar
restores the original tree, including the exact bytes of embedded JSON strings.
Readable factoring and opaque identities use the existing public event codec.
"""
from __future__ import annotations

import copy
import json
import re
from datetime import datetime
from dataclasses import replace
from typing import Any, Callable, Mapping

from . import gef_final_public, public_event_view_v2, public_event_view_v3

VERSION = "forecast-input-view-v1"
NEW_VERSION = "forecast-input-view-v2"

# Opt-in only: the running v1 collector and archived v1 sidecars retain their
# original contract. These are classified transport/provenance fields, never
# matches inside native notice prose.
_RESIDUAL_ARCHIVE_FIELDS = frozenset({
    "source_row", "source_member", "native_source_url", "source_url",
    "catalog_view", "native_catalog_url", "native_url",
    "source_view_id", "country_containment_source",
})
_RESIDUAL_ADMIN_FIELDS = frozenset({
    "optimizer_steps", "strict_ood_reservation", "source_field_preservation",
})
_RESIDUAL_LINEAGE_FIELDS = frozenset({
    "exogenous_source_missing_counts", "historical_covariate_missingness_policy",
})
_ROUTING_FIELDS = frozenset({
    "BLOCKCHANNEL", "WMOidentifier", "AWIPSidentifier", "PIL", "EAS-ORG",
})
_REFERENCE_FIELDS = frozenset({"native_references", "expiredReferences"})
_EVENT_ID_FIELDS = frozenset({"event_id", "event_record_uid", "native_event_id", "record_id"})
_REFERENCE_ID_FIELDS = frozenset({"referenced_source_message_ids"})
_NATIVE_MESSAGE = re.compile(r"(?:urn:oid:2\.49\.0\.1\.840\.0\.[A-Za-z0-9.-]+|NWS-IDP-PROD-[A-Za-z0-9-]+)")
_PJM_NAVIGATION = re.compile(
    r'<br/><br/><a href="https://emergencyprocedures\.pjm\.com/ep/pages/viewposting\.jsf\?id=\d+">View Posting</a>$')
_INHERITED_EIA_AGGREGATION = (
    "hourly_mean_of_complete_native_values; EIA fields retain MWh-per-original-one-hour-interval, not relabelled MW")
_LCL_ARCHIVE_NOTICE = (
    "This complete daily schedule uses the 48 saved Normal/High/Low table entries referenced by the selected-day sidecar.")
_LCL_PUBLIC_NOTICE = (
    "This complete daily schedule uses the 48 saved Normal/High/Low table entries.")
_ADMISSION_COVERAGE_FIELDS = frozenset({"event_coverage_hours", "event_coverage_mask", "coverage_zero_semantics"})


def strip_archive_navigation(text: str) -> str:
    """Remove only the verified PJM terminal View Posting navigation wrapper."""
    return _PJM_NAVIGATION.sub("", text)


def _same(left: Any, right: Any) -> bool:
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(_same(left[key], right[key]) for key in left)
    if isinstance(left, list):
        return len(left) == len(right) and all(_same(a, b) for a, b in zip(left, right))
    return left == right


def recursive_subset(left: Any, right: Any) -> bool:
    """Verify a richer carrier preserves every value and ordered list member.

    Dictionary keys may be added by the richer carrier. List lengths and order,
    scalar types, values, and any existing relation qualifiers must agree.
    """
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() <= right.keys() and all(recursive_subset(child, right[key]) for key, child in left.items())
    if isinstance(left, list):
        return len(left) == len(right) and all(recursive_subset(a, b) for a, b in zip(left, right))
    return left == right


def _derived_cap_uid(value: Mapping[str, Any], uid: Any) -> bool:
    identity, publication, ordinal = (value.get("event_id"), value.get("publication_time"),
                                      value.get("info_ordinal_zero_based"))
    return (isinstance(identity, str) and isinstance(publication, str)
            and type(ordinal) is int and ordinal >= 0
            and uid == f"{identity}@{publication}:info{ordinal}")


def _duplicate_expiry(value: Mapping[str, Any], epoch: Any) -> bool:
    clock = value.get("info", {}).get("expires_time_utc") if isinstance(value.get("info"), Mapping) else None
    if not isinstance(clock, str) or type(epoch) not in {int, float}:
        return False
    try:
        parsed = datetime.fromisoformat(clock.replace("Z", "+00:00"))
        return parsed.tzinfo is not None and parsed.timestamp() == epoch
    except ValueError:
        return False


def _admission_zero_coverage(value: Mapping[str, Any]) -> bool:
    mask = value.get("event_coverage_mask")
    hours = value.get("event_coverage_hours")
    return (value.get("coverage_zero_semantics") == "no_eligible_announced_event_not_evidence_of_no_event"
            and type(hours) in {int, float} and hours == 0
            and isinstance(mask, list) and bool(mask)
            and all(type(entry) in {int, float} and entry == 0 for entry in mask))


def _reference_triples(value: Any) -> list[dict[str, str]] | None:
    """Parse CAP sender,identifier,sent triples without guessing relation text.

    Invalid/nonstandard references remain literal; dropping an unparsed
    reference could erase an update or cancellation link.
    """
    if not isinstance(value, str) or not value.strip():
        return None
    result = []
    for token in value.split():
        parts = token.split(",")
        if len(parts) != 3 or not all(parts):
            return None
        sender, identity, sent = parts
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}T\S+", sent):
            return None
        result.append({"event_id": identity, "sent": sent, "sender": sender})
    return result

# Container names such as quality and lineage are deliberately absent: they
# also carry units, missingness, clock uncertainty and geographic evidence.
ADMIN_FIELDS = frozenset({
    "schema_version", "grid_version", "window_version", "mapping_version",
    "sample_id", "native_payload_sample_id", "context_condition", "split",
    "formal_splits", "formal_split_assigned", "formal_splits_assigned",
    "training_admission", "production_approval", "operational_admission",
    "event_admission", "training_started", "teacher_or_model_calls",
    "model_calls", "proposal_only", "new_hashes", "no_new_hashes",
    "input_capacity_verified", "actor_budget_approved",
    "candidate_eligibility_holds", "consumer_diagnostics", "exposure_status",
    "measurement_counts", "ood_axis", "tier", "selection_policy",
    "irrelevant_distractor_present", "numeric_control_only",
    "retrospective_event_only", "evaluator_only", "supervised_label",
    "relevance_label", "negative_type", "sample_category", "input_classification",
    "event_evidence_status", "research_data_usable",
    "complete_origin_known_text_preserved", "research_event_input_present",
    "event_input_eligible", "research_input_eligible", "research_context_input_eligible",
    "research_training_eligible", "served_load_event_input_eligible", "numeric_eligible",
    "strict_public_availability_approved",
})
ARCHIVE_FIELDS = frozenset({
    "native_source_file", "native_source_locator", "normalized_source_file",
    "normalized_source_byte_offset", "normalized_source_byte_length",
    "normalized_source_physical_line", "full_text_locator", "source_locator",
    "source_record_locator", "record_locator", "catalog_artifact", "catalog_url",
    "catalog_row", "native_row_html", "artifact", "byte_offset", "byte_length",
    "physical_line", "selector", "raw_text_and_report_outcomes_location",
    "input_classification_evidence", "evidence_span", "catalog_id",
})
_ADMIN_COUNTERS = frozenset({
    "event_count", "episode_count", "research_context_event_count",
    "strict_operational_event_count", "research_event_count",
})
_ARCHIVE_REFERENCES = {
    "full_original_text": {"retained_in_evaluator_event_catalog_and_exact_source_locators"},
}
_NATIVE_PROSE_FIELDS = frozenset({
    "text", "body", "full_native_text", "native_text", "description",
    "headline", "title", "instruction", "instructions", "sentence_excerpt",
})


def _native_prose_path(path: list[Any]) -> bool:
    return any(isinstance(step, str) and (step in _NATIVE_PROSE_FIELDS or
               (step != "context_text" and step.endswith(("_text", "_body", "_excerpt"))))
               for step in path)


def _reason(key: str, value: Any, path: list[Any]) -> str | None:
    if key in ADMIN_FIELDS:
        return "internal_administration_or_selection"
    if key in ARCHIVE_FIELDS or key.endswith(("_path", "_paths", "_sha256")):
        return "archive_logistics"
    if key in _ADMIN_COUNTERS and any(p in {"metadata", "quality", "research_context"} for p in path if isinstance(p, str)):
        return "administrative_counter_without_input_event_scope"
    if isinstance(value, str) and value in _ARCHIVE_REFERENCES.get(key, set()):
        return "archive_logistics"
    # Remove complete local locators only. A body mentioning a local path is
    # substantive prose and is never sliced or discarded by a regex.
    if not _native_prose_path(path + [key]) and isinstance(value, str) and value.startswith(("/data2/", "/data/", "/root/")) and not any(c.isspace() for c in value):
        return "local_archive_locator"
    return None


def _embedded_objects(text: str) -> list[tuple[int, int, Any]]:
    """Find complete JSON objects/arrays while retaining surrounding prose."""
    decoder = json.JSONDecoder()
    result = []
    offset = 0
    while offset < len(text):
        starts = [p for p in (text.find("{", offset), text.find("[", offset)) if p >= 0]
        if not starts:
            break
        start = min(starts)
        try:
            value, length = decoder.raw_decode(text[start:])
        except ValueError:
            offset = start + 1
            continue
        end = start + length
        result.append((start, end, value))
        offset = end
    return result


def project_model_input(model: Any, *, policy: str = VERSION) -> tuple[Any, dict[str, Any]]:
    """Return a forecast tree and private edits sufficient for exact recovery.

No case split or label is consulted. The default v1 policy is unchanged. The
opt-in v2 policy also removes residual provenance and routing fields, aliases
event identities, and preserves compact CAP relationships and sent clocks.
"""
    if policy not in {VERSION, NEW_VERSION}:
        raise ValueError("unknown forecast input projection policy")
    residual = policy == NEW_VERSION
    edits: list[dict[str, Any]] = []
    aliases: dict[str, str] = {}
    known_ids: set[str] = set()
    reserved: set[str] = set()
    reference_senders: dict[str, set[str]] = {}

    def collect(value: Any) -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                if key in _EVENT_ID_FIELDS and isinstance(child, str):
                    known_ids.add(child)
                if key == "version" and isinstance(child, str) and _NATIVE_MESSAGE.fullmatch(child):
                    known_ids.add(child)
                if key in _REFERENCE_FIELDS or (key == "value" and value.get("name") in _REFERENCE_FIELDS):
                    for reference in _reference_triples(child) or []:
                        reference_senders.setdefault(reference["event_id"], set()).add(reference["sender"])
                collect(child)
        elif isinstance(value, list):
            for child in value:
                collect(child)
        elif isinstance(value, str):
            reserved.update(re.findall(r"(?<![A-Za-z0-9_])[EX][0-9]+(?![A-Za-z0-9_])", value))
            for _, _, embedded in _embedded_objects(value):
                collect(embedded)

    if residual:
        collect(model)

    def alias(identity: str) -> str:
        if identity not in aliases:
            prefix = "E" if identity in known_ids else "X"
            ordinal = sum(a.startswith(prefix) for a in aliases.values()) + 1
            candidate = prefix + str(ordinal)
            while candidate in reserved or candidate in aliases.values():
                ordinal += 1
                candidate = prefix + str(ordinal)
            aliases[identity] = candidate
        return aliases[identity]

    def replace(value: Any, path: list[Any], journal: list[dict[str, Any]], reason: str) -> None:
        journal.append({"path": path[:], "operation": "restore_value",
                        "value": copy.deepcopy(value), "reason": reason})

    def walk(value: Any, path: list[Any], journal: list[dict[str, Any]], cap_context: bool = False) -> Any:
        if isinstance(value, Mapping):
            cap_context = cap_context or "native_references" in value or (
                isinstance(value.get("event_id"), str) and bool(_NATIVE_MESSAGE.match(value["event_id"])))
            zero_coverage = residual and path == ["source_context", "metadata"] and _admission_zero_coverage(value)
            result = {}
            for key, child in value.items():
                reason = _reason(key, child, path)
                if residual:
                    if (key in _RESIDUAL_ARCHIVE_FIELDS or
                            (key == "row" and path and path[-1] == "provenance") or
                            (key == "url" and "catalog_views" in path) or
                            (key == "version" and child == "ercot_official_archive_v1")):
                        reason = "archive_logistics"
                    elif key in _RESIDUAL_ADMIN_FIELDS or (key in {"grid_anchor", "stride_hours"} and "metadata" in path):
                        reason = "internal_administration_or_selection"
                    elif key in _RESIDUAL_LINEAGE_FIELDS and path and path[-1] == "lineage":
                        reason = "archive_statistics_or_worker_instruction"
                    elif zero_coverage and key in _ADMISSION_COVERAGE_FIELDS:
                        reason = "admission_derived_zero_coverage"
                    elif key in _ROUTING_FIELDS and cap_context:
                        reason = "message_transport_routing"
                    elif (key == "status" and child == "source_product_field_semantics_pending_independent_unit_audit"
                          and "exogenous_unit_evidence" in path):
                        reason = "internal_unit_audit_workflow"
                    elif key == "context_source_spans" and isinstance(child, list) and all(
                            isinstance(span, Mapping) and isinstance(span.get("text"), str) for span in child):
                        text = "".join(span["text"] for span in child)
                        if text == value.get("event_text"):
                            reason = "duplicate_extraction_carrier"
                    elif key == "version" and _same(child, value.get("event_record_uid")):
                        reason = "duplicate_event_identity"
                    elif (key == "event_record_uid" and cap_context and
                          path and path[0] == "temporal_text_events" and _derived_cap_uid(value, child)):
                        reason = "duplicate_event_identity"
                    elif (key == "event_record_uid" and isinstance(model, Mapping) and model.get("dataset_id") == "aemo"
                          and path[:3] == ["source_context", "research_context", "event_records"]
                          and isinstance(child, str) and re.fullmatch(r"aemo-history:s\d+:l\d+", child)
                          and isinstance(value.get("native_notice_id"), str) and value["native_notice_id"]):
                        reason = "archive_event_identity"
                    elif key in {"expiry_epoch", "CAP_information_expires_utc_epoch"} and _duplicate_expiry(value, child):
                        reason = "duplicate_expiry_clock"
                    elif (key == "research_context" and path == ["context_text"] and isinstance(model, Mapping)
                          and model.get("dataset_id") == "entsoe" and isinstance(model.get("source_context"), Mapping)
                          and isinstance(child, Mapping) and "research_context" in model["source_context"]
                          and recursive_subset(child, model["source_context"]["research_context"])):
                        reason = "duplicate_research_context_carrier"
                    elif path == ["source_context"] and isinstance(model, Mapping):
                        if key == "forecast_timestamps" and _same(child, model.get("future_timestamps")):
                            reason = "duplicate_forecast_axis"
                        elif key == "exogenous_history" and (
                                _same(child, model.get("exogenous_histories")) or
                                any(_same(child, item) for item in model.get("exogenous_histories", []))):
                            reason = "duplicate_exogenous_history"
                if reason:
                    journal.append({"path": path + [key], "operation": "restore_field",
                                    "value": copy.deepcopy(child), "reason": reason})
                    continue
                if (residual and key == "aggregation" and child == _INHERITED_EIA_AGGREGATION and
                        value.get("field_id") in {"WDR_ESTIMATE", "OPERATIONAL_DEMAND_ADJUSTMENT"} and
                        "exogenous_unit_evidence" in path):
                    replace(child, path + [key], journal, "inherited_irrelevant_EIA_unit_annotation")
                    result[key] = "hourly_mean_of_complete_native_values"
                    continue
                if residual and key == "source" and child == "NOAA NWS CAP via IEM MTArchive":
                    replace(child, path + [key], journal, "archive_host_attribution")
                    result[key] = "NOAA NWS CAP"
                    continue
                if (residual and key == "inferred_day_ahead_schedule_notice" and
                        isinstance(child, str) and child.endswith(_LCL_ARCHIVE_NOTICE)):
                    replace(child, path + [key], journal, "generated_schedule_archive_annotation")
                    result[key] = child[:-len(_LCL_ARCHIVE_NOTICE)] + _LCL_PUBLIC_NOTICE
                    continue
                reference_key = key
                if key == "value" and value.get("name") in _REFERENCE_FIELDS:
                    reference_key = value["name"]
                if residual and reference_key in _REFERENCE_FIELDS:
                    if child is None or child == "" or child == []:
                        journal.append({"path": path + [key], "operation": "restore_field",
                                        "value": copy.deepcopy(child), "reason": "empty_message_relationship"})
                        continue
                    compact = _reference_triples(child)
                    if compact is not None:
                        for relation in compact:
                            native_id = relation["event_id"]
                            relation["event_id"] = alias(native_id)
                            # CAP IDs are sender-scoped. Retain the namespace
                            # if it actually disambiguates colliding IDs.
                            if len(reference_senders[native_id]) == 1:
                                del relation["sender"]
                        replace(child, path + [key], journal, "compact_message_relationships")
                        result[key] = compact
                        continue
                before = len(journal)
                projected = walk(child, path + [key], journal, cap_context)
                if isinstance(child, Mapping) and child and not projected:
                    # An emptied archive carrier is logistics too. Replace its
                    # child edits with one complete restoration operation.
                    del journal[before:]
                    journal.append({"path": path + [key], "operation": "restore_field",
                                    "value": copy.deepcopy(child), "reason": "empty_logistics_container"})
                    continue
                result[key] = projected
            return result
        if isinstance(value, list):
            if residual and cap_context and path and path[-1] == "parameters":
                before = len(journal)
                retained = [child for child in value if not (
                    isinstance(child, Mapping) and child.get("name") in _ROUTING_FIELDS)]
                if len(retained) != len(value):
                    projected = [walk(child, path + [i], journal, cap_context) for i, child in enumerate(retained)]
                    # List positions changed: restore the entire original list
                    # rather than recording unstable original-index edits.
                    del journal[before:]
                    replace(value, path, journal, "message_transport_parameters")
                    return projected
            return [walk(child, path + [i], journal, cap_context) for i, child in enumerate(value)]
        if isinstance(value, str):
            if residual and path and path[-1] == "raw_text":
                text = strip_archive_navigation(value)
                if text != value:
                    replace(value, path, journal, "terminal_archive_navigation")
                    return text
            if _native_prose_path(path):
                return value
            key = next((step for step in reversed(path) if isinstance(step, str)), None)
            if residual and (key in _EVENT_ID_FIELDS or key in _REFERENCE_ID_FIELDS or
                             (key == "version" and _NATIVE_MESSAGE.match(value))):
                projected = alias(value)
                replace(value, path, journal, "local_event_identity")
                return projected
            pieces = []
            offset = 0
            nested_edits = []
            for start, end, embedded in _embedded_objects(value):
                local_edits: list[dict[str, Any]] = []
                projected = walk(embedded, path + ["<embedded_json>", start], local_edits, cap_context)
                if local_edits:
                    pieces.extend([value[offset:start], json.dumps(projected, ensure_ascii=False, separators=(",", ":"), allow_nan=False)])
                    offset = end
                    nested_edits.extend(local_edits)
            if nested_edits:
                pieces.append(value[offset:])
                journal.append({"path": path[:], "operation": "restore_value", "value": value,
                                "reason": "embedded_json_projection", "embedded_edits": nested_edits})
                return "".join(pieces)
        return copy.deepcopy(value)

    clean = walk(model, [], edits)
    sidecar = {"version": policy, "edits": edits}
    if residual:
        sidecar["identity_aliases"] = aliases
    return clean, sidecar


def restore_model_input(clean: Any, sidecar: Mapping[str, Any]) -> Any:
    """Restore the source tree, including exact original embedded text."""
    if sidecar.get("version") not in {VERSION, NEW_VERSION}:
        raise ValueError("unknown forecast input sidecar version")
    result = copy.deepcopy(clean)
    for edit in reversed(sidecar["edits"]):
        path = edit["path"]
        if not path:
            result = copy.deepcopy(edit["value"])
            continue
        parent = result
        for step in path[:-1]:
            parent = parent[step]
        parent[path[-1]] = copy.deepcopy(edit["value"])
    return result


def forecast_clean_text(text: str) -> str:
    """Correct only the legend; reuse an existing encoded/readable payload."""
    for old, new in ((public_event_view_v2.EVENT_VIEW_LEGEND, public_event_view_v2.FORECAST_CLEAN_LEGEND),
                     (public_event_view_v3.LEGEND, public_event_view_v3.FORECAST_CLEAN_LEGEND)):
        if text.startswith(new):
            return text
        if text.startswith(old):
            return new + text[len(old):]
    raise ValueError("unknown event view legend")


def forecast_clean_view(view: Any) -> Any:
    """Change opt-in model-visible legends without touching encoded values."""
    updates = {"text": forecast_clean_text(view.text)}
    if hasattr(view, "readable_text"):
        updates["readable_text"] = forecast_clean_text(view.readable_text)
    return replace(view, **updates)


def render_projected_forecast_view(projected: Any, *, source: str | None = None,
                                   count_units: Callable[[str], int] | None = None) -> Any:
    """Share the exact v2/v3 choice between exports and opt-in consumers.

    A deployment tokenizer can supply the cost function. Without it, costs
    use characters, matching the public codec's standalone fallback. This is
    a factoring trigger only: every row remains present regardless of size.
    """
    view = public_event_view_v2.render_public(projected, source=source)
    cost = count_units or len
    original_cost = cost(view.text)
    if original_cost > 24000:
        candidate = public_event_view_v3.render_clean_text(
            view.text, source=source, aliases=view.aliases, count_units=count_units)
        if cost(candidate.text) < original_cost:
            return forecast_clean_view(candidate)
    return forecast_clean_view(view)


def render_case(case: Mapping[str, Any], *, compressed: bool = True, policy: str = VERSION,
                count_units: Callable[[str], int] | None = None) -> str:
    """Render one policy for all splits, preserving the saved GEF prompt."""
    if case.get("source") in {"GEF_original_only", "GEF_synthetic_final"}:
        return gef_final_public.render_public(case, compressed=compressed)
    clean, _ = project_model_input(case["model_input"], policy=policy)
    if compressed:
        if policy == NEW_VERSION:
            return render_projected_forecast_view(clean, source=case.get("source"), count_units=count_units).text
        return public_event_view_v2.render_public(clean, source=case.get("source")).text
    public, _ = public_event_view_v2.project_public(clean, source=case.get("source"))
    return json.dumps(public, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
