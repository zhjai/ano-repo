"""Deterministic public-event presentation; never reconstruct a source body.

The input is the existing public JSON case, not canonical/private source rows.
Only exact, explicitly checked duplicate representations are removed. Record
order, distinct views and supplied free text survive; unknown fields are shown.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from typing import Any
import json
import re


_TEXT_KEYS = {"text", "raw_text", "event_text", "body", "headline", "description",
              "instruction", "title", "source_text", "country_containment_quote"}
_QUALIFICATIONS = {
    "source_available_at_proved", "hazard_end_not_inferred_from_CAP_expires",
    "exact_CAP_geometry_binding", "physical_event_or_load_effect_proved",
    "hazard_interval_basis", "target_equivalence", "quantified_effect",
    "native_publication_known_before_origin", "asof_known", "basis",
    "complete_native_text_in_model", "relation", "availability_basis",
    "asof_status", "research_role", "entity_match_basis",
    "projection_excludes_final_end_cancellation_and_revision_state",
    "body_version_asof_status", "reader_clock_status", "join_basis",
    "research_timing", "text_projection_kind", "entity_scope",
    "publication_basis", "operational_input_claim", "source_availability_basis",
    "context_role", "first_body_availability_proved", "availability_category",
    "status", "scope", "info.language", "enrichment_annotation.geographic_target_scope",
    "enrichment_annotation.subregion_exposure_proved",
    "enrichment_annotation.publication_status", "enrichment_annotation.operational_input_claim",
    "embedded_clock_basis", "embedded_point_duration", "role",
}
_CURATOR_ANNOTATIONS = {
    "mechanism_family", "weather_family", "revision_family", "candidate_episode_ids",
    "candidate_episode_evidence",
}
_LABELS = {
    "event_id": "event ID", "record_id": "record ID", "native_event_id": "native event ID",
    "native_notice_id": "native notice ID", "notice_id": "notice ID",
    "raw_text": "text", "event_text": "text", "publication_time": "publication time",
    "publication_time_utc": "publication time UTC", "published_at": "published",
    "available_at": "body available at", "available_time": "available time", "native_sent": "sent native",
    "associated_UGC_codes": "associated UGC", "active_association_UGC_codes": "active UGC",
    "hazard_start_utc_epoch": "hazard start UTC epoch", "hazard_end_utc_epoch": "hazard end UTC epoch",
    "native_references": "referenced messages", "referenced_source_message_ids": "referenced message IDs",
    "info_ordinal_zero_based": "CAP info ordinal (zero based)",
}
_CAP_PARAMETER_LABELS = {
    "eventEndingTime": "CAP event ending time (eventEndingTime)",
    "NWSheadline": "CAP headline (NWSheadline)",
    "HazardType": "CAP hazard type (HazardType)",
    "VTEC": "CAP VTEC",
}

EVENT_STRUCTURE_V2 = "compact-event-structure-v2"
_STRUCTURE_QUALIFICATIONS = _QUALIFICATIONS | {
    "geographic_association_method", "historical_UGC_BA_overlap_status",
    "historical_UGC_BA_overlap_qualification", "effective_notice_clock_basis",
}

# Exact saved qualifications become readable evidence statements. Unrecognised
# values fall back literally, rather than inferring a policy or availability.
_QUALIFICATION_STATEMENTS = {
    ("source_available_at_proved", "false"): "Body availability: unproved",
    ("hazard_end_not_inferred_from_CAP_expires", "true"): "Hazard end: independent of CAP expiry",
    ("exact_CAP_geometry_binding", "null"): "Exact CAP geometry binding: unknown",
    ("physical_event_or_load_effect_proved", "false"): "Physical event or load effect: unproved",
    ("hazard_interval_basis", "literal_VTEC_or_native_CAP_onset_and_eventEndingTime; no_expiry_imputation"):
        "Hazard interval basis: literal VTEC or native CAP onset and eventEndingTime; expiry is not used to infer the interval",
    ("target_equivalence", "false"): "Context area equivalence to target: not established",
    ("quantified_effect", "false"): "Quantified load effect: not established",
    ("native_publication_known_before_origin", "true"): "Native publication: known before forecast origin",
    ("asof_known", "false"): "Known at forecast origin: not established",
    ("complete_native_text_in_model", "false"): "Native text coverage: incomplete",
    ("relation", "contextual_relation"): "Relation to target: contextual",
    ("availability_basis", "historical_snapshot_initial_body_availability_unverified"):
        "Availability basis: historical snapshot; initial body availability unverified",
    ("asof_status", "unknown"): "Known at forecast origin: unknown",
    ("research_role", "oracle_notification_body_context"):
        "Evidence timing: retrospectively supplied notification body",
    ("entity_match_basis", "native_PJM-RTO_regionType_RTO"):
        "Target matching: native PJM RTO region type",
    ("projection_excludes_final_end_cancellation_and_revision_state", "true"):
        "Text coverage: final end, cancellation and revision state excluded",
    ("publication_basis", "local_calendar_day_uncertainty_envelope_not_availability"):
        "Publication basis: local calendar-day uncertainty envelope; not body availability",
    ("publication_basis", "native_issue_clock_not_availability"):
        "Publication basis: native issue clock; not body availability",
    ("operational_input_claim", "false"): "Operational replay: not claimed",
    ("enrichment_annotation.operational_input_claim", "false"): "Operational replay: not claimed",
    ("source_availability_basis", "first_version_availability_unverified"):
        "Availability basis: first-version availability unverified",
    ("context_role", "advance_mechanism_announcement_exact_native_excerpt"):
        "Text coverage: exact native excerpt of advance announcement",
    ("first_body_availability_proved", "false"): "Initial body availability: unproved",
    ("availability_category", "oracle_same_publication_day_clock_ambiguous_first_body_unverified"):
        "Evidence timing: retrospective same-publication-day context; clock ambiguous and initial body availability unverified",
    ("availability_category", "publication_envelope_preorigin_first_body_unverified"):
        "Evidence timing: publication envelope precedes forecast origin; initial body availability unverified",
    ("enrichment_annotation.geographic_target_scope", "ERCO_aggregate_only"):
        "Weather target scope: ERCO aggregate only",
    ("enrichment_annotation.subregion_exposure_proved", "false"):
        "Weather subregion exposure: unproved",
    ("enrichment_annotation.publication_status", "native_CAP_sent_preorigin_available_at_unknown"):
        "CAP timing: native sent time precedes forecast origin; body availability unknown",
    ("body_version_asof_status", "unverified_acquired_historical_representation"):
        "Body version: acquired historical representation; as-of availability unverified",
    ("reader_clock_status", "unresolved_cms_display_timezone"):
        "Reader clock: CMS display timezone unresolved",
    ("reader_clock_status", "native_file_or_MMSDM_clock"):
        "Reader clock: native file or MMSDM clock",
    ("join_basis", "explicit_region_plus_native_record_clock_or_embedded_interval_overlap"):
        "Context matching: explicit region and native record clock or embedded interval overlap",
    ("research_timing", "oracle_embedded_interval_or_postorigin_notice_clock"):
        "Evidence timing: retrospective embedded interval or post-origin notice clock",
    ("research_timing", "native_record_clock_preorigin_body_availability_unverified"):
        "Evidence timing: native record clock precedes forecast origin; body availability unverified",
    ("text_projection_kind", "native_text_excerpt_full_body_in_evaluator"):
        "Text coverage: native excerpt; full body not supplied",
    ("available_at", "null"): "Body availability time: unknown",
    ("embedded_clock_basis", "explicit_embedded_NEM_market_time_fixed_AEST_not_publication"):
        "All embedded intervals and points: NEM market time, fixed AEST (UTC+10); event clocks, not publication clocks",
    ("embedded_point_duration", "null"): "All embedded point durations: unknown",
    ("role", "generation_transmission_outage"): "Event role: generation/transmission outage",
    ("role", "consumption_unavailability_research_context"):
        "Event role: retrospective consumption-unavailability context",
    ("role", "consumption_capacity_unavailability_context"):
        "Native notice subject: consumption capacity unavailability; aggregate load effect unproven",
}


def _label(path: str) -> str:
    if path.startswith("info."):
        path = path[5:]  # The surrounding NWS record already supplies CAP scope.
    # Keep original array paths in the receipt, but expose their native names.
    parameter = re.fullmatch(r"parameters\[\d+\]\.([^\.\[]+)(.*)", path)
    if parameter:
        name, suffix = parameter.groups()
        if name == "expiredReferences":
            return _label("expired_references" + suffix)
        label = _CAP_PARAMETER_LABELS.get(name, "CAP parameter " + name)
        return label + ("." + _label(suffix[1:]) if suffix.startswith(".") else suffix)
    reference = re.fullmatch(r"(native_references|expired_references)\[\d+\](.*)", path)
    if reference:
        kind, suffix = reference.groups()
        label = "Referenced message" if kind == "native_references" else "Expired referenced message"
        return label + ("." + _label(suffix[1:]) if suffix.startswith(".") else suffix)
    event_code = re.fullmatch(r"(?:event_codes|eventCode)\[\d+\]\.(.*)", path)
    if event_code:
        return "CAP event code " + event_code.group(1)
    path = path.replace("contextual_relation.", "relation.")
    path = path.replace("enrichment_annotation.", "qualification.")
    return ".".join(_LABELS.get(p, p.replace("_", " ")) for p in path.split("."))


def _scalar(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def _vtec_native_only(record: dict) -> bool:
    """Remove the parsed copy only if every field recreates the exact string."""
    keys = {"class", "action", "office", "phenomenon", "significance",
            "event_tracking_number", "native_start", "native_end", "native_string"}
    if set(record) != keys:
        return False
    native = ("/{class}.{action}.{office}.{phenomenon}.{significance}."
              "{event_tracking_number}.{native_start}-{native_end}/").format(**record)
    return native == record["native_string"]


def _entries(content: dict) -> list[tuple[str, str]]:
    entries: list[tuple[str, str]] = []
    # The same market-time qualification applies to parsed intervals and points,
    # with explicit scope. Independent points and all native/UTC values remain.
    embedded = [item for field in ("event_intervals", "embedded_event_points")
                for item in content.get(field, []) if isinstance(item, dict)]
    embedded_basis = None
    if embedded and all("clock_basis" in item for item in embedded):
        candidate = embedded[0]["clock_basis"]
        if all(item["clock_basis"] == candidate for item in embedded):
            embedded_basis = candidate
            entries.append(("embedded_clock_basis", _scalar(candidate)))
    points = content.get("embedded_event_points", [])
    shared_null_duration = bool(points) and all(
        isinstance(point, dict) and "duration" in point and point["duration"] is None for point in points)
    if shared_null_duration:
        entries.append(("embedded_point_duration", "null"))
    vtec_strings = [v.get("native_string") for v in content.get("vtec", [])
                    if isinstance(v, dict)]

    def walk(value: Any, path: str) -> None:
        if isinstance(value, dict):
            if not value:
                return
            if path.startswith(("event_intervals[", "embedded_event_points[")):
                value = {key: child for key, child in value.items()
                         if not (key == "clock_basis" and embedded_basis is not None)
                         and not (key == "duration" and shared_null_duration
                                  and path.startswith("embedded_event_points["))}
            if path.startswith(("event_intervals[", "embedded_event_points[", "native_references[")) and all(
                    not isinstance(child, (dict, list))
                    and not (isinstance(child, str) and (";" in child or "\n" in child))
                    for child in value.values()):
                entries.append((path, "; ".join(_label(key) + ": " + _scalar(child)
                                               for key, child in value.items())))
                return
            if path.startswith("vtec[") and _vtec_native_only(value):
                entries.append((path, value["native_string"]))
                return
            # Name/value pairs are CAP-native data; the name remains literal.
            if set(value) == {"name", "value"} and isinstance(value["name"], str):
                if (path.startswith("info.parameters[") and value["name"] == "VTEC"
                        and value["value"] in vtec_strings):
                    return
                walk(value["value"], path + "." + value["name"])
                return
            for key, child in value.items():
                if not path and key == "source":
                    continue  # The exact source is printed in this record's heading.
                if not path and key == "episode_id":
                    continue  # Curated episode classification, not native evidence.
                if not path and key == "mechanism" and content.get("source") == "ERCOT":
                    continue
                if path == "enrichment_annotation" and key in _CURATOR_ANNOTATIONS:
                    continue
                walk(child, path + "." + key if path else key)
            return
        if isinstance(value, list):
            if not value:
                if path.split(".")[-1] not in {"resources", "polygons", "circles"}:
                    entries.append((path, "[]"))
                return
            if path == "referenced_source_message_ids":
                references = content.get("native_references")
                if (isinstance(references, list) and all(isinstance(r, dict) for r in references)
                        and value == [r.get("event_id") for r in references]):
                    return  # Ordered IDs already label each timed referenced message.
            if all(not isinstance(item, (dict, list)) for item in value):
                entries.append((path, ", ".join(_scalar(item) for item in value)))
                return
            # Geocodes retain original ordering, including interleaved code families.
            if path.endswith(".geocodes") and all(
                    isinstance(item, dict) and set(item) == {"name", "value"}
                    and isinstance(item["name"], str) for item in value):
                runs: list[tuple[str, list[str]]] = []
                for item in value:
                    if not runs or runs[-1][0] != item["name"]:
                        runs.append((item["name"], []))
                    runs[-1][1].append(_scalar(item["value"]))
                entries.append((path, "; ".join(name + ": " + ", ".join(codes)
                                               for name, codes in runs)))
                return
            for i, item in enumerate(value):
                walk(item, f"{path}[{i}]")
            return
        # Null and false are evidence states. Empty strings contain no prose.
        if value != "":
            entries.append((path, _scalar(value)))

    walk(content, "")
    return entries


def _records(public_case: dict) -> list[tuple[str, str, dict]]:
    records = []
    for event in public_case.get("events") or []:
        if not isinstance(event, dict) or not isinstance(event.get("content"), dict):
            raise ValueError("Public event must contain an object content")
        carrier = str(event.get("carrier", "events"))
        content = event["content"]
        extra = {key: value for key, value in event.items() if key not in {"content", "carrier"}}
        if extra:
            content = dict(content, record_metadata=extra)
        source = str(content.get("source") or content.get("issuer") or "source unspecified")
        if carrier == "synthetic_announced_programmes":
            source = "Synthetic programmes"
        if "supplemental_notices" in carrier:
            scope = "supplemental notices"
        elif "event_records" in carrier:
            scope = "public notice views"
        else:
            scope = "timed events" if carrier == "temporal_text_events" else carrier
        records.append((source, scope, content))
    context = public_case.get("context") or {}
    def check_notice_paths(value: Any, path: str = "") -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                location = path + "." + key if path else key
                if key == "retrospective_oracle_events" and location != "source_context.research_context.retrospective_oracle_events":
                    raise ValueError("Unrecognised retrospective notice container: " + location)
                check_notice_paths(child, location)
        elif isinstance(value, list):
            for i, child in enumerate(value):
                check_notice_paths(child, f"{path}[{i}]")
    check_notice_paths(context)
    research = ((context.get("source_context") or {}).get("research_context") or {})
    for content in research.get("retrospective_oracle_events") or []:
        if not isinstance(content, dict):
            raise ValueError("Retrospective public notice must be an object")
        source = str(content.get("source") or content.get("issuer") or "source unspecified")
        records.append((source, "retrospective notices", content))
    return records


def _lines(entries: list[tuple[str, str]]) -> list[str]:
    """Keep prose bytes literal; pack only short metadata onto shared lines."""
    # Pair native expired-reference IDs and sent clocks only in the display;
    # individual original paths and values remain unchanged in the receipt.
    displayed = []
    index = 0
    while index < len(entries):
        path, value = entries[index]
        reference = re.fullmatch(r"(info\.parameters\[\d+\]\.expiredReferences\[\d+\])\.(event_id|sent)", path)
        if reference and index + 1 < len(entries):
            root, field = reference.groups()
            next_path, next_value = entries[index + 1]
            other_field = "sent" if field == "event_id" else "event_id"
            if next_path == root + "." + other_field:
                displayed.append((root, _label(field) + ": " + value + "; "
                                  + _label(other_field) + ": " + next_value))
                index += 2
                continue
        displayed.append((path, value))
        index += 1
    lines, metadata = [], []
    for path, value in displayed:
        text = _QUALIFICATION_STATEMENTS.get((path, value), _label(path) + ": " + value)
        if path.split(".")[-1] in _TEXT_KEYS or ";" in text or "\n" in value or len(text) > 180:
            if metadata:
                lines.append("; ".join(metadata))
                metadata = []
            lines.append(text)
        else:
            if metadata and sum(len(t) + 2 for t in metadata) + len(text) > 180:
                lines.append("; ".join(metadata))
                metadata = []
            metadata.append(text)
    if metadata:
        lines.append("; ".join(metadata))
    return lines


def _sharing_field(content: dict, path: str) -> tuple[str, str] | None:
    """Whitelist non-clock evidence and bind its exact typed source value.

    CAP parameters are deliberately restricted to the non-clock HazardType.
    VTEC and native/expired references are never candidates, even if identical.
    """
    if path in {"text", "raw_text", "event_text", "body", "title",
                "info.headline", "info.description", "info.instruction"}:
        category = "prose"
    elif (path in {"associated_UGC_codes", "active_association_UGC_codes"}
          or re.fullmatch(r"info\.areas\[\d+\]\.(description|geocodes|polygons|circles)", path)):
        category = "geography"
    elif re.fullmatch(r"info\.(event_codes|eventCode)\[\d+\]\.(SAME|NationalWeatherService)", path):
        category = "codes"
    elif re.fullmatch(r"info\.parameters\[\d+\]\.HazardType", path):
        category = "codes"
    else:
        return None
    found, current = _source_field(content, path)
    if not found:
        return None
    # JSON with original insertion/list order protects against equality after
    # flattening (e.g. different code arrays that happen to print identically).
    identity = json.dumps(current, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    return category, identity


def _source_field(content: dict, path: str) -> tuple[bool, Any]:
    """Resolve an original field, including the native CAP name/value carrier."""
    current: Any = content
    for part in path.split("."):
        match = re.fullmatch(r"([^\[]+)(?:\[(\d+)\])?", part)
        if match is None:
            return False, None
        name, index = match.groups()
        if isinstance(current, dict) and name in current:
            current = current[name]
        elif isinstance(current, dict) and current.get("name") == name and "value" in current:
            current = current["value"]
        else:
            return False, None
        if index is not None:
            if not isinstance(current, list) or int(index) >= len(current):
                return False, None
            current = current[int(index)]
    return True, current


def _qualification_identity(content: dict, path: str) -> str | None:
    found, value = _source_field(content, path)
    if not found and path == "embedded_clock_basis":
        embedded = [item for field in ("event_intervals", "embedded_event_points")
                    for item in content.get(field, []) if isinstance(item, dict)]
        values = [item.get("clock_basis") for item in embedded]
        if not embedded or not all("clock_basis" in item for item in embedded):
            return None
        identities = [json.dumps(item, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
                      for item in values]
        if len(set(identities)) != 1:
            return None
        return identities[0]
    if not found and path == "embedded_point_duration":
        points = content.get("embedded_event_points", [])
        if points and all(isinstance(p, dict) and p.get("duration", "missing") is None for p in points):
            return "null"
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")) if found else None


def _structure_lines(entries: list[tuple[str, str]], content: dict) -> list[str]:
    """Use reference tables only for exact ordinary ID/sent pairs.

    Every ID and complete native sent clock remains literal in its own record.
    Unsafe cells, extra native fields and nonstandard representations fall back
    to the established literal renderer, without parsing or rewriting prose.
    """
    output, pending = [], []
    index = 0
    while index < len(entries):
        path, value = entries[index]
        native = re.fullmatch(r"native_references\[\d+\]", path)
        expired = re.fullmatch(r"(info\.parameters\[\d+\]\.expiredReferences\[\d+\])\.(event_id|sent)", path)
        root = path if native else expired.group(1) if expired else None
        found, original = _source_field(content, root) if root else (False, None)
        size = 1 if native else 2
        safe = (found and isinstance(original, dict) and set(original) == {"event_id", "sent"}
                and all(isinstance(v, str) and not any(c in v for c in "|;\n\r") for v in original.values()))
        if safe and native:
            safe = value == "; ".join(_label(k) + ": " + _scalar(v) for k, v in original.items())
        elif safe:
            safe = (index + 1 < len(entries) and entries[index + 1][0] == root + "." +
                    ("sent" if expired.group(2) == "event_id" else "event_id")
                    and dict((p.rsplit(".", 1)[1], v) for p, v in entries[index:index + 2]) == original)
        if not safe:
            pending.append((path, value))
            index += 1
            continue
        output.extend(_lines(pending))
        pending = []
        kind = "Referenced messages" if native else "Expired referenced messages"
        # A table is continued only within the same adjacent native array.
        array = root.rsplit("[", 1)[0]
        table = [kind + " (original order; sent is native message clock):", "event ID|sent"]
        while safe:
            table.append(original["event_id"] + "|" + original["sent"])
            index += size
            if index >= len(entries):
                break
            next_path = entries[index][0]
            next_root = next_path if native else next_path.rsplit(".", 1)[0]
            safe = bool(re.fullmatch(re.escape(array) + r"\[\d+\]", next_root))
            found, original = _source_field(content, next_root) if safe else (False, None)
            safe = (safe and found and isinstance(original, dict) and set(original) == {"event_id", "sent"}
                    and all(isinstance(v, str) and not any(c in v for c in "|;\n\r") for v in original.values()))
            if safe and native:
                safe = entries[index][1] == "; ".join(_label(k) + ": " + _scalar(v) for k, v in original.items())
            elif safe:
                safe = (index + 1 < len(entries) and
                        {p for p, _ in entries[index:index + 2]} == {next_root + ".event_id", next_root + ".sent"}
                        and dict((p.rsplit(".", 1)[1], v) for p, v in entries[index:index + 2]) == original)
        output.extend(table)
    output.extend(_lines(pending))
    return output


def expand_event_receipt(receipt: dict) -> list[list[tuple[str, str]]]:
    """Expand direct backward field references, rejecting chains or scope drift."""
    records = receipt["records"]
    expanded: list[list[tuple[str, str]]] = []
    for i, record in enumerate(records):
        row = []
        if len({field["path"] for field in record["fields"]}) != len(record["fields"]):
            raise ValueError("Duplicate event field path in receipt")
        for field in record["fields"]:
            path = field["path"]
            if "reference_record" not in field:
                row.append((path, field["value"]))
                continue
            ordinal = field["reference_record"]
            if isinstance(ordinal, bool) or not isinstance(ordinal, int):
                raise ValueError("Event reference ordinal must be an integer")
            anchor = ordinal - 1
            if not 0 <= anchor < i:
                raise ValueError("Event reference must point directly backward")
            earlier = records[anchor]
            if any(record[key] != earlier[key] for key in ("source", "scope", "carrier")):
                raise ValueError("Event reference crosses source or scope")
            target = next((item for item in earlier["fields"] if item["path"] == path), None)
            if target is None or "reference_record" in target:
                raise ValueError("Event reference has no direct literal field")
            if field.get("source_value") != target.get("source_value") or "source_value" not in field:
                raise ValueError("Event reference changes typed source value")
            row.append((path, target["value"]))
        expanded.append(row)
    layout = receipt.get("layout", {})
    for shared in layout.get("shared_qualifications", []):
        if shared["path"] not in _STRUCTURE_QUALIFICATIONS:
            raise ValueError("Field is not an allowed shared qualification")
        ordinals = shared["records"]
        if len(ordinals) < 2 or len(set(ordinals)) != len(ordinals):
            raise ValueError("Invalid qualification record group")
        for ordinal in ordinals:
            if isinstance(ordinal, bool) or not isinstance(ordinal, int) or not 1 <= ordinal <= len(records):
                raise ValueError("Invalid qualification record ordinal")
            record = records[ordinal - 1]
            if any(record[k] != shared[k] for k in ("source", "scope", "carrier")):
                raise ValueError("Shared qualification crosses source, scope or carrier")
            field = next((f for f in record["fields"] if f["path"] == shared["path"]), None)
            if (field is None or "reference_record" in field or field.get("value") != shared["value"]
                    or field.get("source_value") != shared["source_value"]):
                raise ValueError("Shared qualification changes literal or typed value")
    for local in layout.get("local_geography", []):
        if {local["path"], local["anchor_path"]} != {"associated_UGC_codes", "active_association_UGC_codes"}:
            raise ValueError("Field is not an allowed local UGC alias")
        ordinal = local["record"]
        if isinstance(ordinal, bool) or not isinstance(ordinal, int) or not 1 <= ordinal <= len(records):
            raise ValueError("Invalid local geography record ordinal")
        if any(f.get("reference_record") == ordinal and f["path"] == local["path"]
               for record in records for f in record["fields"]):
            raise ValueError("Local geography hides a visible backward literal anchor")
        fields = {f["path"]: f for f in records[ordinal - 1]["fields"]}
        paths = list(fields)
        target, anchor = fields.get(local["path"]), fields.get(local["anchor_path"])
        if (target is None or anchor is None or target is anchor or "reference_record" in target
                or "reference_record" in anchor or target["value"] != anchor["value"]
                or paths.index(local["anchor_path"]) >= paths.index(local["path"])
                or target.get("source_value") != local["source_value"]
                or anchor.get("source_value") != local["source_value"]):
            raise ValueError("Local geography has no exact direct literal anchor")
    return expanded


def _published_after_origin(content: dict, public_case: dict) -> list[str]:
    """Compare only explicit timezone-aware public publication/origin clocks.

    Missing, naive and unparseable clocks are unknown, not inferred from the
    forecast grid. The note makes no body-availability or event-interval claim.
    """
    def known_clock(value: Any) -> datetime | None:
        if not isinstance(value, str) or "T" not in value:
            return None
        try:
            clock = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
        return clock if clock.utcoffset() is not None else None

    origin_literal = (public_case.get("task") or {}).get("origin_time")
    origin = known_clock(origin_literal)
    if origin is None:
        return []
    notes = []
    for field in ("published_at", "publication_time_utc"):
        clock = known_clock(content.get(field))
        if clock is not None and clock > origin:
            notes.append(f"Publication timing: {_label(field)} is after forecast origin {origin_literal}.")
    return notes


def render_events_with_receipt(public_case: dict, *, structure_format: str | None = None) -> tuple[str, dict]:
    """Render all supplied public records in original order, without summaries.

    Record ordinals are presentation positions, never replacement citation IDs.
    Shared fields apply only to the explicitly listed source/container records.
    Source clocks remain per view, even when some happen to be identical.
    """
    if structure_format is not None:
        if structure_format != EVENT_STRUCTURE_V2:
            raise ValueError("Unknown event structure format: " + structure_format)
        return _render_structure_events(public_case)
    records = _records(public_case)
    if not records:
        return "", {"version": "compact-event-sharing-v1", "records": [], "references": []}
    rows = [_entries(record) for _, _, record in records]
    if any(len({path for path, _ in row}) != len(row) for row in rows):
        raise ValueError("Duplicate public event field path")
    carriers = [str(event.get("carrier", "events")) for event in public_case.get("events") or []]
    carriers += ["source_context.research_context.retrospective_oracle_events"] * (len(records) - len(carriers))
    receipt = {"version": "compact-event-sharing-v1", "records": [
        {"source": source, "scope": scope, "carrier": carriers[i],
         "fields": [{"path": p, "value": v} for p, v in rows[i]]}
        for i, (source, scope, _) in enumerate(records)], "references": []}
    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for i, (source, scope, _) in enumerate(records):
        groups[(source, scope)].append(i)
    shared = {}
    for group, indices in groups.items():
        if len(indices) < 2:
            continue
        candidate = [(p, v) for p, v in rows[indices[0]] if p in _QUALIFICATIONS]
        common = [(p, v) for p, v in candidate if all((p, v) in rows[i] for i in indices[1:])]
        if common:
            shared[group] = common
    visible_rows = [[(p, v) for p, v in row if (p, v) not in shared.get(records[i][:2], [])]
                    for i, row in enumerate(rows)]
    first: dict[tuple[str, str, str, str, str], int] = {}
    for i, row in enumerate(visible_rows):
        source, scope, content = records[i]
        labels = [_label(path) for path, _ in row]
        for j, (path, value) in enumerate(row):
            if labels.count(_label(path)) != 1:
                continue  # A readable backward target must identify one field.
            candidate = _sharing_field(content, path)
            if candidate is None or not value or value == "[]":
                continue
            category, identity = candidate
            field = next(item for item in receipt["records"][i]["fields"] if item["path"] == path)
            field["source_value"] = identity
            key = (source, scope, carriers[i], path, identity)
            if key not in first:
                first[key] = i
                continue
            anchor = first[key]
            reference = f"Same as Record {anchor + 1}, {_label(path)} (exact)."
            if len(reference) >= len(value):
                continue  # No arbitrary threshold or model/tokenizer dependency.
            visible_rows[i][j] = (path, reference)
            field.pop("value")
            field["reference_record"] = anchor + 1
            receipt["references"].append({"record": i + 1, "path": path,
                                          "anchor_record": anchor + 1, "category": category})
    output = ["Events:"]
    if receipt["references"]:
        output.append("'Same as Record N, field (exact)' uses that earlier literal field in full; all other fields belong to this record.")
    for group, common in shared.items():
        scope = ", ".join(str(i + 1) for i in groups[group])
        output.append(f"Shared qualifications — {group[0]}, {group[1]}, records {scope}:")
        output.extend(_lines(common))
    if any(record.get("vtec") for _, _, record in records):
        output.append("VTEC native format: /class.action.office.phenomenon.significance.tracking_number.start-end/.")
    for i, (source, scope, content) in enumerate(records):
        description = source if source == "Synthetic programmes" else source + ", " + scope
        output.append(f"\nRecord {i + 1} — {description}:")
        output.extend(_lines(visible_rows[i]))
        output.extend(_published_after_origin(content, public_case))
    return "\n".join(output), receipt


def _render_structure_events(public_case: dict) -> tuple[str, dict]:
    """Explicit candidate layout; the default and frozen inputs stay unchanged."""
    _, receipt = render_events_with_receipt(public_case)
    records = _records(public_case)
    rows = expand_event_receipt(receipt)
    receipt["version"] = EVENT_STRUCTURE_V2
    receipt["layout"] = {"shared_qualifications": [], "local_geography": []}
    if not records:
        return "", receipt
    visible = []
    for row in receipt["records"]:
        visible.append([(field["path"], field.get("value", "Same as Record " + str(field.get("reference_record")) +
                        ", " + _label(field["path"]) + " (exact).")) for field in row["fields"]])
    # Subset sharing is explicitly scoped by source, container, carrier and
    # typed original value. It never shares a structured event timestamp.
    groups: dict[tuple[str, str, str, str, str, str], list[int]] = defaultdict(list)
    for i, (source, scope, content) in enumerate(records):
        for path, value in rows[i]:
            if path in _STRUCTURE_QUALIFICATIONS:
                identity = _qualification_identity(content, path)
                if identity is not None:
                    groups[(source, scope, receipt["records"][i]["carrier"], path, identity, value)].append(i)
    shared_blocks: dict[tuple[str, str, str, tuple[int, ...]], list[tuple[str, str]]] = defaultdict(list)
    for (source, scope, carrier, path, identity, value), indices in groups.items():
        if len(indices) < 2:
            continue
        shared_blocks[(source, scope, carrier, tuple(indices))].append((path, value))
        receipt["layout"]["shared_qualifications"].append(
            {"source": source, "scope": scope, "carrier": carrier, "path": path,
             "source_value": identity, "value": value, "records": [i + 1 for i in indices]})
        for i in indices:
            next(f for f in receipt["records"][i]["fields"] if f["path"] == path)["source_value"] = identity
            visible[i] = [(p, v) for p, v in visible[i] if p != path]
    # Only a same-record direct literal may anchor an identical ordered UGC
    # list. Different subsets, code families, empty lists and references remain.
    literal_anchors = {(ref["anchor_record"], ref["path"]) for ref in receipt["references"]}
    for i, (_, _, content) in enumerate(records):
        anchor = None
        for j, (path, value) in enumerate(visible[i]):
            if path not in {"associated_UGC_codes", "active_association_UGC_codes"}:
                continue
            found, codes = _source_field(content, path)
            field = next(f for f in receipt["records"][i]["fields"] if f["path"] == path)
            if not found or not isinstance(codes, list) or not codes or "reference_record" in field:
                continue
            identity = json.dumps(codes, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
            if anchor is None:
                anchor = (path, identity)
                continue
            replacement = "Same as this record's " + _label(anchor[0]) + " (exact ordered codes)."
            if (identity == anchor[1] and len(replacement) < len(value)
                    and (i + 1, path) not in literal_anchors):
                visible[i][j] = (path, replacement)
                field["source_value"] = identity
                next(f for f in receipt["records"][i]["fields"] if f["path"] == anchor[0])["source_value"] = identity
                receipt["layout"]["local_geography"].append(
                    {"record": i + 1, "path": path, "anchor_path": anchor[0], "source_value": identity})
    output = ["Events:"]
    if receipt["references"]:
        output.append("'Same as Record N, field (exact)' uses that earlier literal field in full; all other fields belong to this record.")
    if receipt["layout"]["local_geography"]:
        output.append("'Same as this record's field' copies its direct literal ordered codes; code families and field roles remain separate.")
    for (source, scope, carrier, indices), common in shared_blocks.items():
        output.append(f"Shared qualifications — {source}, {scope}, carrier {carrier}, records " +
                      ", ".join(str(i + 1) for i in indices) + ":")
        output.extend(_lines(common))
    if any(record.get("vtec") for _, _, record in records):
        output.append("VTEC native format: /class.action.office.phenomenon.significance.tracking_number.start-end/.")
    for i, (source, scope, content) in enumerate(records):
        description = source if source == "Synthetic programmes" else source + ", " + scope
        output.append(f"\nRecord {i + 1} — {description}:")
        output.extend(_structure_lines(visible[i], content))
        output.extend(_published_after_origin(content, public_case))
    if expand_event_receipt(receipt) != rows:
        raise ValueError("Event structure receipt does not expand exactly")
    return "\n".join(output), receipt


def render_events(public_case: dict, *, structure_format: str | None = None) -> str:
    """Render public events with exact backward sharing and per-record clocks."""
    return render_events_with_receipt(public_case, structure_format=structure_format)[0]
