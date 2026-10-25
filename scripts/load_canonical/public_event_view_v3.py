"""Opt-in second-pass factoring of the exact, readable v2 public event tree.

This module never changes the clean projection or the existing v2 renderer.
It factors literal repeated lines/sentences and structures created by v2's
first pass (for example identical nested table schemas). ISO clocks remain
literal in each record; identical clock fields can share a named field locally.
Definitions remain visible in the model input and use v2's exact decoder.
"""
from __future__ import annotations

import copy
import json
import re
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Callable, Mapping, Optional

from . import public_event_view_v2 as v2

VERSION = "public-event-view-v3"
_ISO_CLOCK = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})$")
LEGEND = (
    "Event view v3: all literal definitions are visible below; same_as uses the "
    "complete exact definition. text_parts concatenate in order, adding nothing. "
    "Record-table rows follow columns; common fields apply to every row; "
    "absent:true distinguishes omitted fields from null. time_axis expands "
    "start+i*step_seconds for count original ISO timestamps. E/X identity aliases "
    "keep the v2 visible-message/outside-window distinction and literal reference "
    "senders/clocks. literal_array_runs lists [count,value] pairs, repeating each "
    "exact value count times in order; literal_object escapes a native object "
    "as key/value pairs. table_rows applies its named schema to every ordered "
    "row; schemas contain exact common fields, columns and nesting flags. "
    "same_column uses the exact named field in this record; identity_template "
    "means event_id@publication_time:info{info_ordinal_zero_based}. "
    "named_value_runs lists ordered [name,values] runs of exact name/value records. "
    "native_relation_literal escapes a native object as literal key/value pairs. "
    "All notices, values, codes and revision relations survive.\n"
)
# Alternate prefix for the residual projection; default v3 rendering remains
# byte-compatible with historical LEGEND output.
FORECAST_CLEAN_LEGEND = LEGEND.replace(
    "E/X identity aliases keep the v2 visible-message/outside-window distinction and literal reference senders/clocks.",
    "E aliases identify records present in this case; X aliases identify referenced records absent from it. CAP reference sent times remain literal; sender is retained when needed to distinguish namespaces.",
    1,
)


def _dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _strings(value: Any):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from _strings(child)
    elif isinstance(value, list):
        for child in value:
            yield from _strings(child)


def _fragments(value: str):
    # Only complete literal lines and sentence/semicolon spans; no word-level
    # token codes, normalization, trimming or changes to punctuation/spacing.
    for line in value.splitlines(keepends=True):
        if len(line) >= 48:
            yield line
        for part in re.split(r"(?<=[.;])(?=\s)", line):
            if len(part) >= 64:
                yield part


def factor_encoded(encoded: Mapping[str, Any], *, count_units: Optional[Callable[[str], int]] = None,
                   rule_counts: Optional[dict[str, int]] = None) -> dict[str, Any]:
    """Apply exact literal and post-layout structural deduplication to v2."""
    original = copy.deepcopy(encoded)
    cost = lru_cache(maxsize=None)(count_units or len)
    statistics = rule_counts if rule_counts is not None else {}

    def hit(rule: str) -> None:
        statistics[rule] = statistics.get(rule, 0) + 1

    def actual(value: Any) -> Any:
        return v2.decode_public({"definitions": original["definitions"], "public": value})

    def relations(value: Any) -> Any:
        if isinstance(value, dict):
            if set(value) in ({"same_column"}, {"identity_template"}, {"named_value_runs"}):
                return {"native_relation_literal": [[key, relations(child)] for key, child in value.items()]}
            if set(value) == {"record_table"}:
                spec = value["record_table"]
                columns = spec["columns"]
                rows = [[relations(cell) for cell in row] for row in spec["rows"]]
                # Consecutive exact names retain both group order and code order.
                if set(columns) == {"name", "value"} and not spec["common"]:
                    ni, vi = columns.index("name"), columns.index("value")
                    groups = []
                    seen = set()
                    valid = True
                    for row_index, row in enumerate(spec["rows"]):
                        name = actual(row[ni])
                        if not isinstance(name, str) or any(cell == {"absent": True} for cell in row):
                            valid = False
                            break
                        if groups and groups[-1][0] == name:
                            groups[-1][1].append(rows[row_index][vi])
                        elif name in seen:
                            valid = False
                            break
                        else:
                            seen.add(name)
                            groups.append([name, [relations(row[vi])]])
                    candidate = {"named_value_runs": groups}
                    if valid and cost(_dump(candidate)) < cost(_dump(value)):
                        hit("contiguous_name_value_runs")
                        return candidate
                for row, original_row in zip(rows, spec["rows"]):
                    values = {key: actual(cell) for key, cell in zip(columns, original_row)}
                    values.update({key: actual(cell) for key, cell in spec["common"].items()})
                    clock_origins = {}
                    preferred = [key for key in ("publication_time", "info/effective_time_utc", "info/onset_time_utc", "info/expires_time_utc", "native_sent") if key in columns]
                    for key in preferred + [key for key in columns if key not in preferred]:
                        literal = values.get(key)
                        if not isinstance(literal, str) or not _ISO_CLOCK.fullmatch(literal):
                            continue
                        if literal not in clock_origins:
                            clock_origins[literal] = key
                            continue
                        marker = {"same_column": clock_origins[literal]}
                        index = columns.index(key)
                        if cost(_dump(marker)) < cost(_dump(row[index])):
                            row[index] = marker
                            hit("same_column_exact_clock")
                    for destination, origin in (("version", "event_record_uid"),
                                                ("active_association_UGC_codes", "associated_UGC_codes")):
                        if destination in columns and origin in values and destination in values and _dump(values[destination]) == _dump(values[origin]):
                            marker = {"same_column": origin}
                            index = columns.index(destination)
                            if cost(_dump(marker)) < cost(_dump(row[index])):
                                row[index] = marker
                                hit("same_column_" + destination)
                    if all(key in values for key in ("event_record_uid", "event_id", "publication_time", "info_ordinal_zero_based")) and "event_record_uid" in columns:
                        expected = str(values["event_id"]) + "@" + str(values["publication_time"]) + ":info" + str(values["info_ordinal_zero_based"])
                        if values["event_record_uid"] == expected:
                            index = columns.index("event_record_uid")
                            marker = {"identity_template": True}
                            if cost(_dump(marker)) < cost(_dump(row[index])):
                                row[index] = marker
                                hit("identity_template")
                return {"record_table": dict(spec, common=relations(spec["common"]), rows=rows)}
            return {key: relations(child) for key, child in value.items()}
        if isinstance(value, list):
            return [relations(child) for child in value]
        return value

    prepared_original = relations(original)
    string_values = list(_strings(prepared_original))
    counts = Counter(fragment for value in string_values for fragment in set(_fragments(value)))
    candidates = sorted((s for s, count in counts.items() if count >= 2),
                        key=lambda s: (-len(s), s))
    # Longest literal match at each position keeps reconstruction unambiguous.
    pattern = re.compile("|".join(re.escape(s) for s in candidates)) if candidates else None

    def split(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: split(child) for key, child in value.items()}
        if isinstance(value, list):
            return [split(child) for child in value]
        if not isinstance(value, str) or pattern is None:
            return value
        parts = []
        start = 0
        for match in pattern.finditer(value):
            if match.start() > start:
                parts.append(value[start:match.start()])
            parts.append(match.group(0))
            start = match.end()
        if not parts:
            return value
        if start < len(value):
            parts.append(value[start:])
        return {"text_parts": parts} if len(parts) > 1 else parts[0]

    prepared = split(prepared_original)
    structural_counts = Counter()

    def eligible(value: Any) -> bool:
        return (isinstance(value, (dict, list, str))
                and not (isinstance(value, str) and _ISO_CLOCK.fullmatch(value))
                and len(_dump(value)) >= (12 if isinstance(value, str) else 48)
                and not (isinstance(value, dict) and set(value) == {"same_as"}))

    def count(value: Any) -> None:
        if eligible(value):
            structural_counts[_dump(value)] += 1
        if isinstance(value, dict):
            for child in value.values():
                count(child)
        elif isinstance(value, list):
            for child in value:
                count(child)

    count(prepared)
    definitions = prepared["definitions"]
    extras = {}
    names = {}

    def pack(value: Any, skip: bool = False) -> Any:
        key = _dump(value) if eligible(value) else None
        repetitions = structural_counts[key] if key is not None else 0
        reference_cost = cost('{"same_as":"L999"}')
        gain = ((repetitions - 1) * cost(key) - repetitions * reference_cost
                - cost('"L999":')) if key is not None else 0
        if not skip and key is not None and repetitions >= 2 and gain > 0:
            if key not in names:
                name = "L" + str(len(names) + 1)
                names[key] = name
                extras[name] = pack(value, True)
            return {"same_as": names[key]}
        if isinstance(value, dict):
            if set(value) == {"record_table"}:
                spec = value["record_table"]
                return {"record_table": dict(spec, common=pack(spec["common"]),
                    columns=pack(spec["columns"]), rows=[pack(row) for row in spec["rows"]])}
            return {key: pack(child) for key, child in value.items()}
        if isinstance(value, list):
            return [pack(child) for child in value]
        return value

    public = pack(prepared["public"])
    packed_definitions = {key: pack(value) for key, value in definitions.items()}
    packed_definitions.update(extras)
    result = {"version": VERSION, "definitions": packed_definitions, "public": public}

    def runs(value: Any) -> Any:
        if isinstance(value, dict):
            if set(value) in ({"literal_array_runs"}, {"literal_object"}, {"table_rows"}):
                return {"literal_object": [[key, runs(child)] for key, child in value.items()]}
            if set(value) == {"record_table"}:
                spec = value["record_table"]
                return {"record_table": dict(spec, common=runs(spec["common"]),
                    columns=runs(spec["columns"]), rows=[runs(row) for row in spec["rows"]])}
            return {key: runs(child) for key, child in value.items()}
        if isinstance(value, list):
            children = [runs(child) for child in value]
            groups = []
            for child in children:
                if groups and _dump(groups[-1][1]) == _dump(child):
                    groups[-1][0] += 1
                else:
                    groups.append([1, child])
            candidate = {"literal_array_runs": groups}
            if any(count >= 2 for count, _ in groups) and cost(_dump(candidate)) < cost(_dump(children)):
                return candidate
            return children
        return value

    result = runs(result)
    schemas = {}
    schema_names = {}

    def table_schemas(value: Any) -> Any:
        if isinstance(value, dict):
            if set(value) == {"literal_object"}:
                return value
            if set(value) == {"record_table"}:
                spec = value["record_table"]
                schema = {key: table_schemas(child) for key, child in spec.items() if key != "rows"}
                key = _dump(schema)
                if key not in schema_names:
                    name = "S" + str(len(schema_names) + 1)
                    schema_names[key] = name
                    schemas[name] = schema
                return {"table_rows": {"schema": schema_names[key],
                    "rows": [table_schemas(row) for row in spec["rows"]]}}
            return {key: table_schemas(child) for key, child in value.items()}
        if isinstance(value, list):
            return [table_schemas(child) for child in value]
        return value

    result = table_schemas(result)
    result["schemas"] = schemas
    schema_uses = Counter()

    def count_schema_uses(value: Any) -> None:
        if isinstance(value, dict):
            if set(value) == {"table_rows"}:
                schema_uses[value["table_rows"]["schema"]] += 1
            for child in value.values():
                count_schema_uses(child)
        elif isinstance(value, list):
            for child in value:
                count_schema_uses(child)

    count_schema_uses(result)

    def inline_single_schema(value: Any) -> Any:
        if isinstance(value, dict):
            if set(value) == {"table_rows"}:
                spec = value["table_rows"]
                if schema_uses[spec["schema"]] <= 1:
                    return {"record_table": inline_single_schema(dict(schemas[spec["schema"]], rows=spec["rows"]))}
            return {key: inline_single_schema(child) for key, child in value.items()}
        if isinstance(value, list):
            return [inline_single_schema(child) for child in value]
        return value

    result = inline_single_schema(result)
    result["schemas"] = {key: value for key, value in result["schemas"].items() if schema_uses[key] >= 2}
    result = {key: result[key] for key in ("version", "definitions", "schemas", "public")}
    if _dump(decode_encoded(result)) != _dump(v2.decode_public(original)):
        raise ValueError("v3 factoring changed the exact public tree")
    return result


def _parse_encoded(text: str) -> dict[str, Any]:
    lines = iter(text.splitlines())

    def parse(first: str) -> Any:
        content = first.strip()
        if content == "{":
            result = {}
            for line in lines:
                stripped = line.strip()
                if stripped == "}":
                    return result
                key, child = stripped.split(": ", 1)
                result[key] = parse(child)
            raise ValueError("unterminated object")
        if content.startswith("record table (rows="):
            expected = int(content.split("=", 1)[1].rstrip(")"))
            spec = {"common": {}, "columns": [], "rows": [], "flattened_paths": True}
            for line in lines:
                stripped = line.strip()
                if stripped == "end table":
                    if len(spec["rows"]) != expected:
                        raise ValueError("table row count mismatch")
                    return {"record_table": spec}
                key, payload = stripped.split(": ", 1)
                if key == "row":
                    spec["rows"].append(json.loads(payload))
                else:
                    spec[key] = json.loads(payload)
            raise ValueError("unterminated table")
        if content.startswith("schema table (rows="):
            expected = int(content.split("=", 1)[1].rstrip(")"))
            spec = {"schema": None, "rows": []}
            for line in lines:
                stripped = line.strip()
                if stripped == "end table":
                    if len(spec["rows"]) != expected or spec["schema"] is None:
                        raise ValueError("schema table row count/schema mismatch")
                    return {"table_rows": spec}
                key, payload = stripped.split(": ", 1)
                if key == "row":
                    spec["rows"].append(json.loads(payload))
                elif key == "schema":
                    spec["schema"] = json.loads(payload)
                else:
                    raise ValueError("unknown schema table field")
            raise ValueError("unterminated schema table")
        if content.startswith("same as "):
            return {"same_as": content[len("same as "):]}
        return json.loads(content)

    result = parse(next(lines))
    if next(lines, None) is not None:
        raise ValueError("trailing readable text")
    return result


def _readable(value: Any, indent: int = 0, label: Optional[str] = None) -> list[str]:
    """Keep schemas visible and put each factored table row on its own line."""
    pad = " " * indent
    prefix = (label + ": ") if label else ""
    if isinstance(value, dict) and set(value) == {"record_table"}:
        spec = value["record_table"]
        return ([pad + prefix + "record table (rows=" + str(len(spec["rows"])) + ")",
                 pad + "common: " + _dump(spec["common"]),
                 pad + "columns: " + _dump(spec["columns"])]
                + [pad + "row: " + _dump(row) for row in spec["rows"]]
                + [pad + "end table"])
    if isinstance(value, dict) and set(value) == {"table_rows"}:
        spec = value["table_rows"]
        return ([pad + prefix + "schema table (rows=" + str(len(spec["rows"])) + ")",
                 pad + "schema: " + _dump(spec["schema"])]
                + [pad + "row: " + _dump(row) for row in spec["rows"]]
                + [pad + "end table"])
    if isinstance(value, dict) and set(value) == {"same_as"}:
        return [pad + prefix + "same as " + value["same_as"]]
    if isinstance(value, dict):
        lines = [pad + prefix + "{"]
        for key, child in value.items():
            lines.extend(_readable(child, indent + 2, str(key)))
        lines.append(pad + "}")
        return lines
    return [pad + prefix + _dump(value)]


def decode_encoded(encoded: Mapping[str, Any]) -> Any:
    """Resolve all schemas before applying the unchanged v2 semantic decoder."""
    definitions = encoded["definitions"]
    schemas = encoded.get("schemas", {})

    def resolve(value: Any, active: tuple = ()) -> Any:
        if isinstance(value, dict):
            if set(value) == {"native_relation_literal"}:
                return {"native_relation_literal": [[key, resolve(child, active)] for key, child in resolve(value["native_relation_literal"], active)]}
            if set(value) == {"named_value_runs"}:
                return [{"name": name, "value": child}
                        for name, children in resolve(value["named_value_runs"], active) for child in children]
            if set(value) == {"table_rows"}:
                spec = value["table_rows"]
                schema = resolve(schemas[spec["schema"]], active)
                return resolve({"record_table": dict(schema, rows=spec["rows"])}, active)
            if set(value) == {"record_table"}:
                spec = value["record_table"]
                common = resolve(spec["common"], active)
                columns = resolve(spec["columns"], active)
                rows = []
                for row in spec["rows"]:
                    cells = resolve(row, active)
                    fields = dict(common, **dict(zip(columns, cells)))

                    def local(key: str, trail: tuple = ()) -> Any:
                        if key in trail:
                            raise ValueError("cyclic same-column relation")
                        child = fields[key]
                        if isinstance(child, dict) and set(child) == {"same_column"}:
                            return local(child["same_column"], trail + (key,))
                        if isinstance(child, dict) and child == {"identity_template": True}:
                            return str(local("event_id", trail + (key,))) + "@" + str(local("publication_time", trail + (key,))) + ":info" + str(local("info_ordinal_zero_based", trail + (key,)))
                        return child

                    rows.append([local(key) for key in columns])
                return {"record_table": dict(spec, common=common, columns=columns, rows=rows)}
            if set(value) == {"literal_object"}:
                return {key: resolve(child, active) for key, child in value["literal_object"]}
            if set(value) == {"literal_array_runs"}:
                result = []
                for count, child in value["literal_array_runs"]:
                    if type(count) is not int or count < 1:
                        raise ValueError("invalid literal run count")
                    result.extend(copy.deepcopy(resolve(child, active)) for _ in range(count))
                return result
            if set(value) == {"same_as"}:
                name = value["same_as"]
                if name in active:
                    raise ValueError("cyclic literal definition")
                return resolve(definitions[name], active + (name,))
            if set(value) == {"text_parts"}:
                return "".join(resolve(value["text_parts"], active))
            return {key: resolve(child, active) for key, child in value.items()}
        if isinstance(value, list):
            return [resolve(child, active) for child in value]
        return value

    def unescape(value: Any) -> Any:
        if isinstance(value, dict):
            if set(value) == {"native_relation_literal"}:
                return {key: unescape(child) for key, child in value["native_relation_literal"]}
            return {key: unescape(child) for key, child in value.items()}
        if isinstance(value, list):
            return [unescape(child) for child in value]
        return value

    return unescape(v2.decode_public({"definitions": {}, "public": resolve(encoded["public"])}))


def decode_text(text: str) -> Any:
    """Decode either selected v3 text or a shorter retained v2 representation."""
    if text.startswith("Event view compact v1:"):
        from . import public_event_view_compact
        if not text.startswith(public_event_view_compact.LEGEND):
            raise ValueError("Invalid compact event-view legend")
        return public_event_view_compact.decode_text(text)
    for legend in (LEGEND, FORECAST_CLEAN_LEGEND):
        if text.startswith(legend):
            return decode_encoded(_parse_encoded(text[len(legend):]))
    return v2.decode_text(text)


@dataclass(frozen=True)
class PublicEventView:
    version: str
    source: Optional[str]
    public: Any
    aliases: dict[str, Any]
    text: str
    encoded: dict[str, Any]
    selected_v3: bool
    baseline_units: int
    candidate_units: int
    rule_counts: dict[str, int]


def render_clean_text(text: str, *, source: Optional[str] = None,
                      aliases: Optional[dict[str, Any]] = None,
                      count_units: Optional[Callable[[str], int]] = None) -> PublicEventView:
    """Factor already clean v2 text, retaining it unless the candidate is shorter.

    Exporters pass the deployment tokenizer as ``count_units``; the standalone
    default is character length. The rule depends only on text, never source.
    """
    public = v2.decode_text(text)
    def check_reserved(value: Any) -> None:
        if isinstance(value, dict):
            if set(value) == {"native_relation_literal"}:
                raise ValueError("native literal conflicts with v3 relation operator")
            for child in value.values():
                check_reserved(child)
        elif isinstance(value, list):
            for child in value:
                check_reserved(child)
    try:
        check_reserved(public)
    except ValueError:
        count = count_units or len
        units = count(text)
        return PublicEventView(v2.VERSION, source, public, copy.deepcopy(aliases or {}),
            text, {}, False, units, units, {"native_operator_literal_fallback": 1})
    rule_counts = {}
    encoded = factor_encoded(v2.encode_public(public), count_units=count_units, rule_counts=rule_counts)
    candidate = LEGEND + "\n".join(_readable(encoded))
    if _dump(decode_text(candidate)) != _dump(public):
        raise ValueError("v3 readable text did not round trip exactly")
    count = count_units or len
    baseline_units, candidate_units = count(text), count(candidate)
    selected = candidate_units < baseline_units
    return PublicEventView(VERSION if selected else v2.VERSION, source, public,
                           copy.deepcopy(aliases or {}), candidate if selected else text,
                           encoded, selected, baseline_units, candidate_units, rule_counts)


def render_public(value: Any, *, source: Optional[str] = None,
                  count_units: Optional[Callable[[str], int]] = None) -> PublicEventView:
    """Create the unchanged v2 projection, then opt in to exact further factoring."""
    baseline = v2.render_public(value, source=source)
    projected, _ = v2.project_public(value, source=source)
    if _dump(v2.decode_text(baseline.text)) != _dump(projected):
        raise ValueError("v2 baseline cannot preserve this literal tree exactly")
    return render_clean_text(baseline.text, source=source, aliases=baseline.aliases,
                             count_units=count_units)
