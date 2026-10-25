"""Exact display compaction of v3 inputs; old renderers remain frozen.

Only codec reference syntax and single-use definition indirection change.
Native JSON strings are never searched/replaced. Every candidate must decode
to the identical public tree and be shorter under the supplied tokenizer.
"""
from __future__ import annotations

import copy
import json
from collections import Counter
from typing import Any, Callable

from . import public_event_view_v3 as v3

VERSION = "forecast-clean-compact-v1"
LEGEND = v3.FORECAST_CLEAN_LEGEND.replace("Event view v3:", "Event view compact v1:", 1) + (
    'Compact row notation: @"ID" means {"same_as":"ID"}; '
    '^"field" means {"same_column":"field"}. '
    'These markers are operators only outside quoted strings. '
    'Definitions and local field names stay visible; quoted native text is literal.\n'
)
_MARKERS = {"@": "same_as", "^": "same_column"}


def _dump(value: Any) -> str:
    if isinstance(value, dict):
        for marker, key in _MARKERS.items():
            if set(value) == {key} and isinstance(value[key], str):
                return marker + json.dumps(value[key], ensure_ascii=False)
        return "{" + ",".join(json.dumps(k, ensure_ascii=False) + ":" + _dump(v)
                              for k, v in sorted(value.items())) + "}"
    if isinstance(value, list):
        return "[" + ",".join(_dump(v) for v in value) + "]"
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def _expand(line: str) -> str:
    """Lex quoted strings atomically, so native marker-looking prose is inert."""
    decoder = json.JSONDecoder()
    parts = []
    index = 0
    # The readable table/object grammar puts an unquoted label before ': '.
    # Leave that label untouched; marker operators exist only in its value.
    stripped = line.lstrip()
    if not stripped.startswith(('"', '[', '{', '@', '^')) and ': ' in line:
        label, line = line.split(': ', 1)
        return label + ': ' + _expand(line)
    while index < len(line):
        char = line[index]
        if char == '"':
            _, end = decoder.raw_decode(line, index)
            parts.append(line[index:end])
            index = end
        elif char in _MARKERS and index + 1 < len(line) and line[index + 1] == '"':
            value, end = decoder.raw_decode(line, index + 1)
            parts.append(json.dumps({_MARKERS[char]: value}, ensure_ascii=False, separators=(",", ":")))
            index = end
        else:
            parts.append(char)
            index += 1
    return "".join(parts)


def _readable(value: Any, indent: int = 0, label: str | None = None) -> list[str]:
    pad = " " * indent
    prefix = label + ": " if label else ""
    if isinstance(value, dict) and set(value) in ({"record_table"}, {"table_rows"}):
        is_schema = "table_rows" in value
        spec = value["table_rows" if is_schema else "record_table"]
        header = "schema table" if is_schema else "record table"
        lines = [pad + prefix + header + " (rows=" + str(len(spec["rows"])) + ")"]
        fields = ("schema",) if is_schema else ("common", "columns")
        lines += [pad + key + ": " + _dump(spec[key]) for key in fields]
        lines += [pad + "row: " + _dump(row) for row in spec["rows"]]
        return lines + [pad + "end table"]
    if isinstance(value, dict) and set(value) == {"same_as"}:
        return [pad + prefix + "same as " + value["same_as"]]
    if isinstance(value, dict):
        lines = [pad + prefix + "{"]
        for key, child in value.items():
            lines.extend(_readable(child, indent + 2, str(key)))
        return lines + [pad + "}"]
    return [pad + prefix + _dump(value)]


def _refs(value: Any) -> Counter:
    result = Counter()
    if isinstance(value, dict):
        if set(value) == {"same_as"}:
            result[value["same_as"]] += 1
        for child in value.values():
            result.update(_refs(child))
    elif isinstance(value, list):
        for child in value:
            result.update(_refs(child))
    return result


def _inline_once(encoded: dict) -> dict:
    result = copy.deepcopy(encoded)
    # Existing inputs have acyclic definitions. Each iteration removes at
    # least one definition; no new references or clock abbreviations appear.
    while True:
        uses = _refs({key: child for key, child in result.items() if key != "definitions"})
        for child in result["definitions"].values():
            uses.update(_refs(child))
        singles = {key for key in result["definitions"] if uses[key] == 1}
        if not singles:
            return result
        definitions = result["definitions"]

        def replace(value: Any, active: tuple = ()) -> Any:
            if isinstance(value, dict):
                if set(value) == {"same_as"} and value["same_as"] in singles:
                    name = value["same_as"]
                    if name in active:
                        raise ValueError("cyclic single-use definition")
                    return replace(definitions[name], active + (name,))
                return {key: replace(child, active) for key, child in value.items()}
            if isinstance(value, list):
                return [replace(child, active) for child in value]
            return value

        result = {key: replace(child) for key, child in result.items() if key != "definitions"}
        result["definitions"] = {key: replace(child) for key, child in definitions.items() if key not in singles}
        result = {key: result[key] for key in ("version", "definitions", "schemas", "public") if key in result}


def decode_text(text: str) -> Any:
    if not text.startswith(LEGEND):
        return v3.decode_text(text)
    expanded = "\n".join(_expand(line) for line in text[len(LEGEND):].splitlines())
    return v3.decode_encoded(v3._parse_encoded(expanded))


def _same(left: Any, right: Any) -> bool:
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(_same(left[key], right[key]) for key in left)
    if isinstance(left, list):
        return len(left) == len(right) and all(_same(a, b) for a, b in zip(left, right))
    return left == right


def render_text(text: str, *, count_units: Callable[[str], int]) -> str:
    """Use compact syntax only when it reduces an already-v3 input."""
    legend = next((prefix for prefix in (v3.FORECAST_CLEAN_LEGEND, v3.LEGEND)
                   if text.startswith(prefix)), None)
    if legend is None:
        return text
    encoded = v3._parse_encoded(text[len(legend):])
    public = v3.decode_encoded(encoded)
    candidates = [text]
    for value in (encoded, _inline_once(encoded)):
        # Braces, labels, table headers and one-event-per-row lines remain.
        # Only layout indentation is omitted, never whitespace in literals.
        candidate = LEGEND + "\n".join(line.lstrip() for line in _readable(value))
        if not _same(decode_text(candidate), public):
            raise ValueError("Compact display changed public values or types")
        candidates.append(candidate)
    return min(candidates, key=count_units)
