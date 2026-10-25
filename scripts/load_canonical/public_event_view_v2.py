"""Deterministic, lossless public event views with private opaque-ID aliases.

The module deliberately operates on already loaded JSON-compatible canonical
objects.  It does not inspect or mutate source datasets.  ``project_public``
returns a JSON-compatible public tree and a sidecar containing every alias
mapping; ``render_public`` additionally creates a readable, deterministic text
form suitable for a model prompt.  IDs are the only values projected: semantic
codes (VTEC, UGC, timestamps, notices and instructions) are retained verbatim.
"""
from __future__ import annotations

import argparse
import copy
import json
import re
from collections import Counter
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

try:
    from .layout_codec import LEGEND as LOSSLESS_LAYOUT_LEGEND, decode as layout_decode, dumps as layout_dumps, encode as layout_encode, embedded_json
except ImportError:  # direct invocation uses the adjacent production module
    from layout_codec import LEGEND as LOSSLESS_LAYOUT_LEGEND, decode as layout_decode, dumps as layout_dumps, encode as layout_encode, embedded_json


VERSION = "public-event-view-v2"
EVENT_VIEW_LEGEND = ("Event view v2: native opaque identities use local E aliases for visible messages and X aliases for outside-window references; reference senders and sent times remain literal. "
    "All dictionary definitions are part of this input. same_as selects one complete exact definition. text_parts concatenate in order with no added separators. "
    "record_table rows follow the named columns; common fields apply to every row; absent:true means the original record omitted that field (different from null). "
    "time_axis expands start+i*step_seconds for count original ISO timestamps. No notice, body, instruction, numeric value or semantic code is omitted.\n")
# Accepted only when an opt-in forecast projection selects this prefix. The
# original renderer still emits EVENT_VIEW_LEGEND unchanged.
FORECAST_CLEAN_LEGEND = EVENT_VIEW_LEGEND.replace(
    "native opaque identities use local E aliases for visible messages and X aliases for outside-window references; reference senders and sent times remain literal.",
    "E aliases identify records present in this case; X aliases identify referenced records absent from it. CAP reference sent times remain literal; sender is retained when needed to distinguish namespaces.",
    1,
)
_UUID = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$", re.I)
_HEX = re.compile(r"^[0-9a-f]{32,}$", re.I)
_OPAQUE = re.compile(r"^[A-Za-z0-9_-]{24,}$")
_SEMANTIC = re.compile(r"(?:vtec|ugc|same|geocode|timezone|timestamp|time|date|code|area|zone|body|description|headline|instruction|title|text|url|path|file|source)", re.I)
_ID_KEY = re.compile(r"(?:^|[_\-.])(?:id|uid|uuid|hash|digest|message|record|episode|segment|version|locator)(?:$|[_\-.])", re.I)
_NATIVE_MESSAGE = re.compile(r"(?:urn:oid:2\.49\.0\.1\.840\.0\.[A-Za-z0-9.-]+|NWS-IDP-PROD-[A-Za-z0-9-]+)")
_IDENTITY_FIELDS = frozenset({"event_id", "event_record_uid", "native_event_id", "version", "referenced_source_message_ids", "native_references", "expiredReferences", "candidate_episode_ids", "revision_family"})


def _json_default(value: Any) -> Any:
    """Convert common scalar wrappers while refusing silent information loss."""
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "isoformat"):
        return value.isoformat()
    raise TypeError(f"not JSON compatible: {type(value).__name__}")


def _path_text(path: tuple[str, ...]) -> str:
    return "$" + "".join("[" + p + "]" if p.isdigit() else "." + p for p in path)


def _is_opaque_id(value: str, key: str | None) -> bool:
    if not key or key not in _IDENTITY_FIELDS or _SEMANTIC.search(key):
        return False
    if _UUID.fullmatch(value) or _HEX.fullmatch(value):
        return True
    return False


@dataclass
class AliasSidecar:
    """Private native-to-alias mapping and audit paths."""

    aliases: dict[str, str] = field(default_factory=dict)
    paths: dict[str, list[str]] = field(default_factory=dict)
    known_ids: set[str] = field(default_factory=set, repr=False)
    reserved_tokens: set[str] = field(default_factory=set, repr=False)

    def add(self, native: str, path: tuple[str, ...]) -> str:
        if native not in self.aliases:
            # E aliases identify in-window event records; X aliases identify
            # referenced native records outside this case's visible window.
            prefix = "E" if native in self.known_ids else "X"
            ordinal = sum(alias.startswith(prefix) for alias in self.aliases.values()) + 1
            alias = prefix + str(ordinal)
            while alias in self.reserved_tokens or alias in self.aliases.values():
                ordinal += 1
                alias = prefix + str(ordinal)
            self.aliases[native] = alias
        alias = self.aliases[native]
        self.paths.setdefault(native, []).append(_path_text(path))
        return alias

    def as_dict(self) -> dict[str, Any]:
        return {
            "version": VERSION,
            "mapping": dict(sorted(self.aliases.items())),
            "paths": {key: sorted(set(value)) for key, value in sorted(self.paths.items())},
        }


def project_public(value: Any, *, source: str | None = None) -> tuple[Any, dict[str, Any]]:
    """Return a deep public projection and a private alias sidecar.

    All mapping keys, list members, text, numbers and nulls survive exactly,
    except eligible opaque identity values.  ``source`` is metadata only and
    is included in the sidecar to make cross-source measurements auditable.
    """
    sidecar = AliasSidecar()

    def collect_ids(item: Any) -> None:
        if isinstance(item, Mapping):
            for key, child in item.items():
                if key in {"event_id", "event_record_uid", "native_event_id", "version"} and isinstance(child, str):
                    sidecar.known_ids.add(child)
                collect_ids(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                collect_ids(child)
        elif isinstance(item, str):
            sidecar.reserved_tokens.update(re.findall(r"(?<![A-Za-z0-9_])[EX][0-9]+(?![A-Za-z0-9_])", item))
    collect_ids(value)

    def identity(value: str, key: str | None, path: tuple[str, ...]) -> str:
        if key in _IDENTITY_FIELDS:
            return _NATIVE_MESSAGE.sub(lambda m: sidecar.add(m.group(0), path), value)
        if _is_opaque_id(value, key):
            return sidecar.add(value, path)
        return value

    def walk(item: Any, path: tuple[str, ...], key: str | None = None) -> Any:
        if isinstance(item, Mapping):
            return {str(k): walk(v, path + (str(k),), "expiredReferences" if str(k) == "value" and item.get("name") == "expiredReferences" else str(k)) for k, v in item.items()}
        if isinstance(item, (list, tuple)):
            return [walk(v, path + (str(i),), key) for i, v in enumerate(item)]
        if isinstance(item, str):
            if len(item) > 180:
                candidate = embedded_json(item)
                if candidate:
                    spec = candidate["$embedded_json"]
                    prepared = walk(spec["value"], path)
                    # Alias paths inside this lexical string refer to the string
                    # itself; all nested JSON serialization stays exact.
                    for native, paths in sidecar.paths.items():
                        prefix = _path_text(path)
                        nested = [p for p in paths if p.startswith(prefix + ".") or p.startswith(prefix + "[")]
                        if nested:
                            sidecar.paths[native] = [p for p in paths if p not in nested] + [prefix]
                    kwargs = {"ensure_ascii": spec["ascii"], "allow_nan": False}
                    if spec["compact"]:
                        kwargs["separators"] = (",", ":")
                    return spec["prefix"] + json.dumps(prepared, **kwargs) + spec["suffix"]
            return identity(item, key, path)
        if item is None or isinstance(item, (str, int, float, bool)):
            return item
        return _json_default(item)

    projected = walk(copy.deepcopy(value), ())
    result = sidecar.as_dict()
    if source is not None:
        result["source"] = source
    return projected, result


def _compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=_json_default)


def restore_aliases(public: Any, aliases: Mapping[str, Any]) -> Any:
    """Restore only recorded paths, avoiding substitutions inside ordinary text."""
    result = copy.deepcopy(public)
    for native, alias in aliases["mapping"].items():
        for path in aliases["paths"].get(native, []):
            parts = re.findall(r"\.([^.[\]]+)|\[(\d+)\]", path)
            steps = [int(index) if index else key for key, index in parts]
            target = result
            for step in steps[:-1]:
                target = target[step]
            if steps:
                target[steps[-1]] = re.sub(r"(?<![A-Za-z0-9_])" + re.escape(alias) + r"(?![A-Za-z0-9_])", lambda m: native, target[steps[-1]])
            else:
                result = re.sub(r"(?<![A-Za-z0-9_])" + re.escape(alias) + r"(?![A-Za-z0-9_])", lambda m: native, result)
    return result


def encode_public(public: Any) -> dict[str, Any]:
    """Factor exact paragraphs, repeated values, axes and homogeneous rows."""
    counts: Counter[str] = Counter()
    definitions: dict[str, Any] = {}
    names: dict[str, str] = {}

    def prepare(value: Any) -> Any:
        if isinstance(value, dict):
            return {k: prepare(v) for k, v in value.items()}
        if isinstance(value, list):
            return [prepare(v) for v in value]
        if isinstance(value, str) and len(value) > 180:
            embedded = embedded_json(value)
            if embedded:
                spec = embedded["$embedded_json"]
                return {"embedded_text": {"prefix": spec["prefix"], "suffix": spec["suffix"], "ascii": spec["ascii"], "compact": spec["compact"], "value": prepare(spec["value"])}}
        if isinstance(value, str) and len(value) > 160 and "\n\n" in value:
            # Separators stay in the fragments, preserving every byte and final
            # newline. Factoring only exact paragraphs preserves negation.
            parts = re.split(r"(?<=\n)\n", value)
            if len(parts) > 1:
                chunks = [p + "\n" if i < len(parts) - 1 else p for i, p in enumerate(parts)]
                if "".join(chunks) == value:
                    return {"text_parts": chunks}
        return value

    prepared = prepare(public)

    def eligible(v: Any) -> bool:
        return isinstance(v, (dict, list, str)) and len(_compact(v)) >= 48

    def flatten(record: Mapping[str, Any], path: tuple[str, ...] = ()) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, child in record.items():
            parts = path + (key,)
            if isinstance(child, dict) and child and not set(child).intersection({"same_as", "text_parts", "time_axis", "record_table", "embedded_text"}):
                result.update(flatten(child, parts))
            else:
                result["/".join(parts)] = child
        return result

    def count(v: Any) -> None:
        if eligible(v):
            counts[_compact(v)] += 1
        if isinstance(v, dict):
            for c in v.values():
                count(c)
        elif isinstance(v, list):
            for c in v:
                count(c)
    count(prepared)

    def pack(v: Any, skip: bool = False) -> Any:
        key = _compact(v) if eligible(v) else None
        if not skip and key is not None and counts[key] > 1:
            if key not in names:
                name = "T" + str(len(names) + 1)
                names[key] = name
                definitions[name] = pack(v, True)
            return {"same_as": names[key]}
        if isinstance(v, list) and len(v) >= 3 and all(isinstance(x, str) and "T" in x for x in v):
            try:
                start = datetime.fromisoformat(v[0])
                delta = (datetime.fromisoformat(v[1]) - start).total_seconds()
                if [(start + timedelta(seconds=i * delta)).isoformat() for i in range(len(v))] == v:
                    return {"time_axis": {"start": v[0], "step_seconds": delta, "count": len(v)}}
            except ValueError:
                pass
        if isinstance(v, list) and len(v) >= 2 and all(isinstance(x, dict) for x in v):
            flat = [flatten(x) for x in v]
            keys = sorted(set().union(*(x.keys() for x in flat)))
            common = [k for k in keys if all(k in x and type(x[k]) == type(flat[0][k]) and x[k] == flat[0][k] for x in flat)]
            columns = [k for k in keys if k not in common]
            # A membership mask distinguishes an absent key from explicit null.
            rows = [[pack(x[k]) if k in x else {"absent": True} for k in columns] for x in flat]
            return {"record_table": {"common": {k: pack(flat[0][k]) for k in common}, "columns": columns, "rows": rows, "flattened_paths": True}}
        if isinstance(v, dict):
            return {k: pack(c) for k, c in v.items()}
        if isinstance(v, list):
            return [pack(c) for c in v]
        return v

    body = pack(prepared)
    return {"version": VERSION, "definitions": definitions, "public": body}


def decode_public(encoded: Mapping[str, Any]) -> Any:
    definitions = encoded["definitions"]
    def unpack(v: Any) -> Any:
        if isinstance(v, dict):
            if set(v) == {"same_as"}:
                return unpack(definitions[v["same_as"]])
            if set(v) == {"text_parts"}:
                return "".join(unpack(v["text_parts"]))
            if set(v) == {"embedded_text"}:
                spec = v["embedded_text"]
                kwargs = {"ensure_ascii": spec["ascii"], "allow_nan": False}
                if spec["compact"]:
                    kwargs["separators"] = (",", ":")
                return spec["prefix"] + json.dumps(unpack(spec["value"]), **kwargs) + spec["suffix"]
            if set(v) == {"time_axis"}:
                a = v["time_axis"]
                start = datetime.fromisoformat(a["start"])
                return [(start + timedelta(seconds=i * a["step_seconds"])).isoformat() for i in range(a["count"])]
            if set(v) == {"record_table"}:
                spec = v["record_table"]
                common = unpack(spec["common"])
                records = [dict(copy.deepcopy(common), **{k: unpack(c) for k, c in zip(spec["columns"], row) if c != {"absent": True}}) for row in spec["rows"]]
                if not spec.get("flattened_paths"):
                    return records
                restored = []
                for record in records:
                    result = {}
                    for key, child in record.items():
                        target = result
                        parts = key.split("/")
                        for part in parts[:-1]:
                            target = target.setdefault(part, {})
                        target[parts[-1]] = child
                    restored.append(result)
                return restored
            return {k: unpack(c) for k, c in v.items()}
        if isinstance(v, list):
            return [unpack(c) for c in v]
        return v
    return unpack(encoded["public"])


def _readable(value: Any, indent: int = 0, label: str | None = None) -> list[str]:
    """Render JSON-compatible data with deterministic homogeneous-list factoring."""
    pad = " " * indent
    prefix = f"{label}: " if label is not None else ""
    if isinstance(value, Mapping):
        if not value:
            return [pad + prefix + "{}"]
        lines = [pad + prefix.rstrip() + "{"]
        for key in sorted(value):
            lines.extend(_readable(value[key], indent + 2, str(key)))
        lines.append(pad + "}")
        return lines
    if isinstance(value, list):
        if not value:
            return [pad + prefix + "[]"]
        if all(isinstance(x, Mapping) for x in value):
            keys = sorted(set().union(*(x.keys() for x in value)))
            common = [k for k in keys if all(k in x and x[k] == value[0][k] for x in value)]
            varying = [k for k in keys if k not in common]
            lines = [pad + prefix.rstrip() + f"[records={len(value)}"]
            if common:
                lines.append(pad + "  common: " + _compact({k: value[0][k] for k in common}))
            lines.append(pad + "  columns: " + ",".join(varying))
            for i, row in enumerate(value):
                lines.append(pad + f"  row[{i}]: " + _compact({k: row[k] for k in varying if k in row}))
            lines.append(pad + "]")
            return lines
        lines = [pad + prefix.rstrip() + "["]
        for i, item in enumerate(value):
            lines.extend(_readable(item, indent + 2, str(i)))
        lines.append(pad + "]")
        return lines
    return [pad + prefix + _compact(value)]


def _readable_codec(value: Any, indent: int = 0, label: str | None = None) -> list[str]:
    """Show named columns once; exact definitions may reference other definitions."""
    pad = " " * indent
    prefix = (label + ": ") if label else ""
    if isinstance(value, dict) and set(value) == {"same_as"}:
        return [pad + prefix + "same as " + value["same_as"]]
    if isinstance(value, dict) and set(value) == {"record_table"}:
        spec = value["record_table"]
        lines = [pad + prefix + "record table (rows=" + str(len(spec["rows"])) + ")"]
        if spec["common"]:
            lines.append(pad + "common: " + _compact(spec["common"]))
        lines.append(pad + "columns: " + _compact(spec["columns"]))
        for row in spec["rows"]:
            lines.append(pad + "row: " + _compact(row))
        lines.append(pad + "end table")
        return lines
    if isinstance(value, dict):
        lines = [pad + prefix + "{"]
        for key in value:
            lines.extend(_readable_codec(value[key], indent + 2, str(key)))
        lines.append(pad + "}")
        return lines
    return [pad + prefix + _compact(value)]


def decode_text(text: str) -> Any:
    """Parse the readable tables and dictionaries back to the public tree."""
    legend = next((prefix for prefix in (EVENT_VIEW_LEGEND, FORECAST_CLEAN_LEGEND) if text.startswith(prefix)), None)
    if legend is None:
        raise ValueError("unknown event view legend")
    lines = iter(text[len(legend):].splitlines())

    def parse_value(first: str) -> Any:
        content = first.strip()
        if content == "{":
            result = {}
            for line in lines:
                stripped = line.strip()
                if stripped == "}":
                    return result
                key, child = stripped.split(": ", 1)
                result[key] = parse_value(child)
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
                field, payload = stripped.split(": ", 1)
                if field == "row":
                    spec["rows"].append(json.loads(payload))
                else:
                    spec[field] = json.loads(payload)
            raise ValueError("unterminated record table")
        if content.startswith("same as "):
            return {"same_as": content[len("same as "):]}
        return json.loads(content)

    encoded = parse_value(next(lines))
    if next(lines, None) is not None:
        raise ValueError("trailing readable text")
    return decode_public(encoded)


@dataclass(frozen=True)
class PublicEventView:
    version: str
    source: str | None
    public: Any
    aliases: dict[str, Any]
    text: str
    encoded: dict[str, Any]
    readable_text: str

    def as_dict(self) -> dict[str, Any]:
        return {"version": self.version, "source": self.source, "public": self.public, "aliases": self.aliases, "text": self.text, "encoded": self.encoded, "readable_text": self.readable_text}


def render_public(value: Any, *, source: str | None = None) -> PublicEventView:
    public, aliases = project_public(value, source=source)
    encoded = encode_public(public)
    if decode_public(encoded) != public:
        raise ValueError("public view did not round trip exactly")
    packed, _ = layout_encode(encoded)
    if layout_decode(packed) != encoded:
        raise ValueError("layout codec did not round trip exactly")
    readable = EVENT_VIEW_LEGEND + "\n".join(_readable_codec(encoded))
    if decode_text(readable) != public:
        raise ValueError("readable text did not round trip exactly")
    return PublicEventView(VERSION, source, public, aliases, readable, {"factored": encoded, "layout": packed}, readable)


def render_jsonl(input_path: str | Path, output_path: str | Path, sidecar_path: str | Path | None = None) -> dict[str, Any]:
    """Render JSONL rows, writing one view per line and merged alias metadata."""
    aliases: dict[str, str] = {}
    paths: dict[str, list[str]] = {}
    count = 0
    with Path(input_path).open(encoding="utf-8") as src, Path(output_path).open("w", encoding="utf-8") as dst:
        for line_no, line in enumerate(src, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            payload = row.get("model_input", row) if isinstance(row, Mapping) else row
            source = row.get("source") if isinstance(row, Mapping) else None
            view = render_public(payload, source=source)
            for native, alias in view.aliases["mapping"].items():
                aliases[native] = alias
                paths.setdefault(native, []).extend(view.aliases["paths"].get(native, []))
            artifact = view.as_dict()
            if isinstance(row, Mapping):
                artifact["case_id"] = row.get("case_id", row.get("sample_id"))
            dst.write(json.dumps(artifact, ensure_ascii=False, sort_keys=True, default=_json_default) + "\n")
            count += 1
    merged = {"version": VERSION, "records": count, "mapping": dict(sorted(aliases.items())), "paths": {k: sorted(set(v)) for k, v in sorted(paths.items())}}
    if sidecar_path:
        Path(sidecar_path).write_text(json.dumps(merged, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return merged


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="input JSONL")
    parser.add_argument("output", help="output JSONL")
    parser.add_argument("--sidecar", help="merged alias JSON path")
    args = parser.parse_args(argv)
    summary = render_jsonl(args.input, args.output, args.sidecar)
    print(json.dumps({"records": summary["records"], "aliases": len(summary["mapping"])}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
