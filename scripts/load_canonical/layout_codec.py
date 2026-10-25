"""Offline reversible public-input layout experiment; no live renderer imports."""
import copy
import json
from collections import Counter
from datetime import datetime, timedelta


def dumps(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def embedded_json(text):
    """Only parse a serialization whose exact lexical form is reproducible."""
    decoder = json.JSONDecoder()
    for start, char in enumerate(text):
        if char not in "{[":
            continue
        try:
            value, end = decoder.raw_decode(text[start:])
        except ValueError:
            continue
        if not isinstance(value, (dict, list)):
            continue
        raw = text[start:start + end]
        for ascii_only in (False, True):
            for compact in (False, True):
                kwargs = {"ensure_ascii": ascii_only, "allow_nan": False}
                if compact:
                    kwargs["separators"] = (",", ":")
                try:
                    exact = json.dumps(value, **kwargs)
                except ValueError:
                    continue
                if raw == exact:
                    return {"$embedded_json": {"prefix": text[:start], "suffix": text[start + end:],
                            "ascii": ascii_only, "compact": compact, "value": value}}
        # Avoid parsing a nested fragment while retaining the enclosing JSON text.
        return None
    return None


def encode(model, *, parse_strings=True, regular_axes=True):
    counts = Counter()
    stats = Counter()

    def prepare(value):
        if isinstance(value, dict):
            if any(k.startswith("$") for k in value):
                raise ValueError("Reserved codec key collision")
            return {k: prepare(v) for k, v in value.items()}
        if isinstance(value, list):
            return [prepare(v) for v in value]
        if parse_strings and isinstance(value, str) and len(value) > 180:
            candidate = embedded_json(value)
            if candidate:
                stats["embedded_json_strings"] += 1
                candidate["$embedded_json"]["value"] = prepare(candidate["$embedded_json"]["value"])
                return candidate
        return value

    prepared = prepare(model)

    def eligible(value):
        return isinstance(value, (dict, list, str)) and len(dumps(value)) > 180

    def count(value):
        if eligible(value):
            counts[dumps(value)] += 1
        if isinstance(value, dict):
            for child in value.values():
                count(child)
        elif isinstance(value, list):
            for child in value:
                count(child)

    count(prepared)
    shared, names = {}, {}

    def pack(value, skip_ref=False):
        key = dumps(value) if eligible(value) else None
        if not skip_ref and key is not None and counts[key] > 1:
            if key not in names:
                name = "shared_" + str(len(names))
                names[key] = name
                shared[name] = pack(value, skip_ref=True)
                stats["shared_definitions"] += 1
            stats["references"] += 1
            return {"$ref": names[key]}
        if isinstance(value, list) and len(value) >= 3:
            if regular_axes and all(isinstance(v, str) and "T" in v for v in value):
                try:
                    start = datetime.fromisoformat(value[0])
                    step = (datetime.fromisoformat(value[1]) - start).total_seconds()
                    restored = [(start + timedelta(seconds=i * step)).isoformat() for i in range(len(value))]
                    if restored == value:
                        stats["regular_time_axes"] += 1
                        return {"$time_axis": {"start": value[0], "step_seconds": step, "count": len(value)}}
                except ValueError:
                    pass
            if all(isinstance(v, (str, bool)) or v is None for v in value):
                runs = []
                for item in value:
                    if runs and item == runs[-1][0] and type(item) is type(runs[-1][0]):
                        runs[-1][1] += 1
                    else:
                        runs.append([item, 1])
                candidate = {"$rle": runs}
                if len(dumps(candidate)) < len(dumps(value)):
                    stats["rle_arrays"] += 1
                    return candidate
        if isinstance(value, dict):
            return {k: pack(v) for k, v in value.items()}
        if isinstance(value, list):
            return [pack(v) for v in value]
        return value

    body = pack(prepared)
    return {"layout": "lossless_public_v1", "shared": shared, "public_model": body}, dict(stats)


def decode(encoded):
    shared = encoded["shared"]

    def unpack(value):
        if isinstance(value, dict):
            if "$ref" in value:
                return unpack(shared[value["$ref"]])
            if "$time_axis" in value:
                spec = value["$time_axis"]
                start = datetime.fromisoformat(spec["start"])
                return [(start + timedelta(seconds=i * spec["step_seconds"])).isoformat()
                        for i in range(spec["count"])]
            if "$rle" in value:
                return [copy.deepcopy(item) for item, count in value["$rle"] for _ in range(count)]
            if "$embedded_json" in value:
                spec = value["$embedded_json"]
                kwargs = {"ensure_ascii": spec["ascii"], "allow_nan": False}
                if spec["compact"]:
                    kwargs["separators"] = (",", ":")
                return spec["prefix"] + json.dumps(unpack(spec["value"]), **kwargs) + spec["suffix"]
            return {k: unpack(v) for k, v in value.items()}
        if isinstance(value, list):
            return [unpack(v) for v in value]
        return value

    return unpack(encoded["public_model"])


LEGEND = """Lossless layout: shared entries are reused by $ref. $time_axis expands exactly as start + i*step_seconds, i=0..count-1, ISO timestamps. $rle contains [value,count] runs in original order. $embedded_json stores parsed string content with original prefix/suffix and JSON serialization settings. Read all referenced content as part of the public model; availability, units and event timing retain their original meaning.\n"""
