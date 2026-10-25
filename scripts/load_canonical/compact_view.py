"""Deterministic compact presentation of the existing public forecast case.

This changes presentation only. Primary numeric tables may shorten binary64
representation tails under a bounded display policy; source objects, citation
identities, availability and clock qualifications remain unchanged.
"""
from __future__ import annotations

import copy
import io
from datetime import datetime, timedelta
from decimal import Decimal
import json
import math
from pathlib import Path
import re
import stat
import struct
import sys
import threading
from typing import Any

INPUT_VERSION = "compact-load-input-v1"
EXPLICIT_TIME_INPUT_VERSION = "compact-load-input-v2-explicit-time-column"
NUMERIC_DISPLAY_POLICY = {
    "name": "binary64-tail-column-display-v2",
    "scope": "primary_numeric_table_cells_only",
    "max_binary64_steps": 2,
    "tail_candidate_max_fractional_digits": 6,
    "extended_tail_min_fractional_digits": 7,
    "extended_tail_min_discarded_digits": 7,
    "extended_tail_min_absolute_magnitude": 1e-6,
    "extended_tail_discarded_suffix_regex": r"(?:0{5,}[0-9]{1,2}|9{5,}[0-9]{1,2})",
    "original_fractional_digits_must_exceed": 6,
    "finite_normal_abs_range": [sys.float_info.min, 2 ** 53],
    "upper_bound_exclusive": True,
    "canonical_values_changed": False,
    "column_precision": "minimum_common_fractional_places_preserving_cleaned_values",
}
SELECTED_REVIEWED_TEXT = Path(__file__).resolve().parents[2] / "datasets/load_datasets/compact_v1/reviewed_text.jsonl"
SELECTED_PUBLIC_DIRECTORY = Path(__file__).resolve().parents[2] / "datasets/load_datasets/json_rows_v1"
_SELECTED_ID_CACHE: tuple[Path, set[str]] | None = None
_SELECTED_ASSET_CACHE: tuple[Path, Path] | None = None
_REVIEW_CACHE: tuple[tuple[int, int] | None, dict[str, list[dict]]] | None = None
_V2_PUBLIC_DIRECTORY = Path(__file__).resolve().parents[2] / "datasets/load_datasets/inputs_v2/json_rows_v2"
_V2_REVIEWED_TEXT = Path(__file__).resolve().parents[2] / "datasets/load_datasets/inputs_v2/reviewed_text.jsonl"
_V2_ASSET_CACHE: tuple[Any, dict, dict] | None = None
_V2_ASSET_LOCK = threading.RLock()
_STAMP = re.compile(r"^(\d{4}-\d{2}-\d{2})([T ])(\d{2}:\d{2})(:\d{2})?(\.\d{1,6})?(Z|[+-]\d{2}:\d{2})?$")
_STATIONS = ", ".join("w" + str(i) for i in range(1, 26))
_GEF_GENERIC_GUIDANCE = (
    "- Avoid exactly copying the last 24/96 observed target values or an entire historical day. "
    "If an analog shape is reasonable, adjust it using exogenous evidence and other context, "
    "including future level, peaks, troughs, and timing.\n"
    "- Consider calendar and time effects when forecasting, such as holidays, special dates, "
    "weekday/weekend changes, and seasonal patterns.\n"
    "- Use exogenous covariates carefully. When the observed window shows weak, nonlinear, "
    "saturated, or regime-dependent response, do not blindly follow a single covariate."
)


def _literal(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def binary64_steps(left: float, right: float) -> int:
    """Count representable steps between finite, same-sign binary64 values."""
    if (not math.isfinite(left) or not math.isfinite(right)
            or math.copysign(1.0, left) != math.copysign(1.0, right)):
        raise ValueError("Binary64 step comparison requires finite same-sign values")
    left_bits = struct.unpack(">Q", struct.pack(">d", left))[0]
    right_bits = struct.unpack(">Q", struct.pack(">d", right))[0]
    return abs(left_bits - right_bits)


def compact_numeric_literal(value: Any) -> str:
    """Shorten only long binary64 tails in primary numerical evidence cells.

    Try zero through six fractional places, accepting the shortest precision
    only within two representable-value steps. Beyond six places, require the
    policy's redundant zero/nine suffix and magnitude/discarded-digit bounds.
    Meaningful longer precision without that suffix, zero, subnormal/extreme
    values, integers and nulls keep their JSON literal.
    Float type is retained (e.g. an integer-valued float displays as ``1.0``).
    This is a display tolerance, not exact canonical-value equivalence.
    """
    original = _literal(value)
    if type(value) is not float or not sys.float_info.min <= abs(value) < 2 ** 53:
        return original
    if -Decimal(original).as_tuple().exponent <= 6:
        return original
    for places in range(NUMERIC_DISPLAY_POLICY["tail_candidate_max_fractional_digits"] + 1):
        candidate = float(format(value, f".{places}f"))
        if (candidate != 0.0 and binary64_steps(value, candidate) <= NUMERIC_DISPLAY_POLICY["max_binary64_steps"]):
            return _literal(candidate)
    if abs(value) >= NUMERIC_DISPLAY_POLICY["extended_tail_min_absolute_magnitude"]:
        fixed = format(Decimal(original), "f")
        fractional = fixed.split(".", 1)[1] if "." in fixed else ""
        for places in range(NUMERIC_DISPLAY_POLICY["extended_tail_min_fractional_digits"],
                            len(fractional) - NUMERIC_DISPLAY_POLICY["extended_tail_min_discarded_digits"] + 1):
            if not re.fullmatch(NUMERIC_DISPLAY_POLICY["extended_tail_discarded_suffix_regex"], fractional[places:]):
                continue
            candidate = float(format(value, f".{places}f"))
            if candidate != 0.0 and binary64_steps(value, candidate) <= NUMERIC_DISPLAY_POLICY["max_binary64_steps"]:
                return _literal(candidate)
    return original


def _exact(left: Any, right: Any) -> bool:
    """Compare public values including scalar types, order and nulls."""
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(_exact(left[k], right[k]) for k in left)
    if isinstance(left, list):
        return len(left) == len(right) and all(_exact(a, b) for a, b in zip(left, right))
    return left == right


def load_reviewed_text(path: str | Path) -> dict[str, dict]:
    """Load reviewed case-bound natural notes without hashing source identity."""
    result = {}
    with io.StringIO(Path(path).read_text(encoding="utf-8")) as handle:
        for line in handle:
            entry = json.loads(line)
            cid = entry.get("case_id")
            if not isinstance(cid, str) or not cid or cid in result:
                raise ValueError("Reviewed text has missing/duplicate case identity")
            if not isinstance(entry.get("public_case"), dict):
                raise ValueError("Reviewed text needs the exact complete public case")
            result[cid] = entry
    return result


def clear_reviewed_text_cache() -> None:
    """Clear selected public-note cache after a controlled sidecar publication."""
    global _REVIEW_CACHE, _SELECTED_ID_CACHE, _SELECTED_ASSET_CACHE, _V2_ASSET_CACHE
    with _V2_ASSET_LOCK:
        _REVIEW_CACHE = None
        _SELECTED_ID_CACHE = None
        _SELECTED_ASSET_CACHE = None
        _V2_ASSET_CACHE = None


def configure_compact_assets(input_version: str, public_directory: str | Path,
                             reviewed_path: str | Path) -> None:
    """Explicitly bind staged/published revision assets for this process."""
    global SELECTED_PUBLIC_DIRECTORY, SELECTED_REVIEWED_TEXT, _V2_PUBLIC_DIRECTORY, _V2_REVIEWED_TEXT
    with _V2_ASSET_LOCK:
        if input_version == INPUT_VERSION:
            SELECTED_PUBLIC_DIRECTORY, SELECTED_REVIEWED_TEXT = Path(public_directory), Path(reviewed_path)
        elif input_version == EXPLICIT_TIME_INPUT_VERSION:
            _V2_PUBLIC_DIRECTORY, _V2_REVIEWED_TEXT = Path(public_directory), Path(reviewed_path)
        else:
            raise ValueError("Unsupported compact input revision")
        clear_reviewed_text_cache()


def _v2_selected_assets() -> tuple[dict, dict]:
    # Cold workflow threads must not each parse/validate the entire corpus.
    # Recheck identity under the same lock as construction and publication.
    with _V2_ASSET_LOCK:
        return _load_v2_selected_assets_locked()


def _load_v2_selected_assets_locked() -> tuple[dict, dict]:
    global _V2_ASSET_CACHE
    paths = [_V2_PUBLIC_DIRECTORY / (split + ".jsonl") for split in ("train", "validation", "id_test", "ood")]
    paths.append(_V2_REVIEWED_TEXT)
    signature = []
    for path in paths:
        try:
            status = path.stat()
        except FileNotFoundError as error:
            raise ValueError("Compact v2 requires its own public identity inventory and reviewed-text sidecar") from error
        if not stat.S_ISREG(status.st_mode):
            raise ValueError("Compact v2 requires its own public identity inventory and reviewed-text sidecar")
        signature.append((str(path), status.st_mtime_ns, status.st_size))
    signature = tuple(signature)
    if _V2_ASSET_CACHE is None or _V2_ASSET_CACHE[0] != signature:
        public = {}
        for path in paths[:-1]:
            with io.StringIO(path.read_text(encoding="utf-8")) as handle:
                for line in handle:
                    row = json.loads(line)
                    if row["case_id"] in public:
                        raise ValueError("Duplicate v2 public identity")
                    public[row["case_id"]] = row
        reviews = load_reviewed_text(_V2_REVIEWED_TEXT)
        if set(public) != set(reviews):
            raise ValueError("Compact v2 public/reviewed identities disagree")
        for cid, row in public.items():
            if not _exact(row["case"], reviews[cid]["public_case"]):
                raise ValueError("Compact v2 lacks exact reviewed public binding")
        _V2_ASSET_CACHE = (signature, public, reviews)
    return _V2_ASSET_CACHE[1], _V2_ASSET_CACHE[2]


def validate_compact_profile_assets(input_version: str = INPUT_VERSION) -> None:
    """Check required assets once before an identity-aware selected lookup."""
    global _SELECTED_ASSET_CACHE
    if input_version == EXPLICIT_TIME_INPUT_VERSION:
        _v2_selected_assets()
        return
    if input_version != INPUT_VERSION:
        raise ValueError("Unsupported compact input revision")
    installation = (SELECTED_PUBLIC_DIRECTORY, SELECTED_REVIEWED_TEXT)
    if _SELECTED_ASSET_CACHE == installation:
        return
    for split in ("train", "validation", "id_test", "ood"):
        path = SELECTED_PUBLIC_DIRECTORY / (split + ".jsonl")
        if not path.is_file():
            raise ValueError("Compact selected lookup requires its public identity inventory: " + str(path))
    if not SELECTED_REVIEWED_TEXT.is_file():
        raise ValueError("Compact selected lookup requires its published reviewed-text sidecar: "
                         + str(SELECTED_REVIEWED_TEXT))
    _SELECTED_ASSET_CACHE = installation


def is_selected_public_case(case_id: str | None) -> bool:
    """Recognize selected identities without retaining their full public cases."""
    global _SELECTED_ID_CACHE
    if case_id is None:
        return False
    if _SELECTED_ID_CACHE is None or _SELECTED_ID_CACHE[0] != SELECTED_PUBLIC_DIRECTORY:
        identities = set()
        for split in ("train", "validation", "id_test", "ood"):
            path = SELECTED_PUBLIC_DIRECTORY / (split + ".jsonl")
            if path.exists():
                with path.open(encoding="utf-8") as handle:
                    identities.update(json.loads(line)["case_id"] for line in handle)
        _SELECTED_ID_CACHE = (SELECTED_PUBLIC_DIRECTORY, identities)
    return case_id in _SELECTED_ID_CACHE[1]


def _selected_review(public_case: dict, *, case_id: str | None = None, required: bool = False) -> dict | None:
    global _REVIEW_CACHE
    if not SELECTED_REVIEWED_TEXT.exists():
        if required:
            raise ValueError("Selected compact case requires its published reviewed-text sidecar")
        return None
    stat = SELECTED_REVIEWED_TEXT.stat()
    fingerprint = (stat.st_mtime_ns, stat.st_size)
    if _REVIEW_CACHE is None or _REVIEW_CACHE[0] != fingerprint:
        entries = load_reviewed_text(SELECTED_REVIEWED_TEXT)
        grouped: dict[str, list[dict]] = {}
        for entry in entries.values():
            public = entry["public_case"]
            task = public["task"]
            # Coarse routing reduces comparisons, but is never the final binding:
            # different synthetic arms share these task/clock coordinates.
            route = _literal([task.get("dataset_id"), task.get("series_id"),
                              public["grid"]["future_timestamps"]])
            grouped.setdefault(route, []).append(entry)
        _REVIEW_CACHE = (fingerprint, grouped)
    task = public_case["task"]
    route = _literal([task.get("dataset_id"), task.get("series_id"), public_case["grid"]["future_timestamps"]])
    candidates = _REVIEW_CACHE[1].get(route, [])
    if case_id is not None:
        candidates = [entry for entry in candidates if entry["case_id"] == case_id]
    matches = [entry for entry in candidates if _exact(entry["public_case"], public_case)]
    if required and not matches:
        raise ValueError("Selected compact case lacks an exact reviewed-text binding")
    if len(matches) > 1:
        first = matches[0]
        if any(not _exact(first.get("notes"), e.get("notes")) or
               not _exact(first.get("replaced_fields"), e.get("replaced_fields")) for e in matches[1:]):
            raise ValueError("Same public case has conflicting reviewed natural text")
    return matches[0] if matches else None


def _apply_reviewed_text(public_case: dict, entry: dict) -> tuple[dict, list[str]]:
    if not _exact(entry.get("public_case"), public_case):
        raise ValueError("Reviewed notes do not match the complete public case")
    notes = entry.get("notes")
    replacements = entry.get("replaced_fields")
    if (not isinstance(notes, list) or not notes or any(not isinstance(s, str) or not s.strip() for s in notes)
            or not isinstance(replacements, list)):
        raise ValueError("Reviewed text requires nonempty natural notes and replacement list")
    result = copy.deepcopy(public_case)
    seen = set()
    protected = {"values", "timestamps", "rows", "columns", "text", "body", "raw_text", "full_native_text",
                 "scenario_text", "static_attribute_text", "synthetic_programme_rule",
                 "retrospective_oracle_events", "inferred_day_ahead_tariff_schedule"}
    def generated_location(value: Any) -> bool:
        # This generated wrapper contains descriptive categorical attributes,
        # not a native bulletin or a categorical time series.
        return (isinstance(value, dict) and set(value) == {"parts"}
                and isinstance(value["parts"], list) and len(value["parts"]) == 2
                and value["parts"][0] == "iFlex household location: "
                and isinstance(value["parts"][1], dict)
                and set(value["parts"][1]) == {"Municipality", "Region"}
                and all(isinstance(v, str) for v in value["parts"][1].values()))
    for replacement in replacements:
        if not isinstance(replacement, dict) or set(replacement) != {"path", "value"}:
            raise ValueError("Malformed reviewed replacement")
        path = replacement["path"]
        if not isinstance(path, list) or any(not isinstance(k, str) for k in path):
            raise ValueError("Reviewed replacement targets protected evidence or invalid path")
        location_metadata = (path == ["context", "source_context", "context", "static_attribute_text"]
                             and generated_location(replacement["value"]))
        identity_duplicate = (len(path) >= 2 and path[-1] in {"entity_id", "parent_entity_id", "physical_system_id", "field_id"}
                              and path[0] == "context" and "lineage" in path)
        if (len(path) < 2 or path[0] not in {"task", "variables", "context"} or tuple(path) in seen
                or any((k in protected and not location_metadata)
                       or (k.endswith(("_id", "_ids")) and not identity_duplicate) for k in path)):
            raise ValueError("Reviewed replacement targets protected evidence or invalid path")
        seen.add(tuple(path))
        parent = result
        for key in path[:-1]:
            if not isinstance(parent, dict) or key not in parent:
                raise ValueError("Reviewed replacement path not found")
            parent = parent[key]
        key = path[-1]
        if not isinstance(parent, dict) or key not in parent or not _exact(parent[key], replacement["value"]):
            raise ValueError("Reviewed replacement original value/type mismatch")
        duplicate_history_cutoff = (path == ["task", "cutoff_time"]
                                    and _exact(parent[key], public_case["grid"]["history_timestamps"][-1]))
        if parent[key] is not None and not duplicate_history_cutoff and (key.endswith(
                ("_time", "_utc", "_at", "_date", "_start", "_end", "_timestamp"))
                or key in {"forecast_origin", "effective_from", "effective_to"}
                or key.startswith("history_last_interval_")):
            raise ValueError("Reviewed text cannot erase a supplied source clock")
        def descriptive_object(value):
            return isinstance(value, dict) and all(
                isinstance(k, str) and k not in protected and not k.endswith(("_id", "_ids"))
                and not k.endswith(("_time", "_utc", "_at", "_date", "_start", "_end", "_timestamp"))
                and k not in {"forecast_origin", "effective_from", "effective_to"}
                and not k.startswith("history_last_interval_")
                and not any(token in k for token in ("time", "clock", "date", "timestamp"))
                and (descriptive_object(v) if isinstance(v, dict) else not isinstance(v, list))
                for k, v in value.items())
        field_metadata = (path == ["context", "source_context", "metadata", "lineage", "exogenous_unit_evidence"]
                          and isinstance(parent[key], dict) and all(
                              name in public_case["history"]["columns"] and isinstance(value, dict)
                              and set(value) == {"field_id", "aggregation"}
                              and _exact(value["field_id"], name) and isinstance(value["aggregation"], str)
                              for name, value in parent[key].items()))
        if isinstance(parent[key], (dict, list)) and not (
                key == "observed_mask" and isinstance(parent[key], list)
                and all(type(flag) is bool for flag in parent[key])) and not (
                    descriptive_object(parent[key]) or parent[key] == []
                    or location_metadata or field_metadata):
            raise ValueError("Reviewed replacement may replace metadata but cannot erase IDs/clocks/arrays")
        if key == "observed_mask":
            if "values" in parent:
                values = parent["values"]
            elif path[0] == "variables" and len(path) == 4:
                scope = path[2]
                table = public_case[scope]
                name = path[1]
                if name not in table["columns"]:
                    raise ValueError("Reviewed mask has no aligned channel")
                values = [row[table["columns"].index(name)] for row in table["rows"]]
            else:
                raise ValueError("Reviewed mask needs explicit checked aligned values")
            if len(values) != len(parent[key]) or any(value is not None and flag is False
                                                     for value, flag in zip(values, parent[key])):
                raise ValueError("Reviewed text cannot erase non-null unobserved-cell evidence")
        parent.pop(key)
    return result, notes


def regular_step_seconds(timestamps: list[str]) -> float | None:
    """Return a step only if anchors reproduce every timestamp byte exactly.

    Different offset/precision spellings, irregular clocks and submicrosecond
    stamps fall back to an explicit mapping rather than silently losing them.
    """
    if not timestamps:
        return None
    match = _STAMP.fullmatch(timestamps[0])
    if not match:
        return None
    try:
        first = datetime.fromisoformat(timestamps[0].replace("Z", "+00:00"))
        second = (datetime.fromisoformat(timestamps[1].replace("Z", "+00:00"))
                  if len(timestamps) > 1 else first)
        delta = second - first
        if len(timestamps) > 1 and delta <= timedelta(0):
            return None
        for i, original in enumerate(timestamps):
            value = first + i * delta
            reconstructed = value.strftime("%Y-%m-%d") + match[2] + value.strftime("%H:%M")
            if match[4]:
                reconstructed += value.strftime(":%S")
            if match[5]:
                reconstructed += "." + f"{value.microsecond:06d}"[:len(match[5]) - 1]
            reconstructed += match[6] or ""
            if reconstructed != original:
                return None
    except (TypeError, ValueError, OverflowError):
        return None
    return delta.total_seconds() if len(timestamps) > 1 else None


def _clock_lines(label: str, timestamps: list[str], start: int, coordinate: str = "position") -> list[str]:
    step = regular_step_seconds(timestamps)
    if step is not None:
        amount = _literal(int(step) if step.is_integer() else step)
        return [f"{label}: {len(timestamps)} points; {coordinate} {start}={timestamps[0]}; "
                f"step={amount} seconds; {coordinate} {start + len(timestamps) - 1}={timestamps[-1]}."]
    return [f"{label}: explicit timestamp mapping (no uniform clock inferred):",
            coordinate + "|timestamp", "---|---"] + [
                str(start + i) + "|" + _literal(t) for i, t in enumerate(timestamps)]


def _validate_table(table: dict, positions: list[int], *, future: bool = False) -> None:
    columns, rows = table["columns"], table["rows"]
    if (not isinstance(columns, list) or not columns or columns[0] != "position"
            or len(set(columns)) != len(columns)
            or any(not isinstance(c, str) or "|" in c or "\n" in c for c in columns)):
        raise ValueError("Invalid compact table columns")
    if len(rows) != len(positions) or any(
            not isinstance(row, list) or len(row) != len(columns)
            or type(row[0]) is not int or row[0] != p for row, p in zip(rows, positions)):
        raise ValueError("Compact table positions/width do not match declared grid")
    if future and "load" in columns:
        raise ValueError("Future public covariates must not contain load")
    for row in rows:
        for value in row[1:]:
            if value is not None and type(value) not in {int, float}:
                raise ValueError("Numeric evidence cells must be numbers or null")
            _literal(value)


def _table_lines(table: dict) -> list[str]:
    return ["|".join(table["columns"]), "|".join("---" for _ in table["columns"])] + [
        "|".join(_literal(value) for value in row) for row in table["rows"]]


def _column_numeric_literals(values: list[Any]) -> tuple[list[str], int]:
    """Use one minimal exact decimal precision after bounded tail cleanup."""
    cleaned = [None if v is None else Decimal(compact_numeric_literal(v)) for v in values]
    places = 0
    for value in cleaned:
        if value is None:
            continue
        digits, exponent = list(value.as_tuple().digits), value.as_tuple().exponent
        while exponent < 0 and digits and digits[-1] == 0:
            digits.pop()
            exponent += 1
        places = max(places, max(0, -exponent))
    return ["null" if v is None else format(v, f".{places}f") for v in cleaned], places


def _source_calendar(stamp: str) -> tuple[str, list[int | None]]:
    try:
        moment = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
    except (ValueError, AttributeError, TypeError):
        return stamp, [None, None, None]
    if not _STAMP.fullmatch(stamp):
        # datetime accepts finer fractions by truncating them; do not shorten
        # a source clock whose spelling exceeds Python's microsecond precision.
        return stamp, [moment.hour, moment.weekday(), moment.month]
    precision = "minutes" if moment.second == 0 and moment.microsecond == 0 else "auto"
    return moment.isoformat(sep=" ", timespec=precision), [moment.hour, moment.weekday(), moment.month]


def build_compact_display_table(case: dict, scope: str) -> dict:
    """Bind each displayed row to an exact source step and calendar.

    Original clocks remain in the checked grid. Calendar covariates already
    present in the source table must agree with that source clock before they
    are deduplicated into the explicit calendar columns.
    """
    table = case[scope]
    history = scope == "history"
    stamps = case["grid"]["history_timestamps" if history else "future_timestamps"]
    positions = list(range(-len(stamps), 0)) if history else list(range(len(stamps)))
    rows = table["rows"] or [[p] for p in positions]
    from .display_clock import display_timestamps
    clock = display_timestamps(case, scope)
    source_time_rows = [_source_calendar(stamp) for stamp in stamps]
    time_rows = [_source_calendar(stamp) for stamp in clock["timestamps"]]
    for name, offset in (("hod", 0), ("dow", 1), ("month", 2)):
        if name in table["columns"]:
            index = table["columns"].index(name)
            for row, (_, calendar) in zip(rows, source_time_rows):
                if calendar[offset] is None or row[index] != calendar[offset]:
                    raise ValueError("Source calendar column disagrees with timestamp: " + name)
    absent = [name for i, name in enumerate(table["columns"]) if i and name != "load"
              and rows and all(row[i] is None for row in rows)]
    retained = [name for name in table["columns"][1:]
                if name not in absent and name not in {"hod", "dow", "month", "load"}]
    if set(retained) & {"step", "time", "target"}:
        raise ValueError("Source covariate conflicts with explicit display column")
    evidence_columns = (["load"] if history else []) + retained
    columns = ["step", "time", "hod", "dow", "month"] + [
        "target" if name == "load" else name for name in evidence_columns]
    display_rows = [[str(p), time, *(_literal(v) for v in calendar)]
                    for p, (time, calendar) in zip(positions, time_rows)]
    precision = {}
    evidence_values = {}
    for name in evidence_columns:
        index = table["columns"].index(name)
        values = [row[index] for row in rows]
        cells, digits = _column_numeric_literals(values)
        display_name = "target" if name == "load" else name
        precision[display_name] = digits
        evidence_values[display_name] = values
        for row, cell in zip(display_rows, cells):
            row.append(cell)
    return {"columns": columns, "rows": display_rows, "all_null_columns": absent,
            "column_fractional_places": precision, "evidence_values": evidence_values,
            "source_timestamps": stamps, "calendar_basis": clock["policy"]["label"],
            "display_clock_policy": clock["policy"], "clock_converted": clock["converted"]}


def _display_table_lines(table: dict) -> list[str]:
    return ["|".join(table["columns"]), "|".join("---" for _ in table["columns"])] + [
        "|".join(row) for row in table["rows"]]


def _merge_partial(context: Any, history: dict, variables: dict, past: list[str]) -> None:
    """Merge explicitly aligned channels wherever a source stored them."""
    if isinstance(context, dict):
        partial = context.pop("partial_historical_covariates", None)
        if partial is not None:
            if not isinstance(partial, dict):
                raise ValueError("Partial historical covariates must be a channel map")
            for name, channel in partial.items():
                if not isinstance(channel, dict) or not isinstance(channel.get("values"), list):
                    raise ValueError("Partial historical channel lacks values")
                values = channel["values"]
                if len(values) != len(past) or ("timestamps" in channel and channel["timestamps"] != past):
                    raise ValueError("Partial historical channel clock/length mismatch")
                if name in {"position", "load"}:
                    raise ValueError("Reserved partial channel name")
                if name in history["columns"]:
                    index = history["columns"].index(name)
                    if [row[index] for row in history["rows"]] != values:
                        raise ValueError("Conflicting duplicate partial historical channel")
                else:
                    history["columns"].append(name)
                    for row, value in zip(history["rows"], values):
                        row.append(value)
                metadata = {k: copy.deepcopy(v) for k, v in channel.items() if k not in {"values", "timestamps"}}
                current = variables.setdefault(name, {}).setdefault("history", {})
                for key, value in metadata.items():
                    if key in current and current[key] != value:
                        raise ValueError("Conflicting partial channel metadata")
                    current[key] = value
        for child in context.values():
            _merge_partial(child, history, variables, past)
    elif isinstance(context, list):
        for child in context:
            _merge_partial(child, history, variables, past)


def prepare_compact_case(public_case: dict) -> dict:
    """Return a checked copied public view with partial channels aligned.

    The audit view still includes all-null channels. Only their *display* is
    replaced with scoped absence statements, so parity checks can recover them.
    """
    result = copy.deepcopy(public_case)
    if set(result) != {"task", "grid", "variables", "history", "future_known", "context", "events"}:
        raise ValueError("Compact renderer expects the shared public JSON case")
    past, future = result["grid"]["history_timestamps"], result["grid"]["future_timestamps"]
    if not past or not future:
        raise ValueError("Public history and forecast grids must be nonempty")
    for key, actual in (("context_length", len(past)), ("prediction_length", len(future))):
        if key in result["task"] and result["task"][key] != actual:
            raise ValueError("Task/grid length mismatch")
    _merge_partial(result["context"], result["history"], result["variables"], past)
    _validate_table(result["history"], list(range(-len(past), 0)))
    if "load" not in result["history"]["columns"]:
        raise ValueError("History requires public load observations")
    known = result["future_known"]
    _validate_table(known, list(range(len(future))) if len(known["columns"]) > 1 else [], future=True)
    for scope, table in (("history", result["history"]), ("future_known", known)):
        for index, name in enumerate(table["columns"][1:], 1):
            metadata = result["variables"].get(name, {}).get(scope, {})
            values = [row[index] for row in table["rows"]]
            mask = metadata.get("observed_mask")
            if mask is not None and (not isinstance(mask, list) or len(mask) != len(values)
                    or any(type(flag) is not bool or (flag and value is None)
                           for flag, value in zip(mask, values))):
                raise ValueError("Observed mask disagrees with public numeric cells")
            if "missing_count" in metadata and metadata["missing_count"] != (
                    mask.count(False) if mask is not None else values.count(None)):
                raise ValueError("Missing count disagrees with public numeric cells")
    _literal(result)
    return result


def _runs(values: list) -> list[tuple[int, int, Any]]:
    result: list[tuple[int, int, Any]] = []
    for i, value in enumerate(values):
        if result and type(value) is type(result[-1][2]) and value == result[-1][2]:
            a, _, saved = result[-1]
            result[-1] = (a, i + 1, saved)
        else:
            result.append((i, i + 1, value))
    return result


def _array_fact(values: list, start: int = 0) -> str:
    runs = _runs(values)
    return "; ".join(f"[{a + start},{b + start})={_literal(v)}" for a, b, v in runs)


def _categorical_table(table: dict, label: str) -> list[str]:
    columns, rows = table["columns"], table["rows"]
    if not columns or columns[0] != "timestamp" or any(len(r) != len(columns) for r in rows):
        raise ValueError("Invalid categorical timestamp table")
    timestamps = [row[0] for row in rows]
    if not timestamps:
        return []
    if regular_step_seconds(timestamps) is None:
        return [label + ":"] + ["|".join(columns), "|".join("---" for _ in columns)] + [
            "|".join(_literal(v) for v in row) for row in rows]
    lines = _clock_lines(label + " native slots", timestamps, 0, "slot")
    lines.append("Native slot ranges are zero-based half-open and separate from target positions.")
    for index, name in enumerate(columns[1:], 1):
        lines.append(name + ": " + _array_fact([row[index] for row in rows]))
    return lines


def _tariff_schedule(records: list[dict], label: str) -> list[str]:
    if not records:
        return []
    if any(not isinstance(r, dict) for r in records):
        raise ValueError("Tariff schedule must contain records")
    times = [r.get("effective_at") for r in records]
    regular = all(isinstance(t, str) for t in times) and regular_step_seconds(times) is not None
    common = {key: value for key, value in records[0].items()
              if key not in {"event_id", "tariff", "effective_at"}
              and all(key in r and r[key] == value for r in records)}
    lines = ["Future tariff schedule (native slots, separate from target positions):"]
    if regular:
        lines.extend(_clock_lines("Tariff schedule", times, 0, "slot"))
    if common:
        lines.extend(_context_lines(common, "All tariff slots"))
    columns = ["event_id", "slot", "tariff"] + ([] if regular else ["effective_at"])
    lines += ["|".join(columns), "|".join("---" for _ in columns)]
    for slot, record in enumerate(records):
        cells = [record.get("event_id"), slot, record.get("tariff")]
        if not regular:
            cells.append(record.get("effective_at"))
        lines.append("|".join(_literal(v) if not isinstance(v, str) else v for v in cells))
        exceptions = {k: v for k, v in record.items()
                      if k not in common and k not in {"event_id", "tariff", "effective_at"}}
        if exceptions:
            lines.extend(_context_lines(exceptions, f"Tariff {record.get('event_id')} slot {slot}"))
    return lines


def _friendly(key: str) -> str:
    return key.replace("_", " ").replace(".", " / ")


_NATURAL = {
    "historical_only": "historical observations only",
    "oracle_future_observed": "realized future weather supplied as offline oracle covariates",
    "calendar_from_supplied_timestamp": "calendar values derived from the supplied timestamps",
    "research_only_plan_clock_and_episode_notification_not_certified":
        "research price plan; advance availability and episode notification are not certified",
    "offline_final_vintage_history_not_production_ingestion_evidence":
        "offline final-vintage history; this does not prove real-time ingestion",
    "offline_final_or_revised_historical_values_not_realtime_replay":
        "offline final or revised historical values; not a real-time replay",
    "offline_final_or_revised_not_realtime_replay":
        "offline final or revised observations; not a real-time replay",
    "preorigin_notification_body_asof_unknown": "pre-origin notification; first-body availability is unknown",
    "retrospective_oracle_context": "retrospective/oracle context",
    "inferred_official_lcl_day_ahead_0830_protocol": "inferred from the official LCL D-1 08:30 notification protocol",
    "protocol_inferred_no_household_delivery_proof": "protocol inference; individual household delivery is unproven",
    "native_UK_clock_GMT_season_only": "native UK clock in the GMT season only",
    "research_consumer_probe_not_operational_known_future_certification":
        "research plan; not certified as operational known-future evidence",
    "exact_literal_From_and_Date_Hour_profile_join": "exact literal From/Date/Hour profile alignment",
    "contextual_relation": "contextual relation only",
    "physical_utc": "physical UTC",
    "interval_start": "interval start",
    "hourly_mean_of_complete_native_values": "hourly mean of complete native values",
    "hourly_mean_of_complete_native_values; EIA fields retain MWh-per-original-one-hour-interval, not relabelled MW":
        "hourly mean of complete native values; EIA energy stays MWh per original one-hour interval",
    "unknown": "unknown",
    "native_revision_publication_before_origin; research assumption distinct from operational availability":
        "for timed/native event records only, native revision publication precedes forecast origin under a research assumption; nested retrospective notices retain their own later publication and unknown as-of availability",
    "mentioned_NEM_region_context_not_proven_causal_demand_response_exposure":
        "mentioned NEM region provides context; causal demand-response exposure is unproven",
    "available_at_unknown; native creation/effective/event clocks do not prove body availability":
        "body availability unknown; native creation, effective and event clocks do not prove it",
    "conservative_full_body_outcome_or_ambiguous_text_excluded_to_evaluator":
        "full bodies containing outcomes or ambiguous text are excluded from this model evidence",
    "explicit_oracle_injection_of_advance_mechanism_excerpt_no_source_availability_or_exact_issue_claim":
        "explicit oracle injection of an advance-mechanism excerpt; source availability and exact issue time are not established",
    "aemo_full_history": "AEMO historical market notices",
}


def _fact(key: str, value: Any) -> str:
    name = _friendly(key)
    if value is None:
        return name + " unknown / not supplied"
    if isinstance(value, str):
        return name + ": " + _NATURAL.get(value, value)
    if type(value) is bool:
        return name + ": " + ("yes" if value else "no")
    return name + ": " + _literal(value)


def _context_lines(value: Any, label: str, history_length: int | None = None) -> list[str]:
    """Readable scoped fallback: unknown public facts are never discarded.

    Arrays become exact runs or record tables, not a large JSON wrapper tree.
    Native strings are literal prose and are never parsed or summarized here.
    """
    label = _friendly(label)
    for prefix, readable in (("Context / source context / context", "Household context"),
                             ("Context / source context", "Source"),
                             ("Context / context text", "Source description")):
        if label.startswith(prefix):
            label = readable + label[len(prefix):]
            break
    if isinstance(value, dict):
        if not value:
            return []
        if set(value) == {"columns", "rows"} and value["columns"] and value["columns"][0] == "timestamp":
            return _categorical_table(value, label)
        if set(value) == {"parts"} and isinstance(value["parts"], list):
            lines = [label + ":"]
            for i, child in enumerate(value["parts"]):
                lines.extend([child] if isinstance(child, str) else _context_lines(child, "part " + str(i), history_length))
            return lines
        lines, facts = [], []
        for key, child in value.items():
            if key == "retrospective_oracle_events":
                # The event renderer consumes this original public container.
                continue
            if key == "inferred_day_ahead_tariff_schedule":
                lines.extend(_tariff_schedule(child, label + "." + key))
            elif child == {} or child == []:
                continue
            elif isinstance(child, (dict, list)):
                lines.extend(_context_lines(child, label + "." + key, history_length))
            elif isinstance(child, str) and ("\n" in child or len(child) > 180):
                lines.extend([label + "." + key + ":", child])
            else:
                facts.append(_fact(key, child))
        return ([label + ": " + "; ".join(facts)] if facts else []) + lines
    if isinstance(value, list):
        if not value:
            return []
        if all(not isinstance(v, (dict, list)) for v in value):
            start = -history_length if history_length is not None and len(value) == history_length and label.endswith(("Participation Phase", "Price signal")) else 0
            coordinate = "history positions" if start else "array indices"
            return [label + f" ({coordinate}, half-open): " + _array_fact(value, start)]
        lines = []
        for i, child in enumerate(value):
            lines.extend(_context_lines(child, label + "[" + str(i) + "]", history_length))
        return lines
    return [label + ": " + (value if isinstance(value, str) else _literal(value))]


def _variables_lines(variables: dict, history: dict, future: dict) -> list[str]:
    grouped: dict[str, list[str]] = {}
    lines = []
    historical_only = []
    if any(_STATIONS in str(scopes) for scopes in variables.values()):
        lines.append("Weather summary station set: w1–w25 (the supplied 25 stations).")
    for name, scopes in variables.items():
        if name == "load":
            # Native target meaning and unit are already in the task header.
            scopes = {scope: {k: v for k, v in raw.items() if k not in {"name", "unit"}}
                      for scope, raw in scopes.items()}
        for scope, raw in scopes.items():
            metadata = copy.deepcopy(raw)
            if metadata.get("name") == name:
                metadata.pop("name")
            if "description" in metadata and isinstance(metadata["description"], str):
                metadata["description"] = metadata["description"].replace(
                    " across stations " + _STATIONS, " across the supplied station set")
            # Verified internal switches / redundant counts do not convey source
            # evidence. A new policy value is retained rather than guessed away.
            for key in ("enabled", "enabled_for_model"):
                if metadata.get(key) is True:
                    metadata.pop(key)
            metadata.pop("missing_count", None)  # checked against mask/cells above
            if metadata.get("missingness_policy") == "complete_numeric_or_partial_text_with_mask_v1":
                metadata.pop("missingness_policy")
            mask = metadata.pop("observed_mask", None)
            table = history if scope == "history" else future
            if mask is not None and name in table["columns"]:
                index = table["columns"].index(name)
                values = [row[index] for row in table["rows"]]
                if any(value is not None and not flag for value, flag in zip(values, mask)):
                    start = -len(values) if scope == "history" else 0
                    lines.append(name + " (" + scope.replace("_", " ") +
                                 "): supplied non-null cells are not all observed; observed flags by position: " +
                                 _array_fact(mask, start))
            if metadata.get("availability") == "historical_only" and scope == "history":
                historical_only.append(name)
                metadata.pop("availability")
            # Full common scientific facts are hoisted with exact channel scope.
            for key, value in metadata.items():
                signature = _fact(key, value)
                grouped.setdefault(signature, []).append(name + " (" + scope.replace("_", " ") + ")")
    for fact, scopes in grouped.items():
        lines.append(", ".join(scopes) + ": " + fact)
    if historical_only:
        lines.insert(1, "Historical covariates (" + ", ".join(dict.fromkeys(historical_only)) + ") are observed history only; no future values are supplied for these channels.")
    return lines


def _task_lines(task: dict, case: dict, *, explicit_time: bool = False) -> list[str]:
    """Separate scientific qualifications from selector/profile configuration."""
    lines = []
    mode = task.get("research_mode")
    if mode == "offline_oracle_future_temperature":
        weather = {"temp_mean", "temp_min", "temp_max", "hdd65", "cdd65"}
        if not all(case["variables"].get(name, {}).get("future_known", {}).get("availability") == "oracle_future_observed"
                   for name in weather):
            lines.append("Evidence qualification: offline benchmark with realized future temperature as oracle covariates.")
    elif mode == "retrospective_oracle_context":
        lines.append("Evidence qualification: retrospective/oracle context; operational availability is not certified.")
    elif mode not in {None, "historical_only", "preorigin"}:
        lines.append("Research qualification: " + str(mode))
    protocol = task.get("input_protocol")
    if protocol == "retrospective_research_unknown_event_asof":
        lines.append("Notification body availability as of the forecast origin is unknown.")
    elif protocol in {"research_oracle_context_not_operational_replay", "explicit_research_context_injection_not_operational_asof"}:
        lines.append("The supplied event context is a research injection, not an operational as-of replay.")
    elif protocol not in {None, "historical_plus_known_future", "asof_operational"}:
        lines.append("Input qualification: " + str(protocol))
    # The legacy asof_operational selector does not certify NWS body availability;
    # source-level availability statements and retrospective mode govern instead.
    known = {"dataset_id", "series_id", "freq", "context_length", "prediction_length",
             "domain", "track", "input_protocol", "research_mode", "target_start"}
    clocks = {key: task[key] for key in ("cutoff_time", "origin_time", "issue_time") if key in task}
    missing_origin_issue = []
    for key, value in clocks.items():
        if key == "cutoff_time" and value == case["grid"]["history_timestamps"][-1]:
            continue
        if explicit_time and key in {"origin_time", "issue_time"} and value is None:
            missing_origin_issue.append(_friendly(key))
            continue
        lines.append("Task " + _fact(key, value) + ".")
    if missing_origin_issue:
        lines.append("Task origin/issuance metadata: " + ", ".join(missing_origin_issue)
                     + " unknown / not supplied; forecast start is position 0 on the Forecast grid.")
    if task.get("target_start") is not None:
        lines.append("Task " + _fact("target_start", task["target_start"]) + ".")
    for key, value in task.items():
        if key not in known and key not in clocks:
            lines.append("Task " + _fact(key, value) + ".")
    return lines


def _clean_context(case: dict) -> dict:
    """Remove only checked redundant/configuration fields from the copied view."""
    context = copy.deepcopy(case["context"])
    if context.get("normalization") is None or context.get("normalization") == {"method": "none"}:
        context.pop("normalization", None)
    if context.get("static_attributes") is None:
        context.pop("static_attributes", None)
    source = context.get("source_context", {})
    if isinstance(source, dict):
        research = source.get("research_context")
        if isinstance(research, dict):
            if research.get("mode") == case["task"].get("research_mode"):
                research.pop("mode", None)
            geographic = research.get("geographic_relation_policy")
            if isinstance(geographic, dict):
                for key in ("native_country_field_used", "prefix_country_derivation_used"):
                    if geographic.get(key) is False:
                        geographic.pop(key)
            relations = research.get("native_zone_context_relations")
            if isinstance(relations, list):
                # The literal area/EIC/target remains in context. Qualifications
                # may be omitted here only when the same mapped source event
                # already carries their exact values in its evidence scope.
                event_relations = []
                def collect_event_relations(value):
                    if isinstance(value, dict):
                        if isinstance(value.get("contextual_relation"), dict):
                            event_relations.append(value)
                        for child in value.values():
                            collect_event_relations(child)
                    elif isinstance(value, list):
                        for child in value:
                            collect_event_relations(child)
                collect_event_relations(case["events"])
                for relation in relations:
                    if not isinstance(relation, dict):
                        continue
                    matches = [event for event in event_relations if all(
                        key in relation and source_key in event["contextual_relation"]
                        and _exact(relation[key], event["contextual_relation"][source_key])
                        for key, source_key in (("native_area", "native_area"),
                                                ("native_area_eic", "code"),
                                                ("target_entity", "target_entity")))]
                    if not matches:
                        continue
                    if relation.get("relation") == "contextual_relation":
                        relation.pop("relation")
                    for key in ("target_equivalence", "quantified_effect"):
                        if key in relation and any(key in event and _exact(relation[key], event[key])
                                                   for event in matches):
                            relation.pop(key)
        plan = source.get("research_price_plan_scope")
        if isinstance(plan, dict) and "plan_id" in plan:
            shared_ids = {scope.get("source_plan_id") for variable in case["variables"].values()
                          for scope in variable.values() if isinstance(scope, dict) and "source_plan_id" in scope}
            if plan["plan_id"] in shared_ids:
                plan.pop("plan_id")  # same original ID remains beside the future price channel
        if source.get("exogenous_history") is None:
            source.pop("exogenous_history", None)
        quality = source.get("quality")
        if isinstance(quality, dict):
            counts = quality.get("historical_covariate_missing_counts")
            if isinstance(counts, dict) and all(
                    name in case["history"]["columns"] and count == sum(
                        row[case["history"]["columns"].index(name)] is None for row in case["history"]["rows"])
                    for name, count in counts.items()):
                quality.pop("historical_covariate_missing_counts")
            if quality.get("partial_historical_covariate_channels") == []:
                quality.pop("partial_historical_covariate_channels")
        meta = source.get("metadata")
        if isinstance(meta, dict):
            grid = case["grid"]
            for key, expected in (("history_last_interval_start", grid["history_timestamps"][-1]),
                                  ("target_start", grid["future_timestamps"][0])):
                if meta.get(key) == expected:
                    meta.pop(key)
            # Endpoints retain their interval-end role, even when equal to an
            # anchor. An issue timestamp is never erased as a target duplicate.
            if meta.get("lead_gap_hours") == 0:
                try:
                    last = datetime.fromisoformat(grid["history_timestamps"][-1])
                    first = datetime.fromisoformat(grid["future_timestamps"][0])
                    if (first - last).total_seconds() == 3600:
                        meta.pop("lead_gap_hours")
                except (TypeError, ValueError):
                    pass
            lineage = meta.get("lineage")
            if isinstance(lineage, dict):
                if lineage.get("clock_basis") == meta.get("clock_basis"):
                    lineage.pop("clock_basis", None)
                if lineage.get("source_timestamp_convention") == meta.get("timestamp_convention"):
                    lineage.pop("source_timestamp_convention", None)
                if lineage.get("entity_id") == case["task"].get("series_id"):
                    lineage.pop("entity_id", None)
                units = lineage.get("exogenous_units")
                if isinstance(units, dict):
                    for name, unit in list(units.items()):
                        if case["variables"].get(name, {}).get("history", {}).get("unit") == unit:
                            units.pop(name)
                evidence = lineage.get("exogenous_unit_evidence")
                if isinstance(evidence, dict):
                    for name, record in evidence.items():
                        if isinstance(record, dict) and record.get("field_id") == name:
                            record.pop("field_id")
                # The EIA generated metadata sentence is redundant only when
                # every component is explicitly supplied in these structures.
                generated = ("reported_demand_MWh_per_hour_converted_to_mean_MW; unit=MW; "
                             "clock_basis=physical_utc; hour labels are declared coordinate interval starts; source timezone=UTC.")
                metadata_text = context.get("context_text")
                if (isinstance(metadata_text, dict) and metadata_text.get("metadata_text") == generated
                        and case["variables"].get("load", {}).get("history") == {
                            "name": "reported_demand_MWh_per_hour_converted_to_mean_MW", "unit": "MW"}
                        and meta.get("clock_basis") == "physical_utc"
                        and meta.get("timestamp_convention") == "interval_start"
                        and lineage.get("timezone") == "UTC"):
                    metadata_text.pop("metadata_text")
    if (case["task"].get("dataset_id") == "gefcom2014"
            and context.get("forecasting_guidance") == _GEF_GENERIC_GUIDANCE):
        context.pop("forecasting_guidance")
    weather = {"temp_mean", "temp_min", "temp_max", "hdd65", "cdd65"}
    if case["task"].get("dataset_id") == "gefcom2014" and all(
            case["variables"].get(name, {}).get("future_known", {}).get("availability") == "oracle_future_observed"
            for name in weather):
        exact = {
            "track_guidance": "- This is an offline-aligned benchmark with oracle future temperature covariates over the forecast window.\n- The load forecast should use both recent target behavior and the future temperature profile.",
            "covariate_qualification": "Use the observed target history and the provided future temperature covariates. For load, the forecast-window temperatures are oracle future covariates in this offline-aligned benchmark.",
        }
        for key, value in exact.items():
            if context.get(key) == value:
                context.pop(key)
    return context


def render_compact_case(public_case: dict, *, reviewed_text: dict | None = None,
                        selected_case_id: str | None = None, require_reviewed: bool = False,
                        use_selected_reviewed: bool = True, input_version: str = INPUT_VERSION) -> str:
    """Render one role-independent public evidence input deterministically."""
    from .compact_events import EVENT_STRUCTURE_V2, render_events
    if input_version not in {INPUT_VERSION, EXPLICIT_TIME_INPUT_VERSION}:
        raise ValueError("Unsupported compact input revision")
    explicit_time = input_version == EXPLICIT_TIME_INPUT_VERSION
    original = prepare_compact_case(public_case)
    notes = []
    required = require_reviewed
    if explicit_time:
        if reviewed_text is None and use_selected_reviewed and selected_case_id is not None:
            public_rows, reviews = _v2_selected_assets()
            if selected_case_id in public_rows:
                if not _exact(public_case, public_rows[selected_case_id]["case"]):
                    raise ValueError("Compact v2 selected public case differs from revision-bound assets")
                reviewed_text, required = reviews[selected_case_id], True
    else:
        required = required or is_selected_public_case(selected_case_id)
        if reviewed_text is None and use_selected_reviewed:
            reviewed_text = _selected_review(public_case, case_id=selected_case_id, required=required)
    if required and reviewed_text is None:
        raise ValueError("Selected compact case requires exact reviewed natural notes")
    if reviewed_text is not None and selected_case_id is not None and reviewed_text.get("case_id") != selected_case_id:
        raise ValueError("Reviewed text differs from selected case identity")
    event_structure = None
    if reviewed_text is not None and "event_structure_format" in reviewed_text:
        if not explicit_time:
            raise ValueError("Reviewed event structure format requires compact v2")
        event_structure = reviewed_text["event_structure_format"]
        if event_structure != EVENT_STRUCTURE_V2:
            raise ValueError("Unknown reviewed event structure format")
    if reviewed_text is not None:
        changed, notes = _apply_reviewed_text(public_case, reviewed_text)
        case = prepare_compact_case(changed)
        for scope in ("grid", "history", "future_known", "events"):
            if not _exact(case[scope], original[scope]):
                raise ValueError("Reviewed text altered protected numeric/clock/event evidence")
    else:
        case = original
    task, grid = case["task"], case["grid"]
    past, future = grid["history_timestamps"], grid["future_timestamps"]
    lines = ["Forecast task:", "Dataset: " + str(task.get("dataset_id", "unknown"))
             + "; Entity: " + str(task.get("series_id", "unknown")),
             f"History: {len(past)} points; Forecast: {len(future)} points; source frequency={task.get('freq', 'unknown')}.",
             "Target: " + str(original["variables"]["load"]["history"].get("name", "load"))
             + " (" + str(original["variables"]["load"]["history"].get("unit", "unknown")) + ").",
             f"Positions: history [{-len(past)},0), forecast [0,{len(future)}); each anchor uses its supplied clock spelling."]
    lines.extend(_clock_lines("History grid", past, -len(past)))
    lines.extend(_clock_lines("Forecast grid", future, 0))
    if explicit_time:
        lines.append("Table steps bind to these exact grids; calendar columns use the declared display clock.")
    try:
        no_zone = all(datetime.fromisoformat(t.replace("Z", "+00:00")).tzinfo is None for t in (past[0], future[0]))
    except (ValueError, TypeError, AttributeError):
        no_zone = True
    if no_zone:
        lines.append("Clock: source-label time; timezone and interval convention unknown unless separately supplied.")
    else:
        lines.append("Clock: retain supplied offsets; an interval convention is used only where explicitly supplied below.")
    lines.extend(_task_lines(task, case, explicit_time=explicit_time))
    variable_lines = _variables_lines(case["variables"], case["history"], case["future_known"])
    if variable_lines:
        lines += ["", "Variable meanings and availability:"] + variable_lines
    if notes:
        lines.extend(["", "Public context and variable notes:"] + notes)
    if explicit_time:
        from .display_clock import display_clock_for_case
        lines.append("Table calendar: " + display_clock_for_case(case)["label"]
                     + "; hod=hour, dow=Monday 0 through Sunday 6, month=1..12. "
                     "Row offsets distinguish repeated hours; source/event clocks remain unchanged.")
    for scope, label in (("history", "Historical observations"), ("future_known", "Future covariates")):
        table = case[scope]
        if not explicit_time and not table["rows"]:
            continue
        if explicit_time:
            displayed = build_compact_display_table(case, scope)
        else:
            absent = [i for i, name in enumerate(table["columns"]) if i and name != "load"
                      and all(row[i] is None for row in table["rows"])]
            retained = [i for i in range(len(table["columns"])) if i not in absent]
            displayed = {"columns": [table["columns"][i] for i in retained],
                         "rows": [[row[i] for i in retained] for row in table["rows"]],
                         "all_null_columns": [table["columns"][i] for i in absent]}
        missing = displayed["all_null_columns"]
        if missing:
            lines.append(label + ": " + ", ".join(missing)
                         + f" unavailable throughout all {len(displayed['rows'])} supplied positions (all null).")
        if explicit_time or len(displayed["columns"]) > 1:
            lines.extend(["", label + ":"] + (_display_table_lines(displayed) if explicit_time else _table_lines(displayed)))
    context_lines = _context_lines(_clean_context(case), "Context", len(past))
    if context_lines:
        lines.extend(["", "Public context:"] + context_lines)
    event_text = render_events(public_case, structure_format=event_structure)
    if event_text:
        lines.extend(["", event_text])
    return "\n".join(lines)


def build_public_compact_case(raw_shared_row: dict, *, reviewed_text: dict | None = None,
                              selected_case_id: str | None = None,
                              use_selected_reviewed: bool = True,
                              input_version: str = INPUT_VERSION) -> dict:
    """Resolve the same revision-owned evidence for rendering and Agent state.

    A corrected event tree may retain nonconsecutive public aliases. Never
    rebuild those aliases differently in a workflow than in its public input.
    The existing physical binding is preserved; returned state is owned.
    """
    from .json_rows_view import build_public_json_case
    if input_version not in {INPUT_VERSION, EXPLICIT_TIME_INPUT_VERSION}:
        raise ValueError("Unsupported compact input revision")
    public_case = build_public_json_case(raw_shared_row)
    if input_version == EXPLICIT_TIME_INPUT_VERSION:
        saved = reviewed_text.get("public_case") if reviewed_text is not None else None
        if saved is None and selected_case_id is not None and use_selected_reviewed:
            public_rows, reviews = _v2_selected_assets()
            if selected_case_id in public_rows:
                saved = public_rows[selected_case_id]["case"]
        if saved is not None:
            # Numeric/clock bindings establish this physical case. Revision
            # overlays retain stable public evidence aliases instead of
            # recreating them after a corrected event relation is removed.
            for key in ("grid", "history", "future_known"):
                if not _exact(public_case[key], saved[key]):
                    raise ValueError("Compact v2 raw/public physical binding differs: " + key)
            if any(not _exact(public_case["task"].get(key), saved["task"].get(key))
                   for key in ("dataset_id", "series_id")):
                raise ValueError("Compact v2 raw/public task identity differs")
            public_case = copy.deepcopy(saved)
    return public_case


def render_public_compact_case(raw_shared_row: dict, *, reviewed_text: dict | None = None,
                               selected_case_id: str | None = None, require_reviewed: bool = False,
                               use_selected_reviewed: bool = True, input_version: str = INPUT_VERSION) -> str:
    """Build and render the existing label-free shared public projection."""
    public_case = build_public_compact_case(raw_shared_row, reviewed_text=reviewed_text,
        selected_case_id=selected_case_id, use_selected_reviewed=use_selected_reviewed,
        input_version=input_version)
    return render_compact_case(public_case, reviewed_text=reviewed_text,
                               selected_case_id=selected_case_id, require_reviewed=require_reviewed,
                               use_selected_reviewed=use_selected_reviewed, input_version=input_version)
