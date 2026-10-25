"""Explicit Agentic JSON-row answer contract for SFT and public runtime.

Numerical tables are validated in global forecast coordinates, then checked
against the existing role geometry/semantic validators. This module does not
select a runtime renderer or implicitly migrate any running job.
"""
from copy import deepcopy
import json
from typing import Any

from castfactory.agentic import distillation_protocol as protocol

VERSION = "agentic-json-rows-v1"
COLUMNS = ["position", "load"]


def parse_object(text: str) -> dict:
    """Read one strict JSON object, rejecting duplicate keys and nonfinites."""
    try:
        result = json.loads(text, object_pairs_hook=protocol._pairs, parse_constant=protocol._constant)
    except (ValueError, TypeError) as error:
        raise ValueError("Expected one strict JSON USER object") from error
    if not isinstance(result, dict):
        raise ValueError("USER must contain one JSON object")
    protocol._finite_tree(result)
    return result


def assert_no_legacy_role_state(tree: Any, path: str = "request") -> None:
    """Reject legacy numeric arrays/report strings outside the shared case."""
    if isinstance(tree, dict):
        if "values" in tree:
            raise ValueError("Legacy values key remains in role state: " + path)
        for key, child in tree.items():
            assert_no_legacy_role_state(child, path + "." + key)
    elif isinstance(tree, list):
        for index, child in enumerate(tree):
            assert_no_legacy_role_state(child, path + "[" + str(index) + "]")
    elif isinstance(tree, str) and tree.lstrip().startswith(("{", "[")):
        try:
            decoded = json.loads(tree)
        except ValueError:
            return
        if isinstance(decoded, (dict, list)):
            # Structured reports must be structured JSON fields, not nested
            # opaque serialized reports with a second numeric contract.
            def report(value):
                if isinstance(value, dict):
                    return ("values" in value or {"columns", "rows"}.issubset(value)
                            or any(report(child) for child in value.values()))
                return isinstance(value, list) and any(report(child) for child in value)
            if report(decoded):
                raise ValueError("Opaque JSON numeric report remains in role state: " + path)


def _restore_table(table: dict, h: int) -> dict:
    result = deepcopy(table)
    start, stop = result.get("start"), result.get("stop")
    if type(start) is not int or type(stop) is not int or not 0 <= start < stop <= h:
        raise ValueError("Row-table bounds must be global positions inside the forecast horizon")
    if result.pop("columns", None) != COLUMNS:
        raise ValueError("Forecast columns must be exactly position,load")
    rows = result.pop("rows", None)
    if not isinstance(rows, list) or len(rows) != stop - start:
        raise ValueError("Row count must equal the complete assigned span")
    values = []
    for position, row in enumerate(rows, start):
        if (not isinstance(row, list) or len(row) != 2
                or type(row[0]) is not int or row[0] != position):
            raise ValueError("Forecast rows require consecutive integer global positions")
        values.append(row[1])
    if "values" in result:
        raise ValueError("A JSON-row report cannot also carry legacy values")
    result["values"] = protocol.validate_values(values, stop - start)
    return result


def _restore_reports(tree: Any, h: int) -> Any:
    if isinstance(tree, list):
        return [_restore_reports(child, h) for child in tree]
    if not isinstance(tree, dict):
        return deepcopy(tree)
    if {"start", "stop"}.issubset(tree) and ("columns" in tree or "rows" in tree):
        return _restore_table(tree, h)
    if "values" in tree and ("segment_id" in tree or "boundary_id" in tree):
        raise ValueError("Legacy numeric report found in JSON-row request")
    return {key: _restore_reports(child, h) for key, child in tree.items()}


def normalize_role_reply(parsed: dict, role: str, h: int) -> dict:
    """Derive an owned values representation for existing runtime mechanics.

    This conversion never replaces the actual parsed rows in model receipts.
    """
    restored = deepcopy(parsed)
    if role in ("T2", "T2_refine"):
        restored["answer"] = _restore_table(parsed["answer"], h)
    elif role == "T4":
        answer = deepcopy(parsed["answer"])
        if not isinstance(answer.get("edits"), list):
            raise ValueError("T4 requires an edits list")
        answer["edits"] = [_restore_table(edit, h) for edit in answer["edits"]]
        restored["answer"] = answer
    return restored


def parse_role_answer(text: str, role: str, request: dict, *, profile=None) -> dict:
    """Parse JSON-row supervision and verify the full existing role semantics.

    Return the original row-table representation. Conversion to values is
    temporary validation only, never a replacement of the supervised target.
    """
    parsed = protocol.parse_reply(text, "student")
    geometry = request.get("geometry") or {}
    h = geometry.get("H")
    if type(h) is not int or h <= 0:
        raise ValueError("JSON role request requires a positive forecast horizon")
    case = request.get("case") or {}
    times = (case.get("grid") or {}).get("future_timestamps")
    if not isinstance(times, list) or len(times) != h or times != geometry.get("future_timestamps"):
        raise ValueError("JSON role request and forecast specification must share exact timestamps")
    restored = normalize_role_reply(parsed, role, h)
    variables = _restore_reports({key: value for key, value in request.items()
                                  if key not in ("case", "role")}, h)
    from castfactory.agentic.full_agent_v4 import validate_public_role
    validate_public_role(role, restored, variables, profile=profile)
    return parsed


def validate_sft_row(row: dict) -> dict:
    """Validate the opt-in JSON SFT messages/answers before native tokenization."""
    messages = row["messages"]
    if (not isinstance(messages, list) or len(messages) < 2 or len(messages) % 2
            or messages[0].get("role") != "system" or messages[1].get("role") != "user"):
        raise ValueError("Expected system/user and complete actual repair pairs")
    request = parse_object(messages[1]["content"])
    assert_no_legacy_role_state({key: value for key, value in request.items() if key != "case"})
    expected_roles = ("T2", "T2_refine") if row["role"] == "T2" else (row["role"],)
    if request.get("role") not in expected_roles:
        raise ValueError("JSON request role does not match supervision role")
    if row["role"] == "T1" and any(key in request.get("compute_policy", {})
                                   for key in ("max_context_tokens", "max_response_tokens")):
        raise ValueError("T1 JSON request must use public semantic compute policy")
    for offset in range(2, len(messages), 2):
        if messages[offset].get("role") != "assistant" or messages[offset + 1].get("role") != "user":
            raise ValueError("Actual repair turns must alternate assistant/user")
        repair = parse_object(messages[offset + 1]["content"])
        if (repair.get("role") != request["role"] or set(repair) != {
                "role", "repair_instruction", "assignment_and_policy"}
                or not isinstance(repair["repair_instruction"], str) or not repair["repair_instruction"].strip()):
            raise ValueError("JSON repair request lost its actual role or instruction")
    return parse_role_answer(row["assistant_response"], row["role"], request)
