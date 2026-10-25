"""Opt-in mixed JSON decisions and indexed Markdown numerical answers.

Parsing preserves table rows. Only normalize_role_reply produces legacy values
for existing mechanics; neither operation changes the sampled response text.
"""
from copy import deepcopy
import json
import re

from castfactory.agentic import distillation_protocol as protocol
from castfactory.agentic.json_rows_protocol import normalize_role_reply, parse_object
from castfactory.time_r1.markdown_table import parse_position_table, render_position_table, POSITION_LOAD_FORMAT, STEP_TARGET_FORMAT

VERSION = "agentic-compact-markdown-v1"
STEP_TARGET_VERSION = "agentic-compact-markdown-v2"


def answer_format_for_version(protocol_version: str) -> str:
    if protocol_version == VERSION:
        return POSITION_LOAD_FORMAT
    if protocol_version == STEP_TARGET_VERSION:
        return STEP_TARGET_FORMAT
    raise ValueError("Unsupported compact protocol version: " + str(protocol_version))
_REPORT_FIELDS = {"segment_id", "start", "stop", "uncertainty", "self_check"}
_EDIT_FIELDS = {"boundary_id", "start", "stop"}


def _span(start, stop, h):
    if type(start) is not int or type(stop) is not int or not 0 <= start < stop <= h:
        raise ValueError("Bounds must be global integer positions within the horizon")
    return list(range(start, stop))


def _body(text: str, response_format: str = "student", *, forbid_rationale: bool = False) -> tuple:
    protocol._format(response_format)
    if not isinstance(text, str) or "```" in text:
        raise ValueError("Reply must be text without code fences")
    if forbid_rationale and re.search(r"<\s*/?\s*rationale\b(?:[^>]*>|$)", text, re.I):
        raise ValueError("Compact-v2 teacher explanation must not contain rationale tags")
    expected = ["<answer>", "</answer>"]
    pattern = r"(.*?)<answer>(.*?)</answer>\s*"
    if response_format == "student":
        expected = ["<think>", "</think>"] + expected
        pattern = r"\s*<think>(.*?)</think>\s*<answer>(.*?)</answer>\s*"
    if protocol._PROTOCOL_TAG.findall(text) != expected:
        raise ValueError("Reply requires exactly one native explanation and answer wrapper")
    match = re.fullmatch(pattern, text, re.S)
    if match is None:
        raise ValueError("Reply must end after its single answer")
    return protocol._explanation(match.group(1).strip()), match.group(2).strip()


def _parse_body(body: str, role: str, h: int, *, protocol_version: str = VERSION) -> dict:
    answer_format = answer_format_for_version(protocol_version)
    if role in ("T1", "T3"):
        return parse_object(body)
    decoder = json.JSONDecoder(object_pairs_hook=protocol._pairs, parse_constant=protocol._constant)
    try:
        answer, end = decoder.raw_decode(body)
    except (ValueError, TypeError) as error:
        raise ValueError("Mixed answer must start with one strict JSON object") from error
    if not isinstance(answer, dict):
        raise ValueError("Role metadata must be one JSON object")
    protocol._finite_tree(answer)
    table = body[end:].strip()
    if role in ("T2", "T2_refine"):
        if set(answer) != _REPORT_FIELDS:
            raise ValueError("T2 metadata requires exactly the five report fields")
        positions = _span(answer["start"], answer["stop"], h)
        values = parse_position_table(table, positions, answer_format=answer_format)
        answer.update(columns=["position", "load"], rows=[[p, v] for p, v in zip(positions, values)])
        return answer
    if role != "T4" or set(answer) != {"action", "edits"} or not isinstance(answer["edits"], list):
        raise ValueError("Unsupported role or invalid T4 metadata")
    if answer["action"] == "KEEP":
        if answer["edits"] or table:
            raise ValueError("KEEP requires empty edits and no table")
        return answer
    if answer["action"] != "REGENERATE" or not answer["edits"]:
        raise ValueError("REGENERATE requires edit descriptors and one union table")
    positions = []
    widths = []
    for edit in answer["edits"]:
        if not isinstance(edit, dict) or set(edit) != _EDIT_FIELDS:
            raise ValueError("Every edit requires exactly boundary_id,start,stop")
        span = _span(edit["start"], edit["stop"], h)
        positions.extend(span)
        widths.append(len(span))
    values = parse_position_table(table, positions, answer_format=answer_format)
    offset = 0
    for edit, width in zip(answer["edits"], widths):
        edit.update(columns=["position", "load"],
                    rows=[[p, v] for p, v in zip(positions[offset:offset + width], values[offset:offset + width])])
        offset += width
    return answer


def parse_envelope(text: str, response_format: str = "student") -> dict:
    """Validate native packaging only; callers separately validate semantics."""
    explanation, body = _body(text, response_format)
    if not body:
        raise ValueError("Answer body must be nonempty")
    return {"explanation": explanation, "answer_body": body}


def parse_role_answer(text: str, role: str, request: dict, *, profile=None, protocol_version: str = VERSION,
                      response_format: str = "student") -> dict:
    """Parse syntax, then apply the full original semantic assignment/policy."""
    if request.get("role") != role and not (role == "T2" and request.get("role") == "T2_refine"):
        raise ValueError("Semantic request role differs from response role")
    geometry = request.get("geometry") or {}
    h = geometry.get("H")
    if type(h) is not int or h <= 0:
        raise ValueError("Semantic request requires a positive horizon")
    times = (request.get("case", {}).get("grid") or {}).get("future_timestamps")
    if not isinstance(times, list) or len(times) != h or times != geometry.get("future_timestamps"):
        raise ValueError("Semantic case and geometry require identical complete forecast timestamps")
    if response_format == "teacher" and protocol_version != STEP_TARGET_VERSION:
        raise ValueError("Live compact teacher parsing requires the explicit v2 profile")
    explanation, body = _body(text, response_format,
        forbid_rationale=response_format == "teacher" and protocol_version == STEP_TARGET_VERSION)
    parsed = {"explanation": explanation, "answer": _parse_body(body, role, h, protocol_version=protocol_version)}
    normalized = normalize_role_reply(parsed, role, h)
    from castfactory.agentic.full_agent_v4 import validate_public_role
    variables = {key: value for key, value in request.items() if key not in ("role", "case")}
    validate_public_role(role, normalized, variables, profile=profile)
    return parsed


def render_answer(answer: dict, role: str, h: int, *, protocol_version: str = VERSION) -> str:
    """Convert an owned old or indexed answer to its compact answer body."""
    from castfactory.agentic.compact_prompts import dumps
    answer_format = answer_format_for_version(protocol_version)
    answer = deepcopy(answer)
    if role in ("T1", "T3"):
        return dumps(answer)
    rows = []

    def report(value, fields):
        positions = _span(value.get("start"), value.get("stop"), h)
        if "values" in value:
            numbers = protocol.validate_values(value.pop("values"), len(positions))
        else:
            if value.pop("columns", None) != ["position", "load"]:
                raise ValueError("Expected position/load report")
            report_rows = value.pop("rows", None)
            numbers = parse_position_table(render_position_table(report_rows), positions)
        if set(value) != fields:
            raise ValueError("Unexpected report metadata fields")
        rows.extend([[p, number] for p, number in zip(positions, numbers)])

    if role in ("T2", "T2_refine"):
        report(answer, _REPORT_FIELDS)
    elif role == "T4":
        if set(answer) != {"action", "edits"} or not isinstance(answer["edits"], list):
            raise ValueError("Invalid T4 answer")
        if answer["action"] == "KEEP":
            if answer["edits"]:
                raise ValueError("KEEP requires empty edits")
            return dumps(answer)
        if answer["action"] != "REGENERATE" or not answer["edits"]:
            raise ValueError("REGENERATE requires edits")
        for edit in answer["edits"]:
            report(edit, _EDIT_FIELDS)
    else:
        raise ValueError("Unsupported compact role")
    return dumps(answer) + "\n" + render_position_table(rows, answer_format=answer_format)


def render_role_reply(parsed: dict, role: str, h: int, response_format: str = "student", *,
                      protocol_version: str = VERSION) -> str:
    """Render actual archived explanation/prediction, without generating prose."""
    protocol._format(response_format)
    explanation = protocol._explanation(parsed["explanation"])
    body = render_answer(parsed["answer"], role, h, protocol_version=protocol_version)
    prefix = "<think>" + explanation + "</think>" if response_format == "student" else explanation
    result = prefix + "\n<answer>" + body + "</answer>"
    _body(result, response_format,
        forbid_rationale=protocol_version == STEP_TARGET_VERSION)
    return result


def validate_sft_row(row: dict) -> dict:
    """Bind supervision to out-of-band state and exactly rendered public input."""
    from castfactory.agentic.compact_prompts import messages_for, repair_message, request_for
    request = row.get("semantic_request")
    if not isinstance(request, dict):
        raise ValueError("Compact SFT requires an out-of-band semantic_request")
    role = request.get("role")
    if role not in (("T2", "T2_refine") if row["role"] == "T2" else (row["role"],)):
        raise ValueError("Semantic role does not match supervision")
    variables = {key: value for key, value in request.items() if key not in ("case", "role")}
    if request != request_for(role, variables, request["case"]):
        raise ValueError("Compact SFT semantic_request differs from its public projection")
    metadata = row.get("metadata") or {}
    protocol_version = row.get("protocol_version", metadata.get("protocol_version", row.get("profile_version", VERSION)))
    if row.get("profile_version", protocol_version) != protocol_version:
        raise ValueError("Compact SFT profile and protocol version disagree")
    if metadata.get("protocol_version", protocol_version) != protocol_version:
        raise ValueError("Compact SFT metadata and protocol version disagree")
    answer_format = answer_format_for_version(protocol_version)
    if row.get("answer_format", metadata.get("answer_format", answer_format)) != answer_format:
        raise ValueError("Compact SFT answer format and protocol version disagree")
    if metadata.get("answer_format", answer_format) != answer_format:
        raise ValueError("Compact SFT metadata answer format and protocol version disagree")
    from castfactory.agentic.root_role_state_view import validated_row_state_view
    display = validated_row_state_view(row)
    display_variables = {key: value for key, value in display.items() if key not in ("case", "role")}
    expected = messages_for(role, display_variables, display["case"], selected_case_id=row.get("case_id"),
                            protocol_version=protocol_version)
    messages = row["messages"]
    if not isinstance(messages, list) or len(messages) < 2 or len(messages) % 2 or messages[:2] != expected:
        raise ValueError("Compact SFT messages do not match their semantic public request")
    for offset in range(2, len(messages), 2):
        assistant, user = messages[offset:offset + 2]
        if assistant.get("role") != "assistant" or user.get("role") != "user":
            raise ValueError("Actual repair turns must alternate assistant/user")
        repair = parse_object(user["content"])
        if not isinstance(repair.get("repair_instruction"), str) or user != repair_message(role, repair["repair_instruction"]):
            raise ValueError("Repair request differs from the compact repair contract")
    return parse_role_answer(row["assistant_response"], row["role"], request, protocol_version=protocol_version)
