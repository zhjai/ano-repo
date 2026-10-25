from __future__ import annotations

import math
import re
from dataclasses import dataclass


GEFCOM_FORECAST_LENGTH = 96
ROWS_ANSWER_FORMAT = "position_load_rows_json_v1"
ROWS_PROTOCOL_VERSION = "reasoning-json-rows-v1"
MARKDOWN_ANSWER_FORMAT = "position_load_markdown_v1"
COMPACT_PROTOCOL_VERSION = "reasoning-compact-markdown-v1"
STEP_TARGET_ANSWER_FORMAT = "step_target_markdown_v2"
STEP_TARGET_PROTOCOL_VERSION = "reasoning-compact-markdown-v2"
MARKDOWN_PROFILES = {
    MARKDOWN_ANSWER_FORMAT: COMPACT_PROTOCOL_VERSION,
    STEP_TARGET_ANSWER_FORMAT: STEP_TARGET_PROTOCOL_VERSION,
}

_STRUCTURE_PATTERN = re.compile(
    r"\A\s*<think>(?P<think>.*?)</think>\s*"
    r"<answer>(?P<answer>.*?)</answer>\s*\Z",
    flags=re.IGNORECASE | re.DOTALL,
)
_CODE_BLOCK_PATTERN = re.compile(
    r"\A\s*```(?:[A-Za-z0-9_+-]+)?[ \t]*\r?\n"
    r"(?P<body>.*?)\r?\n?```\s*\Z",
    flags=re.DOTALL,
)
_NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
_ROW_PATTERN = re.compile(rf"^\s*(\d+)\s*\|\s*({_NUMBER})\s*\|?\s*$")
_SEPARATOR_PATTERN = re.compile(r"^\s*:?-{3,}:?\s*\|\s*:?-{3,}:?\s*\|?\s*$")
_HEADER_NAMES = {"index", "step", "t"}


@dataclass(frozen=True)
class ForecastAnswerContract:
    valid: bool
    values: list[float]
    indices: list[int]
    error: str | None = None
    explanation: str = ""


def parse_values_forecast_answer(
    text: str | None, *, expected_length: int, prompt_owned_prefix: str = "",
) -> ForecastAnswerContract:
    """Strict native Reasoning JSON contract; never use a numeric fallback.

    Keep the historical GEFCOM table parser separate. Only restore a think
    prefix actually owned by the selected tokenizer's generation prompt.
    """
    from castfactory.agentic.distillation_protocol import parse_reply, validate_values
    from castfactory.training.chat_template_compat import restore_prompt_owned_think_prefix

    try:
        if expected_length not in (24, 96):
            raise ValueError("Shared load requires H24 or H96")
        parsed = parse_reply(restore_prompt_owned_think_prefix(str(text or ""), prompt_owned_prefix), "student")
        if set(parsed["answer"]) != {"values"}:
            raise ValueError("Answer object must contain exactly values")
        values = validate_values(parsed["answer"]["values"], expected_length)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("values must be finite in the scoring numeric range")
        return ForecastAnswerContract(True, values, list(range(expected_length)), explanation=parsed["explanation"])
    except (ValueError, OverflowError) as error:
        return _invalid(str(error))


def validate_row_forecast_object(answer: object, expected_length: int) -> list[float]:
    """Validate the explicit rowJSON forecast without sorting or coercing rows.

    Text callers must first use a duplicate-key-rejecting JSON decoder. This
    object validator also serves materializers and answer-only SFT validation.
    """
    if isinstance(expected_length, bool) or not isinstance(expected_length, int) or expected_length not in (24, 96):
        raise ValueError("Shared load rowJSON requires H24 or H96")
    if not isinstance(answer, dict) or set(answer) != {"columns", "rows"}:
        raise ValueError("rowJSON answer must contain exactly columns and rows")
    if answer["columns"] != ["position", "load"]:
        raise ValueError("rowJSON columns must be exactly position, load")
    rows = answer["rows"]
    if not isinstance(rows, list) or len(rows) != expected_length:
        raise ValueError("rowJSON rows must be a list of the required length")
    values: list[float] = []
    for position, row in enumerate(rows):
        if not isinstance(row, list) or len(row) != 2:
            raise ValueError("rowJSON rows must contain exactly position and load")
        index, value = row
        if isinstance(index, bool) or not isinstance(index, int) or index != position:
            raise ValueError("rowJSON positions must be ordered integers from zero through H-1")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("rowJSON loads must be numbers, not booleans or strings")
        try:
            finite = math.isfinite(value)
        except OverflowError as error:
            raise ValueError("rowJSON loads must be finite in the scoring numeric range") from error
        if not finite:
            raise ValueError("rowJSON loads must be finite in the scoring numeric range")
        values.append(value)
    return values


def parse_rows_forecast_answer(
    text: str | None, *, expected_length: int, prompt_owned_prefix: str = "",
) -> ForecastAnswerContract:
    """Parse only the native Reasoning rowJSON protocol, without repairs.

    A missing opening think tag may only be restored from the actual native
    generation prompt's owned prefix. The values and historical table parsers
    remain separate protocols.
    """
    from castfactory.agentic.distillation_protocol import parse_reply
    from castfactory.training.chat_template_compat import restore_prompt_owned_think_prefix

    try:
        parsed = parse_reply(restore_prompt_owned_think_prefix(str(text or ""), prompt_owned_prefix), "student")
        values = validate_row_forecast_object(parsed["answer"], expected_length)
        return ForecastAnswerContract(True, values, list(range(expected_length)), explanation=parsed["explanation"])
    except (ValueError, OverflowError) as error:
        return _invalid(str(error))


def parse_markdown_forecast_answer(
    text: str | None, *, expected_length: int, prompt_owned_prefix: str = "",
    answer_format: str = MARKDOWN_ANSWER_FORMAT,
) -> ForecastAnswerContract:
    """Strict native think + position/load table; never infer or repair rows."""
    from castfactory.agentic.distillation_protocol import _PROTOCOL_TAG, _explanation
    from castfactory.time_r1.markdown_table import parse_position_table
    from castfactory.training.chat_template_compat import restore_prompt_owned_think_prefix

    try:
        if type(expected_length) is not int or expected_length not in (24, 96):
            raise ValueError("Shared load Markdown requires H24 or H96")
        response = restore_prompt_owned_think_prefix(str(text or ""), prompt_owned_prefix)
        if "```" in response or _PROTOCOL_TAG.findall(response) != [
            "<think>", "</think>", "<answer>", "</answer>",
        ]:
            raise ValueError("Missing, malformed, duplicate or unexpected protocol tags/code fences")
        match = re.fullmatch(r"\s*<think>(.*?)</think>\s*<answer>(.*?)</answer>\s*", response, re.S)
        if match is None:
            raise ValueError("Reply must end immediately after its single answer")
        explanation = _explanation(match[1].strip())
        values = parse_position_table(match[2], range(expected_length), answer_format=answer_format)
        return ForecastAnswerContract(True, values, list(range(expected_length)), explanation=explanation)
    except (ValueError, OverflowError) as error:
        return _invalid(str(error))


def parse_gefcom_forecast_answer(
    text: str | None,
    *,
    expected_length: int = GEFCOM_FORECAST_LENGTH,
) -> ForecastAnswerContract:
    """Parse the strict RewardV2 answer contract without accepting prefixes."""
    if text is None or not str(text).strip():
        return _invalid("empty_response")

    response = str(text)
    if not _has_exact_structure_tags(response):
        return _invalid("structural_tag_count")

    structure = _STRUCTURE_PATTERN.fullmatch(response)
    if structure is None:
        return _invalid("structural_block_order")
    if not structure.group("think").strip():
        return _invalid("empty_think_block")

    answer = structure.group("answer")
    if len(re.findall(r"```", answer)) != 2:
        return _invalid("code_block_count")
    code_block = _CODE_BLOCK_PATTERN.fullmatch(answer)
    if code_block is None:
        return _invalid("unclosed_or_malformed_code_block")

    indices: list[int] = []
    values: list[float] = []
    header_seen = False
    data_seen = False
    for raw_line in code_block.group("body").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if not data_seen and not header_seen and _is_header(line):
            header_seen = True
            continue
        if not data_seen and _SEPARATOR_PATTERN.fullmatch(line):
            continue
        match = _ROW_PATTERN.fullmatch(line)
        if match is None:
            return _invalid("non_table_content")
        data_seen = True
        index = int(match.group(1))
        value = float(match.group(2))
        if not math.isfinite(value):
            return _invalid("non_finite_value")
        indices.append(index)
        values.append(value)

    expected_indices = list(range(1, expected_length + 1))
    if indices != expected_indices:
        return ForecastAnswerContract(
            valid=False,
            values=[],
            indices=indices,
            error="answer_indices_or_length",
        )
    return ForecastAnswerContract(valid=True, values=values, indices=indices)


def _has_exact_structure_tags(text: str) -> bool:
    patterns = (r"<think>", r"</think>", r"<answer>", r"</answer>")
    return all(len(re.findall(pattern, text, flags=re.IGNORECASE)) == 1 for pattern in patterns)


def _is_header(line: str) -> bool:
    if "|" not in line:
        return False
    cells = [cell.strip().lower() for cell in line.strip("|").split("|")]
    return len(cells) == 2 and cells[0] in _HEADER_NAMES and cells[1] == "value"


def _invalid(error: str) -> ForecastAnswerContract:
    return ForecastAnswerContract(valid=False, values=[], indices=[], error=error)
