"""Strict shared position/load Markdown tables, without repair or coercion."""
import json
import math
import re
from typing import Iterable, List, Sequence, Union

Number = Union[int, float]
POSITION_LOAD_FORMAT = "position_load_markdown_v1"
STEP_TARGET_FORMAT = "step_target_markdown_v2"
TABLE_HEADERS = {POSITION_LOAD_FORMAT: "position|load", STEP_TARGET_FORMAT: "step|target"}
_INTEGER = re.compile(r"(?:0|[1-9][0-9]*)\Z")
_NUMBER = re.compile(r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?\Z")


def _positions(positions: Iterable[int]) -> List[int]:
    result = list(positions)
    if any(type(p) is not int or p < 0 for p in result):
        raise ValueError("Forecast positions must be nonnegative integers")
    if any(a >= b for a, b in zip(result, result[1:])):
        raise ValueError("Forecast positions must be unique and increasing")
    return result


def _number(value: Number) -> Number:
    if type(value) not in (int, float):
        raise ValueError("Load must be a finite number")
    try:
        finite = math.isfinite(value)
    except OverflowError as error:
        raise ValueError("Load exceeds the finite numeric range") from error
    if not finite:
        raise ValueError("Load must be a finite number")
    return value


def render_position_table(rows: Sequence[Sequence[Number]], *, answer_format: str = POSITION_LOAD_FORMAT) -> str:
    """Render exact increasing global positions and numbers; never round."""
    rows = list(rows)
    if any(not isinstance(row, (list, tuple)) or len(row) != 2 for row in rows):
        raise ValueError("Every forecast row requires position and load")
    _positions(row[0] for row in rows)
    if answer_format not in TABLE_HEADERS:
        raise ValueError("Unknown Markdown answer format: " + str(answer_format))
    return "\n".join([TABLE_HEADERS[answer_format], "---|---"] + [
        str(row[0]) + "|" + json.dumps(_number(row[1]), allow_nan=False) for row in rows])


def parse_position_table(text: str, expected_positions: Iterable[int], *,
                         answer_format: str = POSITION_LOAD_FORMAT) -> List[Number]:
    """Read exactly the requested positions, preserving order and precision."""
    expected = _positions(expected_positions)
    if answer_format not in TABLE_HEADERS:
        raise ValueError("Unknown Markdown answer format: " + str(answer_format))
    if not isinstance(text, str) or "```" in text:
        raise ValueError("Expected one unfenced position/load table")
    lines = text.strip().splitlines()
    if len(lines) != len(expected) + 2:
        raise ValueError("Table must contain exactly all expected positions")

    def cells(line):
        if answer_format == STEP_TARGET_FORMAT:
            return line.split("|")  # v2 has no outer pipes or cell whitespace.
        return [cell.strip() for cell in line.strip().removeprefix("|").removesuffix("|").split("|")]

    if cells(lines[0]) != TABLE_HEADERS[answer_format].split("|") or cells(lines[1]) != ["---", "---"]:
        raise ValueError("Expected " + TABLE_HEADERS[answer_format] + " header and one ---|--- separator")
    values = []
    for expected_position, line in zip(expected, lines[2:]):
        row = cells(line)
        if (len(row) != 2 or not _INTEGER.fullmatch(row[0])
                or int(row[0]) != expected_position or not _NUMBER.fullmatch(row[1])):
            raise ValueError("Invalid, missing, duplicate or unordered position/load row")
        values.append(_number(json.loads(row[1])))
    return values
