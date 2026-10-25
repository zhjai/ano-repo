"""Versioned public/privileged distillation text contract.

This module never reads task state, design documents, labels, or provider-private
reasoning on its own. Basic grid/assignment checks support task rendering;
full role-specific geometric validation belongs to the runtime.
The student format selects the existing native think adapter explicitly; its
tokenizer caller must pass enable_thinking=True and own prefix/EOS handling.
"""

import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union


VERSION = "merged-t1-answer-v1"
PROMPT_REVISION = "natural-task-v1"
STUDENT_ENABLE_THINKING = True
_RESOURCE_DIR = Path(__file__).with_name("prompts") / "merged_t1_answer_v1"
_ROLES = ("reasoning", "reasoning_review", "T1", "T1_review", "T2", "T2_refine", "T3", "T4")
_REVIEW_ROLES = ("reasoning_review", "T1_review")
_PLACEHOLDER = re.compile(r"\{\{([a-z][a-z0-9_]*)\}\}")
_PROTOCOL_TAG = re.compile(r"<\s*/?\s*(?:answer|think|result|payload|metadata)\b[^>]*(?:>|$)", re.I)


def _resource(name: str) -> str:
    return (_RESOURCE_DIR / name).read_text(encoding="utf-8").rstrip("\n")


T1_SCHEMA = _resource("t1_schema.txt")


def prompt_snapshot() -> Dict[str, str]:
    """Return exact resource text for provenance/resume checks, without hashes.

    Read on every call so edits under an unchanged VERSION are observable.
    Rendered-message witnesses remain the caller's separate responsibility.
    """
    return {path.name: path.read_text(encoding="utf-8")
            for path in sorted(_RESOURCE_DIR.glob("*.txt"))}


def _format(response_format: str) -> None:
    if response_format not in ("teacher", "student"):
        raise ValueError("response_format must be teacher or student")


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def _render_template(template: str, variables: Dict[str, Any]) -> str:
    def substitute(match: Any) -> str:
        key = match.group(1)
        if key not in variables:
            raise ValueError("missing prompt variable: " + key)
        value = variables[key]
        return value if isinstance(value, str) else _json(value)

    # One pass: braces/placeholder-like content inside evidence stay data.
    return _PLACEHOLDER.sub(substitute, template)


def _task_variables(role: str, variables: Dict[str, Any]) -> Dict[str, Any]:
    """Derive task wording from the actual forecast grid, never caller hints."""
    supplied = dict(variables)
    geometry = supplied.get("geometry")
    if not isinstance(geometry, dict):
        raise ValueError("geometry must be a dictionary")
    h = geometry.get("H")
    times = geometry.get("future_timestamps")
    if isinstance(h, bool) or not isinstance(h, int) or h <= 0:
        raise ValueError("geometry H must be a positive integer")
    if (not isinstance(times, list) or len(times) != h
            or any(not isinstance(t, str) or not t.strip() for t in times)):
        raise ValueError("geometry future_timestamps must contain H timestamp strings")
    unit = geometry.get("unit")
    unit_phrase = unit.strip() if isinstance(unit, str) and unit.strip() else (
        "the target's own units (the source does not state the unit)")
    supplied.update(horizon_points=h, first_forecast_timestamp=times[0],
                    last_forecast_timestamp=times[-1], unit_phrase=unit_phrase)
    if role in ("T2", "T2_refine"):
        assignment = supplied.get("segment_assignment")
        if not isinstance(assignment, dict):
            raise ValueError("segment_assignment must be a dictionary")
        sid, start, stop = (assignment.get(k) for k in ("segment_id", "start", "stop"))
        if (not isinstance(sid, str) or not sid.strip()
                or any(isinstance(k, bool) or not isinstance(k, int) for k in (start, stop))
                or not 0 <= start < stop <= h):
            raise ValueError("segment_assignment must identify a nonempty interval within [0,H)")
        n = stop - start
        contract = dict(assignment, required_points=n, first_target_timestamp=times[start],
                        last_target_timestamp=times[stop - 1])
        scope = (f"This segment is the complete {h}-position forecast horizon."
                 if start == 0 and stop == h else
                 f"This segment is part of a {h}-position forecast horizon; "
                 "your task in this call is this segment only.")
        supplied.update(segment_id=sid, segment_start=start, segment_stop=stop,
                        required_points=n, last_local_index=n - 1,
                        first_target_timestamp=times[start], last_target_timestamp=times[stop - 1],
                        segment_contract=contract, segment_scope=scope)
    return supplied


def messages_for(role: str, variables: Dict[str, Any], response_format: str = "teacher") -> List[Dict[str, str]]:
    """Render the adopted role request, with no implicit native-thinking toggle."""
    _format(response_format)
    if role not in _ROLES:
        raise ValueError("unsupported protocol role: " + str(role))
    if not isinstance(variables, dict):
        raise ValueError("variables must be a dictionary")
    if role in _REVIEW_ROLES:
        if response_format != "teacher":
            raise ValueError("privileged reviews require the teacher response format")
        system = _resource(role + "_system.txt")
    else:
        role_system = _resource("T2_system.txt") if role == "T2_refine" else _resource(role + "_system.txt")
        if role == "T2_refine":
            role_system += "\n\n" + _resource("T2_refine_system.txt")
        system = "\n\n".join((_resource("public_system.txt"), role_system, _resource(response_format + "_format.txt")))
    supplied = _task_variables(role, variables)
    supplied.setdefault("t1_schema", T1_SCHEMA)
    return [{"role": "system", "content": system},
            {"role": "user", "content": _render_template(_resource(role + "_user.txt"), supplied)}]


def _pairs(pairs: Any) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key: " + key)
        result[key] = value
    return result


def _constant(value: str) -> None:
    raise ValueError("nonfinite JSON number: " + value)


def _finite_tree(value: Any) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("JSON numbers must be finite")
    if isinstance(value, dict):
        for item in value.values():
            _finite_tree(item)
    elif isinstance(value, list):
        for item in value:
            _finite_tree(item)


def _explanation(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("visible public explanation must be nonempty")
    if _PROTOCOL_TAG.search(value) or "```" in value:
        raise ValueError("public explanation cannot contain protocol wrappers or code fences")
    return value


def parse_reply(text: str, response_format: str = "teacher") -> Dict[str, Any]:
    """Parse one visible explanation and one strict JSON-object answer."""
    _format(response_format)
    if not isinstance(text, str) or "```" in text:
        raise ValueError("reply must be text without code fences")
    tags = _PROTOCOL_TAG.findall(text)
    expected = ["<answer>", "</answer>"]
    if response_format == "student":
        expected = ["<think>", "</think>"] + expected
    if tags != expected:
        raise ValueError("reply contains missing, malformed, duplicate or unexpected protocol tags")
    if response_format == "teacher":
        match = re.fullmatch(r"(.*?)<answer>(.*?)</answer>\s*", text, re.S)
    else:
        match = re.fullmatch(r"\s*<think>(.*?)</think>\s*<answer>(.*?)</answer>\s*", text, re.S)
    if match is None:
        raise ValueError("reply must end immediately after its single answer")
    explanation = _explanation(match.group(1).strip())
    try:
        answer = json.loads(match.group(2), object_pairs_hook=_pairs, parse_constant=_constant)
    except (json.JSONDecodeError, ValueError) as exc:
        raise ValueError("invalid answer JSON: " + str(exc)) from exc
    if not isinstance(answer, dict):
        raise ValueError("answer must be one JSON object")
    _finite_tree(answer)
    return {"explanation": explanation, "answer": answer}


def render_reply(explanation: str, answer: Dict[str, Any], response_format: str = "teacher") -> str:
    """Deterministically render public content; never synthesize private reasoning."""
    _format(response_format)
    _explanation(explanation)
    if not isinstance(answer, dict):
        raise ValueError("answer must be a dictionary")
    body = "<answer>" + _json(answer) + "</answer>"
    if response_format == "student":
        result = "<think>" + explanation + "</think>\n" + body
    else:
        result = explanation + "\n" + body
    # Ensure embedded strings cannot smuggle extra protocol wrappers.
    parse_reply(result, response_format)
    return result


def validate_values(values: Any, length: int) -> List[Union[int, float]]:
    """Validate exact finite numeric arrays without rounding or coercion."""
    if isinstance(length, bool) or not isinstance(length, int) or length < 0:
        raise ValueError("length must be a nonnegative integer")
    if not isinstance(values, list) or len(values) != length:
        raise ValueError("values must be a list of the required length")
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("values must contain numbers, not booleans or strings")
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("values must contain only finite numbers")
    return list(values)


def _parsed(reply: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
    if isinstance(reply, str):
        return parse_reply(reply)
    if not isinstance(reply, dict) or set(reply) != {"explanation", "answer"}:
        raise ValueError("parsed reply must contain exactly explanation and answer")
    # Parsed callers receive the same strict checks as text callers.
    parse_reply(render_reply(reply["explanation"], reply["answer"]))
    return reply


def _review(reply: Union[str, Dict[str, Any]], rewrite_keys: Any) -> Dict[str, Any]:
    answer = _parsed(reply)["answer"]
    if answer == {"decision": "KEEP"}:
        return answer
    if answer.get("decision") != "REWRITE" or set(answer) != set(rewrite_keys):
        raise ValueError("review must be exact KEEP or complete REWRITE")
    _explanation(answer["explanation"])
    return answer


def review_reasoning(reply: Union[str, Dict[str, Any]], original_explanation: str) -> str:
    """Select explanation; KEEP preserves the original parsed text exactly."""
    _explanation(original_explanation)
    answer = _review(reply, ("decision", "explanation"))
    return original_explanation if answer["decision"] == "KEEP" else answer["explanation"]


def review_t1(reply: Union[str, Dict[str, Any]], original_reply: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Select complete T1; caller validates its evidence, geometry and policy."""
    original = _parsed(original_reply)
    answer = _review(reply, ("decision", "explanation", "t1"))
    if answer["decision"] == "KEEP":
        return original
    if not isinstance(answer["t1"], dict):
        raise ValueError("T1 REWRITE must supply a complete T1 object")
    return {"explanation": answer["explanation"], "answer": answer["t1"]}


def _student_history(tokenizer: Any, messages: Any) -> List[Dict[str, Any]]:
    from castfactory.training.chat_template_compat import (
        canonical_prompt_messages, has_chat_template, render_chat_template_text,
    )

    if not has_chat_template(tokenizer):
        raise ValueError("selected student adapter requires a native chat template")
    history = canonical_prompt_messages(messages)
    for message in history:
        if message["role"] == "assistant":
            if not isinstance(message.get("content"), str):
                raise ValueError("public assistant repair history must be visible text")
            # The selected Qwen templates extract content inside think tags as
            # reasoning and remove it on older turns. These are PUBLIC reports,
            # including real failed/partial attempts, so keep them in content.
            # An explicit empty channel inhibits that extraction and excludes
            # provider-private reasoning_content from student evidence.
            message["reasoning_content"] = ""
    rendered = render_chat_template_text(
        tokenizer, history, add_generation_prompt=True,
        template_kwargs={"enable_thinking": STUDENT_ENABLE_THINKING},
    )
    for message in history:
        if message["role"] == "assistant" and message["content"].strip():
            if message["content"].strip() not in rendered:
                raise ValueError("chat template removed visible public assistant history")
    return history


def student_prompt_tokens(tokenizer: Any, messages: Any) -> Tuple[List[int], str]:
    """Return native prompt IDs and the prefix owned by that prompt.

    Restore this prefix only on decoded text before student parsing. Never add
    it to generated IDs or the response loss mask.
    """
    from castfactory.training.chat_template_compat import (
        detect_prompt_owned_think_prefix, render_chat_template_ids,
    )

    history = _student_history(tokenizer, messages)
    kwargs = {"enable_thinking": STUDENT_ENABLE_THINKING}
    prompt_ids = render_chat_template_ids(
        tokenizer, history, add_generation_prompt=True, template_kwargs=kwargs,
    )
    prefix = detect_prompt_owned_think_prefix(tokenizer, history, template_kwargs=kwargs)
    return prompt_ids, prefix


def student_training_tokens(tokenizer: Any, messages: Any, output: str) -> Tuple[List[int], List[int]]:
    """Encode public student supervision with explicit existing native mode.

    The native template owns normalization, prompt prefix and assistant EOS;
    final_assistant_training_tokens enforces exact prompt/continuation parity.
    Public repair attempts remain in prompt history, outside response loss.
    Template whitespace after native EOS is not generated assistant content;
    supervision ends at that actually serialized EOS, matching serving.
    """
    from castfactory.training.chat_template_compat import final_assistant_training_tokens

    parse_reply(output, "student")
    history = _student_history(tokenizer, messages)
    prompt_ids, response_ids = final_assistant_training_tokens(
        tokenizer, history + [{"role": "assistant", "content": output}],
        template_kwargs={"enable_thinking": STUDENT_ENABLE_THINKING},
    )
    eos = getattr(tokenizer, "eos_token_id", None)
    if type(eos) is not int or eos < 0:
        raise ValueError("selected student adapter requires one explicit native EOS ID")
    try:
        eos_index = response_ids.index(eos)
    except ValueError as exc:
        raise ValueError("native student training response must contain its EOS") from exc
    tail = response_ids[eos_index + 1:]
    if tail:
        tail_text = tokenizer.decode(tail, skip_special_tokens=False,
                                     clean_up_tokenization_spaces=False)
        if tail_text.strip():
            raise ValueError("native student template contains non-whitespace content after EOS")
    return prompt_ids, response_ids[:eos_index + 1]
