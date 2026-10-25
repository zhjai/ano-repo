"""Versioned boundary between public teacher summaries and student tokens."""
from __future__ import annotations

import copy
import hashlib
import re
from pathlib import Path

from castfactory.training.chat_template_compat import (
    detect_prompt_owned_think_prefix,
    final_assistant_training_tokens,
    render_chat_template_ids,
    restore_prompt_owned_think_prefix,
)

STUDENT_SERIALIZATION_VERSION = "student-think-v1"
V4_STUDENT_SERIALIZATION_VERSION = "merged-t1-answer-v1"
JSON_ROWS_STUDENT_SERIALIZATION_VERSION = "agentic-json-rows-v1"
COMPACT_STUDENT_SERIALIZATION_VERSION = "agentic-compact-markdown-v1"
STEP_TARGET_STUDENT_SERIALIZATION_VERSION = "agentic-compact-markdown-v2"
COMPACT_STUDENT_VERSIONS = (COMPACT_STUDENT_SERIALIZATION_VERSION, STEP_TARGET_STUDENT_SERIALIZATION_VERSION)
_TEACHER = re.compile(
    r"\s*<think_summary>(.*?)</think_summary>\s*<payload>(.*?)</payload>"
    r"\s*<metadata>(.*?)</metadata>\s*", re.S,
)


def teacher_to_student(content: str, *, profile_version=None) -> str:
    """Convert only the visible teacher envelope; preserve raw JSON spellings."""
    if profile_version in (JSON_ROWS_STUDENT_SERIALIZATION_VERSION, *COMPACT_STUDENT_VERSIONS):
        raise ValueError("Converted profile has no live privileged teacher conversion")
    if _v4(profile_version):
        from castfactory.agentic.distillation_protocol import parse_reply, render_reply
        parsed = parse_reply(content, "teacher")
        return render_reply(parsed["explanation"], parsed["answer"], "student")
    from castfactory.agentic.full_agent_v3 import parse_visible
    match = _TEACHER.fullmatch(content)
    if match is None:
        raise ValueError("Expected a complete visible teacher summary envelope")
    parse_visible(content)
    summary, payload, metadata = match.groups()
    return ("<think>\n" + summary.strip() + "\n</think>\n\n<payload>" + payload
            + "</payload><metadata>" + metadata + "</metadata>")


def student_template_messages(messages, *, final_assistant=False):
    """Retain public retry explanations as ordinary historical assistant text.

    Historical malformed/truncated answers remain malformed; only tag names
    change. No missing closing tag, forecast value or metadata is invented.
    Provider reasoning fields are deliberately excluded from template input.
    """
    result = copy.deepcopy(list(messages))
    for index, message in enumerate(result):
        message.pop("reasoning_content", None)
        message.pop("reasoning", None)
        if message.get("role") != "assistant":
            continue
        content = str(message.get("content") or "")
        if final_assistant and index == len(result) - 1:
            if _TEACHER.fullmatch(content):
                content = teacher_to_student(content)
        else:
            if _TEACHER.fullmatch(content):
                try:
                    content = teacher_to_student(content)
                except ValueError:
                    pass
            elif re.fullmatch(r"\s*<think>.*?</think>\s*<payload>.*?</payload>\s*<metadata>.*?</metadata>\s*", content, re.S):
                try:
                    content = student_content(content)
                except ValueError:
                    pass
            content = (content.replace("<think_summary>", "<public_summary>")
                       .replace("</think_summary>", "</public_summary>")
                       .replace("<think>", "<public_summary>")
                       .replace("</think>", "</public_summary>"))
        message["content"] = content
    return result


def _v4(profile_version):
    if profile_version not in (None, "full-agent-v3", V4_STUDENT_SERIALIZATION_VERSION,
                               JSON_ROWS_STUDENT_SERIALIZATION_VERSION, *COMPACT_STUDENT_VERSIONS):
        raise ValueError("Unknown student serialization profile: " + str(profile_version))
    return profile_version in (V4_STUDENT_SERIALIZATION_VERSION, JSON_ROWS_STUDENT_SERIALIZATION_VERSION,
                               *COMPACT_STUDENT_VERSIONS)


def student_content(content, *, profile_version=None):
    """Canonicalize visible teacher or student text without touching JSON."""
    if profile_version in COMPACT_STUDENT_VERSIONS:
        from castfactory.agentic.compact_protocol import parse_envelope
        parse_envelope(content, "student")
        return content
    if _v4(profile_version):
        from castfactory.agentic.distillation_protocol import parse_reply
        parse_reply(content, "student")
        return content
    if _TEACHER.fullmatch(content):
        return teacher_to_student(content)
    match = re.fullmatch(
        r"\s*<think>(.*?)</think>\s*<payload>(.*?)</payload>\s*<metadata>(.*?)</metadata>\s*",
        content, re.S)
    if match is None:
        raise ValueError("Expected a complete visible teacher or student envelope")
    summary, payload, metadata = match.groups()
    return teacher_to_student("<think_summary>" + summary.strip() + "</think_summary>"
                              "<payload>" + payload + "</payload><metadata>" + metadata + "</metadata>")


def student_messages(role, state, profile, original_messages=None, *, profile_version=None):
    if profile_version in COMPACT_STUDENT_VERSIONS:
        from castfactory.agentic.compact_prompts import messages_for
        variables = copy.deepcopy(state)
        case = variables.pop("public_json_case", None)
        if case is None:
            raise ValueError("Compact student messages require out-of-band public_json_case")
        base = messages_for(role, variables, case, protocol_version=profile_version)
        if original_messages is not None:
            base.extend(copy.deepcopy(list(original_messages)[2:]))
        return base
    if profile_version == JSON_ROWS_STUDENT_SERIALIZATION_VERSION:
        from castfactory.agentic.json_rows_prompts import messages_for
        variables = copy.deepcopy(state)
        case = variables.pop("public_json_case", None)
        if case is None and original_messages is not None:
            from castfactory.agentic.json_rows_protocol import parse_object
            case = parse_object(original_messages[1]["content"])["case"]
        if case is None:
            raise ValueError("JSON-row student messages require public_json_case")
        base = messages_for(role, variables, case)
        if original_messages is not None:
            base.extend(copy.deepcopy(list(original_messages)[2:]))
        return base
    if _v4(profile_version):
        from castfactory.agentic.distillation_protocol import messages_for
        base = messages_for(role, state, "student")
        if original_messages is not None:
            base.extend(copy.deepcopy(list(original_messages)[2:]))
        return base
    from castfactory.agentic.full_agent_v3 import messages_for
    base = messages_for(role, state, profile, response_format="student")
    if original_messages is not None:
        history = copy.deepcopy(list(original_messages)[2:])
        index = 0
        while index < len(history):
            message = history[index]
            # A teacher's hidden-budget exhaustion is not a student action.
            empty_budget_retry = (
                message.get("role") == "assistant" and not str(message.get("content") or "").strip()
                and index + 1 < len(history) and history[index + 1].get("role") == "user"
                and "Generation must terminate normally, not at token budget" in str(history[index + 1].get("content")))
            if empty_budget_retry:
                index += 2
                continue
            base.append(message)
            index += 1
    return student_template_messages(base)


def student_prompt(tokenizer, messages, *, profile_version=None):
    if _v4(profile_version):
        from castfactory.agentic.distillation_protocol import student_prompt_tokens
        return student_prompt_tokens(tokenizer, messages)
    normalized = student_template_messages(messages)
    kwargs = {"enable_thinking": True}
    ids = render_chat_template_ids(tokenizer, normalized, add_generation_prompt=True,
                                   template_kwargs=kwargs)
    prefix = detect_prompt_owned_think_prefix(tokenizer, normalized, template_kwargs=kwargs)
    return ids, prefix


def student_training_tokens(tokenizer, messages, output, *, profile_version=None):
    if profile_version in COMPACT_STUDENT_VERSIONS:
        from castfactory.agentic import distillation_protocol as protocol
        from castfactory.agentic.compact_protocol import parse_envelope
        parse_envelope(output, "student")
        history = protocol._student_history(tokenizer, messages)
        prompt, response = final_assistant_training_tokens(
            tokenizer, history + [{"role": "assistant", "content": output}],
            template_kwargs={"enable_thinking": protocol.STUDENT_ENABLE_THINKING})
        eos = getattr(tokenizer, "eos_token_id", None)
        if type(eos) is not int or eos < 0:
            raise ValueError("selected student adapter requires one explicit native EOS ID")
        try:
            eos_index = response.index(eos)
        except ValueError as error:
            raise ValueError("native student training response must contain its EOS") from error
        if response[eos_index + 1:]:
            tail = tokenizer.decode(response[eos_index + 1:], skip_special_tokens=False,
                                    clean_up_tokenization_spaces=False)
            if tail.strip():
                raise ValueError("native student template contains non-whitespace content after EOS")
        return prompt, response[:eos_index + 1]
    if _v4(profile_version):
        from castfactory.agentic.distillation_protocol import student_training_tokens as v4_tokens
        return v4_tokens(tokenizer, messages, output)
    match = re.fullmatch(r"<think>\n(.*?)\n</think>\n\n(<payload>.*</metadata>)", output, re.S)
    if match is None:
        raise ValueError("Student training requires canonical student think output")
    reconstructed = ("<think_summary>" + match.group(1) + "</think_summary>" + match.group(2))
    if teacher_to_student(reconstructed) != output:
        raise ValueError("Student output is not canonical")
    normalized = student_template_messages(
        list(messages) + [{"role": "assistant", "content": output}], final_assistant=True)
    prompt, response = final_assistant_training_tokens(
        tokenizer, normalized, template_kwargs={"enable_thinking": True})
    actual_prompt, prefix = student_prompt(tokenizer, messages)
    if prompt != actual_prompt:
        raise ValueError("Student SFT/serving prompt token mismatch")
    return prompt, response


def restore_student_content(continuation, prefix):
    """Restore prompt ownership for parsing only, never for response IDs/loss."""
    restore_prompt_owned_think_prefix("", prefix)
    return prefix + continuation


def student_serialization_identity(tokenizer, *, profile_version=None):
    from castfactory.agentic.full_agent_v3 import digest
    settings = {name: getattr(tokenizer, name, None) for name in (
        "bos_token", "bos_token_id", "eos_token", "eos_token_id",
        "pad_token", "pad_token_id", "unk_token", "unk_token_id",
        "padding_side", "truncation_side", "model_max_length",
        "clean_up_tokenization_spaces", "add_bos_token", "add_eos_token",
    )}
    settings["all_special_tokens"] = list(getattr(tokenizer, "all_special_tokens", []))
    settings["all_special_ids"] = list(getattr(tokenizer, "all_special_ids", []))
    getter = getattr(tokenizer, "get_chat_template", None)
    template = getter() if callable(getter) else getattr(tokenizer, "chat_template", None)
    if _v4(profile_version):
        return {"version": profile_version,
                "enable_thinking": True, "chat_template": template,
                "tokenizer_settings": settings}
    base = Path(__file__).parent
    return {"version": STUDENT_SERIALIZATION_VERSION,
            "chat_template_sha256": digest(template), "tokenizer_settings": settings,
            "source_sha256": {name: hashlib.sha256((base / name).read_bytes()).hexdigest()
                              for name in ("full_agent_serialization.py", "chat_template_compat.py")}}


serialization_identity = student_serialization_identity
