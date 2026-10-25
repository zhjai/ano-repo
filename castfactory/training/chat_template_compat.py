from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


PROMPT_OWNED_RESPONSE_PREFIX_KEY = "prompt_owned_response_prefix"


def canonical_sft_messages(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    if "messages" in row:
        raw_messages = row["messages"]
        if isinstance(raw_messages, (str, bytes)) or not isinstance(raw_messages, Sequence):
            raise ValueError("SFT messages must be a sequence of message mappings")
        messages = [_copy_message(message) for message in raw_messages]
    else:
        if "input" not in row or "output" not in row:
            raise ValueError("SFT examples require messages or both input and output")
        messages = [
            {"role": "user", "content": str(row["input"])},
            {"role": "assistant", "content": str(row["output"])},
        ]

    if len(messages) < 2:
        raise ValueError("SFT messages require prompt history and a final assistant message")
    if messages[-1].get("role") != "assistant":
        raise ValueError("SFT messages must end with an assistant message")
    if not _assistant_message_has_content(messages[-1]):
        raise ValueError("SFT examples require a non-empty SFT response")
    return messages


def canonical_prompt_messages(prompt: Any) -> list[dict[str, Any]]:
    if isinstance(prompt, str):
        return [{"role": "user", "content": prompt}]
    if isinstance(prompt, (str, bytes)) or not isinstance(prompt, Sequence):
        raise ValueError("prompt must be text or a sequence of message mappings")
    messages = [_copy_message(message) for message in prompt]
    if not messages:
        raise ValueError("prompt messages must not be empty")
    return messages


def has_chat_template(tokenizer: Any) -> bool:
    return bool(
        getattr(tokenizer, "chat_template", None)
        and callable(getattr(tokenizer, "apply_chat_template", None))
    )


def render_chat_template_text(
    tokenizer: Any,
    messages: Sequence[Mapping[str, Any]],
    *,
    add_generation_prompt: bool,
    template_kwargs: Mapping[str, Any] | None = None,
) -> str:
    rendered = tokenizer.apply_chat_template(
        list(messages),
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
        **dict(template_kwargs or {}),
    )
    if not isinstance(rendered, str):
        raise TypeError("tokenizer.apply_chat_template() must return text when tokenize=False")
    return rendered


def render_chat_template_ids(
    tokenizer: Any,
    messages: Sequence[Mapping[str, Any]],
    *,
    add_generation_prompt: bool,
    template_kwargs: Mapping[str, Any] | None = None,
) -> list[int]:
    encoded = tokenizer.apply_chat_template(
        list(messages),
        tokenize=True,
        add_generation_prompt=add_generation_prompt,
        **dict(template_kwargs or {}),
    )
    return _input_ids(encoded)


def final_assistant_training_tokens(
    tokenizer: Any,
    messages: Sequence[Mapping[str, Any]],
    *,
    template_kwargs: Mapping[str, Any] | None = None,
) -> tuple[list[int], list[int]]:
    kwargs = dict(template_kwargs or {})
    if "enable_thinking" not in kwargs:
        chat_template = getattr(tokenizer, "chat_template", "") or ""
        if "enable_thinking" in chat_template:
            last_message = messages[-1] if messages else {}
            content = str(last_message.get("content") or "")
            reasoning = str(last_message.get("reasoning_content") or "")
            if not reasoning and "<think>" not in content:
                kwargs["enable_thinking"] = False

    prompt_ids = render_chat_template_ids(
        tokenizer,
        messages[:-1],
        add_generation_prompt=True,
        template_kwargs=kwargs,
    )
    full_ids = render_chat_template_ids(
        tokenizer,
        messages,
        add_generation_prompt=False,
        template_kwargs=kwargs,
    )
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError(
            "chat template generation prompt tokens must be an exact prefix of the "
            "complete SFT conversation"
        )
    response_ids = full_ids[len(prompt_ids) :]
    if not response_ids:
        raise ValueError("chat template produced no assistant response tokens")
    return prompt_ids, response_ids


def detect_prompt_owned_think_prefix(
    tokenizer: Any,
    prompt: Any,
    *,
    template_kwargs: Mapping[str, Any] | None = None,
) -> str:
    if not has_chat_template(tokenizer):
        return ""
    rendered = render_chat_template_text(
        tokenizer,
        canonical_prompt_messages(prompt),
        add_generation_prompt=True,
        template_kwargs=template_kwargs,
    )
    return prompt_owned_think_prefix_from_rendered(rendered)


def prompt_owned_think_prefix_from_rendered(rendered: str) -> str:
    lowered = rendered.lower()
    open_index = lowered.rfind("<think>")
    close_index = lowered.rfind("</think>")
    if open_index < 0 or open_index < close_index:
        return ""
    suffix = rendered[open_index:]
    if suffix[len("<think>") :].strip():
        return ""
    return suffix


def restore_prompt_owned_think_prefix(raw_output: str, prefix: str) -> str:
    if prefix and (
        not prefix.lower().startswith("<think>")
        or prefix[len("<think>") :].strip()
    ):
        raise ValueError("prompt-owned think prefix must contain only the opening tag and whitespace")
    if prefix and not raw_output.lstrip().lower().startswith("<think>"):
        return prefix + raw_output
    return raw_output


def detect_consistent_prompt_owned_think_prefix(
    tokenizer: Any,
    rows: Sequence[Mapping[str, Any]],
) -> str:
    detected: str | None = None
    for row in rows:
        if "prompt" not in row:
            raise ValueError("RL rows require prompt for chat-template prefix detection")
        prefix = detect_prompt_owned_think_prefix(tokenizer, row["prompt"])
        if detected is None:
            detected = prefix
        elif prefix != detected:
            raise ValueError(
                "inconsistent prompt-owned response prefixes across RL rows"
            )
    return detected or ""


def _copy_message(message: Any) -> dict[str, Any]:
    if not isinstance(message, Mapping):
        raise ValueError("each chat message must be a mapping")
    copied = dict(message)
    role = copied.get("role")
    if not isinstance(role, str) or not role:
        raise ValueError("each chat message requires a non-empty role")
    return copied


def _assistant_message_has_content(message: Mapping[str, Any]) -> bool:
    for key in ("content", "reasoning_content"):
        value = message.get(key)
        if isinstance(value, str) and value.strip():
            return True
    return False


def _input_ids(encoded: Any) -> list[int]:
    if isinstance(encoded, Mapping):
        if "input_ids" not in encoded:
            raise TypeError("tokenized chat template result is missing input_ids")
        encoded = encoded["input_ids"]
    elif hasattr(encoded, "input_ids"):
        encoded = encoded.input_ids
    if hasattr(encoded, "tolist"):
        encoded = encoded.tolist()
    if hasattr(encoded, "ids"):
        encoded = encoded.ids
    if (
        isinstance(encoded, Sequence)
        and encoded
        and not isinstance(encoded, (str, bytes))
        and isinstance(encoded[0], Sequence)
    ):
        if len(encoded) != 1:
            raise TypeError("tokenized chat template must contain exactly one sequence")
        encoded = encoded[0]
    if isinstance(encoded, (str, bytes)) or not isinstance(encoded, Sequence):
        raise TypeError("tokenized chat template must return a token id sequence")
    return [int(token_id) for token_id in encoded]
