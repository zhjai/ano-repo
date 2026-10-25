from __future__ import annotations

from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import re
from typing import Any, Dict, Iterable, List, Mapping

import numpy as np

from castfactory.time_r1.prediction_values import extract_prediction_values
from castfactory.training.chat_template_compat import (
    canonical_sft_messages,
    has_chat_template,
    render_chat_template_ids,
    render_chat_template_text,
)


DEFAULT_STRUCTURE_MARKERS = (
    "<think>",
    "</think>",
    "<answer>",
    "</answer>",
    "Regime:",
    "Future evidence:",
    "Calibration:",
    "Decision:",
    "```markdown",
    "```",
    "index | value",
)

_MISSING_WEIGHT_CONVERSIONS = object()


def _find_token_subsequence(tokens: list[int], needle: list[int]) -> int | None:
    if not needle or len(needle) > len(tokens):
        return None
    width = len(needle)
    for start in range(len(tokens) - width + 1):
        if tokens[start : start + width] == needle:
            return start
    return None


class TransformersSFTBackend:
    def __init__(
        self,
        model=None,
        tokenizer=None,
        trainer_cls=None,
        training_args_cls=None,
        data_collator_cls=None,
        max_length: int | None = None,
        max_seq_length: int | None = None,
        max_prompt_length: int = 4096,
        max_response_length: int = 8192,
        truncation_side: str = "left",
        response_only_logits: bool = False,
        response_loss_mode: str = "all",
        structure_marker_loss: Mapping[str, Any] | None = None,
        compute_forecast_metrics: bool = False,
        forecast_metric_length: int = 96,
        save_original_format: bool | None = None,
        **default_train_args,
    ):
        if max_length is not None and max_seq_length is not None and max_length != max_seq_length:
            raise ValueError("max_length and max_seq_length must match when both are set")
        if max_seq_length is not None:
            max_length = max_seq_length
        if max_length is not None and max_length <= 0:
            raise ValueError("max_length must be positive")
        if max_prompt_length <= 0:
            raise ValueError("max_prompt_length must be positive")
        if max_response_length <= 0:
            raise ValueError("max_response_length must be positive")
        if truncation_side not in {"left", "right"}:
            raise ValueError("truncation_side must be 'left' or 'right'")
        if forecast_metric_length <= 0:
            raise ValueError("forecast_metric_length must be positive")
        self.model = model
        self.tokenizer = tokenizer
        self.trainer_cls = trainer_cls
        self.training_args_cls = training_args_cls
        self.data_collator_cls = data_collator_cls
        self.max_prompt_length = max_prompt_length
        self.max_response_length = max_response_length
        self.max_length = max_length or (max_prompt_length + max_response_length + 1)
        if self.max_length <= 0:
            raise ValueError("max_length must be positive")
        self.truncation_side = truncation_side
        self.response_only_logits = bool(response_only_logits)
        if response_loss_mode not in {"all", "forecast_values_masked", "forecast_numbers_masked"}:
            raise ValueError(
                "response_loss_mode must be 'all', 'forecast_values_masked', or 'forecast_numbers_masked'"
            )
        self.response_loss_mode = response_loss_mode
        marker_loss_config = dict(structure_marker_loss or {})
        unknown_marker_loss_keys = set(marker_loss_config) - {"enable", "weight"}
        if unknown_marker_loss_keys:
            raise ValueError(
                "structure_marker_loss supports only enable and weight; unknown keys: "
                + ", ".join(sorted(unknown_marker_loss_keys))
            )
        self.structure_marker_loss_enabled = bool(marker_loss_config.get("enable", False))
        self.structure_marker_loss_weight = float(marker_loss_config.get("weight", 2.0))
        if (
            not math.isfinite(self.structure_marker_loss_weight)
            or self.structure_marker_loss_weight < 1.0
        ):
            raise ValueError("structure_marker_loss.weight must be finite and at least 1.0")
        if self.structure_marker_loss_enabled and not self.response_only_logits:
            raise ValueError(
                "structure_marker_loss requires training.backend.response_only_logits=true"
            )
        self.compute_forecast_metrics = bool(compute_forecast_metrics)
        self.forecast_metric_length = int(forecast_metric_length)
        self.save_original_format = save_original_format
        self.default_train_args = dict(default_train_args)

    @contextmanager
    def _checkpoint_save_format(self, model):
        if self.save_original_format is not False:
            yield
            return

        previous = getattr(model, "_weight_conversions", _MISSING_WEIGHT_CONVERSIONS)
        model._weight_conversions = []
        try:
            yield
        finally:
            if previous is _MISSING_WEIGHT_CONVERSIONS:
                delattr(model, "_weight_conversions")
            else:
                model._weight_conversions = previous

    def prepare_dataset(self, train_dataset: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
        prepared = []
        for row in train_dataset:
            messages = canonical_sft_messages(row)
            tokenizer = self._require_tokenizer()
            template_kwargs = self._reasoning_v2_template_kwargs(row)
            if template_kwargs and not has_chat_template(tokenizer):
                raise ValueError("Reasoning compact v2 requires a native chat-template tokenizer")
            if self.response_loss_mode == "forecast_numbers_masked" and not has_chat_template(tokenizer):
                raise ValueError("forecast_numbers_masked requires a chat-template tokenizer")
            if has_chat_template(tokenizer):
                messages = self._without_trailing_assistant_eos(messages)
                # Render the complete conversation once.  Besides avoiding a
                # duplicate template call, this gives the offset-based mask
                # the exact text whose ids become the training sequence.
                prompt_ids = render_chat_template_ids(
                    tokenizer,
                    messages[:-1],
                    add_generation_prompt=True,
                    template_kwargs=template_kwargs,
                )
                original_prompt_length = len(prompt_ids)
                full_rendered_text = render_chat_template_text(
                    tokenizer,
                    messages,
                    add_generation_prompt=False,
                    template_kwargs=template_kwargs,
                )
                full_ids = self._encode_text(full_rendered_text, add_special_tokens=False)
                if full_ids[: len(prompt_ids)] != prompt_ids:
                    prompt_text = render_chat_template_text(
                        tokenizer, messages[:-1], add_generation_prompt=True,
                        template_kwargs=template_kwargs,
                    )
                    original_prompt_length = self._merged_prompt_boundary(
                        prompt_text, full_rendered_text, prompt_ids, full_ids,
                    )
                    prompt_ids = full_ids[:original_prompt_length]
                full_ids, full_rendered_text = self._apply_selected_response_boundary(
                    row, full_ids, original_prompt_length, full_rendered_text,
                )
                response_ids = full_ids[len(prompt_ids) :]
                if not response_ids:
                    raise ValueError("chat template produced no assistant response tokens")
                prompt_ids = self._truncate_ids(prompt_ids, self.max_prompt_length)
                if len(response_ids) > self.max_response_length:
                    raise ValueError(
                        "SFT response tokens exceed max_response_length; increase "
                        "max_response_length so labels are not truncated"
                    )
            else:
                prompt, output = self._raw_single_turn_text(messages)
                prompt_ids = self._tokenize_text(prompt, self.max_prompt_length)
                response_ids = self._raw_response_ids(output)
            input_ids = prompt_ids + response_ids
            if len(input_ids) > self.max_length:
                overflow = len(input_ids) - self.max_length
                if overflow >= len(prompt_ids):
                    raise ValueError(
                        "SFT response tokens exceed max_length; increase max_length or "
                        "max_response_length so labels are not truncated"
                    )
                prompt_ids = self._truncate_ids(prompt_ids, len(prompt_ids) - overflow)
                input_ids = prompt_ids + response_ids
            labels = [-100] * len(prompt_ids) + list(response_ids)
            if self.response_loss_mode == "forecast_values_masked":
                labels = self._mask_forecast_value_labels(
                    labels,
                    response_ids=response_ids,
                    response_text=str(messages[-1].get("content", "")),
                    response_offset=len(prompt_ids),
                    full_rendered_text=full_rendered_text,
                    original_prompt_length=original_prompt_length,
                )
            elif self.response_loss_mode == "forecast_numbers_masked":
                labels = self._mask_forecast_numeric_labels(
                    labels,
                    response_ids=response_ids,
                    response_offset=len(prompt_ids),
                    full_rendered_text=full_rendered_text,
                    original_prompt_length=original_prompt_length,
                )
            if row.get("metadata", {}).get("validation_supervision") == "answer_only":
                if (row["metadata"].get("split") != "validation"
                        or self.response_loss_mode != "all" or not has_chat_template(tokenizer)):
                    raise ValueError("answer_only validation requires validation split, all loss mode and a chat tokenizer")
                labels = self._mask_validation_answer_labels(
                    labels, response_ids=response_ids, response_offset=len(prompt_ids),
                    full_rendered_text=full_rendered_text,
                    original_prompt_length=original_prompt_length,
                    response_text=str(messages[-1].get("content", "")),
                    horizon=row["metadata"].get("prediction_length"),
                    answer_format=row["metadata"].get("answer_format"),
                    protocol_version=row["metadata"].get("protocol_version"),
                )
            prepared.append(
                {
                    "input_ids": input_ids,
                    "attention_mask": [1] * len(input_ids),
                    "labels": labels,
                    "metadata": dict(row.get("metadata", {})),
                }
            )
        return prepared

    def _reasoning_v2_template_kwargs(self, row):
        from castfactory.time_r1.forecast_answer_contract import (
            STEP_TARGET_ANSWER_FORMAT, STEP_TARGET_PROTOCOL_VERSION,
        )

        metadata = row.get("metadata", {})
        markers = (metadata.get("answer_format"), metadata.get("protocol_version"))
        expected = (STEP_TARGET_ANSWER_FORMAT, STEP_TARGET_PROTOCOL_VERSION)
        if markers[0] == expected[0] or markers[1] == expected[1]:
            if markers != expected:
                raise ValueError("Reasoning compact v2 requires matching answer_format and protocol_version")
            return {"enable_thinking": True}
        return None

    def _apply_selected_response_boundary(self, row, full_ids, prompt_length, full_text):
        """End explicit Reasoning v2 supervision at its serialized native EOS.

        Older profiles retain their complete template encoding. Keep trimmed
        text and IDs identical so answer-only offsets describe the actual
        sequence, and reject content beyond EOS rather than silently dropping it.
        """
        if not self._reasoning_v2_template_kwargs(row):
            return full_ids, full_text
        tokenizer = self._require_tokenizer()
        eos = getattr(tokenizer, "eos_token_id", None)
        if type(eos) is not int or eos < 0:
            raise ValueError("Reasoning compact v2 requires one explicit native EOS ID")
        try:
            stop = full_ids.index(eos, prompt_length) + 1
        except ValueError as error:
            raise ValueError("Reasoning compact v2 response must contain native EOS") from error
        tail = full_ids[stop:]
        if not tail:
            return full_ids, full_text
        tail_text = tokenizer.decode(tail, skip_special_tokens=False,
                                     clean_up_tokenization_spaces=False)
        if tail_text.strip():
            raise ValueError("Native Reasoning v2 template contains non-whitespace content after EOS")
        trimmed_text = full_text.rstrip()
        trimmed_ids = full_ids[:stop]
        if self._encode_text(trimmed_text, add_special_tokens=False) != trimmed_ids:
            raise ValueError("Native Reasoning v2 EOS boundary differs from rendered text")
        return trimmed_ids, trimmed_text

    def _mask_validation_answer_labels(self, labels, *, response_ids, response_offset,
                                       full_rendered_text, original_prompt_length,
                                       response_text, horizon, answer_format=None,
                                       protocol_version=None):
        """Only score a real answer target, never fabricate/score empty reasoning.

        Native templates may insert an empty think block for a target without
        an explanation. Its tokens and EOS are excluded from this validation
        diagnostic; training rows keep their normal full-response loss.
        """
        from castfactory.agentic.distillation_protocol import _pairs, _constant, validate_values
        from castfactory.time_r1.forecast_answer_contract import (
            ROWS_ANSWER_FORMAT, ROWS_PROTOCOL_VERSION, validate_row_forecast_object,
            MARKDOWN_ANSWER_FORMAT, COMPACT_PROTOCOL_VERSION, MARKDOWN_PROFILES,
        )

        match = re.fullmatch(r"<answer>(.*?)</answer>", response_text, re.S)
        if match is None or not isinstance(horizon, int) or isinstance(horizon, bool) or horizon <= 0:
            raise ValueError("answer_only validation needs a complete answer and positive horizon")
        if answer_format in MARKDOWN_PROFILES or protocol_version in MARKDOWN_PROFILES.values():
            if answer_format not in MARKDOWN_PROFILES or MARKDOWN_PROFILES[answer_format] != protocol_version:
                raise ValueError("Markdown validation requires matching answer_format and protocol_version")
            if horizon not in (24, 96):
                raise ValueError("Shared load Markdown requires H24 or H96")
            from castfactory.time_r1.markdown_table import parse_position_table
            parse_position_table(match[1], range(horizon), answer_format=answer_format)
        elif answer_format == ROWS_ANSWER_FORMAT or protocol_version == ROWS_PROTOCOL_VERSION:
            if answer_format != ROWS_ANSWER_FORMAT or protocol_version != ROWS_PROTOCOL_VERSION:
                raise ValueError("rowJSON validation requires matching answer_format and protocol_version")
            answer = json.loads(match[1], object_pairs_hook=_pairs, parse_constant=_constant)
            validate_row_forecast_object(answer, horizon)
        else:
            answer = json.loads(match[1], object_pairs_hook=_pairs, parse_constant=_constant)
            if not isinstance(answer, dict) or set(answer) != {"values"}:
                raise ValueError("answer_only validation target must contain exactly values")
            validate_values(answer["values"], horizon)
        start = full_rendered_text.rfind(response_text)
        if start < 0:
            raise ValueError("Validation answer is absent from native rendering")
        stop = start + len(response_text)
        encoded = self._encode_with_offsets(full_rendered_text)
        if encoded is None or encoded[0][original_prompt_length:] != list(response_ids):
            raise ValueError("Validation offsets differ from actual response tokens")
        masked = [-100] * len(labels)
        for token_index in range(original_prompt_length, len(encoded[1])):
            left, right = encoded[1][token_index]
            if right > start and left < stop:
                label_index = token_index - original_prompt_length + response_offset
                masked[label_index] = labels[label_index]
        if all(value == -100 for value in masked):
            raise ValueError("Validation answer selected no loss tokens")
        return masked

    def _merged_prompt_boundary(self, prompt_text, full_text, prompt_ids, full_ids):
        error = (
            "chat template generation prompt tokens must be an exact prefix of the "
            "complete SFT conversation"
        )
        if not full_text.startswith(prompt_text):
            raise ValueError(error)
        encoded = self._encode_with_offsets(full_text)
        if encoded is None or encoded[0] != full_ids or len(encoded[1]) != len(full_ids):
            raise ValueError(error)
        offsets = encoded[1]
        boundary = len(prompt_text)
        crossing = [i for i, (start, end) in enumerate(offsets) if start < boundary < end]
        if len(crossing) != 1:
            raise ValueError(error)
        index = crossing[0]
        if index != len(prompt_ids) - 1 or full_ids[:index] != prompt_ids[:index]:
            raise ValueError(error)
        # Keep the exact full-conversation encoding. A token spanning the
        # boundary (e.g. Qwen's double newline) includes response text and is supervised.
        return index

    def _mask_forecast_value_labels(
        self,
        labels: list[int],
        *,
        response_ids: list[int],
        response_text: str,
        response_offset: int,
        full_rendered_text: str | None = None,
        prompt_rendered_text: str | None = None,
        original_prompt_length: int | None = None,
    ) -> list[int]:
        """Mask only values inside ``<forecast>`` while retaining its envelope.

        The forecast span is deliberately present in the target so the model
        learns to continue through the numeric block and emit EOS in the right
        place.  We locate the exact tokenizer subsequence instead of guessing
        token positions from characters; a missing match is a hard contract
        failure and must be fixed before training.
        """

        match = re.search(
            r"<forecast>\s*(.*?)\s*</forecast>",
            response_text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if match is None:
            raise ValueError(
                "forecast_values_masked requires a <forecast>...</forecast> response"
            )
        value_start, value_end = match.span(1)

        # A BPE tokenizer can merge the first/last forecast token with the
        # surrounding whitespace or punctuation.  Tokenizing the value span
        # by itself and searching for that sequence is therefore not stable
        # (notably for Qwen tokenizers).  Prefer the tokenizer's character
        # offsets on the complete response, then retain the subsequence path
        # for small/test tokenizers that do not expose offsets.
        content_ids = list(response_ids)
        eos_ids = self._eos_token_ids()
        if eos_ids and content_ids[-len(eos_ids) :] == eos_ids:
            content_ids = content_ids[: -len(eos_ids)]
        offset_result = None
        offset_base = 0
        offset_prompt_tokens = original_prompt_length
        if full_rendered_text is not None:
            offset_result = self._encode_with_offsets(full_rendered_text)
            # The template may normalize the assistant envelope (so the raw
            # response string is not a literal substring).  Locate the
            # forecast span again in the rendered conversation instead.
            full_match = re.search(
                r"<forecast>\s*(.*?)\s*</forecast>",
                full_rendered_text,
                flags=re.IGNORECASE | re.DOTALL,
            )
            if full_match is not None:
                offset_base = int(full_match.start(1))
                value_start = 0
                value_end = len(full_match.group(1))
            else:
                offset_result = None
        if offset_result is None:
            offset_result = self._encode_with_offsets(response_text)
            offset_base = 0
            offset_prompt_tokens = None
        selected: list[int] = []
        if offset_result is not None:
            offset_ids, offsets = offset_result
            if offset_prompt_tokens is not None and len(offset_ids) == offset_prompt_tokens + len(response_ids):
                selected = [
                    index
                    for index, (start_offset, end_offset) in enumerate(offsets)
                    if int(end_offset) > offset_base + value_start
                    and int(start_offset) < offset_base + value_end
                ]
                selected = [
                    index - offset_prompt_tokens + response_offset
                    for index in selected
                ]
            elif offset_prompt_tokens is None and offset_ids == content_ids:
                selected = [
                    index + response_offset
                    for index, (start_offset, end_offset) in enumerate(offsets)
                    if int(end_offset) > value_start and int(start_offset) < value_end
                ]
        if selected:
            absolute_indices = selected
        else:
            value_text = match.group(1)
            value_ids = self._encode_text(value_text, add_special_tokens=False)
            if not value_ids:
                raise ValueError("forecast value span tokenized to an empty sequence")
            start = _find_token_subsequence(response_ids, value_ids)
            if start is None:
                raise ValueError(
                    "could not locate tokenized forecast values in the assistant response; "
                    "run the template/mask preflight before SFT"
                )
            absolute_indices = list(
                range(response_offset + start, response_offset + start + len(value_ids))
            )
        masked = list(labels)
        for index in absolute_indices:
            masked[index] = -100
        return masked

    def _mask_forecast_numeric_labels(
        self,
        labels: list[int],
        *,
        response_ids: list[int],
        response_offset: int,
        full_rendered_text: str,
        original_prompt_length: int,
    ) -> list[int]:
        """Keep JSON array punctuation/EOS trainable while hiding numeric values."""

        match = re.search(
            r"<forecast>\s*(.*?)\s*</forecast>",
            full_rendered_text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if match is None:
            raise ValueError("forecast_numbers_masked requires a forecast block")
        forecast_text = match.group(1)
        try:
            values = json.loads(forecast_text)
        except json.JSONDecodeError as exc:
            raise ValueError("forecast_numbers_masked requires a JSON array") from exc
        if not isinstance(values, list) or not values or any(
            isinstance(value, bool) or not isinstance(value, (int, float)) for value in values
        ):
            raise ValueError("forecast_numbers_masked requires numeric JSON array values")
        numeric_pattern = re.compile(r"-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?")
        spans = [
            (match.start(1) + item.start(), match.start(1) + item.end())
            for item in numeric_pattern.finditer(forecast_text)
        ]
        if len(spans) != len(values):
            raise ValueError("forecast numeric spans do not match JSON array length")
        encoded = self._encode_with_offsets(full_rendered_text)
        if encoded is None:
            raise ValueError("forecast_numbers_masked requires tokenizer character offsets")
        offset_ids, offsets = encoded
        if offset_ids[original_prompt_length:] != list(response_ids):
            raise ValueError("forecast numeric offsets differ from training response tokens")
        masked = list(labels)
        span_index = 0
        selected = 0
        for token_index in range(original_prompt_length, len(offsets)):
            start, end = offsets[token_index]
            while span_index < len(spans) and spans[span_index][1] <= start:
                span_index += 1
            if span_index == len(spans):
                break
            number_start, number_end = spans[span_index]
            if end > number_start and start < number_end:
                label_index = token_index - original_prompt_length + response_offset
                masked[label_index] = -100
                selected += 1
        if not selected:
            raise ValueError("forecast numeric span selected no response tokens")
        return masked

    def _encode_chat_full_ids(self, full_text: str | None, prompt_text: str | None) -> list[int]:
        """Recover the ids used by the chat template's complete rendering."""

        if full_text is None or prompt_text is None:
            return []
        encoded = self._require_tokenizer()(
            full_text,
            truncation=False,
            max_length=None,
            add_special_tokens=False,
        )
        input_ids = encoded["input_ids"]
        if hasattr(input_ids, "tolist"):
            input_ids = input_ids.tolist()
        return list(input_ids)

    def _encode_with_offsets(self, text: str) -> tuple[list[int], list[tuple[int, int]]] | None:
        """Return complete-text token ids and character offsets when supported."""

        tokenizer = self._require_tokenizer()
        try:
            encoded = tokenizer(
                text,
                truncation=False,
                max_length=None,
                add_special_tokens=False,
                return_offsets_mapping=True,
            )
        except (TypeError, NotImplementedError, ValueError):
            return None
        if not isinstance(encoded, Mapping):
            return None
        input_ids = encoded.get("input_ids")
        offsets = encoded.get("offset_mapping")
        if input_ids is None or offsets is None:
            return None
        if hasattr(input_ids, "tolist"):
            input_ids = input_ids.tolist()
        if hasattr(offsets, "tolist"):
            offsets = offsets.tolist()
        if not isinstance(input_ids, list) or not isinstance(offsets, list):
            return None
        try:
            normalized_offsets = [(int(item[0]), int(item[1])) for item in offsets]
        except (TypeError, ValueError, IndexError):
            return None
        return [int(item) for item in input_ids], normalized_offsets

    def _without_trailing_assistant_eos(
        self,
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        eos_token = getattr(self._require_tokenizer(), "eos_token", None)
        content = messages[-1].get("content")
        if not eos_token or not isinstance(content, str) or not content.endswith(str(eos_token)):
            return messages
        stripped_messages = [dict(message) for message in messages]
        stripped_messages[-1]["content"] = content[: -len(str(eos_token))]
        if not str(stripped_messages[-1]["content"]).strip() and not str(
            stripped_messages[-1].get("reasoning_content", "")
        ).strip():
            raise ValueError("SFT examples require a non-empty SFT response")
        return stripped_messages

    def _raw_single_turn_text(
        self,
        messages: list[dict[str, Any]],
    ) -> tuple[str, str]:
        if (
            len(messages) != 2
            or messages[0].get("role") != "user"
            or messages[1].get("role") != "assistant"
            or not isinstance(messages[0].get("content"), str)
            or not isinstance(messages[1].get("content"), str)
        ):
            raise ValueError(
                "tokenizers without a chat template support only one user message and "
                "one assistant text response"
            )
        return str(messages[0]["content"]), str(messages[1]["content"])

    def _raw_response_ids(self, output: str) -> list[int]:
        response_ids = self._encode_text(output, add_special_tokens=False)
        eos_ids = self._eos_token_ids()
        if eos_ids:
            if response_ids[-len(eos_ids) :] == eos_ids:
                response_ids = response_ids[: -len(eos_ids)]
            response_budget = max(self.max_response_length - len(eos_ids), 0)
            if len(response_ids) > response_budget:
                raise ValueError(
                    "SFT response tokens exceed max_response_length; increase "
                    "max_response_length so labels are not truncated"
                )
            return response_ids + eos_ids
        if len(response_ids) > self.max_response_length:
            raise ValueError(
                "SFT response tokens exceed max_response_length; increase "
                "max_response_length so labels are not truncated"
            )
        return response_ids

    def fit(
        self,
        train_dataset,
        eval_dataset,
        checkpoint_dir: str | Path,
        model=None,
        tokenizer=None,
        train_args: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        if tokenizer is not None:
            self.tokenizer = tokenizer
        model = model or self.model
        if model is None:
            raise ValueError("TransformersSFTBackend.fit requires a model")
        self._ensure_pad_token()
        tokenized_dataset = self.prepare_dataset(train_dataset)
        trainer_dataset = self._training_features(tokenized_dataset)
        trainer_eval_dataset = None
        eval_count = 0
        if eval_dataset is not None:
            tokenized_eval_dataset = self.prepare_dataset(eval_dataset)
            trainer_eval_dataset = self._training_features(tokenized_eval_dataset)
            eval_count = len(tokenized_eval_dataset)
        trainer_cls, training_args_cls, data_collator_cls = self._trainer_types()
        trainer_cls = self._training_trainer_cls(trainer_cls)
        args = dict(self.default_train_args)
        args.update(train_args or {})
        resume_from_checkpoint = args.pop("resume_from_checkpoint", None)
        args.setdefault("output_dir", str(checkpoint_dir))
        args = self._with_available_fsdp_wrap_classes(args, model)
        effective_fsdp_config = args.get("fsdp_config")
        fsdp_wrap_classes = (
            list(effective_fsdp_config.get("transformer_layer_cls_to_wrap", []))
            if isinstance(effective_fsdp_config, Mapping)
            else []
        )
        try:
            training_args = training_args_cls(**args)
        except ImportError as exc:
            if "accelerate" in str(exc):
                raise ImportError(
                    "TransformersSFTBackend requires the CastFactory training extra "
                    "('elecload[train]') with accelerate>=0.26.0."
                ) from exc
            raise
        data_collator = data_collator_cls(
            tokenizer=self._require_tokenizer(),
            model=model,
            padding=True,
        )
        trainer_kwargs = {
            "model": model,
            "args": training_args,
            "train_dataset": trainer_dataset,
            "eval_dataset": trainer_eval_dataset,
            "data_collator": data_collator,
        }
        if self.compute_forecast_metrics:
            trainer_kwargs["compute_metrics"] = self._build_forecast_compute_metrics()
        trainer = trainer_cls(**trainer_kwargs)
        self._initialize_swanlab_run(trainer)
        with self._checkpoint_save_format(model):
            if resume_from_checkpoint is None:
                train_result = trainer.train()
            else:
                train_result = trainer.train(
                    resume_from_checkpoint=resume_from_checkpoint,
                )
            if hasattr(trainer, "save_model"):
                trainer.save_model(str(checkpoint_dir))
        tokenizer = self._require_tokenizer()
        if hasattr(tokenizer, "save_pretrained"):
            tokenizer.save_pretrained(str(checkpoint_dir))
        metrics = getattr(train_result, "metrics", {}) or {}
        return {
            "status": "trained",
            "num_examples": len(tokenized_dataset),
            "num_eval_examples": eval_count,
            "checkpoint_dir": str(checkpoint_dir),
            "metrics": dict(metrics),
            "fsdp_transformer_layer_cls_to_wrap": fsdp_wrap_classes,
        }

    @staticmethod
    def _initialize_swanlab_run(trainer) -> None:
        """Bridge callbacks expecting get_run() to return None before init.

        SwanLab 0.10 raises instead, while Transformers 5.10 checks for None.
        Initialize only the logging rank, retaining the callback's metric logging.
        """
        report_to = getattr(getattr(trainer, "args", None), "report_to", []) or []
        if isinstance(report_to, str):
            report_to = [report_to]
        if "swanlab" not in report_to or not trainer.is_world_process_zero():
            return
        import swanlab

        try:
            active_run = swanlab.get_run()
        except RuntimeError as exc:
            if str(exc) != "No active Run. Call swanlab.init() first.":
                raise
            active_run = None
        if active_run is not None:
            return
        kwargs = {
            "project": os.environ.get("SWANLAB_PROJECT_NAME") or os.environ.get("SWANLAB_PROJECT"),
            "experiment_name": trainer.args.run_name,
        }
        for variable, key in (("SWANLAB_RUN_ID", "id"), ("SWANLAB_RESUME", "resume")):
            if variable in os.environ:
                kwargs[key] = os.environ[variable]
        swanlab.init(**kwargs)

    def _with_available_fsdp_wrap_classes(
        self,
        args: Mapping[str, Any],
        model,
    ) -> Dict[str, Any]:
        configured = dict(args)
        if "auto_wrap" not in str(configured.get("fsdp", "")).lower():
            return configured
        raw_fsdp_config = configured.get("fsdp_config")
        if raw_fsdp_config is not None and not isinstance(raw_fsdp_config, Mapping):
            return configured
        fsdp_config = dict(raw_fsdp_config or {})
        if "transformer_layer_cls_to_wrap" in fsdp_config:
            return configured
        declared = getattr(model, "_no_split_modules", None)
        if not declared:
            return configured
        declared_names = [declared] if isinstance(declared, str) else list(declared)
        module_names = {type(module).__name__ for module in model.modules()}
        available = sorted(
            {str(name) for name in declared_names if str(name) in module_names}
        )
        if not available:
            raise ValueError(
                "FSDP auto-wrap could not find any model-declared `_no_split_modules` "
                f"classes in the loaded model. Declared: {declared_names}"
            )
        fsdp_config["transformer_layer_cls_to_wrap"] = available
        configured["fsdp_config"] = fsdp_config
        return configured

    def _training_features(self, tokenized_dataset: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [
            {key: value for key, value in row.items() if key != "metadata"}
            for row in tokenized_dataset
        ]

    def _training_trainer_cls(self, base_cls):
        trainer_cls = base_cls
        if self.response_only_logits:
            trainer_cls = self._response_only_loss_trainer_cls(trainer_cls)
        if self.compute_forecast_metrics:
            trainer_cls = self._forecast_eval_trainer_cls(trainer_cls)
        return trainer_cls

    def _response_only_loss_trainer_cls(self, base_cls):
        import inspect

        import torch
        import torch.nn.functional as F

        structure_marker_loss_enabled = self.structure_marker_loss_enabled
        structure_marker_loss_weight = self.structure_marker_loss_weight
        marker_token_sequences = (
            self._structure_marker_token_sequences()
            if structure_marker_loss_enabled
            else ()
        )

        class _ResponseOnlyLossTrainer(base_cls):
            @staticmethod
            def _response_logit_window(labels):
                if labels is None:
                    raise ValueError("response-only logits require labels")
                supervised_positions = torch.nonzero(labels.ne(-100), as_tuple=False)
                if supervised_positions.numel() == 0:
                    raise ValueError("response-only logits received no supervised labels")
                first_supervised_position = int(supervised_positions[:, -1].min().item())
                start = max(first_supervised_position - 1, 0)
                return start, labels.shape[-1] - start

            def _accepts_logits_to_keep(self, model):
                inspected_model = model
                unwrap_model = getattr(getattr(self, "accelerator", None), "unwrap_model", None)
                if callable(unwrap_model):
                    try:
                        inspected_model = unwrap_model(model)
                    except (AttributeError, TypeError, ValueError):
                        inspected_model = model
                forward = getattr(inspected_model, "forward", inspected_model)
                try:
                    parameters = tuple(inspect.signature(forward).parameters.values())
                except (TypeError, ValueError):
                    return None
                if any(parameter.name == "logits_to_keep" for parameter in parameters):
                    return True
                if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters):
                    return None
                return False

            def _get_num_items_in_batch(self, batch_samples, device):
                base_count = super()._get_num_items_in_batch(batch_samples, device)
                if not structure_marker_loss_enabled or base_count is None:
                    return base_count
                weighted_count = None
                for batch in batch_samples:
                    labels = batch.get("labels")
                    if labels is None:
                        return base_count
                    valid_count = labels.ne(-100).sum().to(dtype=torch.float32)
                    marker_count = self._structure_marker_mask(labels).sum().to(dtype=torch.float32)
                    batch_count = valid_count + (
                        structure_marker_loss_weight - 1.0
                    ) * marker_count
                    weighted_count = (
                        batch_count
                        if weighted_count is None
                        else weighted_count + batch_count
                    )
                if weighted_count is None:
                    return base_count

                args = getattr(self, "args", None)
                weighted_count = weighted_count.to(device)
                if getattr(args, "average_tokens_across_devices", False):
                    if getattr(args, "world_size", 1) > 1:
                        weighted_count = self.accelerator.gather(weighted_count).sum()
                elif getattr(args, "n_gpu", 0) > 1:
                    weighted_count = weighted_count / args.n_gpu

                if getattr(args, "n_gpu", 0) > 1 and weighted_count.dim() == 0:
                    weighted_count = weighted_count.unsqueeze(0).expand(args.n_gpu, -1)
                parallelism_config = getattr(
                    getattr(self, "accelerator", None),
                    "parallelism_config",
                    None,
                )
                if parallelism_config is not None:
                    weighted_count = (
                        weighted_count / parallelism_config.non_data_parallel_size
                    )
                return weighted_count

            def _model_outputs_with_logits_to_keep(self, model, model_inputs, logits_to_keep):
                accepts_logits_to_keep = self._accepts_logits_to_keep(model)
                if accepts_logits_to_keep is False:
                    raise TypeError(
                        "training.backend.response_only_logits=true requires a model "
                        "that supports the logits_to_keep forward argument"
                    )
                try:
                    return model(**model_inputs, logits_to_keep=logits_to_keep)
                except TypeError as exc:
                    message = str(exc)
                    if (
                        accepts_logits_to_keep is None
                        and "unexpected keyword argument" in message
                        and "logits_to_keep" in message
                    ):
                        raise TypeError(
                            "training.backend.response_only_logits=true requires a model "
                            "that supports the logits_to_keep forward argument"
                        ) from exc
                    raise

            def compute_loss(
                self,
                model,
                inputs,
                return_outputs=False,
                num_items_in_batch=None,
            ):
                labels = inputs.get("labels")
                start, logits_to_keep = self._response_logit_window(labels)
                model_inputs = {key: value for key, value in inputs.items() if key != "labels"}
                outputs = self._model_outputs_with_logits_to_keep(
                    model,
                    model_inputs,
                    logits_to_keep,
                )

                logits = getattr(outputs, "logits", None)
                if logits is None:
                    logits = outputs[0]
                shift_labels = F.pad(labels, (0, 1), value=-100)[..., 1:]
                shift_labels = shift_labels[..., start:]
                if logits.shape[-2] != shift_labels.shape[-1]:
                    raise ValueError(
                        "response-only SFT loss expected logits and shifted labels to have "
                        f"the same sequence length, got {logits.shape[-2]} and "
                        f"{shift_labels.shape[-1]}"
                    )

                logits = logits.float()
                flat_logits = logits.reshape(-1, logits.shape[-1])
                flat_labels = shift_labels.reshape(-1).to(logits.device)
                if structure_marker_loss_enabled:
                    token_losses = F.cross_entropy(
                        flat_logits,
                        flat_labels,
                        ignore_index=-100,
                        reduction="none",
                    )
                    marker_mask = self._structure_marker_mask(labels)
                    shift_marker_mask = F.pad(marker_mask, (0, 1), value=False)[..., 1:]
                    flat_marker_mask = shift_marker_mask[..., start:].reshape(-1).to(logits.device)
                    valid_mask = flat_labels.ne(-100)
                    token_weights = torch.ones_like(token_losses)
                    token_weights = torch.where(
                        flat_marker_mask & valid_mask,
                        torch.full_like(token_weights, structure_marker_loss_weight),
                        token_weights,
                    )
                    effective_weights = token_weights[valid_mask]
                    if effective_weights.numel() == 0:
                        raise ValueError("structure marker loss received no supervised labels")
                    weighted_loss_sum = (
                        token_losses[valid_mask] * effective_weights
                    ).sum()
                    if num_items_in_batch is None:
                        loss = weighted_loss_sum / effective_weights.sum()
                    else:
                        if torch.is_tensor(num_items_in_batch):
                            num_items_in_batch = num_items_in_batch.to(weighted_loss_sum.device)
                        loss = weighted_loss_sum / num_items_in_batch
                    self._accumulate_structure_loss_metrics(
                        token_losses=token_losses,
                        marker_mask=flat_marker_mask & valid_mask,
                        valid_mask=valid_mask,
                    )
                else:
                    reduction = "sum" if num_items_in_batch is not None else "mean"
                    loss = F.cross_entropy(
                        flat_logits,
                        flat_labels,
                        ignore_index=-100,
                        reduction=reduction,
                    )
                    if num_items_in_batch is not None:
                        if torch.is_tensor(num_items_in_batch):
                            num_items_in_batch = num_items_in_batch.to(loss.device)
                        loss = loss / num_items_in_batch
                args = getattr(self, "args", None)
                if (
                    num_items_in_batch is not None
                    and getattr(args, "average_tokens_across_devices", False)
                    and (
                        getattr(self, "model_accepts_loss_kwargs", False)
                        or getattr(self, "compute_loss_func", None) is not None
                    )
                ):
                    if getattr(args, "n_gpu", 0) <= 1:
                        process_count = getattr(getattr(self, "accelerator", None), "num_processes", 1)
                    else:
                        process_count = args.n_gpu
                    loss = loss * process_count
                return (loss, outputs) if return_outputs else loss

            def _accumulate_structure_loss_metrics(self, *, token_losses, marker_mask, valid_mask):
                non_marker_mask = valid_mask & ~marker_mask
                accumulator = getattr(self, "_structure_loss_metric_accumulator", None)
                if accumulator is None:
                    accumulator = {
                        "marker_loss_sum": 0.0,
                        "marker_count": 0,
                        "non_marker_loss_sum": 0.0,
                        "non_marker_count": 0,
                    }
                    self._structure_loss_metric_accumulator = accumulator
                accumulator["marker_loss_sum"] += float(
                    token_losses[marker_mask].detach().sum().cpu()
                )
                accumulator["marker_count"] += int(marker_mask.sum().detach().cpu())
                accumulator["non_marker_loss_sum"] += float(
                    token_losses[non_marker_mask].detach().sum().cpu()
                )
                accumulator["non_marker_count"] += int(non_marker_mask.sum().detach().cpu())

            def log(self, logs, start_time=None):
                enriched_logs = dict(logs)
                accumulator = getattr(self, "_structure_loss_metric_accumulator", None)
                if accumulator:
                    marker_count = int(accumulator["marker_count"])
                    non_marker_count = int(accumulator["non_marker_count"])
                    total_count = marker_count + non_marker_count
                    if marker_count:
                        enriched_logs["marker_ce"] = (
                            float(accumulator["marker_loss_sum"]) / marker_count
                        )
                    if non_marker_count:
                        enriched_logs["non_marker_ce"] = (
                            float(accumulator["non_marker_loss_sum"]) / non_marker_count
                        )
                    if total_count:
                        enriched_logs["marker_token_fraction"] = marker_count / total_count
                    self._structure_loss_metric_accumulator = None
                if start_time is None:
                    return super().log(enriched_logs)
                return super().log(enriched_logs, start_time=start_time)

            @staticmethod
            def _structure_marker_mask(labels):
                marker_mask = torch.zeros_like(labels, dtype=torch.bool)
                for token_sequence in marker_token_sequences:
                    width = len(token_sequence)
                    if width == 0 or labels.shape[-1] < width:
                        continue
                    expected = torch.tensor(
                        token_sequence,
                        dtype=labels.dtype,
                        device=labels.device,
                    )
                    matches = labels.unfold(-1, width, 1).eq(expected).all(dim=-1)
                    for offset in range(width):
                        marker_mask[..., offset : offset + matches.shape[-1]] |= matches
                return marker_mask & labels.ne(-100)

        return _ResponseOnlyLossTrainer

    def _forecast_eval_trainer_cls(self, base_cls):
        """Trainer subclass that skips the model's full-vocab fp32 cross-entropy
        during evaluation. ``metric_for_best_model`` is the forecast MAE, so the
        eval loss is never used; running the eval forward *without* labels avoids
        materialising the ``[seq_len, vocab]`` fp32 logits + log-softmax that
        ``ForCausalLMLoss`` allocates, which OOMs on long GEFCom prompts. It also
        argmaxes the logits to greedy token ids inside ``prediction_step`` so the
        eval loop never pads/gathers a full ``[batch, seq, vocab]`` tensor."""
        import torch

        response_only_logits = self.response_only_logits

        class _ForecastEvalTrainer(base_cls):
            def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
                if prediction_loss_only:
                    return (None, None, None)
                inputs = self._prepare_inputs(inputs)
                labels = inputs.get("labels")
                model_inputs = {key: value for key, value in inputs.items() if key != "labels"}
                response_start = 0
                logits_to_keep = None
                if response_only_logits:
                    response_start, logits_to_keep = self._response_logit_window(labels)
                    model_inputs["use_cache"] = False
                with torch.no_grad():
                    with self.compute_loss_context_manager():
                        if logits_to_keep is None:
                            outputs = model(**model_inputs)
                        else:
                            outputs = self._model_outputs_with_logits_to_keep(
                                model,
                                model_inputs,
                                logits_to_keep,
                            )
                logits = getattr(outputs, "logits", None)
                if logits is None:
                    logits = outputs[0]
                # Reduce to greedy token ids *here*, before the eval loop calls
                # pad_across_processes (which copies the tensor) and gathers it.
                # Returning full [batch, seq, vocab] logits OOMs on long GEFCom
                # sequences; _compute_forecast_metrics consumes [batch, seq] ids.
                prediction_ids = logits.argmax(dim=-1)
                if response_only_logits:
                    expected_length = labels.shape[-1] - response_start
                    if prediction_ids.shape[-1] != expected_length:
                        raise ValueError(
                            "response-only forecast eval expected prediction ids and kept "
                            f"labels to have the same sequence length, got "
                            f"{prediction_ids.shape[-1]} and {expected_length}"
                        )
                    full_prediction_ids = torch.full_like(labels, -100)
                    full_prediction_ids[..., response_start:] = prediction_ids
                    prediction_ids = full_prediction_ids
                return (None, prediction_ids, labels)

            def evaluation_loop(self, *args, **kwargs):
                output = super().evaluation_loop(*args, **kwargs)
                # Each eval step materialises a large transient
                # [batch, seq, vocab] logits tensor. Return the reserved-but-idle
                # blocks so the following training backward pass isn't starved by
                # fragmentation ("reserved >> allocated" OOM).
                accelerator = getattr(torch, "npu", None) or getattr(torch, "cuda", None)
                empty_cache = getattr(accelerator, "empty_cache", None)
                if callable(empty_cache):
                    empty_cache()
                return output

            def _finalize_training(self, trial, num_train_samples, start_time):
                self._evaluate_last_training_step_if_needed(trial)
                return super()._finalize_training(trial, num_train_samples, start_time)

            def _evaluate_last_training_step_if_needed(self, trial):
                if getattr(self, "eval_dataset", None) is None:
                    return
                state = getattr(self, "state", None)
                global_step = int(getattr(state, "global_step", 0) or 0)
                max_steps = int(getattr(state, "max_steps", global_step) or global_step)
                if global_step <= 0 or global_step != max_steps:
                    return
                metric_for_best_model = getattr(getattr(self, "args", None), "metric_for_best_model", None)
                if not metric_for_best_model:
                    return
                metric_key = str(metric_for_best_model)
                if not metric_key.startswith("eval_"):
                    metric_key = f"eval_{metric_key}"
                for log_entry in reversed(getattr(state, "log_history", []) or []):
                    if int(log_entry.get("step", -1) or -1) != global_step:
                        continue
                    if metric_key in log_entry:
                        return
                    if any(str(key).startswith("eval_") for key in log_entry):
                        return
                metrics = self._evaluate(trial, ignore_keys_for_eval=None, skip_scheduler=True)
                is_new_best_metric = self._determine_best_metric(metrics=metrics, trial=trial)
                if is_new_best_metric:
                    self._save_checkpoint(self.model, trial)
                    self.control = self.callback_handler.on_save(self.args, self.state, self.control)

        return _ForecastEvalTrainer

    def _build_forecast_compute_metrics(self):
        def compute_metrics(eval_pred):
            predictions = getattr(eval_pred, "predictions", None)
            label_ids = getattr(eval_pred, "label_ids", None)
            if predictions is None and isinstance(eval_pred, (tuple, list)) and len(eval_pred) >= 2:
                predictions, label_ids = eval_pred[:2]
            return self._compute_forecast_metrics(predictions, label_ids)

        return compute_metrics

    def _compute_forecast_metrics(self, predictions, labels) -> dict[str, float]:
        if predictions is None or labels is None:
            return {
                "mae": math.inf,
                "mse": math.inf,
                "valid_forecast_samples": 0,
                "invalid_forecast_samples": 0,
            }
        if isinstance(predictions, (tuple, list)):
            predictions = predictions[0]
        prediction_array = np.asarray(predictions)
        label_array = np.asarray(labels)
        if prediction_array.ndim == label_array.ndim + 1:
            prediction_ids = prediction_array.argmax(axis=-1)
        else:
            prediction_ids = prediction_array
        tokenizer = self._require_tokenizer()
        squared_error_sum = 0.0
        absolute_error_sum = 0.0
        num_points = 0
        valid_samples = 0
        invalid_samples = 0
        for sample_prediction_ids, sample_label_ids in zip(prediction_ids, label_array):
            label_positions = np.nonzero(sample_label_ids != -100)[0]
            if len(label_positions) == 0:
                invalid_samples += 1
                continue
            prediction_positions = label_positions - 1
            valid_position_mask = prediction_positions >= 0
            label_positions = label_positions[valid_position_mask]
            prediction_positions = prediction_positions[valid_position_mask]
            target_ids = [int(token_id) for token_id in sample_label_ids[label_positions]]
            predicted_ids = [int(token_id) for token_id in sample_prediction_ids[prediction_positions]]
            target_text = tokenizer.decode(target_ids, skip_special_tokens=True)
            predicted_text = tokenizer.decode(predicted_ids, skip_special_tokens=True)
            target_values = extract_prediction_values(target_text)
            predicted_values = extract_prediction_values(predicted_text)
            horizon = self.forecast_metric_length
            if len(target_values) < horizon or len(predicted_values) < horizon:
                invalid_samples += 1
                continue
            target = np.asarray(target_values[:horizon], dtype=np.float64)
            predicted = np.asarray(predicted_values[:horizon], dtype=np.float64)
            errors = predicted - target
            squared_error_sum += float(np.sum(errors * errors))
            absolute_error_sum += float(np.sum(np.abs(errors)))
            num_points += int(horizon)
            valid_samples += 1
        if num_points == 0:
            mae = math.inf
            mse = math.inf
        else:
            mae = absolute_error_sum / num_points
            mse = squared_error_sum / num_points
        return {
            "mae": mae,
            "mse": mse,
            "valid_forecast_samples": valid_samples,
            "invalid_forecast_samples": invalid_samples,
        }

    def _require_tokenizer(self):
        if self.tokenizer is None:
            raise ValueError("TransformersSFTBackend requires a tokenizer")
        return self.tokenizer

    def _ensure_pad_token(self) -> None:
        tokenizer = self._require_tokenizer()
        if getattr(tokenizer, "pad_token", None) is None:
            eos_token = getattr(tokenizer, "eos_token", None)
            if eos_token is None:
                raise ValueError("Tokenizer must define pad_token or eos_token")
            tokenizer.pad_token = eos_token

    def _trainer_types(self):
        if (
            self.trainer_cls is not None
            and self.training_args_cls is not None
            and self.data_collator_cls is not None
        ):
            return self.trainer_cls, self.training_args_cls, self.data_collator_cls
        try:
            from transformers import DataCollatorForSeq2Seq, Trainer, TrainingArguments
        except ImportError as exc:
            raise ImportError(
                "TransformersSFTBackend requires the optional 'transformers' dependency"
            ) from exc
        return (
            self.trainer_cls or Trainer,
            self.training_args_cls or TrainingArguments,
            self.data_collator_cls or DataCollatorForSeq2Seq,
        )

    def _tokenize_text(
        self,
        text: str,
        max_length: int,
        *,
        add_special_tokens: bool = True,
    ) -> list[int]:
        return self._truncate_ids(
            self._encode_text(text, add_special_tokens=add_special_tokens),
            max_length,
        )

    def _encode_text(
        self,
        text: str,
        *,
        add_special_tokens: bool = True,
    ) -> list[int]:
        tokenizer = self._require_tokenizer()
        try:
            encoded = tokenizer(
                text,
                truncation=False,
                max_length=None,
                add_special_tokens=add_special_tokens,
            )
        except TypeError:
            encoded = tokenizer(text, truncation=False)
        return list(encoded["input_ids"])

    def _eos_token_ids(self) -> list[int]:
        tokenizer = self._require_tokenizer()
        eos_token_id = getattr(tokenizer, "eos_token_id", None)
        if eos_token_id is not None:
            return [int(eos_token_id)]
        eos_token = getattr(tokenizer, "eos_token", None)
        if not eos_token:
            return []
        return self._tokenize_text(
            str(eos_token),
            max_length=128,
            add_special_tokens=False,
        )

    def _structure_marker_token_sequences(self) -> tuple[tuple[int, ...], ...]:
        sequences = []
        for marker in DEFAULT_STRUCTURE_MARKERS:
            token_ids = self._tokenize_text(
                marker,
                max_length=128,
                add_special_tokens=False,
            )
            if token_ids:
                sequences.append(tuple(token_ids))
        return tuple(sequences)

    def _truncate_ids(self, ids: list[int], max_length: int) -> list[int]:
        if max_length < 0:
            raise ValueError("max_length must be non-negative")
        if len(ids) <= max_length:
            return ids
        if self.truncation_side == "left":
            return ids[-max_length:] if max_length else []
        return ids[:max_length]
