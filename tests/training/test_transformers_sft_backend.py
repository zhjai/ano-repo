from __future__ import annotations

import math
import unittest
from types import SimpleNamespace

if __import__("importlib.util", fromlist=["find_spec"]).find_spec("numpy") is None:
    raise unittest.SkipTest("install elecload[reasoning] or elecload[train]")
import numpy as np

from castfactory.training.transformers_sft_backend import TransformersSFTBackend


def _tiny_qwen3_model():
    import torch
    from transformers import Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(7)
    config = Qwen3Config(
        vocab_size=41,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        max_position_embeddings=32,
        attention_dropout=0.0,
        use_cache=False,
    )
    return Qwen3ForCausalLM(config)


def _tiny_qwen35_model():
    from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig

    config = Qwen3_5TextConfig(
        vocab_size=41,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        max_position_embeddings=32,
        layer_types=["full_attention"],
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 10000.0,
            "partial_rotary_factor": 0.25,
        },
        use_cache=False,
    )
    return Qwen3_5ForCausalLM(config)


def _answer_table(values: list[float]) -> str:
    lines = ["<answer>", "```", "index | value"]
    lines.extend(f"{index} | {value}" for index, value in enumerate(values))
    lines.extend(["```", "</answer>"])
    return "\n".join(lines)


class FakeDecodeTokenizer:
    pad_token = "<pad>"
    eos_token = "<eos>"
    pad_token_id = 0
    eos_token_id = 99

    def __init__(self, decoded: dict[tuple[int, ...], str]):
        self.decoded = decoded

    def decode(self, token_ids, skip_special_tokens: bool = True) -> str:
        ids = tuple(int(token_id) for token_id in token_ids if int(token_id) != self.pad_token_id)
        if skip_special_tokens:
            ids = tuple(token_id for token_id in ids if token_id != self.eos_token_id)
        return self.decoded.get(ids, "")


class FakeChatTokenizer:
    chat_template = "fake-chat-template"
    bos_token = "<bos>"
    bos_token_id = 1
    eos_token = "<eos>"
    eos_token_id = 2
    pad_token = "<pad>"
    pad_token_id = 0

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        if add_generation_prompt:
            self.assert_single_user_message(messages)
            rendered = f"<user>{messages[0]['content']}</user><assistant>"
        else:
            if len(messages) != 2 or messages[-1].get("role") != "assistant":
                raise AssertionError("expected a complete user/assistant conversation")
            rendered = (
                f"<user>{messages[0]['content']}</user><assistant>"
                f"{messages[-1]['content']}{self.eos_token}"
            )
        if not tokenize:
            return rendered
        return self(rendered, add_special_tokens=False)["input_ids"]

    def __call__(
        self,
        text,
        *,
        truncation=False,
        max_length=None,
        add_special_tokens=True,
    ):
        del truncation, max_length
        source = str(text)
        has_eos = source.endswith(self.eos_token)
        if has_eos:
            source = source[: -len(self.eos_token)]
        ids = [100 + ord(character) for character in source]
        if add_special_tokens:
            ids.insert(0, self.bos_token_id)
        if has_eos:
            ids.append(self.eos_token_id)
        return {"input_ids": ids}

    @staticmethod
    def assert_single_user_message(messages) -> None:
        if len(messages) != 1 or messages[0].get("role") != "user":
            raise AssertionError("expected one user message")


class FakeChatTokenizerWithEncodedEOS(FakeChatTokenizer):
    def __call__(self, text, **kwargs):
        add_special_tokens = kwargs.get("add_special_tokens", True)
        source = str(text)
        has_eos = source.endswith(self.eos_token)
        if has_eos:
            source = source[: -len(self.eos_token)]
        encoded = super().__call__(source, **kwargs)
        if has_eos:
            encoded["input_ids"].append(self.eos_token_id)
        if not add_special_tokens and encoded["input_ids"][:1] == [self.bos_token_id]:
            raise AssertionError("response must not contain BOS")
        return encoded


class FakeOffsetChatTokenizer(FakeChatTokenizer):
    def __call__(self, text, *, return_offsets_mapping=False, **kwargs):
        encoded = super().__call__(text, **kwargs)
        if return_offsets_mapping:
            source = str(text)
            offsets = [(index, index + 1) for index in range(len(source))]
            if source.endswith(self.eos_token):
                offsets = offsets[: -len(self.eos_token)] + [(0, 0)]
            encoded["offset_mapping"] = offsets
        return encoded


class FakeRawTokenizer(FakeChatTokenizer):
    chat_template = None

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        raise AssertionError("raw fallback must not call apply_chat_template")


class FakeMarkerTokenizer:
    pad_token = "<pad>"
    pad_token_id = 0
    eos_token = "<eos>"
    eos_token_id = 2

    def __init__(self):
        self.marker_ids = {
            marker: [20 + index]
            for index, marker in enumerate(
                (
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
            )
        }

    def __call__(
        self,
        text,
        *,
        truncation=False,
        max_length=None,
        add_special_tokens=True,
    ):
        del truncation, max_length, add_special_tokens
        source = str(text)
        return {"input_ids": list(self.marker_ids.get(source, [3]))}


class FakeThinkingChatTokenizer:
    chat_template = "fake-thinking-chat-template"
    bos_token = "<bos>"
    bos_token_id = 1
    eos_token = "<eos>"
    eos_token_id = 2
    pad_token = "<pad>"
    pad_token_id = 0
    prompt_owns_think = False

    def __init__(self):
        self.full_render_calls = 0

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        if not messages or messages[0].get("role") != "user":
            raise AssertionError("expected a user message first")
        prefix = f"<user>{messages[0]['content']}</user><assistant>\n"
        if add_generation_prompt:
            if len(messages) != 1:
                raise AssertionError("generation prompt must exclude the assistant message")
            rendered = prefix + ("<think>\n" if self.prompt_owns_think else "")
        else:
            if len(messages) != 2 or messages[-1].get("role") != "assistant":
                raise AssertionError("expected one final assistant message")
            self.full_render_calls += 1
            content = str(messages[-1].get("content", ""))
            if "</think>" in content:
                reasoning, answer = content.split("</think>", 1)
                reasoning = reasoning.split("<think>")[-1].strip("\n")
                content = f"<think>\n{reasoning}\n</think>\n\n{answer.lstrip()}"
            rendered = prefix + content + self.eos_token
        if not tokenize:
            return rendered
        return {"input_ids": self._encode(rendered, add_special_tokens=False)}

    def __call__(
        self,
        text,
        *,
        truncation=False,
        max_length=None,
        add_special_tokens=True,
    ):
        del truncation, max_length
        return {
            "input_ids": self._encode(
                str(text),
                add_special_tokens=add_special_tokens,
            )
        }

    def decode(self, token_ids, skip_special_tokens: bool = False) -> str:
        pieces = []
        for token_id in token_ids:
            token_id = int(token_id)
            if token_id == self.bos_token_id:
                if not skip_special_tokens:
                    pieces.append(self.bos_token)
            elif token_id == self.eos_token_id:
                if not skip_special_tokens:
                    pieces.append(self.eos_token)
            else:
                pieces.append(chr(token_id - 100))
        return "".join(pieces)

    def _encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        source = str(text)
        has_eos = source.endswith(self.eos_token)
        if has_eos:
            source = source[: -len(self.eos_token)]
        ids = [100 + ord(character) for character in source]
        if add_special_tokens:
            ids.insert(0, self.bos_token_id)
        if has_eos:
            ids.append(self.eos_token_id)
        return ids


class FakeQwen35ThinkingChatTokenizer(FakeThinkingChatTokenizer):
    prompt_owns_think = True


class FakeMultiTurnQwen35ThinkingChatTokenizer(FakeQwen35ThinkingChatTokenizer):
    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        if not messages or messages[0].get("role") != "user":
            raise AssertionError("expected a user message first")
        if add_generation_prompt:
            if messages[-1].get("role") != "user":
                raise AssertionError("generation prompt must end with a user message")
            rendered = self._render_history(messages) + "<assistant>\n<think>\n"
        else:
            if messages[-1].get("role") != "assistant":
                raise AssertionError("complete conversation must end with an assistant")
            self.full_render_calls += 1
            content = str(messages[-1].get("content", ""))
            reasoning, answer = content.split("</think>", 1)
            reasoning = reasoning.split("<think>")[-1].strip("\n")
            rendered = (
                self._render_history(messages[:-1])
                + "<assistant>\n<think>\n"
                + reasoning
                + "\n</think>\n\n"
                + answer.lstrip()
                + self.eos_token
            )
        if not tokenize:
            return rendered
        return {"input_ids": self._encode(rendered, add_special_tokens=False)}

    def _render_history(self, messages) -> str:
        rendered = []
        for message in messages:
            role = message.get("role")
            rendered.append(f"<{role}>{message.get('content', '')}</{role}>")
        return "".join(rendered)


class TransformersSFTBackendCheckpointFormatTest(unittest.TestCase):
    @unittest.skipUnless(__import__("importlib.util", fromlist=["find_spec"]).find_spec("torch") and __import__("importlib.util", fromlist=["find_spec"]).find_spec("transformers"), "install elecload[train]")
    def test_native_save_suppresses_reverse_conversions_and_restores_attribute(self):
        model = _tiny_qwen35_model()
        backend = TransformersSFTBackend(save_original_format=False)
        had_attribute = hasattr(model, "_weight_conversions")

        with backend._checkpoint_save_format(model):
            self.assertEqual(model._weight_conversions, [])

        self.assertEqual(hasattr(model, "_weight_conversions"), had_attribute)

    @unittest.skipUnless(__import__("importlib.util", fromlist=["find_spec"]).find_spec("torch") and __import__("importlib.util", fromlist=["find_spec"]).find_spec("transformers"), "install elecload[train]")
    def test_native_save_restores_attribute_after_error(self):
        model = _tiny_qwen35_model()
        marker = [object()]
        model._weight_conversions = marker
        backend = TransformersSFTBackend(save_original_format=False)

        with self.assertRaisesRegex(RuntimeError, "sentinel"):
            with backend._checkpoint_save_format(model):
                self.assertEqual(model._weight_conversions, [])
                raise RuntimeError("sentinel")

        self.assertIs(model._weight_conversions, marker)

    @unittest.skipUnless(__import__("importlib.util", fromlist=["find_spec"]).find_spec("torch") and __import__("importlib.util", fromlist=["find_spec"]).find_spec("transformers"), "install elecload[train]")
    def test_default_save_mode_does_not_mutate_qwen3(self):
        model = _tiny_qwen3_model()
        backend = TransformersSFTBackend()
        before = getattr(model, "_weight_conversions", None)

        with backend._checkpoint_save_format(model):
            self.assertIs(getattr(model, "_weight_conversions", None), before)


class TransformersSFTBackendResumeTest(unittest.TestCase):
    def test_resume_checkpoint_is_passed_to_trainer_not_training_arguments(self) -> None:
        class FakeTrainingArguments:
            last_kwargs = None

            def __init__(self, **kwargs):
                type(self).last_kwargs = kwargs

        class FakeDataCollator:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        class FakeTrainer:
            last_train_kwargs = None

            def __init__(self, **kwargs):
                self.kwargs = kwargs

            def train(self, **kwargs):
                type(self).last_train_kwargs = kwargs
                return SimpleNamespace(metrics={})

            def save_model(self, checkpoint_dir):
                self.checkpoint_dir = checkpoint_dir

        resume_checkpoint = "/tmp/castfactory-checkpoint-10"
        backend = TransformersSFTBackend(
            model=object(),
            tokenizer=FakeChatTokenizer(),
            trainer_cls=FakeTrainer,
            training_args_cls=FakeTrainingArguments,
            data_collator_cls=FakeDataCollator,
        )

        backend.fit(
            [{"input": "prompt", "output": "response"}],
            None,
            "/tmp/castfactory-output",
            train_args={
                "learning_rate": 2.0e-5,
                "resume_from_checkpoint": resume_checkpoint,
            },
        )

        self.assertNotIn("resume_from_checkpoint", FakeTrainingArguments.last_kwargs)
        self.assertEqual(
            FakeTrainer.last_train_kwargs,
            {"resume_from_checkpoint": resume_checkpoint},
        )


class TransformersSFTBackendFSDPTest(unittest.TestCase):
    @unittest.skipUnless(__import__("importlib.util", fromlist=["find_spec"]).find_spec("torch") and __import__("importlib.util", fromlist=["find_spec"]).find_spec("transformers"), "install elecload[train]")
    def test_model_declared_wrap_classes_are_filtered_to_loaded_modules(self) -> None:
        import torch

        class Qwen3_5DecoderLayer(torch.nn.Module):
            pass

        class PureCausalLM(torch.nn.Module):
            _no_split_modules = {"Qwen3_5DecoderLayer", "Qwen3_5VisionBlock"}

            def __init__(self) -> None:
                super().__init__()
                self.layer = Qwen3_5DecoderLayer()

        backend = TransformersSFTBackend()

        configured = backend._with_available_fsdp_wrap_classes(
            {
                "fsdp": "full_shard auto_wrap",
                "fsdp_config": {"state_dict_type": "SHARDED_STATE_DICT"},
            },
            PureCausalLM(),
        )

        self.assertEqual(
            configured["fsdp_config"]["transformer_layer_cls_to_wrap"],
            ["Qwen3_5DecoderLayer"],
        )

    @unittest.skipUnless(__import__("importlib.util", fromlist=["find_spec"]).find_spec("torch") and __import__("importlib.util", fromlist=["find_spec"]).find_spec("transformers"), "install elecload[train]")
    def test_missing_declared_wrap_classes_are_named_in_error(self) -> None:
        import torch

        class PureCausalLM(torch.nn.Module):
            _no_split_modules = ("MissingDecoderLayer", "MissingVisionBlock")

        backend = TransformersSFTBackend()

        with self.assertRaises(ValueError) as caught:
            backend._with_available_fsdp_wrap_classes(
                {"fsdp": "full_shard auto_wrap"},
                PureCausalLM(),
            )

        message = str(caught.exception)
        self.assertIn("MissingDecoderLayer", message)
        self.assertIn("MissingVisionBlock", message)


class TransformersSFTBackendPromptBoundaryTest(unittest.TestCase):
    def test_prepare_dataset_matches_chat_prefix_without_response_bos(self) -> None:
        tokenizer = FakeChatTokenizer()
        backend = TransformersSFTBackend(
            tokenizer=tokenizer,
            max_prompt_length=4096,
            max_response_length=4096,
        )
        prompt = "Forecast the next values."
        response = "<think>Reasoning.</think><answer>1 | 2</answer>"

        prepared = backend.prepare_dataset(
            [{"input": prompt, "output": response, "metadata": {"sample_id": "x"}}]
        )[0]

        rendered_prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        expected_prompt_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True,
            add_generation_prompt=True,
        )
        expected_response_ids = tokenizer(response, add_special_tokens=False)["input_ids"]
        response_start = len(expected_prompt_ids)

        self.assertEqual(prepared["input_ids"][:response_start], expected_prompt_ids)
        self.assertEqual(
            prepared["input_ids"][response_start:],
            expected_response_ids + [tokenizer.eos_token_id],
        )
        self.assertEqual(prepared["labels"][:response_start], [-100] * response_start)
        self.assertEqual(
            prepared["labels"][response_start:],
            expected_response_ids + [tokenizer.eos_token_id],
        )
        self.assertNotEqual(prepared["input_ids"][response_start], tokenizer.bos_token_id)
        self.assertEqual(prepared["metadata"], {"sample_id": "x"})

    def test_prepare_dataset_does_not_duplicate_existing_eos(self) -> None:
        tokenizer = FakeChatTokenizerWithEncodedEOS()
        backend = TransformersSFTBackend(tokenizer=tokenizer)

        prepared = backend.prepare_dataset(
            [{"input": "Prompt", "output": f"<think>x</think>{tokenizer.eos_token}"}]
        )[0]

        self.assertEqual(prepared["input_ids"].count(tokenizer.eos_token_id), 1)
        self.assertEqual(prepared["labels"].count(tokenizer.eos_token_id), 1)

    def test_forecast_values_mask_keeps_envelope_and_eos_supervised(self) -> None:
        tokenizer = FakeChatTokenizer()
        backend = TransformersSFTBackend(
            tokenizer=tokenizer,
            response_loss_mode="forecast_values_masked",
            max_response_length=4096,
        )
        workspace = (
            '{"context_status":"no_event_context","drivers":[],'
            '"regimes":"history_exogenous","confidence":0.5,'
            '"weather_vintage":"oracle_weather"}'
        )
        values = "[1.0,2.0,3.0]"
        response = (
            "<think><workspace>"
            + workspace
            + "</workspace></think><answer><forecast>"
            + values
            + "</forecast></answer>"
        )
        prepared = backend.prepare_dataset([{"input": "Prompt", "output": response}])[0]
        prompt_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": "Prompt"}],
            tokenize=True,
            add_generation_prompt=True,
        )
        response_ids = tokenizer(response, add_special_tokens=False)["input_ids"]
        labels = prepared["labels"][len(prompt_ids) :]
        value_ids = tokenizer(values, add_special_tokens=False)["input_ids"]
        start = next(
            offset
            for offset in range(len(response_ids) - len(value_ids) + 1)
            if response_ids[offset : offset + len(value_ids)] == value_ids
        )
        self.assertEqual(labels[start : start + len(value_ids)], [-100] * len(value_ids))
        self.assertNotEqual(labels[start - 1], -100)  # <forecast> boundary
        self.assertNotEqual(labels[start + len(value_ids)], -100)  # </forecast>
        self.assertEqual(labels[-1], tokenizer.eos_token_id)

    def test_forecast_numbers_mask_keeps_json_structure_and_eos_supervised(self) -> None:
        tokenizer = FakeOffsetChatTokenizer()
        backend = TransformersSFTBackend(
            tokenizer=tokenizer,
            response_loss_mode="forecast_numbers_masked",
            max_response_length=4096,
        )
        response = (
            '<think><workspace>{"context_status":"no_event_context","drivers":[],'
            '"regimes":"history","confidence":0.5,"weather_vintage":"oracle_weather"}'
            '</workspace></think><answer><forecast>[1.0,-2.5,3e2]</forecast></answer>'
        )
        prepared = backend.prepare_dataset([{"input": "Prompt", "output": response}])[0]
        prompt_length = len(tokenizer.apply_chat_template(
            [{"role": "user", "content": "Prompt"}], tokenize=True, add_generation_prompt=True,
        ))
        labels = prepared["labels"][prompt_length:]
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": "Prompt"}, {"role": "assistant", "content": response}],
            tokenize=False, add_generation_prompt=False,
        )
        for token_offset, character in enumerate(response):
            label = labels[token_offset]
            in_number = any(
                start <= token_offset < end
                for start, end in [(response.index("1.0"), response.index("1.0") + 3),
                                   (response.index("-2.5"), response.index("-2.5") + 4),
                                   (response.index("3e2"), response.index("3e2") + 3)]
            )
            if in_number:
                self.assertEqual(label, -100)
            else:
                self.assertNotEqual(label, -100, (token_offset, character))
        self.assertEqual(labels[-1], tokenizer.eos_token_id)

    def test_forecast_numbers_mask_rejects_nonnumeric_array(self) -> None:
        backend = TransformersSFTBackend(
            tokenizer=FakeOffsetChatTokenizer(), response_loss_mode="forecast_numbers_masked",
        )
        with self.assertRaisesRegex(ValueError, "numeric JSON array"):
            backend.prepare_dataset([{"input": "Prompt", "output": "<forecast>[true,1]</forecast>"}])

    def test_prepare_dataset_preserves_raw_fallback_without_chat_template(self) -> None:
        tokenizer = FakeRawTokenizer()
        backend = TransformersSFTBackend(tokenizer=tokenizer)

        prepared = backend.prepare_dataset([{"input": "Prompt", "output": "<think>x</think>"}])[0]
        expected_prompt_ids = tokenizer("Prompt", add_special_tokens=True)["input_ids"]

        self.assertEqual(prepared["input_ids"][: len(expected_prompt_ids)], expected_prompt_ids)
        self.assertEqual(prepared["labels"][: len(expected_prompt_ids)], [-100] * len(expected_prompt_ids))

    def test_prepare_dataset_rejects_empty_response(self) -> None:
        backend = TransformersSFTBackend(tokenizer=FakeChatTokenizer())

        with self.assertRaisesRegex(ValueError, "non-empty SFT response"):
            backend.prepare_dataset([{"input": "Prompt", "output": "   "}])

    def test_prepare_dataset_rejects_response_over_budget_without_truncating(self) -> None:
        backend = TransformersSFTBackend(
            tokenizer=FakeChatTokenizer(),
            max_response_length=8,
        )

        with self.assertRaisesRegex(ValueError, "max_response_length"):
            backend.prepare_dataset(
                [{"input": "Prompt", "output": "<think>too long</think>"}]
            )

    def test_legacy_input_output_matches_canonical_messages(self) -> None:
        response = "<think>\nreasoning\n</think>\n<answer>x</answer>"
        legacy_backend = TransformersSFTBackend(tokenizer=FakeThinkingChatTokenizer())
        messages_backend = TransformersSFTBackend(tokenizer=FakeThinkingChatTokenizer())

        legacy = legacy_backend.prepare_dataset(
            [{"input": "Prompt", "output": response, "metadata": {"sample_id": "x"}}]
        )[0]
        canonical = messages_backend.prepare_dataset(
            [
                {
                    "messages": [
                        {"role": "user", "content": "Prompt"},
                        {"role": "assistant", "content": response},
                    ],
                    "metadata": {"sample_id": "x"},
                }
            ]
        )[0]

        self.assertEqual(canonical, legacy)

    def test_qwen3_and_qwen35_templates_derive_assistant_loss_boundary(self) -> None:
        response = "<think>\nreasoning\n</think>\n<answer>x</answer>"
        cases = (
            (FakeThinkingChatTokenizer(), "<think>\n"),
            (FakeQwen35ThinkingChatTokenizer(), "reasoning"),
        )

        for tokenizer, expected_supervised_prefix in cases:
            with self.subTest(prompt_owns_think=tokenizer.prompt_owns_think):
                backend = TransformersSFTBackend(tokenizer=tokenizer)

                prepared = backend.prepare_dataset(
                    [{"input": "Prompt", "output": response}]
                )[0]

                full_text = tokenizer.decode(
                    prepared["input_ids"],
                    skip_special_tokens=False,
                )
                supervised_text = tokenizer.decode(
                    [
                        token_id
                        for token_id, label in zip(
                            prepared["input_ids"],
                            prepared["labels"],
                        )
                        if label != -100
                    ],
                    skip_special_tokens=False,
                )
                self.assertEqual(full_text.count("<think>"), 1)
                self.assertTrue(supervised_text.startswith(expected_supervised_prefix))
                self.assertEqual(prepared["input_ids"].count(tokenizer.eos_token_id), 1)
                self.assertEqual(prepared["labels"].count(tokenizer.eos_token_id), 1)
                self.assertEqual(tokenizer.full_render_calls, 1)

    def test_qwen35_empty_think_boundary_merge_preserves_full_tokens(self) -> None:
        class BoundaryTokenizer(FakeQwen35ThinkingChatTokenizer):
            def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
                prefix = "user prompt assistant<think>\n"
                rendered = prefix if add_generation_prompt else prefix + "\n</think>\n" + messages[-1]["content"]
                return self(rendered, add_special_tokens=False)["input_ids"] if tokenize else rendered

            def __call__(self, text, *, return_offsets_mapping=False, **kwargs):
                ids, offsets = [], []
                i = 0
                while i < len(text):
                    width = 2 if text[i:i + 2] == "\n\n" else 1
                    ids.append(500 if width == 2 else 100 + ord(text[i]))
                    offsets.append((i, i + width))
                    i += width
                return {"input_ids": ids, **({"offset_mapping": offsets} if return_offsets_mapping else {})}

        tokenizer = BoundaryTokenizer()
        backend = TransformersSFTBackend(tokenizer=tokenizer)
        row = {"input": "Prompt", "output": "<answer>1 | 2</answer>"}
        prepared = backend.prepare_dataset([row])[0]
        full = tokenizer.apply_chat_template(
            [{"role": "user", "content": "Prompt"}, {"role": "assistant", "content": row["output"]}],
            tokenize=True, add_generation_prompt=False,
        )
        self.assertEqual(prepared["input_ids"], full)
        crossing = full.index(500)
        self.assertEqual(prepared["labels"][:crossing], [-100] * crossing)
        self.assertEqual(prepared["labels"][crossing:], full[crossing:])

        with self.assertRaisesRegex(ValueError, "exact prefix"):
            backend._merged_prompt_boundary("wrong", "different", [1], [2])
        with self.assertRaisesRegex(ValueError, "exact prefix"):
            backend._merged_prompt_boundary("a", "ab", [1], [2, 3])

    def test_qwen35_multi_turn_masks_history_and_prompt_owned_opener(self) -> None:
        tokenizer = FakeMultiTurnQwen35ThinkingChatTokenizer()
        backend = TransformersSFTBackend(tokenizer=tokenizer)

        prepared = backend.prepare_dataset(
            [
                {
                    "messages": [
                        {"role": "user", "content": "First prompt"},
                        {
                            "role": "assistant",
                            "content": "<think>first reasoning</think><answer>first</answer>",
                        },
                        {"role": "user", "content": "Second prompt"},
                        {
                            "role": "assistant",
                            "content": "<think>second reasoning</think><answer>second</answer>",
                        },
                    ]
                }
            ]
        )[0]

        supervised_text = tokenizer.decode(
            [
                token_id
                for token_id, label in zip(prepared["input_ids"], prepared["labels"])
                if label != -100
            ],
            skip_special_tokens=False,
        )

        self.assertTrue(supervised_text.startswith("second reasoning"))
        self.assertNotIn("first reasoning", supervised_text)
        self.assertNotIn("<think>", supervised_text)
        self.assertEqual(prepared["labels"].count(tokenizer.eos_token_id), 1)
        self.assertEqual(tokenizer.full_render_calls, 1)


class TransformersSFTBackendForecastMetricsTest(unittest.TestCase):
    def test_compute_forecast_metrics_decodes_supervised_logits(self) -> None:
        label_values = [1.0, 2.0, 3.0]
        pred_values = [2.0, 2.0, 4.0]
        tokenizer = FakeDecodeTokenizer(
            {
                (1, 2, 3): _answer_table(label_values),
                (4, 5, 6): _answer_table(pred_values),
            }
        )
        backend = TransformersSFTBackend(
            tokenizer=tokenizer,
            compute_forecast_metrics=True,
            forecast_metric_length=3,
        )
        labels = np.array([[-100, -100, 1, 2, 3]], dtype=np.int64)
        logits = np.zeros((1, 5, 7), dtype=np.float32)
        logits[0, 1, 4] = 10.0
        logits[0, 2, 5] = 10.0
        logits[0, 3, 6] = 10.0

        metrics = backend._compute_forecast_metrics(logits, labels)

        self.assertAlmostEqual(metrics["mae"], 2.0 / 3.0)
        self.assertAlmostEqual(metrics["mse"], 2.0 / 3.0)
        self.assertEqual(metrics["valid_forecast_samples"], 1)
        self.assertEqual(metrics["invalid_forecast_samples"], 0)

    @unittest.skipUnless(__import__("importlib.util", fromlist=["find_spec"]).find_spec("torch") and __import__("importlib.util", fromlist=["find_spec"]).find_spec("transformers"), "install elecload[train]")
    def test_forecast_eval_trainer_skips_loss_forward(self) -> None:
        import torch
        from contextlib import nullcontext

        class FakeBase:
            prepared = False
            used_context = False

            def _prepare_inputs(self, inputs):
                self.prepared = True
                prepared = dict(inputs)
                prepared["prepared_marker"] = torch.tensor([1])
                return prepared

            def compute_loss_context_manager(self):
                self.used_context = True
                return nullcontext()

        backend = TransformersSFTBackend(compute_forecast_metrics=True, forecast_metric_length=3)
        wrapped_cls = backend._forecast_eval_trainer_cls(FakeBase)
        self.assertTrue(issubclass(wrapped_cls, FakeBase))
        trainer = wrapped_cls.__new__(wrapped_cls)

        seen_keys: set[str] = set()

        logits_tensor = torch.zeros((1, 4, 7))
        logits_tensor[0, :, 3] = 1.0  # argmax over vocab -> token id 3 at every step

        class FakeOutputs:
            logits = logits_tensor

        class FakeModel:
            def __call__(self, **kwargs):
                seen_keys.update(kwargs)
                return FakeOutputs()

        inputs = {
            "input_ids": torch.zeros((1, 4), dtype=torch.long),
            "labels": torch.tensor([[-100, 1, 2, 3]]),
        }

        loss, prediction_ids, labels = trainer.prediction_step(
            FakeModel(), inputs, prediction_loss_only=False
        )

        self.assertTrue(trainer.prepared)  # mirrors Trainer.prediction_step device prep
        self.assertTrue(trainer.used_context)  # keeps bf16/autocast eval context
        self.assertIsNone(loss)  # eval loss unused; no fp32 cross-entropy computed
        self.assertNotIn("labels", seen_keys)  # forward ran without labels -> no CE
        self.assertIn("prepared_marker", seen_keys)
        # Reduced to [batch, seq] greedy ids before pad/gather (no full-vocab tensor).
        self.assertEqual(tuple(prediction_ids.shape), (1, 4))
        self.assertEqual(prediction_ids.tolist(), [[3, 3, 3, 3]])
        self.assertIsNotNone(labels)  # labels still returned for the forecast metric

    @unittest.skipUnless(__import__("importlib.util", fromlist=["find_spec"]).find_spec("torch") and __import__("importlib.util", fromlist=["find_spec"]).find_spec("transformers"), "install elecload[train]")
    def test_response_only_forecast_eval_slices_logits_and_restores_positions(self) -> None:
        import torch
        from contextlib import nullcontext

        tokenizer = FakeDecodeTokenizer(
            {
                (1, 2, 3): _answer_table([1.0, 2.0, 3.0]),
                (4, 5, 6): _answer_table([2.0, 2.0, 4.0]),
            }
        )
        backend = TransformersSFTBackend(
            tokenizer=tokenizer,
            response_only_logits=True,
            compute_forecast_metrics=True,
            forecast_metric_length=3,
        )

        class FakeBase:
            def _prepare_inputs(self, inputs):
                return dict(inputs)

            def compute_loss_context_manager(self):
                return nullcontext()

        trainer_cls = backend._training_trainer_cls(FakeBase)
        trainer = trainer_cls.__new__(trainer_cls)
        trainer.accelerator = SimpleNamespace(unwrap_model=lambda model: model)
        seen: dict[str, object] = {}

        class FakeModel:
            def __call__(
                self,
                input_ids,
                attention_mask,
                logits_to_keep=None,
                use_cache=None,
            ):
                seen["logits_to_keep"] = logits_to_keep
                seen["use_cache"] = use_cache
                keep = logits_to_keep or input_ids.shape[-1]
                token_ids = [4, 5, 6, 0] if keep == 4 else [0, 4, 5, 6, 0]
                logits = torch.zeros((1, keep, 7), dtype=torch.float32)
                for position, token_id in enumerate(token_ids):
                    logits[0, position, token_id] = 1.0
                return SimpleNamespace(logits=logits)

        inputs = {
            "input_ids": torch.tensor([[9, 8, 1, 2, 3]], dtype=torch.long),
            "attention_mask": torch.ones((1, 5), dtype=torch.long),
            "labels": torch.tensor([[-100, -100, 1, 2, 3]], dtype=torch.long),
        }

        loss, prediction_ids, labels = trainer.prediction_step(
            FakeModel(),
            inputs,
            prediction_loss_only=False,
        )

        self.assertIsNone(loss)
        self.assertEqual(seen["logits_to_keep"], 4)
        self.assertIs(seen["use_cache"], False)
        self.assertEqual(prediction_ids.tolist(), [[-100, 4, 5, 6, 0]])
        self.assertEqual(labels.tolist(), inputs["labels"].tolist())
        metrics = backend._compute_forecast_metrics(
            prediction_ids.numpy(),
            labels.numpy(),
        )
        self.assertEqual(metrics["valid_forecast_samples"], 1)
        self.assertAlmostEqual(metrics["mae"], 2.0 / 3.0)
        self.assertAlmostEqual(metrics["mse"], 2.0 / 3.0)

    @unittest.skipUnless(__import__("importlib.util", fromlist=["find_spec"]).find_spec("torch") and __import__("importlib.util", fromlist=["find_spec"]).find_spec("transformers"), "install elecload[train]")
    def test_forecast_eval_trainer_empties_cache_after_eval_loop(self) -> None:
        import torch

        class FakeBase:
            def evaluation_loop(self, *args, **kwargs):
                self.evaluation_args = args
                self.evaluation_kwargs = kwargs
                return {"metrics": {"eval_mae": 1.0}}

        calls: list[str] = []

        class FakeAccelerator:
            @staticmethod
            def empty_cache():
                calls.append("empty_cache")

        backend = TransformersSFTBackend(compute_forecast_metrics=True, forecast_metric_length=3)
        wrapped_cls = backend._forecast_eval_trainer_cls(FakeBase)
        trainer = wrapped_cls.__new__(wrapped_cls)
        original_npu = getattr(torch, "npu", None)
        torch.npu = FakeAccelerator()
        try:
            output = trainer.evaluation_loop("loader", description="Evaluation")
        finally:
            if original_npu is None:
                delattr(torch, "npu")
            else:
                torch.npu = original_npu

        self.assertEqual(output, {"metrics": {"eval_mae": 1.0}})
        self.assertEqual(calls, ["empty_cache"])
        self.assertEqual(trainer.evaluation_args, ("loader",))
        self.assertEqual(trainer.evaluation_kwargs, {"description": "Evaluation"})

    def test_compute_forecast_metrics_accepts_preargmaxed_predictions(self) -> None:
        label_values = [1.0, 2.0, 3.0]
        pred_values = [2.0, 2.0, 4.0]
        tokenizer = FakeDecodeTokenizer(
            {
                (1, 2, 3): _answer_table(label_values),
                (4, 5, 6): _answer_table(pred_values),
            }
        )
        backend = TransformersSFTBackend(
            tokenizer=tokenizer,
            compute_forecast_metrics=True,
            forecast_metric_length=3,
        )
        labels = np.array([[-100, -100, 1, 2, 3]], dtype=np.int64)
        # Already argmaxed to [N, S] as the eval prediction_step now emits.
        prediction_ids = np.array([[0, 4, 5, 6, 0]], dtype=np.int64)

        metrics = backend._compute_forecast_metrics(prediction_ids, labels)

        self.assertAlmostEqual(metrics["mae"], 2.0 / 3.0)
        self.assertAlmostEqual(metrics["mse"], 2.0 / 3.0)
        self.assertEqual(metrics["valid_forecast_samples"], 1)

    def test_compute_forecast_metrics_counts_unparseable_predictions_as_invalid(self) -> None:
        tokenizer = FakeDecodeTokenizer({(1, 2, 3): _answer_table([1.0, 2.0, 3.0])})
        backend = TransformersSFTBackend(
            tokenizer=tokenizer,
            compute_forecast_metrics=True,
            forecast_metric_length=3,
        )
        labels = np.array([[-100, 1, 2, 3]], dtype=np.int64)
        logits = np.zeros((1, 4, 8), dtype=np.float32)
        logits[0, 1:, 7] = 10.0

        metrics = backend._compute_forecast_metrics(logits, labels)

        self.assertTrue(math.isinf(metrics["mae"]))
        self.assertTrue(math.isinf(metrics["mse"]))
        self.assertEqual(metrics["valid_forecast_samples"], 0)
        self.assertEqual(metrics["invalid_forecast_samples"], 1)

    @unittest.skipUnless(__import__("importlib.util", fromlist=["find_spec"]).find_spec("torch") and __import__("importlib.util", fromlist=["find_spec"]).find_spec("transformers"), "install elecload[train]")
    def test_forecast_eval_trainer_finalizes_with_last_step_eval_when_needed(self) -> None:
        backend = TransformersSFTBackend(compute_forecast_metrics=True)

        class FakeTrainer:
            def __init__(self):
                self.state = SimpleNamespace(
                    global_step=141,
                    max_steps=141,
                    log_history=[
                        {"step": 100, "eval_mae": 14.0},
                        {"step": 141, "loss": 0.3},
                    ],
                    best_metric=14.0,
                    best_model_checkpoint="/tmp/checkpoint-100",
                )
                self.args = SimpleNamespace(
                    metric_for_best_model="eval_mae",
                    greater_is_better=False,
                    eval_strategy="steps",
                )
                self.evaluated = 0
                self.saved = 0
                self.events: list[str] = []
                self.eval_dataset = object()
                self.model = object()
                self.control = SimpleNamespace()
                self.callback_handler = SimpleNamespace(
                    on_save=lambda args, state, control: control
                )

            def _evaluate(self, trial, ignore_keys_for_eval, skip_scheduler=False):
                self.evaluated += 1
                self.events.append("evaluate")
                return {"eval_mae": 13.0, "eval_mse": 100.0}

            def _determine_best_metric(self, metrics, trial):
                if metrics["eval_mae"] < self.state.best_metric:
                    self.state.best_metric = metrics["eval_mae"]
                    self.state.best_global_step = self.state.global_step
                    return True
                return False

            def _save_checkpoint(self, model, trial):
                self.saved += 1
                self.events.append("save")
                self.state.best_model_checkpoint = f"/tmp/checkpoint-{self.state.global_step}"

            def _finalize_training(self, trial, num_train_samples, start_time):
                self.events.append("finalize")
                return {"status": "done"}

        wrapped_cls = backend._forecast_eval_trainer_cls(FakeTrainer)
        trainer = wrapped_cls()

        output = trainer._finalize_training(trial=None, num_train_samples=10, start_time=0.0)

        self.assertEqual(output, {"status": "done"})
        self.assertEqual(trainer.evaluated, 1)
        self.assertEqual(trainer.saved, 1)
        self.assertEqual(trainer.state.best_metric, 13.0)
        self.assertEqual(trainer.state.best_model_checkpoint, "/tmp/checkpoint-141")
        self.assertEqual(trainer.events, ["evaluate", "save", "finalize"])

    @unittest.skipUnless(__import__("importlib.util", fromlist=["find_spec"]).find_spec("torch") and __import__("importlib.util", fromlist=["find_spec"]).find_spec("transformers"), "install elecload[train]")
    def test_forecast_eval_trainer_finalizes_without_duplicate_last_step_eval(self) -> None:
        backend = TransformersSFTBackend(compute_forecast_metrics=True)

        class FakeTrainer:
            def __init__(self):
                self.state = SimpleNamespace(
                    global_step=100,
                    max_steps=100,
                    log_history=[
                        {"step": 100, "eval_mae": 14.0},
                    ],
                    best_metric=14.0,
                    best_model_checkpoint="/tmp/checkpoint-100",
                )
                self.args = SimpleNamespace(metric_for_best_model="eval_mae", eval_strategy="steps")
                self.evaluated = 0
                self.eval_dataset = object()

            def _evaluate(self, trial, ignore_keys_for_eval, skip_scheduler=False):
                self.evaluated += 1
                return {"eval_mae": 13.0}

            def _finalize_training(self, trial, num_train_samples, start_time):
                return {"status": "done"}

        wrapped_cls = backend._forecast_eval_trainer_cls(FakeTrainer)
        trainer = wrapped_cls()
        output = trainer._finalize_training(trial=None, num_train_samples=10, start_time=0.0)

        self.assertEqual(output, {"status": "done"})
        self.assertEqual(trainer.evaluated, 0)


class TransformersSFTBackendResponseOnlyLossTest(unittest.TestCase):
    def _trainer(self, **backend_kwargs):
        class FakeBaseTrainer:
            pass

        backend = TransformersSFTBackend(response_only_logits=True, **backend_kwargs)
        self.assertTrue(
            hasattr(backend, "_response_only_loss_trainer_cls"),
            "response-only Trainer wrapper is not implemented",
        )
        trainer_cls = backend._response_only_loss_trainer_cls(FakeBaseTrainer)
        return trainer_cls.__new__(trainer_cls)

    def test_response_only_logits_is_opt_in(self) -> None:
        class FakeBaseTrainer:
            pass

        backend = TransformersSFTBackend()

        self.assertIs(getattr(backend, "response_only_logits", None), False)
        self.assertTrue(
            hasattr(backend, "_training_trainer_cls"),
            "training Trainer composition is not implemented",
        )
        self.assertIs(backend._training_trainer_cls(FakeBaseTrainer), FakeBaseTrainer)

    def test_structure_marker_loss_requires_response_only_logits(self) -> None:
        with self.assertRaisesRegex(ValueError, "structure_marker_loss.*response_only_logits"):
            TransformersSFTBackend(
                structure_marker_loss={"enable": True, "weight": 2.0},
            )

    @unittest.skipUnless(__import__("importlib.util", fromlist=["find_spec"]).find_spec("torch") and __import__("importlib.util", fromlist=["find_spec"]).find_spec("transformers"), "install elecload[train]")
    def test_structure_marker_loss_uses_weighted_token_mean(self) -> None:
        import torch
        import torch.nn.functional as F

        tokenizer = FakeMarkerTokenizer()
        marker_id = tokenizer.marker_ids["<think>"][0]
        normal_left = 3
        normal_right = 4
        vocab_size = 40
        logits = torch.zeros((1, 4, vocab_size), dtype=torch.float32)
        logits[0, 0, normal_left] = 4.0
        logits[0, 1, 5] = 4.0
        logits[0, 2, normal_right] = 4.0

        class FixedLogitsModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.logits = torch.nn.Parameter(logits.clone())

            def forward(self, input_ids, logits_to_keep=0):
                del input_ids
                return SimpleNamespace(logits=self.logits[:, -logits_to_keep:, :])

        labels = torch.tensor(
            [[-100, normal_left, marker_id, normal_right]],
            dtype=torch.long,
        )
        trainer = self._trainer(
            tokenizer=tokenizer,
            structure_marker_loss={"enable": True, "weight": 3.0},
        )
        model = FixedLogitsModel()

        loss = trainer.compute_loss(
            model,
            {"input_ids": torch.ones((1, 4), dtype=torch.long), "labels": labels},
        )

        shifted_labels = torch.tensor(
            [normal_left, marker_id, normal_right, -100],
            dtype=torch.long,
        )
        token_losses = F.cross_entropy(
            logits.reshape(-1, vocab_size),
            shifted_labels,
            ignore_index=-100,
            reduction="none",
        )
        expected = (
            token_losses[0] + 3.0 * token_losses[1] + token_losses[2]
        ) / 5.0
        unweighted = token_losses[:3].mean()

        torch.testing.assert_close(loss, expected)
        self.assertGreater(float(loss.detach()), float(unweighted.detach()))

    @unittest.skipUnless(__import__("importlib.util", fromlist=["find_spec"]).find_spec("torch") and __import__("importlib.util", fromlist=["find_spec"]).find_spec("transformers"), "install elecload[train]")
    def test_structure_marker_loss_injects_component_metrics_at_log_step(self) -> None:
        import torch

        tokenizer = FakeMarkerTokenizer()
        marker_id = tokenizer.marker_ids["<think>"][0]

        class RecordingTrainer:
            def log(self, logs, start_time=None):
                self.logged = dict(logs)
                self.logged_start_time = start_time
                return self.logged

        class FixedLogitsModel(torch.nn.Module):
            def forward(self, input_ids, logits_to_keep=0):
                del input_ids
                logits = torch.zeros((1, logits_to_keep, 40), dtype=torch.float32)
                logits[0, 0, 3] = 4.0
                logits[0, 1, 5] = 4.0
                logits[0, 2, 4] = 4.0
                return SimpleNamespace(logits=logits)

        backend = TransformersSFTBackend(
            tokenizer=tokenizer,
            response_only_logits=True,
            structure_marker_loss={"enable": True, "weight": 2.0},
        )
        trainer_cls = backend._response_only_loss_trainer_cls(RecordingTrainer)
        trainer = trainer_cls.__new__(trainer_cls)
        labels = torch.tensor([[-100, 3, marker_id, 4]], dtype=torch.long)

        trainer.compute_loss(
            FixedLogitsModel(),
            {"input_ids": torch.ones((1, 4), dtype=torch.long), "labels": labels},
        )
        trainer.log({"loss": 1.0}, start_time=7.0)

        self.assertIn("marker_ce", trainer.logged)
        self.assertIn("non_marker_ce", trainer.logged)
        self.assertIn("marker_token_fraction", trainer.logged)
        self.assertGreater(trainer.logged["marker_ce"], trainer.logged["non_marker_ce"])
        self.assertAlmostEqual(trainer.logged["marker_token_fraction"], 1.0 / 3.0)
        self.assertEqual(trainer.logged_start_time, 7.0)

    @unittest.skipUnless(__import__("importlib.util", fromlist=["find_spec"]).find_spec("torch") and __import__("importlib.util", fromlist=["find_spec"]).find_spec("transformers"), "install elecload[train]")
    def test_structure_marker_loss_uses_accumulation_wide_weighted_item_count(self) -> None:
        import torch

        tokenizer = FakeMarkerTokenizer()
        marker_id = tokenizer.marker_ids["<think>"][0]

        class FakeBaseTrainer:
            def _get_num_items_in_batch(self, batch_samples, device):
                del device
                return sum(
                    int(batch["labels"].ne(-100).sum())
                    for batch in batch_samples
                )

        backend = TransformersSFTBackend(
            tokenizer=tokenizer,
            response_only_logits=True,
            structure_marker_loss={"enable": True, "weight": 3.0},
        )
        trainer_cls = backend._response_only_loss_trainer_cls(FakeBaseTrainer)
        trainer = trainer_cls.__new__(trainer_cls)
        trainer.args = SimpleNamespace(average_tokens_across_devices=False, n_gpu=0)
        batches = [
            {"labels": torch.tensor([[-100, 3, marker_id]], dtype=torch.long)},
            {"labels": torch.tensor([[-100, 4, 5]], dtype=torch.long)},
        ]

        count = trainer._get_num_items_in_batch(batches, torch.device("cpu"))

        self.assertEqual(float(count), 6.0)

    @unittest.skipUnless(__import__("importlib.util", fromlist=["find_spec"]).find_spec("torch") and __import__("importlib.util", fromlist=["find_spec"]).find_spec("transformers"), "install elecload[train]")
    def test_structure_marker_loss_divides_sum_by_weighted_num_items(self) -> None:
        import torch
        import torch.nn.functional as F

        tokenizer = FakeMarkerTokenizer()
        marker_id = tokenizer.marker_ids["<think>"][0]
        logits = torch.zeros((1, 4, 40), dtype=torch.float32)
        logits[0, 0, 3] = 4.0
        logits[0, 1, 5] = 4.0
        logits[0, 2, 4] = 4.0

        class FixedLogitsModel(torch.nn.Module):
            def forward(self, input_ids, logits_to_keep=0):
                del input_ids
                return SimpleNamespace(logits=logits[:, -logits_to_keep:, :])

        labels = torch.tensor([[-100, 3, marker_id, 4]], dtype=torch.long)
        trainer = self._trainer(
            tokenizer=tokenizer,
            structure_marker_loss={"enable": True, "weight": 3.0},
        )

        loss = trainer.compute_loss(
            FixedLogitsModel(),
            {"input_ids": torch.ones((1, 4), dtype=torch.long), "labels": labels},
            num_items_in_batch=torch.tensor(10.0),
        )

        shifted_labels = torch.tensor([3, marker_id, 4, -100], dtype=torch.long)
        token_losses = F.cross_entropy(
            logits.reshape(-1, 40),
            shifted_labels,
            ignore_index=-100,
            reduction="none",
        )
        expected = (token_losses[0] + 3.0 * token_losses[1] + token_losses[2]) / 10.0
        torch.testing.assert_close(loss, expected)

    @unittest.skipUnless(__import__("importlib.util", fromlist=["find_spec"]).find_spec("torch") and __import__("importlib.util", fromlist=["find_spec"]).find_spec("transformers"), "install elecload[train]")
    def test_response_only_and_forecast_wrappers_compose(self) -> None:
        class FakeBaseTrainer:
            def compute_loss(self, model, inputs, return_outputs=False):
                return "base-loss"

            def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
                return "base-prediction"

        backend = TransformersSFTBackend(
            response_only_logits=True,
            compute_forecast_metrics=True,
        )

        self.assertTrue(
            hasattr(backend, "_training_trainer_cls"),
            "training Trainer composition is not implemented",
        )
        trainer_cls = backend._training_trainer_cls(FakeBaseTrainer)

        self.assertTrue(issubclass(trainer_cls, FakeBaseTrainer))
        self.assertIsNot(trainer_cls.compute_loss, FakeBaseTrainer.compute_loss)
        self.assertIsNot(trainer_cls.prediction_step, FakeBaseTrainer.prediction_step)

    @unittest.skipUnless(__import__("importlib.util", fromlist=["find_spec"]).find_spec("torch") and __import__("importlib.util", fromlist=["find_spec"]).find_spec("transformers"), "install elecload[train]")
    def test_tiny_qwen3_loss_and_all_parameter_gradients_match_full_logits(self) -> None:
        import torch

        full_model = _tiny_qwen3_model()
        optimized_model = _tiny_qwen3_model()
        optimized_model.load_state_dict(full_model.state_dict())
        full_model.train()
        optimized_model.train()
        input_ids = torch.tensor(
            [
                [3, 4, 5, 6, 7, 8],
                [9, 10, 11, 12, 13, 14],
            ],
            dtype=torch.long,
        )
        attention_mask = torch.ones_like(input_ids)
        labels = torch.tensor(
            [
                [-100, -100, -100, 6, 7, 8],
                [-100, -100, -100, -100, 13, 14],
            ],
            dtype=torch.long,
        )

        full_loss = full_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        ).loss
        full_loss.backward()
        full_gradients = {
            name: None if parameter.grad is None else parameter.grad.detach().clone()
            for name, parameter in full_model.named_parameters()
        }

        trainer = self._trainer()
        optimized_loss = trainer.compute_loss(
            optimized_model,
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            },
        )
        optimized_loss.backward()

        torch.testing.assert_close(optimized_loss, full_loss, rtol=1e-6, atol=1e-7)
        optimized_parameters = dict(optimized_model.named_parameters())
        self.assertEqual(set(optimized_parameters), set(full_gradients))
        for name, parameter in optimized_parameters.items():
            expected_gradient = full_gradients[name]
            if expected_gradient is None:
                self.assertIsNone(parameter.grad, name)
            else:
                self.assertIsNotNone(parameter.grad, name)
                torch.testing.assert_close(
                    parameter.grad,
                    expected_gradient,
                    rtol=1e-5,
                    atol=1e-6,
                    msg=lambda message, parameter_name=name: f"{parameter_name}: {message}",
                )

    @unittest.skipUnless(__import__("importlib.util", fromlist=["find_spec"]).find_spec("torch") and __import__("importlib.util", fromlist=["find_spec"]).find_spec("transformers"), "install elecload[train]")
    def test_model_receives_response_length_plus_one_logits(self) -> None:
        import torch

        class RecordingModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model = _tiny_qwen3_model()
                self.logits_to_keep = None

            def forward(self, *args, logits_to_keep=0, **kwargs):
                self.logits_to_keep = logits_to_keep
                return self.model(*args, logits_to_keep=logits_to_keep, **kwargs)

        model = RecordingModel()
        trainer = self._trainer()
        labels = torch.tensor([[-100, -100, -100, 6, 7]], dtype=torch.long)

        _, outputs = trainer.compute_loss(
            model,
            {
                "input_ids": torch.tensor([[3, 4, 5, 6, 7]], dtype=torch.long),
                "attention_mask": torch.ones((1, 5), dtype=torch.long),
                "labels": labels,
            },
            return_outputs=True,
        )

        self.assertEqual(model.logits_to_keep, 3)
        self.assertEqual(outputs.logits.shape[-2], 3)

    @unittest.skipUnless(__import__("importlib.util", fromlist=["find_spec"]).find_spec("torch") and __import__("importlib.util", fromlist=["find_spec"]).find_spec("transformers"), "install elecload[train]")
    def test_variable_prompt_lengths_right_padding_and_all_ignore_row_are_exact(self) -> None:
        import torch

        full_model = _tiny_qwen3_model()
        optimized_model = _tiny_qwen3_model()
        optimized_model.load_state_dict(full_model.state_dict())
        input_ids = torch.tensor(
            [
                [3, 4, 5, 6, 0, 0],
                [7, 8, 9, 10, 11, 0],
                [12, 13, 0, 0, 0, 0],
            ],
            dtype=torch.long,
        )
        attention_mask = torch.tensor(
            [
                [1, 1, 1, 1, 0, 0],
                [1, 1, 1, 1, 1, 0],
                [1, 1, 0, 0, 0, 0],
            ],
            dtype=torch.long,
        )
        labels = torch.tensor(
            [
                [-100, -100, 5, 6, -100, -100],
                [-100, -100, -100, -100, 11, -100],
                [-100, -100, -100, -100, -100, -100],
            ],
            dtype=torch.long,
        )

        full_loss = full_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        ).loss
        optimized_loss = self._trainer().compute_loss(
            optimized_model,
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            },
        )

        torch.testing.assert_close(optimized_loss, full_loss, rtol=1e-6, atol=1e-7)

    @unittest.skipUnless(__import__("importlib.util", fromlist=["find_spec"]).find_spec("torch") and __import__("importlib.util", fromlist=["find_spec"]).find_spec("transformers"), "install elecload[train]")
    def test_num_items_in_batch_uses_sum_then_count_reduction(self) -> None:
        import torch
        import torch.nn.functional as F

        class FixedLogitsModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.full_logits = torch.nn.Parameter(
                    torch.tensor(
                        [
                            [
                                [0.1, 0.2, 0.3, 0.4],
                                [0.5, 0.6, 0.7, 0.8],
                                [0.9, 1.0, 1.1, 1.2],
                                [1.3, 1.4, 1.5, 1.6],
                            ]
                        ],
                        dtype=torch.float32,
                    )
                )

            def forward(self, input_ids, logits_to_keep=0):
                return SimpleNamespace(logits=self.full_logits[:, -logits_to_keep:, :])

        model = FixedLogitsModel()
        labels = torch.tensor([[-100, -100, 1, 2]], dtype=torch.long)
        num_items = torch.tensor(7)
        trainer = self._trainer()

        loss = trainer.compute_loss(
            model,
            {"input_ids": torch.ones((1, 4), dtype=torch.long), "labels": labels},
            num_items_in_batch=num_items,
        )

        expected_shift_labels = torch.tensor([[1, 2, -100]], dtype=torch.long)
        expected = F.cross_entropy(
            model.full_logits[:, -3:, :].reshape(-1, 4),
            expected_shift_labels.reshape(-1),
            ignore_index=-100,
            reduction="sum",
        ) / num_items
        torch.testing.assert_close(loss, expected)

    @unittest.skipUnless(__import__("importlib.util", fromlist=["find_spec"]).find_spec("torch") and __import__("importlib.util", fromlist=["find_spec"]).find_spec("transformers"), "install elecload[train]")
    def test_distributed_token_average_scaling_matches_transformers_trainer(self) -> None:
        import torch
        from transformers import Trainer

        full_model = _tiny_qwen3_model()
        optimized_model = _tiny_qwen3_model()
        optimized_model.load_state_dict(full_model.state_dict())
        input_ids = torch.tensor(
            [
                [3, 4, 5, 6, 7, 8],
                [9, 10, 11, 12, 13, 14],
            ],
            dtype=torch.long,
        )
        labels = torch.tensor(
            [
                [-100, -100, -100, 6, 7, 8],
                [-100, -100, -100, -100, 13, 14],
            ],
            dtype=torch.long,
        )
        num_items = torch.tensor(11)

        def configure_token_average(trainer) -> None:
            trainer.args = SimpleNamespace(average_tokens_across_devices=True, n_gpu=0)
            trainer.accelerator = SimpleNamespace(parallelism_config=None, num_processes=8)
            trainer.model_accepts_loss_kwargs = True
            trainer.compute_loss_func = None
            trainer.label_smoother = None

        full_trainer = Trainer.__new__(Trainer)
        configure_token_average(full_trainer)
        full_loss = Trainer.compute_loss(
            full_trainer,
            full_model,
            {"input_ids": input_ids, "labels": labels},
            num_items_in_batch=num_items,
        )
        full_loss.backward()

        backend = TransformersSFTBackend(response_only_logits=True)
        trainer_cls = backend._response_only_loss_trainer_cls(Trainer)
        optimized_trainer = trainer_cls.__new__(trainer_cls)
        configure_token_average(optimized_trainer)
        optimized_loss = optimized_trainer.compute_loss(
            optimized_model,
            {"input_ids": input_ids, "labels": labels},
            num_items_in_batch=num_items,
        )
        optimized_loss.backward()

        torch.testing.assert_close(optimized_loss, full_loss, rtol=1e-6, atol=1e-7)
        full_parameters = dict(full_model.named_parameters())
        for name, parameter in optimized_model.named_parameters():
            torch.testing.assert_close(
                parameter.grad,
                full_parameters[name].grad,
                rtol=1e-5,
                atol=1e-6,
                msg=lambda message, parameter_name=name: f"{parameter_name}: {message}",
            )

    @unittest.skipUnless(__import__("importlib.util", fromlist=["find_spec"]).find_spec("torch") and __import__("importlib.util", fromlist=["find_spec"]).find_spec("transformers"), "install elecload[train]")
    def test_return_outputs_returns_the_sliced_model_output(self) -> None:
        import torch

        model = _tiny_qwen3_model()
        trainer = self._trainer()

        loss, outputs = trainer.compute_loss(
            model,
            {
                "input_ids": torch.tensor([[3, 4, 5, 6]], dtype=torch.long),
                "labels": torch.tensor([[-100, -100, 5, 6]], dtype=torch.long),
            },
            return_outputs=True,
        )

        self.assertEqual(loss.ndim, 0)
        self.assertEqual(tuple(outputs.logits.shape), (1, 3, 41))

    @unittest.skipUnless(__import__("importlib.util", fromlist=["find_spec"]).find_spec("torch") and __import__("importlib.util", fromlist=["find_spec"]).find_spec("transformers"), "install elecload[train]")
    def test_all_ignore_batch_raises_before_calling_model(self) -> None:
        import torch

        class ModelMustNotRun:
            def __call__(self, **kwargs):
                raise AssertionError("model should not be called")

        trainer = self._trainer()

        with self.assertRaisesRegex(ValueError, "no supervised labels"):
            trainer.compute_loss(
                ModelMustNotRun(),
                {
                    "input_ids": torch.ones((2, 4), dtype=torch.long),
                    "labels": torch.full((2, 4), -100, dtype=torch.long),
                },
            )

    @unittest.skipUnless(__import__("importlib.util", fromlist=["find_spec"]).find_spec("torch") and __import__("importlib.util", fromlist=["find_spec"]).find_spec("transformers"), "install elecload[train]")
    def test_unsupported_logits_to_keep_has_clear_compatibility_error(self) -> None:
        import torch

        class UnsupportedModel:
            def __call__(self, input_ids):
                return SimpleNamespace(logits=torch.zeros((1, 4, 8)))

        trainer = self._trainer()

        with self.assertRaisesRegex(TypeError, "response_only_logits.*logits_to_keep"):
            trainer.compute_loss(
                UnsupportedModel(),
                {
                    "input_ids": torch.ones((1, 4), dtype=torch.long),
                    "labels": torch.tensor([[-100, -100, 1, 2]], dtype=torch.long),
                },
            )

    @unittest.skipUnless(__import__("importlib.util", fromlist=["find_spec"]).find_spec("torch") and __import__("importlib.util", fromlist=["find_spec"]).find_spec("transformers"), "install elecload[train]")
    def test_unrelated_model_type_error_is_not_masked(self) -> None:
        import torch

        class BrokenModel:
            def __call__(self, input_ids, logits_to_keep=0):
                raise TypeError("internal sentinel")

        trainer = self._trainer()

        with self.assertRaisesRegex(TypeError, "^internal sentinel$"):
            trainer.compute_loss(
                BrokenModel(),
                {
                    "input_ids": torch.ones((1, 4), dtype=torch.long),
                    "labels": torch.tensor([[-100, -100, 1, 2]], dtype=torch.long),
                },
            )

    @unittest.skipUnless(__import__("importlib.util", fromlist=["find_spec"]).find_spec("torch") and __import__("importlib.util", fromlist=["find_spec"]).find_spec("transformers"), "install elecload[train]")
    def test_internal_keyword_shaped_type_error_is_not_masked(self) -> None:
        import torch

        class BrokenModel:
            def __call__(self, input_ids, logits_to_keep=0):
                raise TypeError("helper() got an unexpected keyword argument 'logits_to_keep'")

        trainer = self._trainer()

        with self.assertRaisesRegex(
            TypeError,
            "^helper\\(\\) got an unexpected keyword argument 'logits_to_keep'$",
        ):
            trainer.compute_loss(
                BrokenModel(),
                {
                    "input_ids": torch.ones((1, 4), dtype=torch.long),
                    "labels": torch.tensor([[-100, -100, 1, 2]], dtype=torch.long),
                },
            )

    @unittest.skipUnless(__import__("importlib.util", fromlist=["find_spec"]).find_spec("torch") and __import__("importlib.util", fromlist=["find_spec"]).find_spec("transformers"), "install elecload[train]")
    def test_forwarding_wrapper_uses_unwrapped_model_for_compatibility(self) -> None:
        import torch

        class ForwardingWrapper:
            def __init__(self, module):
                self.module = module

            def __call__(self, *args, **kwargs):
                return self.module(*args, **kwargs)

        class UnsupportedLeaf:
            def __call__(self, input_ids):
                return SimpleNamespace(logits=torch.zeros((1, 4, 8)))

        class BrokenSupportedLeaf:
            def __call__(self, input_ids, logits_to_keep=0):
                raise TypeError("helper() got an unexpected keyword argument 'logits_to_keep'")

        trainer = self._trainer()
        trainer.accelerator = SimpleNamespace(unwrap_model=lambda model: model.module)
        inputs = {
            "input_ids": torch.ones((1, 4), dtype=torch.long),
            "labels": torch.tensor([[-100, -100, 1, 2]], dtype=torch.long),
        }

        with self.assertRaisesRegex(TypeError, "response_only_logits.*logits_to_keep"):
            trainer.compute_loss(ForwardingWrapper(UnsupportedLeaf()), inputs)

        with self.assertRaisesRegex(
            TypeError,
            "^helper\\(\\) got an unexpected keyword argument 'logits_to_keep'$",
        ):
            trainer.compute_loss(ForwardingWrapper(BrokenSupportedLeaf()), inputs)


if __name__ == "__main__":
    unittest.main()
