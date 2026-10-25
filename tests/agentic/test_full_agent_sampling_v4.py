"""CPU/tokenizer witnesses; no model weights, accelerator or API traffic."""
import asyncio
from datetime import datetime, timedelta
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from castfactory.agentic import distillation_protocol as protocol
from castfactory.agentic.full_agent_v4 import Profile
from castfactory.training.full_agent_sampling_v4 import (
    ExactIDRoleCallerV4, NativeFullRoleSamplerV4, SamplingTransportErrorV4,
    excess_t2_values_v4, protocol_terminal_v4, sampling_params_v4,
)
from castfactory.training import full_agent_serialization as serialization


class Tokenizer:
    eos_token_id = 0

    def decode(self, ids, **kwargs):
        return "".join(chr(i) if i else "<EOS>" for i in ids)


def token_ids(text):
    return [ord(c) for c in text]


class Backend:
    def __init__(self, ids, logs=None, *, extra=None, stop="completed"):
        self.output = {"token_ids": ids, "log_probs": [-1.] * len(ids) if logs is None else logs,
                       "extra_fields": {"actor_version": "actor-7"} if extra is None else extra,
                       "stop_reason": stop}
        self.requests = []

    async def generate(self, **request):
        self.requests.append(request)
        return self.output


class V4SamplingTests(unittest.TestCase):
    def setUp(self):
        self.messages = [{"role": "user", "content": "Public evidence"}]
        self.tokenizer = Tokenizer()
        self.prompt_patch = patch.object(protocol, "student_prompt_tokens", return_value=([41, 42], "<think>\n"))
        self.prompt_patch.start()
        self.addCleanup(self.prompt_patch.stop)

    def call(self, ids, *, budget=None, logs=None, extra=None, stochastic=True, role="T3"):
        backend = Backend(ids, logs, extra=extra)
        caller = ExactIDRoleCallerV4(backend, self.tokenizer,
                                    Profile(max_response_tokens=budget or len(ids)),
                                    "actor-7", seed=19, stochastic=stochastic,
                                    request_prefix="fixture")
        return caller, backend, lambda: asyncio.run(caller.call(role, self.messages))

    def test_exact_sampled_tokens_logprobs_and_prompt_owned_prefix(self):
        continuation = 'Evidence.</think>\n<answer>{"action":"STOP","segment_id":null,"directive":null}</answer>'
        ids = token_ids(continuation) + [0]
        logs = [-0.25 - i / 1000 for i in range(len(ids))]
        caller, backend, invoke = self.call(ids, logs=logs)
        receipt = invoke()
        self.assertEqual(receipt["prompt_ids"], [41, 42])
        self.assertEqual(receipt["response_ids"], ids)
        self.assertEqual(receipt["old_log_probs"], logs)
        self.assertEqual(receipt["content"], "<think>\n" + continuation)
        self.assertEqual(receipt["continuation_content"], continuation)
        self.assertEqual(receipt["response_loss_mask"], [1] * len(ids))
        self.assertEqual(receipt["finish_reason"], "stop")
        self.assertEqual(receipt["decode_tokens"], len(ids))
        self.assertEqual(backend.requests[0]["prompt_ids"], [41, 42])
        self.assertEqual(backend.requests[0]["sampling_params"]["stop"], [])
        self.assertEqual(backend.requests[0]["sampling_params"]["temperature"], 1.)
        self.assertFalse(any("sha" in key or "hash" in key for key in receipt))
        self.assertEqual(caller.transport_receipts, [])

    def test_partial_full_budget_retained_and_no_eos_invented(self):
        ids = token_ids('Evidence.</think><answer>{"values":[1,')
        _, _, invoke = self.call(ids)
        receipt = invoke()
        self.assertEqual(receipt["response_ids"], ids)
        self.assertEqual(receipt["finish_reason"], "length")
        self.assertTrue(receipt["generation_limit_reached"])
        self.assertTrue(receipt["content"].endswith("[1,"))

    def test_actual_eos_invalid_json_and_duplicate_prefix_are_failures(self):
        for text in ('Evidence.</think><answer>{"values":[NaN]}</answer>',
                     'Evidence.</think><answer>{"x":1,"x":2}</answer>',
                     'Evidence.</think><answer>{}</answer><answer>{}</answer>',
                     '<think>Evidence.</think><answer>{}</answer>'):
            ids = token_ids(text) + [0]
            _, _, invoke = self.call(ids)
            receipt = invoke()
            self.assertEqual(receipt["finish_reason"], "contract_error")
            self.assertEqual(receipt["response_ids"], ids)

    def test_transport_failures_retain_actual_calls_and_actor_ack(self):
        scenarios = [(token_ids("partial"), 30, [-1.] * 7, None),
                     (token_ids("partial"), 7, [], None),
                     (token_ids("partial"), 7, [-1.] * 7, {}),
                     ([65, 0, 66], 3, [-1.] * 3, None)]
        for ids, budget, logs, extra in scenarios:
            caller, _, invoke = self.call(ids, budget=budget, logs=logs, extra=extra)
            with self.assertRaises(SamplingTransportErrorV4):
                invoke()
            self.assertEqual(caller.transport_receipts[0]["response_ids"], ids)
            self.assertEqual(caller.transport_receipts[0]["old_log_probs"], logs)
            self.assertIn("content", caller.transport_receipts[0])

    def test_explicit_deterministic_policy_and_no_numeric_logit_processors(self):
        _, backend, invoke = self.call(token_ids("partial"), stochastic=False)
        invoke()
        params = backend.requests[0]["sampling_params"]
        self.assertEqual(params["temperature"], 0.)
        self.assertEqual(params["top_p"], 1.)
        self.assertEqual(params["top_k"], -1)
        for supplied in ({"temperature": 0.7}, {"stop": ["</answer>"]},
                         {"logits_processors": ["numbers"]}):
            with self.assertRaises(ValueError):
                sampling_params_v4(Profile(), 17, 0, supplied, True)

    def test_backend_exception_has_prompt_but_no_invented_generation(self):
        class BrokenBackend:
            async def generate(self, **request):
                raise RuntimeError("fixture transport interrupted")

        caller = ExactIDRoleCallerV4(BrokenBackend(), self.tokenizer, Profile(),
                                    "actor-7", seed=19)
        with self.assertRaises(SamplingTransportErrorV4):
            asyncio.run(caller.call("T1", self.messages))
        receipt = caller.transport_receipts[0]
        self.assertEqual(receipt["prompt_ids"], [41, 42])
        self.assertEqual(receipt["response_ids"], [])
        self.assertIsNone(receipt["old_log_probs"])
        self.assertEqual(receipt["finish_reason"], "transport_error")

    def test_overlong_partial_values_h24_h96_and_segment_lengths(self):
        for length in (6, 24, 48, 96):
            values = ",".join(str(i) for i in range(length))
            base = '<think>Evidence.</think><answer>{"note":"values", "values":[' + values
            self.assertFalse(excess_t2_values_v4(base, length))
            self.assertFalse(excess_t2_values_v4(base + "]", length))
            self.assertTrue(excess_t2_values_v4(base + ",", length))
            self.assertTrue(excess_t2_values_v4(base + ",99]", length))
            self.assertFalse(excess_t2_values_v4('<answer>{"nested":{"values":[1,2,3]},"values":[1]', 1))

    def test_role_validation_rejects_overlong_without_repairing_values(self):
        for h, start, stop in ((24, 0, 24), (96, 0, 96), (96, 24, 48)):
            segment = {"segment_id": "S0", "start": start, "stop": stop}
            expected = stop - start
            variables = {"geometry": {"H": h}, "segment_assignment": segment}
            answer = {**segment, "values": [1] * (expected + 1),
                      "uncertainty": "Unknown", "self_check": "Public check"}
            content = protocol.render_reply("Public segment.", answer, "student")
            result = protocol_terminal_v4("T2", variables, Profile(), content)
            self.assertEqual(result[0], "contract_error")
            self.assertEqual(len(protocol.parse_reply(content, "student")["answer"]["values"]), expected + 1)

    def test_native_cpu_records_actual_selected_tokens_and_eos(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch not installed")
        ids = token_ids('Evidence.</think><answer>{"action":"STOP","segment_id":null,"directive":null}</answer>') + [0]

        class Model:
            def __init__(self):
                self.index = 0
                self.calls = []

            def eval(self):
                return self

            def __call__(self, **kwargs):
                self.calls.append(kwargs)
                logits = torch.full((1, 1, 128), -12.)
                logits[0, 0, ids[self.index]] = 4.
                self.index += 1
                return SimpleNamespace(logits=logits, past_key_values=(self.index,))

        model = Model()
        caller = NativeFullRoleSamplerV4(model, self.tokenizer,
                                        Profile(max_response_tokens=len(ids) + 2),
                                        "frozen-actor", stochastic=False, device="cpu")
        receipt = caller("T3", self.messages)
        expected = float(torch.log_softmax(torch.tensor([4.] + [-12.] * 127), -1)[0])
        self.assertEqual(receipt["response_ids"], ids)
        self.assertTrue(all(abs(lp - expected) < 1e-6 for lp in receipt["old_log_probs"]))
        self.assertEqual(model.calls[0]["input_ids"].tolist(), [[41, 42]])
        self.assertEqual(model.calls[1]["past_key_values"], (1,))
        self.assertEqual(receipt["finish_reason"], "stop")

    def test_native_failure_retains_tokens_sampled_before_failure(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch not installed")

        class BrokenModel:
            calls = 0

            def eval(self):
                return self

            def __call__(self, **kwargs):
                self.calls += 1
                if self.calls == 2:
                    raise RuntimeError("fixture model interrupted")
                logits = torch.full((1, 1, 128), -12.)
                logits[0, 0, ord("E")] = 4.
                return SimpleNamespace(logits=logits, past_key_values=(1,))

        caller = NativeFullRoleSamplerV4(BrokenModel(), self.tokenizer,
                                        Profile(max_response_tokens=3), "frozen-actor",
                                        stochastic=False, device="cpu")
        with self.assertRaises(SamplingTransportErrorV4):
            caller("T1", self.messages)
        receipt = caller.transport_receipts[0]
        self.assertEqual(receipt["response_ids"], [ord("E")])
        self.assertEqual(len(receipt["old_log_probs"]), 1)
        self.assertEqual(receipt["content"], "<think>\nE")
        self.assertEqual(receipt["finish_reason"], "transport_error")


class NativeTokenizerParityTests(unittest.TestCase):
    @unittest.skipUnless(Path(os.environ.get("FULL_AGENT_TOKENIZER", str(Path(__import__("os").environ.get("ELECLOAD_MODELS_DIR", "models")) / "Qwen3.5-4B"))).is_dir(),
                         "local Qwen3.5-4B tokenizer absent")
    def test_qwen35_distillation_and_production_exact_prompt_response_parity(self):
        try:
            from transformers import AutoTokenizer
        except ImportError:
            self.skipTest("transformers not installed")
        tokenizer = AutoTokenizer.from_pretrained(
            os.environ.get("FULL_AGENT_TOKENIZER", str(Path(__import__("os").environ.get("ELECLOAD_MODELS_DIR", "models")) / "Qwen3.5-4B")), local_files_only=True)
        for h, start, stop in ((24, 0, 24), (96, 0, 96), (96, 24, 72)):
            variables = {"geometry": {"H": h, "unit": None, "frequency": "1h",
                         "future_timestamps": [(datetime(2026, 10, 10) + timedelta(hours=i)).isoformat()
                                               for i in range(h)]}, "public_case": "History supports the cycle.",
                         "segment_assignment": {"segment_id": "S0", "start": start, "stop": stop},
                         "local_t1_information": {}}
            messages = protocol.messages_for("T2", variables, "student")
            failed = '<think>Actual failed public explanation.</think><answer>{"values":[1,'
            messages += [{"role": "assistant", "content": failed, "reasoning_content": "PRIVATE_SENTINEL"},
                         {"role": "user", "content": "Retry the unchanged full segment."}]
            output = protocol.render_reply("The public history supports a stable level.",
                                           {**variables["segment_assignment"], "values": [123.456789] * (stop-start),
                                            "uncertainty": "Unknown", "self_check": "Public check"}, "student")
            prompt, prefix = protocol.student_prompt_tokens(tokenizer, messages)
            self.assertEqual(prefix, "<think>\n")
            self.assertEqual(serialization.student_prompt(tokenizer, messages, profile_version=protocol.VERSION),
                             (prompt, prefix))
            trained_prompt, response = protocol.student_training_tokens(tokenizer, messages, output)
            self.assertEqual(prompt, trained_prompt)
            self.assertEqual(serialization.student_training_tokens(tokenizer, messages, output,
                             profile_version=protocol.VERSION), (prompt, response))
            rendered = tokenizer.decode(prompt, skip_special_tokens=False)
            self.assertIn(failed, rendered)
            self.assertNotIn("PRIVATE_SENTINEL", rendered)
            self.assertIn(tokenizer.eos_token_id, response)
            self.assertEqual(response[-1], tokenizer.eos_token_id)
            self.assertEqual(response.count(tokenizer.eos_token_id), 1)
            eos_index = response.index(tokenizer.eos_token_id)
            decoded = tokenizer.decode(response[:eos_index], skip_special_tokens=False,
                                       clean_up_tokenization_spaces=False)
            self.assertFalse(decoded.startswith("<think>"))
            parsed = protocol.parse_reply(prefix + decoded, "student")
            self.assertEqual(parsed["answer"]["values"], [123.456789] * (stop-start))
            identity = serialization.student_serialization_identity(tokenizer, profile_version=protocol.VERSION)
            self.assertTrue(identity["enable_thinking"])
            self.assertFalse(any("sha" in key or "hash" in key for key in identity))


if __name__ == "__main__":
    unittest.main()
