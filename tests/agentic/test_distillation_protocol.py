import unittest
import copy
import json
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

from castfactory.agentic.distillation_protocol import (
    STUDENT_ENABLE_THINKING, T1_SCHEMA, VERSION, messages_for, parse_reply,
    render_reply, review_reasoning, review_t1, validate_values,
    student_prompt_tokens, student_training_tokens,
    prompt_snapshot,
)


class DistillationProtocolTests(unittest.TestCase):
    def test_prompt_snapshot_preserves_exact_resource_text_without_hashes(self):
        snapshot = prompt_snapshot()
        self.assertEqual(len(snapshot), 20)
        self.assertEqual(snapshot["t1_schema.txt"].rstrip("\n"), T1_SCHEMA)
        self.assertTrue(snapshot["teacher_format.txt"].endswith("\n"))
        self.assertEqual(set(snapshot), {"public_system.txt", "teacher_format.txt",
                                        "student_format.txt", "t1_schema.txt"} |
                         {role + "_" + part + ".txt"
                          for role in ("reasoning", "reasoning_review", "T1", "T1_review", "T2", "T2_refine", "T3", "T4")
                          for part in ("system", "user")})
        snapshot["teacher_format.txt"] = "changed in caller memory"
        self.assertNotEqual(prompt_snapshot()["teacher_format.txt"], snapshot["teacher_format.txt"])

    def test_teacher_and_student_roundtrip_preserve_values(self):
        values = [9007199254740993, 0.12345678901234568, -0.0, 2]
        for response_format in ("teacher", "student"):
            with self.subTest(response_format=response_format):
                explanation = 'History supports the level; "oracle" context remains qualified.'
                result = parse_reply(render_reply(explanation, {"values": values}, response_format), response_format)
                self.assertEqual(result, {"explanation": explanation, "answer": {"values": values}})
                self.assertIsInstance(result["answer"]["values"][0], int)
        self.assertTrue(STUDENT_ENABLE_THINKING)
        self.assertEqual(VERSION, "merged-t1-answer-v1")

    def test_teacher_student_formats_are_explicit(self):
        teacher = render_reply("Evidence supports this forecast.", {"values": [1]})
        student = render_reply("Evidence supports this forecast.", {"values": [1]}, "student")
        for text, response_format in ((teacher, "student"), (student, "teacher")):
            with self.assertRaises(ValueError):
                parse_reply(text, response_format)
        with self.assertRaises(ValueError):
            parse_reply(student.removeprefix("<think>"), "student")

    def test_strict_json_and_single_answer_reject_invalid_output(self):
        invalid = [
            '<answer>{"values":[1]}</answer>',
            '  <answer>{"values":[1]}</answer>',
            'Evidence. <answer>{"values":[1],"values":[2]}</answer>',
            'Evidence. <answer>{"nested":{"x":1,"x":2}}</answer>',
            'Evidence. <answer>{"values":[NaN]}</answer>',
            'Evidence. <answer>{"values":[Infinity]}</answer>',
            'Evidence. <answer>{"values":[-Infinity]}</answer>',
            'Evidence. <answer>{"values":[1e999]}</answer>',
            'Evidence. <answer>[1]</answer>',
            'Evidence. <answer>{"values":[1]} {}</answer>',
            'Evidence. <answer>{"values":[1]}</answer> done',
            'Evidence. <answer>{}</answer><answer>{}</answer>',
            'Evidence. <answer >{}</answer>',
            'Evidence. <ANSWER>{}</ANSWER>',
            'Evidence. <answer>{}</answer></answer',
            'Evidence. <payload>{}</payload><answer>{}</answer>',
            'Evidence. ``` <answer>{}</answer>',
            'Evidence. <answer>{"note":"<answer>"}</answer>',
        ]
        for text in invalid:
            with self.subTest(text=text):
                with self.assertRaises(ValueError):
                    parse_reply(text)

    def test_student_requires_single_nonempty_visible_think(self):
        invalid = [
            '<think> </think><answer>{}</answer>',
            '<think>Evidence.</think>outside<answer>{}</answer>',
            '<think>Evidence.</think><think>Other.</think><answer>{}</answer>',
            '<think>Evidence.</think><answer>{}</answer>trailing',
        ]
        for text in invalid:
            with self.assertRaises(ValueError):
                parse_reply(text, "student")

    def test_values_validation_does_not_coerce_or_round(self):
        values = [1, 1.23456789012345, -0.0]
        accepted = validate_values(values, 3)
        self.assertEqual(accepted, values)
        self.assertIsNot(accepted, values)
        self.assertIsInstance(accepted[0], int)
        for values, length in (([True], 1), (["1"], 1), ([float("nan")], 1),
                               ([float("inf")], 1), ([1], True), ([1], 2), ((1,), 1)):
            with self.assertRaises(ValueError):
                validate_values(values, length)

    def test_render_rejects_nonfinite_and_wrapper_injection(self):
        for explanation, answer in (("", {}), ("<think>Secret</think>", {}),
                                    ("Evidence.", {"values": [float("inf")]}),
                                    ("Evidence.", {"note": "</answer>"})):
            with self.assertRaises(ValueError):
                render_reply(explanation, answer)

    def test_reasoning_keep_rewrite_are_complete_exact_decisions(self):
        original = "Evidence remains uncertain.\n\n  Retrospective context is qualified."
        keep = render_reply("Audit: grounded.", {"decision": "KEEP"})
        self.assertEqual(review_reasoning(keep, original), original)
        replacement = "History supplies the main evidence."
        rewrite = render_reply("Audit: unsupported cause removed.", {"decision": "REWRITE", "explanation": replacement})
        self.assertEqual(review_reasoning(rewrite, original), replacement)
        invalid = [{"decision": "KEEP", "explanation": replacement},
                   {"decision": "rewrite", "explanation": replacement},
                   {"decision": "REWRITE"},
                   {"decision": "REWRITE", "explanation": " "},
                   {"decision": "REWRITE", "explanation": replacement, "values": [1]}]
        for answer in invalid:
            with self.assertRaises(ValueError):
                review_reasoning(render_reply("Audit.", answer), original)

    def test_t1_review_returns_complete_accepted_public_reply(self):
        t1 = {"drivers": [], "global_uncertainty": "Unknown drivers.", "evidence_ids_used": [],
              "mode": "DIRECT", "spans": [{"segment_id": "S0", "start": 0, "stop": 24}]}
        original = parse_reply(render_reply("History drives the plan.", t1))
        keep = render_reply("Audit: suitable.", {"decision": "KEEP"})
        self.assertEqual(review_t1(keep, original), original)
        changed = dict(t1, global_uncertainty="Context is retrospective.")
        rewrite = render_reply("Audit: qualified context.", {"decision": "REWRITE", "explanation": "Context is uncertain.", "t1": changed})
        self.assertEqual(review_t1(rewrite, original), {"explanation": "Context is uncertain.", "answer": changed})
        for answer in ({"decision": "REWRITE", "explanation": "New."},
                       {"decision": "REWRITE", "explanation": "New.", "t1": []},
                       {"decision": "KEEP", "t1": t1}):
            with self.assertRaises(ValueError):
                review_t1(render_reply("Audit.", answer), original)

    def test_all_roles_render_declared_structured_variables(self):
        variables = {
            "public_case": {"history": [1, 2], "context": "quoted {{canonical_labels}} is data"},
            "geometry": {"H": 24, "unit": "kWh", "frequency": "1h",
                         "future_timestamps": [(datetime(2026, 10, 10) + timedelta(hours=i)).isoformat()
                                               for i in range(24)]}, "evidence_manifest": [],
            "compute_policy": {"max_segments": 2}, "local_t1_information": {},
            "segment_assignment": {"segment_id": "S0", "start": 0, "stop": 24},
            "accepted_t1": {}, "current_segment_reports": [], "reflection_history": [],
            "reflection_policy_state": {}, "own_report_history": [], "bound_local_feedback": {},
            "final_segment_reports": [], "reflection_history_and_termination": {}, "t4_policy": {},
            "pass1_public_response": "Original.", "pass1_values": [1],
            "t1_pass1_public_response": "Original plan.", "canonical_labels": [987654321],
        }
        public_roles = ("reasoning", "T1", "T2", "T2_refine", "T3", "T4")
        for role in public_roles + ("reasoning_review", "T1_review"):
            messages = messages_for(role, variables)
            self.assertEqual([m["role"] for m in messages], ["system", "user"])
            self.assertIn('quoted {{canonical_labels}} is data', messages[1]["content"])
            if role in public_roles:
                self.assertNotIn("987654321", messages[1]["content"])
                student = messages_for(role, variables, "student")
                self.assertEqual(student[1], messages[1])
                self.assertIn("inside one <think>", student[0]["content"])
            else:
                self.assertIn("987654321", messages[1]["content"])
                self.assertIn("offline", messages[0]["content"])
                with self.assertRaises(ValueError):
                    messages_for(role, variables, "student")
        self.assertIn(T1_SCHEMA, messages_for("T1", variables)[1]["content"])
        with self.assertRaises(ValueError):
            messages_for("reasoning", {"public_case": "case"})
        with self.assertRaises(ValueError):
            messages_for("T0", variables)
        with self.assertRaises(ValueError):
            messages_for("T1", variables, "native")

    def _segment_variables(self, h=96, start=24, stop=72):
        return {
            "geometry": {"H": h, "unit": None, "frequency": "1h",
                         "future_timestamps": [(datetime(2026, 10, 10) + timedelta(hours=i)).isoformat()
                                               for i in range(h)]},
            "public_case": "Exact evidence: oracle temperatures; event {{required_points}} is source text.",
            "segment_assignment": {"segment_id": "S1", "start": start, "stop": stop},
            "local_t1_information": {"qualification": "unproven mechanism"},
            "own_report_history": [{"values": [3] * (stop - start)}],
            "bound_local_feedback": {"attention": [start, stop], "concern": "check recovery"},
        }

    def test_segment_requirements_use_actual_grid_and_preserve_inputs(self):
        variables = self._segment_variables()
        variables["required_points"] = 999
        original = copy.deepcopy(variables)
        teacher = messages_for("T2", variables)
        self.assertEqual(teacher[1], messages_for("T2", variables, "student")[1])
        text = teacher[1]["content"]
        self.assertIn("exactly 48 load values", text)
        self.assertIn("positions [24, 72)", text)
        self.assertIn("values[47] to 2026-10-12T23:00:00", text)
        self.assertIn("first value is for 2026-10-11T00:00:00", text)
        self.assertIn("the source does not state the unit", text)
        self.assertIn(variables["public_case"], text)
        self.assertIn(json.dumps(variables["geometry"], separators=(",", ":")), text)
        self.assertNotIn("999", text)
        self.assertNotIn("00:00:00Z", text)
        self.assertEqual(variables, original)

    def test_direct_segments_allow_full_horizon(self):
        for h in (24, 96):
            text = messages_for("T2", self._segment_variables(h, 0, h))[1]["content"]
            self.assertIn(f"complete {h}-position forecast horizon", text)
            self.assertIn(f"exactly {h} load values", text)
            self.assertNotIn("part of a", text)

    def test_refinement_keeps_complete_assignment_and_feedback(self):
        variables = self._segment_variables()
        text = messages_for("T2_refine", variables)[1]["content"]
        self.assertIn("exactly 48 values for the complete original segment", text)
        self.assertIn("including any unchanged values", text)
        self.assertIn("not a patch", text)
        for key in ("own_report_history", "bound_local_feedback"):
            self.assertIn(json.dumps(variables[key], separators=(",", ":")), text)

    def test_natural_prompts_preserve_entire_runtime_policies(self):
        variables = self._segment_variables()
        variables.update(accepted_t1={}, current_segment_reports=[], reflection_history=[],
                         reflection_policy_state={"remaining": 5, "extra": {"protected": [1, 2]}},
                         final_segment_reports=[], reflection_history_and_termination={},
                         t4_policy={"bands": [[22, 27]], "additional_limit": 7})
        for role, key in (("T3", "reflection_policy_state"), ("T4", "t4_policy")):
            self.assertIn(json.dumps(variables[key], separators=(",", ":")),
                          messages_for(role, variables)[1]["content"])

    def test_invalid_task_geometry_is_rejected_before_rendering(self):
        for start, stop in ((True, 24), (-1, 20), (25, 24), (0, 97), (0, 0)):
            with self.assertRaises(ValueError):
                messages_for("T2", self._segment_variables(96, start, stop))
        variables = self._segment_variables()
        variables["geometry"]["future_timestamps"].pop()
        with self.assertRaisesRegex(ValueError, "future_timestamps"):
            messages_for("T2", variables)

    def test_student_adapter_passes_explicit_native_mode_and_preserves_history(self):
        class Tokenizer:
            chat_template = "native template"
            eos_token_id = 2

            def apply_chat_template(self, messages, tokenize=False, **kwargs):
                self.last_kwargs = kwargs
                text = "\n".join(m["content"] for m in messages) + "\n<think>\n"
                return [ord(c) for c in text] if tokenize else text

        tokenizer = Tokenizer()
        history = [{"role": "user", "content": "Forecast."},
                   {"role": "assistant", "content": "<think>Public prior report.</think>\n<answer>{}</answer>",
                    "reasoning_content": "Provider-private sentinel"},
                   {"role": "user", "content": "Repair."}]
        output = render_reply("Final public explanation.", {"values": [1]}, "student")
        with patch("castfactory.training.chat_template_compat.final_assistant_training_tokens", return_value=([1], [2])) as helper:
            self.assertEqual(student_training_tokens(tokenizer, history, output), ([1], [2]))
            args, kwargs = helper.call_args
            self.assertEqual(kwargs["template_kwargs"], {"enable_thinking": True})
            self.assertEqual(args[1][1]["content"], history[1]["content"])
            self.assertEqual(args[1][1]["reasoning_content"], "")
            self.assertEqual(history[1]["reasoning_content"], "Provider-private sentinel")
        ids, prefix = student_prompt_tokens(tokenizer, history)
        self.assertEqual(prefix, "<think>\n")
        self.assertTrue(ids)
        self.assertTrue(tokenizer.last_kwargs["enable_thinking"])
        with self.assertRaises(ValueError):
            student_training_tokens(tokenizer, history, render_reply("Teacher prose.", {"values": [1]}))

    def test_student_training_ends_at_native_eos_and_rejects_nonwhitespace_tail(self):
        class Tokenizer:
            chat_template = "native template"
            eos_token_id = 2

            def apply_chat_template(self, messages, tokenize=False, **kwargs):
                text = "\n".join(m["content"] for m in messages) + "\n<think>\n"
                return [ord(c) for c in text] if tokenize else text

            def decode(self, ids, **kwargs):
                return "".join("<EOS>" if token == 2 else chr(token) for token in ids)

        tokenizer = Tokenizer()
        history = [{"role": "user", "content": "Forecast."}]
        output = render_reply("Public evidence.", {"values": [1]}, "student")
        helper_path = "castfactory.training.chat_template_compat.final_assistant_training_tokens"
        for original in ([65, 2], [65, 2, 10], [65, 2, 10, 32, 9]):
            with self.subTest(response=original):
                before = list(original)
                with patch(helper_path, return_value=([1], original)):
                    self.assertEqual(student_training_tokens(tokenizer, history, output), ([1], [65, 2]))
                self.assertEqual(original, before)
        for invalid in ([65], [65, 2, 66], [65, 2, 2]):
            with self.subTest(response=invalid):
                with patch(helper_path, return_value=([1], invalid)):
                    with self.assertRaisesRegex(ValueError, "EOS"):
                        student_training_tokens(tokenizer, history, output)


class LocalNativeTokenizerTests(unittest.TestCase):
    def test_actual_qwen_tokenizers_repair_history_and_continuation_parity(self):
        try:
            from transformers import AutoTokenizer
        except ImportError:
            self.skipTest("transformers not installed")
        from castfactory.training.chat_template_compat import restore_prompt_owned_think_prefix

        history = [{"role": "user", "content": "Forecast the supplied horizon."},
                   {"role": "assistant", "content": "TEACHER_PUBLIC_REPAIR. <answer>{}</answer>",
                    "reasoning_content": "PRIVATE_SENTINEL"},
                   {"role": "user", "content": "Return complete values."},
                   {"role": "assistant", "content": "<think>STUDENT_PUBLIC_REPAIR.</think>\n<answer>{}</answer>"},
                   {"role": "user", "content": "Repair the complete response."}]
        output = render_reply("Final public explanation.", {"values": [1, 2]}, "student")
        checked = 0
        for name in ("Qwen3.5-4B", "Qwen3-0.6B"):
            path = Path(__import__("os").environ.get("ELECLOAD_MODELS_DIR", "models")) / name
            if not path.exists():
                continue
            with self.subTest(tokenizer=name):
                tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True)
                prompt_ids, prefix = student_prompt_tokens(tokenizer, history)
                train_prompt, response_ids = student_training_tokens(tokenizer, history, output)
                self.assertEqual(prompt_ids, train_prompt)
                self.assertEqual(response_ids[-1], tokenizer.eos_token_id)
                self.assertEqual(response_ids.count(tokenizer.eos_token_id), 1)
                prompt_text = tokenizer.decode(prompt_ids, skip_special_tokens=False)
                self.assertIn("TEACHER_PUBLIC_REPAIR.", prompt_text)
                self.assertIn("<think>STUDENT_PUBLIC_REPAIR.</think>", prompt_text)
                self.assertNotIn("PRIVATE_SENTINEL", prompt_text)
                response = tokenizer.decode(response_ids, skip_special_tokens=False).rstrip()
                self.assertTrue(response.endswith(tokenizer.eos_token))
                response = response[:-len(tokenizer.eos_token)]
                restored = restore_prompt_owned_think_prefix(response, prefix)
                self.assertEqual(restored.count("<think>"), 1)
                self.assertEqual(parse_reply(restored, "student"), parse_reply(output, "student"))
                if name == "Qwen3.5-4B":
                    self.assertEqual(prefix, "<think>\n")
                    self.assertNotIn("<think>", response)
                else:
                    self.assertEqual(prefix, "")
                    self.assertEqual(response.count("<think>"), 1)
                checked += 1
        if not checked:
            self.skipTest("local Qwen tokenizers unavailable")


if __name__ == "__main__":
    unittest.main()
