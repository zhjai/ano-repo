"""Compact syntax binding and unchanged semantic policy, without model calls."""
from copy import deepcopy
from datetime import datetime, timedelta
import json
import unittest

from castfactory.agentic import compact_protocol as compact
from castfactory.agentic import compact_prompts as prompts
from castfactory.agentic.full_agent_v4 import Profile, _t4_policy
from castfactory.time_r1.markdown_table import parse_position_table, render_position_table, STEP_TARGET_FORMAT
from tests.agentic.test_full_agent_v4 import case, plan, STOP, KEEP


def evidence(h=24):
    geometry = case(h)["geometry"]
    origin = datetime.fromisoformat(geometry["future_timestamps"][0])
    return {"task": {"prediction_length": h, "dataset_id": "fixture", "series_id": "fixture", "freq": "1h"},
            "grid": {"future_timestamps": geometry["future_timestamps"],
                     "history_timestamps": [(origin + timedelta(hours=i)).isoformat() for i in range(-24, 0)]},
            "variables": {"load": {"history": {"unit": None}}},
            "history": {"columns": ["position", "load"], "rows": [[i, i + 24] for i in range(-24, 0)]},
            "future_known": {"columns": ["position"], "rows": []}, "context": {}, "events": []}


def reply(answer, role, h=24):
    return compact.render_role_reply({"explanation": "Public evidence is uncertain.", "answer": answer}, role, h)


class MarkdownTableTests(unittest.TestCase):
    def test_step_target_v2_is_explicit_and_rejects_old_header_or_pipe_spaces(self):
        rows = [[0, -0.0], [1, 1.2345678901234567], [4, 1e-30]]
        old = render_position_table(rows)
        new = render_position_table(rows, answer_format=STEP_TARGET_FORMAT)
        self.assertTrue(old.startswith("position|load\n"))
        self.assertTrue(new.startswith("step|target\n"))
        self.assertEqual(parse_position_table(new, [0, 1, 4], answer_format=STEP_TARGET_FORMAT), [r[1] for r in rows])
        for invalid in (old, new.replace("|", " | "), new.replace("0|-0.0", "|0|-0.0|")):
            with self.assertRaises(ValueError):
                parse_position_table(invalid, [0, 1, 4], answer_format=STEP_TARGET_FORMAT)
        with self.assertRaises(ValueError):
            parse_position_table(new, [0, 1, 4])

    def test_exact_numbers_and_disjoint_positions(self):
        rows = [[0, -0.0], [1, 1.2345678901234567], [4, 1e-30]]
        text = render_position_table(rows)
        self.assertEqual(parse_position_table(text, [0, 1, 4]), [r[1] for r in rows])
        padded = "\n" + "\n".join("| " + line.replace("|", " | ") + " |" for line in text.splitlines()) + "\n"
        self.assertEqual(parse_position_table(padded, [0, 1, 4]), [r[1] for r in rows])

    def test_strict_missing_duplicate_extra_nonfinite_and_fences(self):
        bad = ["position|load\n---|---\n0|1", "position|load\n---|---\n0|1\n0|2",
               "position|load\n---|---\n1|1\n0|2", "position|load\n---|---\n0|true\n1|2",
               "position|load\n---|---\n0|NaN\n1|2", "position|load\n---|---\n0|1e999\n1|2",
               "position|load\n---|---\n0|1\n1|2\n2|3", "position|load\n---|---\n00|1\n1|2",
               "position|load\n---|---\n0|1\n\n1|2", "```\nposition|load\n---|---\n0|1\n1|2\n```"]
        for text in bad:
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse_position_table(text, [0, 1])
        for row in ([[False, 1]], [[0, True]], [[0, float("inf")]], [[0, 10**400]]):
            with self.assertRaises(ValueError):
                render_position_table(row)


class CompactProtocolTests(unittest.TestCase):
    def request(self, role, **variables):
        return prompts.request_for(role, {"geometry": case()["geometry"], **variables}, evidence())

    def test_t2_table_preserved_then_normalized_and_assignment_bound(self):
        segment = {"segment_id": "S1", "start": 5, "stop": 8}
        answer = {**segment, "values": [1.125, 2.5, 3], "uncertainty": "Unknown", "self_check": "Checked"}
        request = self.request("T2", segment_assignment=segment)
        text = reply(answer, "T2")
        parsed = compact.parse_role_answer(text, "T2", request)
        self.assertNotIn("values", parsed["answer"])
        self.assertEqual(parsed["answer"]["rows"], [[5, 1.125], [6, 2.5], [7, 3]])
        self.assertEqual(compact.normalize_role_reply(parsed, "T2", 24)["answer"], answer)
        self.assertEqual(compact.render_role_reply(parsed, "T2", 24), text)
        for bad in (text.replace("5|1.125", "6|1.125"), text + "extra", text.replace('"start":5', '"start":4'),
                    text.replace('"stop":8', '"stop":8,"values":[]'), text.replace('"start":5', '"start":5,"start":5')):
            with self.assertRaises(ValueError):
                compact.parse_role_answer(bad, "T2", request)
        other = deepcopy(request); other["segment_assignment"]["segment_id"] = "S0"
        with self.assertRaises(ValueError):
            compact.parse_role_answer(text, "T2", other)

    def test_v2_numeric_contract_global_steps_and_json_controls(self):
        version = compact.STEP_TARGET_VERSION
        segment = {"segment_id": "S1", "start": 5, "stop": 8}
        answer = {**segment, "values": [1.125, 2.5, 3], "uncertainty": "Unknown", "self_check": "Checked"}
        request = self.request("T2", segment_assignment=segment)
        text = compact.render_role_reply({"explanation": "Public evidence is uncertain.", "answer": answer},
                                         "T2", 24, protocol_version=version)
        self.assertIn("step|target\n---|---\n5|1.125\n6|2.5\n7|3", text)
        parsed = compact.parse_role_answer(text, "T2", request, protocol_version=version)
        self.assertEqual(compact.normalize_role_reply(parsed, "T2", 24)["answer"], answer)
        self.assertEqual(compact.render_role_reply(parsed, "T2", 24, protocol_version=version), text)
        with self.assertRaises(ValueError):
            compact.parse_role_answer(text, "T2", request)
        with self.assertRaises(ValueError):
            compact.parse_role_answer(reply(answer, "T2"), "T2", request, protocol_version=version)
        self.assertEqual(compact.render_answer(STOP, "T3", 24),
                         compact.render_answer(STOP, "T3", 24, protocol_version=version))
        t4_request = self.request("T4", t4_policy=_t4_policy(plan(cuts=[8, 16])["spans"], 24, Profile()))
        t4 = {"action": "REGENERATE", "edits": [
            {"boundary_id": "B0", "start": 7, "stop": 8, "values": [71]},
            {"boundary_id": "B1", "start": 16, "stop": 18, "values": [161, 162]}]}
        rendered = compact.render_role_reply({"explanation": "Reconcile boundaries.", "answer": t4},
                                             "T4", 24, protocol_version=version)
        self.assertEqual(rendered.count("step|target"), 1)
        parsed = compact.parse_role_answer(rendered, "T4", t4_request, protocol_version=version)
        self.assertEqual(compact.normalize_role_reply(parsed, "T4", 24)["answer"], t4)

    def test_v2_role_state_reference_tables_match_output_header(self):
        from unittest.mock import patch
        segment = plan()["spans"][0]
        answer = {**segment, "values": [876.543] * 24, "uncertainty": "Unknown", "self_check": "Checked"}
        report = {"explanation": "Own public summary", "answer": answer}
        variables = {"geometry": case()["geometry"], "segment_assignment": segment,
                     "own_report_history": [report, deepcopy(report)], "mode": "REFINE"}
        with patch("scripts.load_canonical.compact_view.render_compact_case", return_value="Explicit time input") as render:
            messages = prompts.messages_for("T2_refine", variables, evidence(), protocol_version=compact.STEP_TARGET_VERSION)
        self.assertIn("step|target", messages[0]["content"])
        self.assertIn("no spaces around pipes", messages[0]["content"])
        self.assertEqual(messages[1]["content"].count("step|target"), 1)
        self.assertNotIn("position|load", messages[1]["content"])
        self.assertEqual(messages[1]["content"].splitlines().count("0|876.543"), 1)
        self.assertEqual(render.call_args.kwargs["input_version"], "compact-load-input-v2-explicit-time-column")
        request = prompts.request_for("T2_refine", variables, evidence())
        row = {"role": "T2", "profile_version": compact.STEP_TARGET_VERSION,
               "answer_format": STEP_TARGET_FORMAT, "messages": messages, "semantic_request": request,
               "assistant_response": compact.render_role_reply(report, "T2", 24,
                                                                 protocol_version=compact.STEP_TARGET_VERSION)}
        with patch("scripts.load_canonical.compact_view.render_compact_case", return_value="Explicit time input"):
            compact.validate_sft_row(row)
            row["metadata"] = {"protocol_version": compact.VERSION}
            with self.assertRaisesRegex(ValueError, "disagree"):
                compact.validate_sft_row(row)

    def test_t1_t3_original_json_semantics_and_wrappers(self):
        p = plan()
        t1 = self.request("T1", evidence_manifest=case()["evidence_manifest"], compute_policy=Profile().record())
        self.assertEqual(compact.parse_role_answer(reply(p, "T1"), "T1", t1)["answer"], p)
        t3 = self.request("T3", accepted_t1={"explanation": "Plan", "answer": p},
                          reflection_policy_state={"eligible_segments": []})
        self.assertEqual(compact.parse_role_answer(reply(STOP, "T3"), "T3", t3)["answer"], STOP)
        text = reply(STOP, "T3")
        for bad in (text.replace("<think>", ""), text + "<answer>{}</answer>", text.replace('"STOP"', '"STOP","values":[]')):
            with self.assertRaises(ValueError):
                compact.parse_role_answer(bad, "T3", t3)

    def test_t4_union_binds_every_edit_and_full_policy(self):
        spans = plan(cuts=[8, 16])["spans"]
        request = self.request("T4", t4_policy=_t4_policy(spans, 24, Profile()))
        answer = {"action": "REGENERATE", "edits": [
            {"boundary_id": "B0", "start": 7, "stop": 8, "values": [71]},
            {"boundary_id": "B1", "start": 16, "stop": 18, "values": [161, 162]}]}
        text = reply(answer, "T4")
        self.assertEqual(text.count("position|load"), 1)
        parsed = compact.parse_role_answer(text, "T4", request)
        self.assertEqual(parsed["answer"]["edits"][1]["rows"], [[16, 161], [17, 162]])
        self.assertEqual(compact.normalize_role_reply(parsed, "T4", 24)["answer"], answer)
        for bad in (text.replace("17|162", "15|162"), text.replace('"B1"', '"B0"'),
                    text.replace('"start":16', '"start":0'), text.replace("7|71", "7|71\n8|72")):
            with self.assertRaises(ValueError):
                compact.parse_role_answer(bad, "T4", request)
        keep = reply(KEEP, "T4")
        self.assertEqual(compact.parse_role_answer(keep, "T4", request)["answer"], KEEP)
        with self.assertRaises(ValueError):
            compact.parse_role_answer(keep.replace("</answer>", "\nposition|load\n---|---</answer>"), "T4", request)

    def test_shared_input_once_report_tables_deduplicated_and_sft_binding(self):
        segment = plan()["spans"][0]
        answer = {**segment, "values": [876.543] * 24, "uncertainty": "Unknown", "self_check": "Checked"}
        report = {"explanation": "Own public summary", "answer": answer}
        variables = {"geometry": case()["geometry"], "public_case": "OLD_CASE_SENTINEL", "segment_assignment": segment,
                     "own_report_history": [report, deepcopy(report)], "mode": "REFINE"}
        request = prompts.request_for("T2_refine", variables, evidence())
        messages = prompts.messages_for("T2_refine", variables, evidence())
        content = messages[1]["content"]
        self.assertEqual(content.count("Forecast task:"), 1)
        self.assertEqual(content.splitlines().count("0|876.543"), 1)
        self.assertNotIn('"values"', content)
        self.assertNotIn('"future_timestamps"', content)
        self.assertNotIn("OLD_CASE_SENTINEL", content)
        self.assertEqual(request["geometry"]["future_timestamps"], variables["geometry"]["future_timestamps"])
        row = {"role": "T2", "messages": messages, "semantic_request": request, "assistant_response": reply(answer, "T2")}
        compact.validate_sft_row(row)
        row["messages"].extend([{"role": "assistant", "content": "Actual malformed attempt"},
                                prompts.repair_message("T2_refine", "Return the complete segment.")])
        compact.validate_sft_row(row)
        row["messages"][1]["content"] += "\nPRIVATE_LABEL_SENTINEL"
        with self.assertRaisesRegex(ValueError, "do not match"):
            compact.validate_sft_row(row)

    def test_private_fields_opaque_reports_and_case_override_are_rejected(self):
        base = {"geometry": case()["geometry"], "segment_assignment": plan()["spans"][0]}
        for field in ("canonical_labels", "future_target", "reward", "case", "role"):
            with self.assertRaises(ValueError):
                prompts.messages_for("T2", {**base, field: [999]}, evidence())
        with self.assertRaisesRegex(ValueError, "Opaque numeric"):
            prompts.messages_for("T2", {**base, "opaque_report": '{"values":[999]}'}, evidence())


if __name__ == "__main__":
    unittest.main()
