"""Scripted teacher calls exercise the new protocol without API traffic."""
from copy import deepcopy
from datetime import datetime, timedelta
import json
import unittest

from castfactory.agentic import distillation_protocol as protocol
from castfactory.agentic.full_agent_v4 import (
    AcceptanceJournalFailure, Profile, PUBLIC_POLICY_VERSION, compute_policy,
    direct_cost_reference, execute, execute_public, public_compute_policy,
    sft_rows, validate_public_role,
)


def case(h=24):
    origin = datetime(2025, 1, 2)
    return {"case_id": "public-" + str(h), "source": "fixture", "split": "train",
            "public_case": "Hourly history H0 supports a stable cycle. Public evidence is retrospective.",
            "geometry": {"H": h, "future_timestamps": [(origin + timedelta(hours=i)).isoformat() for i in range(h)],
                         "frequency": "1h", "unit": None},
            "evidence_manifest": [{"evidence_id": "H0", "kind": "history", "as_of": None,
                                   "availability_basis": "retrospective", "location": "target_history"}],
            "history_values": list(range(24))}


def plan(h=24, cuts=None, drivers=False):
    bounds = [0] + (cuts or []) + [h]
    spans = [{"segment_id": "S" + str(i), "start": a, "stop": b}
             for i, (a, b) in enumerate(zip(bounds, bounds[1:]))]
    return {"drivers": [{"id": "D0", "name": "history", "evidence": [
                {"evidence_id": "H0", "kind": "history", "as_of": None,
                 "availability_basis": "retrospective", "quote_or_feature": "hourly history"}],
              "mechanism": "Cycle is a hypothesis.", "relevance": "high",
              "impact_windows": [{"start": 0, "stop": h, "effect": "uncertain"}],
              "interactions": [], "uncertainty": "Unknown future changes."}] if drivers else [],
            "global_uncertainty": "No future outcomes are observed.",
            "evidence_ids_used": ["H0"] if drivers else [],
            "mode": "FACTORIZE" if cuts else "DIRECT", "spans": spans}


def t2(segment, value=100, explanation="Local independent forecast."):
    return protocol.render_reply(explanation, {**segment, "values": [value] * (segment["stop"] - segment["start"]),
                                              "uncertainty": "Uncertain", "self_check": "Checked local evidence."})


STOP = {"action": "STOP", "segment_id": None, "directive": None}
KEEP = {"action": "KEEP", "edits": []}


def refine(sid="S0", field="summary", refs=None, concern="shape", span=None):
    return {"action": "REFINE", "segment_id": sid,
            "directive": {"concern": concern, "review_span": span, "driver_refs": refs or [], "own_report_field": field}}


def review_keep(raw, messages):
    return {"content": protocol.render_reply("Full merged T1 reviewed.", {"decision": "KEEP"}),
            "usage": {"output_tokens": 20}}


class Script:
    def __init__(self, turns):
        self.turns = list(turns)
        self.calls = []

    def __call__(self, role, messages):
        self.calls.append((role, deepcopy(messages)))
        if not self.turns:
            raise AssertionError("Unexpected call " + role)
        expected, result = self.turns.pop(0)
        if expected != role:
            raise AssertionError("Expected " + expected + " got " + role)
        if isinstance(result, Exception):
            raise result
        return deepcopy(result)


class FourRoleTeacherTests(unittest.TestCase):
    def test_direct_h24_and_h96_use_actual_geometry_and_model_stop(self):
        for h in (24, 96):
            p = plan(h)
            caller = Script([("T1", protocol.render_reply("Whole horizon.", p)),
                             ("T2", t2(p["spans"][0], 123.456789)),
                             ("T3", protocol.render_reply("No supported refinement.", STOP))])
            episode = execute(case(h), caller, reviewer=review_keep)
            self.assertTrue(episode["normal_completion"])
            self.assertEqual(episode["final_values"], [123.456789] * h)
            self.assertEqual(episode["t4_status"], "not_applicable_DIRECT")
            self.assertEqual([r for r, _ in caller.calls], ["T1", "T2", "T3"])
            self.assertEqual(episode["termination"]["kind"], "model_STOP")
            self.assertEqual(episode["context_validation"], "unmeasured")
            json.dumps(episode, allow_nan=False)

    def test_initial_segments_are_independent_and_t4_keep_is_mandatory(self):
        p = plan(cuts=[12])
        caller = Script([("T1", protocol.render_reply("Divide public regimes.", p)),
                         ("T2", t2(p["spans"][0], 771.234, "SIBLING_EXPLANATION_SENTINEL")),
                         ("T2", t2(p["spans"][1], 200)),
                         ("T3", protocol.render_reply("Stop now.", STOP)),
                         ("T4", protocol.render_reply("Public evidence supports keeping the join.", KEEP))])
        episode = execute(case(), caller, reviewer=review_keep)
        second_initial = json.dumps(caller.calls[2][1])
        self.assertNotIn("771.234", second_initial)
        self.assertNotIn("SIBLING_EXPLANATION_SENTINEL", second_initial)
        self.assertTrue(episode["normal_completion"])
        self.assertEqual(episode["t4_status"], "model_KEEP")
        self.assertEqual(episode["final_values"], [771.234] * 12 + [200] * 12)

    def test_final_refinement_must_return_whole_segment_then_final_t3(self):
        p = plan(cuts=[12], drivers=True)
        directive = refine(refs=["D0"], concern="mechanism_consistency", span=[5, 7])
        caller = Script([("T1", protocol.render_reply("Public shared interpretation.", p)),
                         ("T2", t2(p["spans"][0], 100, "OWN_EXPLANATION")),
                         ("T2", t2(p["spans"][1], 876.543, "SIBLING_SENTINEL")),
                         ("T3", protocol.render_reply("GLOBAL_T3_SENTINEL sibling 876.543", directive)),
                         ("T2", t2(p["spans"][0], 100, "Concern declined; complete values unchanged.")),
                         ("T3", protocol.render_reply("No refinement allowance remains.", STOP)),
                         ("T4", protocol.render_reply("Keep supported levels.", KEEP))])
        episode = execute(case(), caller, profile=Profile(max_refinements=1), reviewer=review_keep)
        self.assertTrue(episode["normal_completion"])
        self.assertEqual([r for r, _ in caller.calls], ["T1", "T2", "T2", "T3", "T2", "T3", "T4"])
        revised_request = json.dumps(caller.calls[4][1])
        self.assertNotIn("GLOBAL_T3_SENTINEL", revised_request)
        self.assertNotIn("SIBLING_SENTINEL", revised_request)
        self.assertNotIn("876.543", revised_request)
        self.assertIn("OWN_EXPLANATION", revised_request)
        self.assertIn("H0", revised_request)
        self.assertIn("This call revises your own original segment", revised_request)
        last_t3 = episode["calls"][5]["public_variables"]["reflection_policy_state"]
        self.assertEqual(last_t3["remaining_refinements"], 0)
        self.assertEqual(last_t3["eligible_segments"], [])

    def test_review_rewrites_whole_plan_before_any_downstream_call(self):
        original, accepted = plan(), plan(cuts=[12])
        p1_raw = protocol.render_reply("Original direct plan.", original)
        review_raw = protocol.render_reply("Complete replacement is publicly justified.", {
            "decision": "REWRITE", "explanation": "Accepted factorized plan.", "t1": accepted})
        seen = []

        def reviewer(raw, messages):
            seen.append((raw, deepcopy(messages)))
            messages.clear()  # Owned snapshot cannot mutate P1 receipts.
            return {"content": review_raw, "usage": {"output_tokens": 30}}

        caller = Script([("T1", {"content": p1_raw, "usage": {"output_tokens": 10}}),
                         ("T2", t2(accepted["spans"][0])), ("T2", t2(accepted["spans"][1])),
                         ("T3", protocol.render_reply("Stop.", STOP)),
                         ("T4", protocol.render_reply("Keep.", KEEP))])
        episode = execute(case(), caller, reviewer=reviewer)
        self.assertEqual(seen[0][0], p1_raw)
        self.assertEqual(episode["calls"][0]["raw_response"]["content"], p1_raw)
        self.assertEqual(episode["t1_review"]["receipt"]["content"], review_raw)
        self.assertEqual(episode["t1_pass1"]["reply"]["answer"]["mode"], "DIRECT")
        self.assertEqual(episode["accepted_t1"]["answer"]["mode"], "FACTORIZE")
        self.assertEqual(len(episode["calls"][0]["messages"]), 2)
        rows = sft_rows(episode)
        self.assertEqual(rows[0]["t1_supervision"]["kind"], "label_informed")
        self.assertEqual(rows[0]["t1_supervision"]["decision"], "REWRITE")
        self.assertTrue(rows[0]["derived"])
        parsed = protocol.parse_reply(rows[0]["assistant_response"], "student")
        self.assertEqual(parsed["answer"], accepted)
        self.assertEqual(rows[0]["teacher_messages"], episode["calls"][0]["messages"])
        self.assertIn("<think>", rows[0]["messages"][0]["content"])

    def test_atomic_multiseam_and_one_sided_t4_edits(self):
        p = plan(cuts=[8, 16])
        edits = {"action": "REGENERATE", "edits": [
            {"boundary_id": "B0", "start": 6, "stop": 8, "values": [61, 62]},
            {"boundary_id": "B1", "start": 16, "stop": 18, "values": [161, 162]}]}
        caller = Script([("T1", protocol.render_reply("Three regimes.", p)),
                         *[("T2", t2(s, (i + 1) * 100)) for i, s in enumerate(p["spans"])],
                         ("T3", protocol.render_reply("No useful refinement.", STOP)),
                         ("T4", protocol.render_reply("Bounded evidence-based local changes.", edits))])
        episode = execute(case(), caller, reviewer=review_keep)
        self.assertTrue(episode["normal_completion"])
        self.assertEqual(episode["final_values"], [100] * 6 + [61, 62] + [200] * 8 + [161, 162] + [300] * 6)
        variables = episode["calls"][-1]["public_variables"]
        self.assertNotIn("concat", variables)
        self.assertNotIn("values", variables["reflection_history_and_termination"]["history"][0])
        values = [v for r in variables["final_segment_reports"] for v in r["answer"]["values"]]
        self.assertEqual(len(values), 24)

    def test_invalid_t4_second_edit_never_partially_applies_first(self):
        p = plan(cuts=[8, 16])
        bad = {"action": "REGENERATE", "edits": [
            {"boundary_id": "B0", "start": 7, "stop": 8, "values": [999]},
            {"boundary_id": "B1", "start": 19, "stop": 20, "values": [888]}]}
        caller = Script([("T1", protocol.render_reply("Split.", p)),
                         *[("T2", t2(s, i + 1)) for i, s in enumerate(p["spans"])],
                         ("T3", protocol.render_reply("Stop.", STOP)),
                         ("T4", protocol.render_reply("Invalid second interval.", bad))])
        episode = execute(case(), caller, profile=Profile(structural_retries=0), reviewer=review_keep)
        self.assertFalse(episode["normal_completion"])
        self.assertEqual(episode["t4_status"], "failed")
        self.assertEqual(episode["final_values"], [1] * 8 + [2] * 8 + [3] * 8)
        self.assertEqual(episode["termination"]["kind"], "runtime_failure")
        self.assertEqual(episode["fallback"]["kind"], "last_complete_segment_snapshot")
        self.assertFalse(episode["fallback"]["model_t4_completed"])
        self.assertEqual([r["role"] for r in sft_rows(episode)], ["T1", "T2", "T2", "T2", "T3"])

    def test_retries_preserve_actual_histories_and_independent_t4_allowance(self):
        p = plan(cuts=[12])
        bad_t2 = protocol.render_reply("Incomplete numerical return.", {**p["spans"][0], "values": [1],
                                                                         "uncertainty": "Unknown", "self_check": "Check"})
        bad_t4 = protocol.render_reply("Malformed keep.", {"action": "KEEP", "edits": [{}]})
        caller = Script([("T1", protocol.render_reply("Two regimes.", p)),
                         ("T2", bad_t2), ("T2", t2(p["spans"][0])), ("T2", t2(p["spans"][1])),
                         ("T3", protocol.render_reply("Stop.", STOP)),
                         ("T4", bad_t4), ("T4", protocol.render_reply("Keep after structural repair.", KEEP))])
        episode = execute(case(), caller, reviewer=review_keep)
        self.assertTrue(episode["normal_completion"])
        for i, failed in ((2, bad_t2), (6, bad_t4)):
            self.assertEqual(episode["calls"][i]["messages"][2], {"role": "assistant", "content": failed})
            self.assertEqual(episode["calls"][i]["attempt"], 1)
        self.assertIn("returned 1 values; original assignment requires 12", episode["calls"][2]["messages"][3]["content"])
        rows = sft_rows(episode)
        self.assertEqual(len(rows), 5)
        successful_t2 = next(r for r in rows if r["role"] == "T2")
        self.assertEqual(successful_t2["messages"][2], {"role": "assistant", "content": bad_t2})
        self.assertEqual(successful_t2["messages"][2:], successful_t2["teacher_messages"][2:])
        self.assertEqual(protocol.parse_reply(successful_t2["assistant_response"], "student")["answer"]["values"], [100] * 12)

    def test_failure_is_not_fabricated_stop_and_fallback_replays_h96_history(self):
        p = plan(96)
        caller = Script([("T1", protocol.render_reply("Whole horizon.", p)), ("T2", RuntimeError("transport failure"))])
        episode = execute(case(96), caller, profile=Profile(structural_retries=0), reviewer=review_keep)
        self.assertFalse(episode["normal_completion"])
        self.assertEqual(episode["reflection_history"], [])
        self.assertEqual(episode["termination"]["kind"], "runtime_failure")
        self.assertEqual(episode["final_values"], list(range(24)) * 4)
        self.assertEqual(episode["fallback"]["kind"], "public_history_cycle")
        self.assertTrue(episode["complete_forecast"])
        self.assertEqual([r["role"] for r in sft_rows(episode)], ["T1"])

    def test_refinement_failure_retains_complete_forecast_and_truthful_status(self):
        p = plan()
        caller = Script([("T1", protocol.render_reply("Whole horizon.", p)), ("T2", t2(p["spans"][0], 42)),
                         ("T3", protocol.render_reply("Review shape.", refine())), ("T2", RuntimeError("revision failed"))])
        episode = execute(case(), caller, profile=Profile(structural_retries=0), reviewer=review_keep)
        self.assertEqual(episode["final_values"], [42] * 24)
        self.assertEqual(episode["reflection_history"][0]["revision_status"], "failed")
        self.assertFalse(episode["normal_completion"])
        self.assertEqual(episode["termination"]["kind"], "runtime_failure")

    def test_public_boundary_train_only_review_required_and_geometry_checked(self):
        for mutate in (lambda c: c.update(split="validation"), lambda c: c.update(labels=[9] * 24),
                       lambda c: c["geometry"].update(H=96),
                       lambda c: c["geometry"]["future_timestamps"].__setitem__(1, "2025-01-02T03:00:00")):
            c = case()
            mutate(c)
            caller = Script([])
            with self.assertRaises(ValueError):
                execute(c, caller, reviewer=review_keep)
            self.assertEqual(caller.calls, [])
        with self.assertRaises(ValueError):
            execute(case(), Script([]))

    def test_invalid_review_and_invented_availability_never_reach_t2(self):
        p = plan(drivers=True)
        caller = Script([("T1", protocol.render_reply("Interpretation.", p))])
        episode = execute(case(), caller, reviewer=lambda raw, msgs: protocol.render_reply("Incomplete rewrite.", {
            "decision": "REWRITE", "explanation": "No plan provided."}))
        self.assertFalse(episode["normal_completion"])
        self.assertIsNone(episode["accepted_t1"])
        self.assertEqual(sft_rows(episode), [])
        bad = deepcopy(p)
        bad["drivers"][0]["evidence"][0]["availability_basis"] = "operational"
        caller = Script([("T1", protocol.render_reply("Invented availability.", bad))])
        episode = execute(case(), caller, profile=Profile(structural_retries=0), reviewer=review_keep)
        self.assertEqual([r for r, _ in caller.calls], ["T1"])
        self.assertIsNone(episode["t1_review"])

    def test_exact_context_rejection_prevents_external_call_without_false_completion(self):
        with self.assertRaises(ValueError):
            Profile(max_context_tokens=5000)
        caller = Script([])
        profile = Profile(max_context_tokens=5000, message_token_counter=lambda messages: 4999)
        episode = execute(case(), caller, profile=profile, reviewer=review_keep)
        self.assertEqual(caller.calls, [])
        self.assertEqual(episode["calls"][0]["status"], "context_rejected")
        self.assertFalse(episode["normal_completion"])
        self.assertEqual(episode["context_validation"], "per_actual_request")

    def test_invalid_refine_on_final_assessment_repairs_to_real_stop_before_t4(self):
        p = plan(cuts=[12])
        illegal = protocol.render_reply("Attempting extra compute.", refine())
        caller = Script([("T1", protocol.render_reply("Split.", p)),
                         ("T2", t2(p["spans"][0])), ("T2", t2(p["spans"][1])),
                         ("T3", illegal), ("T3", protocol.render_reply("No allowance remains.", STOP)),
                         ("T4", protocol.render_reply("Keep.", KEEP))])
        episode = execute(case(), caller, profile=Profile(max_refinements=0), reviewer=review_keep)
        self.assertTrue(episode["normal_completion"])
        self.assertEqual(episode["termination"]["kind"], "model_STOP")
        self.assertEqual(episode["reflection_history"][0]["t3_reply"]["answer"], STOP)
        self.assertEqual(episode["calls"][3]["status"], "failed")
        self.assertEqual(episode["calls"][4]["messages"][2]["content"], illegal)

    def test_failed_final_t3_keeps_revision_but_never_claims_t4_or_stop(self):
        p = plan(cuts=[12])
        caller = Script([("T1", protocol.render_reply("Split.", p)),
                         ("T2", t2(p["spans"][0], 11)), ("T2", t2(p["spans"][1], 22)),
                         ("T3", protocol.render_reply("Review shape.", refine(span=[5, 7]))),
                         ("T2", t2(p["spans"][0], 33)), ("T3", RuntimeError("final assessment failed"))])
        episode = execute(case(), caller, profile=Profile(max_refinements=1, structural_retries=0), reviewer=review_keep)
        self.assertFalse(episode["normal_completion"])
        self.assertEqual(episode["final_values"], [33] * 12 + [22] * 12)
        self.assertEqual(episode["t4_status"], "not_run_failure")
        self.assertEqual(episode["termination"]["kind"], "runtime_failure")
        self.assertEqual(episode["reflection_history"][0]["revision_status"], "accepted")
        self.assertEqual(episode["reflection_history"][0]["t3_reply"]["answer"]["action"], "REFINE")

    def test_typed_feedback_rejects_nonlocal_driver_and_nonoverlapping_attention(self):
        p = plan(cuts=[12], drivers=True)
        p["drivers"][0]["impact_windows"] = [{"start": 0, "stop": 6, "effect": "uncertain"}]
        for action in (refine("S1", refs=["D0"]), refine("S0", refs=["D0"], span=[8, 10]),
                       refine("S0", refs=["D0"], concern="event_interaction"),
                       refine("S0", field="uncertainty", concern="mechanism_consistency")):
            caller = Script([("T1", protocol.render_reply("Limited early driver.", p)),
                             ("T2", t2(p["spans"][0])), ("T2", t2(p["spans"][1])),
                             ("T3", protocol.render_reply("Invalid typed reference.", action))])
            episode = execute(case(), caller, profile=Profile(structural_retries=0), reviewer=review_keep)
            self.assertFalse(episode["normal_completion"])
            self.assertEqual(len(caller.calls), 4)
            self.assertEqual(episode["reflection_history"], [])

    def test_t4_rejects_overlap_duplicate_boundary_and_aggregate_overwrite(self):
        p = plan(cuts=[8, 16])
        invalid_edits = [
            [{"boundary_id": "B0", "start": 7, "stop": 17, "values": [7] * 10},
             {"boundary_id": "B1", "start": 16, "stop": 17, "values": [8]}],
            [{"boundary_id": "B0", "start": 7, "stop": 8, "values": [7]},
             {"boundary_id": "B0", "start": 8, "stop": 9, "values": [8]}],
            [{"boundary_id": "B0", "start": 0, "stop": 12, "values": [7] * 12},
             {"boundary_id": "B1", "start": 12, "stop": 24, "values": [8] * 12}],
        ]
        for edits in invalid_edits:
            caller = Script([("T1", protocol.render_reply("Split.", p)),
                             *[("T2", t2(s, 100)) for s in p["spans"]],
                             ("T3", protocol.render_reply("Stop.", STOP)),
                             ("T4", protocol.render_reply("Attempt invalid edits.", {"action": "REGENERATE", "edits": edits}))])
            episode = execute(case(), caller, reviewer=review_keep,
                              profile=Profile(structural_retries=0, t4_max_patch_steps=24, t4_max_total_steps=24))
            self.assertFalse(episode["normal_completion"])
            self.assertEqual(episode["final_values"], [100] * 24)
            self.assertEqual(episode["t4_status"], "failed")

    def test_explicit_incomplete_receipt_cannot_be_accepted_despite_valid_json(self):
        p = plan()
        for reason in ("length", "content_filter", None):
            raw_t2 = {"content": t2(p["spans"][0]), "finish_reason": reason, "usage": {"completion_tokens": 1}}
            caller = Script([("T1", protocol.render_reply("Whole horizon.", p)), ("T2", raw_t2)])
            episode = execute(case(), caller, reviewer=review_keep, profile=Profile(structural_retries=0))
            self.assertFalse(episode["normal_completion"])
            self.assertEqual(episode["calls"][1]["raw_response"], raw_t2)
            self.assertEqual(episode["calls"][1]["content"], raw_t2["content"])
            self.assertEqual(episode["calls"][1]["status"], "failed")
            self.assertEqual([r["role"] for r in sft_rows(episode)], ["T1"])
        caller = Script([("T1", protocol.render_reply("Whole horizon.", p))])
        receipt = {"content": protocol.render_reply("Keep.", {"decision": "KEEP"}), "finish_reason": "length"}
        episode = execute(case(), caller, reviewer=lambda raw, messages: receipt)
        self.assertFalse(episode["normal_completion"])
        self.assertEqual(episode["t1_review"]["receipt"], receipt)
        self.assertEqual(episode["t1_review"]["status"], "failed")
        self.assertIsNone(episode["accepted_t1"])

    def test_transport_failure_never_becomes_model_structural_feedback(self):
        p = plan()
        caller = Script([("T1", protocol.render_reply("Whole horizon.", p)),
                         ("T2", RuntimeError("Credential-bearing failure sentinel")),
                         ("T2", t2(p["spans"][0]))])
        episode = execute(case(), caller, reviewer=review_keep, profile=Profile(structural_retries=2))
        self.assertFalse(episode["normal_completion"])
        self.assertEqual([role for role, _ in caller.calls], ["T1", "T2"])
        self.assertEqual(len(episode["calls"]), 2)
        failure = episode["calls"][-1]
        self.assertEqual(failure["status"], "transport_failed")
        self.assertEqual(failure["attempt"], 0)
        self.assertIsNone(failure["content"])
        self.assertEqual(len(failure["messages"]), 2)
        self.assertNotIn("Credential-bearing", json.dumps(episode))
        self.assertNotIn("Structural validation failed", json.dumps(episode))
        self.assertEqual([row["role"] for row in sft_rows(episode)], ["T1"])

    def test_accepted_callback_marks_only_valid_attempts_and_owns_its_snapshot(self):
        p = plan()
        markers = []

        def record(entry):
            markers.append(deepcopy(entry))
            entry["messages"].clear()
            entry["parsed"]["answer"].clear()

        bad = t2(p["spans"][0]).replace('"values":[100,', '"values":[true,', 1)
        caller = Script([("T1", protocol.render_reply("Whole horizon.", p)), ("T2", bad),
                         ("T2", t2(p["spans"][0])), ("T3", protocol.render_reply("Stop.", STOP))])
        episode = execute(case(), caller, reviewer=review_keep, on_accept=record)
        self.assertTrue(episode["normal_completion"])
        self.assertEqual([m["role"] for m in markers], ["T1", "T2", "T3"])
        self.assertEqual([m["call_id"] for m in markers], [0, 2, 3])
        self.assertEqual(markers[1]["messages"][2]["content"], bad)
        self.assertTrue(all(m["status"] == "accepted" for m in markers))
        self.assertEqual(episode["final_values"], [100] * 24)
        self.assertEqual(episode["calls"][0]["messages"], markers[0]["messages"])

    def test_acceptance_journal_error_propagates_without_model_retry_or_fallback(self):
        p = plan()

        def broken_journal(entry):
            raise OSError("Private storage sentinel")

        caller = Script([("T1", protocol.render_reply("Whole horizon.", p)),
                         ("T1", protocol.render_reply("Must not be retried.", p))])
        with self.assertRaisesRegex(AcceptanceJournalFailure, "Acceptance journal failed: OSError") as caught:
            execute(case(), caller, reviewer=review_keep, on_accept=broken_journal, profile=Profile(structural_retries=2))
        self.assertNotIn("Private storage sentinel", str(caught.exception))
        self.assertEqual([role for role, _ in caller.calls], ["T1"])

    def test_privileged_review_journal_error_propagates_before_downstream(self):
        p = plan()
        caller = Script([("T1", protocol.render_reply("Whole horizon.", p))])

        def reviewer(raw, messages):
            raise AcceptanceJournalFailure("Privileged accepted receipt could not be recorded")

        with self.assertRaises(AcceptanceJournalFailure):
            execute(case(), caller, reviewer=reviewer)
        self.assertEqual([role for role, _ in caller.calls], ["T1"])

    def test_counter_failure_never_creates_model_repair_history(self):
        for counter in (lambda messages: -1, lambda messages: True,
                        lambda messages: (_ for _ in ()).throw(OSError("Counter failure sentinel"))):
            caller = Script([])
            episode = execute(case(), caller, reviewer=review_keep,
                              profile=Profile(message_token_counter=counter, structural_retries=2))
            self.assertFalse(episode["normal_completion"])
            self.assertEqual(caller.calls, [])
            self.assertEqual(len(episode["calls"]), 1)
            self.assertEqual(len(episode["calls"][0]["messages"]), 2)
            self.assertEqual(episode["calls"][0]["status"], "context_rejected")
            self.assertNotIn("Structural validation failed", json.dumps(episode))
            self.assertNotIn("Counter failure sentinel", json.dumps(episode))


class FourRolePublicTests(unittest.TestCase):
    @staticmethod
    def reply(explanation, answer):
        return protocol.render_reply(explanation, answer, "student")

    def report(self, segment, value, explanation="Independent forecast."):
        return self.reply(explanation, {**segment, "values": [value] * (segment["stop"] - segment["start"]),
                                        "uncertainty": "Uncertain", "self_check": "Checked public evidence."})


    def test_stored_teacher_t1_is_deterministically_adapted_without_rewriting_receipts(self):
        p = plan()
        teacher_profile = Profile(max_response_tokens=16384)
        teacher_caller = Script([("T1", protocol.render_reply("Public plan.", p)),
                                 ("T2", t2(p["spans"][0])), ("T3", protocol.render_reply("Stop.", STOP))])
        teacher = execute(case(), teacher_caller, profile=teacher_profile, reviewer=review_keep)
        original = deepcopy(teacher)
        teacher_policy = teacher["calls"][0]["public_variables"]["compute_policy"]
        self.assertEqual(teacher_policy, compute_policy(case(), teacher_profile))
        self.assertIn('"max_response_tokens":16384', teacher["calls"][0]["messages"][1]["content"])
        rows = sft_rows(teacher)
        public_caller = Script([("T1", self.reply("Public plan.", p)), ("T2", self.report(p["spans"][0], 100)),
                                ("T3", self.reply("Stop.", STOP))])
        production = execute_public(case(), public_caller, profile=Profile(
            max_response_tokens=4096, max_context_tokens=32768, message_token_counter=lambda messages: 100))
        self.assertEqual(rows[0]["messages"], production["calls"][0]["messages"])
        self.assertEqual(rows[0]["teacher_messages"], original["calls"][0]["messages"])
        self.assertEqual(teacher, original)
        self.assertEqual(rows[0]["serialization_provenance"]["public_policy_adapter"], {
            "kind": "deterministic_public_policy_projection", "version": PUBLIC_POLICY_VERSION,
            "omitted_fields": ["max_context_tokens", "max_response_tokens"]})
        self.assertEqual(rows[0]["t1_supervision"], teacher["t1_supervision"])
        self.assertEqual(protocol.parse_reply(rows[0]["assistant_response"], "student"), teacher["accepted_t1"])

    def test_public_direct_h24_h96_has_no_review_and_preserves_source_geometry(self):
        for h, split in ((24, "validation"), (96, "ood_test")):
            c = case(h)
            c["split"] = split
            c["geometry"]["unit"] = "source-native MW"
            c["geometry"]["clock_basis"] = "source clock, unspecified zone"
            before = deepcopy(c)
            p = plan(h)
            caller = Script([("T1", self.reply("Public interpretation.", p)),
                             ("T2", self.report(p["spans"][0], 123.456789)),
                             ("T3", self.reply("No supported refinement.", STOP))])
            episode = execute_public(c, caller)
            self.assertTrue(episode["normal_completion"])
            self.assertEqual([role for role, _ in caller.calls], ["T1", "T2", "T3"])
            self.assertEqual(episode["final_values"], [123.456789] * h)
            self.assertEqual(episode["geometry"], before["geometry"])
            self.assertEqual(c, before)
            self.assertEqual(episode["execution_mode"], "public")
            self.assertIsNone(episode["t1_review"])
            self.assertIsNone(episode["t1_supervision"])
            self.assertEqual(episode["accepted_t1"], episode["t1_pass1"]["reply"])
            for entry in episode["calls"]:
                self.assertIn("<think>", entry["messages"][0]["content"])
                self.assertEqual(entry["parsed"], protocol.parse_reply(entry["content"], "student"))
                validate_public_role(entry["protocol_role"], entry["parsed"], entry["public_variables"])
                self.assertNotIn("Privileged canonical", json.dumps(entry["messages"]))

    def test_factorized_h24_h96_reassessment_is_mandatory_and_refine_is_isolated(self):
        for h in (24, 96):
            p = plan(h, cuts=[h // 2], drivers=True)
            caller = Script([("T1", self.reply("Common public interpretation.", p)),
                             ("T2", self.report(p["spans"][0], 111, "OWN_REPORT")),
                             ("T2", self.report(p["spans"][1], 987.654, "SIBLING_REPORT")),
                             ("T3", self.reply("GLOBAL_ASSESSMENT sibling 987.654", refine(
                                 refs=["D0"], concern="mechanism_consistency", span=[5, 7]))),
                             ("T2", self.report(p["spans"][0], 112, "Reconsidered entire original segment.")),
                             ("T3", self.reply("Final required assessment.", STOP)),
                             ("T4", self.reply("Keep evidence-supported final segments.", KEEP))])
            episode = execute_public(case(h), caller, profile=Profile(max_refinements=1))
            self.assertTrue(episode["normal_completion"])
            self.assertEqual([role for role, _ in caller.calls], ["T1", "T2", "T2", "T3", "T2", "T3", "T4"])
            initial_second = json.dumps(caller.calls[2][1])
            self.assertNotIn("OWN_REPORT", initial_second)
            self.assertNotIn("111", initial_second)
            revision = json.dumps(caller.calls[4][1])
            self.assertIn("OWN_REPORT", revision)
            for sentinel in ("SIBLING_REPORT", "GLOBAL_ASSESSMENT", "987.654"):
                self.assertNotIn(sentinel, revision)
            self.assertEqual(episode["final_values"], [112] * (h // 2) + [987.654] * (h // 2))
            self.assertEqual(episode["t4_status"], "model_KEEP")
            self.assertEqual(episode["calls"][5]["public_variables"]["reflection_policy_state"]["eligible_segments"], [])
            for entry in episode["calls"]:
                validate_public_role(entry["protocol_role"], entry["parsed"], entry["public_variables"])

    def test_t4_bounded_edits_are_atomic_and_actual_receipts_preserved(self):
        p = plan(cuts=[12])
        edits = {"action": "REGENERATE", "edits": [
            {"boundary_id": "B0", "start": 11, "stop": 13, "values": [41.25, 42.5]}]}
        receipt = {"content": self.reply("Public seam correction.", edits), "finish_reason": "stop",
                   "response_ids": [12, 13], "response_logprobs": [-0.2, -0.3], "usage": {"completion_tokens": 2}}
        caller = Script([("T1", self.reply("Split.", p)),
                         ("T2", self.report(p["spans"][0], 11)), ("T2", self.report(p["spans"][1], 22)),
                         ("T3", self.reply("Stop.", STOP)), ("T4", receipt)])
        episode = execute_public(case(), caller)
        self.assertEqual(episode["final_values"], [11] * 11 + [41.25, 42.5] + [22] * 11)
        self.assertEqual(episode["calls"][-1]["raw_response"], receipt)
        self.assertEqual(episode["t4_status"], "model_REGENERATE")

    def test_failed_t4_is_fallback_not_completed_model_finalization(self):
        for failure in (RuntimeError("transport sentinel"), {
                "content": self.reply("KEEP exists but generation was truncated.", KEEP),
                "finish_reason": "length", "response_ids": [1, 2]}):
            p = plan(cuts=[12])
            caller = Script([("T1", self.reply("Split.", p)),
                             ("T2", self.report(p["spans"][0], 11)), ("T2", self.report(p["spans"][1], 22)),
                             ("T3", self.reply("Real STOP before failed T4.", STOP)), ("T4", failure)])
            episode = execute_public(case(), caller, profile=Profile(structural_retries=0))
            self.assertFalse(episode["normal_completion"])
            self.assertEqual(episode["t4_status"], "failed")
            self.assertEqual(episode["termination"]["kind"], "runtime_failure")
            self.assertFalse(episode["fallback"]["model_t4_completed"])
            self.assertEqual(episode["final_values"], [11] * 12 + [22] * 12)
            if isinstance(failure, dict):
                self.assertEqual(episode["calls"][-1]["raw_response"], failure)

    def test_direct_reference_calls_actual_constrained_t1_full_h_t2_only(self):
        for h in (24, 96):
            p = plan(h)
            caller = Script([("T1", self.reply("Constrained public DIRECT plan.", p)),
                             ("T2", self.report(p["spans"][0], 135.79))])
            episode = direct_cost_reference(case(h), caller)
            self.assertTrue(episode["normal_completion"])
            self.assertEqual([role for role, _ in caller.calls], ["T1", "T2"])
            self.assertEqual(episode["reference_protocol"], "program_DIRECT_STOP_T1_T2")
            self.assertEqual(episode["termination"]["kind"], "program_DIRECT_STOP_T1_T2")
            self.assertEqual(episode["final_values"], [135.79] * h)
            self.assertEqual(episode["reflection_history"], [])
            policy = episode["calls"][0]["public_variables"]["compute_policy"]
            self.assertEqual(policy["max_segments"], 1)
            self.assertEqual(policy["allowed_modes"], ["DIRECT"])
            self.assertEqual(policy["protected_t4_attempts"], 0)

    def test_direct_reference_rejects_factorization_before_any_t2(self):
        caller = Script([("T1", self.reply("Wrong factorized reference.", plan(cuts=[12])))])
        episode = direct_cost_reference(case(), caller, profile=Profile(structural_retries=0))
        self.assertFalse(episode["normal_completion"])
        self.assertEqual([role for role, _ in caller.calls], ["T1"])
        self.assertEqual(episode["termination"]["kind"], "runtime_failure")
        self.assertEqual(episode["fallback"]["kind"], "public_history_cycle")

    def test_public_private_fields_are_rejected_before_calls(self):
        for field in ("labels", "canonical_labels", "reward", "ground_truth"):
            c = case()
            c["private"] = {field: [9000] * 24}
            caller = Script([])
            with self.assertRaises(ValueError):
                execute_public(c, caller)
            self.assertEqual(caller.calls, [])

    def test_role_validator_respects_constrained_t1_and_t2_original_scope(self):
        p = plan()
        caller = Script([("T1", self.reply("Direct.", p)), ("T2", self.report(p["spans"][0], 100))])
        episode = direct_cost_reference(case(), caller)
        with self.assertRaises(ValueError):
            validate_public_role("T1", {"explanation": "Bad reference split.", "answer": plan(cuts=[12])},
                                 episode["calls"][0]["public_variables"])
        bad = deepcopy(episode["calls"][1]["parsed"])
        bad["answer"]["stop"] = 12
        bad["answer"]["values"] = [100] * 12
        with self.assertRaises(ValueError):
            validate_public_role("T2_refine", bad, episode["calls"][1]["public_variables"])


if __name__ == "__main__":
    unittest.main()
