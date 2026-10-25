"""DIRECT termination, model-call accounting and legacy SFT compatibility."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from castfactory.agentic import distillation_protocol as protocol, compact_protocol
from castfactory.agentic.full_agent_v4 import (
    Profile, STEP_TARGET_VERSION, attach_public_json_case, execute_public,
    execute_public_from_t1, direct_cost_reference,
)
from castfactory.training import full_agent_sft_v4 as sft
from tests.agentic.test_full_agent_v4 import case, plan, Script
from tests.agentic.test_compact_protocol import evidence
from tests.agentic.test_full_agent_sft_v4 import trajectory
from castfactory.agentic.full_agent_v4 import sft_rows


class DirectTerminationTests(unittest.TestCase):
    def test_direct_never_requests_reflection_across_profiles_and_horizons(self):
        for h in (24, 96):
            for version in (protocol.VERSION, 'agentic-json-rows-v1',
                            'agentic-compact-markdown-v1', STEP_TARGET_VERSION):
                with self.subTest(h=h, profile=version):
                    p = plan(h)
                    public = case(h)
                    child = {**p['spans'][0], 'values': [17.5]*h,
                             'uncertainty': 'Fixture', 'self_check': 'Complete'}
                    if version == protocol.VERSION:
                        render = lambda a, r: protocol.render_reply('Fixture.', a, 'student')
                    elif version == 'agentic-json-rows-v1':
                        from tests.agentic.test_json_rows_runtime import reply
                        render = lambda a, r: reply(a, r, h)
                    else:
                        render = lambda a, r: compact_protocol.render_role_reply(
                            {'explanation': 'Fixture.', 'answer': a}, r, h, protocol_version=version)
                    if version != protocol.VERSION:
                        public = attach_public_json_case(public, evidence(h))
                    caller = Script([('T1', render(p, 'T1')), ('T2', render(child, 'T2'))])
                    with patch('scripts.load_canonical.compact_view.render_compact_case', return_value='Synthetic input'), \
                         patch('scripts.load_canonical.compact_view.validate_compact_profile_assets'):
                        result = execute_public(public, caller, profile=Profile(profile_version=version))
                    self.assertTrue(result['normal_completion'], result['termination'])
                    self.assertEqual([r for r, _ in caller.calls], ['T1', 'T2'])
                    self.assertEqual(result['final_values'], [17.5]*h)
                    self.assertEqual(result['reflection_history'], [])
                    self.assertEqual(result['termination']['kind'], 'direct_forecast_complete')
                    self.assertEqual(result['t4_status'], 'not_applicable_DIRECT')
                    self.assertEqual(caller.turns, [])

    def test_accepted_t1_continuation_returns_after_only_t2(self):
        public = attach_public_json_case(case(), evidence())
        accepted = {'explanation': 'Existing accepted plan.', 'answer': plan()}
        requested = []
        def handler(**kw):
            requested.append(kw['role'])
            self.assertEqual(kw['role'], 'T2')
            reply = {'explanation': 'Forecast.', 'answer': {
                **kw['variables']['segment_assignment'], 'values': [4]*24,
                'uncertainty': 'Fixture', 'self_check': 'Complete'}}
            kw['validate'](reply)
            receipt = {'call_id': 0, 'role': 'T2', 'status': 'accepted'}
            kw['episode']['calls'].append(receipt)
            return reply, receipt
        with patch('scripts.load_canonical.compact_view.render_compact_case', return_value='Synthetic input'), \
                         patch('scripts.load_canonical.compact_view.validate_compact_profile_assets'):
            result = execute_public_from_t1(public, lambda *args: self.fail("Unexpected direct caller"), accepted,
                profile=Profile(profile_version=STEP_TARGET_VERSION), call_handler=handler)
        self.assertEqual(requested, ['T2'])
        self.assertTrue(result['normal_completion'])

    def test_failed_direct_t2_stays_failure_and_uses_public_history(self):
        p = plan()
        caller = Script([('T1', protocol.render_reply('Plan.', p, 'student')),
                         ('T2', RuntimeError('backend failed'))])
        result = execute_public(case(), caller, profile=Profile(structural_retries=0))
        self.assertEqual([r for r, _ in caller.calls], ['T1', 'T2'])
        self.assertFalse(result['normal_completion'])
        self.assertEqual(result['final_values'], list(range(24)))
        self.assertEqual(result['fallback']['kind'], 'public_history_cycle')
        self.assertEqual(result['termination']['kind'], 'runtime_failure')

    @patch.object(protocol, 'student_training_tokens', return_value=([1], [2]))
    def test_sft_new_direct_and_legacy_reflection_are_distinguished(self, tokenizer):
        current = sft_rows(trajectory(identity='direct'))
        self.assertEqual([r['role'] for r in current], ['T1', 'T2'])
        old_t3 = deepcopy(next(r for r in sft_rows(trajectory(True, 'segmented')) if r['role'] == 'T3'))
        old_t3.update(case_id='direct', episode_id='direct', source_call_id=2)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'rows.jsonl'
            for rows, expected in ((current, 0), (current+[old_t3], 1)):
                path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
                features, report = sft.load_release(path, object())
                self.assertEqual(report['legacy_direct_reflection_episodes'], expected)
                self.assertEqual(len(features), len(rows))

    def test_factorized_sft_requires_assessment(self):
        rows = [r for r in sft_rows(trajectory(True)) if r['role'] != 'T3']
        with self.assertRaisesRegex(ValueError, 'requires T3'):
            sft._complete_rows(rows)
