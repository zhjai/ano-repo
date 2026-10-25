"""Public JSON-role execution witnesses without models, APIs or artifacts."""
from copy import deepcopy
import json
from pathlib import Path
import unittest

from castfactory.agentic import distillation_protocol as protocol
from castfactory.agentic import json_rows_prompts as prompts
from castfactory.agentic.json_rows_protocol import parse_role_answer
from castfactory.agentic.full_agent_v4 import (
    Profile, JSON_ROWS_VERSION, attach_public_json_case, execute, execute_public,
    direct_cost_reference,
)
from tests.agentic.test_full_agent_v4 import case, plan, refine, STOP, KEEP, Script


def public_case(h=24):
    prepared = case(h)
    prepared['split'] = 'validation'
    evidence = {'task': {'prediction_length': h},
                'grid': {'future_timestamps': prepared['geometry']['future_timestamps'],
                         'history_timestamps': []},
                'variables': {'load': {'history': {'unit': None}}},
                'history': {'columns': ['position', 'load'],
                            'rows': [[i-24, v] for i, v in enumerate(prepared['history_values'])]},
                'future_known': {'columns': ['position'], 'rows': []},
                'context': {}, 'events': []}
    return attach_public_json_case(prepared, evidence)


def reply(answer, role='T1', h=24, explanation='Evidence supports this public decision.'):
    return protocol.render_reply(explanation, prompts.table_answer(answer, role, h), 'student')


def child(segment, value, h=24, explanation='Independent local evidence.'):
    return reply({**segment, 'values': [value]*(segment['stop']-segment['start']),
                  'uncertainty': 'Unknown changes.', 'self_check': 'Checked original segment.'},
                 'T2', h, explanation)


def profile(**kwargs):
    return Profile(profile_version=JSON_ROWS_VERSION, **kwargs)


class JSONRuntimeTests(unittest.TestCase):
    def test_direct_h24_h96_raw_rows_and_normalized_receipts(self):
        for h in (24, 96):
            p = plan(h)
            caller = Script([('T1', reply(p, h=h)), ('T2', child(p['spans'][0], 1.23456789, h)),
                             ('T3', reply(STOP, 'T3', h))])
            episode = execute_public(public_case(h), caller, profile=profile())
            self.assertTrue(episode['normal_completion'])
            self.assertEqual(episode['profile_version'], JSON_ROWS_VERSION)
            self.assertEqual(episode['final_values'], [1.23456789]*h)
            for call in episode['calls']:
                request = json.loads(call['messages'][1]['content'])
                self.assertEqual(request, {'role': call['protocol_role'], 'case': call['public_json_case'],
                                           **call['public_variables']})
                self.assertEqual(call['parsed'], parse_role_answer(call['content'], call['protocol_role'], request))
                self.assertNotIn('public_case', call['public_variables'])
                self.assertEqual(call['messages'][0]['content'], prompts.COMMON+'\n\n'+prompts.ROLE_RULES[call['role']])
            child_call = episode['calls'][1]
            self.assertNotIn('values', child_call['parsed']['answer'])
            self.assertEqual(child_call['parsed']['answer']['rows'][0], [0, 1.23456789])
            self.assertEqual(child_call['normalized_parsed']['answer']['values'], [1.23456789]*h)

    def test_factorize_refine_reassessment_and_atomic_multiseam(self):
        p = plan(cuts=[8, 16], drivers=True)
        action = refine(refs=['D0'], concern='mechanism_consistency', span=[4, 6])
        edits = {'action': 'REGENERATE', 'edits': [
            {'boundary_id': 'B0', 'start': 7, 'stop': 8, 'values': [71.125]},
            {'boundary_id': 'B1', 'start': 16, 'stop': 18, 'values': [161.5, 162.25]}]}
        turns = [('T1', reply(p)), ('T2', child(p['spans'][0], 100, explanation='OWN_PUBLIC')),
                 ('T2', child(p['spans'][1], 876.543, explanation='SIBLING_SENTINEL')),
                 ('T2', child(p['spans'][2], 300)),
                 ('T3', reply(action, 'T3', explanation='GLOBAL_SENTINEL 876.543')),
                 ('T2', child(p['spans'][0], 100, explanation='Concern declined; whole segment unchanged.')),
                 ('T3', reply(STOP, 'T3')), ('T4', reply(edits, 'T4'))]
        caller = Script(turns)
        episode = execute_public(public_case(), caller, profile=profile(max_refinements=1))
        self.assertTrue(episode['normal_completion'])
        self.assertEqual(episode['final_values'], [100]*7+[71.125]+[876.543]*8+[161.5,162.25]+[300]*6)
        initial = json.loads(caller.calls[2][1][1]['content'])
        self.assertNotIn('current_segment_reports', initial)
        revised = json.dumps(json.loads(caller.calls[5][1][1]['content']))
        for forbidden in ('GLOBAL_SENTINEL','SIBLING_SENTINEL','876.543','"values"'):
            self.assertNotIn(forbidden, revised)
        self.assertIn('OWN_PUBLIC', revised)
        request = json.loads(caller.calls[6][1][1]['content'])
        self.assertEqual(request['reflection_policy_state']['eligible_segments'], [])
        self.assertEqual(episode['t4_status'], 'model_REGENERATE')
        self.assertEqual(episode['calls'][-1]['parsed']['answer']['edits'][1]['rows'], [[16,161.5],[17,162.25]])

    def test_invalid_rows_retry_remains_actual_json_conversation(self):
        p = plan()
        correct = child(p['spans'][0], 1.5)
        parsed = protocol.parse_reply(correct,'student')
        parsed['answer']['rows'][0][0] = False
        failed = protocol.render_reply(parsed['explanation'],parsed['answer'],'student')
        caller = Script([('T1',reply(p)),('T2',failed),('T2',correct),('T3',reply(STOP,'T3'))])
        episode = execute_public(public_case(),caller,profile=profile())
        self.assertTrue(episode['normal_completion'])
        self.assertEqual(episode['calls'][1]['status'],'failed')
        retry = episode['calls'][2]
        self.assertEqual(retry['messages'][2],{'role':'assistant','content':failed})
        repair = json.loads(retry['messages'][3]['content'])
        self.assertEqual(set(repair), {'role','repair_instruction','assignment_and_policy'})
        self.assertEqual(repair['role'],'T2')
        self.assertEqual(retry['parsed']['answer']['rows'][0],[0,1.5])

    def test_bad_t4_rows_fail_without_partial_application(self):
        p = plan(cuts=[12])
        answer = {'action':'REGENERATE','edits':[{'boundary_id':'B0','start':11,'stop':13,
                   'columns':['position','load'],'rows':[[11,9],[11,10]]}]}
        caller=Script([('T1',reply(p)),('T2',child(p['spans'][0],100)),
                       ('T2',child(p['spans'][1],200)),('T3',reply(STOP,'T3')),
                       ('T4',protocol.render_reply('Invalid second position.',answer,'student'))])
        episode=execute_public(public_case(),caller,profile=profile(structural_retries=0))
        self.assertFalse(episode['normal_completion'])
        self.assertEqual(episode['final_values'],[100]*12+[200]*12)
        self.assertEqual(episode['t4_status'],'failed')
        self.assertFalse(episode['fallback']['model_t4_completed'])

    def test_strict_schema_mixed_values_and_bad_numbers(self):
        p=plan(); base=child(p['spans'][0],2)
        variables={'geometry':public_case()['geometry'],'segment_assignment':p['spans'][0]}
        request=prompts.request_for('T2',variables,public_case()['public_json_case'])
        parsed=protocol.parse_reply(base,'student')
        for rows in ([[i+1,2] for i in range(24)],[[i,True] for i in range(24)],[[0,2]],
                     [[i,2] for i in range(23,-1,-1)]):
            bad=deepcopy(parsed);bad['answer']['rows']=rows
            with self.assertRaises(ValueError):
                parse_role_answer(protocol.render_reply(bad['explanation'],bad['answer'],'student'),'T2',request)
        bad=deepcopy(parsed);bad['answer']['values']=[2]*24
        with self.assertRaisesRegex(ValueError,'also carry legacy'):
            parse_role_answer(protocol.render_reply(bad['explanation'],bad['answer'],'student'),'T2',request)
        with self.assertRaises(ValueError):
            parse_role_answer(base.replace('[0,2]','[0,NaN]'),'T2',request)

    def test_profile_is_explicit_and_private_targets_rejected(self):
        with self.assertRaises(ValueError):
            Profile(profile_version='unknown')
        with self.assertRaisesRegex(ValueError,'requires attached'):
            execute_public(case(),lambda *_:None,profile=profile())
        with self.assertRaisesRegex(ValueError,'public student'):
            execute(public_case(),lambda *_:None,profile=profile(),reviewer=lambda *_:None)
        public=public_case(); evidence=deepcopy(public['public_json_case'])
        evidence['future_known']['columns'].append('load')
        with self.assertRaisesRegex(ValueError,'Future target'):
            attach_public_json_case(case(),evidence)
        evidence=deepcopy(public['public_json_case']);evidence['context']['canonical_labels']=[1]*24
        with self.assertRaisesRegex(ValueError,'Private field'):
            attach_public_json_case(case(),evidence)
        self.assertNotIn('profile_version',Profile().record())

    def test_direct_reference_is_constrained_and_profile_bound(self):
        p=plan(); caller=Script([('T1',reply(p)),('T2',child(p['spans'][0],42))])
        episode=direct_cost_reference(public_case(),caller,profile=profile())
        self.assertTrue(episode['normal_completion'])
        self.assertEqual(episode['profile_version'],JSON_ROWS_VERSION)
        self.assertEqual(episode['termination']['kind'],'program_DIRECT_STOP_T1_T2')
        request=json.loads(caller.calls[0][1][1]['content'])
        self.assertEqual(request['compute_policy']['allowed_modes'],['DIRECT'])

    def test_native_think_prefix_and_eos_preserve_json_answer(self):
        tokenizer_path = Path(str(Path(__import__("os").environ.get("ELECLOAD_MODELS_DIR", "models")) / 'Qwen3.5-4B'))
        if not tokenizer_path.is_dir():
            self.skipTest('local Qwen3.5-4B tokenizer absent')
        try:
            from transformers import AutoTokenizer
        except ImportError:
            self.skipTest('transformers not installed')
        from castfactory.training import full_agent_serialization as serialization
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
        public = public_case(); p = plan()
        variables = {'geometry':public['geometry'],'segment_assignment':p['spans'][0],
                     'local_t1_information':{},'mode':'INITIAL'}
        messages = prompts.messages_for('T2',variables,public['public_json_case'])
        messages += [{'role':'assistant','content':'<think>Failed.</think><answer>{"rows":[',
                      'reasoning_content':'PRIVATE_REASONING_SENTINEL'},
                     prompts.repair_message('T2','Repair the complete original segment.')]
        answer = child(p['spans'][0],1.23456789)
        prompt_ids, prefix = protocol.student_prompt_tokens(tokenizer,messages)
        self.assertEqual(prefix,'<think>\n')
        self.assertEqual(serialization.student_prompt(tokenizer,messages,profile_version=JSON_ROWS_VERSION),
                         (prompt_ids,prefix))
        train_prompt, response = protocol.student_training_tokens(tokenizer,messages,answer)
        self.assertEqual(train_prompt,prompt_ids)
        self.assertEqual(response.count(tokenizer.eos_token_id),1)
        self.assertEqual(response[-1],tokenizer.eos_token_id)
        self.assertNotIn('PRIVATE_REASONING_SENTINEL',tokenizer.decode(prompt_ids,skip_special_tokens=False))
        decoded = tokenizer.decode(response[:-1],skip_special_tokens=False,clean_up_tokenization_spaces=False)
        self.assertFalse(decoded.startswith('<think>'))
        parsed = parse_role_answer(prefix+decoded,'T2',json.loads(messages[1]['content']))
        self.assertEqual(parsed['answer']['rows'],[[i,1.23456789] for i in range(24)])


if __name__=='__main__':
    unittest.main()
