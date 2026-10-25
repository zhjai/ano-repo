"""Synthetic scripted protocol example; no model or forecast-accuracy claim."""
from datetime import datetime, timedelta
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from castfactory.agentic import distillation_protocol as protocol
from castfactory.agentic.full_agent_v4 import execute_public, Profile


def public_case():
    start = datetime(2025, 1, 2)
    return {'case_id': 'synthetic-demo', 'source': 'synthetic-fixture', 'split': 'validation',
        'public_case': 'Synthetic hourly load history is 100 MW. A toy demand shift affects the second half of tomorrow.',
        'geometry': {'H': 24, 'future_timestamps': [(start+timedelta(hours=i)).isoformat() for i in range(24)],
                     'frequency': '1h', 'unit': 'MW'},
        'evidence_manifest': [{'evidence_id': 'H0', 'kind': 'history', 'as_of': None,
                              'availability_basis': 'retrospective', 'location': 'target_history'}],
        'history_values': [100] * 24}


class ScriptedCaller:
    def __init__(self, direct=False):
        self.spans = ([{'segment_id': 'S0', 'start': 0, 'stop': 24}] if direct else
            [{'segment_id': 'S0', 'start': 0, 'stop': 12}, {'segment_id': 'S1', 'start': 12, 'stop': 24}])
        self.direct, self.roles, self.t2_calls, self.t3_calls = direct, [], 0, 0

    def __call__(self, role, messages):
        self.roles.append(role)
        if role == 'T1':
            answer = {'drivers': [], 'global_uncertainty': 'Toy example only.', 'evidence_ids_used': [],
                      'mode': 'DIRECT' if self.direct else 'FACTORIZE', 'spans': self.spans}
        elif role == 'T2':
            span = self.spans[min(self.t2_calls, len(self.spans)-1)]
            value = 105 if self.t2_calls >= len(self.spans) else 100
            self.t2_calls += 1
            answer = {**span, 'values': [value]*(span['stop']-span['start']),
                      'uncertainty': 'Synthetic response.', 'self_check': 'Complete assigned segment.'}
        elif role == 'T3':
            refine = not self.direct and self.t3_calls == 0
            self.t3_calls += 1
            answer = {'action': 'REFINE' if refine else 'STOP', 'segment_id': 'S1' if refine else None,
                      'directive': {'concern': 'shape', 'review_span': None, 'driver_refs': [],
                                    'own_report_field': 'summary'} if refine else None}
        elif role == 'T4':
            answer = {'action': 'KEEP', 'edits': []}
        else:
            raise ValueError(role)
        return protocol.render_reply('Scripted demonstration response; no model was called.', answer, 'student')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--direct', action='store_true')
    p.add_argument('--output', type=Path)
    args = p.parse_args()
    caller = ScriptedCaller(args.direct)
    result = execute_public(public_case(), caller, profile=Profile(max_refinements=1))
    assert result['normal_completion'], result['termination']
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x') as f:
            json.dump(result, f, indent=2, allow_nan=False)
            f.write('\n')
    print(json.dumps({'synthetic_scripted_demo': True, 'roles': caller.roles,
                      'normal_completion': result['normal_completion'], 'forecast': result['final_values']}))

if __name__ == '__main__':
    main()
