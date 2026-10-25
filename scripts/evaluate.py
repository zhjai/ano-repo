"""Score one saved episode against separately supplied labels (H24/H96)."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from castfactory.training.full_agent_metrics import forecast_metrics


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prediction', type=Path, required=True)
    p.add_argument('--labels', type=Path, required=True, help='JSON with matching case_id and values')
    p.add_argument('--case', type=Path, required=True)
    args = p.parse_args()
    prediction, labels, public = [json.loads(x.read_text()) for x in (args.prediction, args.labels, args.case)]
    if prediction['case_id'] != labels['case_id'] or public['case_id'] != labels['case_id']:
        p.error('Case identities differ')
    if prediction['geometry'] != public['geometry']:
        p.error('Prediction geometry differs from the supplied case')
    case = SimpleNamespace(horizon=public['geometry']['H'],
        targets=SimpleNamespace(future_values=labels['values']),
        observation=SimpleNamespace(target_history=public['history_values']))
    metrics = forecast_metrics(prediction['final_values'], case, declared_horizon=True)
    metrics['normal_completion'] = prediction['normal_completion']
    metrics['fallback'] = prediction.get('fallback')
    print(json.dumps(metrics, indent=2, allow_nan=False))

if __name__ == '__main__':
    main()
