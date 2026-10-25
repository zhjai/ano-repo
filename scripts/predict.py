"""Run the released native HF caller on an explicitly supplied public case."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case', type=Path, required=True)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--config', type=Path, default=Path(__file__).resolve().parents[1] / 'configs/inference.json')
    p.add_argument('--device', default='cuda', choices=['cpu', 'cuda'])
    p.add_argument('--dtype', choices=['float32', 'bfloat16'], default='bfloat16')
    p.add_argument('--direct-reference', action='store_true', help='Force DIRECT for measured cost reference')
    args = p.parse_args()
    if args.output.exists():
        p.error('Output already exists; choose a new path')
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from castfactory.agentic.full_agent_v4 import Profile, execute_public, direct_cost_reference
    from castfactory.agentic.distillation_protocol import student_prompt_tokens
    from castfactory.training.full_agent_sampling_v4 import NativeFullRoleSamplerV4
    case = json.loads(args.case.read_text())
    config = json.loads(args.config.read_text())
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    profile = Profile(**config, message_token_counter=lambda m: len(student_prompt_tokens(tokenizer, m)[0]))
    model = AutoModelForCausalLM.from_pretrained(args.model, local_files_only=True,
        dtype=getattr(torch, args.dtype)).to(args.device)
    caller = NativeFullRoleSamplerV4(model, tokenizer, profile, str(args.model.resolve()), stochastic=False)
    runner = direct_cost_reference if args.direct_reference else execute_public
    episode = runner(case, caller, profile=profile)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as f:
        json.dump(episode, f, indent=2, allow_nan=False)
        f.write('\n')
    print(json.dumps({'normal_completion': episode['normal_completion'], 'termination': episode['termination'],
                      'roles': [c['role'] for c in episode['calls']], 'output': str(args.output)}))
    return 0 if episode['normal_completion'] else 2

if __name__ == '__main__':
    raise SystemExit(main())
