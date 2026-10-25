"""Run the released native HF caller on an explicitly supplied public case."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def preflight_case(case, config, public_directory=None, reviewed_text=None):
    """Validate input and optional compact assets before importing model libraries."""
    from castfactory.agentic.full_agent_v4 import Profile, _validate_case, attach_public_json_case
    if 'message_token_counter' in config:
        raise ValueError('Token counting is supplied by the native tokenizer, not config')
    # Only check configuration here; the real counter is bound after loading
    # the tokenizer and is the only counter used for runtime admission.
    Profile(**config, message_token_counter=lambda messages: 0)
    _validate_case(case, train_only=False)
    version = config.get('profile_version', 'merged-t1-answer-v1')
    if case.get('profile_version', version) != version:
        raise ValueError('Case profile differs from the selected config')
    if bool(public_directory) != bool(reviewed_text):
        raise ValueError('Supply both compact public directory and reviewed-text sidecar')
    compact = version in ('agentic-compact-markdown-v1', 'agentic-compact-markdown-v2')
    if public_directory and not compact:
        raise ValueError('Compact assets require a compact profile')
    if compact or version == 'agentic-json-rows-v1':
        if 'public_json_case' not in case:
            raise ValueError('Selected profile requires public_json_case')
        attach_public_json_case(case, case['public_json_case'])
    if compact:
        from scripts.load_canonical.compact_view import (
            configure_compact_assets, validate_compact_profile_assets, render_compact_case,
            INPUT_VERSION, EXPLICIT_TIME_INPUT_VERSION,
        )
        revision = EXPLICIT_TIME_INPUT_VERSION if version.endswith('v2') else INPUT_VERSION
        if public_directory:
            configure_compact_assets(revision, public_directory, reviewed_text)
        validate_compact_profile_assets(revision)
        render_compact_case(case['public_json_case'], selected_case_id=case['case_id'], input_version=revision)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case', type=Path, required=True)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--config', type=Path, default=Path(__file__).resolve().parents[1] / 'configs/inference.json')
    p.add_argument('--device', default='cuda', choices=['cpu', 'cuda'])
    p.add_argument('--dtype', choices=['float32', 'bfloat16'], default='bfloat16')
    p.add_argument('--direct-reference', action='store_true', help='Force DIRECT for measured cost reference')
    p.add_argument('--compact-public-directory', type=Path)
    p.add_argument('--compact-reviewed-text', type=Path)
    args = p.parse_args()
    if args.output.exists():
        p.error('Output already exists; choose a new path')
    case = json.loads(args.case.read_text())
    config = json.loads(args.config.read_text())
    try:
        preflight_case(case, config, args.compact_public_directory, args.compact_reviewed_text)
    except (ValueError, TypeError) as exc:
        p.error(str(exc))
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from castfactory.agentic.full_agent_v4 import Profile, execute_public, direct_cost_reference
    from castfactory.agentic.distillation_protocol import student_prompt_tokens
    from castfactory.training.full_agent_sampling_v4 import NativeFullRoleSamplerV4
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
