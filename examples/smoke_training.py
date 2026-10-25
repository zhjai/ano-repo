"""CPU-only real optimizer/export smoke with random tiny models and synthetic data.

No external model, dataset, GPU, API or experiment result is used.
"""
import argparse
import json
from pathlib import Path
import sys
import tempfile
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def make_model_and_tokenizer(directory):
    import torch
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM
    torch.manual_seed(17)
    raw = Tokenizer(WordLevel({'<pad>': 0, '<unk>': 1, '<eos>': 2,
        '<system>': 3, '<user>': 4, '<assistant>': 5, 'load': 6, 'forecast': 7,
        '100': 8, '101': 9, '102': 10, '24': 11}, unk_token='<unk>'))
    raw.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=raw, pad_token='<pad>',
                                       unk_token='<unk>', eos_token='<eos>')
    tokenizer.add_special_tokens({'additional_special_tokens': ['<system>', '<user>', '<assistant>']})
    tokenizer.chat_template = "{% for message in messages %}{{ '<' + message['role'] + '>' + message['content'] + eos_token }}{% endfor %}{% if add_generation_prompt %}{{ '<assistant>' }}{% endif %}"
    model = Qwen3ForCausalLM(Qwen3Config(vocab_size=len(tokenizer), hidden_size=16,
        intermediate_size=32, num_hidden_layers=1, num_attention_heads=2,
        num_key_value_heads=1, head_dim=8, max_position_embeddings=8192,
        eos_token_id=2, pad_token_id=0, use_cache=False))
    model.save_pretrained(directory)
    tokenizer.save_pretrained(directory)
    return model, tokenizer


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, help='New output directory; otherwise use a temporary directory')
    args = p.parse_args()
    if args.out and args.out.exists():
        p.error('Output exists')
    import torch
    torch.set_num_threads(1)
    from transformers import AutoModelForCausalLM
    from castfactory.training.transformers_sft_backend import TransformersSFTBackend
    from castfactory.training.full_agent_sft_v4 import main as agent_main
    from castfactory.agentic.full_agent_v4 import execute, sft_rows
    from castfactory.agentic import distillation_protocol as protocol
    from examples.demo import public_case, ScriptedCaller
    temporary = tempfile.TemporaryDirectory() if not args.out else None
    root = Path(temporary.name) if temporary else args.out
    root.mkdir(parents=True, exist_ok=True)
    initial = root/'initial'
    model, tokenizer = make_model_and_tokenizer(initial)
    initial_weights = {n: v.detach().clone() for n, v in model.named_parameters()}
    rows = [{'messages': [{'role': 'user', 'content': 'load forecast'},
                         {'role': 'assistant', 'content': '100 101 102'}],
             'metadata': {'split': 'train'}}]
    backend = TransformersSFTBackend(model=model, tokenizer=tokenizer,
        max_length=64, max_prompt_length=32, max_response_length=32)
    reasoning = backend.fit(rows, None, root/'reasoning', train_args={
        'max_steps': 1, 'per_device_train_batch_size': 1, 'learning_rate': .001,
        'warmup_steps': 0, 'lr_scheduler_type': 'constant', 'save_strategy': 'no',
        'report_to': [], 'use_cpu': True, 'disable_tqdm': True})
    reasoning_delta = max(float((p.detach()-initial_weights[n]).abs().max()) for n,p in model.named_parameters())
    assert reasoning_delta > 0
    reasoning_reloaded = AutoModelForCausalLM.from_pretrained(root/'reasoning', local_files_only=True)
    for name, parameter in reasoning_reloaded.named_parameters():
        torch.testing.assert_close(parameter, dict(model.named_parameters())[name])
    del reasoning_reloaded
    role_rows = []
    for direct in (True, False):
        fixture = public_case()
        fixture.update(split='train', case_id='synthetic-direct' if direct else 'synthetic-segmented')
        scripted = ScriptedCaller(direct)
        def teacher(role, messages):
            parsed = protocol.parse_reply(scripted(role, messages), 'student')
            return protocol.render_reply(parsed['explanation'], parsed['answer'], 'teacher')
        episode = execute(fixture, teacher, reviewer=lambda *args: protocol.render_reply(
            'Synthetic test-only review.', {'decision': 'KEEP'}, 'teacher'))
        assert episode['normal_completion']
        role_rows.extend(sft_rows(episode))
    data = root/'teacher-fixture.jsonl'
    data.write_text(''.join(json.dumps(r)+'\n' for r in role_rows))
    agent_main(['--profile-version', protocol.VERSION, '--model', str(initial), '--dataset', str(data),
        '--out', str(root/'agent'), '--epochs', '1', '--batch-size', '2',
        '--gradient-accumulation-steps', '1', '--learning-rate', '.001', '--weight-decay', '0',
        '--warmup-ratio', '0', '--lr-scheduler-type', 'constant', '--max-grad-norm', '1',
        '--seed', '17', '--context-tokens', '8192', '--dtype', 'float32', '--device', 'cpu',
        '--gradient-checkpointing', 'off', '--report-to', 'none'])
    reloaded = AutoModelForCausalLM.from_pretrained(root/'agent/final_hf', local_files_only=True)
    agent_delta = max(float((p.detach()-initial_weights[n]).abs().max()) for n,p in reloaded.named_parameters())
    assert agent_delta > 0
    report = {'synthetic_random_model_smoke': True, 'reasoning_parameter_max_change': reasoning_delta,
              'agent_parameter_max_change': agent_delta, 'reasoning_optimizer_steps': 1,
              'agent_optimizer_steps': json.loads((root/'agent/training_report.json').read_text())['optimizer_steps'],
              'reasoning_hf_reload': True, 'agent_hf_reload': True, 'teacher_role_rows': len(role_rows)}
    (root/'smoke_report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    if temporary:
        temporary.cleanup()

if __name__ == '__main__':
    main()
