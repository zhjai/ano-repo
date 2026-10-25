"""Train Reasoning initialization with the released original HF SFT backend."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def read_rows(path, expected_split):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError('Empty dataset: ' + str(path))
    for row in rows:
        split = row.get('metadata', {}).get('split')
        if split != expected_split:
            raise ValueError('Each row must explicitly declare metadata.split=' + expected_split)
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--train', type=Path, required=True)
    p.add_argument('--validation', type=Path)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    p.add_argument('--dtype', choices=['float32', 'bfloat16'], default='bfloat16')
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()
    if args.out.exists():
        p.error('Output already exists; choose a new directory')
    config = json.loads(args.config.read_text())
    if set(config) != {'backend', 'training_args'}:
        p.error('Config requires exactly backend and training_args')
    if any(key in config['training_args'] for key in ('output_dir', 'resume_from_checkpoint', 'use_cpu', 'bf16', 'fp16')):
        p.error('Output, fresh-run status, device and dtype are CLI-controlled')
    if args.device == 'cpu' and args.dtype != 'float32':
        p.error('CPU training requires --dtype float32')
    rows = read_rows(args.train, 'train')
    validation = read_rows(args.validation, 'validation') if args.validation else None
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    from castfactory.training.transformers_sft_backend import TransformersSFTBackend
    class StrictBackend(TransformersSFTBackend):
        def _truncate_ids(self, ids, max_length):
            if len(ids) > max_length:
                raise ValueError('Prompt exceeds configured budget; increase lengths instead of truncating evidence')
            return super()._truncate_ids(ids, max_length)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    backend = StrictBackend(tokenizer=tokenizer, **config['backend'])
    prepared = backend.prepare_dataset(rows)
    if validation is not None:
        backend.prepare_dataset(validation)
    report = {'training_examples': len(prepared), 'validation_examples': len(validation or []),
              'max_tokens': max(len(r['input_ids']) for r in prepared), 'dry_run': args.dry_run}
    if not args.dry_run:
        model = AutoModelForCausalLM.from_pretrained(args.model, local_files_only=True,
                                                   dtype=getattr(torch, args.dtype)).to(args.device)
        training = dict(config['training_args'], use_cpu=args.device == 'cpu',
                        bf16=args.dtype == 'bfloat16', fp16=False)
        report.update(backend.fit(rows, validation, args.out, model=model, train_args=training))
        (args.out/'training_report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(report, indent=2, allow_nan=False))

if __name__ == '__main__':
    main()
