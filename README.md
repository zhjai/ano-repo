# ElecLoad

Code for **ElecLoad: Adaptive Electricity Load Forecasting Guided by Multiple Events**.
ElecLoad analyzes event relevance and timing, plans the forecast horizon, generates segment forecasts, selectively revises them, and reconciles boundaries.

This release provides the implemented four-role runtime, prompt and response contracts, native Hugging Face inference, evaluation metrics, and a portable full-parameter episode SFT entrypoint. Internal Python imports retain the `castfactory` namespace. No datasets, trained weights, or paper-result artifacts are included; this source snapshot is not identified as the revision that produced the paper's reported numbers.

## Quick start

Python 3.10 or newer is required. The protocol demo and core tests need only Python:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python examples/demo.py
python examples/demo.py --direct
python -m unittest discover -s tests -t . -v
```

The demo uses synthetic inputs and explicitly scripted responses. It verifies execution mechanics and does not measure forecasting accuracy. The segmented example calls `T1 → T2 → T2 → T3 → T2 → T3 → T4`.

In this initial source snapshot, ordinary DIRECT execution calls `T1 → T2 → T3`; T4 is skipped. The separate `direct_cost_reference` entrypoint restricts planning to DIRECT and returns after T2. See [runtime details](docs/runtime.md).

## Inference with a local model

Install PyTorch for your accelerator first, then the model dependencies:

```bash
python -m pip install -e '.[hf]'
python scripts/predict.py \
  --model models/your-trained-model \
  --case examples/public_case.json \
  --config configs/inference.json \
  --output outputs/prediction.json
```

Supply a local causal-language-model checkpoint and its native chat tokenizer. The checkpoint must support the selected response contract; an untrained base model is not an ElecLoad-trained model. No automatic weight download is performed. For CPU use `--device cpu --dtype float32`. The caller records actual token IDs and token costs, rejects truncation, and distinguishes fallback from normal completion. Exit status 2 means a fallback forecast was returned.

`--direct-reference` measures the explicitly forced DIRECT reference workflow. The optional exact-token vLLM adapter is included for integration, but a standalone vLLM server launcher is not part of this release.

## Evaluate saved forecasts

Keep future labels separate from the public input. A labels file has the form `{"case_id":"synthetic-demo","values":[...]}`, with exactly H real numbers.

```bash
python scripts/evaluate.py --prediction outputs/prediction.json \
  --case examples/public_case.json --labels path/to/labels.json
```

The evaluator reports per-case WAPE and NRMSE as fractions, plus MAE/RMSE in source units. It retains fallback status. It does not aggregate unrelated units or manufacture missing labels. See [data format](docs/data.md).

## Supervised fine-tuning

The portable HF entrypoint uses the source implementation's complete-episode objective: sum supervised response-token losses within each episode, divide by that episode's supervised token count, then average episodes. Calls retain separate contexts. This is the historical HF backend; it is not the later cluster-specific FSDP trainer.

```bash
python -m castfactory.training.full_agent_sft_v4 \
  --profile-version merged-t1-answer-v1 \
  --model models/your-initialization --dataset path/to/teacher_roles.jsonl \
  --out outputs/sft --epochs 2 --batch-size 1 --gradient-accumulation-steps 1 \
  --learning-rate 0.000005 --weight-decay 0.01 --warmup-ratio 0.05 \
  --lr-scheduler-type cosine --max-grad-norm 1 --seed 17 \
  --context-tokens 32768 --dtype bfloat16 --device cuda \
  --gradient-checkpointing on --report-to none --dry-run
```

Remove `--dry-run` to train. These are explicit example settings, not a claim to reproduce a paper experiment. The loader requires genuine teacher-role provenance, complete episodes, and natural coverage of T1–T4; it rejects truncation. The output directory must be new. Checkpoints and the final HF export receive reload checks; validation-based BEST selection is not included. See [training and release boundaries](docs/training.md).

## Source map

| Component | Location |
|---|---|
| T1–T4 execution, validation, fallback | `castfactory/agentic/full_agent_v4.py` |
| Natural-language prompt resources | `castfactory/agentic/prompts/merged_t1_answer_v1/` |
| Compact/JSON protocols and rendering | `castfactory/agentic/`, `scripts/load_canonical/` |
| Actual-token HF and vLLM callers | `castfactory/training/full_agent_sampling_v4.py` |
| Episode SFT loader and HF trainer | `castfactory/training/full_agent_sft_v4.py` |
| Episode streaming and loss | `castfactory/training/full_agent_sft_execution.py` |
| Future-normalized metrics and rewards | `castfactory/training/full_agent_metrics.py`, `full_agent_reward.py` |
| Contract, leakage, runtime and objective tests | `tests/agentic/` |

[Release notes](docs/release.md) document provenance, portable changes, and verification boundaries. This project replaces the previous STaR working tree; that unrelated project's history remains in Git.
