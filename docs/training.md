# Training implementation and scope

The HF SFT backend is exported from the existing `full_agent_sft_v4.py` implementation. It keeps its tokenizer mask construction, complete-episode grouping, exact episode sampling, loss normalization, role-wise graph release, optimizer configuration, checkpoint reload checks, and CLI. Private offline corpus-repair paths fail explicitly rather than impersonating teacher provenance. Cluster FSDP receipt readers and RL BEST-selection replay are omitted; HF training does not invoke those paths.

The native FSDP SFT runner and current RL orchestration are not shipped; their reusable loss kernels are included. They depend on a modified VERL tree, frozen source-specific populations, provenance/seed handoff files, and job-specific recovery/admission records. Relocating paths alone would not produce a standalone equivalent trainer: its safety gates bind actual case identities, selected checkpoints, and native optimizer recovery. This release does not relax those checks or substitute the older STaR VERL copy. The existing HF backend is exposed explicitly instead.

`full_agent_reward.py` includes versioned reward arithmetic. Select profiles deliberately; the historical default and newer bounded variants are not interchangeable. Neither these utilities nor a successful protocol test establish trained forecasting quality.

The supplied HF command performs genuine optimization when run without `--dry-run` on valid inputs. A GPU training run and reproduction of paper results have not been performed as part of this release. Full-model export verification loads another model copy and therefore requires additional memory. No trained checkpoint or BEST-selection claim is bundled.

Reference source environment: Python 3.12, PyTorch 2.10.0, Transformers 5.3.0.dev0, Accelerate 1.14.0. Public packaging specifies compatible major-version ranges; accelerator wheels must match the user's system. Optional tests requiring model tokenizers skip when `ELECLOAD_MODELS_DIR` does not contain them. Core protocol tests run without ML dependencies.

After the DIRECT correction, newly collected DIRECT episodes contain T1 and T2 only. The SFT loader still accepts historical DIRECT episodes with T3, retains those actual role targets, and reports `legacy_direct_reflection_episodes`. It does not silently relabel historical trajectories as the corrected workflow. FACTORIZE continues to require T3 and T4.

## Reasoning initialization

`castfactory.training.transformers_sft_backend.TransformersSFTBackend` is the source HF implementation, including native chat encoding, response masking, response-only logit support, forecast-aware validation options and model export. `scripts/train_reasoning.py` wraps it with local-only model loading, explicit train/validation split checks, fresh output directories and strict prompt/response budgets. `configs/reasoning_sft.json` provides an explicit starting configuration. The standalone wrapper supports CPU float32 or CUDA bfloat16; `--dry-run` tokenizes and validates without loading model weights. This backend's ordinary response-token loss is distinct from Agent's complete-episode mean objective.

`castfactory.time_r1.reward_load_v3` exports the existing forecast-answer reward and its parser dependencies. Reward profiles and answer formats must match the training data. Installing `.[reasoning]` provides the NumPy dependency for reward utilities; `.[train]` additionally supplies the model libraries needed by both trainers.

## Reusable Agent training objectives

- `paired_credit_loss.credit_objective(new, data, objective_version=...)` computes the differentiable clipped policy, contract, KL and version-dependent paired/local credit terms. `new` holds current response log probabilities; `data` contains corresponding old probabilities, masks, advantages, globally normalized weights and, where required, complete pair indices and branch scores. Select the objective version explicitly; the historical default is not the newest objective. The function returns loss, metrics and component tensors. Analytical gradient tests demonstrate clipping, frozen labels, global weight preservation across microbatches and paired preference updates.
- `paired_credit_loss.paired_credit_loss` adapts that objective to the native VERL worker's transport and distributed reduction convention.
- `agentic_fsdp_loss.agentic_sft_loss` provides the native nested-tensor episode SFT objective, using actual global episode denominators and coordinated nonfinite failure checks before backward.

The last two hooks require the matching externally supplied modified VERL backend; installing generic VERL is not a verified substitute. This release supplies neither an RL launcher nor the rollout/population/seed/recovery records required by that backend. Loss-kernel tests do not establish an end-to-end RL training run.

## CPU training smoke

`python examples/smoke_training.py` creates a random tiny Qwen3 model and synthetic tokenizer/cases locally, performs one optimizer step in each released SFT backend, and reloads both HF exports. `--out path/to/new-directory` retains the generated artifacts. Both trainers changed parameters in the reference CPU run; Agent checkpoint reload/parity and EOS receipts also passed. Transformers may save a scalar native EOS as a singleton list; receipt verification accepts that equivalent representation and still rejects additional or different EOS IDs. No GPU, downloaded weights or real forecast labels are needed for this smoke.
