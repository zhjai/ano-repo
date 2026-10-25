# Training implementation and scope

The HF SFT backend is exported from the existing `full_agent_sft_v4.py` implementation. It keeps its tokenizer mask construction, complete-episode grouping, exact episode sampling, loss normalization, role-wise graph release, optimizer configuration, checkpoint reload checks, and CLI. Private offline corpus-repair paths fail explicitly rather than impersonating teacher provenance. Cluster FSDP receipt readers and RL BEST-selection replay are omitted; HF training does not invoke those paths.

The native FSDP SFT and current RL orchestration are not shipped. They depend on a modified VERL tree, frozen source-specific populations, provenance/seed handoff files, and job-specific recovery/admission records. Relocating paths alone would not produce a standalone equivalent trainer: its safety gates bind actual case identities, selected checkpoints, and native optimizer recovery. This release does not relax those checks or substitute the older STaR VERL copy. The existing HF backend is exposed explicitly instead.

`full_agent_reward.py` includes versioned reward arithmetic. Select profiles deliberately; the historical default and newer bounded variants are not interchangeable. Neither these utilities nor a successful protocol test establish trained forecasting quality.

The supplied HF command performs genuine optimization when run without `--dry-run` on valid inputs. A GPU training run and reproduction of paper results have not been performed as part of this release. Full-model export verification loads another model copy and therefore requires additional memory. No trained checkpoint or BEST-selection claim is bundled.

Reference source environment: Python 3.12, PyTorch 2.10.0, Transformers 5.3.0.dev0, Accelerate 1.14.0. Public packaging specifies compatible major-version ranges; accelerator wheels must match the user's system. Optional tests requiring model tokenizers skip when `ELECLOAD_MODELS_DIR` does not contain them. Core protocol tests run without ML dependencies.
