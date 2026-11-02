# ElecLoad source release

## Included components

- T1–T4 forecasting runtime, prompt resources, response validation, and fallback handling.
- Native Hugging Face inference and separate-label forecast evaluation.
- Reasoning SFT and complete-episode Agent SFT entrypoints.
- Forecast rewards, differentiable PPO/paired-credit objectives, and an episode SFT loss hook.
- Synthetic workflow examples, CPU training smoke checks, and tests.

DIRECT execution returns after whole-horizon T2 forecasting, without T3 or T4. Segmented execution retains selective refinement and boundary reconciliation.

## Portable adaptations

- Package initializers load included modules without unrelated framework dependencies.
- Local tokenizer assets use `ELECLOAD_MODELS_DIR` (default `models/`).
- Inference and training wrappers validate inputs and use local model assets.
- Reward parser helpers are included with their dependency closure.
- Agent checkpoint verification accepts equivalent single-token EOS representations (`2` and `[2]`) while rejecting mismatched or multiple IDs.

Datasets, trained weights, checkpoints, and full distributed training orchestration are not bundled. The reusable training hooks require their matching backend; see [training details](training.md). This release does not grant additional rights to third-party assets.

## Verification

Version 0.3.0 passed 182 tests in the reference CPU environment with zero skips. In the environment without optional ML dependencies, 43 checks were skipped. Both SFT smoke runs completed an optimizer step, changed model parameters, and successfully reloaded their HF exports. The built package loaded all 20 prompt resources.

The reference environment emitted a nonfatal `multiprocess.resource_tracker` cleanup warning at shutdown after successful tests and saved outputs. These checks validate implementation mechanics; they do not constitute large-model training or reproduction of the paper's results.
