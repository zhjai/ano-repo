# Source release

The release was assembled from the existing CastFactory electricity-load implementation on 2026-10-26. File/module names are retained to make implementation provenance reviewable. A private file-hash manifest and verification logs accompany the author's working copy; source-host paths and private workflow histories are not distributed.

Portable adaptations:

- Package initializers expose the included modules without eagerly importing unrelated framework components.
- Historical absolute tokenizer paths use `ELECLOAD_MODELS_DIR` (default `models/`).
- Private display/provenance asset locators use local relative placeholders; no audit records or private source documents are shipped.
- The ordinary HF SFT path is retained. Unsupported private corpus repair and cluster receipt imports fail explicitly; training arithmetic is unchanged.
- New thin CLI wrappers expose native inference and separate-label evaluation. The synthetic scripted demo is clearly labeled.
- Tests covering unavailable cluster selection/dispatch integrations are outside the exported suite. Core runtime, serialization, geometry, leakage, metrics, token masking and episode-loss tests are retained.

Excluded: credentials, personal host paths, task histories, operational logs, data, model weights, checkpoints, and unrelated STaR/VERL contents. The Git history is preserved. No new license grant for upstream or absent assets is asserted.

Verification distinguishes standard-library contract tests, optional CPU model tests, and actual large-model/GPU experiments. Passing unit tests and scripted workflow demos is not a reproduction of the paper's measurements.

The subsequent DIRECT correction changes the released runtime to return after whole-horizon T2, without T3/T4. It is a separate change from the source export. Existing paper measurements were not rerun or replaced.

Initial publication verification: 115 tests passed in the reference Python/PyTorch environment with GPU visibility disabled, including actual local tokenizer and tiny-model CPU tests (zero skips). The dependency-free environment passed the same suite with 11 optional ML/tokenizer checks skipped. The built wheel loaded all 20 prompt resources; synthetic workflow and separate-label evaluation CLI checks passed. No large-model GPU training or paper-result reproduction was run.

## Training component continuation (0.3.0)

The source Reasoning HF backend and load reward, differentiable paired/PPO credit kernel, and native FSDP SFT loss hook are now included. Reward parser helpers were extracted unchanged as a dependency closure; the backend imports that portable parser module. New wrappers provide strict Reasoning SFT input validation and compact inference asset configuration. Source loss mathematics and objective-version selection are retained.

An actual CPU optimizer/save/reload smoke uncovered a Transformers EOS serialization mismatch (`2` versus `[2]`) in the Agent checkpoint receipt. The reader now accepts exactly those equivalent single-token representations and rejects different/multiple IDs or invalid types. The successful smoke performs real parameter updates in both SFT backends using only random tiny models and synthetic inputs. Full cluster training and paper-result reproduction remain outside the verified scope.

Continuation verification: 182 tests passed with GPU visibility disabled in the reference environment, with zero skips; the local environment without ML dependencies passed with 43 optional checks skipped. The two real CPU SFT smoke runs each completed one optimizer step and changed parameters, followed by successful HF reloads. The reference environment emitted a nonfatal `multiprocess.resource_tracker` cleanup warning at interpreter shutdown; the commands returned success and their reports/checkpoints were written.
