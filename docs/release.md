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
