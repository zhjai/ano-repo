# Data interface

`examples/public_case.json` is a synthetic, label-free example, not a benchmark record.

A prepared public case contains:

- `case_id`, `source`, and `split` identifying one forecast origin.
- `public_case`: the rendered historical load and available covariate/event evidence.
- `geometry`: integer H, H exact future timestamps, hourly frequency, and source unit (or explicit null).
- `evidence_manifest`: available evidence identifiers, kind, availability basis, and source location.
- `history_values`: finite numerical historical target values used for fallback.

Compact/JSON-row profiles also need `public_json_case` with `task`, `grid`, `variables`, `history`, `future_known`, `context`, and `events`. The included renderer preserves time grids and rejects future target load in public evidence. Some historical renderers support optional reviewed-text assets; those private assets are not shipped and their absence must not be bypassed.

Keep `labels`, future target values, reward, and private annotations outside the public case. Public runtime validation rejects known private-field names; this structural check does not replace reviewing free text for leakage. Labels are supplied separately to the evaluator and matched by case identity and horizon.

SFT consumes JSONL role rows exported by `full_agent_v4.sft_rows` from completed, reviewed teacher episodes. Rows contain `case_id`/`episode_id`, `role`, `source_call_id`, `messages`, `assistant_response`, `profile_version`, `response_format`, `target_source`, and the actual supervision provenance. T1 must carry complete label-informed review provenance; downstream targets preserve accepted teacher outputs. Do not manufacture review metadata to pass the loader.

No data redistribution rights are implied by this repository. Obtain the source datasets through their respective providers and prepare inputs matching the interfaces above. Dataset construction, selected splits, and final paper predictions are not distributed here.
