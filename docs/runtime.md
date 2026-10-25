# Runtime

`execute_public(case, caller, profile=...)` is the public inference API. The caller receives `(role, messages)` and returns either a complete response string or a receipt containing `content` and real backend metadata. Public execution does not invoke a label-informed reviewer.

T1 returns event/driver analysis, a DIRECT or FACTORIZE decision, and a contiguous partition. Each initial T2 call receives the original public evidence and its assigned segment without sibling predictions. T3 can STOP or issue typed feedback for a selected segment. A successful T2 revision is reassessed by T3, including the final allowed revision. FACTORIZE always reaches T4 on normal completion; T4 keeps or applies bounded edits without moving segment boundaries.

In the initial exported source, DIRECT runs T1, whole-horizon T2, and T3 reflection, and skips T4. `direct_cost_reference` instead forces DIRECT and terminates after T2. The source's teacher constructor `execute` uses train-only privileged review; use `execute_public` for inference.

Invalid or exhausted model calls are recorded. If complete segment forecasts exist, fallback uses the latest complete snapshot; otherwise it repeats the last 24 public history values. A complete numerical forecast is not the same as normal workflow completion. `normal_completion`, `fallback`, `failures`, and `termination` must remain available to evaluation.

Profiles are explicit. `merged-t1-answer-v1` uses the natural-language templates and JSON answer payloads; compact and JSON-row profiles require their matching public JSON evidence and response serialization. Do not mix a model trained for one contract with another by silently changing the parser.
