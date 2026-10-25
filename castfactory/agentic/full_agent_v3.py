"""Shared-policy full-role forecasting, independent of controller-only v2.

Actor states never contain evaluator labels. Teacher and student have separate
wire formats; SFT, student serving and fresh VERL share student serialization.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import re
from typing import Any, Callable, Mapping

from castfactory.agentic.canonical_requests_v1 import _t0_schema
from castfactory.agentic.t0 import parse_t0_workspace
from castfactory.agentic.diagnostic_refinement import (
    PROFILE_VERSION as DIAGNOSTIC_VERSION,
    bind_feedback, response_schema as diagnostic_schema, validate_action as validate_diagnostic_action,
)


VERSION = "full-agent-v3-min-pilot-v0"
PROFILE_VERSIONS = (VERSION, DIAGNOSTIC_VERSION)
ROLES = ("T0", "T1", "T2", "T3", "T4")
FOCUS = ("level_bias_up", "level_bias_down", "peak_timing", "ramp_shape", "event_effect_missing",
         "event_effect_overstated", "uncertainty_high")
VISIBLE_ENVELOPE = re.compile(r"\s*<think_summary>(.*?)</think_summary>\s*<payload>(.*?)</payload>\s*<metadata>(.*?)</metadata>\s*", re.S)
STUDENT_ENVELOPE = re.compile(r"\s*<think>(.*?)</think>\s*<payload>(.*?)</payload>\s*<metadata>(.*?)</metadata>\s*", re.S)
STRUCTURAL_REPAIR_PROTOCOL = "full-role-structural-count-feedback-v2"
FALLBACK_PROTOCOL = "last-valid-complete-forecast-v1"


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


@dataclass(frozen=True)
class Profile:
    version: str = VERSION
    horizon: int = 96
    frequency_hours: float = 1.0
    max_segments: int = 4
    min_segment_hours: float = 12.0
    max_refinements: int = 2
    max_segment_refinements: int = 2
    max_seam_hours: float = 8.0
    call_schema_retries: int = 1
    episode_schema_retries: int = 3
    max_response_tokens: int = 4096
    max_context_tokens: int = 32768
    cost_weight: float = 0.01

    def __post_init__(self):
        if self.version not in PROFILE_VERSIONS:
            raise ValueError("Unknown full-agent protocol version")
        if self.horizon != 96 or self.frequency_hours != 1:
            raise ValueError("This versioned pilot is hourly H=96")
        if self.max_segments < 1 or self.max_refinements < 0 or self.max_segment_refinements < 0:
            raise ValueError("Invalid compute budget")
        if self.max_response_tokens >= self.max_context_tokens:
            raise ValueError("Response budget must leave prompt space")


def reject_private(value: Any) -> None:
    forbidden = {"label", "target", "future_target", "future_values", "ground_truth", "actual_values",
                 "reward", "branch_utility", "oracle_score", "error_to_truth", "effect_mask", "event_spec"}
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = str(key).lower().replace("-", "_")
            if normalized in forbidden or any(x in normalized for x in ("ground_truth", "oracle_score", "target_future_label")):
                raise ValueError("Evaluator/private field in actor state: " + str(key))
            reject_private(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            reject_private(child)


def add_profile_argument(parser):
    parser.add_argument("--agent-profile-version", choices=PROFILE_VERSIONS, default=VERSION,
                        help="Explicit workflow protocol; diagnostic refinement requires fresh teacher trajectories")


def profile_from_args(args):
    return Profile(version=getattr(args, "agent_profile_version", VERSION))


def require_sft_profile(report, profile):
    """New policy inputs require genuinely new SFT, not a renamed old report."""
    recorded = report.get("identity", {}).get("agent_profile")
    if (profile.version == DIAGNOSTIC_VERSION or recorded is not None) and recorded != asdict(profile):
        raise ValueError("SFT/workflow profile mismatch; diagnostic refinement requires a matching fresh SFT release")


def public_case(case) -> dict[str, Any]:
    return {"sample_id": case.observation.sample_id, "domain": case.domain,
            "issue_time": case.observation.issue_time, "horizon": 96,
            "public_prompt": case.public_prompt, "evidence_manifest": list(case.evidence_manifest),
            "covariate_vintage": "offline_aligned_not_operational",
            "context_origin": "synthetic_scenario_assumed_at_issue"}


def _keys(payload, names):
    if not isinstance(payload, dict) or set(payload) != set(names):
        raise ValueError("Payload keys must be exactly " + ",".join(names))


def schema(role: str, state: dict, profile: Profile) -> dict:
    if role == "T0":
        return _t0_schema(profile.horizon)
    if role == "T1":
        return {"mode": "DIRECT|FACTORIZE", "spans": [{"segment_id": "S0", "start": 0, "stop": 96}]}
    if role == "T2":
        s = state["segment"]
        return {"segment_id": s["segment_id"], "start": s["start"], "stop": s["stop"],
                "values": "exactly stop-start finite numeric values in dataset units",
                "uncertainty": "brief uncertainty assessment", "self_check": "brief evidence-grounded check"}
    if role == "T3":
        if profile.version == DIAGNOSTIC_VERSION:
            return diagnostic_schema()
        return {"action": "STOP|REFINE", "segment_id": "S0 for REFINE; null for STOP",
                "focus": "one of " + "|".join(FOCUS) + "; empty string for STOP"}
    if role == "T4":
        return {"action": "KEEP|REGENERATE", "source_hash": state["source_hash"],
                "boundary_id": "B0 for REGENERATE; null for KEEP", "start": "integer for REGENERATE; null for KEEP",
                "values": "absolute replacements crossing one authorized boundary; empty list for KEEP"}
    raise ValueError("Unknown role")


def messages_for(role: str, state: dict, profile: Profile, *, response_format="student") -> list[dict[str, str]]:
    if response_format not in {"teacher", "student"}:
        raise ValueError("Unknown role response format")
    reject_private(state)
    if profile.version == DIAGNOSTIC_VERSION and role == "T2" and state.get("mode") == "REFINE":
        validate_refinement_state(state, profile)
    descriptions = {
        "T0": "Interpret relevance, mechanism, influence windows, interactions and uncertainty. Do not choose actions or forecast numbers.",
        "T1": "Choose DIRECT for a single regime or FACTORIZE at justified predictive-regime transitions. Never split mechanically per event.",
        "T2": "Forecast ONLY the assigned half-open interval. Use domain-specific dynamics, calendar and supplied covariates/context. On REFINE inspect your own preceding draft and local critique. You do not decide REFINE/STOP and cannot see sibling forecasts.",
        "T3": "You alone allocate compute: STOP or REFINE one eligible segment. Compare potential benefit across segments against remaining budget. Confidence is uncalibrated metadata. The focus is exactly one registered category, never free text, numerical prescriptions or sibling prediction contents. Put public explanation in " + ("think_summary" if response_format == "teacher" else "the explanation block") + " only.",
        "T4": "Review concatenated predictions and typed T2/T3 reports. KEEP when appropriate, or REGENERATE one short contiguous seam span crossing an authorized boundary. Use reasons and uncertainty; a true event jump must not be smoothed by default. Never edit the whole horizon. DIRECT must KEEP.",
    }
    if profile.version == DIAGNOSTIC_VERSION:
        descriptions["T3"] = (
            "You alone allocate compute: STOP or REFINE one eligible segment. "
            "Use directive with exactly concern, review_span, driver_refs, own_report_field. "
            "STOP requires null segment_id and null directive. All REFINE fields are required; "
            "review_span and own_report_field may be null. Use only existing drivers visible in the selected segment; "
            "each referenced driver must overlap a non-null review_span. event_coverage requires at least one "
            "driver; event_interaction exactly two distinct drivers. mechanism_consistency requires the selected "
            "draft's summary or self_check. Other concerns allow empty refs. Return no free instruction, "
            "numeric prescription or expected direction. A concern is a hypothesis, not a verified error. "
            "Confidence is uncalibrated. Choose expected refinement benefit against remaining budget.")
        descriptions["T2"] += (
            " On REFINE use the typed directive and source-bound diagnostic feedback. "
            "The review span focuses attention, not permitted edit support: return the whole assigned segment. "
            "An unchanged forecast is valid; never obey an unverified suspicion as a numerical target.")
    explanation_tag = "think_summary" if response_format == "teacher" else "think"
    system = (
        "You are role " + role + " in a shared-policy energy forecasting agent. " + descriptions[role] +
        " Use only supplied public information. Future covariates are offline benchmark inputs, not verified operational vintages; events are synthetic scenario assumptions. "
        "Indices are zero-based horizon-relative half-open [start,stop). Return exactly "
        f"<{explanation_tag}>Brief public explanation (2-5 sentences), not private reasoning.</{explanation_tag}>"
        "<payload>One JSON object matching response_schema</payload>"
        "<metadata>{\"confidence\":\"low|medium|high\"}</metadata>. "
        "No fences or extra text. Confidence value is recorded but excluded from training loss. "
        "Do not expose or quote private reasoning. All required fields must be present."
    )
    if role == "T0":
        system += " Copy evidence_id, kind and as_of exactly from evidence_manifest; known_future calendar columns are registered covariate. evidence_ids_used must match the referenced evidence IDs."
    # Runtime ownership is validated above, but is not predictive evidence.
    # In particular random serving episode IDs must not change actor prompts.
    wire_state = ({k: v for k, v in state.items() if k != "refinement_binding"}
                  if profile.version == DIAGNOSTIC_VERSION and role == "T2"
                  and state.get("mode") == "REFINE" else state)
    body = {"version": profile.version, "role": role, "state": wire_state,
            "profile": asdict(profile), "response_schema": schema(role, state, profile)}
    return [{"role": "system", "content": system}, {"role": "user", "content": canonical(body)}]


def parse_visible(text: str, *, response_format=None) -> dict:
    if response_format not in {None, "teacher", "student"}:
        raise ValueError("Unknown role response format")
    patterns = {"teacher": VISIBLE_ENVELOPE, "student": STUDENT_ENVELOPE}
    match = (patterns[response_format].fullmatch(text) if response_format else
             (STUDENT_ENVELOPE.fullmatch(text) or VISIBLE_ENVELOPE.fullmatch(text)))
    if not match:
        raise ValueError("Full-role visible envelope mismatch")
    summary, payload, metadata = match.groups()
    if not 1 <= len(summary.strip()) <= 4000 or "<" in summary:
        raise ValueError("Invalid public summary")
    def object_only(raw):
        def unique(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError("Duplicate JSON key")
                result[key] = value
            return result
        return json.loads(raw, object_pairs_hook=unique, parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Nonfinite JSON")))
    payload, metadata = object_only(payload), object_only(metadata)
    _keys(metadata, ("confidence",))
    if metadata["confidence"] not in {"low", "medium", "high"}:
        raise ValueError("Confidence must be low, medium or high")
    return {"summary": summary.strip(), "payload": payload, "metadata": metadata}


def structural_feedback(role, state, content, error, attempt):
    """Same truth-free repair for serving, reference generation and fresh RL."""
    repair = f"Structural validation failed: {error}. Retry {attempt}. Return the complete required envelope."
    if role == "T0":
        try:
            payload = parse_visible(content)["payload"]
            referenced = sorted({e["evidence_id"] for d in payload.get("drivers", []) for e in d.get("evidence", [])})
            repair += " evidence_ids_used must exactly match the payload evidence arrays: " + canonical(referenced) + "."
        except (ValueError, KeyError, TypeError):
            repair += " evidence_ids_used must exactly match the IDs referenced in the payload evidence arrays."
    elif role == "T2":
        expected = state["segment"]["stop"] - state["segment"]["start"]
        try:
            values = parse_visible(content)["payload"].get("values")
            if isinstance(values, list):
                actual = len(values)
                repair += f" Your previous payload.values contains {actual} items; the assigned interval requires {expected}."
                if actual < expected:
                    repair += f" It is short by {expected - actual} items."
                elif actual > expected:
                    repair += f" It has {actual - expected} excess items."
                repair += " Rewrite the complete forecast for the assigned interval and check its count before returning."
        except (ValueError, KeyError, TypeError, AttributeError):
            # Do not infer a count from truncated JSON or invent forecast values.
            pass
        repair += (f" payload.values must contain exactly {expected} finite numeric values in horizon order. "
                   "Close the values array after the final assigned point, complete all other required payload fields, "
                   "then close payload and metadata. Do not add extra forecast points or omit required fields.")
    return repair


def validate(role: str, response: dict, state: dict, profile: Profile) -> None:
    p = response["payload"]
    reject_private(p)
    if role == "T0":
        parse_t0_workspace(p, horizon=96, evidence_manifest=state["public"]["evidence_manifest"],
                           issue_time=state["public"]["issue_time"])
    elif role == "T1":
        _keys(p, ("mode", "spans"))
        spans = p["spans"]
        if p["mode"] not in {"DIRECT", "FACTORIZE"} or not isinstance(spans, list):
            raise ValueError("Invalid factorization plan")
        if not 1 <= len(spans) <= profile.max_segments or (len(spans) == 1) != (p["mode"] == "DIRECT"):
            raise ValueError("Mode/segment count mismatch")
        end = 0
        for i, span in enumerate(spans):
            _keys(span, ("segment_id", "start", "stop"))
            if span["segment_id"] != "S" + str(i) or type(span["start"]) is not int or type(span["stop"]) is not int:
                raise ValueError("Invalid span ID/indices")
            minimum = math.ceil(profile.min_segment_hours / profile.frequency_hours)
            if span["start"] != end or span["stop"] - end < minimum:
                raise ValueError(
                    "Span gap/overlap/minimum length violation: "
                    f"spans[{i}] is [{span['start']},{span['stop']}); "
                    f"expected start={end}, minimum length={minimum}, "
                    f"actual length={span['stop'] - span['start']}")
            end = span["stop"]
        if end != profile.horizon:
            raise ValueError("Spans must cover horizon")
    elif role == "T2":
        _keys(p, ("segment_id", "start", "stop", "values", "uncertainty", "self_check"))
        s = state["segment"]
        if any(p[key] != s[key] for key in ("segment_id", "start", "stop")):
            raise ValueError("T2 assigned interval mismatch")
        check_values(p["values"], s["stop"] - s["start"], state["public"]["domain"])
        if any(not isinstance(p[k], str) or not p[k].strip() for k in ("uncertainty", "self_check")):
            raise ValueError("T2 report missing")
    elif role == "T3":
        if profile.version == DIAGNOSTIC_VERSION:
            validate_diagnostic_action(p, state, local_workspace)
            return
        _keys(p, ("action", "segment_id", "focus"))
        if p["action"] not in {"STOP", "REFINE"} or not isinstance(p["focus"], str):
            raise ValueError("Invalid scheduler action")
        if p["action"] == "STOP" and (p["segment_id"] is not None or p["focus"]):
            raise ValueError("STOP must have null segment and empty focus")
        if p["action"] == "REFINE" and (p["segment_id"] not in state["eligible_segments"] or p["focus"] not in FOCUS):
            raise ValueError("REFINE must select an eligible segment and provide a focus")
    elif role == "T4":
        _keys(p, ("action", "source_hash", "boundary_id", "start", "values"))
        # Legacy wire metadata remains required, but never gates a valid seam edit.
        if p["action"] == "KEEP":
            if p["boundary_id"] is not None or p["start"] is not None or p["values"] != []:
                raise ValueError("KEEP cannot contain edits")
        elif p["action"] == "REGENERATE":
            boundaries = {b["boundary_id"]: b for b in state["boundaries"]}
            b = boundaries.get(p["boundary_id"])
            if not b or type(p["start"]) is not int or not isinstance(p["values"], list):
                raise ValueError("Unrecognized seam/span")
            start, end = p["start"], p["start"] + len(p["values"])
            width = math.ceil(profile.max_seam_hours / profile.frequency_hours)
            if not b["allowed_start"] <= start < b["index"] < end <= b["allowed_stop"] or end - start > width:
                raise ValueError("T4 span must cross one authorized seam within budget")
            check_values(p["values"], end - start, state["public"]["domain"])
        else:
            raise ValueError("Invalid T4 action")


def check_values(values, count, domain):
    if not isinstance(values, list) or len(values) != count:
        raise ValueError("Forecast length mismatch")
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in values):
        raise ValueError("Forecast contains nonfinite/nonnumeric values")
    if domain in {"load", "wind", "solar"} and any(v < 0 for v in values):
        raise ValueError("Negative consumption/generation")


def local_workspace(workspace, span):
    drivers = []
    for driver in workspace["drivers"]:
        windows = [w for w in driver["impact_windows"] if w["start"] < span["stop"] and w["end"] > span["start"]]
        if windows:
            item = dict(driver)
            item["impact_windows"] = windows
            drivers.append(item)
    retained = {d["id"] for d in drivers}
    for driver in drivers:
        driver["interactions"] = [i for i in driver["interactions"] if i["with_driver_id"] in retained]
    return {"schema_version": workspace["schema_version"], "drivers": drivers,
            "global_uncertainty": workspace["global_uncertainty"],
            "evidence_ids_used": sorted({e["evidence_id"] for d in drivers for e in d["evidence"]})}


def refinement_state(public, workspace, span, own, decision, profile, *, episode_id):
    """Project a T3 action into only the selected segment's existing evidence."""
    state = {"public": public, "workspace": local_workspace(workspace, span),
             "segment": span, "mode": "REFINE", "own_history": own}
    if profile.version != DIAGNOSTIC_VERSION:
        state["critique"] = decision["payload"]["focus"]
        return state
    if not own:
        raise ValueError("Diagnostic refinement requires an actual preceding draft")
    bound = bind_feedback(decision["payload"], segment=span, workspace=state["workspace"],
                          draft=own[-1]["draft"], public=public, sample_id=public["sample_id"],
                          depth=own[-1]["depth"], protocol_version=profile.version)
    # The runtime owns this complete snapshot; the actor never supplies binding metadata.
    bound.pop("current_draft")
    state.update(directive=bound.pop("directive"), critique=bound.pop("feedback"),
                 refinement_binding={**bound, "episode_id": episode_id})
    return json.loads(canonical(state))


def validate_refinement_state(state, profile):
    """Also checked when rebuilding SFT/serving messages, not just at execution."""
    own = state.get("own_history")
    sid = state["segment"]["segment_id"]
    if not own or any(h["segment_id"] != sid or type(h["depth"]) is not int or h["depth"] != i
                      for i, h in enumerate(own)):
        raise ValueError("Invalid diagnostic segment history/depth")
    binding = state.get("refinement_binding", {})
    if not isinstance(binding.get("episode_id"), str) or not binding["episode_id"]:
        raise ValueError("Missing runtime diagnostic episode binding")
    expected = refinement_state(state["public"], state["workspace"], state["segment"], own,
                                {"payload": {"action": "REFINE", "segment_id": sid,
                                             "directive": state.get("directive")}}, profile,
                                episode_id=binding["episode_id"])
    if state != expected:
        raise ValueError("Stale or modified diagnostic feedback snapshot")


class InfrastructureFailure(RuntimeError):
    """Transport/device failure: never convert it into model critique."""


class ModelContextOverflow(ValueError):
    """An actor-generated state cannot fit; no inference call was performed."""


def execute(case, caller: Callable, profile: Profile | None = None, *, response_format="student", episode_id=None) -> dict:
    """Run actual role calls; caller receives truth-free (role, state, messages)."""
    profile = profile or Profile()
    public = public_case(case)
    turns, failures = [], []
    schema_retries = 0
    versions = []
    # Commit a recovery snapshot only after a complete forecast passes validation.
    # Partial initial drafts must never masquerade as a full-horizon forecast.
    last_valid_concat = None
    def call(role, state):
        nonlocal schema_retries
        state = json.loads(canonical(state))
        messages = messages_for(role, state, profile, response_format=response_format)
        last = None
        for attempt in range(profile.call_schema_retries + 1):
            record = {"role": role, "turn": len(turns), "state": state, "state_sha256": digest(state),
                      "messages": json.loads(canonical(messages)), "attempt": attempt, "valid": False}
            try:
                try:
                    raw = caller(role, state, messages)
                except ModelContextOverflow:
                    record.update(model_contract_failure=True, prefill_tokens=0, decode_tokens=0)
                    raise
                except Exception as exc:
                    record["infrastructure_failure"] = True
                    record["error"] = type(exc).__name__ + ": " + str(exc)[:1000]
                    turns.append(record)
                    raise InfrastructureFailure(record["error"]) from exc
                record.update(raw)
                if raw.get("finish_reason", "stop") != "stop":
                    raise ValueError(raw.get("termination_reason", "Generation must terminate normally, not at token budget"))
                response = parse_visible(raw["content"], response_format=response_format)
                validate(role, response, state, profile)
                record.update({"response": response, "valid": True})
                turns.append(record)
                return response
            except InfrastructureFailure:
                raise
            except Exception as exc:
                record["error"] = type(exc).__name__ + ": " + str(exc)[:1000]
                turns.append(record)
                failures.append({"role": role, "turn": record["turn"], "error": record["error"]})
                last = exc
                if isinstance(exc, ModelContextOverflow):
                    break
                if attempt >= profile.call_schema_retries or schema_retries >= profile.episode_schema_retries:
                    break
                schema_retries += 1
                # Feedback is structural and contains no evaluator/truth information.
                messages = json.loads(canonical(messages))
                messages.append({"role": "assistant", "content": record.get("content", "")})
                messages.append({"role": "user", "content": structural_feedback(
                    role, state, record.get("content", ""), record["error"], attempt + 1)})
        raise ValueError(role + " failed contract") from last

    result = {"version": profile.version, "profile": asdict(profile), "origin": case.manifest(),
              "fallback_protocol": FALLBACK_PROTOCOL,
              "episode_id": episode_id or digest({"sample": public["sample_id"], "version": profile.version,
                                                  "fallback_protocol": FALLBACK_PROTOCOL}),
              "turns": turns, "failures": failures, "complete": False}
    try:
        t0 = call("T0", {"public": public})
        plan = call("T1", {"public": public, "workspace": t0["payload"]})["payload"]
        drafts, depth = {}, {s["segment_id"]: 0 for s in plan["spans"]}
        for span in plan["spans"]:
            draft = call("T2", {"public": public, "workspace": local_workspace(t0["payload"], span),
                                "segment": span, "mode": "INITIAL", "own_history": [], "critique": ""})
            drafts[span["segment_id"]] = draft
            versions.append({"segment_id": span["segment_id"], "depth": 0, "draft": draft})
        initial_concat = [v for s in plan["spans"] for v in drafts[s["segment_id"]]["payload"]["values"]]
        check_values(initial_concat, profile.horizon, case.domain)
        last_valid_concat = list(initial_concat)
        result.update(plan=plan, initial_concat=list(initial_concat), depth=dict(depth))
        scheduler_history, used = [], 0
        termination = "POLICY_STOP"
        while True:
            eligible = [key for key, d in depth.items() if d < profile.max_segment_refinements]
            if used >= profile.max_refinements or not eligible:
                termination = "FORCED_BUDGET_STOP"
                break
            state = {"public": public, "workspace": t0["payload"], "plan": plan, "reports": drafts,
                     "depth": dict(depth), "remaining_refinements": profile.max_refinements - used,
                     "eligible_segments": eligible, "own_history": scheduler_history}
            decision = call("T3", state)
            scheduler_history.append(decision)
            if decision["payload"]["action"] == "STOP":
                break
            sid = decision["payload"]["segment_id"]
            span = next(s for s in plan["spans"] if s["segment_id"] == sid)
            own = [v for v in versions if v["segment_id"] == sid]
            drafts[sid] = call("T2", refinement_state(public, t0["payload"], span, own, decision, profile,
                                                      episode_id=result["episode_id"]))
            depth[sid] += 1
            used += 1
            versions.append({"segment_id": sid, "depth": depth[sid], "draft": drafts[sid]})
            candidate = [v for s in plan["spans"] for v in drafts[s["segment_id"]]["payload"]["values"]]
            check_values(candidate, profile.horizon, case.domain)
            last_valid_concat = list(candidate)
            result["depth"] = dict(depth)
        concat = [v for s in plan["spans"] for v in drafts[s["segment_id"]]["payload"]["values"]]
        boundaries = []
        half = int(profile.max_seam_hours / profile.frequency_hours) // 2
        for i, (left, right) in enumerate(zip(plan["spans"], plan["spans"][1:])):
            index = left["stop"]
            boundaries.append({"boundary_id": "B" + str(i), "index": index,
                               "allowed_start": max(left["start"], index - half),
                               "allowed_stop": min(right["stop"], index + half)})
        t4state = {"public": public, "workspace": t0["payload"], "plan": plan, "reports": drafts,
                   "scheduler_history": scheduler_history, "termination": termination,
                   "depth": depth, "concat": concat, "source_hash": digest({"concat": concat, "depth": depth}),
                   "boundaries": boundaries}
        final = list(concat)
        fallback = False
        if plan["mode"] == "FACTORIZE":
            try:
                t4 = call("T4", t4state)
                if t4["payload"]["action"] == "REGENERATE":
                    patch = t4["payload"]
                    final[patch["start"]:patch["start"] + len(patch["values"])] = patch["values"]
            except ValueError:
                fallback = True
        check_values(final, 96, case.domain)
        result.update({"complete": True, "sft_eligible": not fallback, "plan": plan, "versions": versions,
                       "initial_concat": initial_concat, "pre_t4_concat": concat, "forecast": final,
                       "depth": depth, "termination": termination, "t4_fallback": fallback})
        result["t4_program_keep"] = plan["mode"] == "DIRECT"
    except Exception as exc:
        # Recovery preserves the latest legal forecast, not the best by future
        # truth. It ends the failed episode without inventing STOP/T4 tokens.
        retained = last_valid_concat is not None
        result.update({"error": str(exc), "sft_eligible": False,
                       "forecast": list(last_valid_concat) if retained else case.observation.target_history[-24:].tolist() * 4,
                       "fallback": "last_valid_concat" if retained else "seasonal_last_day",
                       "termination": "INFRASTRUCTURE_FAILURE" if isinstance(exc, InfrastructureFailure) else "MODEL_CONTRACT_FAILURE",
                       "versions": versions})
        if not retained:
            # Legacy name retained for readers; 24 *samples*, only a day at 1h.
            result["fallback_history_period_steps"] = 24
        if isinstance(exc, InfrastructureFailure):
            result["infrastructure_failure"] = True
    return result


def diagnostic_request_messages(role, state, profile, retries, *, response_format):
    """Rebuild only actual structural retries, including teacher wire format."""
    if not isinstance(retries, list) or len(retries) > profile.call_schema_retries:
        raise ValueError("Invalid diagnostic structural retry count")
    messages = messages_for(role, state, profile, response_format=response_format)
    for attempt, retry in enumerate(retries, 1):
        _keys(retry, ("content", "finish_reason", "termination_reason", "error"))
        try:
            if retry["finish_reason"] != "stop":
                raise ValueError(retry["termination_reason"])
            validate(role, parse_visible(retry["content"], response_format=response_format), state, profile)
        except Exception as exc:
            # Match execute.call's structural-failure path, including numeric
            # overflow from a malformed forecast that was successfully retried.
            error = type(exc).__name__ + ": " + str(exc)[:1000]
        else:
            raise ValueError("Diagnostic retry follows an already valid response")
        if retry["error"] != error:
            raise ValueError("Diagnostic retry error differs from actual response failure")
        messages.extend([{"role": "assistant", "content": retry["content"]},
                         {"role": "user", "content": structural_feedback(
                             role, state, retry["content"], error, attempt)}])
    return messages


def sft_rows(episode: dict, tokenizer) -> list[dict]:
    from castfactory.training.full_agent_serialization import (
        student_content, student_messages, student_training_tokens, serialization_identity)
    if not episode.get("complete") or not episode.get("sft_eligible"):
        return []
    profile = Profile(**episode["profile"])
    if episode.get("version") != profile.version:
        raise ValueError("Episode/profile protocol mismatch")
    rows = []
    for index, turn in enumerate(episode["turns"]):
        if not turn["valid"]:
            continue
        diagnostic_metadata = {}
        if profile.version == DIAGNOSTIC_VERSION:
            # Do not trust legacy valid=True flags after an archive is relabelled.
            if (turn["role"] == "T2" and turn["state"].get("mode") == "REFINE"
                    and turn["state"].get("refinement_binding", {}).get("episode_id") != episode["episode_id"]):
                raise ValueError("Diagnostic SFT feedback belongs to another episode")
            validate(turn["role"], parse_visible(turn["content"]), turn["state"], profile)
            response_format = "teacher" if turn["content"].lstrip().startswith("<think_summary>") else "student"
            attempt = turn.get("attempt")
            if type(attempt) is not int or not 0 <= attempt <= min(index, profile.call_schema_retries):
                raise ValueError("Invalid diagnostic archived retry count")
            retries = []
            for retry_index, previous in enumerate(episode["turns"][index - attempt:index]):
                if (previous["valid"] or previous["role"] != turn["role"]
                        or previous["state"] != turn["state"] or previous["attempt"] != retry_index):
                    raise ValueError("Diagnostic retry belongs to another role/state")
                if previous["messages"] != diagnostic_request_messages(
                        turn["role"], turn["state"], profile, retries, response_format=response_format):
                    raise ValueError("Diagnostic archived retry request mismatch")
                retries.append({"content": previous.get("content", ""),
                                "finish_reason": previous.get("finish_reason", "stop"),
                                "termination_reason": previous.get("termination_reason",
                                    "Generation must terminate normally, not at token budget"),
                                "error": previous["error"]})
            expected = diagnostic_request_messages(turn["role"], turn["state"], profile, retries,
                                                   response_format=response_format)
            if turn["messages"] != expected:
                raise ValueError("Diagnostic SFT requires actual same-protocol teacher requests")
            diagnostic_metadata = {"diagnostic_response_format": response_format,
                                   "diagnostic_retry_trace": retries}
            if turn["role"] == "T2" and turn["state"].get("mode") == "REFINE":
                diagnostic_metadata["diagnostic_refinement_binding"] = dict(turn["state"]["refinement_binding"])
        content = student_content(turn["content"])
        messages = student_messages(turn["role"], turn["state"], profile, turn["messages"])
        prefix, suffix = student_training_tokens(tokenizer, messages, content)
        continuation = tokenizer.decode(suffix, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        supervised = confidence_mask(continuation, suffix, tokenizer)
        if len(prefix) + len(suffix) > episode["profile"]["max_context_tokens"]:
            raise ValueError("SFT context exceeds budget; truncation forbidden")
        rows.append({**diagnostic_metadata, "version": profile.version, "role": turn["role"], "row_id": episode["episode_id"] + ":" + str(turn["turn"]),
                     "episode_id": episode["episode_id"], "split": "train", "domain": episode["origin"]["domain"],
                     "state_sha256": turn["state_sha256"], "messages": messages, "output": content,
                     "teacher_response_sha256": digest(turn["content"]),
                     "serialization": serialization_identity(tokenizer),
                     "input_ids_sha256": digest(prefix), "response_sha256": digest(content),
                     "response_ids_sha256": digest(suffix),
                     "prompt_tokens": len(prefix), "response_tokens": len(suffix), "response_loss_mask": supervised,
                     "confidence_provenance": {"author": "teacher", "value": turn["response"]["metadata"]["confidence"]},
                     "target_source": "teacher_demonstration"})
    return rows


def confidence_value_span(content: str):
    """Locate the raw JSON string value, including legal escaped spellings."""
    blocks = list(re.finditer(r"<metadata>(.*?)</metadata>", content, re.S))
    if not blocks:
        return None
    block = blocks[-1]
    raw = block.group(1)
    try:
        metadata = json.loads(raw)
    except ValueError:
        return None
    if not isinstance(metadata, dict) or set(metadata) != {"confidence"}:
        return None
    if metadata["confidence"] not in {"low", "medium", "high"}:
        return None
    string_pair = r'("(?:\\.|[^"\\])*")\s*:\s*("(?:\\.|[^"\\])*")'
    for pair in re.finditer(string_pair, raw):
        if json.loads(pair.group(1)) == "confidence":
            start, stop = pair.span(2)
            return block.start(1) + start + 1, block.start(1) + stop - 1
    raise ValueError("Valid confidence metadata has no identifiable value")


def confidence_mask(content: str, response_ids: list[int], tokenizer) -> list[int]:
    """Mask only tokens overlapping the confidence value, retaining envelope/EOS."""
    mask = [1] * len(response_ids)
    span = confidence_value_span(content)
    if span is None:
        # Invalid fresh sampled output still has policy loss; no confidence to mask.
        return mask
    encoded = tokenizer(content, add_special_tokens=False, return_offsets_mapping=True)
    content_ids = list(encoded["input_ids"])
    if list(response_ids[:len(content_ids)]) != content_ids:
        raise ValueError("Response tokenization differs from its recorded visible content")
    start, stop = span
    for index, (a, b) in enumerate(encoded["offset_mapping"]):
        if a < stop and b > start:
            mask[index] = 0
    if not any(v == 0 for v in mask):
        raise ValueError("Confidence value has no excluded token")
    return mask
