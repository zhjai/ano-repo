"""Four-role teacher construction and public production execution.

``case`` is a prepared public projection. Only the reviewer closure may own
training labels. Raw calls and label-informed accepted T1 are separate records.
Public execution never runs privileged review. Native thinking/template control
belongs to the selected caller; this module renders and validates its contract.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, fields
from datetime import datetime, timedelta
import json
from typing import Any, Callable, Optional

from castfactory.agentic import distillation_protocol as protocol
from castfactory.agentic.diagnostic_refinement import CONCERNS, INSTRUCTIONS


VERSION = protocol.VERSION
WORKFLOW_REVISION = "elecload-direct-t2-return-v1"
JSON_ROWS_VERSION = "agentic-json-rows-v1"
COMPACT_VERSION = "agentic-compact-markdown-v1"
STEP_TARGET_VERSION = "agentic-compact-markdown-v2"
COMPACT_VERSIONS = (COMPACT_VERSION, STEP_TARGET_VERSION)
ROLES = ("T1", "T2", "T3", "T4")
PUBLIC_POLICY_VERSION = "public-t1-compute-policy-v1"
_PUBLIC_POLICY_FIELDS = (
    "max_segments", "min_segment_length", "max_refinements", "structural_retries",
    "t4_max_patch_steps", "t4_max_total_steps", "horizon_steps",
    "mandatory_post_revision_t3", "protected_post_revision_t3_attempts",
    "protected_t4_attempts", "independent_per_call_retry_budgets",
    "allowed_modes", "reference_protocol",
)


@dataclass(frozen=True)
class Profile:
    max_segments: int = 4
    min_segment_length: int = 6
    max_refinements: int = 2
    structural_retries: int = 1
    t4_max_patch_steps: int = 4
    t4_max_total_steps: int = 8
    max_response_tokens: int = 4096
    max_context_tokens: Optional[int] = None
    message_token_counter: Optional[Callable] = None
    profile_version: str = VERSION

    def __post_init__(self):
        if self.profile_version not in (VERSION, JSON_ROWS_VERSION, *COMPACT_VERSIONS):
            raise ValueError("Unsupported Agent profile: " + str(self.profile_version))
        positive = ("max_segments", "min_segment_length", "t4_max_patch_steps",
                    "t4_max_total_steps", "max_response_tokens")
        nonnegative = ("max_refinements", "structural_retries")
        for name in positive + nonnegative:
            value = getattr(self, name)
            if type(value) is not int or value < (1 if name in positive else 0):
                raise ValueError("Invalid profile setting: " + name)
        if self.max_context_tokens is not None:
            if (type(self.max_context_tokens) is not int
                    or self.max_context_tokens <= self.max_response_tokens
                    or not callable(self.message_token_counter)):
                raise ValueError("Context limit requires an exact message counter and response space")
        if self.message_token_counter is not None and not callable(self.message_token_counter):
            raise ValueError("message_token_counter must be callable")

    def record(self):
        return {f.name: getattr(self, f.name) for f in fields(self)
                if f.name != "message_token_counter" and not
                (f.name == "profile_version" and self.profile_version == VERSION)}


def _keys(obj, names, name):
    if not isinstance(obj, dict) or set(obj) != set(names):
        raise ValueError(name + " must contain exactly " + ", ".join(names))


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(name + " must be nonempty text")


def _span(start, stop, h, name):
    if type(start) is not int or type(stop) is not int or not 0 <= start < stop <= h:
        raise ValueError(name + " must be a zero-based half-open integer interval")


def _json_copy(value):
    """Receipts must be honest JSON artifacts, never repr of hidden objects."""
    return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))


def _reject_private(value):
    forbidden = {"labels", "label", "target", "targets", "canonical_labels", "supervised_label",
                 "future_labels", "future_values", "future_target", "ground_truth", "actual_values",
                 "reward", "oracle_score", "error_to_truth", "effect_mask", "event_spec"}
    if isinstance(value, dict):
        for key, child in value.items():
            if str(key).lower().replace("-", "_") in forbidden:
                raise ValueError("Private field in public case: " + str(key))
            _reject_private(child)
    elif isinstance(value, list):
        for child in value:
            _reject_private(child)


def attach_public_json_case(case: dict, public_json_case: dict) -> dict:
    """Attach an owned public renderer result, without canonical labels.

    The caller constructs this projection from the selected canonical source
    with build_public_json_case; runtime code never imports preparation scripts.
    """
    _reject_private(case)
    _reject_private(public_json_case)
    geometry = case.get("geometry", {})
    if (not isinstance(public_json_case, dict)
            or set(public_json_case) != {"task", "grid", "variables", "history",
                                        "future_known", "context", "events"}
            or public_json_case.get("grid", {}).get("future_timestamps") != geometry.get("future_timestamps")):
        raise ValueError("Public JSON case must preserve the exact forecast timestamp grid")
    for name in ("history", "future_known"):
        table = public_json_case[name]
        if not isinstance(table, dict) or not isinstance(table.get("columns"), list):
            raise ValueError("Public JSON case requires structured evidence tables")
    if "load" in public_json_case["future_known"]["columns"]:
        raise ValueError("Future target load cannot enter public JSON evidence")
    result = deepcopy(case)
    result["public_json_case"] = _json_copy(public_json_case)
    return result


def _validate_case(case, *, train_only=True):
    _reject_private(case)
    for key in ("case_id", "source", "split", "public_case", "geometry",
                "evidence_manifest", "history_values"):
        if key not in case:
            raise ValueError("Missing public case field: " + key)
    if train_only and case["split"] != "train":
        raise ValueError("Teacher distillation is train-only")
    _text(case["split"], "split")
    _text(case["public_case"], "public_case")
    geometry = case["geometry"]
    h = geometry.get("H")
    if type(h) is not int or h < 2:
        raise ValueError("Geometry requires an integer H >= 2")
    stamps = geometry.get("future_timestamps")
    if not isinstance(stamps, list) or len(stamps) != h:
        raise ValueError("future_timestamps must contain exactly H timestamps")
    try:
        parsed = [datetime.fromisoformat(s.replace("Z", "+00:00")) for s in stamps]
        if any(b - a != timedelta(hours=1) for a, b in zip(parsed, parsed[1:])):
            raise ValueError("Current profile supports the actual hourly timestamp grid only")
    except (TypeError, AttributeError) as exc:
        raise ValueError("Invalid forecast timestamp grid") from exc
    if geometry.get("frequency") not in ("hourly", "1h", "h", "H", "1H", "60min", "60T"):
        raise ValueError("Current profile requires explicitly hourly frequency")
    if "unit" not in geometry or (geometry["unit"] is not None and not isinstance(geometry["unit"], str)):
        raise ValueError("Preserve source unit as text or explicit null")
    manifest = case["evidence_manifest"]
    if not isinstance(manifest, list):
        raise ValueError("evidence_manifest must be a list")
    ids = []
    for entry in manifest:
        if not isinstance(entry, dict):
            raise ValueError("Evidence manifest entry must be an object")
        for key in ("evidence_id", "kind", "as_of", "availability_basis"):
            if key not in entry:
                raise ValueError("Evidence manifest missing " + key)
        _text(entry["evidence_id"], "evidence_id")
        if entry["kind"] not in ("history", "covariate", "calendar", "context"):
            raise ValueError("Unknown evidence kind")
        ids.append(entry["evidence_id"])
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate public evidence ID")
    history = case["history_values"]
    if not isinstance(history, list) or not history:
        raise ValueError("Public history is required for honest failure fallback")
    protocol.validate_values(history, len(history))
    _json_copy(case)


def validate_t1(reply, case, profile):
    a = reply["answer"]
    _keys(a, ("drivers", "global_uncertainty", "evidence_ids_used", "mode", "spans"), "T1")
    _text(reply["explanation"], "T1 explanation")
    _text(a["global_uncertainty"], "global_uncertainty")
    h = case["geometry"]["H"]
    if a["mode"] not in ("DIRECT", "FACTORIZE") or not isinstance(a["spans"], list):
        raise ValueError("Invalid T1 mode/spans")
    spans = a["spans"]
    if not 1 <= len(spans) <= profile.max_segments:
        raise ValueError("Plan exceeds segment allowance")
    if (a["mode"] == "DIRECT") != (len(spans) == 1):
        raise ValueError("DIRECT requires one span; FACTORIZE requires at least two")
    cursor = 0
    for i, span in enumerate(spans):
        _keys(span, ("segment_id", "start", "stop"), "span")
        _span(span["start"], span["stop"], h, "assignment")
        if span["segment_id"] != "S" + str(i) or span["start"] != cursor:
            raise ValueError("Spans must use ordered S0.. IDs and contiguous exact coverage")
        if a["mode"] == "FACTORIZE" and span["stop"] - span["start"] < profile.min_segment_length:
            raise ValueError("Segment shorter than configured minimum")
        cursor = span["stop"]
    if cursor != h:
        raise ValueError("Spans do not cover the actual horizon")
    drivers = a["drivers"]
    if not isinstance(drivers, list):
        raise ValueError("drivers must be a list")
    driver_ids = []
    manifest = {item["evidence_id"]: item for item in case["evidence_manifest"]}
    used = set()
    for driver in drivers:
        _keys(driver, ("id", "name", "evidence", "mechanism", "relevance", "impact_windows",
                       "interactions", "uncertainty"), "driver")
        for field in ("id", "name", "mechanism", "uncertainty"):
            _text(driver[field], "driver " + field)
        driver_ids.append(driver["id"])
        if driver["relevance"] not in ("high", "medium", "low"):
            raise ValueError("Unknown relevance")
        if not all(isinstance(driver[k], list) for k in ("evidence", "impact_windows", "interactions")):
            raise ValueError("Driver evidence/windows/interactions must be lists")
        for item in driver["evidence"]:
            _keys(item, ("evidence_id", "kind", "as_of", "availability_basis", "quote_or_feature"), "evidence")
            source = manifest.get(item["evidence_id"])
            if source is None or any(item[k] != source[k] for k in ("kind", "as_of", "availability_basis")):
                raise ValueError("Evidence ID/qualification differs from the public manifest")
            _text(item["quote_or_feature"], "quote_or_feature")
            used.add(item["evidence_id"])
        for window in driver["impact_windows"]:
            _keys(window, ("start", "stop", "effect"), "impact window")
            _span(window["start"], window["stop"], h, "impact window")
            if window["effect"] not in ("up", "down", "mixed", "uncertain"):
                raise ValueError("Invalid impact effect")
        for interaction in driver["interactions"]:
            _keys(interaction, ("with_driver_id", "description"), "interaction")
            _text(interaction["description"], "interaction description")
    if len(set(driver_ids)) != len(driver_ids):
        raise ValueError("Duplicate driver ID")
    for driver in drivers:
        for item in driver["interactions"]:
            if item["with_driver_id"] not in driver_ids or item["with_driver_id"] == driver["id"]:
                raise ValueError("Interaction must reference another existing driver")
    eids = a["evidence_ids_used"]
    if (not isinstance(eids, list) or any(not isinstance(e, str) for e in eids)
            or len(set(eids)) != len(eids) or set(eids) != used):
        raise ValueError("evidence_ids_used must equal driver evidence references")


def _local_t1(t1, segment):
    drivers = [deepcopy(d) for d in t1["answer"]["drivers"] if any(
        w["start"] < segment["stop"] and w["stop"] > segment["start"] for w in d["impact_windows"])]
    local_ids = {d["id"] for d in drivers}
    for driver in drivers:
        driver["interactions"] = [i for i in driver["interactions"] if i["with_driver_id"] in local_ids]
    return {"explanation": t1["explanation"], "drivers": drivers,
            "global_uncertainty": t1["answer"]["global_uncertainty"],
            "segment_assignment": deepcopy(segment)}


def _validate_t2(reply, segment):
    a = reply["answer"]
    _keys(a, ("segment_id", "start", "stop", "values", "uncertainty", "self_check"), "T2")
    for k in ("segment_id", "start", "stop"):
        if a[k] != segment[k] or type(a[k]) is not type(segment[k]):
            raise ValueError("T2 must return its unchanged original assignment")
    expected = segment["stop"] - segment["start"]
    if isinstance(a["values"], list) and len(a["values"]) != expected:
        raise ValueError("T2 returned " + str(len(a["values"])) + " values; original assignment requires " + str(expected))
    protocol.validate_values(a["values"], expected)
    for k in ("uncertainty", "self_check"):
        _text(a[k], k)
    _text(reply["explanation"], "T2 explanation")


def _validate_t3(reply, segments, eligible, t1):
    a = reply["answer"]
    _keys(a, ("action", "segment_id", "directive"), "T3")
    if a["action"] == "STOP":
        if a["segment_id"] is not None or a["directive"] is not None:
            raise ValueError("STOP requires null segment_id/directive")
        return
    if a["action"] != "REFINE" or a["segment_id"] not in eligible:
        raise ValueError("REFINE must select an eligible segment")
    segment = next(s for s in segments if s["segment_id"] == a["segment_id"])
    d = a["directive"]
    _keys(d, ("concern", "review_span", "driver_refs", "own_report_field"), "directive")
    if d["concern"] not in CONCERNS:
        raise ValueError("Unknown diagnostic concern")
    span = d["review_span"]
    if span is not None:
        if not isinstance(span, list) or len(span) != 2:
            raise ValueError("review_span requires [start,stop)")
        _span(span[0], span[1], segment["stop"], "review_span")
        if span[0] < segment["start"]:
            raise ValueError("review_span leaves selected original segment")
    refs = d["driver_refs"]
    if (not isinstance(refs, list) or len(refs) > 2 or any(not isinstance(r, str) for r in refs)
            or len(set(refs)) != len(refs)):
        raise ValueError("driver_refs requires at most two distinct IDs")
    if (d["concern"] == "event_coverage" and not refs) or (d["concern"] == "event_interaction" and len(refs) != 2):
        raise ValueError("Concern requires cited drivers")
    local = {v["id"]: v for v in _local_t1(t1, segment)["drivers"]}
    for ref in refs:
        if ref not in local or (span is not None and not any(
                w["start"] < span[1] and w["stop"] > span[0] for w in local[ref]["impact_windows"])):
            raise ValueError("Driver reference is not local or does not overlap attention window")
    if d["own_report_field"] not in (None, "summary", "self_check", "uncertainty"):
        raise ValueError("Unknown own report field")
    if d["concern"] == "mechanism_consistency" and d["own_report_field"] not in ("summary", "self_check"):
        raise ValueError("mechanism_consistency requires own summary or self_check")


def _bind_feedback(action, segment, t1, report, case):
    d = action["directive"]
    by_id = {x["id"]: x for x in _local_t1(t1, segment)["drivers"]}
    drivers = [deepcopy(by_id[x]) for x in d["driver_refs"]]
    evidence_ids = {e["evidence_id"] for driver in drivers for e in driver["evidence"]}
    field = d["own_report_field"]
    own = None if field is None else {"field": field, "text": (
        report["explanation"] if field == "summary" else report["answer"][field])}
    return {"directive": deepcopy(d), "drivers": drivers,
            "source_evidence": [deepcopy(x) for x in case["evidence_manifest"] if x["evidence_id"] in evidence_ids],
            "own_report": own, "instruction": INSTRUCTIONS[d["concern"]] +
            " This is a review hypothesis, not a verified error. T1 mechanisms and interactions are hypotheses. "
            "Retain source qualifications. Keeping values unchanged is legal. The attention window is not an edit mask; "
            "return your complete original segment, considering lag/rebound within it."}


def _t4_policy(spans, h, profile):
    return {"boundaries": [{"boundary_id": "B" + str(i), "position": s["stop"],
                            "start": max(0, s["stop"] - profile.t4_max_patch_steps),
                            "stop": min(h, s["stop"] + profile.t4_max_patch_steps)}
                           for i, s in enumerate(spans[:-1])],
            "max_patch_steps": profile.t4_max_patch_steps, "max_edits": len(spans) - 1,
            "max_total_steps": min(profile.t4_max_total_steps, h - 1),
            "atomic_snapshot": True, "one_model_decision": True}


def _validate_t4(reply, policy, h):
    a = reply["answer"]
    _keys(a, ("action", "edits"), "T4")
    if not isinstance(a["edits"], list):
        raise ValueError("T4 edits must be a list")
    if a["action"] == "KEEP":
        if a["edits"]:
            raise ValueError("KEEP requires empty edits")
        return
    if a["action"] != "REGENERATE" or not 1 <= len(a["edits"]) <= policy["max_edits"]:
        raise ValueError("REGENERATE requires bounded edits")
    bands = {b["boundary_id"]: b for b in policy["boundaries"]}
    seen, previous_stop, total = set(), -1, 0
    for edit in a["edits"]:
        _keys(edit, ("boundary_id", "start", "stop", "values"), "edit")
        band = bands.get(edit["boundary_id"])
        if band is None or edit["boundary_id"] in seen:
            raise ValueError("Each edit requires a distinct authorized boundary")
        _span(edit["start"], edit["stop"], h, "T4 edit")
        start, stop = edit["start"], edit["stop"]
        width = stop - start
        b = band["position"]
        if (start < band["start"] or stop > band["stop"] or width > policy["max_patch_steps"]
                or not (start <= b - 1 < stop or start <= b < stop)):
            raise ValueError("Edit must be bounded and touch boundary-adjacent position")
        if start < previous_stop:
            raise ValueError("T4 edits must be sorted and non-overlapping")
        protocol.validate_values(edit["values"], width)
        seen.add(edit["boundary_id"])
        previous_stop, total = stop, total + width
    if total > policy["max_total_steps"] or total >= h:
        raise ValueError("T4 aggregate edited steps exceeds strict sub-horizon allowance")


def validate_public_role(role: str, parsed: dict, variables: dict, *,
                         profile: Optional[Profile] = None) -> None:
    """Validate a public role against its exact runtime assignment and policy.

    Sampling consumers pass the same public_variables used for rendering. No
    canonical labels, reviewer or legacy role mapping is needed. T2_refine is
    the protocol name for a T2 revision, not an additional trainable role.
    """
    _keys(parsed, ("explanation", "answer"), "parsed public reply")
    protocol.render_reply(parsed["explanation"], parsed["answer"])
    geometry = variables["geometry"]
    if role == "T1":
        if profile is None:
            settings = variables["compute_policy"]
            profile = Profile(**{f.name: settings[f.name] for f in fields(Profile)
                                 if f.name in settings and f.name != "message_token_counter"
                                 and f.name != "max_context_tokens"})
        validate_t1(parsed, {"geometry": geometry,
                             "evidence_manifest": variables["evidence_manifest"]}, profile)
        allowed = variables["compute_policy"].get("allowed_modes", ("DIRECT", "FACTORIZE"))
        if parsed["answer"]["mode"] not in allowed:
            raise ValueError("T1 mode is not permitted by the supplied compute policy")
    elif role in ("T2", "T2_refine"):
        _validate_t2(parsed, variables["segment_assignment"])
    elif role == "T3":
        t1 = variables["accepted_t1"]
        _validate_t3(parsed, t1["answer"]["spans"],
                     variables["reflection_policy_state"]["eligible_segments"], t1)
    elif role == "T4":
        _validate_t4(parsed, variables["t4_policy"], geometry["H"])
    else:
        raise ValueError("Unsupported public role: " + str(role))


class _CallFailure(Exception):
    pass


class AcceptanceJournalFailure(RuntimeError):
    """Infrastructure failure that must not create a replacement episode."""


def compute_policy(case: dict, profile: Optional[Profile] = None) -> dict:
    """Original teacher policy, retained verbatim for live collection bindings.

    Public production and derived student inputs use public_compute_policy;
    engineering capacities remain in the episode/serving reference identity.
    """
    profile = profile or Profile()
    return {**profile.record(), "horizon_steps": case["geometry"]["H"],
            "mandatory_post_revision_t3": True,
            "protected_post_revision_t3_attempts": 1 + profile.structural_retries,
            "protected_t4_attempts": 1 + profile.structural_retries,
            "independent_per_call_retry_budgets": True}


def public_policy(policy: dict) -> dict:
    """Project only semantic planning limits into the public T1 request.

    Native serialization and serving token capacities do not change the
    forecasting instruction. This deterministic adapter leaves its source
    policy untouched, including policies retained in old teacher receipts.
    """
    return {key: deepcopy(policy[key]) for key in _PUBLIC_POLICY_FIELDS if key in policy}


def public_compute_policy(case: dict, profile: Optional[Profile] = None) -> dict:
    return public_policy(compute_policy(case, profile))


def _receipt(result):
    if isinstance(result, str):
        return result, result
    if isinstance(result, dict) and isinstance(result.get("content"), str):
        return _json_copy(result), result["content"]
    raise ValueError("Caller must return public content string or JSON receipt with content")


def _require_completion(receipt):
    # Offline scripted callers may omit provider termination metadata. An
    # explicit provider reason is evidence and must never be ignored merely
    # because a partial generation happens to contain valid answer JSON.
    if isinstance(receipt, dict) and "finish_reason" in receipt and receipt["finish_reason"] != "stop":
        raise ValueError("Teacher response did not terminate normally: " + str(receipt["finish_reason"]))


def execute(case: dict, caller: Callable, *, profile: Optional[Profile] = None,
            reviewer: Optional[Callable] = None, on_accept: Optional[Callable] = None) -> dict:
    """Run fresh train-only teacher calls. Reviewer(raw_p1, public_messages).

    Reviewer returns a real single-answer KEEP/REWRITE string or receipt with
    content. Its private prompt/labels stay in its closure, never actor state.
    Structural retries have independent per-call allowances (including T4).
    Optional context checks measure each actual request with the supplied exact
    counter; they do not certify unspecified future responses/tokenizer modes.
    on_accept(owned_call_record), when supplied, persists validated public role
    receipts before advancing. Its errors propagate as AcceptanceJournalFailure
    rather than creating model repair turns or a replacement fallback episode.
    """
    return _execute(case, caller, profile=profile, reviewer=reviewer,
                    on_accept=on_accept, response_format="teacher", public=False)


def execute_public(case: dict, caller: Callable, *, profile: Optional[Profile] = None,
                   response_format: str = "student", on_accept: Optional[Callable] = None,
                   diagnostic_forced_mode: Optional[str] = None,
                   diagnostic_fixed_plan: Optional[dict] = None) -> dict:
    """Run the adopted workflow using only public evidence and actual model calls.

    Public T1 is accepted directly after validation. Initial T2 assignments are
    independent. DIRECT returns after whole-horizon T2 without T3 or T4.
    On FACTORIZE, every successful revision is reassessed by T3 and every normal
    FACTORIZE completion calls model T4. Failures retain separate fallback status.
    """
    return _execute(case, caller, profile=profile, reviewer=None,
                    on_accept=on_accept, response_format=response_format, public=True,
                    diagnostic_forced_mode=diagnostic_forced_mode,
                    diagnostic_fixed_plan=diagnostic_fixed_plan)


def direct_cost_reference(case: dict, caller: Callable, *, profile: Optional[Profile] = None,
                          response_format: str = "student", on_accept: Optional[Callable] = None) -> dict:
    """Actual constrained DIRECT T1 plus full-H T2 cost reference, without T3/T4.

    This explicit program_DIRECT_STOP_T1_T2 reference is distinct from normal
    DIRECT execution: the reference additionally constrains T1 to DIRECT.
    """
    return _execute(case, caller, profile=profile, reviewer=None,
                    on_accept=on_accept, response_format=response_format, public=True,
                    direct_reference=True)


class DownstreamContinuation:
    """Shared public-state transition mechanics for serial and bounded forks.

    Calls are supplied by the owning executor. Snapshots are immutable copies;
    ordinary endpoints never incorporate a subsequently explored alternative.
    """
    def __init__(self, case, profile, accepted, reports, own_history, episode, call,
                 *, refinements=0, first_refined_segment=None, trace=None):
        self.case, self.profile, self.accepted = case, profile, accepted
        self.spans = accepted["answer"]["spans"]
        self.reports, self.own_history, self.episode, self.call = reports, own_history, episode, call
        self.refinements, self.first_refined_segment = refinements, first_refined_segment
        self.trace = trace
        self.base = {"public_case": case["public_case"], "geometry": case["geometry"]}

    def values(self):
        return [v for s in self.spans for v in self.reports[s["segment_id"]]["answer"]["values"]]

    def snapshot(self):
        return deepcopy({"accepted_t1": self.accepted, "reports": self.reports,
            "own_history": self.own_history, "reflection_history": self.episode["reflection_history"],
            "termination": self.episode.get("termination"), "refinements": self.refinements,
            "first_refined_segment": self.first_refined_segment})

    def _invoke(self, role, variables, phase, validate):
        if self.trace is None:
            return self.call(role, variables, phase, validate)
        event = {"role": role, "phase": phase, "before": self.snapshot(),
                 "call_start": len(self.episode["calls"])}
        self.trace.append(event)
        try:
            return self.call(role, variables, phase, validate)
        finally:
            event["calls"] = deepcopy(self.episode["calls"][event["call_start"]:])

    def assess(self):
        eligible = [s["segment_id"] for s in self.spans] if self.refinements < self.profile.max_refinements else []
        policy = {"remaining_refinements": self.profile.max_refinements - self.refinements,
                  "eligible_segments": eligible, "post_revision_t3_required": True,
                  "protected_t4_attempts": (1 + self.profile.structural_retries if len(self.spans) > 1 else 0),
                  "structural_retries_per_call": self.profile.structural_retries}
        return self._invoke("T3", {**self.base, "accepted_t1": self.accepted,
            "current_segment_reports": [self.reports[s["segment_id"]] for s in self.spans],
            "reflection_history": self.episode["reflection_history"], "reflection_policy_state": policy},
            "ASSESS:" + str(self.refinements), lambda r: _validate_t3(r, self.spans, eligible, self.accepted))

    def apply_assessment(self, t3, t3_call):
        event = {"t3_call_id": t3_call["call_id"], "t3_reply": deepcopy(t3),
                 "bound_local_feedback": None, "revision_call_id": None, "revision_status": None}
        self.episode["reflection_history"].append(event)
        action = t3["answer"]
        if action["action"] == "STOP":
            self.episode["termination"] = {"kind": "model_STOP", "call_id": t3_call["call_id"],
                                           "refinements": self.refinements}
            return False
        sid = action["segment_id"]
        segment = next(s for s in self.spans if s["segment_id"] == sid)
        feedback = _bind_feedback(action, segment, self.accepted, self.reports[sid], self.case)
        event["bound_local_feedback"], event["revision_status"] = deepcopy(feedback), "pending"
        try:
            revised, revision_call = self._invoke("T2", {**self.base,
                "local_t1_information": _local_t1(self.accepted, segment), "segment_assignment": segment,
                "own_report_history": self.own_history[sid], "bound_local_feedback": feedback,
                "mode": "REFINE"}, "REFINE:" + sid, lambda r: _validate_t2(r, segment))
        except _CallFailure:
            event["revision_status"] = "failed"
            raise
        self.reports[sid] = revised
        self.own_history[sid].append(deepcopy(revised))
        event["revision_call_id"], event["revision_status"] = revision_call["call_id"], "accepted"
        self.refinements += 1
        if self.first_refined_segment is None:
            self.first_refined_segment = sid
        return True

    def run_reflection(self):
        while True:
            t3, call = self.assess()
            if not self.apply_assessment(t3, call):
                return self.values()

    def finalize(self):
        snapshot = self.values()
        if self.accepted["answer"]["mode"] != "FACTORIZE":
            self.episode["t4_status"] = "not_applicable_DIRECT"
            return snapshot
        self.episode["t4_status"] = "pending"
        h = self.case["geometry"]["H"]
        policy = _t4_policy(self.spans, h, self.profile)
        t4, _ = self._invoke("T4", {**self.base, "accepted_t1": self.accepted,
            "final_segment_reports": [self.reports[s["segment_id"]] for s in self.spans],
            "reflection_history_and_termination": {"history": self.episode["reflection_history"],
                                                    "termination": self.episode["termination"]},
            "t4_policy": policy}, "FINALIZE", lambda r: _validate_t4(r, policy, h))
        final = list(snapshot)
        for edit in t4["answer"]["edits"]:
            final[edit["start"]:edit["stop"]] = edit["values"]
        self.episode["t4_status"] = "model_" + t4["answer"]["action"]
        return final


def execute_public_from_t1(case, caller, accepted_t1, *, profile, call_handler, initial_calls=None,
                           transition_trace=None):
    """Run the actual T2/reflection/T4 engine from an accepted public plan.

    The paired-credit workflow owns front-end generation and per-attempt
    journaling; the existing engine owns all downstream forecast mechanics.
    """
    if profile.profile_version != STEP_TARGET_VERSION:
        raise ValueError("Accepted-plan paired hooks require the compact-v2 representation")
    return _execute(case, caller, profile=profile, reviewer=None, on_accept=None,
                    response_format="student", public=True, accepted_t1=accepted_t1,
                    call_handler=call_handler, initial_calls=initial_calls, transition_trace=transition_trace)


def _execute(case, caller, *, profile, reviewer, on_accept, response_format, public,
             direct_reference=False, diagnostic_forced_mode=None, diagnostic_fixed_plan=None,
             accepted_t1=None, call_handler=None, initial_calls=None, transition_trace=None):
    if diagnostic_forced_mode is not None and (
            diagnostic_forced_mode not in ("DIRECT", "FACTORIZE")
            or not public or direct_reference):
        raise ValueError("Diagnostic forcing requires a normal public DIRECT/FACTORIZE workflow")
    profile = profile or Profile()
    json_rows = profile.profile_version == JSON_ROWS_VERSION
    compact = profile.profile_version in COMPACT_VERSIONS
    indexed = json_rows or compact
    compact_teacher = compact and profile.profile_version == STEP_TARGET_VERSION and not public and response_format == "teacher"
    public_compact_teacher = public and profile.profile_version == STEP_TARGET_VERSION and response_format == "teacher"
    if indexed and not (compact_teacher or public_compact_teacher) and (not public or response_format != "student"):
        raise ValueError("Indexed profiles require public student execution or explicit compact-v2 teacher format")
    _validate_case(case, train_only=not public)
    if case.get("profile_version", profile.profile_version) != profile.profile_version:
        raise ValueError("Public case profile differs from selected execution profile")
    if indexed:
        if "public_json_case" not in case:
            raise ValueError("Indexed profile requires attached public_json_case")
        case = attach_public_json_case(case, case["public_json_case"])
    if response_format not in ("teacher", "student"):
        raise ValueError("response_format must be teacher or student")
    if not callable(caller) or (not public and not callable(reviewer)):
        raise ValueError("Teacher constructor requires caller and complete-T1 reviewer")
    if on_accept is not None and not callable(on_accept):
        raise ValueError("on_accept must be callable")
    case = deepcopy(case)
    if accepted_t1 is not None:
        if not public or direct_reference or diagnostic_fixed_plan is not None:
            raise ValueError("Accepted T1 requires normal public downstream execution")
        _reject_private(accepted_t1)
        _keys(accepted_t1, ("explanation", "answer"), "accepted T1")
        validate_t1(accepted_t1, case, profile)
        if not callable(call_handler):
            raise ValueError("Accepted T1 execution requires an explicit call handler")
    elif call_handler is not None or initial_calls:
        raise ValueError("Downstream hooks require an accepted public T1")
    if diagnostic_fixed_plan is not None:
        if not public or direct_reference or diagnostic_forced_mode not in (None, 'FACTORIZE'):
            raise ValueError('Fixed plan requires normal public FACTORIZE execution')
        _reject_private(diagnostic_fixed_plan)
        _keys(diagnostic_fixed_plan, ('explanation', 'answer'), 'fixed plan')
        diagnostic_fixed_plan = _json_copy(diagnostic_fixed_plan)
        validate_t1(diagnostic_fixed_plan, case, profile)
        if diagnostic_fixed_plan['answer']['mode'] != 'FACTORIZE':
            raise ValueError('Fixed plan must be a legal FACTORIZE plan')
    h = case["geometry"]["H"]
    episode = {"workflow_revision": WORKFLOW_REVISION, "profile_version": profile.profile_version, "profile": profile.record(),
               "case_id": case["case_id"], "source": case["source"], "split": case["split"],
               "geometry": deepcopy(case["geometry"]), "calls": [], "failures": [],
               "t1_pass1": None, "t1_review": None, "accepted_t1": None, "t1_supervision": None,
               "segment_reports": [], "reflection_history": [], "termination": None,
               "t4_status": "not_started", "normal_completion": False,
               "complete_forecast": False, "final_values": None, "fallback": None,
               "context_validation": "per_actual_request" if profile.message_token_counter else "unmeasured"}
    base = {"public_case": case["public_case"], "geometry": case["geometry"]}
    if initial_calls:
        episode["calls"] = deepcopy(initial_calls)
    policy = public_compute_policy(case, profile) if public or compact_teacher else compute_policy(case, profile)
    if diagnostic_forced_mode is not None:
        policy["allowed_modes"] = [diagnostic_forced_mode]
        episode["diagnostic_forced_mode"] = diagnostic_forced_mode
    if public:
        episode.update(execution_mode="public", response_format=response_format)
    if indexed:
        if compact:
            from castfactory.agentic.compact_protocol import answer_format_for_version
        episode.update(answer_format=answer_format_for_version(profile.profile_version) if compact else "position_load_rows_json_v1",
                       internal_numeric_format="values_for_runtime_mechanics")
    if direct_reference:
        policy.update(max_segments=1, max_refinements=0, allowed_modes=["DIRECT"],
                      mandatory_post_revision_t3=False, protected_post_revision_t3_attempts=0,
                      protected_t4_attempts=0, reference_protocol="program_DIRECT_STOP_T1_T2")
        episode["reference_protocol"] = "program_DIRECT_STOP_T1_T2"

    def call(role, variables, phase, validate):
        protocol_role = "T2_refine" if role == "T2" and variables.get("mode") == "REFINE" else role
        if indexed:
            if compact:
                from castfactory.agentic import compact_prompts as role_prompts, compact_protocol as role_protocol
                request = role_prompts.request_for(protocol_role, variables, case["public_json_case"])
                recorded_variables = variables
            else:
                from castfactory.agentic import json_rows_prompts as role_prompts, json_rows_protocol as role_protocol
                request = role_prompts.request_for(protocol_role, variables, case["public_json_case"])
                recorded_variables = {key: value for key, value in request.items() if key not in ("case", "role")}
            messages = role_prompts.messages_for(protocol_role, variables, case["public_json_case"],
                **({"selected_case_id": case["case_id"], "protocol_version": profile.profile_version,
                    "response_format": response_format,
                    **({"reviewed_text": case["teacher_reviewed_text"]} if compact_teacher else {})} if compact else {}))
        else:
            messages = protocol.messages_for(protocol_role, deepcopy(variables), response_format=response_format)
            recorded_variables = variables
        if call_handler is not None:
            return call_handler(role=role, protocol_role=protocol_role, variables=variables,
                                phase=phase, validate=validate, request=request,
                                messages=messages, episode=episode)
        base_message_count = len(messages)
        for attempt in range(profile.structural_retries + 1):
            entry = {"call_id": len(episode["calls"]), "role": role, "phase": phase,
                     "attempt": attempt, "messages": deepcopy(messages), "status": "pending",
                     "protocol_role": protocol_role, "public_variables": deepcopy(recorded_variables),
                     "base_message_count": base_message_count,
                     "raw_response": None, "content": None, "parsed": None, "error": None,
                     "context": {"measured": False}}
            if indexed:
                entry.update(profile_version=profile.profile_version,
                             public_json_case=deepcopy(case["public_json_case"]),
                             request=deepcopy(request),
                             normalized_parsed=None)
                if compact_teacher:
                    entry["teacher_reviewed_text"] = deepcopy(case["teacher_reviewed_text"])
            episode["calls"].append(entry)
            try:
                if profile.message_token_counter:
                    try:
                        counted_messages = deepcopy(messages)
                        if compact_teacher:
                            counted_messages = role_prompts.messages_for(protocol_role, variables,
                                case["public_json_case"], selected_case_id=case["case_id"],
                                protocol_version=profile.profile_version, response_format="student",
                                reviewed_text=case["teacher_reviewed_text"])
                            counted_messages.extend(deepcopy(messages[base_message_count:]))
                        count = profile.message_token_counter(counted_messages)
                    except Exception as exc:
                        entry["status"] = "context_rejected"
                        raise _CallFailure("Message token counter failed: " + type(exc).__name__) from None
                    if type(count) is not int or count < 0:
                        entry["status"] = "context_rejected"
                        raise _CallFailure("Exact message counter returned invalid token count")
                    entry["context"] = {"measured": True, "prompt_tokens": count,
                                        "reserved_response_tokens": profile.max_response_tokens}
                    if compact_teacher:
                        entry["context"]["measurement"] = "native_student_projection_with_actual_repair_history"
                    if profile.max_context_tokens is not None and count + profile.max_response_tokens > profile.max_context_tokens:
                        entry["status"] = "context_rejected"
                        raise _CallFailure("Actual serialized request exceeds configured context capacity")
                try:
                    result = caller(role, deepcopy(messages))
                except Exception as exc:
                    # Transport retry policy belongs to the caller/collector.
                    # With no model reply there is no structural repair turn to
                    # supervise. Exception text may contain transport secrets.
                    entry["status"] = "transport_failed"
                    raise _CallFailure("Caller transport failed: " + type(exc).__name__) from exc
                # Archive even a JSON receipt with an invalid/missing content
                # field before response-contract validation rejects it.
                entry["raw_response"] = _json_copy(result)
                entry["raw_response"], entry["content"] = _receipt(result)
                if compact and isinstance(entry["raw_response"], dict) and entry["raw_response"].get(
                        "profile_version", profile.profile_version) != profile.profile_version:
                    raise ValueError("Sampled response profile differs from compact execution profile")
                _require_completion(entry["raw_response"])
                if indexed:
                    parsed = role_protocol.parse_role_answer(entry["content"], protocol_role, request,
                        profile=profile, **({"protocol_version": profile.profile_version,
                                           "response_format": response_format} if compact else {}))
                    normalized = role_protocol.normalize_role_reply(parsed, protocol_role, h)
                    validate(normalized)
                    entry["normalized_parsed"] = deepcopy(normalized)
                else:
                    parsed = protocol.parse_reply(entry["content"], response_format=response_format)
                    validate(parsed)
                    normalized = parsed
                entry["parsed"], entry["status"] = deepcopy(parsed), "accepted"
            except Exception as exc:
                entry["error"] = str(exc)
                if entry["status"] == "pending":
                    entry["status"] = "failed"
                episode["failures"].append({"call_id": entry["call_id"], "role": role, "error": str(exc)})
                if isinstance(exc, _CallFailure) or attempt == profile.structural_retries:
                    raise _CallFailure(role + " " + phase + " failed: " + str(exc)) from exc
                if entry["content"] is not None:
                    messages.append({"role": "assistant", "content": entry["content"]})
                repair = "Structural validation failed: " + str(exc) + ". Return one complete corrected response for the same original assignment and policy."
                messages.append(role_prompts.repair_message(protocol_role, repair) if indexed else
                                {"role": "user", "content": repair})
                continue
            # Infrastructure journaling is outside model generation/validation
            # retries. Durable accepted markers can recover a process crash
            # even before the final episode record exists.
            if on_accept is not None:
                try:
                    on_accept(deepcopy(entry))
                except Exception as exc:
                    raise AcceptanceJournalFailure("Acceptance journal failed: " + type(exc).__name__) from None
            return deepcopy(normalized), entry

    def review_t1(p1, p1_call):
        review_record = {"receipt": None, "content": None, "parsed": None, "status": "pending", "error": None}
        episode["t1_review"] = review_record
        try:
            result = reviewer(p1_call["content"], deepcopy(p1_call["messages"]))
            review_record["receipt"] = _json_copy(result)
            review_record["receipt"], review_record["content"] = _receipt(result)
            _require_completion(review_record["receipt"])
            review = protocol.parse_reply(review_record["content"], response_format="teacher")
            review_record["parsed"] = deepcopy(review)
            accepted = protocol.review_t1(review, p1)
            validate_t1(accepted, case, profile)
            review_record["status"] = "accepted"
        except AcceptanceJournalFailure:
            # The reviewer closure owns its separate privileged accepted marker.
            raise
        except Exception as exc:
            review_record["status"], review_record["error"] = "failed", str(exc)
            episode["failures"].append({"role": "T1_P2", "error": str(exc)})
            raise _CallFailure("Complete T1 review failed: " + str(exc)) from exc
        return accepted, review

    reports, own_history, spans = {}, {}, []
    try:
        def check_t1(reply):
            validate_t1(reply, case, profile)
            if diagnostic_forced_mode is not None and reply["answer"]["mode"] != diagnostic_forced_mode:
                raise ValueError("T1 mode is not permitted by the supplied compute policy")
            if direct_reference and reply["answer"]["mode"] != "DIRECT":
                raise ValueError("program_DIRECT_STOP_T1_T2 requires a DIRECT whole-horizon plan")

        if accepted_t1 is not None:
            accepted = deepcopy(accepted_t1)
            episode["t1_supervision"] = {"kind": "assembled_public_interpretation_and_plan",
                                         "t1_generated": False}
        elif diagnostic_fixed_plan is not None:
            accepted = deepcopy(diagnostic_fixed_plan)
            episode['diagnostic_fixed_plan'] = {'kind': 'externally_frozen_legal_T1',
                't1_generated': False, 'cost_scope': 'downstream_T2_T3_T4_only'}
        else:
            p1, p1_call = call("T1", {**base, "evidence_manifest": case["evidence_manifest"],
                                     "compute_policy": policy, "t1_schema": protocol.T1_SCHEMA},
                               "PASS1", check_t1)
            episode["t1_pass1"] = {"call_id": p1_call["call_id"], "reply": deepcopy(p1)}
            if public:
                accepted = deepcopy(p1)
            else:
                accepted, review = review_t1(p1, p1_call)
                episode["t1_supervision"] = {"kind": "label_informed", "decision": review["answer"]["decision"],
                                             "pass1_call_id": p1_call["call_id"], "review_record": "t1_review",
                                             "derived": True}
        episode["accepted_t1"] = deepcopy(accepted)
        spans = accepted["answer"]["spans"]
        for segment in spans:
            sid = segment["segment_id"]
            result, _ = call("T2", {**base, "local_t1_information": _local_t1(accepted, segment),
                                    "segment_assignment": segment, "mode": "INITIAL"},
                             "INITIAL:" + sid, lambda r, s=segment: _validate_t2(r, s))
            reports[sid], own_history[sid] = result, [deepcopy(result)]
        continuation = DownstreamContinuation(case, profile, accepted, reports, own_history,
                                             episode, call, trace=transition_trace)
        if direct_reference or accepted["answer"]["mode"] == "DIRECT":
            episode["termination"] = {"kind": "program_DIRECT_STOP_T1_T2" if direct_reference else "direct_forecast_complete",
                                      "refinements": 0}
        else:
            continuation.run_reflection()
        final = continuation.finalize()
        episode["final_values"], episode["normal_completion"] = final, True
    except _CallFailure as exc:
        episode["termination"] = {"kind": "runtime_failure", "error": str(exc)}
        episode["t4_status"] = "failed" if episode["t4_status"] == "pending" else "not_run_failure"
        if spans and all(s["segment_id"] in reports for s in spans):
            final = [v for s in spans for v in reports[s["segment_id"]]["answer"]["values"]]
            kind = "last_complete_segment_snapshot"
        else:
            cycle = case["history_values"][-24:]
            final = [cycle[i % len(cycle)] for i in range(h)]
            kind = "public_history_cycle"
        episode["final_values"] = final
        episode["fallback"] = {"kind": kind, "reason": str(exc), "model_t4_completed": False}
    episode["segment_reports"] = [deepcopy(reports[s["segment_id"]]) for s in spans if s["segment_id"] in reports]
    protocol.validate_values(episode["final_values"], h)
    episode["complete_forecast"] = True
    return _json_copy(episode)


def sft_rows(episode: dict, response_format: str = "student") -> list:
    """Export accepted T1 and real successful role-local calls, audit elsewhere.

    The chosen format clause is rendered against the original variables and
    real repair history is retained unchanged. ``teacher_messages`` archives
    the exact successful teacher request. Native tokenizer prefix/EOS handling
    remains the explicitly selected consumer's responsibility.
    """
    compact = episode.get("profile_version") == STEP_TARGET_VERSION
    if episode.get("profile_version") not in (VERSION, STEP_TARGET_VERSION) or episode.get("split") != "train":
        raise ValueError("Expected a train episode from the new teacher profile")
    if response_format not in ("teacher", "student"):
        raise ValueError("Unknown SFT response format")
    rows = []

    def request(call):
        original = deepcopy(call["messages"])
        variables = deepcopy(call["public_variables"])
        adaptation = None
        if call["role"] == "T1":
            original_policy = variables["compute_policy"]
            variables["compute_policy"] = public_policy(original_policy)
            adaptation = {"kind": "deterministic_public_policy_projection", "version": PUBLIC_POLICY_VERSION,
                          "omitted_fields": sorted(set(original_policy) - set(variables["compute_policy"]))}
        if compact:
            from castfactory.agentic.compact_prompts import messages_for, request_for
            rendered = messages_for(call["protocol_role"], variables, call["public_json_case"],
                selected_case_id=episode["case_id"], protocol_version=STEP_TARGET_VERSION,
                response_format=response_format, reviewed_text=call["teacher_reviewed_text"])
        else:
            rendered = protocol.messages_for(call["protocol_role"], variables,
                                             response_format=response_format)
        rendered.extend(deepcopy(original[call["base_message_count"]:]))
        provenance = {"kind": "deterministic_format_adapter",
                                              "original_response_format": "teacher",
                                              "selected_response_format": response_format,
                                              "repair_history": "retained_exactly",
                                              "enable_thinking": True if response_format == "student" else None}
        if adaptation is not None:
            provenance["public_policy_adapter"] = adaptation
        result = {"messages": rendered, "teacher_messages": original,
                  "response_format": response_format, "serialization_provenance": provenance}
        if compact:
            result.update(protocol_version=STEP_TARGET_VERSION, answer_format="step_target_markdown_v2",
                          semantic_request=request_for(call["protocol_role"], variables, call["public_json_case"]))
            provenance.update(teacher_request_was_compact=True, fresh_teacher_request=True,
                              semantic_request="out_of_band_validation_only_not_model_input")
        return result

    def render(parsed, role):
        if compact:
            from castfactory.agentic.compact_protocol import render_role_reply
            return render_role_reply(parsed, role, episode["geometry"]["H"],
                response_format=response_format, protocol_version=STEP_TARGET_VERSION)
        return protocol.render_reply(parsed["explanation"], parsed["answer"], response_format=response_format)
    accepted = episode.get("accepted_t1")
    if accepted is not None:
        supervision = episode.get("t1_supervision")
        if not supervision or supervision.get("kind") != "label_informed":
            raise ValueError("Accepted T1 requires explicit label-informed review provenance")
        p1 = next(c for c in episode["calls"] if c["call_id"] == supervision["pass1_call_id"])
        rows.append({"profile_version": episode["profile_version"], "case_id": episode["case_id"],
                     "episode_id": episode["case_id"], "split": "train", "role": "T1",
                     "target_source": "teacher_demonstration", **request(p1),
                     "assistant_response": render(accepted, "T1"),
                     "source_call_id": p1["call_id"], "derived": True,
                     "t1_supervision": deepcopy(supervision)})
    for call in episode["calls"]:
        if call["role"] == "T1" or call["status"] != "accepted":
            continue
        rows.append({"profile_version": episode["profile_version"], "case_id": episode["case_id"],
                     "episode_id": episode["case_id"], "split": "train", "role": call["role"],
                     "target_source": "teacher_demonstration", **request(call),
                     "assistant_response": (call["content"] if response_format == "teacher" else
                                            render(call["parsed"], call["protocol_role"])),
                     "source_call_id": call["call_id"], "derived": False, "phase": call["phase"]})
    return _json_copy(rows)
