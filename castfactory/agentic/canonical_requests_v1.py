"""Full-field request-only boundaries for the existing T0/T1/T2/T3 callbacks.

No model, transport, default token budget, training label, or new identity hash
is introduced here. Mock callbacks are CPU plumbing fixtures, not trajectories.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass
from datetime import date, datetime
import json
import math
from types import MappingProxyType
from typing import Any, Callable

import numpy as np

from castfactory.agentic import t0 as t0_contract
from castfactory.agentic.contracts import (
    ContextWorkspace, FactorizationPlan, ForecastObservation, GlobalAction,
    GlobalSchedulerState, SegmentDraft, SegmentRequest, Span, TrainingTargets,
    _reject_label_keys,
)
from castfactory.training.chat_template_compat import (
    has_chat_template, render_chat_template_ids, render_chat_template_text,
)


STAGES = ("T0", "T1", "T2", "T3")
REQUEST_VERSION = "canonical-actor-request-v1"


def json_safe(value: Any) -> Any:
    """Preserve dataclass fields without deepcopying immutable mappingproxy."""
    if isinstance(value, TrainingTargets):
        raise ValueError("TrainingTargets cannot enter actor requests")
    if is_dataclass(value) and not isinstance(value, type):
        return {item.name: json_safe(getattr(value, item.name)) for item in fields(value)}
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("actor JSON mapping keys must be strings")
        if any(key.strip().lower().replace("-", "_").startswith("target_future_label") for key in value):
            raise ValueError("canonical future labels cannot enter actor requests")
        _reject_label_keys(value, "actor request")
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        if np.issubdtype(value.dtype, np.datetime64):
            return json_safe(value[()]) if value.ndim == 0 else [json_safe(item) for item in value]
        return json_safe(value.tolist())
    if isinstance(value, np.datetime64):
        if np.isnat(value):
            raise ValueError("actor requests cannot contain NaT timestamps")
        return np.datetime_as_string(value)
    if isinstance(value, np.generic):
        converted = value.item()
        if isinstance(converted, np.generic):
            raise TypeError("unsupported numpy scalar precision; refusing lossy coercion")
        return json_safe(converted)
    if isinstance(value, (tuple, list)):
        return [json_safe(item) for item in value]
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("actor requests cannot contain non-finite JSON numbers")
        return value
    raise TypeError(f"unsupported actor JSON value: {type(value).__name__}")


def _json(value: Any) -> str:
    return json.dumps(json_safe(value), ensure_ascii=True, allow_nan=False, separators=(",", ":"))


@dataclass(frozen=True)
class ActorTokenLimits:
    max_prompt_tokens: int
    generation_tokens: int
    context_tokens: int

    def __post_init__(self) -> None:
        for item in fields(self):
            value = getattr(self, item.name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{item.name} must be an explicit positive integer")
        if self.generation_tokens >= self.context_tokens:
            raise ValueError("generation reservation must leave room for a prompt")


class RequestBudgetError(ValueError):
    def __init__(self, stage: str, reason: str, prompt_tokens: int, limits: ActorTokenLimits):
        self.stage, self.reason = stage, reason
        self.prompt_tokens, self.limits = prompt_tokens, limits
        super().__init__(f"{stage}: {reason}; prompt={prompt_tokens}, generation={limits.generation_tokens}, "
                         f"context={limits.context_tokens}, max_prompt={limits.max_prompt_tokens}")


@dataclass(frozen=True)
class PreparedActorRequestV1:
    stage: str
    messages: tuple[Mapping[str, str], ...]
    chat_text: str
    input_ids: tuple[int, ...]
    payload_json: str
    limits: ActorTokenLimits
    version: str = REQUEST_VERSION

    @property
    def prompt_tokens(self) -> int:
        return len(self.input_ids)

    @property
    def payload(self) -> dict[str, Any]:
        return json.loads(self.payload_json)

    def to_payload(self) -> dict[str, Any]:
        return {"version": self.version, "stage": self.stage,
                "messages": [dict(message) for message in self.messages],
                "chat_text": self.chat_text, "input_ids": list(self.input_ids),
                "prompt_tokens": self.prompt_tokens, "limits": json_safe(self.limits),
                "payload": self.payload, "truncation": False,
                "generation_reservation_is_declared_not_observed": True}


def _object(properties: dict[str, Any]) -> dict[str, Any]:
    return {"type": "object", "properties": properties, "required": list(properties),
            "additionalProperties": False}


def _text(maximum: int = 2048) -> dict[str, Any]:
    return {"type": "string", "minLength": 1, "maxLength": maximum}


def _array(items: dict[str, Any], *, minimum: int = 0, maximum: int | None = None) -> dict[str, Any]:
    result = {"type": "array", "items": items, "minItems": minimum}
    if maximum is not None:
        result["maxItems"] = maximum
    return result


def _t0_schema(horizon: int) -> dict[str, Any]:
    text = _text(t0_contract._MAX_TEXT)
    evidence = _object({"evidence_id": text, "kind": {"enum": sorted(t0_contract._KINDS)},
                        "as_of": text, "quote_or_feature": text})
    window = _object({"start": {"type": "integer", "minimum": 0, "maximum": horizon - 1},
                      "end": {"type": "integer", "minimum": 1, "maximum": horizon},
                      "effect": {"enum": sorted(t0_contract._EFFECTS)}})
    interaction = _object({"with_driver_id": _text(128), "description": text})
    driver = _object({"id": _text(128), "name": text,
                      "evidence": _array(evidence, minimum=1, maximum=t0_contract._MAX_EVIDENCE_PER_DRIVER),
                      "mechanism": text, "relevance": {"enum": sorted(t0_contract._RELEVANCE)},
                      "impact_windows": _array(window, minimum=1, maximum=t0_contract._MAX_WINDOWS_PER_DRIVER),
                      "interactions": _array(interaction, maximum=t0_contract._MAX_INTERACTIONS_PER_DRIVER),
                      "uncertainty": text})
    return _object({"schema_version": {"const": "t0.v1"},
                    "drivers": _array(driver, maximum=t0_contract._MAX_DRIVERS),
                    "global_uncertainty": text, "evidence_ids_used": _array(_text())})


def _span_schema(horizon: int) -> dict[str, Any]:
    return _object({"start": {"type": "integer", "minimum": 0, "maximum": horizon - 1},
                    "stop": {"type": "integer", "minimum": 1, "maximum": horizon}})


def _evidence_manifest(observation: ForecastObservation) -> list[dict[str, Any]]:
    result = []
    for record in observation.provenance:
        kind = "covariate" if record.field.startswith("known_future:") else "context"
        if "calendar" in record.field:
            kind = "calendar"
        elif record.field.startswith("history:"):
            kind = "history"
        result.append({"evidence_id": record.field, "kind": kind,
                       "as_of": json_safe(record.available_at), "source": record.source,
                       "version": record.version, "publish_time": json_safe(record.publish_time)})
    return result


class CanonicalActorRequestBuilderV1:
    """Build exact requests; limits are required for every stage, without defaults."""

    def __init__(self, tokenizer: Any, *, limits: Mapping[str, ActorTokenLimits]) -> None:
        if set(limits) != set(STAGES) or any(not isinstance(item, ActorTokenLimits) for item in limits.values()):
            raise ValueError("provide explicit ActorTokenLimits for exactly T0/T1/T2/T3")
        if not has_chat_template(tokenizer):
            raise ValueError("a tokenizer with the actual shared chat template is required")
        self.tokenizer = tokenizer
        self.limits = MappingProxyType(dict(limits))

    def _prepare(self, stage: str, inputs: Any, schema: Any, constraints: str) -> PreparedActorRequestV1:
        payload = {"request_version": REQUEST_VERSION, "stage": stage,
                   "inputs": json_safe(inputs), "response_schema": schema,
                   "response_constraints": constraints}
        payload_json = _json(payload)
        messages = [
            {"role": "system", "content": f"You are the {stage} forecasting callback. "
             "Use only the supplied issue-time input. Return exactly one JSON object matching "
             "response_schema and response_constraints, without fences, hidden reasoning or extra keys. "
             "Input clock_basis and provenance are source claims; do not relabel native coordinates as UTC."},
            {"role": "user", "content": payload_json},
        ]
        template_kwargs = {}
        chat_template = getattr(self.tokenizer, "chat_template", "") or ""
        if "enable_thinking" in chat_template and stage in ("T1", "T2", "T3"):
            template_kwargs["enable_thinking"] = False
        chat_text = render_chat_template_text(self.tokenizer, messages, add_generation_prompt=True,
                                              template_kwargs=template_kwargs or None)
        ids = render_chat_template_ids(self.tokenizer, messages, add_generation_prompt=True,
                                       template_kwargs=template_kwargs or None)
        complete_ids = self.tokenizer(chat_text, add_special_tokens=False, truncation=False,
                                      max_length=None)["input_ids"]
        if list(complete_ids) != ids:
            raise ValueError("chat-template IDs disagree with complete untruncated chat text")
        limits = self.limits[stage]
        if len(ids) > limits.max_prompt_tokens:
            raise RequestBudgetError(stage, "prompt_limit_exceeded_no_truncation", len(ids), limits)
        if len(ids) + limits.generation_tokens > limits.context_tokens:
            raise RequestBudgetError(stage, "prompt_plus_generation_exceeds_context", len(ids), limits)
        return PreparedActorRequestV1(stage, tuple(MappingProxyType(dict(m)) for m in messages),
                                      chat_text, tuple(ids), payload_json, limits)

    def t0(self, observation: ForecastObservation) -> PreparedActorRequestV1:
        if not isinstance(observation, ForecastObservation) or observation.issue_time is None:
            raise ValueError("T0 requires ForecastObservation with its actual issue_time")
        evidence = _evidence_manifest(observation)
        # Reuse the strict parser to check manifest clocks before fixture/provider work.
        t0_contract.parse_t0_workspace(
            {"schema_version": "t0.v1", "drivers": [], "global_uncertainty": "Request-only validation.",
             "evidence_ids_used": []}, horizon=observation.horizon,
            evidence_manifest=evidence, issue_time=json_safe(observation.issue_time))
        return self._prepare("T0", {"observation": observation, "evidence_manifest": evidence},
                             _t0_schema(observation.horizon),
                             "Driver IDs must be unique; referenced evidence kind/as_of must match the manifest. "
                             "Evidence must be available at issue_time; evidence_ids_used must equal referenced IDs. "
                             "Every impact window satisfies 0 <= start < end <= horizon; interactions reference other drivers.")

    def t1(self, observation: ForecastObservation, workspace: ContextWorkspace) -> PreparedActorRequestV1:
        if not isinstance(observation, ForecastObservation) or not isinstance(workspace, ContextWorkspace):
            raise TypeError("T1 requires ForecastObservation and actual ContextWorkspace")
        schema = _object({"mode": {"enum": ["DIRECT", "FACTORIZE"]},
                          "spans": _array(_span_schema(observation.horizon), minimum=1), "version": _text()})
        return self._prepare("T1", {"observation": observation, "workspace": workspace}, schema,
                             "Spans are contiguous, non-overlapping and cover [0,horizon). "
                             "DIRECT has one span; FACTORIZE has at least two. No forecast or segmentation evidence from labels.")

    def t2(self, request: SegmentRequest) -> PreparedActorRequestV1:
        if not isinstance(request, SegmentRequest):
            raise TypeError("T2 requires actual SegmentRequest")
        schema = _object({"segment_id": {"const": request.segment_id},
                          "span": {"const": json_safe(request.span)},
                          "values": _array({"type": "number"}, minimum=request.span.length, maximum=request.span.length),
                          "cost": {"type": "integer", "minimum": 0},
                          "metadata": {"type": "object"},
                          "replaces_hash": {"const": request.previous_draft_hash}})
        return self._prepare("T2", {"segment_request": request}, schema,
                             "Values are finite and ordered at the exact segment timestamps. "
                             "Identity/span match the request; replaces_hash matches its existing previous_draft_hash. "
                             "Cost is declared executor cost, not tokens; metadata contains no labels.")

    def t3(self, state: GlobalSchedulerState) -> PreparedActorRequestV1:
        if not isinstance(state, GlobalSchedulerState):
            raise TypeError("T3 requires actual GlobalSchedulerState")
        enabled = [segment.segment_id for segment in state.segments
                   if state.action_mask.get("REFINE:" + segment.segment_id, False)]
        choices = [_object({"action": {"const": "STOP"}, "segment_id": {"type": "null"}})]
        if enabled:
            choices.append(_object({"action": {"const": "REFINE"}, "segment_id": {"enum": enabled}}))
        return self._prepare("T3", {"scheduler_state": state.to_payload()}, {"oneOf": choices},
                             "Select exactly one enabled action. STOP has null segment_id. "
                             "REFINE targets exactly one enabled segment. Existing identity hashes are copied, not recreated. "
                             "The bounded policy snapshot deliberately excludes runtime-internal full numeric drafts.")

    def parse_response(self, prepared: PreparedActorRequestV1, text: str) -> tuple[Any, int]:
        if not isinstance(prepared, PreparedActorRequestV1):
            raise TypeError("response parsing requires its prepared request")
        if not isinstance(text, str):
            raise TypeError("actor fixture response must be text")
        response_ids = self.tokenizer(text, add_special_tokens=False, truncation=False,
                                      max_length=None)["input_ids"]
        if len(response_ids) > prepared.limits.generation_tokens:
            raise ValueError("fixture response exceeds declared generation reservation; no truncation")
        raw = t0_contract.load_t0_json(text)
        if not isinstance(raw, dict):
            raise ValueError("actor response must be one JSON object")
        inputs = prepared.payload["inputs"]
        if prepared.stage == "T0":
            result = t0_contract.parse_t0_workspace(raw, horizon=len(inputs["observation"]["future_timestamps"]),
                evidence_manifest=inputs["evidence_manifest"], issue_time=inputs["observation"]["issue_time"])
        elif prepared.stage == "T1":
            if (set(raw) != {"mode", "spans", "version"} or not isinstance(raw["version"], str)
                    or not raw["version"].strip() or len(raw["version"]) > 2048):
                raise ValueError("T1 response keys/version do not match schema")
            if not isinstance(raw["spans"], list) or any(not isinstance(s, dict) or set(s) != {"start", "stop"} for s in raw["spans"]):
                raise ValueError("T1 spans do not match schema")
            if raw["mode"] not in {"DIRECT", "FACTORIZE"}:
                raise ValueError("T1 mode does not match schema")
            result = FactorizationPlan(raw["mode"], tuple(Span(**span) for span in raw["spans"]), raw["version"])
            result.validate(len(inputs["observation"]["future_timestamps"]))
        elif prepared.stage == "T2":
            if set(raw) != {"segment_id", "span", "values", "cost", "metadata", "replaces_hash"}:
                raise ValueError("T2 response keys do not match schema")
            request = inputs["segment_request"]
            if raw["segment_id"] != request["segment_id"] or raw["span"] != request["span"]:
                raise ValueError("T2 response identity does not match request")
            if raw["replaces_hash"] != request["previous_draft_hash"]:
                raise ValueError("T2 replaces_hash does not match existing request identity")
            if not isinstance(raw["values"], list) or any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in raw["values"]):
                raise ValueError("T2 values must be finite JSON numbers")
            if not isinstance(raw["metadata"], dict):
                raise ValueError("T2 metadata must be an object")
            json_safe(raw["metadata"])
            result = SegmentDraft(raw["segment_id"], Span(**raw["span"]), raw["values"],
                                  cost=raw["cost"], metadata=raw["metadata"], replaces_hash=raw["replaces_hash"])
        elif prepared.stage == "T3":
            if set(raw) != {"action", "segment_id"} or raw["action"] not in {"STOP", "REFINE"}:
                raise ValueError("T3 response does not match action schema")
            if raw["action"] == "REFINE" and not isinstance(raw["segment_id"], str):
                raise ValueError("T3 REFINE segment_id must be a string")
            result = GlobalAction(**raw)
            if not inputs["scheduler_state"]["action_mask"].get(result.key, False):
                raise ValueError("T3 action is not enabled")
        else:
            raise ValueError("unsupported actor stage")
        return result, len(response_ids)


@dataclass(frozen=True)
class RecordedFixtureCallV1:
    request: PreparedActorRequestV1
    fixture_response: str
    fixture_response_tokens: int

    def to_payload(self) -> dict[str, Any]:
        return {"status": "explicit_mock_fixture_not_teacher_or_training_trajectory",
                "request": self.request.to_payload(), "fixture_response": self.fixture_response,
                "fixture_response_tokens": self.fixture_response_tokens}


class MockRequestOnlyCallbacksV1:
    """Record real callback requests; responses must be explicitly supplied fixtures."""

    def __init__(self, builder: CanonicalActorRequestBuilderV1, *,
                 fixture_response: Callable[[PreparedActorRequestV1], str]) -> None:
        if not callable(fixture_response):
            raise TypeError("an explicit fixture response callable is mandatory")
        self.builder, self.fixture_response = builder, fixture_response
        self.calls: list[RecordedFixtureCallV1] = []

    def _respond(self, request: PreparedActorRequestV1) -> Any:
        text = self.fixture_response(request)
        result, count = self.builder.parse_response(request, text)
        self.calls.append(RecordedFixtureCallV1(request, text, count))
        return result

    def reasoner(self, observation: ForecastObservation) -> ContextWorkspace:
        return self._respond(self.builder.t0(observation))

    def controller(self, observation: ForecastObservation, workspace: ContextWorkspace) -> FactorizationPlan:
        return self._respond(self.builder.t1(observation, workspace))

    def executor(self, request: SegmentRequest) -> SegmentDraft:
        return self._respond(self.builder.t2(request))

    def scheduler_policy(self, state: GlobalSchedulerState) -> GlobalAction:
        return self._respond(self.builder.t3(state))

    def callbacks(self) -> dict[str, Callable[..., Any]]:
        return {"reasoner": self.reasoner, "controller": self.controller,
                "executor": self.executor, "scheduler_policy": self.scheduler_policy}
