"""Runtime contracts for the context adaptive forecasting agent.

The contracts deliberately keep future labels out of the runtime observation.  A
training caller may hold :class:`TrainingTargets` separately when it computes a
reward, but it cannot accidentally pass those values to a reasoner or segment
executor through the normal request objects.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from collections.abc import Mapping as MappingABC
from datetime import datetime
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np


_RESERVED_LABEL_KEYS = frozenset(
    {
        "future_values",
        "future_target",
        "target_future",
        "target",
        "target_values",
        "actual",
        "actual_values",
        "ground_truth",
        "label",
        "labels",
        "valid_mask",
        "reward",
        "action_label",
        "branch_utility",
        "effect_mask",
        "gold_turning_point",
        "post_event_summary",
    }
)


def _is_reserved_label_key(key: Any) -> bool:
    normalized = str(key).strip().lower().replace("-", "_")
    if normalized in _RESERVED_LABEL_KEYS:
        return True
    # Metadata is often produced by adapters and may use a qualified key.
    # Reject only unambiguous label-bearing components; ordinary fields such as
    # ``future_timestamps`` remain valid runtime observations.
    return any(
        token in normalized
        for token in (
            "future_target",
            "target_values",
            "ground_truth",
            "oracle_score",
            "branch_utility",
            "effect_mask",
            "gold_turning_point",
            "post_event_summary",
        )
    )


def _reject_label_keys(value: Any, name: str) -> None:
    """Reject obvious target-bearing fields from runtime-visible mappings."""

    training_targets = globals().get("TrainingTargets")
    if training_targets is not None and isinstance(value, training_targets):
        raise ValueError(f"{name} contains a TrainingTargets object")
    if isinstance(value, MappingABC):
        for key, item in value.items():
            if _is_reserved_label_key(key):
                raise ValueError(f"{name} contains reserved training-label key: {key!r}")
            _reject_label_keys(item, name)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _reject_label_keys(item, name)


def _nonnegative_int(value: Any, name: str, *, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    value = int(value)
    if value < (1 if positive else 0):
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"{name} must be {qualifier}")
    return value


def _finite_vector(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty 1D array")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


def _freeze(value: Any) -> Any:
    """Copy nested input data into immutable containers for callback isolation."""

    if isinstance(value, np.ndarray):
        copied = np.asarray(value).copy()
        return np.frombuffer(copied.tobytes(), dtype=copied.dtype).reshape(copied.shape)
    if isinstance(value, MappingABC):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_freeze(item) for item in value)
    return value


def _timestamps(value: Sequence[Any], name: str) -> tuple[Any, ...]:
    result = tuple(value)
    if not result:
        raise ValueError(f"{name} must not be empty")
    return result


def _as_of_order(left: Any, right: Any) -> bool:
    """Compare provenance timestamps without silently accepting mixed clocks."""

    if isinstance(left, str) and isinstance(right, str):
        try:
            parse = lambda value: datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
            left, right = parse(left), parse(right)
        except ValueError as exc:
            raise ValueError("string provenance timestamps must be ISO-8601") from exc
    try:
        result = left <= right
    except TypeError:
        raise ValueError("issue_time and available_at must use comparable timestamp values")
    if not isinstance(result, (bool, np.bool_)):
        raise ValueError("issue_time and available_at must use scalar timestamp values")
    return bool(result)


@dataclass(frozen=True)
class ProvenanceRecord:
    """As-of metadata for one runtime-visible context or covariate field."""

    field: str
    source: str
    version: str
    available_at: Any
    publish_time: Any | None = None

    def __post_init__(self) -> None:
        for value, name in ((self.field, "field"), (self.source, "source"), (self.version, "version")):
            if not str(value).strip():
                raise ValueError(f"{name} must not be empty")
        if self.publish_time is not None and not _as_of_order(self.publish_time, self.available_at):
            raise ValueError("publish_time must be no later than available_at")
        object.__setattr__(self, "field", str(self.field))
        object.__setattr__(self, "source", str(self.source))
        object.__setattr__(self, "version", str(self.version))


def validate_as_of_provenance(issue_time: Any, records: Sequence[ProvenanceRecord]) -> tuple[ProvenanceRecord, ...]:
    """Validate that every runtime-visible field existed by the issue time."""

    if issue_time is None:
        raise ValueError("issue_time is required for as-of validation")
    normalized = tuple(records)
    if len({record.field for record in normalized}) != len(normalized):
        raise ValueError("provenance fields must be unique")
    for record in normalized:
        if not _as_of_order(record.available_at, issue_time):
            raise ValueError(
                f"provenance for {record.field!r} is not as-of valid: "
                f"available_at={record.available_at!r} > issue_time={issue_time!r}"
            )
    return normalized


@dataclass(frozen=True)
class ForecastObservation:
    """Information available at forecast issue time."""

    sample_id: str
    target_history: Any
    history_timestamps: Sequence[Any]
    future_timestamps: Sequence[Any]
    known_future: Mapping[str, Any] = field(default_factory=dict)
    context: Mapping[str, Any] = field(default_factory=dict)
    unit: str = ""
    issue_time: Any | None = None
    provenance: Sequence[ProvenanceRecord] = ()

    def __post_init__(self) -> None:
        history = _finite_vector(self.target_history, "target_history")
        history_times = _timestamps(self.history_timestamps, "history_timestamps")
        future_times = _timestamps(self.future_timestamps, "future_timestamps")
        _reject_label_keys(self.known_future, "known_future")
        _reject_label_keys(self.context, "context")
        provenance = tuple(self.provenance)
        if any(not isinstance(record, ProvenanceRecord) for record in provenance):
            raise TypeError("provenance entries must be ProvenanceRecord instances")
        if self.issue_time is not None:
            validate_as_of_provenance(self.issue_time, provenance)
        if len(history) != len(history_times):
            raise ValueError("target_history and history_timestamps must have equal length")
        if len(set(map(str, future_times))) != len(future_times):
            raise ValueError("future_timestamps must be unique")
        if set(map(str, history_times)).intersection(map(str, future_times)):
            raise ValueError("history_timestamps and future_timestamps must be disjoint")
        try:
            comparable = (
                not isinstance(history_times[-1], str)
                and not isinstance(future_times[0], str)
                and type(history_times[-1]) is type(future_times[0])
            )
            if not comparable:
                raise TypeError("adapter-owned timestamp representation")
            if any(left >= right for left, right in zip(history_times, history_times[1:])):
                raise ValueError("history_timestamps must be strictly increasing")
            if any(left >= right for left, right in zip(future_times, future_times[1:])):
                raise ValueError("future_timestamps must be strictly increasing")
            if history_times[-1] >= future_times[0]:
                raise ValueError("history must end before future horizon begins")
        except TypeError:
            pass
        for key, item in self.known_future.items():
            if isinstance(item, (list, tuple, np.ndarray)) and len(item) != len(future_times):
                raise ValueError(f"known_future[{key!r}] length must equal horizon")
        object.__setattr__(self, "sample_id", str(self.sample_id))
        object.__setattr__(self, "target_history", history)
        object.__setattr__(self, "history_timestamps", history_times)
        object.__setattr__(self, "future_timestamps", future_times)
        object.__setattr__(self, "known_future", _freeze(self.known_future))
        object.__setattr__(self, "context", _freeze(self.context))
        object.__setattr__(self, "provenance", provenance)

    @property
    def horizon(self) -> int:
        return len(self.future_timestamps)


@dataclass(frozen=True)
class TrainingTargets:
    """Training/evaluation labels kept outside runtime agent state."""

    future_values: Any
    valid_mask: Any | None = None

    def __post_init__(self) -> None:
        values = np.asarray(self.future_values, dtype=float)
        if values.ndim != 1 or values.size == 0:
            raise ValueError("future_values must be a non-empty 1D array")
        mask = np.ones(values.size, dtype=bool) if self.valid_mask is None else np.asarray(self.valid_mask, dtype=bool)
        if mask.ndim != 1 or len(mask) != len(values):
            raise ValueError("valid_mask must be a 1D mask matching future_values")
        if np.any(mask) and not np.all(np.isfinite(values[mask])):
            raise ValueError("valid future_values must be finite")
        values = np.frombuffer(values.tobytes(), dtype=values.dtype).reshape(values.shape)
        object.__setattr__(self, "future_values", values)
        mask = np.frombuffer(mask.tobytes(), dtype=mask.dtype).reshape(mask.shape)
        object.__setattr__(self, "valid_mask", mask)


@dataclass(frozen=True)
class ContextDriver:
    name: str
    relevance: str = "unknown"
    direction: str = "unknown"
    mechanism: str = ""
    impact_window: tuple[int, int] | None = None
    persistence: str = ""
    confidence: float | None = None
    evidence: tuple[str, ...] = ()
    impact_windows: tuple[tuple[int, int], ...] = ()
    uncertainty: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "evidence", tuple(map(str, self.evidence)))
        windows = tuple(self.impact_windows)
        if self.impact_window is not None:
            windows = (tuple(self.impact_window),) + windows
        normalized_windows = []
        for window in windows:
            if len(window) != 2:
                raise ValueError("impact windows must contain (start, stop) pairs")
            start, stop = (int(window[0]), int(window[1]))
            if start < 0 or stop <= start:
                raise ValueError("impact_window must be a valid half-open horizon span")
            normalized_windows.append((start, stop))
        # Keep the legacy singular field for callers, while exposing the full
        # t0.v1 list to span filtering and serializers.
        object.__setattr__(self, "impact_windows", tuple(dict.fromkeys(normalized_windows)))
        object.__setattr__(self, "impact_window", self.impact_windows[0] if self.impact_windows else None)
        if self.confidence is not None and not 0.0 <= float(self.confidence) <= 1.0:
            raise ValueError("confidence must be in [0, 1]")


@dataclass(frozen=True)
class ContextWorkspace:
    """Structured context interpretation; it contains no segmentation action."""

    drivers: tuple[ContextDriver, ...] = ()
    interactions: tuple[str, ...] = ()
    regime_hints: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "drivers", tuple(self.drivers))
        object.__setattr__(self, "interactions", tuple(map(str, self.interactions)))
        _reject_label_keys(self.metadata, "context workspace metadata")
        object.__setattr__(self, "regime_hints", tuple(map(str, self.regime_hints)))
        object.__setattr__(self, "metadata", _freeze(self.metadata))

    def for_span(self, start: int, stop: int) -> "ContextWorkspace":
        selected = []
        for driver in self.drivers:
            windows = driver.impact_windows
            if not windows or any(window[0] < stop and start < window[1] for window in windows):
                selected.append(driver)
        return ContextWorkspace(tuple(selected), self.interactions, self.regime_hints, self.metadata)


@dataclass(frozen=True)
class Span:
    start: int
    stop: int

    def __post_init__(self) -> None:
        if isinstance(self.start, bool) or isinstance(self.stop, bool) or not isinstance(self.start, (int, np.integer)) or not isinstance(self.stop, (int, np.integer)):
            raise TypeError("span bounds must be integers")
        object.__setattr__(self, "start", int(self.start))
        object.__setattr__(self, "stop", int(self.stop))
        if self.start < 0 or self.stop <= self.start:
            raise ValueError("span must be a non-empty half-open interval")

    @property
    def length(self) -> int:
        return self.stop - self.start


@dataclass(frozen=True)
class FactorizationPlan:
    mode: str
    spans: tuple[Span, ...]
    version: str = "v1"

    def __post_init__(self) -> None:
        mode = str(self.mode).upper()
        if mode not in {"DIRECT", "FACTORIZE"}:
            raise ValueError("mode must be DIRECT or FACTORIZE")
        spans = tuple(self.spans)
        if not spans:
            raise ValueError("at least one span is required")
        if mode == "DIRECT" and len(spans) != 1:
            raise ValueError("DIRECT requires exactly one span")
        if mode == "FACTORIZE" and len(spans) == 1:
            raise ValueError("FACTORIZE requires at least two spans")
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "spans", spans)

    @classmethod
    def direct(cls, horizon: int) -> "FactorizationPlan":
        return cls("DIRECT", (Span(0, horizon),))

    def validate(self, horizon: int, *, min_segment_length: int = 1, max_segments: int | None = None) -> None:
        if self.spans[0].start != 0 or self.spans[-1].stop != horizon:
            raise ValueError("plan must cover [0, horizon)")
        if max_segments is not None and len(self.spans) > max_segments:
            raise ValueError("plan exceeds max_segments")
        previous = 0
        for span in self.spans:
            if span.start != previous:
                raise ValueError("plan spans must be contiguous and non-overlapping")
            if span.length < min_segment_length:
                raise ValueError("plan contains a segment shorter than min_segment_length")
            previous = span.stop


@dataclass(frozen=True)
class BudgetAllocation:
    total: int
    shared: int = 0
    segment_limits: Mapping[str, int] = field(default_factory=dict)
    max_refinements: int | None = None
    max_refinements_per_segment: int | None = None

    def __post_init__(self) -> None:
        total = _nonnegative_int(self.total, "total", positive=True)
        shared = _nonnegative_int(self.shared, "shared")
        max_refinements = (
            None if self.max_refinements is None
            else _nonnegative_int(self.max_refinements, "max_refinements")
        )
        max_refinements_per_segment = (
            None if self.max_refinements_per_segment is None
            else _nonnegative_int(self.max_refinements_per_segment, "max_refinements_per_segment")
        )
        if shared > total:
            raise ValueError("invalid total/shared budget")
        limits = {str(k): _nonnegative_int(v, f"segment_limits[{k!r}]") for k, v in self.segment_limits.items()}
        if any(limit > total - shared for limit in limits.values()):
            raise ValueError("a segment limit cannot exceed the global budget after shared cost")
        object.__setattr__(self, "total", total)
        object.__setattr__(self, "shared", shared)
        object.__setattr__(self, "max_refinements", max_refinements)
        object.__setattr__(self, "max_refinements_per_segment", max_refinements_per_segment)
        object.__setattr__(self, "segment_limits", _freeze(limits))


@dataclass(frozen=True)
class SegmentRequest:
    observation: ForecastObservation
    workspace: ContextWorkspace
    segment_id: str
    span: Span
    mode: str = "FORECAST"
    previous: np.ndarray | None = None
    remaining_budget: int = 0
    global_iteration: int = 0
    scheduler_version: str = "global-scheduler-v1"
    previous_draft_hash: str | None = None

    def __post_init__(self) -> None:
        if self.mode not in {"FORECAST", "REFINE"}:
            raise ValueError("segment mode must be FORECAST or REFINE")
        if self.span.stop > self.observation.horizon:
            raise ValueError("segment span exceeds observation horizon")
        _nonnegative_int(self.remaining_budget, "remaining_budget")
        _nonnegative_int(self.global_iteration, "global_iteration")
        if not str(self.scheduler_version).strip():
            raise ValueError("scheduler_version must not be empty")
        if self.previous is not None:
            if self.mode == "FORECAST":
                raise ValueError("FORECAST requests cannot include previous forecast")
            previous = _finite_vector(self.previous, "previous")
            if len(previous) != self.span.length:
                raise ValueError("previous forecast length must match segment span")
            object.__setattr__(self, "previous", previous)
        elif self.mode == "REFINE":
            raise ValueError("REFINE requests require previous forecast")
        if self.mode == "FORECAST" and self.previous_draft_hash is not None:
            raise ValueError("FORECAST requests cannot reference a previous draft")
        if self.mode == "REFINE":
            if not isinstance(self.previous_draft_hash, str) or len(self.previous_draft_hash) != 64:
                raise ValueError("REFINE requests require a SHA-256 previous_draft_hash")
        object.__setattr__(self, "scheduler_version", str(self.scheduler_version))


@dataclass(frozen=True)
class SegmentDraft:
    segment_id: str
    span: Span
    values: Any
    cost: int = 1
    metadata: Mapping[str, Any] = field(default_factory=dict)
    replaces_hash: str | None = None

    def __post_init__(self) -> None:
        values = _finite_vector(self.values, "segment forecast")
        if len(values) != self.span.length:
            raise ValueError("segment forecast length must equal span length")
        cost = _nonnegative_int(self.cost, "draft cost")
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "cost", cost)
        _reject_label_keys(self.metadata, "segment draft metadata")
        object.__setattr__(self, "metadata", _freeze(self.metadata))
        if self.replaces_hash is not None and (
            not isinstance(self.replaces_hash, str) or len(self.replaces_hash) != 64
        ):
            raise ValueError("replaces_hash must be a SHA-256 digest")


@dataclass(frozen=True)
class SegmentResult:
    segment_id: str
    span: Span
    draft: SegmentDraft
    refinements: int
    decisions: tuple[str, ...]
    initial_draft: SegmentDraft | None = None

    def __post_init__(self) -> None:
        if self.initial_draft is not None:
            if self.initial_draft.segment_id != self.segment_id or self.initial_draft.span != self.span:
                raise ValueError("initial draft identity does not match segment result")


def segment_draft_hash(draft: SegmentDraft) -> str:
    """Hash the numeric draft identity and values for replacement/replay checks."""

    digest = hashlib.sha256()
    digest.update(draft.segment_id.encode("utf-8"))
    digest.update(b"\0")
    digest.update(f"{draft.span.start}:{draft.span.stop}".encode("ascii"))
    digest.update(b"\0")
    executor_version = str(draft.metadata.get("executor_version", ""))
    digest.update(executor_version.encode("utf-8"))
    digest.update(b"\0")
    digest.update(np.asarray(draft.values, dtype="<f8").tobytes(order="C"))
    return digest.hexdigest()


@dataclass(frozen=True)
class SegmentSnapshot:
    """Immutable T3 view of one segment, including its current draft."""

    segment_id: str
    span: Span
    draft: SegmentDraft
    workspace: ContextWorkspace
    refinements: int
    segment_budget_remaining: int
    draft_hash: str = ""

    def __post_init__(self) -> None:
        if self.draft.segment_id != self.segment_id or self.draft.span != self.span:
            raise ValueError("segment snapshot draft identity does not match")
        _nonnegative_int(self.refinements, "refinements")
        _nonnegative_int(self.segment_budget_remaining, "segment_budget_remaining")
        expected = segment_draft_hash(self.draft)
        if self.draft_hash and self.draft_hash != expected:
            raise ValueError("segment snapshot draft_hash does not match its draft")
        object.__setattr__(self, "draft_hash", expected)


@dataclass(frozen=True)
class SeamMetadata:
    left_segment_id: str
    right_segment_id: str
    left_value: float
    right_value: float
    absolute_jump: float

    def __post_init__(self) -> None:
        values = (float(self.left_value), float(self.right_value), float(self.absolute_jump))
        if not all(np.isfinite(value) for value in values) or values[2] < 0.0:
            raise ValueError("seam values must be finite and absolute_jump non-negative")
        if not np.isclose(values[2], abs(values[0] - values[1]), rtol=0.0, atol=1e-12):
            raise ValueError("absolute_jump must equal the boundary value difference")
        object.__setattr__(self, "left_value", values[0])
        object.__setattr__(self, "right_value", values[1])
        object.__setattr__(self, "absolute_jump", values[2])


@dataclass(frozen=True)
class GlobalAction:
    """One global T3 decision: stop the episode or replace one segment."""

    action: str
    segment_id: str | None = None

    def __post_init__(self) -> None:
        action = str(self.action).upper()
        if action not in {"STOP", "REFINE"}:
            raise ValueError("global action must be STOP or REFINE")
        if action == "STOP" and self.segment_id is not None:
            raise ValueError("STOP cannot name a segment")
        if action == "REFINE" and not str(self.segment_id or "").strip():
            raise ValueError("REFINE requires a segment_id")
        object.__setattr__(self, "action", action)
        if self.segment_id is not None:
            object.__setattr__(self, "segment_id", str(self.segment_id))

    @classmethod
    def stop(cls) -> "GlobalAction":
        return cls("STOP")

    @classmethod
    def refine(cls, segment_id: str) -> "GlobalAction":
        return cls("REFINE", segment_id)

    @property
    def key(self) -> str:
        return "STOP" if self.action == "STOP" else f"REFINE:{self.segment_id}"


@dataclass(frozen=True)
class GlobalSchedulerState:
    """Label-free, JSON-round-trippable snapshot exposed to the global T3 policy."""

    iteration: int
    remaining_budget: int
    scheduler_version: str
    action_mask: Mapping[str, bool]
    segments: tuple[SegmentSnapshot, ...]
    seams: tuple[SeamMetadata, ...] = ()

    def __post_init__(self) -> None:
        _nonnegative_int(self.iteration, "iteration")
        _nonnegative_int(self.remaining_budget, "remaining_budget")
        if not str(self.scheduler_version).strip():
            raise ValueError("scheduler_version must not be empty")
        segments = tuple(self.segments)
        if not segments or any(not isinstance(item, SegmentSnapshot) for item in segments):
            raise ValueError("scheduler state requires SegmentSnapshot entries")
        ids = [item.segment_id for item in segments]
        if len(set(ids)) != len(ids):
            raise ValueError("scheduler segment ids must be unique")
        mask = {str(key): bool(value) for key, value in self.action_mask.items()}
        expected = {"STOP", *(f"REFINE:{segment_id}" for segment_id in ids)}
        if set(mask) != expected or not mask["STOP"]:
            raise ValueError("action_mask must contain STOP and every segment's REFINE action")
        if any(not isinstance(item, SeamMetadata) for item in self.seams):
            raise TypeError("seams must contain SeamMetadata entries")
        object.__setattr__(self, "segments", segments)
        object.__setattr__(self, "seams", tuple(self.seams))
        object.__setattr__(self, "action_mask", _freeze(mask))
        object.__setattr__(self, "scheduler_version", str(self.scheduler_version))

    @property
    def snapshot_hash(self) -> str:
        encoded = json.dumps(self.to_payload(), sort_keys=True, separators=(",", ":"), allow_nan=False)
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def to_payload(self, *, include_drafts: bool = False) -> dict[str, Any]:
        """Return a bounded policy-visible snapshot.

        Full numeric drafts remain runtime-internal.  ``include_drafts=True``
        is reserved for local persistence/replay and is never the scheduler
        policy payload.  The default summary is stable and finite-size even for
        long horizons.
        """
        _reject_label_keys(self.seams, "scheduler seams")

        def draft_payload(item: SegmentSnapshot) -> dict[str, Any]:
            values = np.asarray(item.draft.values, dtype=float)
            summary = {
                "length": int(values.size),
                "first": float(values[0]),
                "last": float(values[-1]),
                "mean": float(np.mean(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }
            payload: dict[str, Any] = {
                "summary": summary,
                "cost": int(item.draft.cost),
                "executor_version": str(item.draft.metadata.get("executor_version", "")),
            }
            if include_drafts:
                payload["values"] = values.tolist()
                payload["metadata"] = dict(item.draft.metadata)
            return payload

        return {
            "schema_version": "global-scheduler-state-v1",
            "iteration": int(self.iteration),
            "remaining_budget": int(self.remaining_budget),
            "scheduler_version": self.scheduler_version,
            "action_mask": dict(self.action_mask),
            "segments": [
                {
                    "segment_id": item.segment_id,
                    "span": {"start": item.span.start, "stop": item.span.stop},
                    "draft": draft_payload(item),
                    "draft_hash": item.draft_hash,
                    "workspace": {
                        "drivers": [
                            {
                                "name": driver.name,
                                "relevance": driver.relevance,
                                "direction": driver.direction,
                                "mechanism": driver.mechanism,
                                "impact_window": driver.impact_window,
                                "impact_windows": list(driver.impact_windows),
                                "persistence": driver.persistence,
                                "confidence": driver.confidence,
                                "evidence": list(driver.evidence),
                                "uncertainty": driver.uncertainty,
                            }
                            for driver in item.workspace.drivers
                        ],
                        "interactions": list(item.workspace.interactions),
                        "regime_hints": list(item.workspace.regime_hints),
                        "metadata": dict(item.workspace.metadata),
                    },
                    "refinements": item.refinements,
                    "segment_budget_remaining": item.segment_budget_remaining,
                }
                for item in self.segments
            ],
            "seams": [
                {
                    "left_segment_id": seam.left_segment_id,
                    "right_segment_id": seam.right_segment_id,
                    "left_value": seam.left_value,
                    "right_value": seam.right_value,
                    "absolute_jump": seam.absolute_jump,
                }
                for seam in self.seams
            ],
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "GlobalSchedulerState":
        if payload.get("schema_version") != "global-scheduler-state-v1":
            raise ValueError("unsupported global scheduler state schema")
        snapshots = []
        for raw in payload["segments"]:
            workspace_raw = raw["workspace"]
            workspace = ContextWorkspace(
                drivers=tuple(ContextDriver(**driver) for driver in workspace_raw["drivers"]),
                interactions=tuple(workspace_raw["interactions"]),
                regime_hints=tuple(workspace_raw["regime_hints"]),
                metadata=workspace_raw["metadata"],
            )
            span = Span(**raw["span"])
            draft_raw = raw["draft"]
            if "values" not in draft_raw:
                raise ValueError("bounded scheduler payload cannot be reconstructed; use include_drafts=True")
            draft = SegmentDraft(
                raw["segment_id"], span, draft_raw["values"],
                cost=draft_raw["cost"], metadata=draft_raw.get("metadata", {}),
            )
            snapshots.append(
                SegmentSnapshot(
                    raw["segment_id"], span, draft, workspace,
                    raw["refinements"], raw["segment_budget_remaining"], raw["draft_hash"],
                )
            )
        seams = tuple(SeamMetadata(**item) for item in payload["seams"])
        return cls(
            iteration=payload["iteration"],
            remaining_budget=payload["remaining_budget"],
            scheduler_version=payload["scheduler_version"],
            action_mask=payload["action_mask"],
            segments=tuple(snapshots),
            seams=seams,
        )


@dataclass(frozen=True)
class MergeResult:
    forecast: np.ndarray
    segments: tuple[SegmentResult, ...]
    workspace: ContextWorkspace
    plan: FactorizationPlan
    total_cost: int
    summary: str = ""

    def __post_init__(self) -> None:
        values = _finite_vector(self.forecast, "merged forecast")
        object.__setattr__(self, "total_cost", _nonnegative_int(self.total_cost, "total_cost"))
        self.plan.validate(len(values))
        if len(self.segments) != len(self.plan.spans):
            raise ValueError("merged segments must match plan spans")
        for result, span in zip(self.segments, self.plan.spans):
            if result.span != span or result.draft.span != span:
                raise ValueError("merged segment spans must match plan order")
            if result.draft.segment_id != result.segment_id:
                raise ValueError("segment result and draft identities must match")
        expected = np.concatenate([result.draft.values for result in self.segments])
        if not np.array_equal(values, expected):
            raise ValueError("merged forecast must equal the ordered segment drafts")
        object.__setattr__(self, "forecast", values)

    @property
    def initial_forecast(self) -> np.ndarray:
        """Return the assembled pre-refinement forecast for factorization credit."""

        if any(result.initial_draft is None for result in self.segments):
            raise ValueError("all segments must preserve an initial draft")
        return np.concatenate(
            [result.initial_draft.values for result in self.segments if result.initial_draft is not None]
        )

    @property
    def initial_cost(self) -> int:
        if any(result.initial_draft is None for result in self.segments):
            raise ValueError("all segments must preserve an initial draft")
        return sum(result.initial_draft.cost for result in self.segments if result.initial_draft is not None)


@dataclass(frozen=True)
class AgentTrace:
    sample_id: str
    plan: FactorizationPlan
    decisions: tuple[Mapping[str, Any], ...]
    total_cost: int
    fallback: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)
    policy_steps: tuple["PolicyStepRecord", ...] = ()
    scheduler_version: str = ""
    termination_reason: str = ""
    global_iterations: int = 0
    scheduler_snapshots: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "decisions", tuple(_freeze(decision) for decision in self.decisions))
        object.__setattr__(self, "total_cost", _nonnegative_int(self.total_cost, "total_cost"))
        object.__setattr__(self, "metadata", _freeze(self.metadata))
        if self.scheduler_version and not str(self.scheduler_version).strip():
            raise ValueError("scheduler_version must not be blank")
        if self.termination_reason and self.termination_reason not in {
            "POLICY_STOP", "FORCED_STOP", "DEPTH_LIMIT"
        }:
            raise ValueError("invalid termination_reason")
        object.__setattr__(
            self, "global_iterations", _nonnegative_int(self.global_iterations, "global_iterations")
        )
        snapshots = tuple(self.scheduler_snapshots)
        if any(not isinstance(item, str) or len(item) != 64 for item in snapshots):
            raise ValueError("scheduler_snapshots must contain SHA-256 digests")
        object.__setattr__(self, "scheduler_snapshots", snapshots)
        steps = tuple(self.policy_steps)
        if any(not isinstance(step, PolicyStepRecord) for step in steps):
            raise TypeError("policy_steps must contain PolicyStepRecord instances")
        object.__setattr__(self, "policy_steps", steps)


@dataclass(frozen=True)
class PolicyStepRecord:
    """Replay metadata for one sampled controller or local stopping action."""

    stage: str
    action: str
    segment_id: str | None = None
    step: int = 0
    log_prob: float | None = None
    action_mask: Mapping[str, bool] = field(default_factory=dict)
    policy_version: str = ""
    prompt_hash: str = ""
    next_state: Mapping[str, Any] = field(default_factory=dict)
    sampling: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not str(self.stage).strip() or not str(self.action).strip():
            raise ValueError("policy step stage and action must not be empty")
        if isinstance(self.step, bool) or not isinstance(self.step, (int, np.integer)) or int(self.step) < 0:
            raise ValueError("policy step index must be a non-negative integer")
        if self.log_prob is not None and not np.isfinite(float(self.log_prob)):
            raise ValueError("policy step log_prob must be finite")
        if not str(self.policy_version).strip():
            raise ValueError("policy step policy_version must not be empty")
        object.__setattr__(self, "stage", str(self.stage))
        object.__setattr__(self, "action", str(self.action))
        object.__setattr__(self, "segment_id", None if self.segment_id is None else str(self.segment_id))
        object.__setattr__(self, "step", int(self.step))
        object.__setattr__(self, "action_mask", _freeze(self.action_mask))
        object.__setattr__(self, "next_state", _freeze(self.next_state))
        object.__setattr__(self, "sampling", _freeze(self.sampling))
