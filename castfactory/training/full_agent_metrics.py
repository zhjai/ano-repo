"""Evaluator-only future-normalized metrics and overflow-safe scalar arithmetic.

WAPE and NRMSE have no denominator floor. A zero future denominator is
undefined, including for a perfect all-zero forecast. Decimal intermediates
preserve finite binary64 inputs even when their difference, sum or square would
overflow. Report scalars outside binary64's nonzero finite range are decimal
strings; they must be parsed before arithmetic and never passed to a tracker as
infinite floats. Irrational quantities use 4096 significant decimal digits.
"""
from __future__ import annotations

from decimal import Context, Decimal, InvalidOperation, localcontext
import math
from numbers import Integral, Real
import sys
from typing import Any, Iterable, Union


METRIC_VERSION = "full_agent_future_wape_v1"
METRIC_PRECISION = 4096
MetricScalar = Union[float, str]
_ZERO = Decimal(0)
_MAX_FLOAT = Decimal.from_float(sys.float_info.max)
_MIN_POSITIVE_FLOAT = Decimal.from_float(float.fromhex("0x0.0000000000001p-1022"))


def metric_context():
    """Isolated precision for full binary64 range, differences and squares."""
    return localcontext(Context(prec=METRIC_PRECISION))


def metric_decimal(value: Any) -> Decimal:
    """Parse a finite metric, preserving the exact value of binary64 inputs."""
    if isinstance(value, bool):
        raise ValueError("Boolean is not a metric scalar")
    try:
        if isinstance(value, Decimal):
            result = value
        elif isinstance(value, Integral):
            result = Decimal(int(value))
        elif isinstance(value, Real):
            result = Decimal.from_float(float(value))
        elif isinstance(value, str) and value.strip():
            result = Decimal(value)
        else:
            raise ValueError("Expected a numeric metric scalar or decimal string")
    except (InvalidOperation, OverflowError, TypeError) as exc:
        raise ValueError("Invalid metric scalar") from exc
    if not result.is_finite():
        raise ValueError("Nonfinite metric scalar")
    return result


def serialize_metric(value: Any) -> MetricScalar:
    """Use ordinary floats when in range, otherwise retain the decimal value."""
    value = metric_decimal(value)
    magnitude = value.copy_abs()
    if magnitude > _MAX_FLOAT or (magnitude and magnitude < _MIN_POSITIVE_FLOAT):
        return str(value)
    number = float(value)
    if math.isfinite(number) and (number != 0.0 or value.is_zero()):
        return number
    # Formatting never rounds (unlike Decimal.normalize under ambient context).
    return str(value)


def metric_float(value: Any) -> float:
    """Convert a tracker-compatible scalar; reject overflow and underflow."""
    number = serialize_metric(value)
    if isinstance(number, str):
        raise ValueError("Metric requires extended decimal serialization")
    return number


def scoring_wape(metrics: dict[str, Any]) -> Decimal:
    """Read the unrounded scoring value and verify its presentation companion."""
    exact = metrics.get("wape_decimal")
    if not isinstance(exact, str) or metrics.get("normalization_status") != "ok":
        raise ValueError("Scoring requires a defined exact WAPE companion")
    value = metric_decimal(exact)
    if value < 0 or serialize_metric(value) != metrics.get("wape"):
        raise ValueError("Exact WAPE differs from its presentation value")
    return value


def evaluator_preflight(cases: Iterable[Any], *, reject_zero_denominator: bool = True,
                        declared_horizon: bool = False) -> dict[str, Any]:
    """Validate evaluator labels before loading a model or starting workers.

    Labels are evaluator-only and may be signed in every dataset domain. The
    actor's nonnegative-output contract is not a label-validity rule: published
    synthetic solar interventions include negative labels. Report their counts
    without clipping or filtering. Zero future absolute truth fails closed by
    default because WAPE and NRMSE are undefined without inventing a floor.
    """
    records = list(cases)
    zero_denominator: list[str] = []
    domains: dict[str, int] = {}
    negative_labels_by_domain: dict[str, dict[str, Any]] = {}
    for index, case in enumerate(records):
        domain = str(getattr(case, "domain", ""))
        if domain not in {"load", "price", "wind", "solar"}:
            raise ValueError("Unsupported evaluator domain: " + domain)
        values = list(getattr(getattr(case, "targets", None), "future_values", ()))
        horizon = _forecast_horizon(case, declared_horizon)
        if len(values) != horizon:
            raise ValueError(f"Evaluator labels must contain {horizon} points (case {index})")
        if any(isinstance(value, bool) or not isinstance(value, Real) for value in values):
            raise ValueError(f"Evaluator labels must be finite numeric values (case {index})")
        labels = [metric_decimal(value) for value in values]
        sample_id = str(getattr(getattr(case, "observation", None), "sample_id", index))
        negatives = [value for value in labels if value < 0]
        diagnostic = negative_labels_by_domain.setdefault(domain, {"windows": 0, "points": 0, "minimum": None})
        if negatives:
            diagnostic["windows"] += 1
            diagnostic["points"] += len(negatives)
            minimum = min(negatives)
            if diagnostic["minimum"] is None or minimum < metric_decimal(diagnostic["minimum"]):
                diagnostic["minimum"] = serialize_metric(minimum)
        if not any(labels):
            zero_denominator.append(sample_id)
        domains[domain] = domains.get(domain, 0) + 1
    report = {"cases": len(records), "domains": domains,
              "negative_labels_by_domain": negative_labels_by_domain,
              "zero_future_denominator": zero_denominator,
              "zero_future_denominator_count": len(zero_denominator),
              "metric_version": METRIC_VERSION,
              "normalization_status": "ok" if not zero_denominator else "zero_future_denominator"}
    if zero_denominator and reject_zero_denominator:
        raise ValueError("Future WAPE denominator undefined for origins: " + ", ".join(zero_denominator))
    return report


def stable_sum(values: Iterable[Any]) -> MetricScalar:
    with metric_context():
        return serialize_metric(sum((metric_decimal(v) for v in values), _ZERO))


def stable_mean(values: Iterable[Any]) -> MetricScalar:
    numbers = [metric_decimal(v) for v in values]
    if not numbers:
        raise ValueError("Cannot average an empty metric sequence")
    with metric_context():
        return serialize_metric(sum(numbers, _ZERO) / len(numbers))


def _mean_std(numbers: list[Decimal]) -> tuple[Decimal, Decimal]:
    if len(numbers) < 2:
        raise ValueError("Sample standard deviation requires at least two values")
    mean = sum(numbers, _ZERO) / len(numbers)
    variance = sum(((v - mean) ** 2 for v in numbers), _ZERO) / (len(numbers) - 1)
    return mean, variance.sqrt()


def stable_sample_std(values: Iterable[Any]) -> MetricScalar:
    with metric_context():
        _, std = _mean_std([metric_decimal(v) for v in values])
        return serialize_metric(std)


def normalized_group_advantages(values: Iterable[Any], epsilon: float = 1e-6) -> list[float]:
    """Sample-std GRPO advantages; retain the existing additive 1e-6 epsilon.

    Convert only the final bounded advantages to the policy's binary64 input.
    Tiny advantages can round to zero at that boundary; the original rewards
    and report metrics remain losslessly available in their serialized form.
    """
    numbers = [metric_decimal(v) for v in values]
    if epsilon != 1e-6:
        raise ValueError("The full-agent GRPO epsilon is fixed at 1e-6")
    with metric_context():
        mean, std = _mean_std(numbers)
        # Use the actual binary64 epsilon used by the historical implementation.
        denominator = std + metric_decimal(epsilon)
        return [float((v - mean) / denominator) for v in numbers]


def _history_scale(case: Any) -> Decimal:
    """Original legacy diagnostic scale: max(population std, .05 meanabs, 1e-6)."""
    manifest = getattr(case, "manifest", None)
    if callable(manifest):
        scale_value = manifest()["history_scale"]
        try:
            scale = metric_decimal(scale_value)
        except ValueError:
            # The legacy numpy implementation can overflow on finite histories.
            # Recompute its same formula below without redefining the diagnostic.
            pass
        else:
            if scale <= 0:
                raise ValueError("Invalid legacy history scale")
            return scale
    observation = getattr(case, "observation", None)
    history = getattr(observation, "target_history", None)
    if history is None:
        raise ValueError("Missing finite legacy history scale/history")
    numbers = [metric_decimal(v) for v in history]
    if not numbers:
        raise ValueError("Empty legacy history")
    mean = sum(numbers, _ZERO) / len(numbers)
    std = (sum(((v - mean) ** 2 for v in numbers), _ZERO) / len(numbers)).sqrt()
    mean_abs = sum((abs(v) for v in numbers), _ZERO) / len(numbers)
    return max(std, metric_decimal(0.05) * mean_abs, metric_decimal(1e-6))


def _forecast_horizon(case: Any, declared_horizon: bool) -> int:
    if not declared_horizon:
        return 96
    horizon = getattr(case, "horizon", None)
    if type(horizon) is not int or horizon not in (24, 96):
        raise ValueError("Shared evaluator requires an explicit H24/H96 horizon")
    return horizon


def forecast_metrics(values: Iterable[Any], case: Any, *, declared_horizon: bool = False) -> dict[str, Any]:
    """Score a full window, with future truth used only here.

    Legacy callers retain the 96-point contract. Shared v4 callers explicitly
    select their validated public H24/H96 geometry with ``declared_horizon``.

    Missing forecasts retain explicit unscored status. Invalid labels and
    nonnumeric/nonfinite predictions raise errors, whereas finite extreme
    predictions remain valid. ``nmae_hist`` is a legacy diagnostic, never an
    alias for WAPE and never a substitute for undefined WAPE.
    """
    truth = list(case.targets.future_values)
    horizon = _forecast_horizon(case, declared_horizon)
    if len(truth) != horizon:
        raise ValueError(f"Evaluator requires {horizon} future labels")
    target = [metric_decimal(v) for v in truth]
    values = list(values)
    scalar_names = ("wape", "wape_decimal", "nrmse", "rmse", "mae", "future_abs_mean",
                    "abs_error_sum", "abs_truth_sum", "squared_error_sum", "nmae_hist")
    if len(values) != horizon:
        return {**dict.fromkeys(scalar_names), "points": 0, "metric_version": METRIC_VERSION,
                "normalization_status": "invalid_forecast_shape", "numeric_status": "not_scored",
                "extended_decimal_fields": []}
    if any(isinstance(v, bool) or not isinstance(v, Real) for v in values):
        raise ValueError("Nonfinite/nonnumeric evaluation forecast")
    prediction = [metric_decimal(v) for v in values]
    with metric_context():
        differences = [a - b for a, b in zip(prediction, target)]
        absolute_error = sum((abs(d) for d in differences), _ZERO)
        absolute_truth = sum((abs(t) for t in target), _ZERO)
        squared_error = sum((d * d for d in differences), _ZERO)
        mean_abs = absolute_truth / horizon
        mae = absolute_error / horizon
        rmse = (squared_error / horizon).sqrt()
        raw = {"wape": absolute_error / absolute_truth if absolute_truth else None,
               "nrmse": rmse / mean_abs if absolute_truth else None,
               "rmse": rmse, "mae": mae, "future_abs_mean": mean_abs,
               "abs_error_sum": absolute_error, "abs_truth_sum": absolute_truth,
               "squared_error_sum": squared_error, "nmae_hist": mae / _history_scale(case)}
        result = {key: serialize_metric(value) if value is not None else None
                  for key, value in raw.items()}
    extended = [key for key, value in result.items() if isinstance(value, str)]
    result["wape_decimal"] = str(raw["wape"]) if raw["wape"] is not None else None
    result.update(points=horizon, metric_version=METRIC_VERSION,
                  normalization_status="ok" if absolute_truth else "zero_future_denominator",
                  numeric_status="extended_decimal" if extended else "finite_binary64",
                  extended_decimal_fields=extended)
    return result
