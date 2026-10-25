from __future__ import annotations

from typing import Any, Mapping

import numpy as np


class SampleScaler:
    """Per-sample scaler fitted on the observed history only.

    Sample-level scaling stabilises training and makes MSE/MAE comparable across
    domains and scales. Every scaler computes its ``center_`` and ``scale_`` from
    the history window so no future value ever influences normalization, and
    :meth:`inverse_transform` maps model-space forecasts back to raw units before
    metrics are computed.
    """

    name = "identity"
    min_scale = 1e-8

    def __init__(self) -> None:
        self.center_: float = 0.0
        self.scale_: float = 1.0
        self._fitted = False

    def fit(self, history: np.ndarray) -> "SampleScaler":
        values = self._finite_history(history)
        self.center_, self.scale_ = self._compute(values)
        scale = float(self.scale_)
        self.scale_ = scale if np.isfinite(scale) and abs(scale) >= self.min_scale else 1.0
        self._fitted = True
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        self._check_fitted()
        array = np.asarray(values, dtype=float)
        return (array - self.center_) / self.scale_

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        self._check_fitted()
        array = np.asarray(values, dtype=float)
        return array * self.scale_ + self.center_

    def fit_transform(self, history: np.ndarray) -> np.ndarray:
        return self.fit(history).transform(history)

    @property
    def params(self) -> dict[str, float]:
        return {"name": self.name, "center": float(self.center_), "scale": float(self.scale_)}

    def _compute(self, values: np.ndarray) -> tuple[float, float]:
        raise NotImplementedError

    def _finite_history(self, history: np.ndarray) -> np.ndarray:
        values = np.asarray(history, dtype=float).reshape(-1)
        if values.size == 0:
            raise ValueError("SampleScaler.fit requires a non-empty history")
        if not np.all(np.isfinite(values)):
            raise ValueError("SampleScaler.fit requires finite history values")
        return values

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError(f"{type(self).__name__} must be fit before transform")


class IdentityScaler(SampleScaler):
    name = "identity"

    def _compute(self, values: np.ndarray) -> tuple[float, float]:
        return 0.0, 1.0


class StandardScaler(SampleScaler):
    name = "standard"

    def _compute(self, values: np.ndarray) -> tuple[float, float]:
        return float(values.mean()), float(values.std())


class MeanAbsScaler(SampleScaler):
    """Scale by the mean absolute value of the history (DeepAR/Chronos style)."""

    name = "mean_abs"

    def _compute(self, values: np.ndarray) -> tuple[float, float]:
        return 0.0, float(np.mean(np.abs(values)))


class RobustScaler(SampleScaler):
    """Center on the median and scale by the inter-quartile range."""

    name = "robust"

    def _compute(self, values: np.ndarray) -> tuple[float, float]:
        median = float(np.median(values))
        q75, q25 = np.percentile(values, [75, 25])
        return median, float(q75 - q25)


_SCALERS = {
    "identity": IdentityScaler,
    "none": IdentityScaler,
    "no_op": IdentityScaler,
    "noop": IdentityScaler,
    "standard": StandardScaler,
    "mean_abs": MeanAbsScaler,
    "mean_absolute": MeanAbsScaler,
    "mean_scaling": MeanAbsScaler,
    "robust": RobustScaler,
}


def build_scaler(config: Any) -> SampleScaler:
    """Build a fresh scaler from a name or ``{"name": ...}`` mapping."""

    if config is None:
        return IdentityScaler()
    if isinstance(config, SampleScaler):
        return type(config)()
    if isinstance(config, str):
        name = config
    elif isinstance(config, Mapping):
        name = str(config.get("name", "identity"))
    else:
        raise TypeError("scaler config must be None, a string, or a mapping")
    key = str(name).strip().lower()
    if key not in _SCALERS:
        raise ValueError(
            f"Unknown scaler '{name}'. Available scalers: {sorted(set(_SCALERS))}"
        )
    return _SCALERS[key]()


def is_identity_scaler_config(config: Any) -> bool:
    if config is None:
        return True
    if isinstance(config, str):
        return config.strip().lower() in {"identity", "none", "no_op", "noop", ""}
    if isinstance(config, Mapping):
        return str(config.get("name", "identity")).strip().lower() in {
            "identity",
            "none",
            "no_op",
            "noop",
            "",
        }
    return False
