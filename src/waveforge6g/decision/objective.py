"""Bounded, dimensionless link loss and known action-transition costs."""

from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from typing import Any

import numpy as np

from .base import N_ACTIONS, validate_action

DEFAULT_OBJECTIVE = {
    "ber_weight": 1.0,
    "bler_weight": 0.2,
    "papr_weight": 0.05,
    "complexity_weight": 0.05,
    "switching_weight": 0.1,
    "papr_reference_db": 12.0,
    "complexity_reference": 100_000.0,
    "switching_matrix": [[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]],
}


class Objective:
    """Normalize every term before summing; larger loss always means worse.

    The common denominator includes all five weights, including switching.
    Runtime is deliberately absent: complexity_proxy is a deterministic
    operation/work estimate so hardware scheduling does not alter rewards.
    """

    def __init__(self, config: Mapping[str, Any] | Any | None = None) -> None:
        supplied = asdict(config) if is_dataclass(config) else dict(config or {})
        unknown = set(supplied) - set(DEFAULT_OBJECTIVE)
        if unknown:
            raise ValueError(f"Unknown objective options: {sorted(unknown)}")
        self.config = {**DEFAULT_OBJECTIVE, **supplied}
        names = ("ber", "bler", "papr", "complexity", "switching")
        self.weights = {name: float(self.config[f"{name}_weight"]) for name in names}
        if not all(np.isfinite(value) and value >= 0 for value in self.weights.values()):
            raise ValueError("objective weights must be finite and nonnegative")
        self.total_weight = sum(self.weights.values())
        if not np.isfinite(self.total_weight) or self.total_weight <= 0:
            raise ValueError("at least one objective weight must be positive")
        self.papr_reference_db = float(self.config["papr_reference_db"])
        self.complexity_reference = float(self.config["complexity_reference"])
        if not all(np.isfinite(value) and value > 0 for value in (
            self.papr_reference_db, self.complexity_reference
        )):
            raise ValueError("objective normalization references must be finite and positive")
        matrix = np.asarray(self.config["switching_matrix"], dtype=float)
        if matrix.shape != (N_ACTIONS, N_ACTIONS) or not np.all(np.isfinite(matrix)):
            raise ValueError("switching_matrix must be a finite 3x3 matrix")
        if np.any((matrix < 0) | (matrix > 1)) or np.any(np.diag(matrix) != 0):
            raise ValueError("switching_matrix requires entries in [0,1] and an exactly zero diagonal")
        self.switching_matrix = np.frombuffer(matrix.tobytes(), dtype=float).reshape(matrix.shape)
        self.config["switching_matrix"] = matrix.tolist()

    def base_loss(self, metrics: Mapping[str, float]) -> float:
        """Compute weighted BER/BLER/PAPR/complexity loss before transition cost.

        Required positive-weight metrics: ber, bler, papr_db, complexity_proxy.
        papr_mean_db and complexity are accepted compatibility aliases. A term
        with zero weight need not be provided. Rates outside [0,1] are errors;
        PAPR and complexity are divided by fixed references then clipped.
        """
        terms: dict[str, float] = {}
        keys = {"ber": ("ber",), "bler": ("bler",),
                "papr": ("papr_db", "papr_mean_db", "papr"),
                "complexity": ("complexity_proxy", "complexity")}
        for name, alternatives in keys.items():
            if self.weights[name] == 0:
                terms[name] = 0.0
                continue
            key = next((key for key in alternatives if key in metrics), None)
            if key is None:
                raise ValueError(f"objective requires metric {alternatives[0]!r}")
            value = float(metrics[key])
            if not np.isfinite(value):
                raise ValueError(f"objective metric {key!r} must be finite")
            if name in ("ber", "bler"):
                if not 0 <= value <= 1:
                    raise ValueError(f"{key} must lie in [0, 1]")
                terms[name] = value
            elif name == "papr":
                terms[name] = float(np.clip(value / self.papr_reference_db, 0, 1))
            else:
                if value < 0:
                    raise ValueError("complexity_proxy must be nonnegative")
                terms[name] = float(np.clip(value / self.complexity_reference, 0, 1))
        return float(sum(self.weights[name] * value for name, value in terms.items()) / self.total_weight)

    def switch_cost(self, previous: int | None, action: int) -> float:
        """Known weighted/normalized transition cost; initial action is free."""
        action = validate_action(action)
        if previous is None:
            return 0.0
        previous = validate_action(previous)
        return float(self.weights["switching"] * self.switching_matrix[previous, action] / self.total_weight)

    def total_loss(self, metrics: Mapping[str, float], previous: int | None, action: int) -> float:
        """Deployment loss = normalized base loss + known switching cost."""
        return self.base_loss(metrics) + self.switch_cost(previous, action)

    def to_dict(self) -> dict[str, Any]:
        """Return independent JSON-compatible settings for reproducibility."""
        return {**self.config, "switching_matrix": self.switching_matrix.tolist()}
