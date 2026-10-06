"""Evaluation-only current-step oracle; never supplied to deployed policies."""

import numpy as np
from numpy.typing import ArrayLike

from .base import N_ACTIONS, DecisionResult
from .objective import Objective


def checked_base_losses(base_losses: ArrayLike) -> np.ndarray:
    """Validate one complete action-loss table in stable action order."""
    values = np.asarray(base_losses, dtype=float)
    if values.shape != (N_ACTIONS,) or not np.all(np.isfinite(values)):
        raise ValueError("base_losses must contain three finite values")
    if np.any((values < 0) | (values > 1)):
        raise ValueError("base_losses must lie in [0,1]")
    return values


class OracleSelector:
    """Minimize measured loss conditional on the learner's previous action.

    This is an evaluation object, deliberately not a BaseSelector. It sees
    all three paired simulator outcomes; online selectors must not receive
    it or its loss table. Its confidence=1 means the *measured* table is known,
    not that noisy finite-frame estimates equal expected link performance.
    """

    def __init__(self, objective: Objective | None = None) -> None:
        self.objective = objective if objective is not None else Objective()

    def evaluate(self, base_losses: ArrayLike, previous: int | None = None) -> DecisionResult:
        """Select argmin(base_loss[a]+cost(previous,a)), with stable tie breaking."""
        base = checked_base_losses(base_losses)
        total = base + np.array([self.objective.switch_cost(previous, action) for action in range(N_ACTIONS)])
        action = int(np.argmin(total))
        return DecisionResult(action, -base, 1.0, 0.0, {
            "oracle_loss": float(total[action]),
            "total_losses": total.tolist(),
            "conditional_previous_action": previous,
            "oracle_kind": "paired_empirical_conditional",
        })
