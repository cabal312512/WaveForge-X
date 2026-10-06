"""A restricted, causal selector interface with no ground-truth handles."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np

ACTIONS = ("ofdm", "otfs", "afdm")
N_ACTIONS = len(ACTIONS)


def validate_action(action: int) -> int:
    """Validate the stable integer encoding OFDM=0, OTFS=1, AFDM=2."""
    if isinstance(action, bool) or not isinstance(action, int | np.integer):
        raise ValueError("action must be an integer in [0, 2]")
    if not 0 <= action < N_ACTIONS:
        raise ValueError("action must be an integer in [0, 2]")
    return int(action)


def validate_reward(reward: float) -> float:
    """Learner feedback is negative normalized base loss, bounded in [-1, 0]."""
    if not np.isfinite(reward) or not -1 <= reward <= 0:
        raise ValueError("learner reward must be finite and lie in [-1, 0]")
    return float(reward)


@dataclass(frozen=True, slots=True)
class DecisionContext:
    """Estimated features available at time t, never the hidden channel state.

    The runner owns feature definitions/scaling. A copied tuple and slots
    prevent mutating the context or attaching a future trajectory reference.
    No reward table, simulator, oracle, or true-state object is stored here.
    """

    features: tuple[float, ...]
    previous_action: int | None
    time: int

    def __post_init__(self) -> None:
        values = tuple(float(value) for value in self.features)
        if not values or not np.all(np.isfinite(values)):
            raise ValueError("context features must be nonempty and finite")
        if self.previous_action is not None:
            validate_action(self.previous_action)
        if isinstance(self.time, bool) or not isinstance(self.time, int | np.integer) or self.time < 0:
            raise ValueError("context time must be a nonnegative integer")
        object.__setattr__(self, "features", values)


@dataclass(frozen=True)
class DecisionResult:
    """Chosen action with predicted base rewards and explicit uncertainty.

    scores[a] is a predicted mean base reward (higher is better), not an
    exploration bonus or oracle value. Bounds/posterior samples used for
    selection belong in diagnostics. Confidence and uncertainty lie in [0,1]
    but are algorithm-specific diagnostics, not calibrated probabilities.
    """

    action: int
    scores: np.ndarray
    confidence: float
    uncertainty: float
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        validate_action(self.action)
        scores = np.asarray(self.scores, dtype=float)
        if scores.shape != (N_ACTIONS,) or not np.all(np.isfinite(scores)):
            raise ValueError("scores must contain three finite predicted rewards")
        if np.any((scores < -1) | (scores > 0)):
            raise ValueError("predicted mean reward scores must lie in [-1, 0]")
        if not 0 <= self.confidence <= 1 or not 0 <= self.uncertainty <= 1:
            raise ValueError("confidence and uncertainty must lie in [0, 1]")
        # Read-only backing bytes prevent accidental learner-state mutation.
        object.__setattr__(self, "scores", np.frombuffer(scores.tobytes(), dtype=float))
        object.__setattr__(self, "diagnostics", dict(self.diagnostics))


class BaseSelector(ABC):
    """Every deployed selector sees only current context and chosen feedback."""

    def observe(self, context: DecisionContext) -> None:
        """Optional pre-decision observation; never receives oracle information."""
        return None

    @abstractmethod
    def select(self, context: DecisionContext) -> DecisionResult:
        """Choose using current estimated context and past observed rewards."""

    def update(self, context: DecisionContext, action: int, reward: float) -> None:
        """Observe only the selected action's negative normalized base loss."""
        validate_action(action)
        validate_reward(reward)

    def reset(self) -> None:
        """Reset learned state; stateless selectors need no implementation."""
        return None

    def state_dict(self) -> dict[str, Any]:
        """Return JSON-compatible diagnostic state, excluding simulator handles."""
        return {"selector": type(self).__name__}
