"""Known transition costs, sticky decisions, and descriptive switching metrics."""

from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from .base import ACTIONS, BaseSelector, DecisionContext, DecisionResult, validate_action
from .objective import Objective


class SwitchingSelector(BaseSelector):
    """Wrap a causal learner while keeping known switching cost out of feedback.

    Under greedy, follow the learner's candidate. Sticky requires predicted
    base-reward gain > known transition cost + threshold. Uncertainty-aware
    adds alpha*uncertainty. Candidate exploration explicitly tagged by the
    learner always bypasses these filters, preventing lock-in. max_hold gives
    a second escape after repeated rejection of a different candidate.
    """

    def __init__(
        self, base: BaseSelector, objective: Objective, policy: str = "uncertainty",
        base_threshold: float = 0.01, alpha: float = 0.1, max_hold: int = 20, cost_aware: bool = True,
    ) -> None:
        if policy not in ("greedy", "sticky", "uncertainty", "uncertainty_aware"):
            raise ValueError("policy must be greedy, sticky, or uncertainty")
        if not np.isfinite(base_threshold) or base_threshold < 0 or not np.isfinite(alpha) or alpha < 0:
            raise ValueError("switch thresholds must be finite and nonnegative")
        if isinstance(max_hold, bool) or not isinstance(max_hold, int) or max_hold < 1:
            raise ValueError("max_hold must be a positive integer")
        self.base, self.objective, self.policy = base, objective, policy
        self.base_threshold, self.alpha, self.max_hold = base_threshold, alpha, max_hold
        self.cost_aware = cost_aware
        self._rejections = 0

    def observe(self, context: DecisionContext) -> None:
        self.base.observe(context)

    def select(self, context: DecisionContext) -> DecisionResult:
        candidate = self.base.select(context)
        action, previous = candidate.action, context.previous_action
        diagnostics = dict(candidate.diagnostics)
        bypass = bool(diagnostics.get("forced_exploration", False))
        gain, threshold = 0.0, 0.0
        if previous is not None and action != previous and self.policy != "greedy":
            gain = float(candidate.scores[action] - candidate.scores[previous])
            threshold = (self.objective.switch_cost(previous, action) if self.cost_aware else 0) + self.base_threshold
            if self.policy in ("uncertainty", "uncertainty_aware"):
                threshold += self.alpha * candidate.uncertainty
            if gain <= threshold and not bypass:
                self._rejections += 1
                if self._rejections >= self.max_hold:
                    bypass = True
                    diagnostics["exploration_escape"] = True
                    self._rejections = 0
                else:
                    action = previous
            else:
                self._rejections = 0
        else:
            self._rejections = 0
        diagnostics.update({"switching_policy": self.policy, "candidate_action": candidate.action,
                            "predicted_gain": gain, "switching_threshold": threshold,
                            "switch_suppressed": action != candidate.action,
                            "exploration_bypass": bypass})
        return DecisionResult(action, candidate.scores, candidate.confidence, candidate.uncertainty, diagnostics)

    def update(self, context: DecisionContext, action: int, reward: float) -> None:
        self.base.update(context, action, reward)

    def reset(self) -> None:
        self.base.reset()
        self._rejections = 0

    def state_dict(self) -> dict[str, Any]:
        return {"selector": type(self).__name__, "policy": self.policy,
                "base_threshold": self.base_threshold, "alpha": self.alpha,
                "max_hold": self.max_hold, "consecutive_rejections": self._rejections,
                "cost_aware": self.cost_aware,
                "base": self.base.state_dict()}


def switching_statistics(actions: ArrayLike) -> dict[str, Any]:
    """Switch count, dwell lengths, and occupancy; epochs are the time unit.

    First/last observed dwell periods are included and may be censored by the
    experiment window. The first action is not counted as a switch.
    """
    path = np.asarray(actions)
    if path.ndim != 1:
        raise ValueError("actions must be a one-dimensional vector")
    for action in path:
        validate_action(action)
    if not len(path):
        return {"total_switch_count": 0, "switches_per_100_frames": 0.0,
                "average_dwell_time": None, "minimum_dwell_time": None,
                "occupancy": {name: 0.0 for name in ACTIONS}}
    changes = np.flatnonzero(path[1:] != path[:-1]) + 1
    dwell = np.diff(np.concatenate(([0], changes, [len(path)])))
    return {"total_switch_count": int(len(changes)),
            "switches_per_100_frames": float(100 * len(changes) / len(path)),
            "average_dwell_time": float(np.mean(dwell)),
            "minimum_dwell_time": int(np.min(dwell)),
            "occupancy": {name: float(np.mean(path == index)) for index, name in enumerate(ACTIONS)}}
