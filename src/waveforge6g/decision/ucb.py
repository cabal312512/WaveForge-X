"""UCB1-style bounded-reward bandit with selected-action feedback only."""

import numpy as np

from .base import BaseSelector, DecisionContext, DecisionResult


class UCBSelector(BaseSelector):
    """Mean + alpha*sqrt(2 log(total+1)/count); explicit warm-up for each arm."""

    def __init__(self, alpha: float = 1.0) -> None:
        if not np.isfinite(alpha) or alpha < 0:
            raise ValueError("alpha must be finite and nonnegative")
        self.alpha = float(alpha)
        self.reset()

    def reset(self) -> None:
        self.counts = np.zeros(3)
        self.reward_sums = np.zeros(3)

    def _prepare(self, time: int) -> None:
        return None

    def select(self, context: DecisionContext) -> DecisionResult:
        self._prepare(context.time)
        means = np.divide(self.reward_sums, self.counts, out=np.full(3, -0.5), where=self.counts > 1e-12)
        means = np.clip(means, -1, 0)
        unobserved = np.flatnonzero(self.counts < 1e-8)
        bonus = self.alpha * np.sqrt(2 * np.log(max(2.0, self.counts.sum() + 1)) /
                                    np.maximum(self.counts, 1e-8))
        action = int(unobserved[0]) if len(unobserved) else int(np.argmax(means + bonus))
        forced = bool(len(unobserved) or action != int(np.argmax(means)))
        uncertainty = float(np.clip(bonus[action], 0, 1)) if not len(unobserved) else 1.0
        return DecisionResult(action, means, 1 - uncertainty, uncertainty,
                              {"forced_exploration": forced, "counts": self.counts.tolist(),
                               "exploration_bonus": bonus.tolist(), "warmup": bool(len(unobserved))})

    def update(self, context: DecisionContext, action: int, reward: float) -> None:
        super().update(context, action, reward)
        self._prepare(context.time)
        self.counts[action] += 1
        self.reward_sums[action] += reward

    def state_dict(self) -> dict:
        return {"selector": type(self).__name__, "alpha": self.alpha,
                "counts": self.counts.tolist(), "reward_sums": self.reward_sums.tolist()}
