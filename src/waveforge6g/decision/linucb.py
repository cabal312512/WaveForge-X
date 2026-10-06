"""Disjoint linear UCB with optional finite global-epoch history."""

from collections import deque

import numpy as np

from .base import BaseSelector, DecisionContext, DecisionResult


class LinUCBSelector(BaseSelector):
    """Per action A=lambda*I+sum(xx'), b=sum(x*r), theta=solve(A,b).

    Choose clipped predicted reward + alpha*sqrt(x' solve(A,x)). The model
    sees only estimated, normalized context. A window expires samples by
    elapsed global epoch, including epochs when an action was not selected.
    No linear realizability or stationary reward assumption is claimed for PHY.
    """

    def __init__(self, context_dimension: int = 9, alpha: float = 0.5,
                 regularization: float = 1.0, window: int | None = None) -> None:
        if isinstance(context_dimension, bool) or not isinstance(context_dimension, int) or context_dimension < 1:
            raise ValueError("context_dimension must be a positive integer")
        if not np.isfinite(alpha) or alpha < 0 or not np.isfinite(regularization) or regularization <= 0:
            raise ValueError("require alpha>=0 and regularization>0")
        if window is not None and (isinstance(window, bool) or not isinstance(window, int) or window < 3):
            raise ValueError("window must be None or an integer >=3")
        self.dimension, self.alpha = context_dimension, float(alpha)
        self.regularization, self.window = float(regularization), window
        self.reset()

    def reset(self) -> None:
        self.gram = np.repeat((self.regularization * np.eye(self.dimension))[None, :, :], 3, axis=0)
        self.targets = np.zeros((3, self.dimension))
        self.counts = np.zeros(3, dtype=int)
        self.history = deque()

    def _vector(self, context: DecisionContext) -> np.ndarray:
        vector = np.asarray(context.features, dtype=float)
        if vector.shape != (self.dimension,):
            raise ValueError(f"LinUCB expects {self.dimension} context features")
        return vector

    def _expire(self, time: int) -> None:
        if self.window is None:
            return
        while self.history and self.history[0][0] < time - self.window:
            _, action, x, reward = self.history.popleft()
            self.gram[action] -= np.outer(x, x)
            self.targets[action] -= x * reward
            self.counts[action] -= 1
            if self.counts[action] == 0:
                self.gram[action] = self.regularization * np.eye(self.dimension)
                self.targets[action] = 0

    def select(self, context: DecisionContext) -> DecisionResult:
        self._expire(context.time)
        x = self._vector(context)
        means, bonus = np.empty(3), np.empty(3)
        for action in range(3):
            solved = np.linalg.solve(self.gram[action], np.column_stack((self.targets[action], x)))
            means[action] = x @ solved[:, 0]
            bonus[action] = self.alpha * np.sqrt(max(0.0, float(x @ solved[:, 1])))
        means = np.clip(means, -1, 0)
        unobserved = np.flatnonzero(self.counts == 0)
        action = int(unobserved[0]) if len(unobserved) else int(np.argmax(means + bonus))
        uncertainty = 1.0 if len(unobserved) else float(np.clip(bonus[action], 0, 1))
        return DecisionResult(action, means, 1 - uncertainty, uncertainty,
                              {"forced_exploration": bool(len(unobserved) or action != np.argmax(means)),
                               "exploration_bonus": bonus.tolist(), "counts": self.counts.tolist(),
                               "window": self.window})

    def update(self, context: DecisionContext, action: int, reward: float) -> None:
        super().update(context, action, reward)
        self._expire(context.time)
        x = self._vector(context)
        self.gram[action] += np.outer(x, x)
        self.targets[action] += x * reward
        self.counts[action] += 1
        if self.window is not None:
            self.history.append((context.time, action, x.copy(), reward))

    def state_dict(self) -> dict:
        return {"selector": type(self).__name__, "dimension": self.dimension,
                "alpha": self.alpha, "regularization": self.regularization, "window": self.window,
                "gram": self.gram.tolist(), "targets": self.targets.tolist(),
                "counts": self.counts.tolist(), "retained_samples": len(self.history)}
