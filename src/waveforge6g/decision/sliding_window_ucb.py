"""UCB using only feedback from the last configured decision epochs."""

from collections import deque

from .base import DecisionContext
from .ucb import UCBSelector


class SlidingWindowUCBSelector(UCBSelector):
    """Forget by global epoch age, not by each arm's number of recent pulls."""

    def __init__(self, window: int = 20, alpha: float = 1.0) -> None:
        if isinstance(window, bool) or not isinstance(window, int) or window < 3:
            raise ValueError("window must be an integer >=3")
        self.window = window
        super().__init__(alpha)

    def reset(self) -> None:
        super().reset()
        self.history = deque()

    def _prepare(self, time: int) -> None:
        while self.history and self.history[0][0] < time - self.window:
            _, action, reward = self.history.popleft()
            self.counts[action] -= 1
            self.reward_sums[action] -= reward
        self.counts[self.counts < 1e-10] = 0
        self.reward_sums[self.counts == 0] = 0

    def update(self, context: DecisionContext, action: int, reward: float) -> None:
        super().update(context, action, reward)
        self.history.append((context.time, action, reward))

    def state_dict(self) -> dict:
        return {**super().state_dict(), "window": self.window, "history": list(self.history)}
