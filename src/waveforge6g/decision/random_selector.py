"""Uniform randomized control with resettable private random state."""

import numpy as np

from .base import BaseSelector, DecisionContext, DecisionResult


class RandomSelector(BaseSelector):
    def __init__(self, seed: int = 0) -> None:
        self.seed = seed
        self.reset()

    def reset(self) -> None:
        self.rng = np.random.default_rng(self.seed)

    def select(self, context: DecisionContext) -> DecisionResult:
        return DecisionResult(int(self.rng.integers(3)), np.full(3, -0.5), 0, 1,
                              {"forced_exploration": True, "scores_available": False})

    def state_dict(self) -> dict:
        return {"selector": type(self).__name__, "seed": self.seed,
                "rng_state": self.rng.bit_generator.state}
