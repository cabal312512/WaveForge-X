"""Beta pseudo-posterior Thompson baseline for fractional bounded rewards.

Adding fractional success/failure counts is a transparent heuristic for
continuous rewards, not an exact Bernoulli likelihood posterior.
"""

import numpy as np

from .base import BaseSelector, DecisionContext, DecisionResult


class ThompsonSelector(BaseSelector):
    def __init__(self, seed: int = 0, prior: float = 1.0) -> None:
        if not np.isfinite(prior) or prior <= 0:
            raise ValueError("prior must be finite and positive")
        self.seed, self.prior = seed, float(prior)
        self.reset()

    def reset(self) -> None:
        self.rng = np.random.default_rng(self.seed)
        self.successes = np.full(3, self.prior)
        self.failures = np.full(3, self.prior)
        self.counts = np.zeros(3, dtype=int)

    def select(self, context: DecisionContext) -> DecisionResult:
        total = self.successes + self.failures
        means = self.successes / total - 1
        samples = self.rng.beta(self.successes, self.failures) - 1
        unobserved = np.flatnonzero(self.counts == 0)
        action = int(unobserved[0]) if len(unobserved) else int(np.argmax(samples))
        variance = self.successes * self.failures / (total**2 * (total + 1))
        uncertainty = float(np.clip(2 * np.sqrt(variance[action]), 0, 1))
        return DecisionResult(action, means, 1 - uncertainty, uncertainty,
                              {"forced_exploration": bool(len(unobserved) or action != np.argmax(means)),
                               "posterior_variances": variance.tolist(), "posterior_samples": samples.tolist(),
                               "posterior_kind": "fractional_beta_pseudo_posterior"})

    def update(self, context: DecisionContext, action: int, reward: float) -> None:
        super().update(context, action, reward)
        bounded_success = reward + 1
        self.successes[action] += bounded_success
        self.failures[action] += 1 - bounded_success
        self.counts[action] += 1

    def state_dict(self) -> dict:
        return {"selector": type(self).__name__, "successes": self.successes.tolist(),
                "failures": self.failures.tolist(), "counts": self.counts.tolist(),
                "prior": self.prior, "seed": self.seed, "rng_state": self.rng.bit_generator.state}
