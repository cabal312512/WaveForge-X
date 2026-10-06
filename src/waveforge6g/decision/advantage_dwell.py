"""Empirical advantage/dwell gate with finite paid probing; no safety theorem."""

import numpy as np

from .base import BaseSelector, DecisionResult
from .linucb import LinUCBSelector


class BlockLinUCB(BaseSelector):
    """Commit an action for a block; incorporate its feedback at block end."""

    def __init__(self, block=8, alpha=0.4):
        if not isinstance(block, int) or block < 1:
            raise ValueError("block must be positive")
        self.block = block
        self.base = LinUCBSelector(alpha=alpha)
        self.pending = []
        self.last = None

    def select(self, context):
        boundary = context.time % self.block == 0
        if boundary:
            for args in self.pending:
                self.base.update(*args)
            self.pending.clear()
            self.last = self.base.select(context)
        return DecisionResult(self.last.action, self.last.scores, self.last.confidence,
                              self.last.uncertainty, {"gate_reason": "block_start" if boundary else "block_hold",
                                                     "proposed_action": self.last.action})

    def update(self, context, action, reward):
        super().update(context, action, reward)
        self.pending.append((context, action, reward))


class AdvantageDwell(BaseSelector):
    """Joint Poisson-bootstrap ridge predictions; calibration multiplier is empirical.

    Ensemble differences retain prediction covariance. Only executed feedback
    updates models. The 3*probe block initial exploration is explicitly charged.
    H is fixed, rho a sensitivity parameter, not a learned bound on future drift.
    """

    def __init__(self, objective, seed=0, horizon=16, probe=8, multiplier=2.0, rho=0.0,
                 probe_order=(0, 1, 2)):
        if (isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 1
                or isinstance(probe, bool) or not isinstance(probe, int) or probe < 0
                or not np.isfinite(multiplier) or multiplier < 0 or not np.isfinite(rho) or rho < 0):
            raise ValueError("invalid gate parameters")
        self.objective, self.horizon, self.probe = objective, horizon, probe
        self.multiplier, self.rho = multiplier, rho
        self.rng = np.random.default_rng(seed)
        self.tie_rng = np.random.default_rng(np.random.SeedSequence([seed, 31003]))
        if sorted(probe_order) != [0, 1, 2]:
            raise ValueError("probe_order must be a permutation of three actions")
        self.probe_order = tuple(probe_order)
        self.feedback_sum = np.zeros(3)
        self.feedback_count = np.zeros(3, dtype=int)
        self.gram = np.tile(np.eye(5) * 0.1, (12, 3, 1, 1))
        self.targets = np.zeros((12, 3, 5))
        self.age = 0
        self.last_action = None

    def select(self, context):
        x = np.asarray(context.features[:5])
        predictions = np.clip(np.einsum("mad,d->ma", np.linalg.solve(self.gram, self.targets[..., None])[..., 0], x), -1, 0)
        means = predictions.mean(axis=0)
        previous = context.previous_action
        candidate = int(self.tie_rng.choice(np.flatnonzero(np.isclose(means, means.max(), atol=1e-12, rtol=0))))
        incumbent = candidate if previous is None else previous
        if self.probe and context.time == 3 * self.probe:
            empirical = self.feedback_sum / np.maximum(self.feedback_count, 1)
            eligible = self.feedback_count > 0
            best = empirical[eligible].max()
            candidate = int(self.tie_rng.choice(np.flatnonzero(eligible & np.isclose(empirical, best, atol=1e-12, rtol=0))))
        difference = predictions[:, candidate] - predictions[:, incumbent]
        gain = float(difference.mean())
        scale = max(float(difference.std(ddof=1)), 0.001)
        radius = self.multiplier * scale
        lower = gain - radius
        margin = self.horizon * lower - self.rho * self.horizon * (self.horizon - 1) / 2
        if context.time < 3 * self.probe:
            action, reason = self.probe_order[context.time // self.probe], "budgeted_probe"
        elif self.probe and context.time == 3 * self.probe:
            action, reason = candidate, "empirical_probe_commit"
        elif self.age < self.horizon and previous is not None:
            action, reason = incumbent, "dwell_hold"
        elif candidate != incumbent and margin > self.objective.switch_cost(previous, candidate):
            action, reason = candidate, "gain_exceeds_cost"
        else:
            action, reason = incumbent, "insufficient_gain"
        return DecisionResult(action, means, 0.0, min(radius, 1.0), {
            "proposed_action": candidate, "incumbent": incumbent, "gate_reason": reason,
            "gain": gain, "gain_scale": scale, "radius": radius, "gain_lower": lower,
            "dwell_age": self.age, "exploration_flag": reason == "budgeted_probe",
            "reset_type": "none", "change_alarm": False})

    def update(self, context, action, reward):
        super().update(context, action, reward)
        x = np.asarray(context.features[:5])
        weights = self.rng.poisson(1, size=12)
        self.gram[:, action] += weights[:, None, None] * np.outer(x, x)
        self.targets[:, action] += weights[:, None] * x * reward
        self.age = self.age + 1 if action == self.last_action else 1
        self.last_action = action
        self.feedback_sum[action] += reward
        self.feedback_count[action] += 1


class ExploreThenCommit(BaseSelector):
    """Same paid probe schedule, then empirical selected-feedback winner forever."""

    def __init__(self, seed=0, probe=8, probe_order=(0, 1, 2)):
        if not isinstance(probe, int) or probe < 1 or sorted(probe_order) != [0, 1, 2]:
            raise ValueError("positive probe length and action permutation required")
        self.probe, self.order = probe, tuple(probe_order)
        self.tie_rng = np.random.default_rng(np.random.SeedSequence([seed, 31003]))
        self.sums, self.counts = np.zeros(3), np.zeros(3, dtype=int)
        self.committed = None

    def select(self, context):
        means = self.sums / np.maximum(self.counts, 1)
        if context.time < 3 * self.probe:
            action = self.order[context.time // self.probe]
            reason = "budgeted_probe"
        else:
            if self.committed is None:
                eligible = self.counts > 0
                best = means[eligible].max()
                ties = np.flatnonzero(eligible & np.isclose(means, best, atol=1e-12, rtol=0))
                self.committed = int(self.tie_rng.choice(ties))
            action, reason = self.committed, "empirical_commit"
        return DecisionResult(action, means, 0, 1, {"proposed_action": action, "gate_reason": reason})

    def update(self, context, action, reward):
        super().update(context, action, reward)
        self.sums[action] += reward
        self.counts[action] += 1
