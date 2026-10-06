"""Small selected-feedback ridge models with hard work-unit feasibility."""

import numpy as np


def observable_features(channel, snr, n, modulation):
    power = abs(channel.gains)**2
    total = max(float(power.sum()), 1e-12)
    weights = power/total
    delay = channel.delays/max(n/8, 1)
    fd = float(weights @ abs(channel.dopplers_hz))/2400
    return np.array([1., snr/20, np.log1p(total), channel.n_paths/8,
                     fd, float(weights @ delay), n/256,
                     float(modulation == "qam16"), fd*channel.n_paths/8])


class BudgetSelector:
    """No oracle argument: choose from current observables, update one executed arm.

    Equal offline initial models for all waveform restrictions. A scheduled two-
    frame exploration block starts at epochs 0, 10 and 20; minimum dwell is two.
    No configuration fits => None (declared outage, never a free transmission).
    """

    def __init__(self, inverse, target, allowed, seed, min_dwell=2):
        self.inverse = inverse.copy()
        self.target = target.copy()
        self.allowed = np.asarray(allowed, dtype=int)
        self.rng = np.random.default_rng(seed)
        self.previous = None
        self.since = -100
        self.min_dwell = min_dwell

    def overhead(self, paths):
        a, d = self.target.shape
        # Feature extraction, all-arm prediction (inverse@target and dot), scan,
        # and a conservative rank-one inverse/target update on the selected arm.
        return float(64*paths+100+a*(2*d*d+2*d+112)+12*d*d+10*d)

    def choose(self, features, costs, budget, epoch):
        eligible = self.allowed[costs[self.allowed] <= budget]
        if not len(eligible):
            self.previous = None
            return None, "outage"
        if self.previous in eligible and epoch-self.since < self.min_dwell:
            return self.previous, "dwell"
        if epoch in (0, 10, 20):
            chosen = int(self.rng.choice(eligible))
            reason = "explore"
        else:
            predictions = np.clip(np.einsum("aij,aj,i->a", self.inverse, self.target, features), 0, 1)
            near = eligible[predictions[eligible] <= predictions[eligible].min()+.002]
            cheap = near[np.isclose(costs[near], costs[near].min(), rtol=0, atol=1e-8)]
            chosen = int(self.rng.choice(cheap))
            reason = "predict"
        if chosen != self.previous:
            self.since = epoch
        self.previous = chosen
        return chosen, reason

    def update(self, features, action, observed_ber):
        if action is None:
            return
        if not 0 <= observed_ber <= 1:
            raise ValueError("selected-action BER must lie in [0,1]")
        v = self.inverse[action] @ features
        self.inverse[action] -= np.outer(v, v)/(1+features @ v)
        self.target[action] += features*observed_ber
