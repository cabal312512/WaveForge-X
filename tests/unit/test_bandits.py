"""Bandit state updates, actual selection, forgetting, and reproducibility."""

import numpy as np
import pytest

from waveforge6g.decision.base import DecisionContext
from waveforge6g.decision.discounted_ucb import DiscountedUCBSelector
from waveforge6g.decision.sliding_window_ucb import SlidingWindowUCBSelector
from waveforge6g.decision.thompson import ThompsonSelector
from waveforge6g.decision.ucb import UCBSelector


def _context(time):
    return DecisionContext((1.0, 0.2), None, time)


@pytest.mark.parametrize("factory", [UCBSelector, SlidingWindowUCBSelector, DiscountedUCBSelector, ThompsonSelector])
def test_selected_only_feedback_and_initial_exploration(factory):
    selector = factory()
    actions = []
    for time in range(3):
        result = selector.select(_context(time))
        actions.append(result.action)
        assert result.diagnostics["forced_exploration"]
        selector.update(_context(time), result.action, -0.2)
    assert actions == [0, 1, 2]
    with pytest.raises(ValueError, match="reward"):
        selector.update(_context(3), 0, 2)
    selector.reset()
    assert selector.select(_context(0)).action == 0


def test_stationary_ucb_learns_a_separated_best_arm():
    selector = UCBSelector()
    actions = []
    for time in range(300):
        result = selector.select(_context(time))
        actions.append(result.action)
        selector.update(_context(time), result.action, [0, -0.8, -0.9][result.action])
    assert actions.count(0) > 220
    np.testing.assert_array_equal(selector.counts, np.bincount(actions, minlength=3))


def test_sliding_window_expires_by_global_time_and_reexplores():
    selector = SlidingWindowUCBSelector(window=3)
    for time in range(3):
        selector.update(_context(time), time, -0.1 * time)
    selector.select(_context(4))
    np.testing.assert_array_equal(selector.counts, [0, 1, 1])
    assert selector.select(_context(4)).action == 0
    assert selector.select(_context(10)).diagnostics["warmup"]
    assert selector.counts.sum() == 0


def test_discounted_statistics_age_once_per_epoch():
    selector = DiscountedUCBSelector(discount=0.5)
    selector.update(_context(0), 0, -0.6)
    selector.select(_context(2))
    assert selector.counts[0] == 0.25
    assert selector.reward_sums[0] == pytest.approx(-0.15)
    selector.select(_context(2))
    assert selector.counts[0] == 0.25
    selector.update(_context(2), 1, -0.2)
    assert selector.counts[1] == 1


def test_thompson_fractional_updates_and_seed_reproduction():
    def run():
        selector = ThompsonSelector(seed=91)
        actions = []
        for time in range(50):
            action = selector.select(_context(time)).action
            actions.append(action)
            selector.update(_context(time), action, [-0.1, -0.3, -0.9][action])
        return actions, selector
    actions, selector = run()
    assert actions == run()[0]
    np.testing.assert_allclose(selector.successes + selector.failures, selector.counts + 2)
