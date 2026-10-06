"""Disjoint ridge estimates and sliding contextual-history checks."""

import numpy as np
import pytest

from waveforge6g.decision import create_selector
from waveforge6g.decision.base import DecisionContext
from waveforge6g.decision.linucb import LinUCBSelector
from waveforge6g.decision.objective import Objective


def test_contextual_model_fits_different_optima_without_counterfactual_updates():
    model = LinUCBSelector(context_dimension=2, alpha=0, regularization=0.001)
    for time, value in enumerate(np.linspace(-1, 1, 90)):
        context = DecisionContext((1, value), None, time)
        action = time % 3
        reward = [-0.4 + 0.3 * value, -0.4 - 0.3 * value, -0.9][action]
        model.update(context, action, reward)
    assert model.select(DecisionContext((1, 1), None, 91)).action == 0
    assert model.select(DecisionContext((1, -1), None, 92)).action == 1
    np.testing.assert_array_equal(model.counts, [30, 30, 30])
    assert model.history == __import__("collections").deque()


def test_window_subtraction_matches_rebuilt_ridge_system():
    model = LinUCBSelector(context_dimension=2, window=3)
    for time in range(7):
        model.update(DecisionContext((1, time / 10), None, time), time % 3, -0.5)
    model.select(DecisionContext((1, 0.7), None, 7))
    for action in range(3):
        retained = [np.array([1, time / 10]) for time in range(4, 7) if time % 3 == action]
        expected = np.eye(2) + sum((np.outer(x, x) for x in retained), np.zeros((2, 2)))
        np.testing.assert_allclose(model.gram[action], expected, atol=1e-14)
    model.select(DecisionContext((1, 1), None, 20))
    np.testing.assert_array_equal(model.counts, [0, 0, 0])
    np.testing.assert_array_equal(model.gram, np.repeat(np.eye(2)[None], 3, axis=0))


def test_factory_context_dimension_and_bad_input():
    selector = create_selector({"type": "linucb", "window": 6}, 1, Objective(), context_dimension=2)
    assert selector.select(DecisionContext((1, 0), None, 0)).action == 0
    with pytest.raises(ValueError, match="features"):
        selector.select(DecisionContext((1,), None, 1))
