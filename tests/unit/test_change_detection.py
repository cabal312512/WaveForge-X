"""Deterministic false-alarm/delay checks and causal reset lifecycle."""

import numpy as np
import pytest

from waveforge6g.decision import create_selector, validate_selector_spec
from waveforge6g.decision.base import DecisionContext
from waveforge6g.decision.change_detection import ChangeAwareSelector, PageHinkley
from waveforge6g.decision.objective import Objective
from waveforge6g.decision.ucb import UCBSelector


@pytest.mark.parametrize("shift", [0.3, -0.3])
def test_stationary_noise_no_false_alarm_and_abrupt_shift_short_delay(shift):
    monitor = PageHinkley(threshold=0.1, delta=0.005, min_instances=20)
    rng = np.random.default_rng(517)
    before = rng.normal(0, 0.003, 400)
    assert sum(monitor.update(value) for value in before) == 0
    alarms = [time for time, value in enumerate(rng.normal(shift, 0.003, 50)) if monitor.update(value)]
    assert alarms and alarms[0] < 5


def test_change_aware_resets_before_current_decision_without_double_observe():
    base = UCBSelector()
    model = ChangeAwareSelector(base, threshold=0.05, min_instances=3, feature_index=1)
    for time in range(6):
        context = DecisionContext((1, 0), None, time)
        model.observe(context)
        action = model.select(context).action
        model.update(context, action, -0.2)
    assert model.detector.total_samples == 6
    changed = DecisionContext((1, 1), None, 6)
    model.observe(changed)
    result = model.select(changed)
    assert result.diagnostics["detected_change"]
    assert result.diagnostics["forced_exploration"]
    assert base.counts.sum() == 0
    assert model.detection_times == [6]


def test_full_factory_has_all_components_and_ablation_flags():
    spec = {"name": "full", "type": "linucb", "window": 6, "policy": "uncertainty",
            "ignore_switching_cost": True, "change_detection": {"feature_index": 1}}
    model = create_selector(spec, seed=10, objective=Objective(), context_dimension=2)
    result = model.select(DecisionContext((1, 0), None, 0))
    assert result.action == 0
    assert model.base.cost_aware is False
    assert model.base.base.window == 6


@pytest.mark.parametrize("spec", [
    {"type": "ucb", "window": 10}, {"type": "linucb", "alpha": True},
    {"type": "ucb", "change_detection": {"typo": 1}}, {"type": "static", "action": 3},
    {"type": "linucb", "policy": "unknown"}, {"type": "ucb", "ignore_switching_cost": 1},
    {"type": "sliding_window_ucb", "window": 0},
    {"type": "ucb", "base_threshold": 0.1}, {"type": "rule", "policy": "sticky"},
])
def test_factory_rejects_silently_ignored_options(spec):
    with pytest.raises(ValueError):
        validate_selector_spec(spec)
