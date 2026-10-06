"""Normalization and scalar-feedback boundary checks for sequential decisions."""

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from waveforge6g.decision.base import BaseSelector, DecisionContext, DecisionResult
from waveforge6g.decision.objective import Objective
from waveforge6g.decision.switching import SwitchingSelector, switching_statistics


def test_objective_normalizes_units_and_includes_all_weights():
    objective = Objective()
    metrics = {"ber": 0.1, "bler": 0.5, "papr_db": 6, "complexity_proxy": 50_000}
    assert objective.base_loss(metrics) == pytest.approx((0.1 + 0.2 * 0.5 + 0.05 * 0.5 + 0.05 * 0.5) / 1.4)
    assert objective.switch_cost(None, 1) == 0
    assert objective.switch_cost(1, 1) == 0
    assert objective.switch_cost(0, 1) == pytest.approx(0.1 / 1.4)
    largest = {"ber": 1, "bler": 1, "papr_db": 1000, "complexity_proxy": 1e20}
    assert objective.total_loss(largest, 0, 1) == pytest.approx(1)
    assert 0 <= objective.base_loss({**metrics, "papr_db": -1e-15}) <= 1


def test_asymmetric_cost_and_zero_weight_terms():
    objective = Objective({"ber_weight": 0, "bler_weight": 0, "papr_weight": 0,
                           "complexity_weight": 0, "switching_weight": 1,
                           "switching_matrix": [[0, 0.2, 1], [0.8, 0, 1], [0.4, 0.3, 0]]})
    assert objective.base_loss({}) == 0
    assert objective.switch_cost(0, 1) == 0.2
    assert objective.switch_cost(1, 0) == 0.8
    copied = objective.to_dict()
    copied["switching_matrix"][0][1] = 1
    assert objective.switch_cost(0, 1) == 0.2


@pytest.mark.parametrize("options", [
    {"ber_weight": -1}, {"papr_reference_db": 0}, {"complexity_reference": np.inf},
    {"switching_matrix": [[0, 2, 1], [1, 0, 1], [1, 1, 0]]},
    {"switching_matrix": [[1, 1, 1], [1, 0, 1], [1, 1, 0]]},
    {"unknown": 1},
])
def test_invalid_objective_configuration(options):
    with pytest.raises(ValueError):
        Objective(options)


def test_missing_or_invalid_rates_fail_clearly():
    with pytest.raises(ValueError, match="requires metric"):
        Objective().base_loss({})
    with pytest.raises(ValueError, match="ber must"):
        Objective().base_loss({"ber": 2, "bler": 0, "papr_db": 3, "complexity_proxy": 5})


def test_context_is_copied_immutable_and_has_no_truth_or_future_handle():
    source = [1.0, 2.0]
    context = DecisionContext(source, None, 0)
    source[0] = 10
    assert context.features == (1.0, 2.0)
    assert not hasattr(context, "truth") and not hasattr(context, "future")
    assert not hasattr(context, "__dict__")
    with pytest.raises(FrozenInstanceError):
        context.time = 1
    scores = np.array([-0.1, -0.2, -0.3])
    result = DecisionResult(0, scores, 0.5, 0.5)
    scores[0] = 0
    assert result.scores[0] == -0.1
    with pytest.raises(ValueError):
        result.scores.setflags(write=True)


class _Candidate(BaseSelector):
    def __init__(self, forced=False):
        self.forced = forced
        self.feedback = []

    def select(self, context):
        return DecisionResult(1, np.array([-0.2, -0.18, -0.5]), 0.1, 0.9,
                              {"forced_exploration": self.forced})

    def update(self, context, action, reward):
        super().update(context, action, reward)
        self.feedback.append((action, reward))


def test_uncertainty_switching_gates_gain_but_cannot_permanently_block_exploration():
    context = DecisionContext((1,), 0, 1)
    sticky = SwitchingSelector(_Candidate(), Objective(), max_hold=3)
    assert sticky.select(context).action == 0
    assert sticky.select(context).action == 0
    escaped = sticky.select(context)
    assert escaped.action == 1 and escaped.diagnostics["exploration_escape"]
    forced = SwitchingSelector(_Candidate(forced=True), Objective())
    assert forced.select(context).action == 1
    greedy = SwitchingSelector(_Candidate(), Objective(), policy="greedy")
    assert greedy.select(context).action == 1
    sticky.update(context, 0, -0.25)
    assert sticky.base.feedback == [(0, -0.25)]


def test_switching_statistics_use_observed_epochs_and_initial_action_is_free():
    result = switching_statistics([0, 0, 1, 1, 1, 2])
    assert result["total_switch_count"] == 2
    assert result["switches_per_100_frames"] == pytest.approx(100 / 3)
    assert result["average_dwell_time"] == 2
    assert result["minimum_dwell_time"] == 1
    assert result["occupancy"] == {"ofdm": 2 / 6, "otfs": 3 / 6, "afdm": 1 / 6}
    assert switching_statistics([])["average_dwell_time"] is None
