"""Information-boundary and physical-regret tests for the sequential runner."""

import json

import numpy as np
import pandas as pd
import pytest

from waveforge6g.config_dynamic import parse_dynamic_config
from waveforge6g.decision.base import BaseSelector, DecisionResult
from waveforge6g.decision.objective import Objective
from waveforge6g.decision.regret import hindsight_optimal_sequence
from waveforge6g.experiments.dynamic import DynamicExperimentRunner, switching_statistics


class AuditedConstant(BaseSelector):
    def __init__(self, action=0):
        self.action = action
        self.history = []

    def select(self, context):
        assert not hasattr(context, "true_state")
        assert not hasattr(context, "channel")
        assert not hasattr(context, "oracle")
        assert not hasattr(context, "trajectory")
        assert len(self.history) == context.time
        assert len(context.features) == 9
        return DecisionResult(self.action, np.full(3, -.5), 0.0, 1.0, {})

    def update(self, context, action, reward):
        super().update(context, action, reward)
        assert action == self.action
        self.history.append((context.time, reward))


def small_config():
    return parse_dynamic_config({"experiment": {"seeds": [13]},
        "dynamic": {"scenario": "abrupt_change", "epochs": 8, "change_interval": 4},
        "system": {"frame_size": 16, "subcarriers": 4, "cp_length": 4}})


def test_dynamic_causality_regret_and_counterfactual_reuse():
    config = small_config()
    policies = {"fixed_ofdm": AuditedConstant(0), "fixed_otfs": AuditedConstant(1)}
    trajectory, outcomes, summary = DynamicExperimentRunner(config, 13, policies).simulate()
    assert len(trajectory) == 16
    assert len(outcomes) == 24
    assert trajectory.groupby("time").channel_hash.nunique().eq(1).all()
    assert trajectory.instantaneous_regret.ge(0).all()
    for name, group in trajectory.groupby("selector"):
        np.testing.assert_allclose(group.cumulative_regret, group.instantaneous_regret.cumsum())
        np.testing.assert_allclose(group.learner_reward, -group.base_loss)
        assert len(policies[name].history) == 8
        assert summary["selectors"][name]["total_switch_count"] == 0
        assert summary["selectors"][name]["dynamic_regret"] >= -1e-12
    base = outcomes.pivot(index="time", columns="action", values="base_loss").to_numpy()
    optimum = hindsight_optimal_sequence(base, Objective(config["objective"]))
    assert summary["selectors"]["fixed_ofdm"]["hindsight_oracle_total_loss"] == optimum.total_loss


def test_dynamic_reproducibility_and_no_future_context():
    def run(config):
        return DynamicExperimentRunner(config, 13, {"fixed": AuditedConstant()}).simulate()[0]
    first, second = run(small_config()), run(small_config())
    pd.testing.assert_frame_equal(first.drop(columns="runtime_s"), second.drop(columns="runtime_s"))
    raw = small_config().to_dict()
    raw["dynamic"]["trajectories"] = {"snr_db": {"type": "piecewise_constant",
        "points": [[0, 12], [6, -10]]}}
    altered = run(parse_dynamic_config(raw))
    # First epochs of abrupt default are at SNR12; a change to future SNR at6
    # cannot alter contexts or selected feedback through epoch3.
    for column in ("context_features", "base_loss", "channel_hash"):
        assert first[column].iloc[:4].tolist() == altered[column].iloc[:4].tolist()


def test_dynamic_serialization(tmp_path, monkeypatch):
    monkeypatch.setenv("WAVEFORGE_PROJECT_ROOT", str(tmp_path))
    target = DynamicExperimentRunner(small_config(), 13, {"fixed": AuditedConstant()}).run(
        plot=False, use_cache=False)
    assert (target / "trajectory.csv").is_file()
    assert json.loads((target / "metadata.json").read_text())["status"] == "complete"
    assert len(pd.read_csv(target / "oracle_outcomes.csv")) == 24


def test_switching_metrics_hand_computed():
    metrics = switching_statistics([0, 0, 1, 1, 2, 0])
    assert metrics["total_switch_count"] == 3
    assert metrics["average_dwell_time"] == 1.5
    assert metrics["minimum_dwell_time"] == 1
    assert metrics["occupancy"]["ofdm"] == .5


@pytest.mark.parametrize("raw", [{"experiment": {"seeds": [1, 1]}},
                                  {"dynamic": {"epochs": 0}},
                                  {"observation": {"delay_frames": -1}}])
def test_dynamic_config_rejects_invalid(raw):
    with pytest.raises(ValueError):
        parse_dynamic_config(raw)
