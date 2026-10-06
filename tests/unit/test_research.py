"""Seed-level inference and physical multi-seed dynamic serialization."""

import json

import numpy as np
import pandas as pd
import pytest

from waveforge6g.channels.nonstationary import SCENARIO_NAMES
from waveforge6g.config_dynamic import parse_dynamic_config
from waveforge6g.decision.base import BaseSelector, DecisionResult
from waveforge6g.experiments import research
from waveforge6g.experiments.dynamic import DynamicExperimentRunner, adaptation_statistics


class ConstantSelector(BaseSelector):
    def select(self, context):
        return DecisionResult(0, np.full(3, -0.5), 0.5, 0.5, {})


def test_seed_confidence_interval_matches_small_hand_calculation():
    result = research.confidence_summary(np.array([1.0, 2.0, 3.0]))
    assert result["mean"] == 2 and result["std"] == 1 and result["n_seeds"] == 3
    margin = 4.302652729749462 / np.sqrt(3)
    assert result["ci_low"] == pytest.approx(2 - margin)
    assert result["ci_high"] == pytest.approx(2 + margin)
    single = research.confidence_summary(np.array([2.0]))
    assert single["n_seeds"] == 1 and np.isnan(single["std"]) and np.isnan(single["ci_low"])
    with pytest.raises(ValueError):
        research.confidence_summary(np.array([1.0, np.nan]))


@pytest.mark.parametrize("scenario", SCENARIO_NAMES)
def test_each_scenario_serializes_finite_real_physical_results(scenario, tmp_path, monkeypatch):
    monkeypatch.setenv("WAVEFORGE_PROJECT_ROOT", str(tmp_path))
    config = parse_dynamic_config(
        {
            "system": {"frame_size": 16, "cp_length": 4, "subcarriers": 4, "sample_rate_hz": 32000},
            "dynamic": {"scenario": scenario, "epochs": 12, "change_interval": 3},
            "observation": {"delay_frames": 2},
        }
    )
    runner = DynamicExperimentRunner(config, 7, {"always_ofdm": ConstantSelector()})
    destination = runner.run(plot=False, use_cache=False)
    table = pd.read_csv(destination / "trajectory.csv")
    summary = json.loads((destination / "summary.json").read_text())
    assert len(table) == 12 and table.instantaneous_regret.ge(0).all()
    assert table.time.tolist() == list(range(12))
    assert table.estimated_source_time.tolist() == [max(0, time - 2) for time in range(12)]
    assert not table.change_point.iloc[0]
    assert summary["selectors"]["always_ofdm"]["n_epochs"] == 12
    assert np.isfinite(summary["selectors"]["always_ofdm"]["dynamic_regret"])
    with pytest.raises(RuntimeError, match="new DynamicExperimentRunner"):
        runner.simulate()


def test_adaptation_latency_and_censoring_use_post_change_windows():
    table = pd.DataFrame(
        {
            "time": range(10),
            "change_point": [False, False, True, False, False, False, True, False, False, False],
            "detected_change": [False, True, False, True, False, False, False, False, True, True],
            "instantaneous_regret": [0, 0, 0.3, 0.1, 0, 0, 0.4, 0.4, 0.4, 0.4],
            "objective": [0.1] * 10,
        }
    )
    result = adaptation_statistics(table, window=2, regret_threshold=0.01)
    first, second = result["episodes"]
    assert first["adaptation_latency"] == 3 and first["detection_delay"] == 1
    assert second["adaptation_latency"] is None and second["detection_delay"] == 2
    assert result["unmatched_alarms"] == 2
    table.loc[0, "change_point"] = True
    initial = adaptation_statistics(table, window=2)
    assert initial["episodes"][0]["pre_change_mean_loss"] is None
    json.dumps(initial, allow_nan=False)


def test_real_three_seed_summary_aggregates_seeds_and_preserves_pairing(tmp_path, monkeypatch):
    monkeypatch.setenv("WAVEFORGE_PROJECT_ROOT", str(tmp_path))

    class PairedRunner(DynamicExperimentRunner):
        def __init__(self, config, seed):
            super().__init__(
                config, seed, {"always_ofdm": ConstantSelector(), "same_action": ConstantSelector()}
            )

    monkeypatch.setattr(research, "DynamicExperimentRunner", PairedRunner)
    config = parse_dynamic_config(
        {
            "experiment": {"seeds": [3, 5, 8]},
            "system": {"frame_size": 16, "cp_length": 4, "subcarriers": 4},
            "dynamic": {"scenario": "abrupt_change", "epochs": 5, "change_interval": 3},
            "selectors": [
                {"name": "always_ofdm", "type": "static", "action": 0},
                {"name": "same_action", "type": "static", "action": 0},
            ],
        }
    )
    destination = research.run_research(config, plot=False, use_cache=False)
    summary = pd.read_csv(destination / "summary.csv")
    aggregate = pd.read_csv(destination / "aggregate.csv")
    curves = pd.read_csv(destination / "curves.csv")
    contrasts = pd.read_csv(destination / "paired_contrasts.csv")
    assert len(summary) == 6 and len(aggregate) == 2 and len(curves) == 10
    assert curves.cumulative_regret_n_seeds.eq(3).all()
    for selector, group in summary.groupby("selector"):
        row = aggregate.loc[aggregate.selector == selector].iloc[0]
        assert row.mean_ber_mean == pytest.approx(group.mean_ber.mean())
        assert row.dynamic_regret_std == pytest.approx(group.dynamic_regret.std(ddof=1))
    assert contrasts["mean"].eq(0).all() and contrasts["ci_low"].eq(0).all()
    assert json.loads((destination / "metadata.json").read_text())["status"] == "complete"
