"""Baseline behavior and training/evaluation separation."""

import json

import numpy as np
import pandas as pd
import pytest
import yaml

from waveforge6g.decision.base import DecisionContext
from waveforge6g.decision.offline import OfflineSelector, train_offline
from waveforge6g.decision.random_selector import RandomSelector
from waveforge6g.decision.rule_based import RuleBasedSelector
from waveforge6g.decision.static_selector import StaticSelector


def test_fixed_and_random_controls_reset_reproducibly():
    context = DecisionContext((1, 0.3, 0.2, 0.1, 0), None, 0)
    for action in range(3):
        assert StaticSelector(action).select(context).action == action
    random = RandomSelector(37)
    first = [random.select(context).action for _ in range(40)]
    random.reset()
    assert first == [random.select(context).action for _ in range(40)]
    assert set(first) == {0, 1, 2}
    assert RuleBasedSelector().select(context).action == 1


def _training_directory(tmp_path, split="train"):
    directory = tmp_path / split
    directory.mkdir()
    config = {"experiment": {"split": split}, "system": {"frame_size": 32},
              "receiver": {"detector": "lmmse"}, "dynamic": {"scenario": "fixture"}, "objective": {}}
    (directory / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    trajectory, outcomes = [], []
    for time in range(30):
        x = (time - 15) / 15
        trajectory.append({"seed": 101, "time": time, "split": split,
                           "configuration_hash": "fixture", "context_features": json.dumps([1, x, x, 0, 0, 0, 1, 0, 0])})
        for action, loss in enumerate([0.1, 0.8, 0.9] if x < 0 else [0.8, 0.1, 0.9]):
            outcomes.append({"seed": 101, "time": time, "action": action, "base_loss": loss})
    pd.DataFrame(trajectory).to_csv(directory / "trajectory.csv", index=False)
    pd.DataFrame(outcomes).to_csv(directory / "oracle_outcomes.csv", index=False)
    return directory, config


def test_offline_fits_only_training_and_rejects_seed_domain_overlap(tmp_path):
    directory, config = _training_directory(tmp_path)
    path = tmp_path / "model.json"
    model = train_offline([directory], path)
    assert model["training_seeds"] == [101]
    assert model["training_accuracy"] > 0.9
    selector = OfflineSelector(path)
    negative = selector.select(DecisionContext((1, -1, -1, 0, 0), None, 0))
    positive = selector.select(DecisionContext((1, 1, 1, 0, 0), None, 0))
    assert negative.action == 0 and positive.action == 1
    assert np.sum(negative.diagnostics["probabilities"]) == pytest.approx(1)
    config["experiment"]["split"] = "test"
    with pytest.raises(ValueError, match="overlaps"):
        selector.validate_domain(config, 101)
    selector.validate_domain(config, 11)
    bad_directory, _ = _training_directory(tmp_path, "test")
    with pytest.raises(ValueError, match="split=train"):
        train_offline([bad_directory], tmp_path / "bad.json")
