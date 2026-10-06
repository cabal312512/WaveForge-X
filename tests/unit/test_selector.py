"""Benchmark-label policy, feature scaling, model validation and region holdout."""

import json

import numpy as np
import pandas as pd
import pytest

from waveforge6g.analysis.waveform_selector import (
    NearestRegionSelector,
    benchmark_oracle,
    objective_scores,
    rule_based_selector,
    train_selector,
)


def condition_rows(point=0, snr=0, doppler=0, ber=(0.1, 0.2, 0.3)):
    return pd.DataFrame({
        "point_id": [str(point)] * 3, "waveform": ["ofdm", "otfs", "afdm"],
        "detector": ["lmmse"] * 3, "ber": ber,
        "runtime_mean_s": [0.003, 0.002, 0.001], "papr_mean_db": [6, 4, 2],
        "frame_size": [16] * 3, "bandwidth_hz": [64000] * 3,
        "snr_db": [snr] * 3, "max_doppler_hz": [doppler] * 3,
        "delay_span_s": [2 / 64000] * 3, "modulation": ["qpsk"] * 3,
        "configuration_hash": ["test-fixture"] * 3,
    })


def model():
    return NearestRegionSelector(
        [[0, 0, 0, 2], [10, 1000, 0, 2]], ["ofdm", "afdm"],
        [5, 500, 0, 2], [5, 500, 1, 1], {},
    )


def test_oracle_tie_policy_is_independent_of_table_order():
    rows = condition_rows(ber=(0.1, 0.1, 0.1))
    assert benchmark_oracle(rows.iloc[::-1]) == "ofdm"
    assert benchmark_oracle(condition_rows(ber=(0.2, 0.1, 0.3))) == "otfs"
    with pytest.raises(ValueError, match="three waveforms"):
        benchmark_oracle(rows.iloc[:2])
    rows.loc[0, "detector"] = "zf"
    with pytest.raises(ValueError, match="shared detector"):
        benchmark_oracle(rows)


def test_objective_weights_can_change_preferred_waveform():
    rows = condition_rows()
    assert benchmark_oracle(rows, "ber") == "ofdm"
    assert benchmark_oracle(rows, "utility", 1, 1) == "afdm"
    np.testing.assert_allclose(objective_scores(rows, "utility", 0, 0), rows.ber)


@pytest.mark.parametrize("value", [-1, np.nan, np.inf])
def test_invalid_penalties_and_runtime_are_rejected(value):
    rows = condition_rows()
    with pytest.raises(ValueError, match="penalty weights"):
        objective_scores(rows, "utility", value, 0)
    rows.loc[0, "runtime_mean_s"] = value
    with pytest.raises(ValueError, match="runtime"):
        objective_scores(rows, "utility")


@pytest.mark.parametrize("value", [-0.1, 1.1, np.nan, np.inf])
def test_invalid_error_rates_do_not_silently_choose_a_waveform(value):
    with pytest.raises(ValueError, match="BER"):
        benchmark_oracle(condition_rows(ber=(value, 0.1, 0.2)))


def test_nearest_region_uses_saved_feature_scale_and_survives_json(tmp_path):
    fitted = model()
    queries = np.array([[1, 100, 0, 2], [9, 900, 0, 2], [8, 100, 0, 2]])
    assert fitted.predict(queries) == ["ofdm", "afdm", "ofdm"]
    target = tmp_path / "selector.json"
    fitted.save(target)
    loaded = NearestRegionSelector.load(target)
    assert loaded.predict(queries) == fitted.predict(queries)
    assert json.loads(target.read_text(encoding="utf-8"))["labels"] == ["ofdm", "afdm"]


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [("features", [], "features"), ("features", [[0, 0, 0]], "features"),
     ("labels", ["unknown", "afdm"], "labels"), ("labels", [], "labels"),
     ("scale", [1, 0, 1, 1], "scale"), ("scale", [1, np.nan, 1, 1], "scale"),
     ("center", [0, 0], "center"), ("metadata", [], "metadata")],
)
def test_model_json_is_validated_on_load(tmp_path, field, value, message):
    from dataclasses import asdict

    data = asdict(model())
    data[field] = value
    target = tmp_path / "bad.json"
    target.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        NearestRegionSelector.load(target)


def test_region_holdout_uses_complete_conditions_and_refits_final_model(tmp_path):
    table = pd.concat([condition_rows(i, i * 2, i * 100) for i in range(6)])
    table.to_csv(tmp_path / "metrics.csv", index=False)
    fitted = train_selector([tmp_path])
    assert fitted.metadata["training_regions"] == 6
    assert fitted.metadata["validation"] == {"held_out_regions": 2, "accuracy": 1.0}
    assert len(fitted.features) == 6
    assert fitted.metadata["configuration_hashes"] == ["test-fixture"]
    rerun = tmp_path / "rerun"
    rerun.mkdir()
    table.to_csv(rerun / "metrics.csv", index=False)
    with pytest.raises(ValueError, match="duplicate operating conditions"):
        train_selector([tmp_path, rerun])


def test_mixed_numerologies_are_not_hidden_in_training_data(tmp_path):
    first, second = condition_rows(), condition_rows(1, 10, 100)
    second["frame_size"] = 32
    pd.concat([first, second]).to_csv(tmp_path / "metrics.csv", index=False)
    with pytest.raises(ValueError, match="separate selectors"):
        train_selector([tmp_path])


def test_explicit_rule_baseline_uses_normalized_doppler():
    assert rule_based_selector(10, 10, 0, 2, sample_rate_hz=64000, frame_size=64) == "ofdm"
    assert rule_based_selector(10, 200, 0, 2, sample_rate_hz=64000, frame_size=64) == "otfs"
    assert rule_based_selector(10, 1000, 0, 2, sample_rate_hz=64000, frame_size=64) == "afdm"
