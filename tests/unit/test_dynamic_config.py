"""Strict parsing prevents silent changes to scientific experiment definitions."""

import copy

import pytest

from waveforge6g.config import ConfigError
from waveforge6g.config_dynamic import load_dynamic_config, parse_dynamic_config, research_variants


@pytest.mark.parametrize(
    "raw",
    [
        {"unknown": {}},
        {"system": {"sample_rate": 1}},
        {"dynamic": {"epoch": 3}},
        {"experiment": {"split": "testing"}},
        {"experiment": {"seeds": [-1]}},
        {"experiment": {"seeds": [True]}},
        {"experiment": {"seeds": [1.0]}},
        {"experiment": {"seeds": [1, 1]}},
        {"cache": {"enabled": "false"}},
        {"cache": {"enable": False}},
        {"observation": {"perfect": 1}},
        {"observation": {"snr_std": True}},
        {"observation": {"noise": 1}},
        {"observation": {"biases": {"snr": 1}}},
        {"objective": {"ber_weight": True}},
        {"objective": {"ber_weight": -1}},
        {"objective": {"unknown": 0}},
        {"objective": {"papr_reference_db": 0}},
        {"objective": {"switching_matrix": [[0, True, 1], [1, 0, 1], [1, 1, 0]]}},
        {"dynamic": {"scenario": "not_a_scenario"}},
        {"dynamic": {"epochs": True}},
        {"dynamic": {"trajectories": {"doppler": 5}}},
        {"dynamic": {"trajectories": {"correlation": 2}}},
        {"dynamic": {"trajectories": {"snr_db": {"type": "linear", "slop": 1}}}},
        {
            "dynamic": {
                "trajectories": {
                    "snr_db": {"type": "piecewise_constant", "points": [[0, 1], [0, 2]]}
                }
            }
        },
        {
            "dynamic": {
                "trajectories": {"snr_db": {"type": "smooth", "start_time": 5, "end_time": 2}}
            }
        },
        {"dynamic": {"cluster_dynamics": {"birth_probability": 1.5}}},
        {"dynamic": {"cluster_dynamics": {"birth_probability": True}}},
        {"dynamic": {"cluster_dynamics": {"unknown": 0}}},
        {"sweep": {"observation.snr_std": [1, 1]}},
        {"sweep": {"observation.error_scale": [0, 1], "observation.snr_std": [0, 1]}},
    ],
)
def test_invalid_dynamic_options_are_rejected_before_simulation(raw):
    with pytest.raises(ConfigError):
        parse_dynamic_config(raw)


def test_resolved_config_is_defensive_canonical_and_records_bandwidth():
    raw = {
        "experiment": {"seeds": [4, 8]},
        "system": {"sample_rate_hz": 32000},
        "dynamic": {"scenario": "urban_transition", "epochs": 20},
        "observation": {"perfect": True},
        "cache": {"enabled": False},
    }
    original = copy.deepcopy(raw)
    first = parse_dynamic_config(raw)
    second = parse_dynamic_config(dict(reversed(list(raw.items()))))
    assert first.config_hash == second.config_hash
    assert first["observation"]["reference_bandwidth_hz"] == 32000
    assert first["objective"]["switching_matrix"] == [[0, 1, 1], [1, 0, 1], [1, 1, 0]]
    copy_dict = first.to_dict()
    copy_dict["experiment"]["seeds"].append(999)
    assert first["experiment"]["seeds"] == [4, 8]
    assert raw == original


def test_sweep_variants_are_cartesian_and_record_absolute_and_relative_noise():
    config = parse_dynamic_config(
        {
            "observation": {"snr_std": 2, "doppler_std_hz": 4},
            "sweep": {"observation.error_scale": [0, 2], "objective.switching_weight": [0, 0.5]},
        }
    )
    variants = research_variants(config)
    assert len(variants) == 4 and len({variant.config_hash for _, variant in variants}) == 4
    for label, variant in variants:
        scale = label["observation.error_scale"]
        assert variant["observation"]["snr_std"] == 2 * scale
        assert variant["observation"]["doppler_std_hz"] == 4 * scale
        assert variant["observation"]["doppler_relative_error"] == 0.1 * scale
        assert variant["observation"]["perfect"] is (scale == 0)
        assert variant["sweep"] == {}
    assert config["observation"]["snr_std"] == 2


def test_invalid_yaml_and_zero_total_objective_rejected(tmp_path):
    path = tmp_path / "invalid.yaml"
    path.write_text("selectors: [", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_dynamic_config(path)
    objective = {f"{name}_weight": 0 for name in ("ber", "bler", "papr", "complexity", "switching")}
    with pytest.raises(ConfigError):
        parse_dynamic_config({"objective": objective})
