"""Configuration failures must be explicit before numerical work starts."""

import copy

import pytest

from waveforge6g.config import ConfigError, load_config, parse_config


def test_defaults_are_resolved_and_hash_ignores_input_mapping_order():
    first = parse_config({"experiment": {"seed": 7}, "system": {"modulation": "qpsk"}})
    second = parse_config({"system": {"modulation": "qpsk"}, "experiment": {"seed": 7}})
    assert first.config_hash == second.config_hash
    assert first["system"]["frame_size"] == 64
    assert len(first.config_hash) == 64
    assert first.config_hash != parse_config({"experiment": {"seed": 8}}).config_hash


def test_config_does_not_retain_mutable_input_or_return_internal_lists():
    raw = {"system": {"waveforms": ["ofdm"]}, "sweep": {"snr_db": [0, 10]}}
    config = parse_config(raw)
    expected_hash = config.config_hash
    raw["system"]["waveforms"].append("otfs")
    config["sweep"]["snr_db"].append(20)
    saved = config.to_dict()
    saved["system"]["waveforms"].clear()
    assert config.config_hash == expected_hash
    assert config["system"]["waveforms"] == ["ofdm"]


@pytest.mark.parametrize(
    ("raw", "location"),
    [
        ([], "configuration"),
        ({1: {}}, "section names"),
        ({"unknown": {}}, "unknown sections"),
        ({"system": []}, "system"),
        ({"system": {1: "ofdm"}}, "field names"),
        ({"system": {"fram_size": 64}}, "unknown fields"),
        ({"experiment": {"name": "../escape"}}, "experiment.name"),
        ({"experiment": {"seed": True}}, "experiment.seed"),
        ({"experiment": {"seed": 1.0}}, "experiment.seed"),
        ({"experiment": {"seed": -1}}, "experiment.seed"),
        ({"system": {"frame_size": 64.0}}, "system.frame_size"),
        ({"system": {"cp_length": 64}}, "system.cp_length"),
        ({"system": {"subcarriers": 7}}, "system.subcarriers"),
        ({"system": {"sample_rate_hz": 0}}, "system.sample_rate_hz"),
        ({"system": {"waveforms": ["ofdm", "ofdm"]}}, "duplicates"),
        ({"system": {"modulation": "qam256"}}, "system.modulation"),
        ({"channel": {"snr_db": float("nan")}}, "channel.snr_db"),
        ({"channel": {"snr_db": float("inf")}}, "channel.snr_db"),
        ({"channel": {"normalize_power": "true"}}, "normalize_power"),
        ({"channel": {"paths": [{"power_db": 0}]}}, "delay_samples"),
        ({"channel": {"paths": [{"delay_samples": 0.5}]}}, "delay_samples"),
        ({"channel": {"paths": [{"delay_samples": 0, "gain": [1]}]}}, "gain"),
        ({"receiver": {"detectors": ["zf", "zf"]}}, "duplicates"),
        ({"receiver": {"detectors": ["one_tap_zf"]}}, "one_tap"),
        ({"receiver": {"rtol": 1}}, "receiver.rtol"),
        ({"receiver": {"dense_threshold": 2048}}, "receiver.dense_threshold"),
        ({"monte_carlo": {"max_frames": 2}}, "min_frames"),
        ({"monte_carlo": {"progress": 1}}, "progress"),
        ({"plotting": {"papr_oversampling": 4.0}}, "papr_oversampling"),
        ({"sweep": {"snr_db": []}}, "sweep.snr_db"),
        ({"sweep": {"snr_db": [0, 0]}}, "duplicates"),
        ({"sweep": {"frame_size": [32, 33]}}, "subcarriers"),
        ({"sweep": {"frame_size": [8]}}, "cp_length"),
    ],
)
def test_invalid_values_raise_field_specific_errors(raw, location):
    with pytest.raises(ConfigError, match=location):
        parse_config(raw)


@pytest.mark.parametrize("model", ["static_tdl", "doubly_selective"])
def test_prefix_must_cover_explicit_and_profile_delays(model):
    with pytest.raises(ConfigError, match="cp_length"):
        parse_config({"channel": {"model": model, "paths": [{"delay_samples": 9}]}})
    with pytest.raises(ConfigError, match="cp_length"):
        parse_config({"system": {"cp_length": 2}, "channel": {"model": model}})
    with pytest.raises(ConfigError, match="cp_length"):
        parse_config({"system": {"cp_length": 2},
                      "channel": {"model": model, "profile": "high_speed_train"}})


def test_ambiguous_channel_source_and_nonstatic_one_tap_are_rejected():
    with pytest.raises(ConfigError, match="either profile or paths"):
        parse_config({"channel": {"profile": "pedestrian", "paths": [{"delay_samples": 0}]}})
    with pytest.raises(ConfigError, match="zero Doppler"):
        parse_config({"channel": {"model": "static_tdl",
                                   "paths": [{"delay_samples": 0, "doppler_hz": 1}]}})
    with pytest.raises(ConfigError, match="zero Doppler"):
        parse_config({"system": {"waveforms": ["ofdm"]},
                      "channel": {"model": "flat_rayleigh", "max_doppler_hz": 1},
                      "receiver": {"detectors": ["one_tap_zf"]}})


def test_doppler_validation_includes_sweeps_and_profile_defaults():
    with pytest.raises(ConfigError, match="Doppler"):
        parse_config({"channel": {"model": "doubly_selective"},
                      "sweep": {"max_doppler_hz": [0, 32000]}})
    with pytest.raises(ConfigError, match="Doppler"):
        parse_config({"channel": {"model": "doubly_selective"},
                      "system": {"sample_rate_hz": 600}})


def test_supported_aliases_have_canonical_configuration_hashes():
    first = parse_config({"receiver": {"solver": "iterative"},
                          "system": {"modulation": "16qam"}})
    second = parse_config({"receiver": {"solver": "matrix_free"},
                           "system": {"modulation": "qam16"}})
    assert first.config_hash == second.config_hash


def test_yaml_round_trip_and_syntax_errors(tmp_path):
    import yaml

    raw = {"experiment": {"name": "test", "seed": 19}, "sweep": {"snr_db": [0, 10]}}
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.safe_dump(copy.deepcopy(raw)), encoding="utf-8")
    config = load_config(config_file)
    config_file.write_text(yaml.safe_dump(config.to_dict()), encoding="utf-8")
    assert load_config(config_file).config_hash == config.config_hash
    config_file.write_text("experiment: [", encoding="utf-8")
    with pytest.raises(ConfigError, match="invalid YAML"):
        load_config(config_file)


def test_ofdm_static_one_tap_and_frame_grid_validations():
    config = parse_config({"system": {"waveforms": ["ofdm"], "frame_size": 17},
                           "receiver": {"detectors": ["one_tap_zf", "one_tap_mmse"]}})
    assert config["system"]["frame_size"] == 17
    with pytest.raises(ConfigError, match="1024"):
        parse_config({"receiver": {"solver": "dense"}, "sweep": {"frame_size": [2048]}})
