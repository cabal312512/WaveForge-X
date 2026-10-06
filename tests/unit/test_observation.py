"""Upgrade C: estimate-only contexts, causal latency, and correct noise laws."""

from dataclasses import FrozenInstanceError, fields

import numpy as np
import pytest

from waveforge6g.channels.nonstationary import TrueChannelState
from waveforge6g.channels.observation import ObservationModel, ObservedContext


def _state(time, snr=12.0, doppler=800.0, delay=2e-5):
    return TrueChannelState(time, snr, doppler, delay, 3, 2, 0.9, 0.4, 1, (0.7, 0.3))


def test_public_context_contains_estimates_only_and_no_truth_reference():
    model = ObservationModel({}, 8)
    state = _state(0)
    context = model.observe(state)
    expected = {
        "time",
        "estimated_snr_db",
        "estimated_doppler_hz",
        "estimated_delay_spread_s",
        "estimated_variation",
        "estimated_coherence_time_s",
        "estimated_frequency_selectivity",
    }
    assert {field.name for field in fields(context)} == expected
    assert set(context.to_dict()) == expected
    assert all(isinstance(value, int | float) for value in context.to_dict().values())
    assert not hasattr(context, "__dict__")
    for attribute in ("snr_db", "max_doppler_hz", "path_count", "truth", "state", "future"):
        with pytest.raises(AttributeError):
            getattr(context, attribute)
    assert context.estimated_snr_db != state.snr_db
    with pytest.raises(FrozenInstanceError):
        context.estimated_snr_db = 99
    assert all(
        not isinstance(item, TrueChannelState) for snapshot in model._history for item in snapshot
    )


def test_perfect_context_has_exact_current_descriptors_even_with_other_controls():
    model = ObservationModel(
        {"perfect": True, "snr_std": 50, "delay_frames": 5, "snr_bias_db": 10}, 3
    )
    for time in range(4):
        state = _state(time, snr=time + 5, doppler=time * 40, delay=time * 1e-5)
        context = model.observe(state)
        assert context.time == time and model.last_source_time == time
        assert context.estimated_snr_db == state.snr_db
        assert context.estimated_doppler_hz == state.max_doppler_hz
        assert context.estimated_delay_spread_s == state.delay_spread_s


def test_observation_latency_uses_only_past_measurements_with_current_timestamp():
    model = ObservationModel(
        {
            "snr_std": 0,
            "doppler_relative_error": 0,
            "delay_spread_relative_error": 0,
            "delay_frames": 2,
        },
        5,
    )
    contexts = [model.observe(_state(time, snr=float(time))) for time in range(6)]
    assert [context.time for context in contexts] == list(range(6))
    assert [context.estimated_snr_db for context in contexts] == [0, 0, 0, 1, 2, 3]
    assert model.last_source_time == 3
    assert len(model._history) == 3


def test_noisy_measurement_is_sampled_once_before_delivery():
    model = ObservationModel({"delay_frames": 3, "snr_std": 2}, 88)
    contexts = [model.observe(_state(time)) for time in range(4)]
    assert len({context.estimated_snr_db for context in contexts}) == 1
    assert all(context.estimated_variation == 0 for context in contexts)


def test_noise_bias_and_relative_standard_deviation_match_configuration():
    model = ObservationModel(
        {
            "snr_std": 1.5,
            "doppler_relative_error": 0.1,
            "delay_spread_relative_error": 0.2,
            "biases": {"snr_db": 2, "doppler_hz": 10, "delay_spread_s": 1e-6},
        },
        104,
    )
    contexts = [model.observe(_state(time)) for time in range(6000)]
    snr = np.array([context.estimated_snr_db for context in contexts])
    doppler = np.array([context.estimated_doppler_hz for context in contexts])
    delay = np.array([context.estimated_delay_spread_s for context in contexts])
    assert snr.mean() == pytest.approx(14, abs=0.07)
    assert snr.std() == pytest.approx(1.5, abs=0.05)
    assert doppler.mean() == pytest.approx(810, abs=3)
    assert doppler.std() == pytest.approx(80, abs=3)
    assert delay.mean() == pytest.approx(2.1e-5, abs=2e-7)
    assert delay.std() == pytest.approx(4e-6, abs=2e-7)


def test_seed_reproducibility_and_future_truth_is_irrelevant():
    one, two = ObservationModel({}, 99), ObservationModel({}, 99)
    prefix1, prefix2 = [], []
    for time in range(5):
        prefix1.append(one.observe(_state(time)).to_dict())
        prefix2.append(two.observe(_state(time)).to_dict())
    assert prefix1 == prefix2
    one.observe(_state(5, snr=-100))
    two.observe(_state(5, snr=100))
    assert prefix1 == prefix2
    with pytest.raises(ValueError, match="sequential"):
        one.observe(_state(7))


def test_nonnegative_clipping_and_derived_context_definitions():
    model = ObservationModel(
        {
            "snr_std": 0,
            "doppler_relative_error": 0,
            "delay_spread_relative_error": 0,
            "reference_bandwidth_hz": 1000,
            "doppler_bias_hz": -2000,
            "delay_spread_bias_s": -1,
        },
        1,
    )
    context = model.observe(_state(0, snr=-3))
    assert context.estimated_snr_db == -3
    assert context.estimated_doppler_hz == context.estimated_delay_spread_s == 0
    assert context.estimated_frequency_selectivity == 0
    assert context.estimated_coherence_time_s == 1
    perfect = ObservationModel({"perfect": True, "reference_bandwidth_hz": 1000}, 1)
    context = perfect.observe(_state(0))
    assert context.estimated_coherence_time_s == pytest.approx(1 / (2 * np.pi * 800))
    assert context.estimated_frequency_selectivity == pytest.approx(
        1 - np.exp(-2 * np.pi * 1000 * 2e-5)
    )
    assert perfect.observe(_state(1, snr=22)).estimated_variation == pytest.approx(
        1 - np.exp(-1 / np.sqrt(3))
    )


@pytest.mark.parametrize(
    "config",
    [
        {"snr_std": -1},
        {"delay_frames": -1},
        {"delay_frames": 1.5},
        {"doppler_relative_error": np.nan},
        {"reference_bandwidth_hz": 0},
        {"snr_bias_db": np.inf},
        {"delay_variation_scale_s": 0},
    ],
)
def test_invalid_observation_configuration_is_rejected(config):
    with pytest.raises(ValueError):
        ObservationModel(config, 1)


def test_context_rejects_invalid_numeric_values():
    with pytest.raises(ValueError):
        ObservedContext(0, np.nan, 1, 1, 1, 1, 1)
    with pytest.raises(ValueError):
        ObservedContext(0, 1, -1, 1, 1, 1, 1)
