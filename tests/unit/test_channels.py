"""Physical finite sums, distribution checks, and immutable shared channels."""

import json

import numpy as np
import pytest

from waveforge6g.channels import ChannelRealization, realize_channel
from waveforge6g.channels.doppler import isotropic_dopplers, maximum_doppler_hz
from waveforge6g.channels.flat_fading import rayleigh_gains, rician_gains
from waveforge6g.channels.profiles import SYNTHETIC_PROFILES, get_profile


def test_delay_and_doppler_use_output_time_and_zero_prehistory():
    channel = ChannelRealization(np.array([2]), np.array([2j]), np.array([125.0]), 1000.0)
    transmitted = np.array([1, 2, 3, 4, 5], dtype=complex)
    expected = np.array([0, 0, -2, -2 * np.sqrt(2) - 2j * np.sqrt(2), -6j])
    np.testing.assert_allclose(channel.apply(transmitted), expected, atol=2e-15)


def test_channel_matches_independent_dense_matrix_and_adjoint():
    rng = np.random.default_rng(17)
    channel = ChannelRealization(
        np.array([0, 2, 2, 18]),
        np.array([0.8, 0.1j, -0.2 + 0.3j, 0.7]),
        np.array([0.0, 81.5, -147.2, 20]),
        1600.0,
    )
    length = 16
    matrix = np.zeros((length, length), dtype=complex)
    for n in range(length):
        for m in range(length):
            for delay, gain, doppler in zip(
                channel.delays, channel.gains, channel.dopplers_hz, strict=True
            ):
                if m == n - delay:
                    matrix[n, m] += gain * np.exp(2j * np.pi * doppler * n / channel.sample_rate_hz)
    x = rng.standard_normal(length) + 1j * rng.standard_normal(length)
    y = rng.standard_normal(length) + 1j * rng.standard_normal(length)
    np.testing.assert_allclose(channel.apply(x), matrix @ x, atol=8e-16)
    np.testing.assert_allclose(channel.adjoint(y), matrix.conj().T @ y, atol=8e-16)
    np.testing.assert_allclose(
        np.vdot(channel.apply(x), y), np.vdot(x, channel.adjoint(y)), atol=3e-15
    )


def test_zero_doppler_matches_linear_convolution_with_truncated_tail():
    impulse = np.array([0.8, 0, 0.3j, -0.2])
    tx = np.array([1, 2j, -1, 0.4 - 0.3j, 2, 0])
    channel = ChannelRealization(np.array([0, 2, 3]), impulse[[0, 2, 3]], np.zeros(3), 1000.0)
    np.testing.assert_allclose(channel.apply(tx), np.convolve(tx, impulse)[: tx.size], atol=1e-15)


def test_realization_is_immutable_and_serialization_is_lossless():
    gains = np.array([1 + 2j, -0.5j])
    channel = ChannelRealization(np.array([0, 1]), gains, np.array([0, 2]), 1000.0)
    gains[0] = 999
    assert channel.gains[0] == 1 + 2j
    with pytest.raises(ValueError):
        channel.gains[0] = 2
    with pytest.raises(ValueError):
        channel.gains.setflags(write=True)
    recovered = ChannelRealization.from_dict(json.loads(json.dumps(channel.to_dict())))
    assert recovered.to_dict() == channel.to_dict()
    assert channel.max_delay == 1
    assert channel.n_paths == 2
    assert channel.max_doppler_hz == 2
    np.testing.assert_allclose(channel.time_response(5)[:, 0], channel.gains)


@pytest.mark.parametrize(
    "model", ["awgn", "flat_rayleigh", "flat_rician", "static_tdl", "doubly_selective"]
)
def test_seeded_channel_factory_is_reproducible(model):
    config = {"model": model, "profile": "high_speed_train"}
    one = realize_channel(config, np.random.default_rng(130), 20000.0)
    two = realize_channel(config, np.random.default_rng(130), 20000.0)
    assert one.to_dict() == two.to_dict()
    if model == "static_tdl":
        np.testing.assert_array_equal(one.dopplers_hz, 0)


def test_fixed_path_gains_and_expected_power_normalization():
    config = {
        "model": "static_tdl",
        "normalize_power": False,
        "paths": [{"delay_samples": 0, "gain": [2, 0]}, {"delay_samples": 3, "gain": [0, 1]}],
    }
    channel = realize_channel(config, np.random.default_rng(1), 1000.0)
    np.testing.assert_array_equal(channel.gains, [2, 1j])
    config["normalize_power"] = True
    normalized = realize_channel(config, np.random.default_rng(1), 1000.0)
    np.testing.assert_allclose(normalized.gains, np.array([2, 1j]) / np.sqrt(5))


def test_random_fading_is_not_normalized_per_realization():
    config = {"model": "static_tdl", "paths": [{"delay_samples": 0, "power_db": 0}]}
    rng = np.random.default_rng(90)
    powers = np.array(
        [abs(realize_channel(config, rng, 1000.0).gains[0]) ** 2 for _ in range(3000)]
    )
    assert abs(powers.mean() - 1) < 0.06
    assert powers.std() > 0.8


def test_rayleigh_and_rician_have_expected_ensemble_moments():
    rng = np.random.default_rng(49)
    rayleigh = rayleigh_gains(rng, 100_000)
    rician = rician_gains(rng, k_factor_db=10, count=100_000, los_phase=0.3)
    assert abs(np.mean(rayleigh)) < 0.01
    assert abs(np.mean(np.abs(rayleigh) ** 2) - 1) < 0.01
    assert abs(np.mean(rician) - np.sqrt(10 / 11) * np.exp(0.3j)) < 0.005
    assert abs(np.mean(np.abs(rician) ** 2) - 1) < 0.01


def test_doppler_override_scales_path_pattern_and_keeps_gains():
    config = {
        "model": "doubly_selective",
        "paths": [
            {"delay_samples": 0, "power_db": 0, "doppler_hz": -10},
            {"delay_samples": 2, "power_db": -3, "doppler_hz": 20},
        ],
    }
    original = realize_channel(config, np.random.default_rng(1), 1000.0)
    scaled = realize_channel(config, np.random.default_rng(1), 1000.0, max_doppler_hz=100)
    np.testing.assert_array_equal(scaled.gains, original.gains)
    np.testing.assert_allclose(scaled.dopplers_hz, [-50, 100])
    zero = realize_channel(config, np.random.default_rng(1), 1000.0, max_doppler_hz=0)
    np.testing.assert_array_equal(zero.dopplers_hz, 0)


def test_missing_doppler_offsets_use_supplied_rng():
    config = {"model": "doubly_selective", "max_doppler_hz": 50, "paths": [{"delay_samples": 0}]}
    channel = realize_channel(config, np.random.default_rng(3), 1000.0)
    assert 0 < abs(channel.dopplers_hz[0]) <= 50


@pytest.mark.parametrize("name", list(SYNTHETIC_PROFILES))
def test_profiles_are_synthetic_independent_and_scaled(name):
    paths = get_profile(name, 100.0)
    assert SYNTHETIC_PROFILES[name]["classification"] == "synthetic"
    assert max(abs(path["doppler_hz"]) for path in paths) == 100.0
    paths[0]["delay_samples"] = 999
    assert get_profile(name)[0]["delay_samples"] == 0


def test_doppler_helpers_have_explicit_units_and_bounds():
    assert maximum_doppler_hz(30, 3e9) == pytest.approx(300.2076856783368)
    frequencies = isotropic_dopplers(np.random.default_rng(91), 10_000, 100)
    assert np.max(np.abs(frequencies)) <= 100
    assert np.mean(frequencies**2) == pytest.approx(5000, rel=0.03)


@pytest.mark.parametrize(
    "delays,gains,dopplers,rate",
    [
        ([-1], [1], [0], 1000),
        ([0.5], [1], [0], 1000),
        ([0, 1], [1], [0], 1000),
        ([0], [np.nan], [0], 1000),
        ([0], [1], [500], 1000),
        ([0], [1], [0], 0),
    ],
)
def test_invalid_channel_realization_rejected(delays, gains, dopplers, rate):
    with pytest.raises(ValueError):
        ChannelRealization(np.array(delays), np.array(gains), np.array(dopplers), rate)


def test_static_channel_rejects_nonzero_configured_doppler():
    with pytest.raises(ValueError, match="zero Doppler"):
        realize_channel(
            {"model": "static_tdl", "paths": [{"doppler_hz": 2}]}, np.random.default_rng(0), 1000
        )
