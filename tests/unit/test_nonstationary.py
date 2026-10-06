"""Upgrade B: causal trajectories, correlated paths, and physical consistency."""

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from waveforge6g.channels.nonstationary import (
    SCENARIO_NAMES,
    NonStationaryChannel,
    ScalarTrajectory,
    TrueChannelState,
)


def _trajectory(spec, epochs=5, seed=1):
    generator = ScalarTrajectory(spec, epochs, np.random.default_rng(seed))
    return [generator.step(t) for t in range(epochs)]


def _engine(config=None, seed=4):
    return NonStationaryChannel(
        config or {"scenario": "stable_low_mobility", "epochs": 8}, seed, 32, 8, 32000.0
    )


@pytest.mark.parametrize(
    "spec,expected",
    [
        (3, [3] * 5),
        ({"type": "constant", "value": 2}, [2] * 5),
        ({"type": "linear", "start": 0, "end": 4}, [0, 1, 2, 3, 4]),
        ({"type": "sinusoidal", "offset": 2, "amplitude": 1, "period": 4}, [2, 3, 2, 1, 2]),
        ({"type": "piecewise_constant", "points": [[0, 1], [2, 4]]}, [1, 1, 4, 4, 4]),
        ({"type": "piecewise_linear", "points": [[0, 0], [2, 4], [4, 0]]}, [0, 2, 4, 2, 0]),
        ({"type": "abrupt", "time": 2, "before": 1, "after": -1}, [1, 1, -1, -1, -1]),
        (
            {"type": "smooth", "start_time": 1, "end_time": 3, "before": 0, "after": 2},
            [0, 0, 1, 2, 2],
        ),
    ],
)
def test_trajectory_definitions_against_known_values(spec, expected):
    np.testing.assert_allclose(_trajectory(spec), expected, atol=1e-14)


def test_random_walk_reproducible_bounded_and_causal():
    spec = {"type": "random_walk", "initial": 2, "std": 0.5, "min": 1, "max": 3}
    first, second = _trajectory(spec, 100), _trajectory(spec, 100)
    assert first == second and len(set(first)) > 10
    assert min(first) >= 1 and max(first) <= 3
    trajectory = ScalarTrajectory(spec, 3, np.random.default_rng(1))
    with pytest.raises(ValueError, match="sequential"):
        trajectory.step(1)
    trajectory.step(0)
    with pytest.raises(ValueError, match="sequential"):
        trajectory.step(0)


@pytest.mark.parametrize("name", SCENARIO_NAMES)
def test_all_scenarios_repeat_exactly_and_obey_frame_constraints(name):
    config = {"scenario": name, "epochs": 24, "change_interval": 6}
    first, second = list(_engine(config, 42)), list(_engine(config, 42))
    for (state, channel), (other_state, other_channel) in zip(first, second, strict=True):
        assert state.to_dict() == other_state.to_dict()
        assert channel.to_dict() == other_channel.to_dict()
        assert channel.max_delay <= 8
        assert state.path_count == channel.n_paths
        assert state.max_doppler_hz == channel.max_doppler_hz
        assert sum(state.path_power_distribution) == pytest.approx(1)
        seconds = channel.delays / channel.sample_rate_hz
        powers = np.asarray(state.path_power_distribution)
        reference_rms = np.sqrt(powers @ (seconds - powers @ seconds) ** 2)
        assert state.delay_spread_s == pytest.approx(reference_rms)
        assert np.isfinite(state.snr_db)


def test_stable_control_has_no_scripted_changes_but_fading_changes():
    data = list(_engine({"scenario": "stable_low_mobility", "epochs": 10}))
    assert all(not state.change_point and state.stationarity == 1 for state, _ in data)
    assert all(state.snr_db == data[0][0].snr_db for state, _ in data)
    assert not np.array_equal(data[0][1].gains, data[-1][1].gains)


def test_acceleration_and_abrupt_markers():
    accelerated = list(_engine({"scenario": "acceleration", "epochs": 10}))
    assert np.all(np.diff([state.max_doppler_hz for state, _ in accelerated]) > 0)
    abrupt = list(_engine({"scenario": "abrupt_change", "epochs": 10, "change_interval": 3}))
    assert [state.time for state, _ in abrupt if state.change_point] == [3, 6, 9]
    assert abrupt[2][0].snr_db > abrupt[3][0].snr_db
    assert abrupt[2][0].max_doppler_hz < abrupt[3][0].max_doppler_hz


def test_future_schedule_does_not_affect_prior_channel_realizations():
    first_config = {
        "scenario": "stable_low_mobility",
        "epochs": 10,
        "trajectories": {"snr_db": {"type": "abrupt", "time": 7, "before": 10, "after": -10}},
    }
    other_config = {
        "scenario": "stable_low_mobility",
        "epochs": 10,
        "trajectories": {"snr_db": {"type": "abrupt", "time": 7, "before": 10, "after": 30}},
    }
    first, second = _engine(first_config), _engine(other_config)
    for time in range(7):
        state1, channel1 = first.step(time)
        state2, channel2 = second.step(time)
        assert state1 == state2 and channel1.to_dict() == channel2.to_dict()
    with pytest.raises(ValueError, match="sequential"):
        first.step(9)


def test_doppler_phase_continuity_across_changing_frequencies():
    engine = _engine(
        {
            "scenario": "stable_low_mobility",
            "epochs": 4,
            "path_count": 1,
            "correlation": 1,
            "trajectories": {"max_doppler_hz": {"type": "linear", "start": 20, "end": 800}},
        }
    )
    previous = None
    for _, channel in engine:
        if previous is not None:
            expected = previous.gains * np.exp(
                2j * np.pi * previous.dopplers_hz * engine.frame_duration_s
            )
            np.testing.assert_allclose(channel.gains, expected, atol=2e-15)
        previous = channel


def test_ar1_diffuse_gain_correlation_and_ensemble_power():
    engine = _engine(
        {
            "scenario": "stable_low_mobility",
            "epochs": 1800,
            "path_count": 1,
            "correlation": 0.8,
            "max_doppler_hz": 0,
            "k_factor_db": -120,
        }
    )
    gains = np.array([channel.gains[0] for _, channel in engine])
    lag_one = np.vdot(gains[:-1], gains[1:]) / np.vdot(gains[:-1], gains[:-1])
    assert abs(lag_one - 0.8) < 0.05
    assert np.mean(np.abs(gains) ** 2) == pytest.approx(1, abs=0.15)
    assert np.std(np.abs(gains) ** 2) > 0.7


def test_cluster_lifetimes_birth_death_and_power_ramps_are_real():
    engine = _engine(
        {
            "scenario": "cluster_birth_death",
            "epochs": 35,
            "cluster_dynamics": {
                "birth_probability": 0.5,
                "death_probability": 0.2,
                "lifetime": 8,
                "power_ramp_frames": 3,
                "max_paths": 8,
            },
        }
    )
    ids, counts, events = [], [], []
    for state, _ in engine:
        ids.append(engine.active_path_ids)
        counts.append(state.path_count)
        events.extend(engine.last_events)
    assert len(set(counts)) > 1
    assert any(event.startswith("path_removed:") for event in events)
    assert max(max(values) for values in ids) > max(ids[0])
    assert min(counts) >= 1 and max(counts) <= 8


def test_expected_power_ramp_grows_without_normalizing_realized_gain():
    engine = _engine(
        {
            "scenario": "stable_low_mobility",
            "epochs": 5,
            "correlation": 1,
            "trajectories": {"path_count": {"type": "abrupt", "time": 1, "before": 1, "after": 2}},
            "cluster_dynamics": {"power_ramp_frames": 4},
        }
    )
    data = list(engine)
    newborn = [state.path_power_distribution[-1] for state, _ in data[1:]]
    assert np.all(np.diff(newborn) > 0)
    assert any(abs(np.sum(np.abs(channel.gains) ** 2) - 1) > 0.05 for _, channel in data)


def test_fractional_severity_zero_stays_on_integer_doppler_grid():
    data = list(
        _engine(
            {
                "scenario": "stable_low_mobility",
                "epochs": 3,
                "max_doppler_hz": 1800,
                "fractional_doppler_severity": 0,
            }
        )
    )
    for state, channel in data:
        np.testing.assert_allclose(channel.dopplers_hz / 1000, np.rint(channel.dopplers_hz / 1000))
        assert state.fractional_doppler_severity == 0


def test_truth_state_is_deeply_immutable_and_engine_ends():
    engine = _engine({"scenario": "stable_low_mobility", "epochs": 1})
    state, _ = engine.step(0)
    assert isinstance(state, TrueChannelState)
    assert isinstance(state.path_power_distribution, tuple)
    with pytest.raises(FrozenInstanceError):
        state.snr_db = 1
    result = state.to_dict()
    result["path_power_distribution"][0] = 999
    assert state.path_power_distribution[0] < 1
    with pytest.raises(StopIteration):
        next(engine)


def test_delay_spread_target_is_quantized_and_constrained_by_guard():
    engine = _engine(
        {
            "scenario": "stable_low_mobility",
            "epochs": 1,
            "trajectories": {"delay_spread_s": 0.00002},
        }
    )
    state, channel = engine.step(0)
    assert abs(state.delay_spread_s - 0.00002) < 1 / channel.sample_rate_hz
    assert channel.max_delay <= 8


@pytest.mark.parametrize(
    "config",
    [
        {"scenario": "unknown"},
        {"scenario": "stable_low_mobility", "epochs": 0},
        {"scenario": "stable_low_mobility", "correlation": 2},
        {"scenario": "stable_low_mobility", "max_doppler_hz": 16000},
        {"scenario": "stable_low_mobility", "trajectories": {"future_truth": 1}},
    ],
)
def test_invalid_nonstationary_configuration_is_rejected(config):
    with pytest.raises(ValueError):
        next(_engine(config))
