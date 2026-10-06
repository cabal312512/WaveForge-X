"""Definitions, empty observations, and uncertainty edge cases."""

import numpy as np
import pytest

from waveforge6g.analysis.complexity import estimate_complexity
from waveforge6g.analysis.statistics import frame_mean_interval, wilson_interval
from waveforge6g.core.metrics import (
    bit_error_rate,
    block_error_rate,
    ccdf,
    evm_rms,
    papr_db,
    spectral_efficiency,
    symbol_error_rate,
)


def test_ber_bler_and_ser_distinguish_counts():
    expected = np.array([[0, 1, 1], [1, 0, 0]])
    observed = np.array([[1, 0, 1], [1, 0, 0]])
    assert bit_error_rate(expected, observed) == pytest.approx(1 / 3)
    assert block_error_rate(expected, observed) == 0.5
    assert block_error_rate(expected[0], observed[0]) == 1
    assert symbol_error_rate([0, 1, 2], [0, 2, 2]) == pytest.approx(1 / 3)
    for metric in (bit_error_rate, symbol_error_rate, block_error_rate, evm_rms):
        assert np.isnan(metric([], []))
        with pytest.raises(ValueError):
            metric([0], [0, 1])


def test_papr_and_evm_reference_values():
    assert papr_db([1, -1, 1j, -1j]) == pytest.approx(0)
    assert papr_db([2, 0, 0, 0]) == pytest.approx(10 * np.log10(4))
    assert papr_db([0, 0]) == 0
    assert np.isnan(papr_db([]))
    assert evm_rms([1, -1], [1.1, -0.9]) == pytest.approx(0.1)
    assert evm_rms([0], [0]) == 0
    assert evm_rms([0], [1]) == np.inf


def test_spectral_efficiency_and_ccdf():
    assert spectral_efficiency(2, 64, 80) == 1.6
    with pytest.raises(ValueError):
        spectral_efficiency(2, 64, 0)
    with pytest.raises(ValueError):
        spectral_efficiency(2, 65, 64)
    np.testing.assert_allclose(ccdf([1, 2, 2, 3], [-np.inf, 1, 2, 3, np.inf]), [1, 0.75, 0.25, 0, 0])
    assert np.all(np.isnan(ccdf([], [0, 1])))


def test_wilson_reference_and_missing_observations():
    low, high = wilson_interval(50, 100)
    assert low == pytest.approx(0.4038315304, abs=1e-9)
    assert high == pytest.approx(0.5961684696, abs=1e-9)
    zero_low, zero_high = wilson_interval(0, 100)
    assert zero_low == pytest.approx(0, abs=1e-16)
    assert 0 < zero_high < 0.04
    assert all(np.isnan(wilson_interval(0, 0)))
    with pytest.raises(ValueError):
        wilson_interval(2, 1)


def test_frame_cluster_interval_limits():
    low, high = frame_mean_interval([0.1, 0.2, 0.3, 0.4] * 20)
    assert low < 0.25 < high
    assert all(np.isnan(frame_mean_interval([0, 0, 0])))
    assert all(np.isnan(frame_mean_interval([0.1])))
    with pytest.raises(ValueError):
        frame_mean_interval([-0.1])


def test_complexity_estimates_are_explicit_not_empirical():
    dense = estimate_complexity(128, n_paths=3, solver="dense")
    iterative = estimate_complexity(128, n_paths=3, solver="iterative", iterations=40)
    assert dense.working_memory_bytes > iterative.working_memory_bytes
    assert "O(N^3" in dense.detector_time
    assert "K=40" in iterative.assumptions
    assert dense.to_dict()["working_memory_bytes"] == dense.working_memory_bytes
