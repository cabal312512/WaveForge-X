"""Fourier reconstruction checks cover odd/even lengths and Nyquist bins."""

import numpy as np
import pytest

from waveforge6g.analysis.papr import oversample, papr_ccdf, papr_samples
from waveforge6g.core.metrics import papr_db
from waveforge6g.waveforms.ofdm import OFDM


@pytest.mark.parametrize("n", [1, 5, 6, 31, 32])
def test_interpolation_preserves_original_samples_and_mean_power(n):
    rng = np.random.default_rng(n)
    signal = rng.normal(size=n) + 1j * rng.normal(size=n)
    interpolated = oversample(signal, 4)
    np.testing.assert_allclose(interpolated[::4], signal, atol=1e-14)
    assert np.mean(abs(interpolated) ** 2) == pytest.approx(np.mean(abs(signal) ** 2), rel=1e-13)
    assert papr_db(interpolated) >= papr_db(signal) - 1e-12


@pytest.mark.parametrize("n,frequency", [(8, -4), (8, 3), (7, -3), (7, 3)])
def test_interpolated_single_tone_has_exact_phase_and_constant_envelope(n, frequency):
    signal = np.exp(2j * np.pi * frequency * np.arange(n) / n)
    expected = np.exp(2j * np.pi * frequency * np.arange(4 * n) / (4 * n))
    np.testing.assert_allclose(oversample(signal, 4), expected, atol=1e-14)
    assert papr_db(oversample(signal, 4)) == pytest.approx(0, abs=1e-13)


def test_papr_frames_and_exceedance_are_reproducible():
    waveform = OFDM(32, 8)
    first = papr_samples(waveform, "qpsk", 8, np.random.default_rng(41))
    second = papr_samples(waveform, "qpsk", 8, np.random.default_rng(41))
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(papr_ccdf(first, [-1, 100]), [1, 0])
    with pytest.raises(ValueError):
        oversample([1, 2], 0)
