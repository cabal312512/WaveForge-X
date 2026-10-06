"""Transform scaling and empirical circular-noise variance."""

import numpy as np
import pytest

from waveforge6g.channels.awgn import complex_awgn, ebn0_to_esn0, noise_variance
from waveforge6g.core.fft import fft_unitary, ifft_unitary
from waveforge6g.core.validation import run_validations


@pytest.mark.parametrize("length", [1, 7, 32, 127])
def test_fft_roundtrip_parseval_and_sign(length):
    rng = np.random.default_rng(81)
    values = rng.normal(size=(3, length)) + 1j * rng.normal(size=(3, length))
    transformed = fft_unitary(values)
    np.testing.assert_allclose(ifft_unitary(transformed), values, atol=2e-14)
    np.testing.assert_allclose(np.sum(abs(transformed) ** 2, axis=1), np.sum(abs(values) ** 2, axis=1))
    impulse = np.zeros(length)
    impulse[0] = 1
    np.testing.assert_allclose(fft_unitary(impulse), np.ones(length) / np.sqrt(length))


def test_complex_noise_variance_and_empirical_snr():
    variance = noise_variance(7.0)
    samples = complex_awgn(200_000, variance, np.random.default_rng(71))
    assert np.mean(abs(samples) ** 2) == pytest.approx(variance, rel=0.012)
    assert samples.real.var() == pytest.approx(variance / 2, rel=0.018)
    assert samples.imag.var() == pytest.approx(variance / 2, rel=0.018)
    assert abs(np.mean(samples**2)) < 0.01 * variance
    assert 10 * np.log10(1 / np.mean(abs(samples) ** 2)) == pytest.approx(7, abs=0.06)


def test_noise_zero_and_bad_variances():
    np.testing.assert_array_equal(complex_awgn(7, 0, np.random.default_rng(1)), np.zeros(7))
    assert noise_variance(np.inf) == 0
    for snr in (np.nan, -np.inf, -4000):
        with pytest.raises(ValueError):
            noise_variance(snr)
    for variance in (-1, np.nan, np.inf):
        with pytest.raises(ValueError):
            complex_awgn(3, variance, np.random.default_rng(1))


def test_ebn0_guard_convention_is_explicit():
    assert ebn0_to_esn0(0, 2) == pytest.approx(10 * np.log10(2))
    assert ebn0_to_esn0(0, 2, efficiency=64 / 80) == pytest.approx(10 * np.log10(1.6))
    with pytest.raises(ValueError):
        ebn0_to_esn0(0, 0)
    with pytest.raises(ValueError):
        ebn0_to_esn0(0, 2, efficiency=1.1)


def test_numeric_validation_entrypoint():
    assert run_validations()["passed"] is True
