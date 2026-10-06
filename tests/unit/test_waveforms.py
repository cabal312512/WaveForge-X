"""Independent finite-sum references catch matched forward/inverse sign bugs."""

import numpy as np
import pytest

from waveforge6g.channels import ChannelRealization
from waveforge6g.waveforms import AFDM, OFDM, OTFS, create_waveform


def _complex(rng, length):
    return rng.standard_normal(length) + 1j * rng.standard_normal(length)


@pytest.mark.parametrize("name", ["ofdm", "otfs", "afdm"])
@pytest.mark.parametrize("cp", [0, 5, 24])
def test_waveform_roundtrip_energy_and_adjoint(name, cp):
    rng = np.random.default_rng(381)
    waveform = create_waveform(name, 24, cp, 6)
    symbols, test_vector = _complex(rng, 24), _complex(rng, 24)
    useful = waveform.synthesis(symbols)
    np.testing.assert_allclose(waveform.analysis(useful), symbols, atol=2e-14)
    np.testing.assert_allclose(waveform.demodulate(waveform.modulate(symbols)), symbols, atol=2e-14)
    np.testing.assert_allclose(np.vdot(useful, useful), np.vdot(symbols, symbols), atol=5e-14)
    np.testing.assert_allclose(
        np.vdot(useful, test_vector),
        np.vdot(symbols, waveform.synthesis_adjoint(test_vector)),
        atol=3e-14,
    )
    full_vector = _complex(rng, 24 + cp)
    np.testing.assert_allclose(
        np.vdot(waveform.add_prefix(useful), full_vector),
        np.vdot(useful, waveform.prefix_adjoint(full_vector)),
        atol=3e-14,
    )
    assert waveform.modulate(symbols).shape == (24 + cp,)


def test_ofdm_matches_positive_exponent_finite_sum():
    symbols = np.array([1.0, 2j, -3.0, 4.0 - 1j, -2.0])
    expected = np.array(
        [
            sum(symbols[k] * np.exp(2j * np.pi * n * k / 5) for k in range(5)) / np.sqrt(5)
            for n in range(5)
        ]
    )
    np.testing.assert_allclose(OFDM(5, 2).synthesis(symbols), expected, atol=5e-15)


def test_otfs_isfft_matches_explicit_two_dimensional_sum():
    rng = np.random.default_rng(19)
    k_count, m_count = 3, 4
    grid = _complex(rng, k_count * m_count).reshape((k_count, m_count), order="F")
    waveform = OTFS(12, 2, 3)
    expected = np.zeros_like(grid)
    for frequency in range(k_count):
        for time in range(m_count):
            for delay in range(k_count):
                for doppler in range(m_count):
                    expected[frequency, time] += (
                        grid[delay, doppler]
                        * np.exp(
                            2j * np.pi * (time * doppler / m_count - frequency * delay / k_count)
                        )
                        / np.sqrt(k_count * m_count)
                    )
    np.testing.assert_allclose(waveform.isfft(grid), expected, atol=5e-15)
    np.testing.assert_allclose(waveform.sfft(expected), grid, atol=5e-15)


def test_otfs_serialization_matches_kronecker_matrix():
    # vec_F(X F_M^H) = (F_M^H kron I_K) vec_F(X).
    rng = np.random.default_rng(4)
    symbols = _complex(rng, 15)
    indices = np.arange(5)
    idft = np.exp(2j * np.pi * indices[:, None] * indices[None, :] / 5) / np.sqrt(5)
    expected = np.kron(idft, np.eye(3)) @ symbols
    np.testing.assert_allclose(OTFS(15, 2, 3).synthesis(symbols), expected, atol=3e-15)


@pytest.mark.parametrize("n,c1,c2", [(11, 0.037, -0.029), (16, 3 / 32, 0.017)])
def test_afdm_matches_dense_daft_and_continuation(n, c1, c2):
    rng = np.random.default_rng(28)
    symbols = _complex(rng, n)
    waveform = AFDM(n, 4, c1=c1, c2=c2)
    rows, columns = np.arange(n)[:, None], np.arange(n)[None, :]
    inverse_daft = np.exp(
        2j * np.pi * (c1 * rows**2 + rows * columns / n + c2 * columns**2)
    ) / np.sqrt(n)
    expected = inverse_daft @ symbols
    np.testing.assert_allclose(waveform.synthesis(symbols), expected, atol=3e-13)
    np.testing.assert_allclose(waveform.analysis(expected), symbols, atol=3e-13)
    np.testing.assert_allclose(inverse_daft.conj().T @ inverse_daft, np.eye(n), atol=3e-14)
    # Prefix must equal evaluating the very same chirp basis at negative time.
    negative = np.arange(-4, 0)[:, None]
    continued = (
        np.exp(2j * np.pi * (c1 * negative**2 + negative * columns / n + c2 * columns**2))
        @ symbols
        / np.sqrt(n)
    )
    np.testing.assert_allclose(waveform.modulate(symbols)[:4], continued, atol=3e-13)


@pytest.mark.parametrize("n", [15, 16])
def test_default_afdm_guard_is_periodic_for_odd_and_even_lengths(n):
    np.testing.assert_allclose(AFDM(n, 5).prefix_phases, np.ones(5), atol=3e-14)


def test_zero_chirps_reduce_afdm_to_ofdm():
    symbols = _complex(np.random.default_rng(1), 17)
    np.testing.assert_allclose(AFDM(17, 3, 0, 0).modulate(symbols), OFDM(17, 3).modulate(symbols))


def test_static_channel_ofdm_has_expected_frequency_response():
    waveform = OFDM(32, 5)
    symbols = _complex(np.random.default_rng(81), 32)
    channel = ChannelRealization(
        np.array([0, 2, 5]), np.array([1, 0.2j, -0.1]), np.zeros(3), 1000.0
    )
    impulse = np.zeros(32, dtype=complex)
    impulse[channel.delays] = channel.gains
    expected = np.fft.fft(impulse) * symbols
    observed = waveform.demodulate(channel.apply(waveform.modulate(symbols)))
    np.testing.assert_allclose(observed, expected, atol=3e-15)


@pytest.mark.parametrize("name", ["ofdm", "otfs", "afdm"])
def test_identity_channel_is_exact_for_each_waveform(name):
    waveform = create_waveform(name, 24, 4, 6)
    symbols = _complex(np.random.default_rng(88), 24)
    channel = ChannelRealization(np.array([0]), np.array([1]), np.array([0]), 1000.0)
    np.testing.assert_allclose(
        waveform.demodulate(channel.apply(waveform.modulate(symbols))), symbols, atol=3e-15
    )


@pytest.mark.parametrize("n,cp", [(0, 0), (8, -1), (8, 9), (3.5, 0), (True, 0), (8, 1.5)])
def test_invalid_waveform_sizes_rejected(n, cp):
    with pytest.raises(ValueError):
        OFDM(n, cp)


def test_invalid_grid_and_nonfinite_values_rejected():
    with pytest.raises(ValueError):
        OTFS(10, 1, 3)
    with pytest.raises(ValueError):
        OTFS(12, 2, 3).isfft(np.zeros((4, 3)))
    with pytest.raises(ValueError):
        AFDM(12, 2, np.nan)
    with pytest.raises(ValueError):
        OFDM(4, 1).synthesis([1, 2, 3, np.inf])
    with pytest.raises(ValueError):
        OFDM(4, 1).demodulate(np.ones(4))
    with pytest.raises(ValueError):
        create_waveform("unknown", 4, 1, 2)
