"""Alphabet checks independent of the modulator/demodulator inverse pair."""

import numpy as np
import pytest

from waveforge6g.core.bits import random_bits
from waveforge6g.core.modulation import bits_per_symbol, constellation, demodulate, modulate


@pytest.mark.parametrize("name,width", [("bpsk", 1), ("qpsk", 2), ("16-QAM", 4), ("QAM64", 6)])
def test_all_labels_roundtrip_and_unit_energy(name, width):
    labels = np.arange(2**width)
    bits = ((labels[:, None] >> np.arange(width - 1, -1, -1)) & 1).astype(np.uint8).ravel()
    symbols = modulate(bits, name)
    assert bits_per_symbol(name) == width
    np.testing.assert_array_equal(demodulate(symbols, name), bits)
    np.testing.assert_array_equal(symbols, constellation(name))
    assert np.mean(abs(symbols) ** 2) == pytest.approx(1, abs=1e-14)


@pytest.mark.parametrize("name", ["qpsk", "qam16", "qam64"])
def test_nearest_neighbors_have_one_bit_difference(name):
    alphabet = constellation(name)
    distances = abs(alphabet[:, None] - alphabet[None, :]) ** 2
    minimum = np.min(distances[distances > 0])
    rows, columns = np.where(np.isclose(distances, minimum))
    assert all((int(row) ^ int(column)).bit_count() == 1 for row, column in zip(rows, columns, strict=True))


def test_mapping_is_explicit_and_does_not_renormalize_frames():
    np.testing.assert_array_equal(modulate([0, 1], "bpsk"), [1, -1])
    np.testing.assert_allclose(modulate([0, 0, 0, 1, 1, 0, 1, 1], "qpsk"),
                               np.array([-1-1j, -1+1j, 1-1j, 1+1j]) / np.sqrt(2))
    assert abs(modulate([0, 0, 0, 0], "16qam")[0]) ** 2 == pytest.approx(1.8)


def test_random_bits_are_reproducible_without_global_rng():
    first = random_bits(1000, np.random.default_rng(7))
    second = random_bits(1000, np.random.default_rng(7))
    np.testing.assert_array_equal(first, second)
    assert first.dtype == np.uint8
    assert random_bits(0, np.random.default_rng()).size == 0
    with pytest.raises(ValueError, match="nonnegative integer"):
        random_bits(-1, np.random.default_rng())


@pytest.mark.parametrize("bits,name", [([0, 2], "bpsk"), ([0], "qpsk"), ([[0, 1]], "qpsk")])
def test_invalid_bits_rejected(bits, name):
    with pytest.raises(ValueError):
        modulate(bits, name)


def test_invalid_modulation_and_nonfinite_symbols():
    with pytest.raises(ValueError, match="Unsupported modulation"):
        constellation("8psk")
    with pytest.raises(ValueError, match="finite"):
        demodulate([np.nan + 0j], "qpsk")
    assert modulate([], "qam64").size == 0
    assert demodulate([], "qam64").size == 0
