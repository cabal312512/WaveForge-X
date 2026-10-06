"""Executed link-level statistical reference, not an invented BER fixture."""

import numpy as np
from scipy.special import erfc

from waveforge6g.channels.awgn import complex_awgn, noise_variance
from waveforge6g.core.bits import random_bits
from waveforge6g.core.modulation import demodulate, modulate
from waveforge6g.receivers.equalizers import one_tap_mmse
from waveforge6g.waveforms.ofdm import OFDM


def test_ofdm_qpsk_awgn_agrees_with_theory():
    """At Es/N0=gamma, Gray QPSK BER=0.5*erfc(sqrt(gamma/2))."""
    waveform = OFDM(64, 16)
    frame_count = 1200
    bits_per_frame = 128
    rates = []
    for snr_index, snr_db in enumerate((0.0, 4.0, 8.0)):
        rng = np.random.default_rng(np.random.SeedSequence([194, snr_index]))
        variance = noise_variance(snr_db)
        errors = 0
        for _ in range(frame_count):
            bits = random_bits(bits_per_frame, rng)
            transmitted = waveform.modulate(modulate(bits, "qpsk"))
            received = transmitted + complex_awgn(transmitted.shape, variance, rng)
            estimates = one_tap_mmse(waveform.demodulate(received), 1.0, variance)
            errors += np.count_nonzero(demodulate(estimates, "qpsk") != bits)
        total = frame_count * bits_per_frame
        measured = errors / total
        theory = 0.5 * erfc(np.sqrt(10 ** (snr_db / 10) / 2))
        standard_error = np.sqrt(theory * (1 - theory) / total)
        assert abs(measured - theory) < 6 * standard_error + 1 / total
        rates.append(measured)
    assert rates[0] > rates[1] > rates[2]
