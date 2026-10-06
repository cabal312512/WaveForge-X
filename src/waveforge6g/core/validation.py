"""Small deterministic numeric checks used by the validate CLI."""

import numpy as np

from waveforge6g.channels.awgn import complex_awgn
from waveforge6g.core.bits import random_bits
from waveforge6g.core.fft import fft_unitary, ifft_unitary
from waveforge6g.core.modulation import bits_per_symbol, demodulate, modulate


def run_validations() -> dict[str, float | bool]:
    """Execute reproducible low-cost core checks; return measured diagnostics."""
    rng = np.random.default_rng(1731)
    signal = rng.normal(size=127) + 1j * rng.normal(size=127)
    reconstruction_error = float(np.max(np.abs(ifft_unitary(fft_unitary(signal)) - signal)))
    result: dict[str, float | bool] = {"fft_roundtrip_max_error": reconstruction_error}
    for name in ("bpsk", "qpsk", "qam16", "qam64"):
        bits = random_bits(256 * bits_per_symbol(name), rng)
        result[f"{name}_bit_perfect"] = bool(np.array_equal(bits, demodulate(modulate(bits, name), name)))
    noise = complex_awgn(100_000, 0.25, rng)
    result["noise_variance_relative_error"] = float(abs(np.mean(abs(noise) ** 2) / 0.25 - 1))
    result["passed"] = bool(
        reconstruction_error < 1e-12
        and all(result[f"{name}_bit_perfect"] for name in ("bpsk", "qpsk", "qam16", "qam64"))
        and result["noise_variance_relative_error"] < 0.02
    )
    return result
