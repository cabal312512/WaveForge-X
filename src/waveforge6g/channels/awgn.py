"""Complex AWGN under the project's useful-symbol Es/N0 convention.

Constellations have E|s|² = 1 and waveform transforms are unitary. For
``snr_db = 10 log10(Es/N0)``, noise variance is E|w[n]|² = N0; each real
component has variance N0/2. Noise never depends on a fading realization
or a particular frame's measured signal power.
"""

import numpy as np
from numpy.typing import NDArray


def noise_variance(snr_db: float) -> float:
    """Return complex noise variance for useful-symbol Es=1 and Es/N0 in dB."""
    if np.isnan(snr_db) or snr_db == -np.inf:
        raise ValueError("snr_db must be finite or +infinity")
    with np.errstate(over="ignore"):
        variance = float(np.power(10.0, -float(snr_db) / 10.0))
    if not np.isfinite(variance):
        raise ValueError("snr_db gives an unrepresentable noise variance")
    return variance


def complex_awgn(
    shape: int | tuple[int, ...], variance: float, rng: np.random.Generator
) -> NDArray[np.complex128]:
    """Generate circular complex Gaussian samples with E|w|²=``variance``."""
    if not np.isfinite(variance) or variance < 0:
        raise ValueError("variance must be finite and nonnegative")
    scale = np.sqrt(variance / 2.0)
    return scale * (rng.standard_normal(shape) + 1j * rng.standard_normal(shape))


def ebn0_to_esn0(
    ebn0_db: float, bits_per_symbol: int, code_rate: float = 1.0, efficiency: float = 1.0
) -> float:
    """Convert Eb/N0 to useful-symbol Es/N0, all ratios supplied in dB.

    ``Es/N0 = (Eb/N0) * bits_per_symbol * code_rate * efficiency``.
    For Eb that excludes guard energy use efficiency=1. For total transmitted
    energy per information bit with a cyclic prefix, use N/(N+CP), assuming
    ensemble-equal sample power. The guard overhead is never silently applied.
    """
    if not isinstance(bits_per_symbol, int | np.integer) or bits_per_symbol <= 0:
        raise ValueError("bits_per_symbol must be a positive integer")
    if not 0 < code_rate <= 1 or not 0 < efficiency <= 1:
        raise ValueError("code_rate and efficiency must lie in (0, 1]")
    if not np.isfinite(ebn0_db):
        raise ValueError("ebn0_db must be finite")
    return float(ebn0_db + 10 * np.log10(bits_per_symbol * code_rate * efficiency))
