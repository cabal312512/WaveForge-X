"""Dimensionless error metrics, sample PAPR (dB), and nominal bits/s/Hz."""

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _matched(reference: ArrayLike, estimate: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    reference_array, estimate_array = np.asarray(reference), np.asarray(estimate)
    if reference_array.shape != estimate_array.shape:
        raise ValueError("reference and estimate must have identical shapes")
    return reference_array, estimate_array


def bit_error_rate(reference: ArrayLike, estimate: ArrayLike) -> float:
    """Fraction of unequal bits; return NaN for no observations."""
    expected, observed = _matched(reference, estimate)
    if np.any((expected != 0) & (expected != 1)) or np.any((observed != 0) & (observed != 1)):
        raise ValueError("bit arrays must contain only 0 and 1")
    return float(np.mean(expected != observed)) if expected.size else float("nan")


def symbol_error_rate(reference: ArrayLike, estimate: ArrayLike) -> float:
    """Fraction of unequal hard symbol labels (or exact constellation points)."""
    expected, observed = _matched(reference, estimate)
    return float(np.mean(expected != observed)) if expected.size else float("nan")


def block_error_rate(reference: ArrayLike, estimate: ArrayLike) -> float:
    """Fraction of blocks with any error; shape [blocks, bits] (1-D = one block)."""
    expected, observed = _matched(reference, estimate)
    if expected.ndim not in (1, 2):
        raise ValueError("block arrays must have one or two dimensions")
    if not expected.size:
        return float("nan")
    return float(np.mean(np.any(expected != observed, axis=-1)))


def papr_db(signal: ArrayLike) -> float:
    """Sample peak/average power in dB; empty signal NaN and zero signal 0 dB.

    This function measures the supplied samples only. Use periodic Fourier
    oversampling before calling for the project's approximate intersample PAPR.
    """
    samples = np.asarray(signal)
    if not np.all(np.isfinite(samples)):
        raise ValueError("signal must be finite")
    if not samples.size:
        return float("nan")
    peak_amplitude = float(np.max(np.abs(samples)))
    if peak_amplitude == 0:
        return 0.0
    # Scaling avoids overflow for valid large finite amplitudes.
    relative_power = np.abs(samples / peak_amplitude) ** 2
    return float(-10 * np.log10(np.mean(relative_power)))


def evm_rms(reference: ArrayLike, estimate: ArrayLike) -> float:
    """sqrt(sum |estimate-reference|² / sum |reference|²), dimensionless.

    Empty arrays return NaN. An all-zero reference gives zero for an exact
    all-zero estimate and infinity otherwise.
    """
    expected, observed = _matched(reference, estimate)
    if not np.all(np.isfinite(expected)) or not np.all(np.isfinite(observed)):
        raise ValueError("reference and estimate must be finite")
    if not expected.size:
        return float("nan")
    energy = float(np.sum(np.abs(expected) ** 2))
    error = float(np.sum(np.abs(observed - expected) ** 2))
    if energy == 0:
        return 0.0 if error == 0 else float("inf")
    return float(np.sqrt(error / energy))


def spectral_efficiency(bits_per_symbol: int, n_useful: int, n_total: int) -> float:
    """Uncoded nominal bits/s/Hz = bits_per_symbol*N_useful/N_total.

    Assumes occupied bandwidth equals complex sample rate; excludes coding,
    pilots, filtering, packet loss, and physical guard bands. This is not goodput.
    """
    values = (bits_per_symbol, n_useful, n_total)
    if any(not isinstance(x, int | np.integer) or isinstance(x, bool) for x in values):
        raise ValueError("symbol width and lengths must be integers")
    if bits_per_symbol <= 0 or not 0 <= n_useful <= n_total or n_total <= 0:
        raise ValueError("Require positive symbol width and 0 <= n_useful <= n_total > 0")
    return float(bits_per_symbol * n_useful / n_total)


def ccdf(samples: ArrayLike, thresholds: ArrayLike) -> NDArray[np.float64]:
    """Empirical strict exceedance probability P(sample > threshold).

    Empty samples return NaN at each threshold. Searchsorted avoids a large
    samples-by-thresholds allocation. Thresholds may be +/-infinity.
    """
    values = np.asarray(samples, dtype=float)
    levels = np.asarray(thresholds, dtype=float)
    if values.ndim != 1 or not np.all(np.isfinite(values)) or np.any(np.isnan(levels)):
        raise ValueError("samples must be finite and one-dimensional; thresholds cannot be NaN")
    if not values.size:
        return np.full(levels.shape, np.nan)
    return (values.size - np.searchsorted(np.sort(values), levels, side="right")) / values.size
