"""Unit-mean-power flat fading distributions; no per-draw normalization."""

import numpy as np
from numpy.typing import NDArray


def rayleigh_gains(rng: np.random.Generator, count: int = 1) -> NDArray[np.complex128]:
    """Circular CN(0,1) gains, with independent half-variance quadratures."""
    if not isinstance(count, int | np.integer) or count < 1:
        raise ValueError("count must be a positive integer")
    return (rng.standard_normal(count) + 1j * rng.standard_normal(count)) / np.sqrt(2)


def rician_gains(
    rng: np.random.Generator, k_factor_db: float = 6.0, count: int = 1, los_phase: float = 0.0
) -> NDArray[np.complex128]:
    """sqrt(K/(K+1))*exp(j*phase) + CN(0,1/(K+1)); K is in dB."""
    if not np.isfinite(k_factor_db) or not np.isfinite(los_phase):
        raise ValueError("Rician K factor and LOS phase must be finite")
    # Logistic weights avoid overflow even for extreme finite dB values.
    log_k = float(k_factor_db) * np.log(10) / 10
    scattered_power = float(np.exp(-np.logaddexp(0.0, log_k)))
    los_power = float(np.exp(-np.logaddexp(0.0, -log_k)))
    return np.sqrt(los_power) * np.exp(1j * los_phase) + np.sqrt(scattered_power) * rayleigh_gains(
        rng, count
    )
