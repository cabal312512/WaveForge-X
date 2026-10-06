"""Scalar frequency-domain equalizers for genuinely diagonal channels only."""

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _inputs(y: ArrayLike, h: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    received = np.asarray(y, dtype=np.complex128)
    channel = np.asarray(h, dtype=np.complex128)
    if not np.all(np.isfinite(received)) or not np.all(np.isfinite(channel)):
        raise ValueError("received samples and channel must be finite")
    if channel.shape != received.shape and channel.ndim != 0:
        raise ValueError("h must be scalar or have the same shape as y")
    return received, channel


def one_tap_zf(y: ArrayLike, h: ArrayLike) -> NDArray[np.complex128]:
    """Return y/h; exact spectral nulls raise rather than silently becoming NaN."""
    received, channel = _inputs(y, h)
    if np.any(channel == 0):
        raise np.linalg.LinAlgError("ZF is undefined at a zero channel coefficient")
    return received / channel


def one_tap_mmse(y: ArrayLike, h: ArrayLike, noise_variance: float) -> NDArray[np.complex128]:
    """Return h* y / (|h|² + N0), assuming E|symbol|²=1.

    At N0=0 this uses the minimum-norm zero estimate for an exactly null tone.
    A one-tap model does not describe intercarrier interference under Doppler.
    """
    received, channel = _inputs(y, h)
    if not np.isfinite(noise_variance) or noise_variance < 0:
        raise ValueError("noise_variance must be finite and nonnegative")
    denominator = abs(channel) ** 2 + noise_variance
    numerator = channel.conj() * received
    return np.divide(numerator, denominator, out=np.zeros_like(received), where=denominator != 0)
