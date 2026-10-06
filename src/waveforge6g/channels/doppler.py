"""Doppler helpers with explicit hertz and speed units."""

import numpy as np
from numpy.typing import NDArray

SPEED_OF_LIGHT_M_S = 299_792_458.0


def maximum_doppler_hz(speed_m_s: float, carrier_frequency_hz: float) -> float:
    """Magnitude v*fc/c for the single-ray narrowband approximation."""
    if not np.isfinite(speed_m_s) or speed_m_s < 0:
        raise ValueError("speed_m_s must be finite and nonnegative")
    if not np.isfinite(carrier_frequency_hz) or carrier_frequency_hz <= 0:
        raise ValueError("carrier_frequency_hz must be finite and positive")
    return float(speed_m_s * carrier_frequency_hz / SPEED_OF_LIGHT_M_S)


def isotropic_dopplers(
    rng: np.random.Generator, count: int, maximum_hz: float
) -> NDArray[np.float64]:
    """One independent uniform arrival angle per path, f=fmax*cos(theta).

    A finite collection of tones is a synthetic specular channel, not an exact
    Clarke/Jakes diffuse fading process.
    """
    if not isinstance(count, int | np.integer) or count < 1:
        raise ValueError("count must be a positive integer")
    if not np.isfinite(maximum_hz) or maximum_hz < 0:
        raise ValueError("maximum_hz must be finite and nonnegative")
    return float(maximum_hz) * np.cos(rng.uniform(-np.pi, np.pi, count))
