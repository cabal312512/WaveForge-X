"""Small synthetic profiles; none is an exact or claimed 3GPP standard model."""

from copy import deepcopy

import numpy as np

SYNTHETIC_PROFILES = {
    "pedestrian": {
        "classification": "synthetic",
        "max_doppler_hz": 5.0,
        "delays": [0, 1, 2],
        "powers_db": [0.0, -4.0, -8.0],
        "doppler_fractions": [0.2, -1.0, 0.6],
    },
    "urban_vehicle": {
        "classification": "synthetic",
        "max_doppler_hz": 300.0,
        "delays": [0, 1, 3, 5],
        "powers_db": [0.0, -3.0, -6.0, -9.0],
        "doppler_fractions": [0.1, -0.6, 1.0, -0.3],
    },
    "high_speed_train": {
        "classification": "synthetic",
        "max_doppler_hz": 1200.0,
        "delays": [0, 1, 4, 7],
        "powers_db": [0.0, -3.0, -7.0, -10.0],
        "doppler_fractions": [1.0, 0.7, -0.4, -0.9],
    },
    "extreme_mobility": {
        "classification": "synthetic",
        "max_doppler_hz": 3000.0,
        "delays": [0, 2, 4, 6, 8],
        "powers_db": [0.0, -2.0, -5.0, -8.0, -11.0],
        "doppler_fractions": [1.0, -0.9, 0.55, -0.35, 0.1],
    },
}


def get_profile(name: str, max_doppler_hz: float | None = None) -> list[dict]:
    """Return independent path dictionaries, with delays measured in samples."""
    if name not in SYNTHETIC_PROFILES:
        raise ValueError(f"unknown synthetic channel profile {name!r}")
    profile = deepcopy(SYNTHETIC_PROFILES[name])
    maximum = profile["max_doppler_hz"] if max_doppler_hz is None else max_doppler_hz
    if not np.isfinite(maximum) or maximum < 0:
        raise ValueError("max_doppler_hz must be finite and nonnegative")
    return [
        {"delay_samples": delay, "power_db": power, "doppler_hz": maximum * fraction}
        for delay, power, fraction in zip(
            profile["delays"], profile["powers_db"], profile["doppler_fractions"], strict=True
        )
    ]
