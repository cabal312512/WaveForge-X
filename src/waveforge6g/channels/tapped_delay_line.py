"""Configuration-to-realization factory for shared synthetic path channels."""

from collections.abc import Mapping
from numbers import Integral

import numpy as np

from .doppler import isotropic_dopplers
from .doubly_selective import ChannelRealization
from .flat_fading import rayleigh_gains, rician_gains
from .profiles import get_profile


def _gain(value: object) -> complex:
    if isinstance(value, list | tuple | np.ndarray):
        if len(value) != 2:
            raise ValueError("a fixed path gain must be [real, imaginary]")
        result = complex(value[0], value[1])
    else:
        result = complex(value)
    if not np.isfinite(result):
        raise ValueError("path gains must be finite")
    return result


def realize_channel(
    config: Mapping,
    rng: np.random.Generator,
    sample_rate_hz: float,
    max_doppler_hz: float | None = None,
) -> ChannelRealization:
    """Draw one realization using only the supplied RNG.

    Configured powers normalize E[sum |gain|^2], never the realized norm.
    ``gain`` overrides a path's ``power_db`` as its deterministic coefficient.
    A Doppler override rescales supplied path offsets; paths without offsets
    receive independent isotropic arrival angles. See docs/theory/channels.md.
    """
    model = str(config.get("model", "awgn")).lower()
    valid_models = {"awgn", "flat_rayleigh", "flat_rician", "static_tdl", "doubly_selective"}
    if model not in valid_models:
        raise ValueError(f"unsupported channel model {model!r}")
    maximum = max_doppler_hz if max_doppler_hz is not None else config.get("max_doppler_hz")
    if maximum is not None and (not np.isfinite(maximum) or maximum < 0):
        raise ValueError("max_doppler_hz must be finite and nonnegative")
    if model == "awgn":
        return ChannelRealization(
            np.array([0]), np.array([1.0 + 0j]), np.array([0.0]), sample_rate_hz
        )
    if model in {"flat_rayleigh", "flat_rician"}:
        if model == "flat_rician":
            gains = rician_gains(
                rng,
                float(config.get("k_factor_db", 6.0)),
                los_phase=float(config.get("los_phase", 0.0)),
            )
        else:
            gains = rayleigh_gains(rng)
        if "gain" in config:
            gains = np.array([_gain(config["gain"])])
        doppler = float(config.get("doppler_hz", 0.0))
        if maximum is not None:
            doppler = float(maximum) * (-1 if doppler < 0 else 1)
        return ChannelRealization(np.array([0]), gains, np.array([doppler]), sample_rate_hz)

    from_profile = config.get("paths") is None
    if from_profile:
        profile_name = str(config.get("profile") or "urban_vehicle")
        paths = get_profile(profile_name, maximum)
    else:
        paths = list(config["paths"])
    if not paths:
        raise ValueError("multipath channels need at least one path")
    delays, powers, fixed, dopplers, supplied_doppler = [], [], [], [], []
    for path in paths:
        if not isinstance(path, Mapping):
            raise ValueError("each channel path must be a mapping")
        delay = path.get("delay_samples", 0)
        if isinstance(delay, bool) or not isinstance(delay, Integral) or delay < 0:
            raise ValueError("delay_samples must be a nonnegative integer")
        delays.append(delay)
        deterministic = _gain(path["gain"]) if "gain" in path else None
        power_db = float(path.get("power_db", 0.0))
        if not np.isfinite(power_db):
            raise ValueError("power_db must be finite")
        with np.errstate(over="ignore"):
            power = (
                abs(deterministic) ** 2
                if deterministic is not None
                else float(np.power(10.0, power_db / 10))
            )
        if not np.isfinite(power):
            raise ValueError("path power is too large to represent")
        powers.append(power)
        fixed.append(deterministic)
        dopplers.append(float(path.get("doppler_hz", 0.0)))
        supplied_doppler.append("doppler_hz" in path)
    powers_array = np.asarray(powers)
    expected_power = float(np.sum(powers_array))
    if not np.isfinite(expected_power) or expected_power <= 0:
        raise ValueError("total configured path power must be finite and positive")
    gains = rayleigh_gains(rng, len(paths)) * np.sqrt(powers_array)
    for index, deterministic in enumerate(fixed):
        if deterministic is not None:
            gains[index] = deterministic
    if bool(config.get("normalize_power", True)):
        gains /= np.sqrt(expected_power)
    frequencies = np.asarray(dopplers, dtype=float)
    if not np.all(np.isfinite(frequencies)):
        raise ValueError("doppler_hz must be finite")
    if model == "static_tdl":
        if not from_profile and np.any(frequencies != 0):
            raise ValueError("static_tdl paths must have zero Doppler")
        frequencies[:] = 0
    elif maximum is not None and not from_profile:
        provided = np.asarray(supplied_doppler)
        largest = float(np.max(np.abs(frequencies)))
        if largest > 0:
            frequencies *= float(maximum) / largest
        missing = ~provided
        if np.any(missing):
            frequencies[missing] = isotropic_dopplers(rng, int(missing.sum()), float(maximum))
    return ChannelRealization(np.asarray(delays), gains, frequencies, sample_rate_hz)
