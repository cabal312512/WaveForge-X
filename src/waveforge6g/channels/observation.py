"""Causal noisy decision contexts with no channel truth or future references."""

from collections import deque
from dataclasses import asdict, dataclass
from numbers import Integral

import numpy as np

from .nonstationary.state import TrueChannelState


@dataclass(frozen=True, slots=True)
class ObservedContext:
    """The complete public context supplied to online decision algorithms."""

    time: int
    estimated_snr_db: float
    estimated_doppler_hz: float
    estimated_delay_spread_s: float
    estimated_variation: float
    estimated_coherence_time_s: float
    estimated_frequency_selectivity: float

    def __post_init__(self) -> None:
        if isinstance(self.time, bool) or not isinstance(self.time, Integral) or self.time < 0:
            raise ValueError("observed time must be a nonnegative integer")
        values = tuple(value for name, value in asdict(self).items() if name != "time")
        if not all(np.isfinite(value) for value in values):
            raise ValueError("observation estimates must all be finite")
        if min(values[1:]) < 0:
            raise ValueError("only estimated SNR may be negative")

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


class ObservationModel:
    """Measurement noise/bias then causal delivery delay, independently seeded.

    Only numeric measurement tuples are retained internally. ``perfect=True``
    disables errors, biases, and latency to define the perfect-context baseline.
    Context time is always the current decision epoch, never the source epoch.
    """

    def __init__(self, config: dict | None, seed: int) -> None:
        config = dict(config or {})
        self.perfect = bool(config.get("perfect", False))
        delay = config.get("delay_frames", 0)
        if isinstance(delay, bool) or not isinstance(delay, Integral) or delay < 0:
            raise ValueError("delay_frames must be a nonnegative integer")
        self.delay_frames = 0 if self.perfect else int(delay)
        self._rng = np.random.default_rng(seed)
        self._next = 0
        self._history: deque[tuple[int, float, float, float]] = deque(maxlen=self.delay_frames + 1)
        self._previous_estimates: np.ndarray | None = None
        self.last_source_time: int | None = None
        self._noise = {
            "snr_std": float(config.get("snr_std", 1.0)),
            "doppler_relative_error": float(config.get("doppler_relative_error", 0.1)),
            "delay_spread_relative_error": float(config.get("delay_spread_relative_error", 0.1)),
            "doppler_std_hz": float(config.get("doppler_std_hz", 0.0)),
            "delay_spread_std_s": float(config.get("delay_spread_std_s", 0.0)),
        }
        if not all(np.isfinite(value) and value >= 0 for value in self._noise.values()):
            raise ValueError(
                "observation standard deviations and relative errors must be finite and nonnegative"
            )
        biases = config.get("biases", {})
        self._bias = np.array(
            [
                config.get("snr_bias_db", biases.get("snr_db", 0.0)),
                config.get("doppler_bias_hz", biases.get("doppler_hz", 0.0)),
                config.get("delay_spread_bias_s", biases.get("delay_spread_s", 0.0)),
            ],
            dtype=float,
        )
        self._scales = np.array(
            [
                config.get("snr_variation_scale_db", 10.0),
                config.get("doppler_variation_scale_hz", 1000.0),
                config.get("delay_variation_scale_s", 1e-4),
            ],
            dtype=float,
        )
        self._bandwidth = float(
            config.get("reference_bandwidth_hz", config.get("sample_rate_hz", 64000.0))
        )
        self._coherence_cap = float(config.get("coherence_cap_s", 1.0))
        if not np.all(np.isfinite(self._bias)):
            raise ValueError("observation biases must be finite")
        if not np.all(np.isfinite(self._scales)) or np.any(self._scales <= 0):
            raise ValueError("observation variation scales must be finite and positive")
        if not np.isfinite(self._bandwidth) or self._bandwidth <= 0:
            raise ValueError("reference_bandwidth_hz must be finite and positive")
        if not np.isfinite(self._coherence_cap) or self._coherence_cap <= 0:
            raise ValueError("coherence_cap_s must be finite and positive")

    def observe(self, state: TrueChannelState) -> ObservedContext:
        if state.time != self._next:
            raise ValueError(f"observation model requires next sequential epoch {self._next}")
        measurement = np.array(
            [state.snr_db, state.max_doppler_hz, state.delay_spread_s], dtype=float
        )
        if not self.perfect:
            noise = self._noise
            standard_deviations = np.array(
                [
                    noise["snr_std"],
                    np.hypot(
                        noise["doppler_std_hz"], noise["doppler_relative_error"] * measurement[1]
                    ),
                    np.hypot(
                        noise["delay_spread_std_s"],
                        noise["delay_spread_relative_error"] * measurement[2],
                    ),
                ]
            )
            measurement += self._bias + standard_deviations * self._rng.standard_normal(3)
        measurement[1:] = np.maximum(measurement[1:], 0.0)
        self._history.append((int(state.time), *(float(value) for value in measurement)))
        # Startup holds the first available measurement, never future-fills latency.
        source_time, snr, doppler, delay = self._history[0]
        estimates = np.array([snr, doppler, delay])
        variation = 0.0
        if self._previous_estimates is not None:
            distance = float(
                np.linalg.norm((estimates - self._previous_estimates) / self._scales) / np.sqrt(3)
            )
            variation = float(-np.expm1(-distance))
        coherence = (
            self._coherence_cap
            if doppler == 0
            else min(self._coherence_cap, 1 / (2 * np.pi * doppler))
        )
        selectivity = float(-np.expm1(-2 * np.pi * self._bandwidth * delay))
        context = ObservedContext(
            time=int(state.time),
            estimated_snr_db=snr,
            estimated_doppler_hz=doppler,
            estimated_delay_spread_s=delay,
            estimated_variation=variation,
            estimated_coherence_time_s=float(coherence),
            estimated_frequency_selectivity=selectivity,
        )
        self.last_source_time = source_time
        self._previous_estimates = estimates
        self._next += 1
        return context
