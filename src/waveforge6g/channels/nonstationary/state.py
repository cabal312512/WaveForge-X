"""Ground truth kept separate from decision-time observations."""

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class TrueChannelState:
    """Realized frame-level channel descriptors, never selector input."""

    time: int
    snr_db: float
    max_doppler_hz: float
    delay_spread_s: float
    k_factor_db: float
    path_count: int
    correlation: float
    fractional_doppler_severity: float
    stationarity: float
    path_power_distribution: tuple[float, ...]
    change_point: bool = False

    def __post_init__(self) -> None:
        powers = tuple(float(value) for value in self.path_power_distribution)
        if self.time < 0 or self.path_count < 1 or len(powers) != self.path_count:
            raise ValueError("state time and path count must agree with the power distribution")
        numeric = (
            self.snr_db,
            self.max_doppler_hz,
            self.delay_spread_s,
            self.k_factor_db,
            self.correlation,
            self.fractional_doppler_severity,
            self.stationarity,
            *powers,
        )
        if not all(np.isfinite(value) for value in numeric):
            raise ValueError("true channel descriptors must be finite")
        if self.max_doppler_hz < 0 or self.delay_spread_s < 0:
            raise ValueError("Doppler and RMS delay spread must be nonnegative")
        if not all(
            0 <= value <= 1
            for value in (
                self.correlation,
                self.fractional_doppler_severity,
                self.stationarity,
            )
        ):
            raise ValueError("correlation, severity and stationarity must lie in [0,1]")
        if any(value < 0 for value in powers) or not np.isclose(sum(powers), 1.0):
            raise ValueError("expected path powers must be nonnegative and sum to one")
        object.__setattr__(self, "path_power_distribution", powers)

    def to_dict(self) -> dict:
        result = asdict(self)
        result["path_power_distribution"] = list(self.path_power_distribution)
        return result
