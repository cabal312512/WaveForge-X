"""Sequential channel engine reusing the validated finite-frame channel kernel."""

from copy import deepcopy
from numbers import Integral

import numpy as np

from ..doubly_selective import ChannelRealization
from .cluster_dynamics import ClusterDynamics
from .scenarios import scenario_config
from .state import TrueChannelState
from .trajectory import ScalarTrajectory


class NonStationaryChannel:
    """Causal correlated paths with slowly or abruptly changing controls.

    ``step(t)`` can only request the next epoch. No future realization sequence
    is constructed or exposed. All external side effects remain in the runner.
    """

    def __init__(
        self, config: dict, seed: int, n_symbols: int, cp_length: int, sample_rate_hz: float
    ) -> None:
        if not isinstance(n_symbols, Integral) or isinstance(n_symbols, bool) or n_symbols < 1:
            raise ValueError("n_symbols must be a positive integer")
        if (
            not isinstance(cp_length, Integral)
            or isinstance(cp_length, bool)
            or not 0 <= cp_length <= n_symbols
        ):
            raise ValueError("cp_length must be an integer between 0 and n_symbols")
        if not np.isfinite(sample_rate_hz) or sample_rate_hz <= 0:
            raise ValueError("sample_rate_hz must be finite and positive")
        self.config = scenario_config(deepcopy(config))
        self.epochs = self.config["epochs"]
        self.n_symbols, self.cp_length = int(n_symbols), int(cp_length)
        self.sample_rate_hz = float(sample_rate_hz)
        self.frame_duration_s = (self.n_symbols + self.cp_length) / self.sample_rate_hz
        self._next = 0
        self._previous_controls: dict | None = None
        specs = self.config["trajectories"]
        allowed = {
            "snr_db",
            "max_doppler_hz",
            "delay_samples",
            "delay_spread_s",
            "k_factor_db",
            "correlation",
            "path_count",
            "fractional_doppler_severity",
        }
        if set(specs) - allowed:
            raise ValueError(
                f"unknown channel trajectory parameters {sorted(set(specs) - allowed)}"
            )
        streams = np.random.SeedSequence(seed).spawn(len(specs) + 1)
        self._trajectories = {
            key: ScalarTrajectory(spec, self.epochs, np.random.default_rng(streams[index]))
            for index, (key, spec) in enumerate(sorted(specs.items()))
        }
        self._clusters = ClusterDynamics(
            self.config["cluster_dynamics"], np.random.default_rng(streams[-1])
        )

    @property
    def last_events(self) -> tuple[str, ...]:
        """Past/current synthetic birth/death audit events; not selector context."""
        return tuple(self._clusters.events)

    @property
    def active_path_ids(self) -> tuple[int, ...]:
        return tuple(path.identifier for path in self._clusters.paths)

    def __iter__(self):
        return self

    def __next__(self) -> tuple[TrueChannelState, ChannelRealization]:
        return self.step(self._next)

    def _delays(self, values: dict, powers: np.ndarray) -> np.ndarray:
        positions = np.array([path.delay_fraction for path in self._clusters.paths])
        if positions.size == 1 or np.ptp(positions) == 0:
            return np.zeros(positions.size, dtype=np.int64)
        positions = (positions - positions.min()) / np.ptp(positions)
        if "delay_spread_s" in values:
            if values["delay_spread_s"] < 0:
                raise ValueError("target delay_spread_s must be nonnegative")
            position_rms = float(
                np.sqrt(np.sum(powers * positions**2) - np.sum(powers * positions) ** 2)
            )
            support = values["delay_spread_s"] * self.sample_rate_hz / max(position_rms, 1e-15)
        else:
            support = values["delay_samples"]
            if support < 0:
                raise ValueError("delay_samples must be nonnegative")
        # CP is a physical resource constraint, not an implicit IBI approximation.
        support = min(float(support), self.cp_length)
        return np.rint(positions * support).astype(np.int64)

    def step(self, time: int) -> tuple[TrueChannelState, ChannelRealization]:
        if isinstance(time, bool) or not isinstance(time, int | np.integer) or time != self._next:
            raise ValueError(f"nonstationary channel requires next sequential epoch {self._next}")
        if time >= self.epochs:
            raise StopIteration
        values = {name: trajectory.step(time) for name, trajectory in self._trajectories.items()}
        correlation = float(values["correlation"])
        maximum = float(values["max_doppler_hz"])
        severity = float(values["fractional_doppler_severity"])
        if not 0 <= correlation <= 1 or not 0 <= severity <= 1:
            raise ValueError("correlation and fractional_doppler_severity must lie in [0,1]")
        if not 0 <= maximum < self.sample_rate_hz / 2:
            raise ValueError("max_doppler_hz must be nonnegative and strictly below Nyquist")
        target = int(np.rint(values["path_count"]))
        if not 1 <= target <= self._clusters.max_paths:
            raise ValueError("path_count trajectory must stay between 1 and cluster max_paths")
        event = self._clusters.update(time, target, correlation, self.frame_duration_s)
        powers = self._clusters.expected_powers(time)
        delays = self._delays(values, powers)
        ratios = np.array([path.doppler_fraction for path in self._clusters.paths])
        ratios /= max(float(np.max(np.abs(ratios))), 1e-15)
        bin_hz = self.sample_rate_hz / self.n_symbols
        raw_bins = maximum * ratios / bin_hz
        # severity=0 snaps to the useful-frame Doppler grid, severity=1 is continuous.
        limit_bins = np.floor(maximum / bin_hz)
        grid_bins = np.clip(np.rint(raw_bins), -limit_bins, limit_bins)
        frequencies = ((1 - severity) * grid_bins + severity * raw_bins) * bin_hz
        frequencies = np.clip(frequencies, -maximum, maximum)
        log_k = values["k_factor_db"] * np.log(10) / 10
        scatter = float(np.exp(-np.logaddexp(0.0, log_k)))
        specular = float(np.exp(-np.logaddexp(0.0, -log_k)))
        coefficients = []
        for index, path in enumerate(self._clusters.paths):
            path.previous_doppler_hz = float(frequencies[index])
            coefficient = (
                np.sqrt(specular) * np.exp(1j * path.los_phase) + np.sqrt(scatter) * path.diffuse
            )
            coefficients.append(
                np.sqrt(powers[index]) * coefficient * np.exp(1j * path.oscillator_phase)
            )
        realization = ChannelRealization(
            delays, np.asarray(coefficients), frequencies, self.sample_rate_hz
        )
        seconds = delays / self.sample_rate_hz
        mean_delay = float(powers @ seconds)
        rms_delay = float(np.sqrt(powers @ (seconds - mean_delay) ** 2))
        normalized_doppler = frequencies / bin_hz
        fractional = float(
            np.clip(2 * powers @ np.abs(normalized_doppler - np.rint(normalized_doppler)), 0, 1)
        )
        # Dimensionless descriptor of control variation, not a statistical test.
        variation = 0.0
        if self._previous_controls is not None:
            scales = {
                "snr_db": 10,
                "max_doppler_hz": bin_hz,
                "delay_samples": max(1, self.cp_length),
                "delay_spread_s": max(1, self.cp_length) / self.sample_rate_hz,
                "k_factor_db": 10,
                "correlation": 1,
                "path_count": 5,
                "fractional_doppler_severity": 1,
            }
            variation = sum(
                abs(value - self._previous_controls[name]) / scales[name]
                for name, value in values.items()
            )
        scripted_change = any(trajectory.change_point for trajectory in self._trajectories.values())
        state = TrueChannelState(
            time=int(time),
            snr_db=float(values["snr_db"]),
            max_doppler_hz=realization.max_doppler_hz,
            delay_spread_s=rms_delay,
            k_factor_db=float(values["k_factor_db"]),
            path_count=len(powers),
            correlation=correlation,
            fractional_doppler_severity=fractional,
            stationarity=float(np.exp(-variation - (0.5 if event else 0.0))),
            path_power_distribution=tuple(float(power) for power in powers),
            change_point=bool(scripted_change or event),
        )
        self._previous_controls = values
        self._next += 1
        return state, realization
