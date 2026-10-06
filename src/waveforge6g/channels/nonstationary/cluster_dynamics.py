"""Causal synthetic cluster and path birth/death with finite power ramps."""

from dataclasses import dataclass

import numpy as np


@dataclass
class PathState:
    identifier: int
    cluster: int
    born: int
    lifetime: int | None
    power: float
    delay_fraction: float
    doppler_fraction: float
    diffuse: complex
    los_phase: float
    oscillator_phase: float = 0.0
    previous_doppler_hz: float = 0.0
    dying_at: int | None = None


class ClusterDynamics:
    """Own path identities; only causal state and past phase are retained."""

    def __init__(self, config: dict, rng: np.random.Generator) -> None:
        self.config = dict(config)
        self.rng = rng
        self.paths: list[PathState] = []
        self.events: list[str] = []
        self._next_id = 0
        self._next_cluster = 0
        self._previous_target: int | None = None
        for key in (
            "birth_probability",
            "death_probability",
            "cluster_birth_probability",
            "cluster_death_probability",
        ):
            probability = float(self.config.get(key, 0.0))
            if not 0 <= probability <= 1:
                raise ValueError(f"{key} must lie in [0,1]")
        self.max_paths = int(self.config.get("max_paths", 10))
        self.ramp = int(self.config.get("power_ramp_frames", 3))
        self.cluster_size = int(self.config.get("cluster_size", 2))
        if min(self.max_paths, self.cluster_size) < 1 or self.ramp < 0:
            raise ValueError("max_paths/cluster_size must be positive and ramp nonnegative")
        lifetime = self.config.get("lifetime")
        if lifetime is not None:
            limits = lifetime if isinstance(lifetime, list | tuple) else [lifetime, lifetime]
            if len(limits) != 2 or min(limits) < 1 or limits[1] < limits[0]:
                raise ValueError("lifetime must be positive or [minimum, maximum]")
        for key in ("delay_drift_std", "doppler_drift_std", "power_drift_std"):
            if float(self.config.get(key, 0.0)) < 0:
                raise ValueError(f"{key} must be nonnegative")

    def _normal(self) -> complex:
        return complex(*self.rng.standard_normal(2)) / np.sqrt(2)

    def _birth(self, time: int, count: int, initial: bool = False) -> None:
        cluster = self._next_cluster
        self._next_cluster += 1
        for _ in range(min(count, self.max_paths - len(self.paths))):
            lifetime = self.config.get("lifetime")
            if isinstance(lifetime, list | tuple):
                lifetime = int(self.rng.integers(lifetime[0], lifetime[1] + 1))
            elif lifetime is not None:
                lifetime = int(lifetime)
            path = PathState(
                identifier=self._next_id,
                cluster=cluster,
                born=time - self.ramp if initial else time,
                lifetime=lifetime,
                power=float(np.exp(self.rng.normal(0, 0.5))),
                delay_fraction=float(self.rng.uniform()),
                doppler_fraction=float(self.rng.uniform(-1, 1)),
                diffuse=self._normal(),
                los_phase=float(self.rng.uniform(-np.pi, np.pi)),
            )
            self.paths.append(path)
            self._next_id += 1
            self.events.append(f"path_birth:{path.identifier}")

    def update(self, time: int, target: int, correlation: float, frame_duration_s: float) -> bool:
        self.events = []
        # Advance the accumulated oscillator using the previous frame's frequency.
        for path in self.paths:
            path.oscillator_phase = float(
                (path.oscillator_phase + 2 * np.pi * path.previous_doppler_hz * frame_duration_s)
                % (2 * np.pi)
            )
            path.diffuse = (
                correlation * path.diffuse + np.sqrt(max(0, 1 - correlation**2)) * self._normal()
            )
            path.delay_fraction = float(
                np.clip(
                    path.delay_fraction
                    + self.rng.normal(0, self.config.get("delay_drift_std", 0.0)),
                    0,
                    1,
                )
            )
            path.doppler_fraction = float(
                np.clip(
                    path.doppler_fraction
                    + self.rng.normal(0, self.config.get("doppler_drift_std", 0.0)),
                    -1,
                    1,
                )
            )
            path.power = float(
                np.exp(
                    np.clip(
                        np.log(path.power)
                        + self.rng.normal(0, self.config.get("power_drift_std", 0.0)),
                        -10,
                        10,
                    )
                )
            )
        if self._previous_target is None:
            self._birth(time, target, initial=True)
        elif target != self._previous_target:
            alive = [path for path in self.paths if path.dying_at is None]
            if len(alive) < target:
                self._birth(time, target - len(alive))
            else:
                for path in alive[target:]:
                    path.dying_at = time
                    self.events.append(f"path_death_started:{path.identifier}")
        self._previous_target = target
        for path in self.paths:
            expired = path.lifetime is not None and time - path.born >= path.lifetime
            random_death = self.rng.random() < self.config.get("death_probability", 0.0)
            if path.dying_at is None and (expired or random_death) and len(self.paths) > 1:
                path.dying_at = time
                self.events.append(f"path_death_started:{path.identifier}")
        if self.paths and self.rng.random() < self.config.get("cluster_death_probability", 0.0):
            cluster = int(self.rng.choice([path.cluster for path in self.paths]))
            for path in self.paths:
                if path.cluster == cluster and path.dying_at is None:
                    path.dying_at = time
                    self.events.append(f"cluster_death:{cluster}")
        remaining = []
        for path in self.paths:
            if path.dying_at is not None and time - path.dying_at >= max(1, self.ramp):
                self.events.append(f"path_removed:{path.identifier}")
            else:
                remaining.append(path)
        self.paths = remaining
        if self.rng.random() < self.config.get("birth_probability", 0.0):
            self._birth(time, 1)
        if self.rng.random() < self.config.get("cluster_birth_probability", 0.0):
            self._birth(time, self.cluster_size)
        if not self.paths:
            self._birth(time, 1)
        return bool(time > 0 and self.events)

    def expected_powers(self, time: int) -> np.ndarray:
        weights = []
        for path in self.paths:
            birth = 1.0 if self.ramp == 0 else min(1.0, (time - path.born + 1) / self.ramp)
            death = (
                1.0
                if path.dying_at is None
                else max(0.0, 1 - (time - path.dying_at) / max(1, self.ramp))
            )
            weights.append(path.power * (birth * death) ** 2)
        weights = np.asarray(weights, dtype=float)
        return weights / weights.sum()
