"""Causal scalar parameter generators; no future-state array is exposed."""

from collections.abc import Mapping
from copy import deepcopy

import numpy as np


class ScalarTrajectory:
    """Evaluate one configured scalar in strictly increasing epoch order.

    Supports constant, linear, sinusoidal, random_walk, piecewise_constant,
    piecewise_linear, abrupt, and smooth. ``points`` are [epoch, value] pairs.
    A deterministic future schedule is internal generator configuration, not
    an observation or decision API.
    """

    KINDS = {
        "constant",
        "linear",
        "sinusoidal",
        "random_walk",
        "piecewise_constant",
        "piecewise_linear",
        "abrupt",
        "smooth",
    }

    def __init__(self, spec: float | dict, epochs: int, rng: np.random.Generator) -> None:
        self._spec = deepcopy(dict(spec)) if isinstance(spec, Mapping) else {"value": float(spec)}
        self._kind = str(self._spec.get("type", self._spec.get("kind", "constant")))
        if self._kind not in self.KINDS:
            raise ValueError(f"unknown trajectory kind {self._kind!r}")
        self._epochs = epochs
        self._rng = rng
        self._next = 0
        self._value = float(self._spec.get("initial", self._spec.get("start", 0.0)))
        self.change_point = False
        self._points: np.ndarray | None = None
        if self._kind.startswith("piecewise"):
            points = self._spec.get("points")
            if points is None:
                values = self._spec.get("values", [])
                breaks = self._spec.get("breakpoints", [])
                if len(values) != len(breaks) + 1:
                    raise ValueError("piecewise values need one more element than breakpoints")
                points = list(zip([0, *breaks], values, strict=True))
            self._points = np.asarray(points, dtype=float)
            if (
                self._points.ndim != 2
                or self._points.shape[1] != 2
                or self._points.shape[0] < 1
                or not np.all(np.isfinite(self._points))
                or self._points[0, 0] != 0
                or np.any(np.diff(self._points[:, 0]) <= 0)
                or np.any(self._points[:, 0] != np.floor(self._points[:, 0]))
            ):
                raise ValueError(
                    "piecewise points must start at epoch 0 and increase by integer epochs"
                )

    def step(self, time: int) -> float:
        if isinstance(time, bool) or not isinstance(time, int | np.integer) or time != self._next:
            raise ValueError(f"trajectory requires the next sequential epoch {self._next}")
        if time >= self._epochs:
            raise StopIteration
        spec, kind = self._spec, self._kind
        self.change_point = False
        if kind == "constant":
            value = float(spec.get("value", spec.get("start", 0.0)))
        elif kind == "linear":
            start = float(spec.get("start", 0.0))
            slope = float(
                spec.get(
                    "slope", (float(spec.get("end", start)) - start) / max(1, self._epochs - 1)
                )
            )
            value = start + time * slope
        elif kind == "sinusoidal":
            period = float(spec.get("period", self._epochs))
            if period <= 0:
                raise ValueError("sinusoidal period must be positive")
            value = float(spec.get("offset", 0.0)) + float(spec.get("amplitude", 1.0)) * np.sin(
                2 * np.pi * time / period + float(spec.get("phase", 0.0))
            )
        elif kind == "random_walk":
            std = float(spec.get("std", spec.get("step_std", 1.0)))
            if std < 0:
                raise ValueError("random-walk std must be nonnegative")
            value = self._value + (float(self._rng.normal(0, std)) if time else 0.0)
        elif kind in {"piecewise_constant", "piecewise_linear"}:
            assert self._points is not None
            times, values = self._points.T
            if kind == "piecewise_constant":
                value = float(values[np.searchsorted(times, time, side="right") - 1])
            else:
                value = float(np.interp(time, times, values))
            self.change_point = bool(time > 0 and time in times)
        elif kind == "abrupt":
            change = int(spec.get("time", spec.get("change_at", self._epochs // 2)))
            value = float(spec.get("before", 0.0) if time < change else spec.get("after", 1.0))
            self.change_point = time == change and time > 0
        else:
            # Finite-duration smoothstep: exactly reaches both endpoints.
            begin = float(spec.get("start_time", self._epochs / 3))
            end = float(spec.get("end_time", 2 * self._epochs / 3))
            if end <= begin:
                raise ValueError("smooth end_time must exceed start_time")
            weight = float(np.clip((time - begin) / (end - begin), 0.0, 1.0))
            weight = weight**2 * (3 - 2 * weight)
            before, after = float(spec.get("before", 0.0)), float(spec.get("after", 1.0))
            value = before + weight * (after - before)
        value = float(np.clip(value, spec.get("min", -np.inf), spec.get("max", np.inf)))
        if not np.isfinite(value):
            raise ValueError("trajectory produced a nonfinite value")
        self._next += 1
        self._value = value
        return value
