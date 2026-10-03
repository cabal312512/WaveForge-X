"""Finite-block linear time-varying paths and their exact Hermitian adjoint."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _immutable_array(values: ArrayLike, dtype: np.dtype) -> np.ndarray:
    """Own immutable backing bytes so callers cannot re-enable writeability."""
    array = np.asarray(values, dtype=dtype)
    if array.ndim != 1:
        raise ValueError("channel path arrays must be one-dimensional")
    return np.frombuffer(array.tobytes(), dtype=dtype)


@dataclass(frozen=True, eq=False)
class ChannelRealization:
    """One shared set of paths; noise is added by the experiment separately.

    Time n=0 is the first transmitted guard sample. Each apply starts at this
    same origin and assumes zero prehistory. Samples beyond the input's length
    are truncated. The adjoint includes that exact truncation.
    """

    delays: NDArray[np.int64]
    gains: NDArray[np.complex128]
    dopplers_hz: NDArray[np.float64]
    sample_rate_hz: float

    def __post_init__(self) -> None:
        raw_delays = np.asarray(self.delays)
        if raw_delays.ndim != 1 or not np.all(np.isfinite(raw_delays)):
            raise ValueError("delays must be a finite one-dimensional integer vector")
        if np.iscomplexobj(raw_delays) or np.any(raw_delays != np.floor(raw_delays)):
            raise ValueError("fractional path delays are not supported")
        if np.any(raw_delays < 0) or np.any(raw_delays >= np.iinfo(np.int64).max):
            raise ValueError("path delays must be nonnegative int64 values")
        delays = _immutable_array(raw_delays, np.dtype(np.int64))
        gains = _immutable_array(self.gains, np.dtype(np.complex128))
        dopplers = _immutable_array(self.dopplers_hz, np.dtype(np.float64))
        if delays.size == 0 or not (delays.size == gains.size == dopplers.size):
            raise ValueError("channel arrays must have the same nonzero length")
        if not np.all(np.isfinite(gains)) or not np.all(np.isfinite(dopplers)):
            raise ValueError("gains and Doppler frequencies must be finite")
        rate = float(self.sample_rate_hz)
        if not np.isfinite(rate) or rate <= 0:
            raise ValueError("sample_rate_hz must be finite and positive")
        if np.any(np.abs(dopplers) >= rate / 2):
            raise ValueError("Doppler frequencies must lie strictly within sample-rate Nyquist")
        object.__setattr__(self, "delays", delays)
        object.__setattr__(self, "gains", gains)
        object.__setattr__(self, "dopplers_hz", dopplers)
        object.__setattr__(self, "sample_rate_hz", rate)

    @property
    def max_delay(self) -> int:
        return int(np.max(self.delays))

    @property
    def n_paths(self) -> int:
        return int(self.delays.size)

    @property
    def max_doppler_hz(self) -> float:
        return float(np.max(np.abs(self.dopplers_hz)))

    def _vector(self, values: ArrayLike) -> NDArray[np.complex128]:
        vector = np.asarray(values, dtype=np.complex128)
        if vector.ndim != 1 or not np.all(np.isfinite(vector)):
            raise ValueError("channel input must be a finite one-dimensional vector")
        return vector

    def apply(self, transmitted: ArrayLike) -> NDArray[np.complex128]:
        """y[n] = sum_l g_l exp(+j 2 pi f_l n/fs) x[n-d_l]."""
        signal = self._vector(transmitted)
        received = np.zeros_like(signal)
        for delay, gain, doppler in zip(self.delays, self.gains, self.dopplers_hz, strict=True):
            if delay >= signal.size:
                continue
            indices = np.arange(delay, signal.size)
            phase = np.exp(2j * np.pi * doppler * indices / self.sample_rate_hz)
            received[delay:] += gain * phase * signal[: signal.size - delay]
        return received

    def adjoint(self, received: ArrayLike) -> NDArray[np.complex128]:
        """Exact adjoint: conjugate phase at output time m+d, then shift back."""
        vector = self._vector(received)
        result = np.zeros_like(vector)
        for delay, gain, doppler in zip(self.delays, self.gains, self.dopplers_hz, strict=True):
            if delay >= vector.size:
                continue
            indices = np.arange(delay, vector.size)
            phase = np.exp(-2j * np.pi * doppler * indices / self.sample_rate_hz)
            result[: vector.size - delay] += gain.conjugate() * phase * vector[delay:]
        return result

    def time_response(self, sample_count: int) -> NDArray[np.complex128]:
        """Path-by-time gains for channel diagnostics, shape (paths, samples)."""
        if not isinstance(sample_count, int | np.integer) or sample_count < 0:
            raise ValueError("sample_count must be a nonnegative integer")
        indices = np.arange(sample_count)
        return self.gains[:, None] * np.exp(
            2j * np.pi * self.dopplers_hz[:, None] * indices[None, :] / self.sample_rate_hz
        )

    def to_dict(self) -> dict:
        """JSON-safe full realization for audits and exact reconstruction."""
        return {
            "delays": self.delays.tolist(),
            "gains": [[float(g.real), float(g.imag)] for g in self.gains],
            "dopplers_hz": self.dopplers_hz.tolist(),
            "sample_rate_hz": self.sample_rate_hz,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ChannelRealization":
        """Reconstruct output of ``to_dict`` without any random draw."""
        return cls(
            delays=np.asarray(data["delays"]),
            gains=np.asarray([complex(*gain) for gain in data["gains"]]),
            dopplers_hz=np.asarray(data["dopplers_hz"]),
            sample_rate_hz=data["sample_rate_hz"],
        )
