"""Common finite-frame interface for unitary waveform transforms."""

from abc import ABC, abstractmethod
from numbers import Integral

import numpy as np
from numpy.typing import ArrayLike, NDArray


class Waveform(ABC):
    """Map N independent complex symbols to N useful samples plus one guard.

    Prefix insertion is not unitary. ``analysis`` is the adjoint/inverse of
    ``synthesis`` only; removing a prefix is not the adjoint of inserting it.
    """

    name: str

    def __init__(self, n_symbols: int, cp_length: int) -> None:
        if isinstance(n_symbols, bool) or not isinstance(n_symbols, Integral) or n_symbols < 1:
            raise ValueError("n_symbols must be a positive integer")
        if isinstance(cp_length, bool) or not isinstance(cp_length, Integral):
            raise ValueError("cp_length must be an integer")
        if not 0 <= cp_length <= n_symbols:
            raise ValueError("cp_length must lie between zero and n_symbols")
        self.n_symbols = int(n_symbols)
        self.cp_length = int(cp_length)

    def _vector(self, values: ArrayLike, length: int) -> NDArray[np.complex128]:
        array = np.asarray(values, dtype=np.complex128)
        if array.ndim != 1 or array.size != length:
            raise ValueError(f"expected a one-dimensional complex vector of length {length}")
        if not np.all(np.isfinite(array)):
            raise ValueError("waveform input must contain only finite values")
        return array

    @property
    def prefix_phases(self) -> NDArray[np.complex128]:
        """Phase multiplying useful samples [-L:] to construct the guard."""
        return np.ones(self.cp_length, dtype=np.complex128)

    @abstractmethod
    def synthesis(self, symbols: ArrayLike) -> NDArray[np.complex128]:
        """Unitary information-domain to useful time-domain transform."""

    @abstractmethod
    def analysis(self, samples: ArrayLike) -> NDArray[np.complex128]:
        """Unitary useful time-domain to information-domain transform."""

    def synthesis_adjoint(self, samples: ArrayLike) -> NDArray[np.complex128]:
        """Adjoint of the useful-sample synthesis transform."""
        return self.analysis(samples)

    def add_prefix(self, samples: ArrayLike) -> NDArray[np.complex128]:
        useful = self._vector(samples, self.n_symbols)
        if self.cp_length == 0:
            return useful.copy()
        return np.concatenate((self.prefix_phases * useful[-self.cp_length :], useful))

    def remove_prefix(self, received: ArrayLike) -> NDArray[np.complex128]:
        frame = self._vector(received, self.n_symbols + self.cp_length)
        return frame[self.cp_length :].copy()

    def prefix_adjoint(self, samples: ArrayLike) -> NDArray[np.complex128]:
        """Exact adjoint of prefix insertion, needed by matrix-free receivers."""
        frame = self._vector(samples, self.n_symbols + self.cp_length)
        useful = frame[self.cp_length :].copy()
        if self.cp_length:
            useful[-self.cp_length :] += self.prefix_phases.conj() * frame[: self.cp_length]
        return useful

    def modulate(self, symbols: ArrayLike) -> NDArray[np.complex128]:
        return self.add_prefix(self.synthesis(symbols))

    def demodulate(self, received: ArrayLike) -> NDArray[np.complex128]:
        return self.analysis(self.remove_prefix(received))
