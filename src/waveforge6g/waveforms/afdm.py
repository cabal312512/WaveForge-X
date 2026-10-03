"""DAFT-based AFDM with a mathematically consistent chirp-periodic prefix."""

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .base import Waveform


class AFDM(Waveform):
    """DAFT A=Lambda(c2) F Lambda(c1); transmit A^H times the symbols.

    Lambda(c)[n,n]=exp(-2j*pi*c*n**2). Real finite chirps are unitary for
    any c1,c2. Default chirps are reproducible baselines, not a promise of
    full diversity or optimality for the configured delay/Doppler support.
    """

    name = "afdm"

    def __init__(
        self, n_symbols: int, cp_length: int, c1: float | None = None, c2: float | None = None
    ) -> None:
        super().__init__(n_symbols, cp_length)
        self.c1 = float((1 if n_symbols % 2 == 0 else 2) / (2 * n_symbols) if c1 is None else c1)
        self.c2 = float(np.sqrt(2) / n_symbols**2 if c2 is None else c2)
        if not np.isfinite(self.c1) or not np.isfinite(self.c2):
            raise ValueError("AFDM chirp parameters must be finite real numbers")
        indices = np.arange(self.n_symbols, dtype=float)
        self._chirp1 = np.exp(-2j * np.pi * self.c1 * indices**2)
        self._chirp2 = np.exp(-2j * np.pi * self.c2 * indices**2)

    @property
    def prefix_phases(self) -> NDArray[np.complex128]:
        n = np.arange(-self.cp_length, 0, dtype=float)
        return np.exp(-2j * np.pi * self.c1 * (self.n_symbols**2 + 2 * self.n_symbols * n))

    def synthesis(self, symbols: ArrayLike) -> NDArray[np.complex128]:
        data = self._vector(symbols, self.n_symbols)
        return self._chirp1.conj() * np.fft.ifft(self._chirp2.conj() * data, norm="ortho")

    def analysis(self, samples: ArrayLike) -> NDArray[np.complex128]:
        data = self._vector(samples, self.n_symbols)
        return self._chirp2 * np.fft.fft(self._chirp1 * data, norm="ortho")
