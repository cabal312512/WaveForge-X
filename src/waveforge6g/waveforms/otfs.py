"""Rectangular-pulse, reduced-CP OTFS with explicit ISFFT/Heisenberg stages."""

from numbers import Integral

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .base import Waveform


class OTFS(Waveform):
    """Use a (delay, Doppler) grid of shape (K, M), with K*M=N.

    All flattened arrays use column-major ordering. One cyclic prefix protects
    the entire frame; this is not a per-OFDM-symbol CP implementation.
    See docs/equations/otfs.md for signs, normalization, and serialization.
    """

    name = "otfs"

    def __init__(self, n_symbols: int, cp_length: int, subcarriers: int) -> None:
        super().__init__(n_symbols, cp_length)
        if isinstance(subcarriers, bool) or not isinstance(subcarriers, Integral):
            raise ValueError("subcarriers must be a positive integer dividing n_symbols")
        if subcarriers < 1 or self.n_symbols % subcarriers:
            raise ValueError("subcarriers must be a positive integer dividing n_symbols")
        self.subcarriers = int(subcarriers)
        self.time_slots = self.n_symbols // self.subcarriers

    def _grid(self, grid: ArrayLike) -> NDArray[np.complex128]:
        array = np.asarray(grid, dtype=np.complex128)
        if array.shape != (self.subcarriers, self.time_slots):
            raise ValueError("grid must have shape (subcarriers, time_slots)")
        if not np.all(np.isfinite(array)):
            raise ValueError("OTFS grid must contain only finite values")
        return array

    def isfft(self, delay_doppler: ArrayLike) -> NDArray[np.complex128]:
        """DD to TF: F_K X F_M^H, with rows frequency and columns time."""
        grid = self._grid(delay_doppler)
        return np.fft.ifft(np.fft.fft(grid, axis=0, norm="ortho"), axis=1, norm="ortho")

    def sfft(self, time_frequency: ArrayLike) -> NDArray[np.complex128]:
        """TF to DD: F_K^H X F_M."""
        grid = self._grid(time_frequency)
        return np.fft.fft(np.fft.ifft(grid, axis=0, norm="ortho"), axis=1, norm="ortho")

    def synthesis(self, symbols: ArrayLike) -> NDArray[np.complex128]:
        delay_doppler = self._vector(symbols, self.n_symbols).reshape(
            (self.subcarriers, self.time_slots), order="F"
        )
        time_frequency = self.isfft(delay_doppler)
        time_samples = np.fft.ifft(time_frequency, axis=0, norm="ortho")
        return time_samples.reshape(-1, order="F")

    def analysis(self, samples: ArrayLike) -> NDArray[np.complex128]:
        time_samples = self._vector(samples, self.n_symbols).reshape(
            (self.subcarriers, self.time_slots), order="F"
        )
        time_frequency = np.fft.fft(time_samples, axis=0, norm="ortho")
        return self.sfft(time_frequency).reshape(-1, order="F")
