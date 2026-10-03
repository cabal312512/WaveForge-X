"""Fully occupied, single-block cyclic-prefix OFDM."""

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .base import Waveform


class OFDM(Waveform):
    """All N information symbols map to N consecutive FFT bins.

    There are no null, pilot, or DC exclusions in the baseline. FFT bins use
    NumPy's native ordering; ``fftshift`` is for visualization only.
    """

    name = "ofdm"

    def synthesis(self, symbols: ArrayLike) -> NDArray[np.complex128]:
        return np.fft.ifft(self._vector(symbols, self.n_symbols), norm="ortho")

    def analysis(self, samples: ArrayLike) -> NDArray[np.complex128]:
        return np.fft.fft(self._vector(samples, self.n_symbols), norm="ortho")
