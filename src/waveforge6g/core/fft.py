"""Unitary FFT convention: both transforms have a 1/sqrt(N) factor."""

import numpy as np
from numpy.typing import ArrayLike, NDArray


def fft_unitary(x: ArrayLike, axis: int = -1) -> NDArray[np.complex128]:
    """DFT along ``axis`` with negative exponential and energy preservation."""
    return np.fft.fft(np.asarray(x), axis=axis, norm="ortho")


def ifft_unitary(x: ArrayLike, axis: int = -1) -> NDArray[np.complex128]:
    """Inverse DFT along ``axis`` with positive exponential and unitary scaling."""
    return np.fft.ifft(np.asarray(x), axis=axis, norm="ortho")
