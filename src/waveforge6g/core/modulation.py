"""Gray-labeled BPSK and square QAM with ensemble-average symbol energy one.

Bits within a symbol are most-significant first. For square QAM the first
half label the real axis, the second half label the imaginary axis. Increasing
axis amplitudes carry binary-reflected Gray labels. QPSK is square 4-QAM;
BPSK maps 0 to +1 and 1 to -1. Energy normalization is over the alphabet,
not each random frame (which would alter the physical noise convention).
"""

import numpy as np
from numpy.typing import ArrayLike, NDArray

_ORDERS = {"bpsk": 2, "qpsk": 4, "qam16": 16, "qam64": 64}
_ALIASES = {"16qam": "qam16", "64qam": "qam64", "4qam": "qpsk", "qam4": "qpsk"}


def _name(modulation: str) -> str:
    name = str(modulation).lower().replace("-", "").replace("_", "")
    name = _ALIASES.get(name, name)
    if name not in _ORDERS:
        raise ValueError(f"Unsupported modulation {modulation!r}; choose BPSK, QPSK, QAM16, QAM64")
    return name


def bits_per_symbol(modulation: str) -> int:
    """Information bits per uncoded complex symbol: 1, 2, 4, or 6."""
    return _ORDERS[_name(modulation)].bit_length() - 1


def constellation(modulation: str) -> NDArray[np.complex128]:
    """Return complex alphabet in binary-label integer order, shape ``[M]``."""
    name = _name(modulation)
    if name == "bpsk":
        return np.array([1.0, -1.0], dtype=np.complex128)
    order = _ORDERS[name]
    side = int(np.sqrt(order))
    natural = np.arange(side)
    gray = natural ^ (natural >> 1)
    amplitudes_by_label = np.empty(side, dtype=np.float64)
    amplitudes_by_label[gray] = 2 * natural - (side - 1)
    alphabet = amplitudes_by_label[:, None] + 1j * amplitudes_by_label[None, :]
    return (alphabet / np.sqrt(2 * (order - 1) / 3)).ravel()


def modulate(bits: ArrayLike, modulation: str) -> NDArray[np.complex128]:
    """Map flat binary ``bits`` to flat normalized complex symbols."""
    bit_array = np.asarray(bits)
    width = bits_per_symbol(modulation)
    if bit_array.ndim != 1 or np.any((bit_array != 0) & (bit_array != 1)):
        raise ValueError("bits must be a one-dimensional array containing only 0 and 1")
    if bit_array.size % width:
        raise ValueError(f"Bit count must be divisible by {width} for {modulation}")
    weights = 1 << np.arange(width - 1, -1, -1, dtype=np.int64)
    labels = bit_array.reshape(-1, width).astype(np.int64) @ weights
    return constellation(modulation)[labels]


def demodulate(symbols: ArrayLike, modulation: str) -> NDArray[np.uint8]:
    """Hard Euclidean decisions returning flat MSB-first uint8 bits.

    The square-QAM slicer treats each axis independently, avoiding an
    ``n_symbols x constellation_size`` distance matrix. Exact ties resolve
    toward the larger amplitude on each QAM axis and +1 for BPSK.
    """
    values = np.asarray(symbols, dtype=np.complex128)
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise ValueError("symbols must be a finite one-dimensional complex array")
    name = _name(modulation)
    width = bits_per_symbol(name)
    if name == "bpsk":
        return (values.real < 0).astype(np.uint8)
    order = _ORDERS[name]
    side = int(np.sqrt(order))
    scaled = values * np.sqrt(2 * (order - 1) / 3)

    def axis_labels(axis: NDArray[np.float64]) -> NDArray[np.int64]:
        natural = np.clip(np.floor((axis + side) / 2), 0, side - 1).astype(np.int64)
        return natural ^ (natural >> 1)

    labels = (axis_labels(scaled.real) << (width // 2)) | axis_labels(scaled.imag)
    shifts = np.arange(width - 1, -1, -1)
    return ((labels[:, None] >> shifts) & 1).astype(np.uint8).ravel()
