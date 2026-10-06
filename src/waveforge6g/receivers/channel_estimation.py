"""Standalone pilot LS primitive; the benchmark currently uses perfect CSI.

This is intentionally not advertised as an end-to-end pilot-aided receiver:
pilot placement, overhead, interpolation, and doubly-selective estimation are
not implemented in the benchmark configuration.
"""

import numpy as np
from numpy.typing import ArrayLike, NDArray


def pilot_ls(received_pilots: ArrayLike, transmitted_pilots: ArrayLike) -> NDArray[np.complex128]:
    """Estimate scalar channel values y_p/x_p at nonzero pilot coordinates.

    Assumes diagonal pilot observations y_p=h_p*x_p+w_p and does not model
    intercarrier or intersymbol interference. Inputs must have equal shapes.
    """
    received = np.asarray(received_pilots, dtype=np.complex128)
    transmitted = np.asarray(transmitted_pilots, dtype=np.complex128)
    if received.shape != transmitted.shape:
        raise ValueError("received_pilots and transmitted_pilots must have equal shapes")
    if not np.all(np.isfinite(received)) or not np.all(np.isfinite(transmitted)):
        raise ValueError("pilot arrays must be finite")
    if np.any(transmitted == 0):
        raise ValueError("transmitted pilot symbols must be nonzero")
    return received / transmitted
