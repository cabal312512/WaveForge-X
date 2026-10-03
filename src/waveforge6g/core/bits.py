"""Random information bits with caller-owned random number generators."""

import numpy as np
from numpy.typing import NDArray


def random_bits(n_bits: int, rng: np.random.Generator) -> NDArray[np.uint8]:
    """Return ``n_bits`` independent equiprobable bits; never use global RNG state."""
    if isinstance(n_bits, bool) or not isinstance(n_bits, int | np.integer) or n_bits < 0:
        raise ValueError("n_bits must be a nonnegative integer")
    return rng.integers(0, 2, size=n_bits, dtype=np.uint8)
