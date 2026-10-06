"""Explicit returned-bit selection for the unchanged full linear system."""
import copy
import numpy as np
from .certiphy_deep import DeepRefiner


class PayloadSelection:
    def __init__(self, n, width, payload_indices=None, output_bit_mask=None):
        mask = np.ones((n, width), bool)
        if payload_indices is not None:
            idx = np.asarray(payload_indices)
            if idx.ndim != 1 or not np.issubdtype(idx.dtype, np.integer):
                raise ValueError('one-dimensional integer payload indices required')
            if len(np.unique(idx)) != len(idx) or np.any(idx < 0) or np.any(idx >= n):
                raise ValueError('distinct payload indices within full symbol coordinates required')
            mask[:] = False
            mask[idx] = True
        if output_bit_mask is not None:
            bits = np.asarray(output_bit_mask)
            if bits.dtype != np.bool_ or bits.shape not in ((n, width), (n*width,)):
                raise ValueError('boolean output mask in full Gray bit order required')
            mask &= bits.reshape(n, width)
        if not mask.any():
            raise ValueError('at least one returned payload bit required')
        self.mask = mask
        self.symbols = np.flatnonzero(mask.any(axis=1))
        self.count = int(mask.sum())
        # Returned order is canonical symbol order, then existing Gray bit order.


class PayloadRefinement:
    """Remove only unrequested decision groups, never coupled soft variables."""
    def bound(self, symbols, active, target, radius, energy, guard, ledger, margins):
        rx = self.receiver
        rows = np.flatnonzero(active.any(axis=1))
        if not len(rows):
            return 0
        # Explicit scans/indexing are charged, including unsuccessful refinement.
        fee = 4*rx.n*rx.width+8*len(rows)*rx.width
        if not ledger.fits(fee):
            return int(active.sum())
        ledger.add('payload_refinement_selection', fee)
        proxy = copy.copy(rx)
        proxy.n = len(rows)
        weights = self.weights
        if weights is not None and rx.wave.name == 'ofdm':
            order = np.r_[rx.wave.occupied, rx.wave.unused] if hasattr(rx.wave, 'base') else np.arange(rx.n)
            self.weights = weights[order[rows]]
        self.receiver = proxy
        try:
            return DeepRefiner.bound(self, symbols[rows], active[rows], target,
                                     radius, energy, guard, ledger, margins[rows])
        finally:
            self.weights = weights
            self.receiver = rx
