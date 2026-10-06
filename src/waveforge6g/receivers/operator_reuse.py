"""R7 phase-invariant, regularized-Gram reuse validator.

Exact-arithmetic implications are in research_v7/bounds.md. Numerical guards
are diagnostics, not an end-to-end floating-point enclosure. No RHS or payload
enters the validator. Sparse matrices are the complete current supplied model.
"""
from dataclasses import dataclass
import math
import time
import numpy as np
from .certiphy_deep import DeepRefiner, circulant_envelope
from .certiphy_reuse import (ReuseState, ReuseRefiner, ReuseCertiPHY,
                            direction_values, channel_norm, PayloadRegionAlignment)
from .payload_selection import PayloadRefinement
from ..channels.reuse_paths import path_change_bound


def relative_scale(epsilon, noise_anchor, noise_current):
    """Sharp universal scalar, normalized before squaring; 0 means unknown.

    Representable underflow is not repaired by inventing a positive coefficient.
    Downward diagnostics are not IEEE-754 certification of the full formula.
    """
    e, a, b = float(epsilon), float(noise_anchor), float(noise_current)
    if not all(math.isfinite(x) for x in (e, a, b)) or e < 0 or min(a, b) <= 0:
        raise ValueError('finite epsilon>=0 and positive complex noise variances required')
    if e == 0:
        return 1. if b >= a else b/a
    scale = max(math.sqrt(a), math.sqrt(b), e)
    aa, bb, ee = (math.sqrt(a)/scale)**2, (math.sqrt(b)/scale)**2, e/scale
    if aa == 0 or bb == 0:
        return 0.
    d = math.hypot(aa-bb, ee*math.sqrt(2*(aa+bb)+ee*ee))
    g = 2*bb/(aa+bb+ee*ee+d)
    # A local numerical guard. Its adequacy is checked with Decimal in tests.
    return max(0., g*(1-128*np.finfo(float).eps))


def sparse_norm_upper(matrix):
    """min(Frobenius, sqrt(max column sum * max row sum)); no dense Gram."""
    values = abs(matrix)
    row = np.asarray(values.sum(axis=1)).ravel()
    col = np.asarray(values.sum(axis=0)).ravel()
    # hypot accumulation via a scaled vector norm avoids squaring huge entries.
    maximum = float(np.max(abs(matrix.data), initial=0.))
    frob = 0. if maximum == 0 else maximum*float(np.linalg.norm(abs(matrix.data)/maximum))
    prod = math.sqrt(float(row.max(initial=0.)))*math.sqrt(float(col.max(initial=0.)))
    return min(frob, prod)


def aligned_change(current, anchor, mode='row', anchor_norm=None, anchor_rows=None):
    """Align receive phases AFTER coalescing every actual sparse coefficient.

    Correlation minimizes each row's 2-norm, not the spectral norm. q=0 uses 1.
    The stored floating phase is interpreted as a proposed exact unit phase;
    its distance to the unit circle is separately bounded diagnostically.
    """
    if current.shape != anchor.shape or mode not in ('none', 'common', 'row'):
        raise ValueError('matched coordinates and supported alignment required')
    c, a = current.tocsr(copy=False), anchor.tocsr(copy=False)
    if not c.has_canonical_format or not a.has_canonical_format:
        c, a = c.copy(), a.copy()
        c.sum_duplicates(); a.sum_duplicates(); c.sort_indices(); a.sort_indices()
    n = c.shape[0]
    matching = np.array_equal(c.indptr, a.indptr) and np.array_equal(c.indices, a.indices)
    rows = (np.repeat(np.arange(n), np.diff(a.indptr)) if anchor_rows is None else anchor_rows) if matching else None
    phase = np.ones(n, complex)
    if mode != 'none':
        if matching:
            products = c.data*a.data.conj()
            q = np.bincount(rows, weights=products.real, minlength=n)+1j*np.bincount(rows, weights=products.imag, minlength=n)
        else:
            q = np.asarray(c.multiply(a.conj()).sum(axis=1)).ravel()
        if mode == 'common':
            q = np.full(n, q.sum())
        nonzero = abs(q) > 0
        phase[nonzero] = q[nonzero]/abs(q[nonzero])
        # Near an exactly representable quarter-turn, propose that unit phase.
        # This changes only U. It NEVER declares channel equality: the complete
        # actual residual below is still measured and charged.
        real_axis = abs(phase.real) >= abs(phase.imag)
        axis = np.where(real_axis, np.sign(phase.real), 1j*np.sign(phase.imag))
        snap = abs(phase-axis) <= 16*np.finfo(float).eps
        phase[snap] = axis[snap]
    if matching:
        residual_data = c.data-phase[rows]*a.data
        values = abs(residual_data)
        row_sum = np.bincount(rows, weights=values, minlength=n)
        col_sum = np.bincount(a.indices, weights=values, minlength=c.shape[1])
        maximum = float(values.max(initial=0.))
        frob = 0. if maximum == 0 else maximum*float(np.linalg.norm(values/maximum))
        raw = min(frob, math.sqrt(float(row_sum.max(initial=0.)))*math.sqrt(float(col_sum.max(initial=0.))))
        residual_nnz = int(np.count_nonzero(residual_data))
    else:
        scaled = a.multiply(phase[:, None]).tocsr()
        residual = (c-scaled).tocsr()
        residual.sum_duplicates(); residual.eliminate_zeros()
        raw = sparse_norm_upper(residual)
        residual_nnz = residual.nnz
    axis_exact = ((abs(phase.real) == 1) & (phase.imag == 0)) | ((abs(phase.imag) == 1) & (phase.real == 0))
    defect = 0. if axis_exact.all() else float(np.max(abs(np.hypot(phase.real, phase.imag)-1)))+8*np.finfo(float).eps
    anchor_norm = sparse_norm_upper(a) if anchor_norm is None else float(anchor_norm)
    # Exact axis phases with exact zero represented residual retain epsilon=0.
    guard = 0. if raw == 0 and defect == 0 else 64*np.finfo(float).eps*(raw+anchor_norm+np.finfo(float).tiny)
    epsilon = raw+defect*anchor_norm+guard
    cost = float((44 if matching and mode != 'none' else 28 if matching else 72 if mode != 'none' else 36)*(c.nnz+a.nnz)+40*n+16*c.shape[1])
    return dict(epsilon=epsilon, residual_upper=raw, phase_modulus_defect=defect,
                anchor_norm=anchor_norm, work=cost, residual_nnz=residual_nnz,
                storage_bytes=int(a.data.nbytes+a.indices.nbytes+a.indptr.nbytes+phase.nbytes), mode=mode, fused_scan=matching)


@dataclass(frozen=True)
class ValidatorPolicy:
    alignment: str = 'row'
    relative: bool = True
    minimum_scale: float = .25
    failed_age: int = 10


class OperatorReuseValidator:
    def __init__(self, state, policy=None, directions=True):
        self.state, self.policy = state, policy or ValidatorPolicy()
        self.directions = directions
        self.weights = self.inverse_directions = None
        self.transition = {}
        self.preparation_seconds = 0.

    def prepare(self, rx, ledger):
        st, policy = self.state, self.policy
        start = time.perf_counter()
        k = st.frame_index
        action, reason = 'rebuild', 'cold'
        epsilon = absolute = relative = scale = 0.
        detail = {}
        if st.anchor is not None and hasattr(st, 'anchor_matrix'):
            fee = 4*(rx.nnz+st.anchor_matrix.nnz)+8*rx.n
            if not ledger.fits(fee):
                self.transition = dict(action='fallback', reason='validation_budget', valid_envelope=False)
                st.metrics['fallbacks'] += 1
                return
            ledger.add('operator_equality_check', fee)
            a = st.anchor_matrix
            same = (rx.matrix.shape == a.shape and np.array_equal(rx.matrix.indptr, a.indptr)
                    and np.array_equal(rx.matrix.indices, a.indices) and np.array_equal(rx.matrix.data, a.data))
            if same:
                reason = 'exact_operator'
            else:
                matching = np.array_equal(rx.matrix.indptr,a.indptr) and np.array_equal(rx.matrix.indices,a.indices)
                factor = (44 if policy.alignment != 'none' else 28) if matching else (72 if policy.alignment != 'none' else 36)
                fee = factor*(rx.nnz+a.nnz)+40*rx.n+16*rx.n
                if not ledger.fits(fee):
                    self.transition = dict(action='fallback', reason='alignment_budget', valid_envelope=False)
                    st.metrics['fallbacks'] += 1
                    return
                ledger.add('phase_validator', fee)
                detail = aligned_change(rx.matrix, a, policy.alignment, st.anchor_matrix_norm, st.anchor_rows)
                epsilon = detail['epsilon']
            if st.anchor_weights is not None:
                absolute = float(np.min(1-(2*st.anchor_matrix_norm*epsilon-(rx.noise-st.anchor_noise))/st.anchor_weights))
                if not (same and rx.noise == st.anchor_noise):
                    absolute -= 256*np.finfo(float).eps*(1+abs(absolute))
                relative = relative_scale(epsilon, st.anchor_noise, rx.noise) if policy.relative else 0.
                scale = max(absolute, relative)
                if scale >= policy.minimum_scale and np.isfinite(scale):
                    action = 'reuse' if epsilon == 0 and rx.noise == st.anchor_noise else 'update'
                    reason = 'phase_invariant_relative' if relative >= absolute else 'absolute_lower_bound'
                else:
                    st.metrics['weak_updates' if scale > 0 else 'invalidations'] += 1
                    reason = 'valid_but_not_useful' if scale > 0 else 'no_positive_update'
            elif same or (k-st.anchor_last < policy.failed_age and epsilon <= .05*max(st.anchor_matrix_norm, 1e-30)):
                action, reason = 'fallback', 'cached_no_envelope'
        if action in ('reuse', 'update'):
            fee = 12*rx.n
            if ledger.fits(fee):
                ledger.add('scaled_envelope_and_directions', fee)
                self.weights = st.anchor_weights*scale
                self.inverse_directions = None if st.anchor_directions is None else st.anchor_directions/scale
                st.metrics['reuses' if action == 'reuse' else 'updates'] += 1
            else:
                action, reason = 'fallback', 'scale_budget'
        if action == 'rebuild':
            fee = 100*rx.nnz+5*rx.n*np.log2(rx.n)+80*rx.n
            dirs = 20*rx.n*np.log2(rx.n)+40*rx.n if self.directions else 0.
            storage = 12*(rx.nnz+rx.n)+36*rx.nnz+40*rx.n
            if ledger.fits(fee+dirs+storage):
                ledger.add('deep_envelope_preparation', fee)
                self.weights, info = circulant_envelope(rx.matrix, rx.noise)
                if self.weights is not None and self.directions:
                    ledger.add('deep_inverse_directions', dirs)
                    self.inverse_directions = direction_values(rx.wave, self.weights)
                ledger.add('immutable_operator_anchor', storage)
                st.anchor = rx.channel
                st.anchor_matrix = rx.matrix.copy()
                st.anchor_rows = np.repeat(np.arange(rx.n),np.diff(rx.matrix.indptr))
                st.anchor_matrix_norm = sparse_norm_upper(rx.matrix)
                st.anchor_weights = None if self.weights is None else self.weights.copy()
                st.anchor_directions = None if self.inverse_directions is None else self.inverse_directions.copy()
                for array in [st.anchor_matrix.data,st.anchor_matrix.indices,st.anchor_matrix.indptr,st.anchor_rows,st.anchor_weights,st.anchor_directions]:
                    if array is not None:
                        array.setflags(write=False)
                st.anchor_noise = rx.noise
                st.anchor_last = k
                st.metrics['rebuilds'] += 1
            else:
                action, reason = 'fallback', 'rebuild_budget'
        if action == 'fallback':
            st.metrics['fallbacks'] += 1
        self.transition = dict(action=action, reason=reason, delta_c=epsilon, scale=scale,
                               absolute_scale=absolute, relative_scale=relative,
                               valid_envelope=self.weights is not None, anchor_age=k-st.anchor_last,
                               **detail)
        self.preparation_seconds = time.perf_counter()-start


class ValidatedRefiner(PayloadRefinement, PayloadRegionAlignment, DeepRefiner):
    def __init__(self, validator):
        super().__init__('spectral')
        self.validator = validator
        self.transition = {}

    def prepare(self, rx, ledger):
        self.receiver = rx
        self.validator.prepare(rx, ledger)
        self.weights = self.validator.weights
        self.inverse_directions = self.validator.inverse_directions
        self.transition = dict(self.validator.transition)
        self.metrics.update(self.transition)
        self.metrics['preparation_seconds'] += self.validator.preparation_seconds


class PayloadReuseRefiner(PayloadRefinement, ReuseRefiner):
    pass


class ScalarEnvelopePreparation:
    """The same legal envelope can improve Radau's scalar spectral lower bound."""
    def __init__(self, validator):
        self.validator = validator

    def prepare(self, rx, ledger):
        self.validator.prepare(rx, ledger)
        if self.validator.weights is not None and ledger.fits(4*rx.n):
            ledger.add('scalar_spectrum_lower_bound', 4*rx.n)
            rx.alpha = max(rx.noise, float(self.validator.weights.min()))
            rx.mu = (1-1e-12)*rx.alpha/float(rx.diagonal.max())
