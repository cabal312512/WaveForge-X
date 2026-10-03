"""R4-Deep: conditional Fourier envelopes and target Gray-region refinement.

Local directed kernels bound represented scalar geometry. FFT, sparse operator,
PCG and energy enclosures remain unvalidated: no end-to-end floating certificate.
"""

import time

import numpy as np

from ..core.modulation import bits_per_symbol, demodulate


def down(x):
    return np.nextafter(x, -np.inf)


def up(x):
    return np.nextafter(x, np.inf)


def directed_sum(values, upward=False):
    """Binary tree with outward rounding at EVERY addition (finite IEEE inputs)."""
    values = np.asarray(values, float).ravel().copy()
    rounding = up if upward else down
    if not len(values):
        return 0.
    while len(values) > 1:
        pairs = len(values)//2
        tail = values[-1:] if len(values) % 2 else np.empty(0)
        values = np.r_[rounding(values[:2*pairs:2]+values[1:2*pairs:2]), tail]
    return float(values[0])


def region_costs(symbols, modulation, active, weights=None):
    """Lower costs per axis/count; regions enclose the actual floating slicer.

    Invert rounded multiply/add/scale operations with adjacent floats at each
    step. Returned costs are conservative for represented centers and weights.
    This does not enclose errors in the upstream center or weight construction.
    """
    symbols = np.asarray(symbols)
    width = bits_per_symbol(modulation)
    p = max(1, width//2)
    coordinates = symbols.real if width == 1 else np.r_[symbols.real, symbols.imag]
    masks = active if width == 1 else np.r_[active[:, :p], active[:, p:]]
    bits = demodulate(symbols, modulation).reshape(-1, width)
    labels = bits if width == 1 else np.r_[bits[:, :p], bits[:, p:]]
    shifts = 1 << np.arange(p-1, -1, -1)
    current, mask = labels @ shifts, masks @ shifts
    side = 2**p
    if width == 1:
        lower, upper = np.array([-np.inf, 0.]), np.array([0., np.inf])
        region_labels = np.array([1, 0])
    else:
        scale = float(np.sqrt(2*(2**width-1)/3))
        cuts = 2*np.arange(1, side, dtype=float)
        # fl(fl(x*scale)+side)/2 crosses the integer boundary at cuts/2.
        low_cut = down(down(down(cuts)-side))
        high_cut = up(up(up(cuts)-side))
        lower = np.r_[-np.inf, down(low_cut/scale)]
        upper = np.r_[up(high_cut/scale), np.inf]
        region_labels = np.arange(side) ^ (np.arange(side) >> 1)
    w = np.ones(len(coordinates)) if weights is None else np.asarray(weights, float)
    if w.shape != coordinates.shape or np.any(w <= 0) or not np.isfinite(w).all():
        raise ValueError("positive finite axis weights required")
    costs = np.full((len(coordinates), p+1), np.inf)
    rows = np.arange(len(coordinates))
    pop = np.array([int(h).bit_count() for h in range(side)])
    for j in range(side):
        h = pop[(current ^ region_labels[j]) & mask]
        distance = np.maximum(0., np.maximum(down(lower[j]-coordinates), down(coordinates-upper[j])))
        cost = np.maximum(0., down(down(distance*distance)*w))
        np.minimum.at(costs, (rows, h), cost)
    costs[:, 0] = 0.
    return costs


def saturated_dp(costs, target, allowed=None):
    """Lower minimum cost of AT LEAST target flips, O(groups*target*axis_bits).

    Saturation retains all dangerous combinations, including counts > target.
    In exact arithmetic this DP is exact for the closed, separable regions.
    """
    table = np.full(target+1, np.inf)
    table[0] = 0.
    for i, row in enumerate(costs):
        new = np.full_like(table, np.inf)
        for h, cost in enumerate(row):
            if not np.isfinite(cost) or (allowed is not None and not allowed[i, h]):
                continue
            values = np.maximum(0., down(table+cost))
            if h == 0:
                new = np.minimum(new, values)
            elif h <= target:
                new[h:] = np.minimum(new[h:], values[:-h])
                new[-1] = min(new[-1], values[-h:].min())
            else:
                new[-1] = min(new[-1], values.min())
        table = new
    return float(table[-1])


def dual_lower(costs, target, multiplier):
    counts = np.arange(costs.shape[1])
    product_lo, product_hi = down(multiplier*counts), up(multiplier*counts)
    low, high = down(costs-product_hi), up(costs-product_lo)
    minimum_lo, minimum_hi = low.min(axis=1), high.min(axis=1)
    bound = down(down(multiplier*target)+directed_sum(minimum_lo))
    return float(bound), low, minimum_hi


def refined_target(costs, energy, target, max_cells=180000, dual_steps=10, exact_only=False):
    """Directed dual lower bound, then reduced-cost safe filtering and small DP.

    Work counts include unsuccessful refinements. Cell cap is a hard limit on
    the DP refinement, not permission to discard dangerous combinations.
    """
    groups, choices = costs.shape
    work = 0
    if not np.isfinite(energy) or energy < 0:
        return False, {"work": 0, "reason": "numerical_unknown", "lower": 0.}
    allowed = costs <= energy  # Every term is nonnegative; safe exclusion.
    lower, best_multiplier = 0., 0.
    if not exact_only:
        finite = costs[np.isfinite(costs)]
        left, right = 0., float(finite.max(initial=0.))
        for _ in range(dual_steps):
            multiplier = (left+right)/2
            trial, _, _ = dual_lower(costs, target, multiplier)
            work += 24*groups*choices+8*groups
            if trial > lower:
                lower, best_multiplier = trial, multiplier
            if lower > energy:
                return True, dict(work=work, reason="dual", lower=lower, critical=0)
            selected = np.argmin(costs-multiplier*np.arange(choices), axis=1).sum()
            if selected < target:
                left = multiplier
            else:
                right = multiplier
        lower, low, minimum_hi = dual_lower(costs, target, best_multiplier)
        work += 28*groups*choices+8*groups
        slack = up(energy-lower)
        reduced_lower = down(low-minimum_hi[:, None])
        allowed &= reduced_lower <= slack
    # A group with exactly one retained choice can be eliminated with its count
    # and cost carried into the target; no soft linear-system variable is fixed.
    numbers = allowed.sum(axis=1)
    work += 8*groups*choices
    if np.any(numbers == 0):
        return True, dict(work=work, reason="empty_region", lower=np.inf, critical=0)
    fixed = numbers == 1
    selections = np.argmax(allowed[fixed], axis=1)
    fixed_cost = max(0., directed_sum(costs[fixed, :][np.arange(fixed.sum()), selections]))
    remaining = max(0, target-int(selections.sum()))
    critical = int((~fixed).sum())
    cells = critical*(remaining+1)*choices
    if cells > max_cells:
        return False, dict(work=work, reason="refinement_cap", lower=lower, critical=critical)
    result = saturated_dp(costs[~fixed], remaining, allowed[~fixed])
    work += 8*cells
    lower = max(lower, float(down(result+fixed_cost)))
    return lower > energy, dict(work=work, reason="dp", lower=lower, critical=critical)


def circulant_envelope(matrix, noise):
    """C=C0+E; sparse row/column norm bound, Young's inequality lower M.

    All actual CPP entries participate. This is row-wise circulant averaging,
    NOT removal of physical paths and NOT diag(A) <= A.
    """
    n = matrix.shape[0]
    coo = matrix.tocoo()
    delays = (coo.row-coo.col) % n
    impulse = np.zeros(n, complex)
    np.add.at(impulse, delays, coo.data/n)
    support = np.unique(delays)
    # Include positions missing from C (cancellation/eliminate_zeros).
    from scipy.sparse import coo_matrix
    rows = np.tile(np.arange(n), len(support))
    cols = (rows-np.repeat(support, n)) % n
    approximate = coo_matrix((np.repeat(impulse[support], n), (rows, cols)), shape=(n, n)).tocsr()
    defect = abs(matrix-approximate)
    epsilon = np.sqrt(float(defect.sum(axis=0).max())*float(defect.sum(axis=1).max()))
    spectrum = abs(np.fft.fft(impulse))
    minimum = float(spectrum.min())
    # Guard is diagnostic only: FFT/sparse norms have no validated enclosure.
    epsilon += 64*np.finfo(float).eps*(1+float(spectrum.max()))
    minimum = max(0., minimum-64*np.finfo(float).eps*(1+float(spectrum.max())))
    if minimum <= epsilon:
        return None, dict(epsilon=epsilon, minimum=minimum, triggered=False)
    tau = epsilon/minimum
    weights = (1-tau)*spectrum**2+noise-(1/tau-1)*epsilon**2
    weights -= 128*np.finfo(float).eps*(1+float(spectrum.max())**2)
    if weights.min() <= 0:
        return None, dict(epsilon=epsilon, minimum=minimum, triggered=False)
    return weights, dict(epsilon=epsilon, minimum=minimum, triggered=True)


class DeepRefiner:
    """Two isolated candidates: 'envelope' and 'dp'; redesigned 'target'."""

    def __init__(self, mode="target", max_cells=180000, dual_steps=10):
        if mode not in ("envelope", "dp", "target", "combined", "spectral"):
            raise ValueError("unknown refinement")
        self.mode, self.max_cells, self.dual_steps = mode, max_cells, dual_steps
        self.metrics = dict(calls=0, successes=0, numerical_unknown=0, cap=0,
                            check_seconds=0., preparation_seconds=0., critical_total=0)
        self.weights = None
        self.inverse_directions = None
        self.current_eta = None

    def prepare(self, receiver, ledger):
        self.receiver = receiver
        if self.mode not in ("envelope", "combined", "spectral"):
            return
        start = time.perf_counter()
        cost = 100*receiver.nnz+5*receiver.n*np.log2(receiver.n)+80*receiver.n
        if ledger.fits(cost):
            ledger.add("deep_envelope_preparation", cost)
            self.weights, details = circulant_envelope(receiver.matrix, receiver.noise)
            self.metrics.update(details)
            if self.weights is not None and self.mode in ("spectral", "combined"):
                cost = 20*receiver.n*np.log2(receiver.n)+40*receiver.n
                if ledger.fits(cost):
                    ledger.add("deep_inverse_directions", cost)
                    self.inverse_directions = inverse_directions(receiver.wave, self.weights)
        self.metrics["preparation_seconds"] += time.perf_counter()-start

    def radii(self, energy, radius, guard, ledger, residual):
        receiver = self.receiver
        self.current_eta = None
        result = np.full(receiver.n, radius)
        if self.weights is None or not ledger.fits(12*receiver.n):
            return result
        ledger.add("deep_direction_radii", 12*receiver.n)
        eta = (np.sqrt(max(0., energy))+np.sqrt(receiver.beta)*guard)**2
        if self.mode in ("spectral", "combined"):
            cost = 5*receiver.n*np.log2(receiver.n)+12*receiver.n
            if ledger.fits(cost):
                start = time.perf_counter()
                ledger.add("deep_residual_energy", cost)
                frequency = np.fft.fft(residual, norm="ortho")
                new_energy = np.sum(abs(frequency)**2/self.weights)
                eta = min(eta, (np.sqrt(new_energy)+np.sqrt(receiver.beta)*guard)**2)
                self.metrics["check_seconds"] += time.perf_counter()-start
            if self.inverse_directions is not None:
                self.current_eta = eta
                return np.minimum(result, np.sqrt(eta*self.inverse_directions))
        self.current_eta = eta
        weights = self.weights if receiver.wave.name == "ofdm" else self.weights.min()
        return np.minimum(result, np.sqrt(eta/weights))

    def bound(self, symbols, active, target, radius, energy, guard, ledger, margins):
        remaining = int(active.sum())
        if target == 0 or (self.mode in ("envelope", "spectral") and self.weights is None):
            return remaining
        receiver = self.receiver
        p = max(1, receiver.width//2)
        groups = receiver.n*(1 if receiver.width == 1 else 2)
        axis_weights = None
        limit = float(up(radius*radius))
        if self.weights is not None:
            eta = self.current_eta if self.current_eta is not None else up((np.sqrt(max(0., energy))+np.sqrt(receiver.beta)*guard)**2)
            if receiver.wave.name == "ofdm":
                axis_weights = self.weights if receiver.width == 1 else np.r_[self.weights, self.weights]
                limit = float(up(eta))
            else:
                limit = min(limit, float(up(eta/self.weights.min())))
        if self.mode in ("target", "spectral", "combined"):
            # This only declines refinement; it never announces certification.
            gate_work = 14*groups*p
            if not ledger.fits(gate_work):
                self.metrics["cap"] += 1
                return remaining
            ledger.add("deep_cheap_gate", gate_work)
            axis = margins if receiver.width == 1 else np.r_[margins[:, :p], margins[:, p:]]
            masks = active if receiver.width == 1 else np.r_[active[:, :p], active[:, p:]]
            cheapest = np.min(np.where(masks, axis, np.inf), axis=1)**2
            if axis_weights is not None:
                cheapest *= axis_weights
            if target < len(cheapest) and np.partition(cheapest, target)[:target+1].sum() <= limit:
                return remaining
        # Reserve worst-case refinement BEFORE constructing or optimizing it.
        region_work = 48*groups*2**p+8*groups*(p+1)
        dual_work = (self.dual_steps+1)*(32*groups*(p+1)+8*groups)
        reservation = region_work+dual_work+8*self.max_cells
        if not ledger.fits(reservation):
            self.metrics["cap"] += 1
            return remaining
        start = time.perf_counter()
        self.metrics["calls"] += 1
        costs = region_costs(symbols, receiver.modulation, active, axis_weights)
        ledger.add("deep_regions", region_work)
        success, info = refined_target(costs, limit, target+1, self.max_cells,
                                       self.dual_steps, exact_only=self.mode == "dp")
        ledger.add("deep_refinement", info["work"])
        self.metrics["successes"] += int(success)
        self.metrics["numerical_unknown"] += int(info["reason"] == "numerical_unknown")
        self.metrics["cap"] += int(info["reason"] == "refinement_cap")
        self.metrics["critical_total"] += info.get("critical", 0)
        self.metrics["check_seconds"] += time.perf_counter()-start
        return target if success else remaining


def inverse_directions(wave, weights):
    """diag(T^H F^H diag(1/m) F T), without constructing any dense transform.

    OTFS support is k mod M = Doppler index. AFDM is a circular correlation
    with the squared Fourier magnitude of its time chirp (arbitrary c1/c2).
    These are individual direction bounds, NOT separable joint axis weights.
    """
    inverse = 1/weights
    if wave.name == "ofdm":
        result = inverse
    elif wave.name == "otfs":
        result = np.repeat(inverse.reshape(wave.subcarriers, wave.time_slots).mean(axis=0), wave.subcarriers)
    elif wave.name == "afdm":
        kernel = abs(np.fft.fft(wave._chirp1.conj())/wave.n_symbols)**2
        result = np.fft.ifft(np.fft.fft(inverse)*np.fft.fft(kernel).conj()).real
    else:
        return np.full(wave.n_symbols, inverse.max())
    # Upstream FFT arithmetic is not enclosed; do not claim a local proof here.
    return np.maximum(0., result)+128*np.finfo(float).eps*(1+inverse.max())
