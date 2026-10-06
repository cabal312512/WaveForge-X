"""Decision-aware, budgeted time-domain PCG (CertiPHY research implementation).

Bounds have exact-arithmetic proofs. Floating evaluations are guarded, NOT
validated interval arithmetic; ``floating_point_certified`` is always False.
No transmit bits, reference solutions, or counterfactual loss enter this API.
"""

import time
from dataclasses import dataclass

import numpy as np

from ..core.modulation import bits_per_symbol, demodulate
from .budgeted import cost_model, time_channel, transform_work


def bit_margins(symbols, modulation):
    """Distance to the nearest boundary that changes EACH Gray bit.

    Equality never certifies a bit, including the slicer's asymmetric tie rules.
    No LMMSE gain normalization is introduced.
    """
    values = np.asarray(symbols)
    width = bits_per_symbol(modulation)
    if width == 1:
        return abs(values.real)[:, None]
    side, axis_bits = 2**(width//2), width//2
    scale = np.sqrt(2*(2**width-1)/3)
    cuts = (2*np.arange(1, side)-side)/scale
    labels = np.arange(side) ^ (np.arange(side) >> 1)
    toggles = labels[:-1] ^ labels[1:]
    result = np.empty((len(values), width))
    for offset, coordinate in ((0, values.real), (axis_bits, values.imag)):
        for bit in range(axis_bits):
            relevant = cuts[(toggles & (1 << (axis_bits-1-bit))) != 0]
            result[:, offset+bit] = abs(coordinate[:, None]-relevant).min(axis=1)
    return result


def gray_energy_bound(symbols, modulation, radius, active, target_count, margins=None):
    """Upper bound on joint Hamming distance inside a Euclidean error ball.

    For QPSK/BPSK, sort the per-axis boundary-crossing energies (exact supremum
    count for the ball). For higher QAM, evaluate valid Lagrangian upper bounds
    of a multiple-choice knapsack over all Gray-labeled axis decision regions.
    Any nonnegative multiplier is valid; five data-dependent trials affect
    tightness only. This does not mark the unidentified bits as individually safe.
    """
    width = bits_per_symbol(modulation)
    remaining = int(active.sum())
    if not remaining:
        return 0
    margins = bit_margins(symbols, modulation) if margins is None else margins
    energy = float(radius)**2
    if width <= 2:
        distances = np.sort(margins[active]**2)
        # Closed boundary regions are conservative for exact slicer ties.
        return int(np.searchsorted(np.cumsum(distances), energy, side="right"))
    p, side = width//2, 2**(width//2)
    scale = np.sqrt(2*(2**width-1)/3)
    coordinates = np.concatenate((symbols.real, symbols.imag))
    masks = np.concatenate((active[:, :p], active[:, p:]))
    shifts = 1 << np.arange(p-1, -1, -1)
    mask_labels = masks @ shifts
    natural = np.clip(np.floor((coordinates*scale+side)/2), 0, side-1).astype(int)
    current = natural ^ (natural >> 1)
    cuts = (2*np.arange(1, side)-side)/scale
    boundaries = np.concatenate(([-np.inf], cuts, [np.inf]))
    costs = np.full((len(coordinates), p+1), np.inf)
    costs[:, 0] = 0.
    popcount = np.array([int(i).bit_count() for i in range(side)])
    rows = np.arange(len(coordinates))
    for q in range(side):
        label = q ^ (q >> 1)
        count = popcount[(current ^ label) & mask_labels]
        distance = np.maximum(np.maximum(boundaries[q]-coordinates, coordinates-boundaries[q+1]), 0)
        np.minimum.at(costs, (rows, count), distance**2)
    available = margins[active]**2
    anchor = np.partition(available, min(target_count, len(available)-1))[min(target_count, len(available)-1)]
    if anchor <= np.finfo(float).tiny:
        return remaining
    best = float(remaining)
    counts = np.arange(p+1)
    for factor in (.25, .5, 1., 2., 4.):
        multiplier = factor/max(anchor, 1e-300)
        upper = multiplier*energy+np.max(counts[None, :]-multiplier*costs, axis=1).sum()
        best = min(best, float(upper))
    # A guard against an ordinary rounding-down at integer-valued bounds, not
    # a claim of a complete floating-point proof for all preceding operations.
    return min(remaining, max(0, int(np.floor(best+1e-9))))


def targeted_gray_bound(symbols, modulation, radius, active, target_count, margins):
    """Target-aware redesign: avoid a full sort and unproductive Gray checks.

    For binary axes, q+1 smallest crossing energies suffice to decide whether
    more than q flips are possible. For higher QAM, skip the dual evaluation
    when q+1 different axes can already cross within the ball.
    """
    width = bits_per_symbol(modulation)
    remaining = int(active.sum())
    if remaining <= target_count:
        return remaining, False
    if width <= 2:
        costs = margins[active]**2
        smallest = np.partition(costs, target_count)[:target_count+1]
        return (target_count if smallest.sum() > radius**2 else remaining), False
    p = width//2
    axis_costs = np.minimum(np.where(active, margins, np.inf).reshape(len(symbols), 2, p).min(axis=2), np.inf).ravel()**2
    if len(axis_costs) > target_count:
        smallest = np.partition(axis_costs, target_count)[:target_count+1]
        if smallest.sum() <= radius**2:
            return remaining, False
    return gray_energy_bound(symbols, modulation, radius, active, target_count, margins), True


def physical_spectral_bounds(channel, noise):
    """Each guarded integer-delay path is gain times a unitary shift/phase.

    Reverse triangle inequality for sigma_min(C); never use a Ritz value here.
    """
    gains = abs(channel.gains)
    total = float(gains.sum())
    minimum = max(0., 2*float(gains.max())-total)
    return float(noise+minimum**2), float(noise+total**2)


@dataclass(frozen=True)
class StopRule:
    method: str = "packing"
    delta: float = 0.
    max_iterations: int = 256
    period: int = 4
    schedule: str = "adaptive"
    tolerance: float = 1e-4
    fixed_iterations: int = 16
    stable_checks: int = 3
    dual_iterations: int = 12
    dual_targets: int = 2


class WorkLedger:
    def __init__(self, maximum):
        self.maximum = float(maximum)
        self.parts = {}

    @property
    def total(self):
        return float(sum(self.parts.values()))

    def fits(self, cost):
        return self.total+cost <= self.maximum

    def add(self, category, cost):
        if not self.fits(cost):
            raise RuntimeError("unreserved work")
        self.parts[category] = self.parts.get(category, 0.)+float(cost)


class CertiPHY:
    """Exact sparse C; fixed SPD Jacobi preconditioner; no dense A or inverse."""

    def __init__(self, waveform, channel, noise_variance, modulation):
        if not np.isfinite(noise_variance) or noise_variance <= 0:
            raise ValueError("positive finite noise variance required")
        start, cpu = time.perf_counter(), time.process_time()
        self.wave, self.channel, self.noise = waveform, channel, float(noise_variance)
        self.modulation = modulation
        self.n, self.width = waveform.n_symbols, bits_per_symbol(modulation)
        self.matrix = time_channel(waveform, channel)
        self.adjoint = self.matrix.conj().T.tocsr()
        self.diagonal = np.asarray(abs(self.matrix).power(2).sum(axis=0)).ravel()+self.noise
        self.alpha, self.beta = physical_spectral_bounds(channel, self.noise)
        # Strict underestimate is needed by the Gauss-Radau recurrence.
        self.mu = (1-1e-12)*self.alpha/float(self.diagonal.max())
        self.nnz = self.n*channel.n_paths  # same conservative convention as R3
        self.transform = transform_work(waveform)
        self.step_work = 16*self.nnz+40*self.n+8
        self.normal_work = 16*self.nnz+12*self.n
        self.decode_work = 4*self.n*self.width
        self.setup_work = cost_model(waveform, channel.n_paths, "time", 0, 0)["preparation_work"]+16*channel.n_paths
        self.setup_elapsed = time.perf_counter()-start
        self.setup_cpu = time.process_time()-cpu

    def normal(self, x):
        return self.adjoint @ (self.matrix @ x)+self.noise*x

    def _dual(self, index, iterations):
        unit = np.zeros(self.n, complex)
        unit[index] = 1
        f = self.wave.synthesis(unit)
        q, r = np.zeros(self.n, complex), f.copy()
        z = r/self.diagonal
        p, rho = z.copy(), float(np.vdot(r, z).real)
        count = 0
        for _ in range(iterations):
            ap = self.normal(p)
            denominator = float(np.vdot(p, ap).real)
            if denominator <= 0 or rho <= 1e-28:
                break
            a = rho/denominator
            q += a*p
            r -= a*ap
            count += 1
            z = r/self.diagonal
            new = float(np.vdot(r, z).real)
            p = z+(new/rho)*p
            rho = new
        defect = np.linalg.norm(f-self.normal(q))
        return q, float(defect), count

    def solve(self, received, rule=None, max_work=np.inf, keep_trace=False, refiner=None,
              payload_indices=None, output_bit_mask=None, preparer=None):
        rule = StopRule() if rule is None else rule
        if rule.method not in ("fixed", "residual", "residual_fast", "stable", "stable_fast", "global", "krylov", "dual", "radau", "packing", "gray"):
            raise ValueError("unknown stopping method")
        if (not 0 <= rule.delta <= 1 or rule.period < 1 or rule.max_iterations < 1
                or rule.schedule not in ("periodic", "adaptive") or rule.tolerance <= 0):
            raise ValueError("invalid stopping rule")
        y = np.asarray(received, complex)
        if y.shape != (self.n,) or not np.all(np.isfinite(y)):
            raise ValueError("finite useful received vector required")
        started, cpu_started = time.perf_counter(), time.process_time()
        from .payload_selection import PayloadSelection
        selection = PayloadSelection(self.n, self.width, payload_indices, output_bit_mask)
        selected_mask, bit_count = selection.mask, selection.count
        live_symbols = len(selection.symbols)
        decode_work = 4*live_symbols*self.width
        decision_work = live_symbols*self.width
        self.output_selection = selection
        ledger = WorkLedger(max_work)
        # Reserve the final output, so budget exhaustion cannot hide a free FFT.
        initial = self.setup_work+8*self.nnz+12*self.n+2*self.transform+8*live_symbols*2**self.width+decode_work+4*self.n*self.width
        if not ledger.fits(initial):
            raise ValueError("budget cannot cover mandatory preparation and output")
        ledger.add("preparation", self.setup_work)
        ledger.add("rhs", 8*self.nnz+12*self.n)
        ledger.add("transmit_and_output", 2*self.transform+8*live_symbols*2**self.width+decode_work)
        ledger.add("output_selection", 4*self.n*self.width)
        if preparer is not None:
            preparer.prepare(self, ledger)
        if refiner is not None:
            if rule.method != "gray":
                raise ValueError("refinement requires the Gray stopping rule")
            refiner.prepare(self, ledger)
        b = self.adjoint @ y
        x, r = np.zeros(self.n, complex), b.copy()
        z = r/self.diagonal
        p, rho = z.copy(), float(np.vdot(r, z).real)
        norm_b = np.linalg.norm(b)
        radau = 1/self.mu
        leverage = np.zeros(self.n)
        # Outside-mask True is bookkeeping only; it is never exposed as certified.
        certificate = ~selected_mask.copy()
        frozen_bits = np.zeros_like(certificate, np.uint8)
        previous_bits, stable = None, 0
        duals, trace = {}, []
        target = int(np.floor(rule.delta*bit_count))
        next_check, previous_radius, previous_iteration = rule.period, None, 0
        count, checks, restarts, bound = 0, 0, 0, bit_count
        status = "iteration_cap"
        check_seconds = 0.
        certified_mode = rule.method in ("global", "krylov", "dual", "radau", "packing", "gray")
        for k in range(1, rule.max_iterations+1):
            extra = self.transform+8*self.n if rule.method == "krylov" else 0
            if not ledger.fits(self.step_work+extra):
                status = "work_cap"
                break
            if rho <= 0:
                # The zero RHS is handled by a checked zero-iteration output below.
                next_check = k
            else:
                ledger.add("pcg", self.step_work)
                ap = self.normal(p)
                denominator = float(np.vdot(p, ap).real)
                if denominator <= 0 or not np.isfinite(denominator):
                    status = "numerical_breakdown"
                    break
                step = rho/denominator
                if rule.method == "krylov":
                    ledger.add("direction_transforms", extra)
                    leverage += abs(self.wave.analysis(p))**2/denominator
                x += step*p
                r -= step*ap
                z = r/self.diagonal
                new_rho = float(np.vdot(r, z).real)
                beta = max(0., new_rho/rho)
                gap = radau-step
                radau = gap/(self.mu*gap+beta) if gap > 0 and self.mu*gap+beta > 0 else np.inf
                p = z+beta*p
                rho = new_rho
                count = k
            due = k >= next_check or k == rule.max_iterations or rho <= 1e-300
            if rule.method == "fixed":
                if k >= rule.fixed_iterations:
                    status = "fixed_k"
                    break
                continue
            if rule.method == "residual_fast":
                if not ledger.fits(4*self.n):
                    status = "work_cap"
                    break
                ledger.add("cheap_residual_trigger", 4*self.n)
                due = np.linalg.norm(r) <= rule.tolerance*max(norm_b, 1e-300) or k == rule.max_iterations
            if not due:
                continue
            check_work = self.normal_work+24*self.n+self.transform+decode_work
            if rule.method == "stable_fast":
                check_work = 12*self.n+self.transform+decode_work+4*bit_count
            if certified_mode:
                check_work += 12*decision_work*max(1, 2**(self.width//2)-1)
                if rule.schedule == "adaptive":
                    check_work += 8*bit_count+40
            if not ledger.fits(check_work):
                status = "work_cap"
                break
            ledger.add("residual_and_check", check_work)
            check_started = time.perf_counter()
            true_r = r if rule.method == "stable_fast" else b-self.normal(x)
            residual = np.linalg.norm(true_r)
            drift = np.linalg.norm(true_r-r)
            symbols = self.wave.analysis(x)
            uncertain_symbols = np.flatnonzero(~certificate.all(axis=1))
            bits = frozen_bits.copy()
            bits[uncertain_symbols] = demodulate(symbols[uncertain_symbols], self.modulation).reshape(-1, self.width)
            checks += 1
            stop = False
            radius = residual/self.noise
            numerical_guard = 64*np.finfo(float).eps*(1+norm_b+self.beta*np.linalg.norm(x))/self.noise
            restarted = drift > 1e-8*max(residual, 1e-30)
            if rule.method in ("residual", "residual_fast"):
                stop = residual <= rule.tolerance*max(norm_b, 1e-300)
            elif rule.method in ("stable", "stable_fast"):
                stable = stable+1 if previous_bits is not None and np.array_equal(bits, previous_bits) else 0
                previous_bits = bits.copy()
                stop = stable >= rule.stable_checks
            else:
                alpha = self.noise if rule.method == "global" else self.alpha
                radius = residual/alpha+numerical_guard
                energy = radius**2*alpha
                if rule.method in ("krylov", "radau", "packing", "gray") and not restarted and np.isfinite(radau):
                    energy = min(energy, max(0., radau*rho))
                    radius = min(radius, np.sqrt(energy/alpha)+numerical_guard)
                radii = np.full(self.n, radius)
                if refiner is not None:
                    radii = refiner.radii(energy, radius, numerical_guard, ledger, true_r)
                if rule.method == "krylov" and not restarted:
                    diagonal_bound = 1/alpha-leverage
                    if np.all(diagonal_bound > 0):
                        radii = np.minimum(radii, np.sqrt(energy*diagonal_bound)+numerical_guard)
                margins = np.full((self.n, self.width), np.inf)
                margins[uncertain_symbols] = bit_margins(symbols[uncertain_symbols], self.modulation)
                newly = (~certificate) & (margins > radii[:, None])
                frozen_bits[newly] = bits[newly]
                certificate |= newly
                if rule.method == "dual":
                    uncertain = np.flatnonzero(~certificate.all(axis=1))
                    if len(uncertain) <= rule.dual_targets:
                        for j in uncertain:
                            if j not in duals:
                                auxiliary_work = self.transform+rule.dual_iterations*self.step_work+self.normal_work+8*self.n
                                if not ledger.fits(auxiliary_work):
                                    continue
                                ledger.add("auxiliary_solve", auxiliary_work)
                                q, defect, _ = self._dual(int(j), rule.dual_iterations)
                                duals[j] = (q, defect)
                            dot_work = 8*self.n+12*self.width*max(1, 2**(self.width//2)-1)
                            if not ledger.fits(dot_work):
                                continue
                            ledger.add("auxiliary_check", dot_work)
                            q, defect = duals[j]
                            center = symbols[j]+np.vdot(q, true_r)
                            dual_radius = defect*residual/alpha+numerical_guard
                            certain = bit_margins(np.array([center]), self.modulation)[0] > dual_radius
                            fresh = certain & (~certificate[j])
                            frozen_bits[j, fresh] = demodulate(np.array([center]), self.modulation)[fresh]
                            certificate[j] |= fresh
                bound = int((~certificate).sum())
                if rule.method == "packing" and target > 0 and bound > target:
                    packing_work = 20*decision_work*max(1, 2**(self.width//2))+12*decision_work*np.ceil(np.log2(bit_count))
                    if ledger.fits(packing_work):
                        ledger.add("gray_energy_packing", packing_work)
                        bound = min(bound, gray_energy_bound(symbols, self.modulation, radius, ~certificate, target))
                if rule.method == "gray" and target > 0 and bound > target:
                    gate_work = 10*decision_work
                    dual_work = live_symbols*(12*2**(self.width//2)+30*(self.width//2+1)+8*self.width) if self.width > 2 else 0
                    if ledger.fits(gate_work+dual_work):
                        ledger.add("gray_target_gate", gate_work)
                        idx = selection.symbols
                        upper, used_dual = targeted_gray_bound(symbols[idx], self.modulation, radius, (~certificate)[idx], target, margins[idx])
                        if used_dual:
                            ledger.add("gray_region_dual", dual_work)
                        bound = min(bound, upper)
                if refiner is not None and bound > target:
                    bound = min(bound, refiner.bound(symbols, ~certificate, target, radius,
                                                    energy, numerical_guard, ledger, margins))
                stop = bound <= target
            output = bits.copy()
            output[certificate] = frozen_bits[certificate]
            check_seconds += time.perf_counter()-check_started
            if keep_trace:
                trace.append(dict(iteration=count, work=ledger.total, coverage=float(certificate[selected_mask].mean()),
                                  disagreement_bound=bound/bit_count, residual=float(residual),
                                  radius=float(radius), recursive_drift=float(drift), output=output[selected_mask].copy(),
                                  energy_bound=float(energy) if certified_mode else None,
                                  refined_radius_max=float(np.max(radii)) if certified_mode else None,
                                  refined_energy=None if refiner is None else refiner.current_eta,
                                  soft=symbols.copy(), certificate=(certificate & selected_mask).ravel().copy()))
            if stop:
                status = "certified_exact_arithmetic" if certified_mode else "heuristic_stop"
                break
            if restarted:
                r = true_r.copy()
                z = r/self.diagonal
                p, rho = z.copy(), float(np.vdot(r, z).real)
                radau, leverage = 1/self.mu, np.zeros(self.n)
                restarts += 1
            gap = rule.period
            if rule.schedule == "adaptive" and certified_mode and previous_radius is not None and 0 < radius < previous_radius:
                available = margins[~certificate]
                if len(available):
                    wanted = np.partition(available, min(target, len(available)-1))[min(target, len(available)-1)]
                    if rule.method in ("packing", "gray"):
                        wanted *= np.sqrt(max(1, target)/max(1, self.width//2))
                    rate = np.log(radius/previous_radius)/max(1, count-previous_iteration)
                    if wanted > 0 and rate < 0:
                        gap = int(np.clip(np.ceil(np.log(wanted/radius)/rate), 2, 12))
            previous_radius, previous_iteration = radius, count
            next_check = k+gap
        symbols = self.wave.analysis(x)
        output = np.zeros((self.n, self.width), np.uint8)
        output[selection.symbols] = demodulate(symbols[selection.symbols], self.modulation).reshape(live_symbols, self.width)
        output[certificate] = frozen_bits[certificate]
        # A packing bound applies to the checked iterate only. If work/iteration
        # exhaustion occurs after that check, only the sticky individual bits
        # retain their guarantee for the new current output.
        if not status.startswith("certified"):
            bound = int((~certificate).sum())
        return {"bits": output[selected_mask], "soft": symbols, "time_estimate": x,
                "output_bit_mask": selected_mask.ravel(), "payload_bit_count": bit_count,
                "certified": (certificate & selected_mask)[selected_mask], "unknown_fraction": float((~certificate).sum()/bit_count),
                "disagreement_bound": bound/bit_count, "status": status,
                "met_requested_bound": bool(certified_mode and bound <= target),
                "floating_point_certified": False, "iterations": count, "checks": checks,
                "restarts": restarts, "work": ledger.total, "work_parts": ledger.parts,
                "elapsed_s": self.setup_elapsed+time.perf_counter()-started,
                "process_cpu_s": self.setup_cpu+time.process_time()-cpu_started,
                "check_seconds": check_seconds,
                "trace": trace, "alpha": self.alpha, "beta": self.beta,
                "refinement": {} if refiner is None else refiner.metrics,
                "array_storage_bytes": int(sum(a.data.nbytes+a.indices.nbytes+a.indptr.nbytes for a in (self.matrix, self.adjoint))
                                           +16*self.n*12+self.diagonal.nbytes+sum(q.nbytes for q, _ in duals.values()))}
