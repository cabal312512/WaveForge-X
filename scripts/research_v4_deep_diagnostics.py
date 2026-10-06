"""Evaluator-only exact Gray enumeration and small high-precision diagnostics."""
import itertools
from fractions import Fraction

import numpy as np
import pandas as pd
from research_v4_deep import OUT, Budget, snapshot

from waveforge6g.channels import ChannelRealization
from waveforge6g.core.modulation import demodulate
from waveforge6g.experiments.research_v2 import sha, write_json
from waveforge6g.experiments.research_v4_precision import decimal_reference
from waveforge6g.receivers.certiphy import CertiPHY, StopRule, gray_energy_bound
from waveforge6g.receivers.certiphy_deep import DeepRefiner, region_costs, saturated_dp
from waveforge6g.waveforms import create_waveform


def exact_gray_enumeration(symbols, side):
    """Independent complete REGION enumeration using exact rational arithmetic.

    Fixed represented normalization constant; ideal real comparisons with that
    constant. Closed cells enclose tie rules; boundary equality remains unknown.
    """
    coordinates = [Fraction(float(v)) for v in np.r_[symbols.real, symbols.imag]]
    scale = Fraction(float(np.sqrt(2*(side*side-1)/3)))
    cuts = [Fraction(2*q-side)/scale for q in range(1, side)]
    current = [sum(x >= cut for cut in cuts) for x in coordinates]
    costs = []
    for x in coordinates:
        row = []
        for j in range(side):
            distance = max(Fraction(0), cuts[j-1]-x if j else Fraction(0),
                           x-cuts[j] if j < side-1 else Fraction(0))
            row.append(distance*distance)
        costs.append(row)
    maximum = len(coordinates)*(side.bit_length()-1)
    best = [None]*(maximum+1)
    for choice in itertools.product(range(side), repeat=len(coordinates)):
        h = sum(((a ^ (a >> 1)) ^ (b ^ (b >> 1))).bit_count() for a,b in zip(choice,current,strict=True))
        cost = sum(costs[i][a] for i,a in enumerate(choice))
        if best[h] is None or cost < best[h]:
            best[h] = cost
    for h in range(maximum-1, -1, -1):
        if best[h+1] is not None and (best[h] is None or best[h+1] < best[h]):
            best[h] = best[h+1]
    return best


def run():
    destination = OUT/"diagnostics"
    if (destination/"manifest.json").exists():
        raise RuntimeError("diagnostics already preserved")
    destination.mkdir(exist_ok=True)
    budget = Budget()
    enum_rows, counterexamples = [], []
    rng = np.random.default_rng(94001)
    for mod, side in (("qam16", 4), ("qam64", 8)):
        for sample in range(6):
            budget.check()
            symbols = rng.uniform(-1.2, 1.2, 2)+1j*rng.uniform(-1.2, 1.2, 2)
            minima = exact_gray_enumeration(symbols, side)
            active = np.ones((2, 2*(side.bit_length()-1)), bool)
            costs = region_costs(symbols, mod, active)
            for q in range(1, active.size):
                # Strictly between consecutive exact costs: no tie ambiguity.
                energy = (minima[q]+minima[q+1])/2
                if minima[q] == minima[q+1]:
                    continue
                radius = np.sqrt(float(energy))
                old = gray_energy_bound(symbols, mod, radius, active, q)
                lower = saturated_dp(costs, q+1)
                assert Fraction(float(lower)) <= minima[q+1]
                exact_u = max(h for h,v in enumerate(minima) if v is not None and v <= energy)
                enum_rows.append(dict(modulation=mod, sample=sample, target=q, exact_u=exact_u,
                                      old_upper=old, dp_lower=lower, exact_minimum=float(minima[q+1]),
                                      energy=float(energy), gap=old-exact_u))
                if old > exact_u and len(counterexamples) < 6:
                    counterexamples.append(dict(modulation=mod, symbols=[[s.real,s.imag] for s in symbols],
                                               radius=float(radius), old_upper=old, exact_u=exact_u,
                                               minimum_numerator=str(minima[q+1].numerator),
                                               minimum_denominator=str(minima[q+1].denominator),
                                               target=q, dp_lower=lower))
    pd.DataFrame(enum_rows).to_csv(destination/"exact_regions.csv", index=False)
    write_json(destination/"counterexamples.json", counterexamples)
    small_rows, checks = [], []
    for name in ("ofdm", "otfs", "afdm"):
        wave = create_waveform(name, 8, 2, 4)
        for condition in ("quasi", "cancellation", "boundary"):
            channel = ChannelRealization([0, 1], [1/np.sqrt(2), -.997/np.sqrt(2)], [.012, -.009], 64000)
            noise = 1e-10 if condition == "cancellation" else 1e-4
            receiver = CertiPHY(wave, channel, noise, "qam64")
            y = rng.normal(size=8)+1j*rng.normal(size=8)
            if condition == "boundary":
                y *= 0.
            gold = decimal_reference(receiver.matrix.toarray(), y, noise, digits=90)
            gold2 = decimal_reference(receiver.matrix.toarray(), y, noise, digits=120)
            ref_symbols = wave.analysis(gold)
            for mode in ("gray", "spectral"):
                budget.check(1)
                result = receiver.solve(y, StopRule(method="gray", delta=0., period=1, max_iterations=32,
                                                    schedule="periodic"), keep_trace=True,
                                        refiner=None if mode == "gray" else DeepRefiner("spectral"))
                ref_bits = demodulate(ref_symbols, "qam64")
                violations = 0
                for point in result["trace"]:
                    difference = point["soft"]-ref_symbols
                    error = wave.synthesis(difference)
                    actual_energy = float(np.vdot(error, receiver.normal(error)).real)
                    active = ~point["certificate"].reshape(8, 6)
                    costs = region_costs(point["soft"], "qam64", active)
                    ball_energy = point["radius"]**2
                    exact_u = max([0]+[q for q in range(1, int(active.sum())+1)
                                       if saturated_dp(costs, q) <= ball_energy])
                    upper = gray_energy_bound(point["soft"], "qam64", point["radius"], active, 1)
                    changed = point["output"] != ref_bits
                    violations += int(np.sum(changed & point["certificate"]))
                    small_rows.append(dict(waveform=name, condition=condition, mode=mode,
                                           iteration=point["iteration"], work=point["work"],
                                           actual_error=float(np.linalg.norm(difference)), radius=point["radius"],
                                           actual_energy=actual_energy, radau_energy=point["energy_bound"],
                                           refined_energy=point["refined_energy"],
                                           refined_radius=point["refined_radius_max"],
                                           closed_dp_upper=exact_u, r4_dual_upper=upper,
                                           actual_hamming=int(changed.sum())))
                checks.append(dict(waveform=name, condition=condition, mode=mode,
                                   precision_difference=float(np.linalg.norm(gold-gold2)),
                                   violations=violations, status=result["status"],
                                   unknown=result["unknown_fraction"], restarts=result["restarts"]))
    pd.DataFrame(small_rows).to_csv(destination/"small_systems.csv", index=False)
    write_json(destination/"numerical_checks.json", checks)
    # A spectral lower bound must decline the exactly cancelling singular mode.
    wave = create_waveform("ofdm", 32, 4, 8)
    channel = ChannelRealization([0, 1], [1/np.sqrt(2), -1/np.sqrt(2)], [0, 0], 64000)
    receiver = CertiPHY(wave, channel, 1e-12, "qpsk")
    budget.check(1)
    result = receiver.solve(np.random.default_rng(77).normal(size=32),
                            StopRule(method="gray", period=2, max_iterations=64),
                            refiner=DeepRefiner("spectral"))
    write_json(destination/"drift.json", {k:result[k] for k in ("status", "restarts", "unknown_fraction", "refinement")})
    write_json(destination/"manifest.json", dict(source_hash=snapshot(),
               files={p.name:sha(p) for p in destination.iterdir() if p.is_file() and p.name != "manifest.json"}))
    budget.save()
    print("Exact Gray regions, Decimal references and drift diagnostics saved")


if __name__ == "__main__":
    run()
