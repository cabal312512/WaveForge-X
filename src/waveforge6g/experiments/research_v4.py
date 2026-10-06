"""Small development and frozen paired confirmation for decision certificates."""

import argparse
import itertools
import json
import time
import zipfile
from dataclasses import asdict

import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve
from scipy.sparse.linalg import LinearOperator, cg

from ..channels import ChannelRealization
from ..channels.awgn import complex_awgn
from ..core.modulation import bits_per_symbol, demodulate, modulate
from ..receivers.budgeted import time_channel
from ..receivers.certiphy import CertiPHY, StopRule, bit_margins, physical_spectral_bounds
from ..waveforms import create_waveform
from .expected_loss import mismatch, role_seed
from .research_cache import implementation_hash
from .research_v2 import ROOT, Budget, sha, write_json

OUT, DOC, CFG = (ROOT/p for p in ("results/research_v4", "docs/research_v4", "configs/research_v4"))


def make_channel(n, condition, seed):
    rng = np.random.default_rng(role_seed(seed, "r4_channel", n, condition))
    p = 8 if condition == "rich" else 3
    delays = np.sort(rng.choice(np.arange(n//8-1), p, replace=False))
    delays[0] = 0
    powers = np.array([.9, .06, .04]) if condition == "dominant" else 10**(-np.linspace(0, 5, p)/10)
    powers /= powers.sum()
    gains = np.sqrt(powers)*np.exp(2j*np.pi*rng.random(p))
    if condition != "dominant":
        gains *= (.5+np.abs(rng.normal(size=p)))
        gains /= np.linalg.norm(gains)
    fd = rng.uniform(-2400 if condition != "dominant" else -5, 2400 if condition != "dominant" else 5, p)
    return ChannelRealization(delays, gains, fd, 64000)


def estimated_channel(channel, wave, nmse, seed):
    if not nmse:
        return channel
    errors = mismatch(channel, nmse, role_seed(seed, "r4_csi"))
    truth, error = time_channel(wave, channel), time_channel(wave, errors)
    scale = np.sqrt(nmse*np.sum(abs(truth.data)**2)/np.sum(abs(error.data)**2))
    return ChannelRealization(channel.delays, channel.gains+scale*errors.gains,
                              channel.dopplers_hz, channel.sample_rate_hz)


def reference(wave, channel, y, noise):
    """Evaluator only. Dense small reference; matrix-free reference for large N.

    Residual enclosure marks ambiguous reference bits. Ordinary floating results
    are not relabeled exact mathematics; Decimal adversarial tests are separate.
    """
    c = time_channel(wave, channel)
    b = c.conj().T @ y
    n = wave.n_symbols
    if n <= 256:
        a = (c.conj().T @ c).toarray()+noise*np.eye(n)
        factor = cho_factor(a, lower=True)
        solution = cho_solve(factor, b)
        # Extended-precision residual and a correction using the same factor.
        wide_c = c.toarray().astype(np.clongdouble)
        wide_y = y.astype(np.clongdouble)
        residual = wide_c.conj().T @ (wide_y-wide_c @ solution.astype(np.clongdouble))-noise*solution
        solution += cho_solve(factor, np.asarray(residual, complex))
        residual = wide_c.conj().T @ (wide_y-wide_c @ solution.astype(np.clongdouble))-noise*solution
    else:
        op = LinearOperator((n, n), matvec=lambda x: c.conj().T @ (c @ x)+noise*x, dtype=complex)
        diag = np.asarray(abs(c).power(2).sum(axis=0)).ravel()+noise
        pre = LinearOperator((n, n), matvec=lambda x: x/diag, dtype=complex)
        solution, info = cg(op, b, M=pre, rtol=2e-14, atol=0, maxiter=4*n)
        if info != 0:
            raise RuntimeError("reference did not converge")
        residual = b-op@solution
    alpha, beta = physical_spectral_bounds(channel, noise)
    uncertainty = float(np.linalg.norm(residual))/alpha+128*np.finfo(float).eps*(1+np.linalg.norm(b)+beta*np.linalg.norm(solution))/alpha
    symbols = wave.analysis(solution)
    return {"solution": solution, "symbols": symbols,
            "uncertainty": uncertainty, "condition_upper": beta/alpha}


def candidate_rules(delta=.01, redesigned=False):
    result = {}
    for method in ("global", "krylov", "dual", "radau", "packing"):
        for schedule in ("periodic", "adaptive"):
            result[f"{method}_{schedule}"] = StopRule(method=method, delta=delta, schedule=schedule)
    result.update({f"fixed_{k}": StopRule(method="fixed", fixed_iterations=k) for k in (8, 16, 32, 64)})
    result.update({f"residual_{tol:g}": StopRule(method="residual", tolerance=tol, period=2, schedule="periodic")
                   for tol in (1e-2, 1e-3, 1e-4, 1e-5)})
    result["stable_3"] = StopRule(method="stable", stable_checks=3, period=2, schedule="periodic")
    if redesigned:
        result.update({f"gray_{schedule}": StopRule(method="gray", delta=delta, schedule=schedule)
                       for schedule in ("periodic", "adaptive")})
        result.update({f"residual_fast_{tol:g}": StopRule(method="residual_fast", tolerance=tol)
                       for tol in (1e-2, 1e-3, 1e-4, 1e-5)})
        result.update({f"fixed_{k}": StopRule(method="fixed", fixed_iterations=k) for k in (4, 12, 24, 48)})
        result.update({f"stable_{s}_p{p}": StopRule(method="stable", stable_checks=s, period=p, schedule="periodic")
                       for s, p in ((2, 1), (3, 1), (5, 1), (3, 4))})
    return result


def frame(n, waveform, modulation, channel, snr, seed, epoch, rules, nmse=0., trace=False):
    cpu, wall = time.process_time(), time.perf_counter()
    wave = create_waveform(waveform, n, n//8, 16)
    wave_cpu, wave_wall = time.process_time()-cpu, time.perf_counter()-wall
    used = estimated_channel(channel, wave, nmse, seed+epoch)
    rng = np.random.default_rng(role_seed(seed, "r4_payload", epoch, modulation))
    truth = rng.integers(0, 2, n*bits_per_symbol(modulation), dtype=np.uint8)
    symbols = modulate(truth, modulation)
    noise = 10**(-snr/10)
    cpu, wall = time.process_time(), time.perf_counter()
    transmitted = wave.modulate(symbols)
    tx_cpu, tx_wall = time.process_time()-cpu, time.perf_counter()-wall
    received = (channel.apply(transmitted)+complex_awgn(n+n//8, noise, rng))[n//8:]
    ref = reference(wave, used, received, noise)
    ref_bits = demodulate(ref["symbols"], modulation)
    ref_known = bit_margins(ref["symbols"], modulation).ravel() > ref["uncertainty"]
    truth_reference_disagreement = 0.
    if nmse:
        true_ref = reference(wave, channel, received, noise)
        truth_reference_disagreement = float(np.mean(demodulate(true_ref["symbols"], modulation) != ref_bits))
    rows, traces, outputs = [], [], []
    for name, rule in rules.items():
        receiver = CertiPHY(wave, used, noise, modulation)
        result = receiver.solve(received, rule, max_work=8e6*(n/128), keep_trace=trace)
        output = result["bits"]
        mismatch_bits = output != ref_bits
        violations = mismatch_bits & result["certified"] & ref_known
        rows.append(dict(method=name, n=n, waveform=waveform, modulation=modulation, snr=snr,
                         seed=seed, epoch=epoch, nmse=nmse, delta=rule.delta, bits=len(truth),
                         ber=float(np.mean(output != truth)), reference_ber=float(np.mean(ref_bits != truth)),
                         disagreement=float(mismatch_bits.mean()), certified_violation_count=int(violations.sum()),
                         reference_ambiguous=int((~ref_known).sum()), reference_uncertainty=ref["uncertainty"],
                         model_mismatch_disagreement=truth_reference_disagreement,
                         bound=result["disagreement_bound"], coverage=1-result["unknown_fraction"],
                         requested_met=result["met_requested_bound"], status=result["status"],
                         iterations=result["iterations"], checks=result["checks"], restarts=result["restarts"],
                         work=result["work"], elapsed_s=result["elapsed_s"]+wave_wall+tx_wall,
                         process_cpu_s=result["process_cpu_s"]+wave_cpu+tx_cpu,
                         array_storage_bytes=result["array_storage_bytes"], condition_upper=ref["condition_upper"],
                         work_parts=json.dumps(result["work_parts"], sort_keys=True)))
        outputs.append(output)
        for point in result["trace"]:
            traces.append(dict(method=name, **{k: v for k, v in point.items() if k not in ("output", "soft", "certificate")},
                               actual_disagreement=float(np.mean(point["output"] != ref_bits)),
                               actual_error_norm=float(np.linalg.norm(point["soft"]-ref["symbols"])),
                               certified_violation_count=int(np.sum((point["output"] != ref_bits) & point["certificate"] & ref_known))))
    raw = {"truth": truth, "received": received, "reference_symbols": ref["symbols"], "reference_known": ref_known,
           "outputs": np.stack(outputs), "methods": np.array(list(rules)), "channel_json": np.array(json.dumps(channel.to_dict())),
           "estimated_channel_json": np.array(json.dumps(used.to_dict()))}
    return rows, traces, raw


def run_development(stage, budget):
    rules = candidate_rules(redesigned=stage == "develop")
    cases = list(itertools.product((128, 256), ("dominant", "sparse", "rich"), (8, 24)))
    if stage == "develop":
        cases = list(itertools.product((128, 256), ("sparse", "rich"), (8, 24)))
    all_rows, all_traces = [], []
    for index, (n, condition, snr) in enumerate(cases):
        mods = ("qpsk",) if stage == "screen" else ("qpsk", "qam16")
        waves = ("ofdm",) if stage == "screen" else ("ofdm", "otfs", "afdm")
        for waveform, modulation in itertools.product(waves, mods):
            seed = (61001 if stage == "screen" else 62001)+index
            channel = make_channel(n, condition, seed)
            budget.reserve(len(rules))
            rows, traces, raw = frame(n, waveform, modulation, channel, snr, seed, 0, rules, trace=True)
            for row in rows:
                row["condition"] = condition
            for row in traces:
                row.update(n=n, condition=condition, snr=snr, waveform=waveform, modulation=modulation, seed=seed)
            all_rows.extend(rows)
            all_traces.extend(traces)
            destination = OUT/stage/f"{seed}_{waveform}_{modulation}"
            destination.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(destination/"raw.npz", **raw)
            print(stage, seed, n, condition, snr, waveform, modulation, flush=True)
    pd.DataFrame(all_rows).to_csv(DOC/f"{stage}.csv", index=False)
    pd.DataFrame(all_traces).to_csv(DOC/f"{stage}_traces.csv", index=False)
    write_json(OUT/stage/"manifest.json", {"stage": stage, "source_hash": implementation_hash(),
               "rules": {name: asdict(rule) for name, rule in rules.items()}, "rows": len(all_rows),
               "summary_sha256": sha(DOC/f"{stage}.csv"), "trace_sha256": sha(DOC/f"{stage}_traces.csv")})
    budget.save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["screen", "develop", "confirm", "report"])
    args = parser.parse_args()
    for p in (OUT, DOC, CFG):
        p.mkdir(exist_ok=True, parents=True)
    source_hash = implementation_hash()
    snapshot = OUT/"sources"/(source_hash+".zip")
    if not snapshot.exists():
        snapshot.parent.mkdir(exist_ok=True)
        with zipfile.ZipFile(snapshot, "w", zipfile.ZIP_DEFLATED) as archive:
            for path in (ROOT/"src/waveforge6g").rglob("*.py"):
                archive.write(path, path.relative_to(ROOT))
    budget = Budget(output_root=OUT)
    if args.stage in ("screen", "develop"):
        if (OUT/args.stage/"manifest.json").exists():
            raise RuntimeError("stage already measured; preserve evidence")
        run_development(args.stage, budget)
    elif args.stage == "confirm":
        from .research_v4_confirm import run
        run(budget)
    else:
        from .research_v4_report import report
        report()
    budget.save()


if __name__ == "__main__":
    main()
