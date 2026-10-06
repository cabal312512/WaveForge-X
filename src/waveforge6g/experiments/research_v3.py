"""R3 bounded compute/BER experiments; reuse R2 RNG, integrity and budget machinery."""

import argparse
import hashlib
import itertools
import json
import time
import traceback
import zipfile
from dataclasses import asdict

import numpy as np
import pandas as pd

from ..channels import ChannelRealization
from ..channels.awgn import complex_awgn, noise_variance
from ..core.modulation import bits_per_symbol, demodulate, modulate
from ..receivers.budgeted import (
    BudgetedReceiver,
    ReceiverAction,
    cost_model,
    symbol_channel,
    time_channel,
)
from ..waveforms import create_waveform
from .expected_loss import role_seed
from .research_cache import implementation_hash
from .research_v2 import ROOT, Budget, sha, write_json

OUT = ROOT / "results/research_v3"
DOC = ROOT / "docs/research_v3"
CFG = ROOT / "configs/research_v3"
PROFILES = {"low_sparse": (2, 5.), "high_sparse": (2, 2400.), "high_rich": (8, 2400.)}
SPARSE_TIERS = ((4, 2), (4, 8), (8, 4), (8, 16), (16, 8), (32, 16))


def actions(n):
    result = []
    for name in ("ofdm", "otfs", "afdm"):
        result.extend(ReceiverAction(name, "symbol", min(n, keep), k) for keep, k in SPARSE_TIERS)
        result.extend(ReceiverAction(name, "time", 0, k) for k in (2, 4, 8, 16))
        result.append(ReceiverAction(name, "dense", n, 0))
    return result


def make_channel(n, profile, seed):
    p, doppler = PROFILES[profile]
    rng = np.random.default_rng(role_seed(seed, "r3_channel", n, profile))
    delays = np.sort(rng.choice(np.arange(n//8-1), size=p, replace=False))
    delays[0] = 0
    powers = 10**(-np.linspace(0, 8, p)/10)
    powers /= powers.sum()
    gains = np.sqrt(powers/2)*(rng.normal(size=p)+1j*rng.normal(size=p))
    frequencies = rng.uniform(-doppler, doppler, p)
    return ChannelRealization(delays, gains, frequencies, 64000)


def link_work(waveform, channel, action, modulation, iterations=None):
    record = cost_model(waveform, channel.n_paths, action.domain, action.keep,
                        action.iterations if iterations is None else iterations)
    bits = bits_per_symbol(modulation)
    record["codec_work"] = 8*waveform.n_symbols*(2**bits)+4*waveform.n_symbols*bits
    record["total_work"] += record["codec_work"]
    return record


def evaluate_state(n, modulation, channel, snr, seed, epoch, replicas=16, candidates=None):
    """Independent selection/validation banks, CRN across candidates and honest setup charging.

    Replicas are fixed-state evaluator draws, NOT physically coherent frames over
    which preparation can be amortized. Every candidate is billed reuse_frames=1.
    """
    candidates = actions(n) if candidates is None else candidates
    width, cp = n*bits_per_symbol(modulation), n//8
    variance = noise_variance(snr)
    banks = {}
    for role in ("selection", "validation", "online"):
        count = 1 if role == "online" else replicas
        bits, symbols, noises = [], [], []
        for r in range(count):
            rng = np.random.default_rng(role_seed(seed, "r3_"+role, epoch, r))
            bit = rng.integers(0, 2, width, dtype=np.uint8)
            bits.append(bit)
            symbols.append(modulate(bit, modulation))
            noises.append(complex_awgn(n+cp, variance, rng))
        banks[role] = (bits, symbols, noises)
    raw = {role+"_errors": np.empty((len(banks[role][0]), len(candidates)), np.int32) for role in banks}
    raw["bit_count"] = np.array(width)
    raw["validation_iterations"] = np.empty((replicas, len(candidates)), np.int16)
    raw["validation_residual"] = np.empty((replicas, len(candidates)))
    raw["validation_actual_work"] = np.empty((replicas, len(candidates)))
    rows, history = [], []
    for name in ("ofdm", "otfs", "afdm"):
        indices = [i for i, a in enumerate(candidates) if a.waveform == name]
        if not indices:
            continue
        start = time.perf_counter()
        wave = create_waveform(name, n, cp, 16)
        matrix_time = time_channel(wave, channel)
        time_setup = time.perf_counter()-start
        start = time.perf_counter()
        matrix_symbol = symbol_channel(wave, matrix_time) if any(candidates[i].domain != "time" for i in indices) else None
        symbol_setup = time.perf_counter()-start
        received = {}
        tx_times = []
        for role, (_, symbols, noises) in banks.items():
            received[role] = []
            for symbol, noise in zip(symbols, noises, strict=True):
                start = time.perf_counter()
                transmitted = wave.modulate(symbol)
                tx_times.append(time.perf_counter()-start)
                # The original full propagation kernel, never the sparsified matrix.
                received[role].append((channel.apply(transmitted)+noise)[cp:])
        for a in indices:
            action = candidates[a]
            receiver = BudgetedReceiver(wave, channel, variance, action, matrix_time, matrix_symbol)
            cpu_times, iteration_counts, residuals = [], [], []
            for role, (bits, _, _) in banks.items():
                for r, (observed, truth_bits) in enumerate(zip(received[role], bits, strict=True)):
                    estimate, diagnostics = receiver.solve(observed)
                    decode_start = time.perf_counter()
                    errors = int(np.count_nonzero(demodulate(estimate, modulation) != truth_bits))
                    decode_s = time.perf_counter()-decode_start
                    raw[role+"_errors"][r, a] = errors
                    if role == "validation":
                        raw["validation_iterations"][r, a] = diagnostics["iterations"]
                        raw["validation_residual"][r, a] = diagnostics["relative_normal_residual"]
                        raw["validation_actual_work"][r, a] = link_work(wave, channel, action, modulation,
                                                                         diagnostics["iterations"])["total_work"]
                        cpu_times.append(diagnostics["solve_cpu_s"]+decode_s)
                        iteration_counts.append(diagnostics["iterations"])
                        residuals.append(diagnostics["relative_normal_residual"])
                        if r == 0:
                            history.append({"action": action.key, "residual": diagnostics["residual_history"]})
            cost = link_work(wave, channel, action, modulation)
            setup_s = time_setup+(symbol_setup if action.domain != "time" else 0)+receiver.setup_cpu_s
            rows.append({"action_index": a, "action": action.key, **asdict(action), **cost,
                         "discarded_energy": receiver.discarded_energy,
                         "selection_ber": float(raw["selection_errors"][:, a].mean()/width),
                         "ber": float(raw["validation_errors"][:, a].mean()/width),
                         "frame_ber_sd": float(raw["validation_errors"][:, a].std(ddof=1)/width),
                         "error_count": int(raw["validation_errors"][:, a].sum()),
                         "bit_count": width*replicas, "frames": replicas,
                         "mean_iterations": float(np.mean(iteration_counts)),
                         "normal_residual": float(np.mean(residuals)),
                         "actual_work_mean": float(raw["validation_actual_work"][:, a].mean()),
                         "setup_cpu_s": setup_s, "frame_cpu_s": float(np.mean(cpu_times)+np.mean(tx_times)),
                         "total_cpu_s": float(setup_s+np.mean(cpu_times)+np.mean(tx_times)),
                         "array_storage_bytes": int(receiver.array_bytes),
                         "n": n, "modulation": modulation, "snr_db": snr,
                         "n_paths": channel.n_paths, "max_doppler_hz": float(max(abs(channel.dopplers_hz)))})
    return pd.DataFrame(rows).sort_values("action_index"), raw, history


def run_state(stage, n, modulation, profile, snr, seed, budget, replicas=16):
    channel = make_channel(n, profile, seed)
    settings = {"n": n, "modulation": modulation, "profile": profile, "snr": snr, "seed": seed,
                "replicas": replicas, "channel": channel.to_dict(), "cost_version": "r3-real-work-v1",
                "source_hash": implementation_hash(), "actions": [asdict(a) for a in actions(n)], "reuse_frames": 1}
    key = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()
    destination = OUT/stage/f"N{n}_{modulation}_{profile}_S{snr}_{seed}_{key[:8]}"
    manifest_path = destination/"manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("status") == "complete" and all(sha(destination/name) == value for name, value in manifest["files"].items()):
            print("cached", destination.name, flush=True)
            return destination
    budget.reserve(len(actions(n))*(2*replicas+1))
    destination.mkdir(parents=True, exist_ok=True)
    snapshot = OUT/"sources"/(settings["source_hash"]+".zip")
    if not snapshot.exists():
        snapshot.parent.mkdir(exist_ok=True, parents=True)
        with zipfile.ZipFile(snapshot, "w", zipfile.ZIP_DEFLATED) as archive:
            for path in (ROOT/"src/waveforge6g").rglob("*.py"):
                archive.write(path, path.relative_to(ROOT))
    manifest = {"status": "running", "settings": settings, "key": key}
    write_json(manifest_path, manifest)
    write_json(OUT/"STATUS.json", {"status": "running", "stage": stage, "run": str(destination)})
    start = time.perf_counter()
    try:
        table, data, history = evaluate_state(n, modulation, channel, snr, seed, 0, replicas)
        table.to_csv(destination/"summary.csv", index=False)
        np.savez_compressed(destination/"replicas.npz", **data)
        np.savez_compressed(destination/"paired_covariance.npz", covariance=np.cov(data["validation_errors"].T/data["bit_count"]))
        write_json(destination/"convergence.json", history)
        manifest.update(status="complete", elapsed_s=time.perf_counter()-start,
                        files={p.name: sha(p) for p in destination.iterdir() if p.name != "manifest.json"})
        write_json(manifest_path, manifest)
        budget.save()
        print(f"complete {destination.name} {manifest['elapsed_s']:.2f}s", flush=True)
        return destination
    except BaseException:
        manifest.update(status="failed", error=traceback.format_exc())
        write_json(manifest_path, manifest)
        budget.save()
        raise


def completed(stage):
    result = []
    for path in sorted((OUT/stage).glob("*/manifest.json")):
        m = json.loads(path.read_text())
        if m["status"] == "complete":
            if not all(sha(path.parent/k) == v for k, v in m["files"].items()):
                raise RuntimeError(f"checksum mismatch: {path}")
            result.append((path.parent, m))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["pilot", "grid", "probe", "adapt", "report"])
    args = parser.parse_args()
    for p in (OUT, DOC, CFG):
        p.mkdir(parents=True, exist_ok=True)
    budget = Budget(output_root=OUT)
    if args.stage == "pilot":
        for n, profile in itertools.product((128, 256), ("low_sparse", "high_rich")):
            run_state("pilot", n, "qpsk", profile, 12, 41001, budget)
    elif args.stage == "grid":
        for n, mod, profile, snr, seed in itertools.product((128, 256), ("qpsk", "qam16"), PROFILES, (8, 18), (42001, 42002, 42003)):
            run_state("grid", n, mod, profile, snr, seed, budget)
    elif args.stage == "probe":
        from .research_v3_probe import run
        run()
    elif args.stage == "adapt":
        from .research_v3_adapt import run
        run(budget)
    else:
        from .research_v3_report import report
        report()
    budget.save()
    write_json(OUT/"STATUS.json", {"status": "stage_complete", "stage": args.stage})


if __name__ == "__main__":
    main()
