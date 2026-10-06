"""Bounded R4-Deep experiments. Development uses R4 caches only."""
import argparse
import json
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from waveforge6g.channels import ChannelRealization
from waveforge6g.channels.awgn import complex_awgn
from waveforge6g.core.modulation import bits_per_symbol, demodulate, modulate
from waveforge6g.experiments.research_cache import implementation_hash
from waveforge6g.experiments.research_v2 import sha, write_json
from waveforge6g.experiments.research_v4 import estimated_channel, reference
from waveforge6g.receivers.certiphy import CertiPHY, StopRule, gray_energy_bound
from waveforge6g.receivers.certiphy_deep import DeepRefiner, region_costs, saturated_dp
from waveforge6g.waveforms import create_waveform

ROOT = Path(__file__).resolve().parents[1]
OUT, DOC = ROOT/"results/research_v4_deep", ROOT/"docs/research_v4_deep"
for directory in (OUT, DOC):
    directory.mkdir(parents=True, exist_ok=True)


class Budget:
    def __init__(self):
        self.path = OUT/"budget.json"
        self.previous = json.loads(self.path.read_text()) if self.path.exists() else {"seconds": 0., "receivers": 0}
        self.start = time.perf_counter()

    def check(self, count=0):
        elapsed = self.previous["seconds"]+time.perf_counter()-self.start
        size = sum(p.stat().st_size for root in (OUT, DOC) for p in root.rglob("*") if p.is_file())
        if elapsed >= 4*3600 or size >= 5*1024**3:
            self.save()
            raise RuntimeError("R4-Deep experiment limit reached; completed evidence retained")
        self.previous["receivers"] += count

    def save(self):
        write_json(self.path, {**self.previous, "seconds": self.previous["seconds"]+time.perf_counter()-self.start})


def snapshot():
    code = implementation_hash()
    path = OUT/"sources"/(code+".zip")
    path.parent.mkdir(exist_ok=True)
    if not path.exists():
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
            for source in (ROOT/"src/waveforge6g").rglob("*.py"):
                archive.write(source, source.relative_to(ROOT))
            archive.write(Path(__file__), Path(__file__).relative_to(ROOT))
    return code


def development(budget, stage):
    if (OUT/stage/"manifest.json").exists():
        raise RuntimeError("completed development preserved")
    destination = OUT/stage
    destination.mkdir(exist_ok=True)
    metadata = pd.read_csv(ROOT/"docs/research_v4/develop.csv").drop_duplicates(["seed", "waveform", "modulation"])
    metadata = pd.concat([metadata, pd.read_csv(ROOT/"docs/research_v4/screen.csv").drop_duplicates(["seed", "waveform", "modulation"])])
    modes = ("r4", "envelope", "dp") if stage == "screen" else ("r4", "target", "spectral")
    rows, diagnostics, trace_rows = [], [], []
    for case in metadata.itertuples():
        oldstage = "screen" if case.seed < 62000 else "develop"
        source = ROOT/f"results/research_v4/{oldstage}/{case.seed}_{case.waveform}_{case.modulation}/raw.npz"
        raw = np.load(source)
        channel = ChannelRealization.from_dict(json.loads(str(raw["estimated_channel_json"])))
        wave = create_waveform(case.waveform, case.n, case.n//8, 16)
        reference = raw["reference_symbols"]
        ref_bits = demodulate(reference, case.modulation)
        for mode in modes:
            budget.check(1)
            receiver = CertiPHY(wave, channel, 10**(-case.snr/10), case.modulation)
            result = receiver.solve(raw["received"], StopRule(method="gray", delta=.01, schedule="periodic"),
                                    max_work=8e6*case.n/128, keep_trace=True,
                                    refiner=None if mode == "r4" else DeepRefiner(mode))
            rows.append(dict(seed=case.seed, n=case.n, waveform=case.waveform, modulation=case.modulation,
                             condition=case.condition, snr=case.snr, mode=mode, work=result["work"],
                             iterations=result["iterations"], checks=result["checks"], elapsed_s=result["elapsed_s"],
                             disagreement=float(np.mean(result["bits"] != ref_bits)), bound=result["disagreement_bound"],
                             met=result["met_requested_bound"], parts=json.dumps(result["work_parts"]),
                             refinement=json.dumps(result["refinement"]), source=str(source.relative_to(ROOT)),
                             source_sha256=sha(source)))
            for tr in result["trace"]:
                trace_rows.append(dict(seed=case.seed, n=case.n, waveform=case.waveform, modulation=case.modulation,
                                       mode=mode, iteration=tr["iteration"], work=tr["work"], bound=tr["disagreement_bound"],
                                       radius=tr["radius"], error=float(np.linalg.norm(tr["soft"]-reference)),
                                       disagreement=float(np.mean(tr["output"] != ref_bits))))
            if mode == "r4":
                # Exact separable DP is evaluator-only, sampled late checks.
                for tr in result["trace"][-3:]:
                    active = ~tr["certificate"].reshape(case.n, -1)
                    q = int(.01*active.size)
                    costs = region_costs(tr["soft"], case.modulation, active)
                    minimum = saturated_dp(costs, q+1)
                    old = gray_energy_bound(tr["soft"], case.modulation, tr["radius"], active, q)
                    diagnostics.append(dict(seed=case.seed, waveform=case.waveform, modulation=case.modulation,
                                            iteration=tr["iteration"], radius=tr["radius"],
                                            actual_error=float(np.linalg.norm(tr["soft"]-reference)),
                                            target=q, exact_closed_cost=minimum, energy=tr["radius"]**2,
                                            r4_upper=old, dp_proves=minimum > tr["radius"]**2))
        pd.DataFrame(rows).to_csv(destination/"frames.csv", index=False)
        print(stage, case.seed, case.waveform, case.modulation, flush=True)
    pd.DataFrame(trace_rows).to_csv(destination/"traces.csv", index=False)
    pd.DataFrame(diagnostics).to_csv(destination/"gaps.csv", index=False)
    write_json(destination/"manifest.json", {"source_hash": snapshot(), "rows": len(rows),
               "files": {p.name: sha(p) for p in destination.glob("*.csv")}})


CASES = [(128, "qam16", "quasi"), (256, "qam64", "quasi"),
         (512, "qam64", "near_singular"), (1024, "qam64", "quasi"),
         (256, "qpsk", "near_singular"), (512, "qpsk", "fast"),
         (128, "qam16", "ordinary"), (1024, "qam64", "fast")]


def channel_sequence(n, condition, seed, epoch):
    rng = np.random.default_rng(seed)
    if condition == "near_singular":
        # A nonzero Doppler perturbation of two nearly cancelling paths.
        delays, gains = np.array([0, 1]), np.array([1., -.997], complex)
        gains *= np.exp(1j*.001*epoch*np.array([1., -1.]))
        fd = np.array([.012, -.009])*(1+.1*epoch)
        snr = 40.
    else:
        paths = 8 if condition == "fast" else 4
        delays = np.sort(rng.choice(np.arange(n//8-1), paths, replace=False))
        delays[0] = 0
        gains = (rng.normal(size=paths)+1j*rng.normal(size=paths))*np.exp(-np.arange(paths)/5)
        gains *= np.exp(1j*.1*epoch*np.arange(paths))
        fd = rng.uniform(-1, 1, paths)*({"quasi": .3, "fast": 2400., "ordinary": 50.}[condition])
        snr = 32. if condition == "quasi" else (28. if condition == "fast" else 8.)
    gains /= np.linalg.norm(gains)
    return ChannelRealization(delays, gains, fd, 64000), snr


def evaluate_case(n, mod, condition, seed, epoch, wave_name, rules, nmse=0., trace=False):
    channel, snr = channel_sequence(n, condition, seed, epoch)
    wave = create_waveform(wave_name, n, n//8, 16)
    used = estimated_channel(channel, wave, nmse, seed+epoch)
    rng = np.random.default_rng(seed*17+epoch)
    truth = rng.integers(0, 2, n*bits_per_symbol(mod), dtype=np.uint8)
    noise = 10**(-snr/10)
    y = (channel.apply(wave.modulate(modulate(truth, mod)))+complex_awgn(n+n//8, noise, rng))[n//8:]
    ref = reference(wave, used, y, noise)
    ref_bits = demodulate(ref["symbols"], mod)
    from waveforge6g.receivers.certiphy import bit_margins
    known = bit_margins(ref["symbols"], mod).ravel() > ref["uncertainty"]
    model_difference = 0.
    if nmse:
        actual = reference(wave, channel, y, noise)
        model_difference = float(np.mean(demodulate(actual["symbols"], mod) != ref_bits))
    rows, outputs, traces = [], [], []
    for name, rule, mode in rules:
        receiver = CertiPHY(wave, used, noise, mod)
        result = receiver.solve(y, rule, max_work=8e6*n/128, keep_trace=trace,
                                refiner=None if mode is None else DeepRefiner(mode))
        mismatch = result["bits"] != ref_bits
        parts = result["work_parts"]
        rows.append(dict(seed=seed, epoch=epoch, n=n, modulation=mod, condition=condition, snr=snr,
                         waveform=wave_name, nmse=nmse, method=name, delta=rule.delta, bits=len(truth),
                         ber=float(np.mean(result["bits"] != truth)), reference_ber=float(np.mean(ref_bits != truth)),
                         disagreement=float(mismatch.mean()), bound=result["disagreement_bound"],
                         violations=int(np.sum(mismatch & result["certified"] & known)),
                         reference_unknown=int((~known).sum()), model_difference=model_difference,
                         unknown=result["unknown_fraction"], requested_met=result["met_requested_bound"],
                         status=result["status"], iterations=result["iterations"], checks=result["checks"],
                         restarts=result["restarts"], work=result["work"], elapsed_s=result["elapsed_s"],
                         cpu_s=result["process_cpu_s"], check_s=result["check_seconds"],
                         storage_bytes=result["array_storage_bytes"]+24*n*(mode in ("spectral", "envelope")),
                         preparation_work=sum(v for k, v in parts.items() if "preparation" in k or "directions" in k),
                         check_work=sum(v for k, v in parts.items() if k not in ("pcg", "preparation", "rhs", "transmit_and_output")),
                         parts=json.dumps(parts), refinement=json.dumps(result["refinement"])))
        outputs.append(result["bits"])
        for tr in result["trace"]:
            traces.append(dict(method=name, **{k:v for k,v in tr.items() if k not in ("output", "soft", "certificate")},
                               error=float(np.linalg.norm(tr["soft"]-ref["symbols"])),
                               disagreement=float(np.mean(tr["output"] != ref_bits))))
    raw = dict(received=y, truth=truth, reference_symbols=ref["symbols"], reference_known=known,
               outputs=np.stack(outputs), methods=np.array([r[0] for r in rules]),
               channel_json=np.array(json.dumps(channel.to_dict())), estimated_channel_json=np.array(json.dumps(used.to_dict())))
    return rows, traces, raw


def rules_for(stage):
    methods = ("gray", "envelope", "target", "spectral") if stage == "hard_develop" else (
        "gray", "radau", "global", "residual_fast", "stable_fast", "spectral")
    result = []
    for delta, label, tol, stable, period in ((0., "d0", 1e-5, 12, 2), (.01, "d001", 1e-4, 2, 1), (.05, "d005", 1e-3, 2, 1)):
        for method in methods:
            mode = method if method in ("envelope", "target", "spectral") else None
            rule = StopRule(method="gray" if mode else method, delta=delta, tolerance=tol,
                            stable_checks=stable, period=period if method == "stable_fast" else 4,
                            schedule="periodic" if stage == "hard_develop" else "adaptive")
            result.append((method+"_"+label, rule, mode))
    return result


def new_trajectories(budget, stage):
    from dataclasses import asdict
    code = snapshot()
    rules = rules_for(stage)
    protocol = dict(source_hash=code, cases=CASES, clusters_per_case=1 if stage == "hard_develop" else 6,
                    epochs=1 if stage == "hard_develop" else 4,
                    seed_base=91001 if stage == "hard_develop" else 92001,
                    rules=[dict(name=n, rule=asdict(r), refiner=m) for n,r,m in rules],
                    primary="spectral versus R4 Gray and scalar Radau; paired channel trajectory cluster",
                    selection="No test-based threshold selection; same R4 solver/Jacobi; all failures retained",
                    limits=dict(wall_seconds=14400, bytes=5*1024**3))
    destination = OUT/stage
    destination.mkdir(exist_ok=True)
    protocol_path = destination/"protocol.json"
    if protocol_path.exists():
        if json.loads(protocol_path.read_text()) != json.loads(json.dumps(protocol)):
            raise RuntimeError("frozen protocol/source differs; preserve completed results")
    else:
        write_json(protocol_path, protocol)
    for index, (n, mod, condition) in enumerate(CASES):
        for repeat in range(protocol["clusters_per_case"]):
            seed = protocol["seed_base"]+index*10+repeat
            folder = destination/str(seed)
            if (folder/"manifest.json").exists():
                manifest = json.loads((folder/"manifest.json").read_text())
                assert all(sha(folder/p)==s for p,s in manifest["files"].items())
                continue
            folder.mkdir(exist_ok=True)
            rows, traces = [], []
            for epoch in range(protocol["epochs"]):
                for wave_name in ("ofdm", "otfs", "afdm"):
                    budget.check(len(rules))
                    part, points, raw = evaluate_case(n, mod, condition, seed, epoch, wave_name, rules,
                                                     trace=repeat == 0 and epoch == 0)
                    rows.extend(part)
                    traces.extend([dict(seed=seed, epoch=epoch, waveform=wave_name, **p) for p in points])
                    np.savez_compressed(folder/f"{epoch}_{wave_name}.npz", **raw)
            pd.DataFrame(rows).to_csv(folder/"frames.csv", index=False)
            pd.DataFrame(traces).to_csv(folder/"traces.csv", index=False)
            write_json(folder/"manifest.json", dict(source_hash=code, protocol_sha256=sha(protocol_path),
                       files={p.name:sha(p) for p in folder.iterdir() if p.is_file() and p.name != "manifest.json"}))
            budget.save()
            print(stage, seed, n, mod, condition, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["screen", "redesign", "hard_develop", "confirm"])
    args = parser.parse_args()
    budget = Budget()
    snapshot()
    try:
        if args.stage in ("screen", "redesign"):
            development(budget, args.stage)
        else:
            new_trajectories(budget, args.stage)
    finally:
        budget.save()
