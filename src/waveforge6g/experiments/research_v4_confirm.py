"""Validation-only stopping choices, followed by untouched trajectory seeds."""

import itertools
import json
from dataclasses import asdict

import numpy as np
import pandas as pd

from ..channels import ChannelRealization
from ..channels.nonstationary.engine import NonStationaryChannel
from ..core.modulation import demodulate
from ..receivers.certiphy import CertiPHY, StopRule
from ..waveforms import create_waveform
from .research_cache import implementation_hash
from .research_v2 import sha, write_json
from .research_v4 import CFG, DOC, OUT, candidate_rules, frame

DELTAS = {"d0": 0., "d001": .01, "d005": .05}


def calibration(budget):
    path = DOC/"calibration.csv"
    if path.exists():
        return pd.read_csv(path)
    rules = {f"fixed_{k}": StopRule(method="fixed", fixed_iterations=k) for k in (96, 128)}
    rules.update({f"stable_{s}_p2": StopRule(method="stable", stable_checks=s, period=2, schedule="periodic") for s in (8, 12)})
    for label, delta in DELTAS.items():
        for method, schedule in itertools.product(("global", "radau", "gray"), ("adaptive", "periodic")):
            rules[f"{method}_{schedule}_{label}"] = StopRule(method=method, delta=delta, schedule=schedule)
    development = pd.read_csv(DOC/"develop.csv")
    rows = []
    for _, saved in development.drop_duplicates(["seed", "waveform", "modulation"]).iterrows():
        raw = np.load(OUT/"develop"/f"{saved.seed}_{saved.waveform}_{saved.modulation}"/"raw.npz")
        channel = ChannelRealization.from_dict(json.loads(str(raw["channel_json"])))
        wave = create_waveform(saved.waveform, int(saved.n), int(saved.n)//8, 16)
        reference_bits = demodulate(raw["reference_symbols"], saved.modulation)
        budget.reserve(len(rules))
        for name, rule in rules.items():
            result = CertiPHY(wave, channel, 10**(-saved.snr/10), saved.modulation).solve(raw["received"], rule,
                                                                                       max_work=8e6*saved.n/128)
            rows.append(dict(method=name, seed=saved.seed, n=saved.n, waveform=saved.waveform, modulation=saved.modulation,
                             delta=rule.delta, work=result["work"], iterations=result["iterations"],
                             disagreement=float(np.mean(result["bits"] != reference_bits)),
                             coverage=1-result["unknown_fraction"], bound=result["disagreement_bound"],
                             requested_met=result["met_requested_bound"], status=result["status"],
                             certified_violation_count=int(np.sum((result["bits"] != reference_bits) & result["certified"] & raw["reference_known"]))))
    result = pd.DataFrame(rows)
    result.to_csv(path, index=False)
    write_json(CFG/"calibration_rules.json", {name: asdict(rule) for name, rule in rules.items()})
    budget.save()
    return result


def freeze(calibrated):
    development = pd.read_csv(DOC/"develop.csv")
    combined = pd.concat((development, calibrated), ignore_index=True)
    lookup = candidate_rules(redesigned=True)
    lookup.update({name: StopRule(**spec) for name, spec in json.loads((CFG/"calibration_rules.json").read_text()).items()})
    chosen, audit = {}, []
    for label, delta in DELTAS.items():
        for prefix in ("fixed", "residual_fast", "stable"):
            subset = combined[combined.method.str.startswith(prefix+"_")]
            grouped = subset.groupby("method").agg(work=("work", "mean"), worst=("disagreement", "max"))
            eligible = grouped[grouped.worst <= delta]
            if eligible.empty:
                name = grouped.sort_values(["worst", "work"]).index[0]
            else:
                name = eligible.sort_values("work").index[0]
            spec = asdict(lookup[name])
            spec["delta"] = delta
            chosen[f"{prefix}_{label}"] = StopRule(**spec)
            audit.append(dict(label=label, family=prefix, selected=name, development_worst=grouped.loc[name, "worst"],
                              development_mean_work=grouped.loc[name, "work"], target=delta,
                              validation_pass=bool(grouped.loc[name, "worst"] <= delta)))
        for method in ("global", "radau", "gray"):
            chosen[f"{method}_{label}"] = StopRule(method=method, delta=delta, schedule="adaptive")
    chosen["gray_periodic_d001"] = StopRule(method="gray", delta=.01, schedule="periodic")
    chosen["r3_fixed8"] = StopRule(method="fixed", fixed_iterations=8)
    chosen["r3_fixed16"] = StopRule(method="fixed", fixed_iterations=16)
    pd.DataFrame(audit).to_csv(DOC/"baseline_selection.csv", index=False)
    protocol = {"source_hash": implementation_hash(), "rules": {name: asdict(rule) for name, rule in chosen.items()},
                "development_sha256": sha(DOC/"develop.csv"), "calibration_sha256": sha(DOC/"calibration.csv"),
                "core": "48 independent seeds 71001..71048; N128/256 x 3 waveforms x 2 modulations x 4; 12 frames",
                "scale": "36 independent seeds 72001..72036; N512/1024 x 3 waveforms x 3 modulations x 2; 8 frames",
                "mismatch": "12 independent seeds 73001..73012; N256 x 3 waveforms x 2 modulations x 2; 8 frames; NMSE 0/.01/.1",
                "max_work_per_frame": "8e6 * N/128", "reference": "same estimated model full LMMSE; ambiguity retained",
                "statistical_unit": "trajectory; paired, stratified bootstrap; no iid-bit inference",
                "preconditioner": "fixed Jacobi for all methods; no new preconditioner", "floating_point_certified": False}
    path = CFG/"confirmation_protocol.json"
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise RuntimeError("frozen confirmation protocol differs; retain old evidence")
    write_json(path, protocol)
    return chosen, protocol


def config(n, epochs, reverse):
    def schedule(values):
        if reverse:
            values = values[::-1]
        return {"type": "piecewise_constant", "points": [[i*(epochs//2), v] for i, v in enumerate(values)]}
    return {"scenario": "stable_low_mobility", "epochs": epochs,
            "trajectories": {"snr_db": schedule([8, 28]), "max_doppler_hz": schedule([15, 2400]),
                             "path_count": schedule([3, 8]), "correlation": .93,
                             "delay_samples": n//8-2, "k_factor_db": -20., "fractional_doppler_severity": 1.},
            "cluster_dynamics": {"max_paths": 8, "power_ramp_frames": 0}}


def run(budget):
    calibrated = calibration(budget)
    rules, protocol = freeze(calibrated)
    cases = []
    for i, (n, wave, mod, repeat) in enumerate(itertools.product((128, 256), ("ofdm", "otfs", "afdm"), ("qpsk", "qam16"), range(4))):
        cases.append(("core", 71001+i, n, wave, mod, repeat, 12, (0.,)))
    for i, (n, wave, mod, repeat) in enumerate(itertools.product((512, 1024), ("ofdm", "otfs", "afdm"), ("qpsk", "qam16", "qam64"), range(2))):
        cases.append(("scale", 72001+i, n, wave, mod, repeat, 8, (0.,)))
    for i, (wave, mod, repeat) in enumerate(itertools.product(("ofdm", "otfs", "afdm"), ("qpsk", "qam16"), range(2))):
        cases.append(("mismatch", 73001+i, 256, wave, mod, repeat, 8, (0., .01, .1)))
    for i, (stage, seed, n, wave, mod, repeat, epochs, mismatches) in enumerate(cases):
        dest = OUT/stage/str(seed)
        if (dest/"manifest.json").exists():
            saved = json.loads((dest/"manifest.json").read_text())
            if saved.get("protocol_sha256") != sha(CFG/"confirmation_protocol.json") or not all(sha(dest/name) == digest for name, digest in saved["files"].items()):
                raise RuntimeError("incompatible/corrupt trajectory cache")
            continue
        dest.mkdir(exist_ok=True, parents=True)
        selected = {name: rule for name, rule in rules.items() if stage != "mismatch" or name.endswith("_d001")}
        budget.reserve(epochs*len(mismatches)*len(selected))
        engine = NonStationaryChannel(config(n, epochs, repeat % 2 == 1), seed, n, n//8, 64000)
        rows = []
        for epoch in range(epochs):
            state, channel = engine.step(epoch)
            for nmse in mismatches:
                evaluated, _, raw = frame(n, wave, mod, channel, state.snr_db, seed, epoch, selected, nmse)
                rows.extend(evaluated)
                np.savez_compressed(dest/f"frame_{epoch}_nmse{nmse:g}.npz", **raw)
        pd.DataFrame(rows).to_csv(dest/"frames.csv", index=False)
        write_json(dest/"manifest.json", {"status": "complete", "stage": stage, "seed": seed, "n": n, "waveform": wave,
                   "modulation": mod, "epochs": epochs, "source_hash": protocol["source_hash"],
                   "protocol_sha256": sha(CFG/"confirmation_protocol.json"),
                   "files": {path.name: sha(path) for path in dest.iterdir() if path.name != "manifest.json"}})
        budget.save()
        print(f"{i+1}/{len(cases)} {stage} {seed} N={n} {wave} {mod}", flush=True)
