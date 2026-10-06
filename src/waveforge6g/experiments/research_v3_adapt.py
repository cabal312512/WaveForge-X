"""Frozen R3 confirmation on fresh, paired causal channel trajectories."""

import itertools
import json
import time
import zipfile
from types import SimpleNamespace

import numpy as np
import pandas as pd

from ..channels import ChannelRealization
from ..channels.nonstationary.engine import NonStationaryChannel
from ..decision.compute_budget import BudgetSelector, observable_features
from .expected_loss import role_seed
from .research_cache import implementation_hash
from .research_v2 import sha, write_json
from .research_v3 import CFG, DOC, OUT, ROOT, actions, completed, evaluate_state, link_work


def train_models():
    """Only one pre-scheduled arm per development feedback draw is revealed."""
    models, tables, schedule = {}, [], []
    for n, mod in itertools.product((128, 256), ("qpsk", "qam16")):
        inverse = np.tile(np.eye(9)[None], (33, 1, 1))
        target = np.zeros((33, 9))
        learner = BudgetSelector(inverse, target, range(33), 711)
        serial = 0
        order = np.random.default_rng(7101).permutation(33)
        for directory, manifest in completed("grid"):
            s = manifest["settings"]
            if (s["n"], s["modulation"]) != (n, mod):
                continue
            ch = ChannelRealization.from_dict(s["channel"])
            features = observable_features(ch, s["snr"], n, mod)
            raw = np.load(directory/"replicas.npz")
            for r in range(len(raw["selection_errors"])):
                a = int(order[serial % 33])
                learner.update(features, a, float(raw["selection_errors"][r, a]/raw["bit_count"]))
                schedule.append(dict(n=n, modulation=mod, source=directory.name, replica=r, action=a))
                serial += 1
            table = pd.read_csv(directory/"summary.csv")
            table["profile"], table["channel_seed"] = s["profile"], s["seed"]
            tables.append(table)
        models[n, mod] = (learner.inverse, learner.target)
    pd.DataFrame(schedule).to_csv(DOC/"training_feedback_schedule.csv", index=False)
    return models, pd.concat(tables, ignore_index=True)


def trajectory_config(n, reverse=False):
    def pw(values):
        values = values[::-1] if reverse else values
        return {"type": "piecewise_constant", "points": [[i*8, v] for i, v in enumerate(values)]}
    return {"scenario": "stable_low_mobility", "epochs": 24,
            "trajectories": {"snr_db": pw([8, 18, 8]), "max_doppler_hz": pw([5, 2400, 2400]),
                             "path_count": pw([2, 2, 8]), "correlation": pw([.995, .95, .8]),
                             "delay_samples": n//8-2, "k_factor_db": -30.,
                             "fractional_doppler_severity": 1.},
            "cluster_dynamics": {"max_paths": 8, "power_ramp_frames": 0}}


def budget_value(n, level):
    return {"low": 180000*(n/128), "medium": 340000*(n/128), "high": 5e6*(n/128)**2}[level]


def fixed_choice(table, budget, overhead, domain=None):
    if domain:
        table = table[table.domain == domain]
    scores = table.groupby("action_index").agg(ber=("ber", "mean"), cost=("total_work", "max"))
    scores = scores[scores.cost+overhead <= budget]
    if scores.empty:
        return None
    return int(scores.sort_values(["ber", "cost"]).index[0])


def run(budget):
    models, development = train_models()
    protocol = {"version": 1, "epochs": 24, "replicas": 8, "seeds": list(range(51001, 51061)),
                "strata": "N=128/256 x QPSK/16QAM x budget low/medium/high; five independent seeds each",
                "budgets": {str(n): {b: budget_value(n, b) for b in ("low", "medium", "high")} for n in (128, 256)},
                "ridge": 1., "minimum_dwell": 2, "exploration_starts": [0, 10, 20], "prediction_tie_ber": .002,
                "feedback": "simulator true BER, only actually selected action, one online replica",
                "preparation_reuse_frames": 1, "source_hash": implementation_hash(),
                "training_schedule_sha256": sha(DOC/"training_feedback_schedule.csv"),
                "grid_manifests": {str(p.relative_to(ROOT)): sha(p/"manifest.json") for p, _ in completed("grid")}}
    frozen = CFG/"confirmation_protocol.json"
    if frozen.exists() and json.loads(frozen.read_text()) != protocol:
        raise RuntimeError("confirmation protocol already frozen; do not silently retune")
    write_json(frozen, protocol)
    archive_path = OUT/"sources"/(protocol["source_hash"]+".zip")
    if not archive_path.exists():
        with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
            for path in (ROOT/"src/waveforge6g").rglob("*.py"):
                archive.write(path, path.relative_to(ROOT))
    for index, (n, mod, level, repeat) in enumerate(itertools.product((128, 256), ("qpsk", "qam16"), ("low", "medium", "high"), range(5))):
        seed, cap = 51001+index, budget_value(n, level)
        dest = OUT/"confirmation"/str(seed)
        manifest_path = dest/"manifest.json"
        if manifest_path.exists():
            previous = json.loads(manifest_path.read_text())
            if previous.get("status") == "complete" and previous["protocol_sha256"] == sha(frozen):
                if not all(sha(dest/k) == v for k, v in previous["files"].items()):
                    raise RuntimeError("confirmation cache corrupted")
                continue
        budget.reserve(24*33*17)
        dest.mkdir(parents=True, exist_ok=True)
        write_json(manifest_path, {"status": "running", "seed": seed, "protocol_sha256": sha(frozen)})
        candidates = actions(n)
        allowed = [i for i, a in enumerate(candidates) if a.domain != "dense"]
        policies = {name: BudgetSelector(*models[n, mod],
                    [i for i in allowed if name == "joint" or candidates[i].waveform == name[3:]],
                    role_seed(seed, "r3_policy", name)) for name in ("joint", "rx_ofdm", "rx_otfs", "rx_afdm")}
        subset = development[(development.n == n) & (development.modulation == mod)]
        fixed = fixed_choice(subset, cap, 100.)
        dense = fixed_choice(subset, float("inf"), 0., "dense")
        engine = NonStationaryChannel(trajectory_config(n, repeat % 2 == 1), seed, n, n//8, 64000)
        records, channels, error_banks, choices = [], [], [], []
        start = time.perf_counter()
        for epoch in range(24):
            state, channel = engine.step(epoch)
            snr = state.snr_db
            feature_start = time.perf_counter()
            features = observable_features(channel, snr, n, mod)
            feature_s = time.perf_counter()-feature_start
            costs = np.array([link_work(SimpleNamespace(name=a.waveform, n_symbols=n, cp_length=n//8, subcarriers=16),
                                       channel, a, mod)["total_work"] for a in candidates])
            # All choices committed before any current PHY feedback bank exists.
            decisions, overheads, decision_times = {}, {}, {}
            for name, policy in policies.items():
                overheads[name] = policy.overhead(channel.n_paths)
                before = time.perf_counter()
                decisions[name] = policy.choose(features, costs+overheads[name], cap, epoch)
                decision_times[name] = time.perf_counter()-before+feature_s
            decisions["fixed"] = (fixed if fixed is not None and costs[fixed]+100 <= cap else None, "fixed")
            decisions["dense_reference"] = (dense, "unconstrained_reference")
            overheads.update(fixed=100., dense_reference=0.)
            decision_times.update(fixed=0., dense_reference=0.)
            table, raw, _ = evaluate_state(n, mod, channel, snr, seed, epoch, 8)
            validation = raw["validation_errors"].mean(axis=0)/raw["bit_count"]
            error_banks.append({k: v for k, v in raw.items()})
            channels.append(dict(epoch=epoch, snr=snr, channel=channel.to_dict(), features=features.tolist()))
            choices.append({k: v[0] for k, v in decisions.items()})
            for name, (chosen, reason) in decisions.items():
                before = time.perf_counter()
                feedback = None if chosen is None else float(raw["online_errors"][0, chosen]/raw["bit_count"])
                if name in policies:
                    policies[name].update(features, chosen, feedback)
                    decision_times[name] += time.perf_counter()-before
                row = None if chosen is None else table.iloc[chosen]
                records.append(dict(seed=seed, n=n, modulation=mod, level=level, budget=cap,
                                    epoch=epoch, policy=name, action_index=chosen,
                                    action="outage" if chosen is None else candidates[chosen].key, reason=reason,
                                    ber=np.nan if chosen is None else validation[chosen], online_ber=feedback,
                                    outage=chosen is None, dropped_bit_fraction=float(chosen is None),
                                    reserved_work=overheads[name]+(0 if row is None else row.total_work),
                                    actual_work=overheads[name]+(0 if row is None else row.actual_work_mean),
                                    cpu_s=decision_times[name]+(0 if row is None else row.total_cpu_s),
                                    policy_work=overheads[name]))
        pd.DataFrame(records).to_csv(dest/"trajectory.csv", index=False)
        np.savez_compressed(dest/"replicas.npz", **{k: np.stack([v[k] for v in error_banks]) for k in error_banks[0]})
        write_json(dest/"channels.json", channels)
        write_json(dest/"committed_choices.json", choices)
        write_json(manifest_path, {"status": "complete", "seed": seed, "n": n, "modulation": mod, "level": level,
                                  "protocol_sha256": sha(frozen), "elapsed_s": time.perf_counter()-start,
                                  "files": {p.name: sha(p) for p in dest.iterdir() if p.name != "manifest.json"}})
        budget.save()
        print(f"confirmation {index+1}/60 seed={seed} {time.perf_counter()-start:.1f}s", flush=True)
