"""Small exact-mean toys and recomputed raw-record diagnostics (no new PHY)."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from waveforge6g.decision.base import DecisionContext
from waveforge6g.decision.change_detection import PageHinkley
from waveforge6g.decision.linucb import LinUCBSelector
from waveforge6g.decision.objective import Objective
from waveforge6g.decision.regret import (
    best_fixed_action,
    hindsight_optimal_sequence,
    sequence_losses,
)
from waveforge6g.experiments.dynamic import adaptation_statistics
from waveforge6g.experiments.expected_loss import role_seed
from waveforge6g.experiments.research_v2 import DOC, OUT, ROOT, write_json
from waveforge6g.experiments.research_v2_report import completed


def toy_oracle():
    records = []
    for weight in (0., .1):
        obj = Objective({"switching_weight": weight})
        for seed in range(100):
            rng = np.random.default_rng(role_seed(seed, "toy_oracle"))
            for k in (1, 16, 64, 256):
                estimate = rng.binomial(k, .5, (120, 3))/k
                reference = hindsight_optimal_sequence(estimate, obj)
                true = np.full((120, 3), .5)
                records.append({"seed": seed, "weight": weight, "k": k,
                                "selected_cost": reference.total_loss,
                                "true_cost": sequence_losses(true, reference.actions, obj).sum(),
                                "true_optimal_cost": 60.,
                                "switches": np.count_nonzero(np.diff(reference.actions))})
    pd.DataFrame(records).to_csv(DOC / "toy_oracle.csv", index=False)


def detectors():
    rows, response_rows = [], []
    for seed in range(40):
        rng = np.random.default_rng(role_seed(seed, "detector_toy"))
        for environment in ("static_forced_switch", "fixed_action_mean_change", "variance_only"):
            detectors = {key: PageHinkley(.5, .01, 10) for key in ("raw_loss", "action_residual", "context")}
            counts = np.full(3, 20.)
            sums = np.array([.2, .6, .4])*counts  # shared declared pretraining information
            alarms = {key: [] for key in detectors}
            for t in range(200):
                action = (t//8) % 3 if environment == "static_forced_switch" else 0
                mean = [.2, .6, .4][action]
                if environment == "fixed_action_mean_change" and t >= 100:
                    mean += .35
                sd = .08 if environment == "variance_only" and t >= 100 else .02
                value = mean + rng.uniform(-np.sqrt(3), np.sqrt(3))*sd
                values = {"raw_loss": value, "action_residual": value-sums[action]/counts[action],
                          "context": .4 if environment == "fixed_action_mean_change" and t >= 100 else 0.}
                for name, detector in detectors.items():
                    if detector.update(values[name]):
                        alarms[name].append(t)
                counts[action] += 1
                sums[action] += value
            for name, times in alarms.items():
                event = 100 if environment == "fixed_action_mean_change" else None
                matches = [t for t in times if event is not None and event <= t < event+40]
                rows.append({"seed": seed, "environment": environment, "signal": name,
                             "alarms": len(times), "events": int(event is not None),
                             "false_alarms": len(times)-int(bool(matches)),
                             "misses": int(event is not None and not matches),
                             "delay": matches[0]-event if matches else None})
        # Four responses see the same causal context and paired action-noise table.
        for response in ("hard_reset", "soft_discount", "budgeted_probe", "none"):
            learner = LinUCBSelector(alpha=.1)
            detector = PageHinkley(.5, .01, 10)
            previous, total, switches, probe_until, last_alarm = None, 0., 0, -1, -30
            alarm_count = probe_count = 0
            noise = np.random.default_rng(role_seed(seed, "response_noise")).uniform(-.02, .02, (200, 3))
            for t in range(200):
                state = float(t >= 100)
                context = DecisionContext((1., state, 0., 0., 0., 0., 0., 0., 0.), previous, t)
                alarm = detector.update(state)
                if alarm and t-last_alarm >= 20:
                    last_alarm = t
                    alarm_count += 1
                    if response == "hard_reset":
                        learner.reset()
                    elif response == "soft_discount":
                        learner.gram = np.eye(9)[None] + .5*(learner.gram-np.eye(9)[None])
                        learner.targets *= .5
                    elif response == "budgeted_probe":
                        probe_until = t+12
                action = learner.select(context).action
                if t < probe_until:
                    action = ((t-last_alarm)//4) % 3
                    probe_count += 1
                mean = ([.1, .3, .4] if t < 100 else [.4, .1, .3])[action]
                loss = mean + noise[t, action]
                switch = previous is not None and action != previous
                total += loss + .1*switch
                switches += int(switch)
                learner.update(context, action, -loss)
                previous = action
            response_rows.append({"seed": seed, "response": response, "total": total,
                                  "switches": switches, "alarms": alarm_count, "probe_frames": probe_count})
    pd.DataFrame(rows).to_csv(DOC / "detector_toys.csv", index=False)
    pd.DataFrame(response_rows).to_csv(DOC / "detector_responses_toy.csv", index=False)


def old_censoring():
    manifest = json.loads((ROOT / "figures/research/manifest.json").read_text())
    parent = Path(manifest["studies"]["10_full_waveforge_x"]["result"])
    rows, seeds = [], []
    for run in json.loads((parent / "runs.json").read_text()):
        table = pd.read_csv(Path(run["directory"]) / "trajectory.csv")
        for policy, group in table.groupby("selector"):
            group = group.sort_values("time").reset_index(drop=True)
            result = adaptation_statistics(group)
            episodes = result["episodes"]
            for item in episodes:
                rows.append({"seed": run["seed"], "policy": policy, **item,
                             "censored": item["adaptation_latency"] is None})
            seeds.append({"seed": run["seed"], "policy": policy, "events": len(episodes),
                          "alarms": int(group.detected_change.sum()), "false_alarms": result["unmatched_alarms"],
                          "misses": sum(e["detection_delay"] is None for e in episodes),
                          "censor_fraction": np.mean([e["adaptation_latency"] is None for e in episodes])})
    pd.DataFrame(rows).to_csv(DOC / "old_recovery_episodes.csv", index=False)
    frame = pd.DataFrame(seeds)
    frame.to_csv(DOC / "old_recovery_counts.csv", index=False)
    values = frame.groupby("policy").censor_fraction.mean()
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.bar(values.index, values, color="#24566d")
    ax.set(ylim=(0, 1), ylabel="Censored recovery fraction")
    ax.tick_params(axis="x", labelrotation=50, labelsize=8)
    fig.tight_layout()
    destination = OUT / "report"
    destination.mkdir(exist_ok=True)
    fig.savefig(destination / "change-reset-diagnostics.png", dpi=170)
    fig.savefig(destination / "change-reset-diagnostics.pdf")
    plt.close(fig)
    endpoint_rows = []
    parent = Path(manifest["studies"]["05_observation_noise_sweep"]["result"])
    for run in json.loads((parent / "runs.json").read_text()):
        path = Path(run["directory"]) / "trajectory.csv"
        observed = pd.read_csv(path, usecols=["time", "estimated_source_time"]).drop_duplicates("time")
        lag = observed.time-observed.estimated_source_time
        endpoint_rows.append({"seed": run["seed"], "error_scale": run["variant"]["observation.error_scale"],
                              "mean_lag": lag.mean(), "max_lag": lag.max(),
                              "contemporaneous_fraction": float((lag == 0).mean()), "raw_path": str(path)})
    pd.DataFrame(endpoint_rows).to_csv(DOC / "noise_endpoint_audit.csv", index=False)


def cost_atlas():
    records, decompositions = [], []
    for directory, manifest in completed("confirm"):
        setting = manifest["settings"]
        raw = np.load(directory / "replicas.npz")
        original = Objective(setting["configuration"]["objective"])
        for weight in (0., .05, .1, .4):
            obj = Objective({**original.to_dict(), "switching_weight": weight})
            scale = original.total_weight/obj.total_weight
            loss = raw["selection_loss"][:, :16].mean(axis=1)*scale
            validation = raw["validation_loss"]*scale
            means = validation.mean(axis=1)
            dynamic, fixed = hindsight_optimal_sequence(loss, obj), best_fixed_action(loss, obj)
            t = np.arange(len(loss))
            paired = validation[t, :, fixed.actions] - validation[t, :, dynamic.actions]
            se = np.sqrt(paired.var(axis=1, ddof=1).sum()/validation.shape[1])
            difference = sequence_losses(means, fixed.actions, obj).sum()-sequence_losses(means, dynamic.actions, obj).sum()
            records.append({"family": setting["family"], "seed": setting["seed"], "weight": weight,
                            "estimated_opportunity": fixed.total_loss-dynamic.total_loss,
                            "reevaluated_difference": difference, "conditional_mc_se": se,
                            "optimal_switches": np.count_nonzero(np.diff(dynamic.actions)),
                            "estimated_mean_gap": np.mean(np.sort(loss, axis=1)[:, 1]-np.min(loss, axis=1))})
        frame = pd.read_csv(directory / "summary.csv")
        if "ber_term" in frame:
            frame["complexity_term"] = frame.base_loss-frame.ber_term-frame.bler_term-frame.papr_term
            frame = frame.assign(family=setting["family"], seed=setting["seed"])
            decompositions.append(frame)
    pd.DataFrame(records).to_csv(DOC / "cost_atlas.csv", index=False)
    if decompositions:
        pd.concat(decompositions).to_csv(DOC / "objective_decomposition.csv", index=False)


if __name__ == "__main__":
    DOC.mkdir(exist_ok=True, parents=True)
    toy_oracle()
    detectors()
    old_censoring()
    cost_atlas()
    write_json(DOC / "diagnostics_manifest.json", {"script": "scripts/research_v2_diagnostics.py",
               "toy_seeds": 40, "oracle_toy_seeds": 100,
               "scope": "exact-mean toy diagnostics and R1 recovery recomputation; no PHY efficacy claim",
               "event_matching": "first alarm in [100,140); further alarms unmatched; variance-only has no mean event"})
