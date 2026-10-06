"""Auditable bounded R2 studies using the established channel and policy interfaces."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
import traceback
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ..channels.nonstationary import NonStationaryChannel
from ..channels.observation import ObservationModel
from ..config_dynamic import parse_dynamic_config
from ..decision import create_selector
from ..decision.advantage_dwell import AdvantageDwell, BlockLinUCB
from ..decision.objective import Objective
from ..decision.regret import best_fixed_action, hindsight_optimal_sequence, sequence_losses
from ..reproducibility import collect_metadata
from .dynamic import make_context
from .expected_loss import PreparedPHY, role_seed
from .research_cache import implementation_hash

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results/research_v2"
DOC = ROOT / "docs/research_v2"
CFG = ROOT / "configs/research_v2"
FAMILIES = {
    "near_tie": {"scenario": "stable_low_mobility", "max_doppler_hz": 0,
                 "delay_samples": 0, "path_count": 1, "correlation": 1, "snr_db": 12},
    "fixed_region": {"scenario": "stable_low_mobility", "max_doppler_hz": 15,
                     "delay_samples": 6, "correlation": 0.995, "snr_db": 12},
    "slow_change": {"scenario": "abrupt_change", "max_doppler_hz": 2400,
                    "delay_samples": 6, "change_interval": 60, "snr_db": 12},
    "fast_change": {"scenario": "mixed_extreme_mobility", "max_doppler_hz": 2400,
                    "delay_samples": 6, "change_interval": 15, "snr_db": 12},
}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    temporary.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def configuration(family, epochs=120, noise=1.0, delay=1):
    raw = yaml.safe_load((ROOT / "configs/research/10_full_waveforge_x.yaml").read_text())
    raw["dynamic"] = {**FAMILIES[family], "epochs": epochs}
    raw["selectors"] = [s for s in raw["selectors"] if s["type"] != "offline"]
    raw["observation"].update(perfect=False, snr_std=noise,
                               doppler_relative_error=0.12 * noise,
                               delay_spread_relative_error=0.12 * noise, delay_frames=delay)
    raw["experiment"]["split"] = "test"
    return parse_dynamic_config(raw)


def policies(config, seed, parameters, ablations=False):
    obj = Objective(config["objective"])
    specs = [s for s in config["selectors"] if s["name"] in
             ("always_ofdm", "always_otfs", "always_afdm", "linucb", "uncertainty_aware")]
    specs.append({"name": "sticky", "type": "linucb", "alpha": 0.4, "policy": "sticky",
                  "base_threshold": 0.01, "max_hold": parameters.get("sticky_hold", 15)})
    specs.append({"name": "offline_historical", "type": "offline",
                  "model": str(ROOT / "data/models/offline_selector.json")})
    result = {s["name"]: create_selector(s, seed=role_seed(seed, "policy", s["name"]), objective=obj)
              for s in specs}
    result["block"] = BlockLinUCB(parameters.get("block", 8))
    result["cadc"] = AdvantageDwell(obj, role_seed(seed, "policy", "cadc"),
                                    horizon=parameters.get("horizon", 16),
                                    multiplier=parameters.get("multiplier", 2.0))
    if ablations:
        result["cadc_no_margin"] = AdvantageDwell(obj, role_seed(seed, "policy", "cadc"), multiplier=0)
        result["cadc_no_probe"] = AdvantageDwell(obj, role_seed(seed, "policy", "cadc"),
                                                probe=0, multiplier=parameters.get("multiplier", 2))
        result["cadc_drift_margin"] = AdvantageDwell(obj, role_seed(seed, "policy", "cadc"),
                                                    rho=0.001, multiplier=parameters.get("multiplier", 2))
    return result


def simulate(config, seed, parameters, selection_k=16, validation_k=16, evaluator_seed=None,
             csi_nmse=0.0, ablations=False, reverse_policies=False):
    s = config["system"]
    engine = NonStationaryChannel(config["dynamic"], role_seed(seed, "channel"), s["frame_size"],
                                  s["cp_length"], s["sample_rate_hz"])
    observer = ObservationModel(config["observation"], role_seed(seed, "observation"))
    deployed = policies(config, seed, parameters, ablations)
    if reverse_policies:
        deployed = dict(reversed(list(deployed.items())))
    previous, recent = dict.fromkeys(deployed), dict.fromkeys(deployed, 0.0)
    rows, channels, records = [], [], {key: [] for key in ("selection_loss", "validation_loss",
        "online_loss", "selection_errors", "validation_errors", "online_errors", "online_papr", "energy")}
    objective = Objective(config["objective"])
    evaluation_seed = seed if evaluator_seed is None else evaluator_seed
    for t in range(config["dynamic"]["epochs"]):
        state, channel = engine.step(t)
        obs = observer.observe(state)
        contexts, decisions, policy_times = {}, {}, {}
        for name, policy in deployed.items():
            start = time.perf_counter()
            contexts[name] = make_context(obs, previous[name], recent[name], s)
            policy.observe(contexts[name])
            decisions[name] = policy.select(contexts[name])
            policy_times[name] = time.perf_counter() - start
        phy = PreparedPHY(config, channel, state.snr_db, csi_nmse, role_seed(seed, "csi", t))
        online = phy.evaluate(seed, t, "online", 1)
        # Commit and selected feedback before constructing any evaluator replicas.
        for name, decision in decisions.items():
            a = decision.action
            start = time.perf_counter()
            deployed[name].update(contexts[name], a, -float(online["loss"][0, a]))
            policy_times[name] += time.perf_counter() - start
            rows.append({**decision.diagnostics, "time": t, "policy": name, "action": a,
                         "policy_runtime_s": policy_times[name], "papr_db": online["papr"][0, a],
                         "proposed_action": decision.diagnostics.get("proposed_action", decision.diagnostics.get("candidate_action", a)),
                         "executed_action": a, "selected_feedback": -float(online["loss"][0, a]),
                         "reset_type": "hard" if decision.diagnostics.get("detected_change", False) else "none",
                         "base_loss": online["loss"][0, a], "ber": online["errors"][0, a] / online["bits"],
                         "bler": int(online["errors"][0, a] > 0),
                         "switch_cost": objective.switch_cost(previous[name], a),
                         "source_time": observer.last_source_time,
                         "context": json.dumps(contexts[name].features),
                         "diagnostics": json.dumps(decision.diagnostics)})
            previous[name], recent[name] = a, online["errors"][0, a] / online["bits"]
        selection = phy.evaluate(evaluation_seed, t, "oracle_selection", selection_k)
        validation = phy.evaluate(evaluation_seed, t, "oracle_validation", validation_k)
        for prefix, measured in (("online", online), ("selection", selection), ("validation", validation)):
            records[prefix + "_loss"].append(measured["loss"])
            records[prefix + "_errors"].append(measured["errors"])
        records["energy"].append(validation["energy"].mean(axis=0))
        records["online_papr"].append(online["papr"])
        channels.append({"time": t, "channel": channel.to_dict(), "state": state.to_dict(),
                         "csi_nmse": phy.nmse})
    records = {key: np.asarray(value) for key, value in records.items()}
    records["bit_count"] = np.array(online["bits"])
    return pd.DataFrame(rows), records, channels


def summarize(table, data, objective):
    select = data["selection_loss"].mean(axis=1)
    valid = data["validation_loss"].mean(axis=1)
    realized = data["online_loss"][:, 0]
    rows, references = [], []
    for k in (1, 16, 64, 256):
        if k > data["selection_loss"].shape[1]:
            continue
        estimate = data["selection_loss"][:, :k].mean(axis=1)
        dynamic = hindsight_optimal_sequence(estimate, objective)
        fixed = best_fixed_action(estimate, objective)
        cross_dynamic = sequence_losses(valid, dynamic.actions, objective).sum()
        cross_fixed = sequence_losses(valid, fixed.actions, objective).sum()
        references.append({"k": k, "estimated_dynamic": dynamic.total_loss,
                           "reevaluated_dynamic": cross_dynamic,
                           "optimism": cross_dynamic - dynamic.total_loss,
                           "estimated_opportunity": fixed.total_loss - dynamic.total_loss,
                           "reevaluated_opportunity": cross_fixed - cross_dynamic,
                           "fixed_action": int(fixed.actions[0]),
                           "dynamic_switches": int(np.count_nonzero(np.diff(dynamic.actions))),
                           "near_tie_fraction": float(np.mean(np.ptp(estimate, axis=1) < 0.002))})
    dp = hindsight_optimal_sequence(select, objective)
    realized_dp = hindsight_optimal_sequence(realized, objective)
    expected_reference = sequence_losses(valid, dp.actions, objective).sum()
    for name, group in table.groupby("policy", sort=True):
        group = group.sort_values("time")
        path = group.action.to_numpy()
        expected = sequence_losses(valid, path, objective).sum()
        actual = (group.base_loss + group.switch_cost).sum()
        rows.append({"policy": name, "expected_objective": expected / len(path),
                     "realized_objective": actual / len(path), "ber": group.ber.mean(),
                     "bler": group.bler.mean(), "switches": np.count_nonzero(np.diff(path)),
                     "switch_cost": group.switch_cost.sum(), "base_loss": group.base_loss.sum(),
                     "policy_runtime_s": group.policy_runtime_s.sum(),
                     "ber_term": objective.weights["ber"] * group.ber.sum() / objective.total_weight,
                     "bler_term": objective.weights["bler"] * group.bler.sum() / objective.total_weight,
                     "papr_term": objective.weights["papr"] * np.clip(group.papr_db / objective.papr_reference_db, 0, 1).sum() / objective.total_weight,
                     "realized_regret": actual - realized_dp.total_loss,
                     "approximate_reference_gap": expected - expected_reference,
                     "same_table_expected_regret": sequence_losses(select, path, objective).sum() - dp.total_loss})
    return pd.DataFrame(rows), pd.DataFrame(references)


class Budget:
    def __init__(self, seconds=14400, evaluations=2000000, bytes_limit=5 * 1024**3, output_root=None):
        self.started = time.perf_counter()
        self.seconds, self.limit, self.bytes_limit = seconds, evaluations, bytes_limit
        self.root = OUT if output_root is None else Path(output_root)
        self.path = self.root / "budget.json"
        old = json.loads(self.path.read_text()) if self.path.exists() else {}
        self.evaluations = old.get("phy_evaluations", 0)
        self.prior_seconds = old.get("wall_seconds", 0.0)

    def reserve(self, count):
        elapsed = self.prior_seconds + time.perf_counter() - self.started
        size = sum(p.stat().st_size for p in self.root.rglob("*") if p.is_file())
        if self.evaluations + count > self.limit or elapsed > self.seconds or size > self.bytes_limit:
            raise RuntimeError("R2 cumulative compute/data budget reached")
        self.evaluations += count
        write_json(self.path, {"phy_evaluations": self.evaluations, "wall_seconds": elapsed,
                              "reserved_before_execution": True, "data_bytes": size})

    def save(self):
        write_json(self.path, {"phy_evaluations": self.evaluations,
                              "wall_seconds": self.prior_seconds + time.perf_counter() - self.started})


def run_one(stage, family, seed, parameters, budget, epochs=120, k=16, kv=None,
            noise=1.0, delay=1, csi_nmse=0.0, ablations=False):
    kv = k if kv is None else kv
    config = configuration(family, epochs, noise, delay)
    settings = {"family": family, "seed": seed, "epochs": epochs, "selection_k": k,
                "validation_k": kv, "noise": noise, "delay": delay, "csi_nmse": csi_nmse,
                "parameters": parameters, "ablations": ablations,
                "configuration": config.to_dict(), "source_hash": implementation_hash(),
                "protocol_sha256": sha(CFG / "frozen.json") if stage in ("confirm", "extensions") else None,
                "offline_model_sha256": sha(ROOT / "data/models/offline_selector.json"),
                "feedback": "simulator-assisted selected-action feedback", "rng_version": "sha256-role-v1"}
    digest = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()
    directory = OUT / stage / f"{family}_{seed}_T{epochs}_{digest[:10]}"
    manifest_path = directory / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("status") == "complete" and all(sha(directory / p) == h for p, h in manifest["files"].items()):
            print(f"cached {directory.name}", flush=True)
            return directory
    budget.reserve(epochs * (1 + k + kv) * 3)
    started = time.perf_counter()
    directory.mkdir(parents=True, exist_ok=True)
    snapshot = OUT / "sources" / (settings["source_hash"] + ".zip")
    if not snapshot.exists():
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(snapshot, "w", zipfile.ZIP_DEFLATED) as archive:
            for path in (ROOT / "src/waveforge6g").rglob("*.py"):
                archive.write(path, path.relative_to(ROOT))
    manifest = {"status": "running", "settings": settings, "key": digest,
                "metadata": collect_metadata(config.config_hash, seed)}
    write_json(manifest_path, manifest)
    write_json(OUT / "STATUS.json", {"stage": stage, "run": str(directory), "status": "running",
                                    "resume": f"python scripts/research_v2.py {stage}"})
    try:
        table, data, channels = simulate(config, seed, parameters, k, kv,
                                         csi_nmse=csi_nmse, ablations=ablations)
        table.to_csv(directory / "trajectory.csv", index=False)
        np.savez_compressed(directory / "replicas.npz", **data)
        write_json(directory / "channels.json", channels)
        summary, references = summarize(table, data, Objective(config["objective"]))
        summary.to_csv(directory / "summary.csv", index=False)
        references.to_csv(directory / "references.csv", index=False)
        # Conditional covariance is paired across waveform actions, with frame replicas as units.
        means = data["validation_loss"].mean(axis=1)
        covariance = np.asarray([np.cov(x, rowvar=False, ddof=1) for x in data["validation_loss"]])
        np.savez_compressed(directory / "moments.npz", mean=means, covariance=covariance,
                            frame_count=kv, error_count=data["validation_errors"].sum(axis=1),
                            bit_count=kv * data["bit_count"])
        manifest.update(status="complete", elapsed_s=time.perf_counter() - started,
                        phy_evaluations=epochs * (1 + k + kv) * 3,
                        files={p.name: sha(p) for p in directory.iterdir() if p.name != "manifest.json"})
        write_json(manifest_path, manifest)
        budget.save()
        print(f"complete {directory.name}: {manifest['elapsed_s']:.2f}s", flush=True)
        return directory
    except BaseException:
        manifest.update(status="failed", error=traceback.format_exc())
        write_json(manifest_path, manifest)
        write_json(OUT / "STATUS.json", manifest)
        budget.save()
        raise


def audit():
    manifest = json.loads((ROOT / "figures/research/manifest.json").read_text())
    records, inputs = [], []
    for study in ("10_full_waveforge_x", "09_uncertainty_ablation"):
        parent = Path(manifest["studies"][study]["result"])
        for run in json.loads((parent / "runs.json").read_text()):
            directory = Path(run["directory"])
            table = pd.read_csv(directory / "trajectory.csv")
            raw = pd.read_csv(directory / "oracle_outcomes.csv")
            config = yaml.safe_load((directory / "config.yaml").read_text())
            objective = Objective(config["objective"])
            loss = raw.pivot(index="time", columns="action", values="base_loss").to_numpy()
            dp = hindsight_optimal_sequence(loss, objective)
            for policy, group in table.groupby("selector"):
                group = group.sort_values("time")
                recalculated = sequence_losses(loss, group.action.to_numpy(), objective)
                if not np.allclose(recalculated, group.objective, atol=1e-12):
                    raise AssertionError("old raw objective mismatch")
                records.append({"study": study, "seed": run["seed"], "policy": policy,
                                "dynamic_regret": recalculated.sum() - dp.total_loss,
                                "base_loss": group.base_loss.sum(), "switch_cost": group.switching_cost.sum(),
                                "ber": group.ber.mean(), "switches": np.count_nonzero(np.diff(group.action))})
            inputs.append({"path": str(directory / "trajectory.csv"), "sha256": sha(directory / "trajectory.csv")})
    table = pd.DataFrame(records)
    table.to_csv(DOC / "baseline_recomputed.csv", index=False)
    write_json(DOC / "baseline_inputs.json", inputs)
    means = table.groupby(["study", "policy"]).mean(numeric_only=True)
    text = "# R1 原始记录复算\n\n由 `python scripts/research_v2.py audit` 从每条 trajectory 和 oracle_outcomes 重算；逐行 objective 与原值一致（1e-12）。DP 复用现有实现。旧种子全部作为开发证据。\n\n"
    text += "```csv\n" + means.to_csv(lineterminator="\n") + "```\n\n源码核查：每 epoch 40/64000=0.000625 秒，T120 为75毫秒；不是120秒。帧内 Doppler 固定，跨帧路径相位积分推进；这是分段恒频离散模型。反馈使用真实发送比特，仅称 simulator-assisted selected-action feedback。功率为有用符号的集合平均 Es=1；总帧平均能量40，有限帧前缀能量不同，无逐帧归一化。OFDM N=32；OTFS 8×4，不能混称同一子载波间隔。相同全维 LMMSE、不使用单抽头弱化 OFDM。\n\n旧 observation.error_scale=0 同时设置 perfect=True；新研究独立设置 noise 和 delay，保留旧结果。旧置信度是 proposal 的算法诊断，不能当执行动作正确率。旧 DP 已含成本和未来，不是缺失功能。319测试是上一轮记录，本轮将重新运行。约93%删失需保留事件定义，不解释成已确认失败。\n"
    (DOC / "baseline_audit.md").write_text(text, encoding="utf-8", newline="\n")
    write_json(DOC / "environment.json", {"git_status": subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True),
               "metadata": collect_metadata("audit", 0), "memory_note": "24 GiB installed; approximately 6.4 GiB free at start; serial execution"})
    print(means[["dynamic_regret", "switches"]].to_string())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["audit", "pilot", "calibrate", "confirm", "extensions", "report"])
    parser.add_argument("--max-seconds", type=float, default=14400)
    parser.add_argument("--max-evaluations", type=int, default=2000000)
    parser.add_argument("--max-bytes", type=int, default=5*1024**3)
    args = parser.parse_args()
    for path in (OUT, DOC, CFG):
        path.mkdir(parents=True, exist_ok=True)
    if args.stage == "audit":
        audit()
        return
    if args.stage == "report":
        from .research_v2_report import report
        report()
        return
    budget = Budget(args.max_seconds, args.max_evaluations, args.max_bytes)
    defaults = {"horizon": 16, "block": 8, "sticky_hold": 15, "multiplier": 2.0}
    if args.stage == "pilot":
        for index, family in enumerate(FAMILIES):
            run_one("pilot", family, 3001 + index, defaults, budget, epochs=120, k=16)
    elif args.stage == "calibrate":
        from .research_v2_report import calibrate
        for index, family in enumerate(FAMILIES):
            run_one("calibrate", family, 3101 + index, defaults, budget, epochs=120, k=16)
        calibrate()
    else:
        protocol = json.loads((CFG / "frozen.json").read_text())
        parameters = protocol["parameters"]
        if args.stage == "confirm":
            for index, family in enumerate(FAMILIES):
                for j in range(20):
                    run_one("confirm", family, 10001 + index * 100 + j, parameters, budget,
                            k=64 if j == 0 else 16)
        else:
            for family, epochs, seed in [("slow_change", 600, 20001), ("fast_change", 600, 20002),
                                         ("slow_change", 2000, 20003)]:
                run_one("extensions", family, seed, parameters, budget, epochs=epochs, k=8, ablations=True)
            for noise, delay in [(0, 0), (0, 1), (0, 4), (1, 0), (1, 1), (1, 4), (2, 1)]:
                run_one("extensions", "fast_change", 21001, parameters, budget, noise=noise, delay=delay, ablations=True)
            for nmse in (0, 0.01, 0.1):
                run_one("extensions", "fast_change", 21002, parameters, budget, csi_nmse=nmse)
            run_one("extensions", "slow_change", 22001, parameters, budget, epochs=12, k=256)
    budget.save()
    write_json(OUT / "STATUS.json", {"status": "stage_complete", "stage": args.stage})


if __name__ == "__main__":
    main()
