"""Multi-seed dynamic sweeps, paired uncertainty summaries and saved analysis."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
import yaml
from scipy.stats import t

from ..config import ExperimentConfig
from ..config_dynamic import research_variants
from ..reproducibility import collect_metadata, configure_local_environment
from .dynamic import DynamicExperimentRunner


def confidence_summary(values: np.ndarray) -> dict:
    """t interval over independent seeds; not over correlated time samples."""
    x = np.asarray(values, dtype=float)
    if x.ndim != 1 or len(x) == 0 or not np.all(np.isfinite(x)):
        raise ValueError("seed summary requires a finite nonempty vector")
    mean = float(x.mean())
    std = float(x.std(ddof=1)) if len(x) > 1 else float("nan")
    margin = float(t.ppf(.975, len(x) - 1) * std / np.sqrt(len(x))) if len(x) > 1 else float("nan")
    return {"mean": mean, "std": std, "ci_low": mean - margin, "ci_high": mean + margin,
            "n_seeds": len(x)}


def run_research(config: ExperimentConfig, *, plot: bool = True,
                  use_cache: bool = True, force: bool = False) -> Path:
    """Run all registered selectors on common trajectories for every seed/variant."""
    root = configure_local_environment()
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    target = root / "results" / f"{stamp}_{config['experiment']['name']}_research_{uuid4().hex[:6]}"
    target.mkdir(parents=True)
    (target / "config.yaml").write_text(yaml.safe_dump(config.to_dict(), sort_keys=False), encoding="utf-8")
    metadata = collect_metadata(config.config_hash, config["experiment"]["seeds"][0])
    metadata.update(seeds=config["experiment"]["seeds"], status="running", kind="research")
    (target / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    records, trajectories, manifest = [], [], []
    try:
        for variant_id, (variant, resolved) in enumerate(research_variants(config)):
            label = json.dumps(variant, sort_keys=True)
            sweep_key = next(iter(variant)) if len(variant) == 1 else "multiple" if variant else "none"
            sweep_value = variant[sweep_key] if len(variant) == 1 else 0
            for seed in config["experiment"]["seeds"]:
                directory = DynamicExperimentRunner(resolved, seed).run(plot=False, use_cache=use_cache, force=force)
                manifest.append({"directory": str(directory), "seed": seed, "variant": variant,
                                 "variant_id": variant_id})
                summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
                table = pd.read_csv(directory / "trajectory.csv")
                table["variant"] = label
                table["variant_id"] = variant_id
                trajectories.append(table)
                for name, values in summary["selectors"].items():
                    record = {"selector": name, "seed": seed, "variant": label, "variant_id": variant_id,
                              "scenario": config["dynamic"]["scenario"], "sweep_key": sweep_key,
                              "sweep_value": sweep_value}
                    record.update({key: value for key, value in values.items() if isinstance(value, int | float)})
                    record.update({f"occupancy_{key}": value for key, value in values["occupancy"].items()})
                    episodes = values["adaptation"]["episodes"]
                    latencies = [episode["adaptation_latency"] for episode in episodes
                                 if episode["adaptation_latency"] is not None]
                    record["mean_adaptation_latency"] = float(np.mean(latencies)) if latencies else np.nan
                    record["adaptation_censored_fraction"] = (1 - len(latencies) / len(episodes)) if episodes else np.nan
                    record["unmatched_alarms_per_100_epochs"] = values["adaptation"]["unmatched_alarms_per_100_epochs"]
                    records.append(record)
        summary_frame = pd.DataFrame(records)
        summary_frame.to_csv(target / "summary.csv", index=False)
        all_trajectories = pd.concat(trajectories, ignore_index=True)
        curves = []
        for (variant, selector, time), group in all_trajectories.groupby(["variant", "selector", "time"], sort=False):
            record = {"variant": variant, "selector": selector, "time": time}
            for metric in ("cumulative_regret", "instantaneous_regret", "ber", "objective", "uncertainty",
                           "hindsight_dynamic_regret"):
                record.update({f"{metric}_{key}": value for key, value in confidence_summary(group[metric].to_numpy()).items()})
            curves.append(record)
        pd.DataFrame(curves).to_csv(target / "curves.csv", index=False)
        aggregates = []
        measures = ("dynamic_regret", "final_cumulative_regret", "mean_ber", "mean_bler", "mean_objective",
                    "total_switch_count", "average_dwell_time")
        for (variant, selector), group in summary_frame.groupby(["variant", "selector"], sort=False):
            record = {"variant": variant, "selector": selector}
            for metric in measures:
                record.update({f"{metric}_{key}": value for key, value in confidence_summary(group[metric].to_numpy()).items()})
            aggregates.append(record)
        pd.DataFrame(aggregates).to_csv(target / "aggregate.csv", index=False)
        # Paired policy contrasts exploit identical channel/noise realizations.
        contrasts = []
        for variant, group in summary_frame.groupby("variant", sort=False):
            pivot = group.pivot(index="seed", columns="selector", values="dynamic_regret")
            names = list(pivot.columns)
            reference = "always_ofdm" if "always_ofdm" in names else names[0]
            for name in names:
                differences = (pivot[name] - pivot[reference]).to_numpy()
                contrasts.append({"variant": variant, "selector": name, "reference": reference,
                                  **confidence_summary(differences)})
        pd.DataFrame(contrasts).to_csv(target / "paired_contrasts.csv", index=False)
        (target / "runs.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        metadata["status"] = "complete"
        (target / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        if plot:
            from ..plotting.research import plot_research
            plot_research(target)
    except Exception as error:
        metadata.update(status="failed", error=f"{type(error).__name__}: {error}")
        (target / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        raise
    return target
