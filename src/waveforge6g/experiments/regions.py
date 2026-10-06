"""Simulation-derived waveform regions, explicit rules and learned maps."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ..config import load_config
from ..decision.base import ACTIONS, DecisionContext
from ..decision.objective import Objective
from ..reproducibility import configure_local_environment
from .link import operation_proxy
from .runner import run_experiment


def waveform_regions(config_path: Path, output_dir: Path, *, offline_model: Path | None = None,
                      occupancy_run: Path | None = None, results_dir: Path | None = None) -> Path:
    """Measure finite-grid oracle regions without encoding preferred waveforms.

    Oracle uses expected empirical base loss, no predecessor/switch term.
    The displayed classifier slice fixes delay and variation; online occupancy
    is a descriptive projection of a different, varying-delay trajectory.
    """
    configure_local_environment()
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    from ..decision import create_selector
    from ..plotting.ber import COLORS
    from ..plotting.publication import save_figure, set_style

    config = load_config(config_path)
    source = results_dir or run_experiment(config, plot=False)
    measured = pd.read_csv(source / "metrics.csv")
    objective = Objective({"complexity_reference": 1000000})
    system, channel = config["system"], config["channel"]
    if measured.detector.nunique() != 1:
        raise ValueError("region comparison needs one common detector")
    n, k, fs, cp = (system[key] for key in ("frame_size", "subcarriers", "sample_rate_hz", "cp_length"))
    paths = channel["paths"]
    if not paths:
        raise ValueError("region configuration must explicitly define fixed path delays/powers")
    delays = np.asarray([path["delay_samples"] for path in paths]) / fs
    power = 10 ** (np.asarray([path.get("power_db", 0) for path in paths]) / 10)
    power /= power.sum()
    rms = float(np.sqrt(np.sum(power * delays ** 2) - np.sum(power * delays) ** 2))
    snrs = np.sort(measured.snr_db.unique())
    dopplers = np.sort(measured.max_doppler_hz.unique())
    if len(snrs) < 2 or len(dopplers) < 2:
        raise ValueError("region map needs at least two SNR and Doppler values")
    panels = {"Measured oracle": np.full((len(dopplers), len(snrs)), np.nan),
              "Explicit rule": np.full((len(dopplers), len(snrs)), np.nan)}
    rule = create_selector({"type": "rule", "name": "map_rule"}, seed=0, objective=objective,
                           context_dimension=9)
    offline = None
    if offline_model:
        offline = create_selector({"type": "offline", "model": str(offline_model), "name": "map_offline"},
                                  seed=0, objective=objective, context_dimension=9)
        panels["Offline classifier"] = np.full_like(panels["Measured oracle"], np.nan)
    records = []
    for iy, fd in enumerate(dopplers):
        for ix, snr in enumerate(snrs):
            group = measured[(measured.snr_db == snr) & np.isclose(measured.max_doppler_hz, fd)]
            if len(group) != 3 or set(group.waveform) != set(ACTIONS):
                raise ValueError("every map cell requires all three waveform measurements")
            losses = []
            for action in ACTIONS:
                row = group[group.waveform == action].iloc[0]
                proxy = operation_proxy(action, n, k, len(paths), row.solver)
                losses.append(objective.base_loss({"ber": row.ber, "bler": row.bler,
                                                    "papr_db": row.papr_mean_db, "complexity_proxy": proxy}))
            oracle = int(np.argmin(losses))
            panels["Measured oracle"][iy, ix] = oracle
            context = DecisionContext((1., float(snr / 30), float(fd * n / fs), rms * fs / max(cp, 1),
                                       0., 0., 0., 0., 0.), None, 0)
            rule_action = rule.select(context).action
            panels["Explicit rule"][iy, ix] = rule_action
            record = {"snr_db": snr, "max_doppler_hz": fd, "delay_spread_s": rms,
                      "oracle": ACTIONS[oracle], "rule": ACTIONS[rule_action],
                      **{f"loss_{action}": loss for action, loss in zip(ACTIONS, losses, strict=True)}}
            if offline:
                predicted = offline.select(context).action
                panels["Offline classifier"][iy, ix] = predicted
                record["offline"] = ACTIONS[predicted]
            records.append(record)
    occupancy_records = []
    if occupancy_run:
        manifests = json.loads((occupancy_run / "runs.json").read_text(encoding="utf-8"))
        frames = [pd.read_csv(Path(entry["directory"]) / "trajectory.csv") for entry in manifests]
        samples = pd.concat(frames, ignore_index=True)
        names = list(samples.selector.unique())
        selected = "uncertainty_aware" if "uncertainty_aware" in names else names[-1]
        samples = samples[samples.selector == selected]
        panels["Online occupancy · varying delay"] = np.full_like(panels["Measured oracle"], np.nan)
        bins: dict[tuple[int, int], list[int]] = {}
        for row in samples.itertuples():
            snr, fd = row.estimated_snr_db, row.estimated_doppler_hz
            if not snrs.min() <= snr <= snrs.max() or not dopplers.min() <= fd <= dopplers.max():
                continue
            index = (int(np.argmin(np.abs(dopplers - fd))), int(np.argmin(np.abs(snrs - snr))))
            bins.setdefault(index, []).append(row.action)
        for (iy, ix), actions in bins.items():
            counts = np.bincount(actions, minlength=3)
            panels["Online occupancy · varying delay"][iy, ix] = int(np.argmax(counts))
            occupancy_records.append({"snr_db": snrs[ix], "max_doppler_hz": dopplers[iy],
                                      "selector": selected, "n_epochs": len(actions),
                                      **{f"occupancy_{action}": counts[i] / len(actions)
                                         for i, action in enumerate(ACTIONS)}})
    set_style()
    cmap = ListedColormap([COLORS[action] for action in ACTIONS])
    cmap.set_bad("#e8e8e5")
    output_dir.mkdir(parents=True, exist_ok=True)
    for title, values in panels.items():
        fig, ax = plt.subplots(figsize=(5.5, 3.8), layout="constrained")
        mesh = ax.pcolormesh(snrs, dopplers, values, shading="nearest", cmap=cmap, vmin=-.5, vmax=2.5)
        ax.set(xlabel=r"$E_s/N_0$ (dB)", ylabel="Maximum Doppler (Hz)", title=title,
               xlim=(snrs.min(), snrs.max()), ylim=(dopplers.min(), dopplers.max()))
        bar = fig.colorbar(mesh, ax=ax, ticks=[0, 1, 2])
        bar.ax.set_yticklabels([action.upper() for action in ACTIONS])
        name = {"Measured oracle": "oracle_selection_map", "Explicit rule": "rule_selection_map",
                "Offline classifier": "offline_selection_map",
                "Online occupancy · varying delay": "online_occupancy_map"}[title]
        save_figure(fig, output_dir, name)
    cols = min(2, len(panels))
    fig, axes = plt.subplots(int(np.ceil(len(panels) / cols)), cols,
                              figsize=(5 * cols, 3.5 * np.ceil(len(panels) / cols)),
                              squeeze=False, layout="constrained")
    for ax, (title, values) in zip(axes.flat, panels.items(), strict=False):
        mesh = ax.pcolormesh(snrs, dopplers, values, shading="nearest", cmap=cmap, vmin=-.5, vmax=2.5)
        ax.set(xlabel=r"$E_s/N_0$ (dB)", ylabel="Maximum Doppler (Hz)", title=title,
               xlim=(snrs.min(), snrs.max()), ylim=(dopplers.min(), dopplers.max()))
    for ax in list(axes.flat)[len(panels):]:
        ax.set_visible(False)
    bar = fig.colorbar(mesh, ax=list(axes.flat), ticks=[0, 1, 2], shrink=.75)
    bar.ax.set_yticklabels([action.upper() for action in ACTIONS])
    save_figure(fig, output_dir, "figure_01_waveform_regions")
    pd.DataFrame(records).to_csv(output_dir / "region_measurements.csv", index=False)
    if occupancy_records:
        pd.DataFrame(occupancy_records).to_csv(output_dir / "occupancy_projection.csv", index=False)
    manifest = {"source": str(source), "configuration_hash": config.config_hash,
                "objective": objective.to_dict(), "fixed_classifier_rms_delay_s": rms,
                "note": "Finite-sample oracle per measured grid cell. Unvisited occupancy cells remain blank. "
                        "No statistical superiority or interpolated boundary is claimed."}
    (output_dir / "region_metadata.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (output_dir / "region_config.yaml").write_text(yaml.safe_dump(config.to_dict()), encoding="utf-8")
    return output_dir
