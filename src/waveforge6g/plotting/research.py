"""Seed-level research comparisons from saved CSV summaries and curves."""

from __future__ import annotations

import html
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t

from ..reproducibility import configure_local_environment
from .ber import COLORS
from .dynamic import POLICY_COLORS, plot_dynamic, policy_label
from .publication import save_figure, set_style


def _variant_label(serialized: str) -> str:
    values = json.loads(serialized)
    if not values:
        return "Default settings"
    aliases = {"objective.switching_weight": "Switch weight", "observation.snr_std": "SNR error σ",
               "observation.doppler_relative_error": "Doppler relative error",
               "observation.error_scale": "Observation error multiplier",
               "dynamic.change_interval": "Regime interval (epochs)",
               "dynamic.change_rate": "Change rate", "dynamic.change_frequency": "Change frequency"}
    return ", ".join(f"{aliases.get(key, key.split('.')[-1].replace('_', ' '))}: {value}"
                     for key, value in values.items())


def _mean_interval(values: pd.Series) -> tuple[float, float]:
    data = values.to_numpy(dtype=float)
    data = data[np.isfinite(data)]
    if not len(data):
        return float("nan"), float("nan")
    mean = float(data.mean())
    if len(data) < 2:
        return mean, float("nan")
    radius = float(t.ppf(.975, len(data) - 1) * data.std(ddof=1) / np.sqrt(len(data)))
    return mean, radius


def _curve_figures(curves: pd.DataFrame, target: Path) -> None:
    variants = curves.variant.drop_duplicates().tolist()
    selectors = curves.selector.drop_duplicates().tolist()
    for start in range(0, len(selectors), 12):
        page = selectors[start:start + 12]
        figure, axes = plt.subplots(len(variants), 2, figsize=(12, 3.8 * len(variants)),
                                    squeeze=False, layout="constrained")
        for row, variant in enumerate(variants):
            for column, (metric, ylabel) in enumerate((
                ("cumulative_regret", "Cumulative conditional regret"),
                ("hindsight_dynamic_regret", "Loss gap to hindsight sequence"),
            )):
                ax = axes[row, column]
                for index, selector in enumerate(page):
                    group = curves.loc[(curves.variant == variant) &
                                       (curves.selector == selector)].sort_values("time")
                    if group.empty or f"{metric}_mean" not in group:
                        continue
                    color = POLICY_COLORS[index]
                    x = group.time.to_numpy()
                    ax.plot(x, group[f"{metric}_mean"], color=color, label=policy_label(selector))
                    low, high = group[f"{metric}_ci_low"], group[f"{metric}_ci_high"]
                    ax.fill_between(x, low, high, color=color, alpha=.12, linewidth=0)
                ax.set(xlabel="Decision epoch", ylabel=ylabel, title=_variant_label(variant))
                ax.grid(axis="y", alpha=.2)
                ax.legend(frameon=False, fontsize=7, ncol=2)
        suffix = "" if start == 0 else f"_part{start // 12 + 1:02}"
        save_figure(figure, target, f"figure_03_cumulative_regret{suffix}")


def _sweep_plot(data: pd.DataFrame, key: str, target: Path, filename: str,
                xlabel: str) -> None:
    selected = data.loc[data.sweep_key == key]
    if selected.empty:
        return
    names = selected.selector.drop_duplicates().tolist()
    for start in range(0, len(names), 12):
        figure, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
        for index, name in enumerate(names[start:start + 12]):
            group = selected.loc[selected.selector == name]
            for ax, metric, ylabel in zip(axes, ("dynamic_regret", "mean_ber"),
                                          ("Final dynamic regret", "Mean BER"), strict=True):
                points = [(value, *_mean_interval(sample[metric]))
                          for value, sample in group.groupby("sweep_value", sort=True)]
                x, means, radii = map(np.asarray, zip(*points, strict=True))
                ax.errorbar(x, means, yerr=radii, color=POLICY_COLORS[index], marker="o",
                            markersize=4, capsize=2, label=policy_label(name))
                ax.set(xlabel=xlabel, ylabel=ylabel)
                ax.grid(axis="y", alpha=.2)
        axes[0].legend(frameon=False, fontsize=7, ncol=2)
        suffix = "" if start == 0 else f"_part{start // 12 + 1:02}"
        save_figure(figure, target, filename + suffix)


def _switching_tradeoff(data: pd.DataFrame, target: Path) -> None:
    selected = data.loc[data.sweep_key.str.contains("switch", case=False, na=False)]
    if selected.empty:
        return
    names = selected.selector.drop_duplicates().tolist()
    all_means = selected.groupby(["selector", "sweep_value"], sort=False)[
        ["total_switch_count", "mean_ber", "dynamic_regret"]].mean()
    for start in range(0, len(names), 11):
        figure, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
        for index, name in enumerate(names[start:start + 11]):
            group = selected.loc[selected.selector == name]
            means = group.groupby("sweep_value", sort=True)[
                ["total_switch_count", "mean_ber", "dynamic_regret"]].mean()
            for ax, metric, ylabel in zip(axes, ("mean_ber", "dynamic_regret"),
                                          ("Mean BER", "Final dynamic regret"), strict=True):
                ax.plot(means.total_switch_count, means[metric], "o-", markersize=4,
                        color=POLICY_COLORS[index], label=policy_label(name))
                ax.set(xlabel="Mean switches per trajectory", ylabel=ylabel)
                ax.grid(alpha=.2)
        # A measured BER/switch frontier only: objectives at different switch
        # weights are not interchangeable, so do not pool their regret frontier.
        frontier = []
        best_ber = float("inf")
        ordered = all_means.sort_values(["total_switch_count", "mean_ber"])
        for point in ordered.itertuples():
            if point.mean_ber < best_ber:
                frontier.append((point.total_switch_count, point.mean_ber))
                best_ber = point.mean_ber
        if frontier:
            x, y = zip(*frontier, strict=True)
            axes[0].plot(x, y, ":", color="#454c50", linewidth=1.2, label="Observed frontier")
        axes[0].legend(frameon=False, fontsize=7, ncol=2)
        suffix = "" if start == 0 else f"_part{start // 11 + 1:02}"
        save_figure(figure, target, "figure_04_switching_cost_tradeoff" + suffix)


def _ablation(data: pd.DataFrame, target: Path, study_name: str) -> None:
    names = data.selector.drop_duplicates().tolist()
    if "ablation" not in study_name and not any("without" in name for name in names):
        return
    variants = data.variant.drop_duplicates().tolist()
    figure, axes = plt.subplots(len(variants), 2, figsize=(12, max(4, len(names) * .33) * len(variants)),
                                squeeze=False, layout="constrained")
    for row, variant in enumerate(variants):
        subset = data.loc[data.variant == variant]
        for column, (metric, label) in enumerate((
            ("dynamic_regret", "Final dynamic regret"), ("total_switch_count", "Switches"),
        )):
            ax = axes[row, column]
            points = [_mean_interval(subset.loc[subset.selector == name, metric]) for name in names]
            means, radii = map(np.asarray, zip(*points, strict=True))
            ax.barh(np.arange(len(names)), means, color="#42677a", height=.65,
                    xerr=radii, capsize=2)
            ax.set_yticks(np.arange(len(names)), [policy_label(name) for name in names])
            ax.invert_yaxis()
            ax.set(xlabel=label, title=_variant_label(variant))
            ax.grid(axis="x", alpha=.2)
    save_figure(figure, target, "figure_07_ablation")


def _occupancy(data: pd.DataFrame, target: Path) -> None:
    variants = data.variant.drop_duplicates().tolist()
    figure, axes = plt.subplots(len(variants), 1, figsize=(8, max(2.1, data.selector.nunique() * .38 + 1.2) * len(variants)),
                                squeeze=False, layout="constrained")
    for ax, variant in zip(axes[:, 0], variants, strict=True):
        group = data.loc[data.variant == variant].groupby("selector", sort=False)[
            ["occupancy_ofdm", "occupancy_otfs", "occupancy_afdm"]].mean()
        left = np.zeros(len(group))
        for waveform in ("ofdm", "otfs", "afdm"):
            values = group[f"occupancy_{waveform}"].to_numpy()
            ax.barh(np.arange(len(group)), values, left=left, color=COLORS[waveform],
                    height=.65, label=waveform.upper())
            left += values
        ax.set_yticks(np.arange(len(group)), [policy_label(name) for name in group.index])
        ax.invert_yaxis()
        ax.set(xlabel="Mean waveform occupancy", xlim=(0, 1), title=_variant_label(variant))
        ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(.5, -.15))
    save_figure(figure, target, "figure_08_waveform_occupancy")


def plot_research(directory: str | Path, output_dir: str | Path | None = None) -> Path:
    """Render actual multi-seed curves, sweep comparisons, ablations and occupancy."""
    root = configure_local_environment()
    set_style()
    source = Path(directory)
    target = Path(output_dir) if output_dir is not None else root / "figures" / "research" / source.name
    data = pd.read_csv(source / "summary.csv")
    curves = pd.read_csv(source / "curves.csv")
    if data.empty or curves.empty:
        raise ValueError("research summaries and curves must be nonempty")
    _curve_figures(curves, target)
    _switching_tradeoff(data, target)
    observation_keys = [key for key in data.sweep_key.unique() if "observation" in key]
    for index, key in enumerate(observation_keys):
        suffix = "" if index == 0 else f"_{index + 1:02}"
        xlabel = {"observation.snr_std": "SNR estimation σ (dB)",
                  "observation.error_scale": "Observation error multiplier"}.get(
                      key, key.split(".")[-1].replace("_", " ").capitalize())
        _sweep_plot(data, key, target, "figure_05_observation_noise" + suffix,
                    xlabel)
    change_keys = [key for key in data.sweep_key.unique() if "change" in key]
    for index, key in enumerate(change_keys):
        suffix = "" if index == 0 else f"_{index + 1:02}"
        _sweep_plot(data, key, target, "figure_06_change_rate" + suffix,
                    "Regime interval (epochs)" if key.endswith("change_interval") else
                    key.split(".")[-1].replace("_", " ").capitalize())
    _ablation(data, target, source.name)
    _occupancy(data, target)
    manifest_path = source / "runs.json"
    if manifest_path.is_file():
        runs = json.loads(manifest_path.read_text(encoding="utf-8"))
        if runs:
            # The first configured seed/variant is fixed independently of outcomes.
            representative = Path(runs[0]["directory"])
            example = plot_dynamic(representative, target / "example_trajectory")
            for suffix in ("png", "pdf"):
                shutil.copy2(example / f"dynamic_trajectory.{suffix}",
                             target / f"figure_02_dynamic_trajectory.{suffix}")
    panels = "\n".join(
        f'<figure><a href="{path.stem}.pdf"><img src="{path.name}" alt="{html.escape(path.stem)}"></a></figure>'
        for path in sorted(target.glob("*.png"))
    )
    title = source.name
    report = f'''<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)}</title><style>
body{{max-width:1160px;margin:48px auto;padding:0 24px;color:#23313a;background:#faf9f6;
font:15px/1.5 system-ui,sans-serif}}h1{{font-size:26px;font-weight:600}}img{{max-width:100%}}
figure{{margin:32px 0;background:white}}small{{color:#647179}}</style>
<h1>WaveForge-X</h1><small>{data.seed.nunique()} seeds · {data.selector.nunique()} selectors ·
95% pointwise intervals across seeds</small>{panels}</html>'''
    (target / "report.html").write_text(report, encoding="utf-8")
    return target
