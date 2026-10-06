"""Trajectory figures from saved dynamic experiments; no simulation or fitting."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from ..decision.base import ACTIONS
from ..reproducibility import configure_local_environment
from .publication import save_figure, set_style

TRUTH_COLOR = "#244a61"
ESTIMATE_COLOR = "#c37837"
POLICY_COLORS = ("#244a61", "#c37837", "#508369", "#906484", "#7796a8", "#957b50",
                 "#476e78", "#b27473", "#828d50", "#70758f", "#9d6747", "#56625e")


def policy_label(name: str) -> str:
    """Short readable legend labels without changing stored identifiers."""
    terms = {"ofdm": "OFDM", "otfs": "OTFS", "afdm": "AFDM", "ucb": "UCB",
             "linucb": "LinUCB", "ts": "TS"}
    return " ".join(terms.get(part, part.capitalize()) for part in name.split("_"))


def representative_selector(names: list[str]) -> str:
    """Prefer the configured full policy, without selecting on observed performance."""
    for candidate in ("uncertainty_aware", "full", "change_aware", "linucb"):
        for name in names:
            if name == candidate or candidate in name:
                return name
    return names[-1]


def _changes(ax: plt.Axes, group: pd.DataFrame) -> None:
    if "change_point" in group:
        times = group.loc[group.change_point.astype(str).str.lower().eq("true"), "time"]
        if len(times) > 12:
            # Frequent cluster events remain visible as a rug without obscuring the traces.
            ax.plot(times, [1.0] * len(times), "|", transform=ax.get_xaxis_transform(),
                    color="#7d8588", alpha=.65, markersize=4, clip_on=False)
            return
        for time in times:
            ax.axvline(time, color="#7d8588", alpha=.55, linewidth=.8, linestyle=":")


def _decorate(ax: plt.Axes, ylabel: str, group: pd.DataFrame, xlabel: bool = True) -> None:
    ax.set_ylabel(ylabel)
    if xlabel:
        ax.set_xlabel("Decision epoch")
    ax.grid(axis="y", alpha=.2)
    _changes(ax, group)


def _state(ax: plt.Axes, group: pd.DataFrame, truth: str, estimate: str,
           label: str, scale: float = 1.0) -> None:
    ax.plot(group.time, group[truth] * scale, color=TRUTH_COLOR, label="True")
    if estimate in group:
        ax.plot(group.time, group[estimate] * scale, color=ESTIMATE_COLOR,
                linewidth=1, alpha=.8, label="Observed")
    _decorate(ax, label, group)


def _timeline(ax: plt.Axes, group: pd.DataFrame) -> None:
    ax.step(group.time, group.action, where="post", color=TRUTH_COLOR, label="Selected")
    ax.step(group.time, group.oracle_action, where="post", color=ESTIMATE_COLOR,
            linestyle="--", alpha=.85, label="Conditional oracle")
    ax.set_yticks(range(3), [name.upper() for name in ACTIONS])
    ax.set_ylim(-.3, 2.3)
    _decorate(ax, "Waveform", group)


def plot_dynamic(directory: str | Path, output_dir: str | Path | None = None) -> Path:
    """Render measured channel, decisions, uncertainty and comparator trajectories."""
    configure_local_environment()
    set_style()
    source = Path(directory)
    target = Path(output_dir) if output_dir is not None else source / "figures"
    data = pd.read_csv(source / "trajectory.csv")
    summary = json.loads((source / "summary.json").read_text(encoding="utf-8"))
    if data.empty:
        raise ValueError("trajectory.csv contains no dynamic observations")
    names = data.selector.drop_duplicates().tolist()
    selected_name = representative_selector(names)
    group = data.loc[data.selector == selected_name].sort_values("time")

    figure, axes = plt.subplots(3, 1, figsize=(7, 6), sharex=True, layout="constrained")
    for ax, arguments in zip(axes, (
        ("true_snr_db", "estimated_snr_db", r"$E_s/N_0$ (dB)", 1),
        ("true_max_doppler_hz", "estimated_doppler_hz", "Doppler (Hz)", 1),
        ("true_delay_spread_s", "estimated_delay_spread_s", "RMS delay (µs)", 1e6),
    ), strict=True):
        _state(ax, group, *arguments)
    axes[0].legend(frameon=False, ncol=2)
    save_figure(figure, target, "channel_state_over_time")

    figure, ax = plt.subplots(figsize=(7, 2.8), layout="constrained")
    _timeline(ax, group)
    ax.set_title(policy_label(selected_name))
    ax.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(.5, -.2))
    save_figure(figure, target, "selected_vs_oracle_waveform")

    for column, label, filename in (
        ("uncertainty", "Uncertainty score", "uncertainty_over_time"),
        ("instantaneous_regret", "Conditional regret", "instantaneous_regret"),
        ("ber", "BER", "ber_over_time"),
    ):
        figure, ax = plt.subplots(figsize=(7, 3), layout="constrained")
        ax.plot(group.time, group[column], color=TRUTH_COLOR)
        ax.set_title(policy_label(selected_name))
        _decorate(ax, label, group)
        save_figure(figure, target, filename)

    pages = [names[start:start + 12] for start in range(0, len(names), 12)]
    figure, axes = plt.subplots(len(pages), 1, figsize=(7.5, 4.2 * len(pages)),
                                squeeze=False, layout="constrained")
    for ax, page in zip(axes[:, 0], pages, strict=True):
        for index, name in enumerate(page):
            series = data.loc[data.selector == name].sort_values("time")
            ax.plot(series.time, series.cumulative_regret, color=POLICY_COLORS[index],
                    label=policy_label(name))
        _decorate(ax, "Cumulative conditional regret", group)
        ax.legend(frameon=False, fontsize=7, ncol=2)
    save_figure(figure, target, "cumulative_regret")

    figure, axes = plt.subplots(4, 2, figsize=(11, 10), layout="constrained")
    _state(axes[0, 0], group, "true_snr_db", "estimated_snr_db", r"$E_s/N_0$ (dB)")
    _state(axes[0, 1], group, "true_max_doppler_hz", "estimated_doppler_hz", "Doppler (Hz)")
    _state(axes[1, 0], group, "true_delay_spread_s", "estimated_delay_spread_s", "RMS delay (µs)", 1e6)
    _timeline(axes[1, 1], group)
    axes[0, 0].legend(frameon=False, ncol=2, fontsize=8)
    axes[1, 1].legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(.5, -.2))
    for ax, column, label in (
        (axes[2, 0], "uncertainty", "Uncertainty score"),
        (axes[2, 1], "instantaneous_regret", "Conditional regret"),
        (axes[3, 0], "cumulative_regret", "Cumulative conditional regret"),
        (axes[3, 1], "ber", "BER"),
    ):
        ax.plot(group.time, group[column], color=TRUTH_COLOR)
        _decorate(ax, label, group)
    figure.suptitle(f"{summary.get('scenario', 'Dynamic trajectory').replace('_', ' ').title()} · "
                   f"{policy_label(selected_name)} · seed {summary.get('seed', '')}", fontsize=12)
    save_figure(figure, target, "dynamic_trajectory")

    fields = ("total_switch_count", "switches_per_100_frames", "average_dwell_time",
              "minimum_dwell_time", "occupancy")
    statistics = {name: {field: values[field] for field in fields if field in values}
                  for name, values in summary["selectors"].items()}
    (target / "waveform_switching_statistics.json").write_text(
        json.dumps(statistics, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return target
