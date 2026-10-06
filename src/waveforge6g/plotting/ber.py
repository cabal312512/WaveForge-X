"""Measured BER curves with explicit treatment of zero observed errors."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLORS = {"ofdm": "#24566f", "otfs": "#c07937", "afdm": "#467462"}
MARKERS = {"ofdm": "o", "otfs": "s", "afdm": "D"}


def plot_ber(data: pd.DataFrame, x: str = "snr_db") -> plt.Figure:
    """One fixed-condition slice; zero-error points show an IID Wilson upper bound."""
    fig, ax = plt.subplots(figsize=(5.6, 3.8), layout="constrained")
    visible = np.where(data.ber.to_numpy() > 0, data.ber, data.ber_iid_ci_high)
    floor = max(float(np.nanmin(visible)) / 5, 1e-8)
    zeros = False
    for (waveform, detector), group in data.groupby(["waveform", "detector"], sort=False):
        group = group.sort_values(x)
        positive = group.ber.to_numpy() > 0
        values = np.where(positive, group.ber, np.nan)
        color = COLORS[waveform]
        label = f"{waveform.upper()} · {detector.upper()}"
        ax.semilogy(group[x], values, marker=MARKERS[waveform], color=color,
                    linestyle="--" if detector == "zf" else "-", label=label)
        if not np.all(positive):
            zeros = True
            ax.semilogy(group.loc[~positive, x], group.loc[~positive, "ber_iid_ci_high"],
                        "v", color=color, markerfacecolor="none")
        low, high = group.ber_frame_ci_low.to_numpy(), group.ber_frame_ci_high.to_numpy()
        valid = positive & np.isfinite(low) & np.isfinite(high)
        ax.fill_between(group[x], np.maximum(low, floor), high, where=valid,
                        color=color, alpha=.10)
    ax.set_xlabel({"snr_db": r"$E_s/N_0$ (dB)", "max_doppler_hz": "Maximum Doppler (Hz)",
                   "frame_size": "Useful symbols per frame"}[x])
    ax.set_ylabel("Bit error rate")
    ax.set_ylim(floor, min(1.05, max(.1, float(np.nanmax(visible)) * 3)))
    ax.grid(True, which="major", alpha=.22)
    ax.legend(frameon=False, fontsize=8)
    if zeros:
        ax.text(.02, .02, "▽ Zero errors: IID 95% upper bound", transform=ax.transAxes,
                fontsize=7, color="#555555")
    return fig
