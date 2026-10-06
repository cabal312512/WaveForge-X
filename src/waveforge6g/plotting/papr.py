"""Empirical PAPR complementary cumulative distribution."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .ber import COLORS, MARKERS


def plot_papr(trials: pd.DataFrame) -> plt.Figure:
    """Draw P(PAPR > threshold) using one sample per waveform and paired frame."""
    fig, ax = plt.subplots(figsize=(5.6, 3.8), layout="constrained")
    unique = trials.drop_duplicates(["point_id", "waveform", "trial"])
    thresholds = np.linspace(0, max(12, unique.papr_db.max() + 1), 150)
    for waveform, group in unique.groupby("waveform", sort=False):
        probability = (group.papr_db.to_numpy()[:, None] > thresholds).mean(axis=0)
        valid = probability > 0
        ax.semilogy(thresholds[valid], probability[valid], color=COLORS[waveform],
                    marker=MARKERS[waveform], markevery=14, markersize=3,
                    label=waveform.upper())
    ax.set_xlabel("PAPR threshold (dB)")
    ax.set_ylabel("Exceedance probability")
    ax.set_ylim(max(.5 / len(unique), 1e-5), 1.05)
    ax.grid(True, which="major", alpha=.22)
    ax.legend(frameon=False)
    return fig
