"""Constellations from saved detector output, without synthetic display points."""

import matplotlib.pyplot as plt
import numpy as np

from .ber import COLORS


def plot_constellations(records: list[dict]) -> plt.Figure:
    """Plot the first measured frame, one panel for each receiver candidate."""
    fig, axes = plt.subplots(1, len(records), figsize=(3 * len(records), 3),
                             squeeze=False, layout="constrained")
    for ax, record in zip(axes.ravel(), records, strict=True):
        observed, reference = np.asarray(record["estimated"]), np.asarray(record["reference"])
        ax.scatter(observed[:, 0], observed[:, 1], s=10, alpha=.65,
                   color=COLORS[record["waveform"]])
        ax.scatter(reference[:, 0], reference[:, 1], marker="+", color="#333333", s=45)
        ax.set(title=f"{record['waveform'].upper()} · {record['detector'].upper()}",
               xlabel="In-phase", ylabel="Quadrature")
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(alpha=.18)
    return fig
