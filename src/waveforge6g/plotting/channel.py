"""Channel figures from an actual saved realization."""

import matplotlib.pyplot as plt
import numpy as np


def plot_channel(channel: dict) -> plt.Figure:
    """Plot path power and signed Doppler. Coincident delays remain separate paths."""
    gains = np.asarray(channel["gains"])
    power = np.maximum(np.sum(gains ** 2, axis=1), 1e-15)
    delay_us = np.asarray(channel["delays"]) / channel["sample_rate_hz"] * 1e6
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2), layout="constrained")
    axes[0].vlines(delay_us, -60, 10 * np.log10(power), color="#24566f")
    axes[0].plot(delay_us, 10 * np.log10(power), "o", color="#24566f", markersize=4)
    axes[0].set(xlabel="Delay (µs)", ylabel="Path power (dB)")
    scatter = axes[1].scatter(delay_us, channel["dopplers_hz"], c=10 * np.log10(power),
                              s=65, cmap="cividis", edgecolor="white")
    axes[1].set(xlabel="Delay (µs)", ylabel="Doppler (Hz)")
    fig.colorbar(scatter, ax=axes[1], label="Path power (dB)", shrink=.8)
    for ax in axes:
        ax.grid(alpha=.2)
    return fig
