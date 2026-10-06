"""Independent PDF/PNG figure rendering and a small static research report."""

from __future__ import annotations

import html
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.colors import ListedColormap

from ..analysis.waveform_selector import WAVEFORMS, NearestRegionSelector
from ..reproducibility import configure_local_environment
from .ber import COLORS, plot_ber
from .channel import plot_channel
from .constellation import plot_constellations
from .papr import plot_papr


def set_style() -> None:
    """Restrained colors, embedded vector fonts, no external TeX requirement."""
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titlesize": 10, "lines.linewidth": 1.4,
                         "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 180})


def save_figure(figure: plt.Figure, directory: Path, name: str) -> list[Path]:
    """Save both vector PDF and raster PNG, then release figure memory."""
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for suffix in ("pdf", "png"):
        target = directory / f"{name}.{suffix}"
        figure.savefig(target, bbox_inches="tight")
        paths.append(target)
    plt.close(figure)
    return paths


def plot_results(directory: str | Path, output_dir: str | Path | None = None) -> Path:
    """Regenerate all figures from CSV/JSON; never rerun a simulation."""
    root = configure_local_environment()
    source = Path(directory)
    target = Path(output_dir) if output_dir else root / "figures" / source.name
    config = yaml.safe_load((source / "config.yaml").read_text(encoding="utf-8"))
    data, trials = pd.read_csv(source / "metrics.csv"), pd.read_csv(source / "trials.csv")
    set_style()
    kind = config["experiment"]["kind"]
    if kind != "papr":
        x = "frame_size" if kind == "complexity" else (
            "max_doppler_hz" if len(config["sweep"].get("max_doppler_hz", [])) > 1
            and len(config["sweep"].get("snr_db", [])) <= 1 else "snr_db")
        fixed = [key for key in ("snr_db", "max_doppler_hz", "frame_size") if key != x]
        for index, (_, group) in enumerate(data.groupby(fixed, dropna=False, sort=False)):
            figure = plot_ber(group, x)
            save_figure(figure, target, f"ber_{x}_{index:02}")
        if kind == "complexity":
            figure, ax = plt.subplots(figsize=(5.6, 3.8), layout="constrained")
            for (waveform, detector), group in data.groupby(["waveform", "detector"], sort=False):
                group = group.sort_values("frame_size")
                ax.loglog(group.frame_size, group.runtime_mean_s * 1e3, "o-",
                           color=COLORS[waveform], label=f"{waveform.upper()} · {detector.upper()}")
            ax.set(xlabel="Useful symbols per frame", ylabel="Mean link runtime (ms)")
            ax.grid(alpha=.22)
            ax.legend(frameon=False, fontsize=8)
            save_figure(figure, target, "runtime_frame_size")
    # PAPR depends on frame size/modulation, not channel; avoid pooling frame sizes.
    for n, group in trials.groupby("frame_size", sort=True):
        save_figure(plot_papr(group), target, f"papr_ccdf_n{n}")
    examples = json.loads((source / "channel_examples.json").read_text(encoding="utf-8"))
    if examples:
        save_figure(plot_channel(examples[0]["channel"]), target, "channel")
        if examples[0].get("constellations"):
            save_figure(plot_constellations(examples[0]["constellations"]), target, "constellation")
    _write_report(source, target, data)
    return target


def _write_report(source: Path, target: Path, data: pd.DataFrame) -> None:
    metadata = json.loads((source / "metadata.json").read_text(encoding="utf-8"))
    title = yaml.safe_load((source / "config.yaml").read_text(encoding="utf-8"))["experiment"]["name"]
    panels = "\n".join(f'<figure><a href="{p.stem}.pdf"><img src="{p.name}" '
                       f'alt="{html.escape(p.stem)}"></a></figure>' for p in sorted(target.glob("*.png")))
    columns = ["waveform", "detector", "snr_db", "max_doppler_hz", "n_frames", "n_bits",
               "bit_errors", "ber", "papr_mean_db", "runtime_mean_s"]
    table = data[columns].to_html(index=False, float_format=lambda x: f"{x:.5g}", border=0)
    report = f'''<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>{html.escape(title)}</title>
<style>body{{max-width:1080px;margin:56px auto;padding:0 28px;color:#23313a;background:#faf9f6;
font:15px/1.55 system-ui,sans-serif}}h1{{font-size:30px;font-weight:600;margin-bottom:8px}}
small{{color:#657079}}figure{{margin:36px 0;background:white}}img{{max-width:100%;display:block}}
table{{border-collapse:collapse;font-size:12px;width:100%}}th,td{{padding:9px 10px;border-bottom:1px solid #dce0df;
text-align:right}}th{{font-weight:600}}.table{{overflow:auto}}code{{font-size:12px;overflow-wrap:anywhere}}</style>
<h1>{html.escape(title)}</h1><small>{html.escape(metadata['utc_timestamp'])} · seed {metadata['random_seed']}</small>
{panels}<div class="table">{table}</div><p><small>Configuration <code>{metadata['configuration_hash']}</code>
<br>Measured simulation output · Es/N0 at transmitter · timing includes channel and receiver.</small></p></html>'''
    (target / "report.html").write_text(report, encoding="utf-8")


def plot_selection_map(model: NearestRegionSelector, output_dir: Path,
                       delay_span_s: float | None = None, modulation_bits: int | None = None) -> Path:
    """Render a within-training-range slice; markers show observed training labels."""
    configure_local_environment()
    set_style()
    features = np.asarray(model.features)
    delay = float(np.median(features[:, 2])) if delay_span_s is None else delay_span_s
    bits = int(np.median(features[:, 3])) if modulation_bits is None else modulation_bits
    snrs = np.linspace(features[:, 0].min(), features[:, 0].max(), 121)
    dopplers = np.linspace(features[:, 1].min(), features[:, 1].max(), 101)
    if np.ptp(snrs) == 0 or np.ptp(dopplers) == 0:
        raise ValueError("selection map needs at least two measured SNRs and Dopplers")
    xx, yy = np.meshgrid(snrs, dopplers)
    query = np.column_stack((xx.ravel(), yy.ravel(), np.full(xx.size, delay), np.full(xx.size, bits)))
    indices = np.asarray([WAVEFORMS.index(label) for label in model.predict(query)]).reshape(xx.shape)
    fig, ax = plt.subplots(figsize=(6, 4), layout="constrained")
    cmap = ListedColormap([COLORS[w] for w in WAVEFORMS])
    color = ax.pcolormesh(xx, yy, indices, shading="nearest", cmap=cmap, vmin=-.5, vmax=2.5,
                          rasterized=True, alpha=.88)
    selected = np.isclose(features[:, 2], delay) & (features[:, 3] == bits)
    ax.scatter(features[selected, 0], features[selected, 1], s=20, facecolors="none", edgecolors="white")
    ax.set(xlabel=r"$E_s/N_0$ (dB)", ylabel="Maximum Doppler (Hz)",
           title=f"{bits} bits/symbol · delay span {delay * 1e6:g} µs")
    bar = fig.colorbar(color, ax=ax, ticks=[0, 1, 2], shrink=.8)
    bar.ax.set_yticklabels([w.upper() for w in WAVEFORMS])
    save_figure(fig, output_dir, "waveform_selection")
    pd.DataFrame({"snr_db": xx.ravel(), "max_doppler_hz": yy.ravel(),
                  "selected_waveform": [WAVEFORMS[i] for i in indices.ravel()]}).to_csv(
                      output_dir / "selection_grid.csv", index=False)
    return output_dir
