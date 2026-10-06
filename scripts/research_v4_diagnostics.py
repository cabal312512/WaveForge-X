"""Focused trace, allocation and adversarial checks; uses saved R4 inputs."""

import json
import time
import tracemalloc

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from waveforge6g.channels import ChannelRealization
from waveforge6g.core.modulation import demodulate
from waveforge6g.experiments.research_v2 import Budget, write_json
from waveforge6g.experiments.research_v4 import CFG, DOC, OUT
from waveforge6g.experiments.research_v4_precision import decimal_reference
from waveforge6g.experiments.research_v4_report import COLORS, save
from waveforge6g.receivers.certiphy import CertiPHY, StopRule
from waveforge6g.waveforms import create_waveform

budget = Budget(output_root=OUT)
protocol = json.loads((CFG/"confirmation_protocol.json").read_text())
cases = [("core", 71025, 6), ("core", 71029, 6), ("scale", 72019, 4), ("scale", 72023, 4)]
names = ["global_d001", "radau_d001", "gray_d001", "gray_periodic_d001", "gray_d0", "residual_fast_d0"]
records, profiles = [], []
for stage, seed, epoch in cases:
    directory = OUT/stage/str(seed)
    manifest = json.loads((directory/"manifest.json").read_text())
    n, mod, name = manifest["n"], manifest["modulation"], manifest["waveform"]
    raw = np.load(directory/f"frame_{epoch}_nmse0.npz")
    channel = ChannelRealization.from_dict(json.loads(str(raw["estimated_channel_json"])))
    wave = create_waveform(name, n, n//8, 16)
    expected = demodulate(raw["reference_symbols"], mod)
    for method in names:
        budget.reserve(1)
        result = CertiPHY(wave, channel, 10**(-2.8), mod).solve(raw["received"], StopRule(**protocol["rules"][method]),
                                                             max_work=8e6*n/128, keep_trace=True)
        for point in result["trace"]:
            records.append(dict(seed=seed, n=n, modulation=mod, method=method, iteration=point["iteration"],
                                work=point["work"], coverage=point["coverage"], bound=point["disagreement_bound"],
                                disagreement=float(np.mean(point["output"] != expected)), radius=point["radius"],
                                actual_error_norm=float(np.linalg.norm(point["soft"]-raw["reference_symbols"]))))
    for method in ("global_d001", "gray_d001", "residual_fast_d001"):
        rule = StopRule(**protocol["rules"][method])
        tracemalloc.start()
        receiver = CertiPHY(wave, channel, 10**(-2.8), mod)
        result = receiver.solve(raw["received"], rule, max_work=8e6*n/128)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        start, cpu = time.perf_counter(), time.process_time()
        budget.reserve(17)
        for _ in range(16):
            receiver = CertiPHY(wave, channel, 10**(-2.8), mod)
            receiver.solve(raw["received"], rule, max_work=8e6*n/128)
        profiles.append(dict(n=n, modulation=mod, method=method, seed=seed, traced_peak_bytes=peak,
                             array_storage_bytes=result["array_storage_bytes"],
                             repeated_elapsed_s=(time.perf_counter()-start)/16,
                             repeated_process_cpu_s=(time.process_time()-cpu)/16))
pd.DataFrame(records).to_csv(DOC/"mechanism_traces.csv", index=False)
pd.DataFrame(profiles).to_csv(DOC/"memory_cpu_profile.csv", index=False)

data = pd.DataFrame(records)
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
for metric, title, output_name in (("coverage", "Individually certified bit fraction", "certified_vs_compute"),
                                  ("disagreement", "Measured disagreement with reference", "disagreement_vs_compute")):
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout="constrained")
    for ax, (_, seed, _) in zip(axes.flat, cases, strict=True):
        part = data[data.seed == seed]
        for method, points in part.groupby("method"):
            family = "residual_fast" if method.startswith("residual_fast") else method.split("_")[0]
            style = ":" if "periodic" in method else ("--" if method.endswith("_d0") else "-")
            ax.plot(points.work, points[metric], style, marker=".", markersize=4, c=COLORS[family], label=method)
        ax.set(xscale="log", xlabel="Cumulative work (all preparation charged)", ylabel=title,
               title=f"N={part.n.iloc[0]}, {part.modulation.iloc[0].upper()}, 28 dB")
        if metric == "disagreement":
            ax.set_yscale("symlog", linthresh=1e-4)
            ax.set_ylim(bottom=0)
        else:
            ax.set_ylim(0, 1.03)
        ax.grid(alpha=.18)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3, frameon=False, fontsize=8)
    save(fig, output_name)

# Independent 80-decimal-digit solve, a nearly singular drift case and an
# explicit invalid-Ritz example. No adversarial input is discarded as a PHY win.
wave = create_waveform("afdm", 8, 2, 4, c1=.017)
channel = ChannelRealization([0, 1, 2], [.8, .35j, -.17], [20, 1600, -701], 64000)
y = np.random.default_rng(89001).normal(size=8)+1j*np.random.default_rng(89002).normal(size=8)
receiver = CertiPHY(wave, channel, .003, "qam16")
gold = decimal_reference(receiver.matrix.toarray(), y, .003)
result = receiver.solve(y, StopRule(method="gray", period=1), keep_trace=True)
wave2 = create_waveform("ofdm", 32, 4, 8)
channel2 = ChannelRealization([0, 1], [1/np.sqrt(2), -1/np.sqrt(2)], [0, 0], 64000)
drift = CertiPHY(wave2, channel2, 1e-12, "qpsk").solve(np.random.default_rng(77).normal(size=32),
              StopRule(method="gray", period=2, max_iterations=64), keep_trace=True)
budget.reserve(2)
write_json(DOC/"adversarial_checks.json", {
    "decimal_digits": 80, "decimal_reference_time_error": float(np.linalg.norm(result["time_estimate"]-gold)),
    "decimal_reference_bit_differences": int(np.sum(result["bits"] != demodulate(wave.analysis(gold), "qam16"))),
    "decimal_bound_violations": int(sum(np.linalg.norm(t["soft"]-wave.analysis(gold)) > t["radius"]+1e-12 for t in result["trace"])),
    "drift_restarts": drift["restarts"], "drift_unknown_fraction": drift["unknown_fraction"],
    "drift_status": drift["status"], "max_recursive_drift": max(t["recursive_drift"] for t in drift["trace"]),
    "invalid_ritz_example": {"A_diagonal": [.001, 1.], "start": [0., 1.], "ritz": 1., "lambda_min": .001,
                             "error_norm": 1., "false_bound": .001},
    "hard_decision_plateau": {"development_seed": 62002, "waveform": "afdm", "modulation": "qam16", "N": 128,
                              "unchanged_checks": [16, 20, 24, 28], "reference_bit_changes_after_plateau": 1},
    "floating_point_certified": False})
budget.save()
