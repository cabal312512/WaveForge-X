"""Trajectory-level R4 statistics and measured certificate/cost figures."""

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .research_v2 import sha, write_json
from .research_v4 import DOC, OUT

COLORS = {"global": "#a05a46", "radau": "#8971ad", "gray": "#246b66", "residual_fast": "#265d8d",
          "fixed": "#555555", "stable": "#bb8729", "krylov": "#8971ad", "dual": "#bd8e38"}


def save(fig, name):
    fig.savefig(DOC/(name+".png"), dpi=170, bbox_inches="tight")
    fig.savefig(DOC/(name+".pdf"), bbox_inches="tight")
    plt.close(fig)


def load_confirmation():
    frames = []
    for stage, expected in (("core", 48), ("scale", 36), ("mismatch", 12)):
        files = sorted((OUT/stage).glob("*/manifest.json"))
        if len(files) != expected:
            raise RuntimeError(f"incomplete {stage}: {len(files)}/{expected}")
        for path in files:
            manifest = json.loads(path.read_text())
            if not all(sha(path.parent/name) == digest for name, digest in manifest["files"].items()):
                raise RuntimeError(f"corrupt evidence: {path}")
            part = pd.read_csv(path.parent/"frames.csv")
            part["stage"] = stage
            frames.append(part)
    result = pd.concat(frames, ignore_index=True)
    refinement = OUT/"stability_refinement"
    if (refinement/"manifest.json").exists():
        manifest = json.loads((refinement/"manifest.json").read_text())
        if not all(sha(refinement/name) == digest for name, digest in manifest["files"].items()):
            raise RuntimeError("strengthened baseline evidence corrupted")
        result = pd.concat([result[~result.method.str.startswith("stable_")], pd.read_csv(refinement/"frames.csv")], ignore_index=True)
    result["adjusted_bound"] = np.minimum(1., result.bound+result.reference_ambiguous/result.bits)
    result["bound_violated"] = result.disagreement > result.adjusted_bound+1e-14
    result["measured_exceedance"] = result.disagreement > result.delta+1e-14
    result["budget_unmet"] = result.method.str.startswith(("gray", "radau", "global")) & (~result.requested_met)
    result.to_csv(DOC/"confirmation_frames.csv", index=False)
    return result


def paired_statistics(data):
    metrics = ["ber", "reference_ber", "disagreement", "bound", "coverage", "work", "iterations", "checks",
               "elapsed_s", "process_cpu_s", "budget_unmet", "measured_exceedance", "reference_ambiguous",
               "model_mismatch_disagreement", "certified_violation_count", "bound_violated"]
    trajectory = data.groupby(["stage", "seed", "n", "waveform", "modulation", "nmse", "method"], as_index=False)[metrics].mean()
    trajectory.to_csv(DOC/"trajectory_summary.csv", index=False)
    summary = trajectory.groupby(["stage", "nmse", "method"])[metrics].mean()
    summary.to_csv(DOC/"method_summary.csv")
    data.groupby(["stage", "nmse", "method"]).agg(work_p50=("work", "median"), work_p95=("work", lambda s: s.quantile(.95)),
                    work_max=("work", "max"), elapsed_p95=("elapsed_s", lambda s: s.quantile(.95)),
                    iterations_max=("iterations", "max"), disagreement_max=("disagreement", "max")).to_csv(DOC/"tails.csv")
    comparisons = []
    for stage in ("core", "scale"):
        subset = trajectory[trajectory.stage == stage]
        for label in ("d0", "d001", "d005"):
            pivot = subset.pivot(index=["n", "waveform", "modulation", "seed"], columns="method", values="work")
            columns = list(pivot.columns)
            rng = np.random.default_rng(81001)
            groups = [part.to_numpy() for _, part in pivot.groupby(level=[0, 1, 2])]
            draws = np.mean([g[rng.integers(0, len(g), (5000, len(g)))].mean(axis=1) for g in groups], axis=0)
            for baseline in ("fixed", "residual_fast", "stable", "global", "radau"):
                a, b = "gray_"+label, baseline+"_"+label
                ratio = draws[:, columns.index(a)]/draws[:, columns.index(b)]
                lo, hi = np.quantile(ratio, [.025, .975])
                comparisons.append(dict(stage=stage, delta=label, baseline=baseline, ratio=pivot[a].mean()/pivot[b].mean(),
                                        paired95_low=lo, paired95_high=hi, trajectories=len(pivot),
                                        mean_work_difference=float((pivot[a]-pivot[b]).mean())))
    pd.DataFrame(comparisons).to_csv(DOC/"paired_cost_ratios.csv", index=False)
    matched = []
    # Descriptive convex-mixture interpolation, never a tuned deployed policy.
    for stage in ("core", "scale"):
        values = summary.loc[(stage, 0.)]
        for label in ("d0", "d001", "d005"):
            candidate = values.loc["gray_"+label]
            for family in ("fixed", "residual_fast", "stable"):
                points = values.loc[[family+"_"+d for d in ("d0", "d001", "d005")]].sort_values("disagreement")
                best = None
                for i in range(len(points)-1):
                    left, right = points.iloc[i], points.iloc[i+1]
                    if left.disagreement <= candidate.disagreement <= right.disagreement and right.disagreement > left.disagreement:
                        weight = (candidate.disagreement-left.disagreement)/(right.disagreement-left.disagreement)
                        work = (1-weight)*left.work+weight*right.work
                        best = work if best is None else min(work, best)
                matched.append(dict(stage=stage, delta=label, baseline=family, measured_disagreement=candidate.disagreement,
                                    gray_work=candidate.work, interpolated_baseline_work=best,
                                    gray_over_matched_work=np.nan if best is None else candidate.work/best,
                                    scope="evaluation-only interpolation; no quality equivalence claim or retuning"))
    pd.DataFrame(matched).to_csv(DOC/"empirical_quality_matching.csv", index=False)
    write_json(DOC/"numerical_checks.json", {"certified_bit_violations": int(data.certified_violation_count.sum()),
               "adjusted_bound_violations": int(data.bound_violated.sum()),
               "reference_ambiguous_records": int((data.reference_ambiguous > 0).sum()),
               "work_cap_violations": int((data.work > 8e6*data.n/128).sum()),
               "algorithm_frame_evaluations": len(data), "floating_point_certified": False,
               "inference_unit": "paired independent trajectory, stratified by N/waveform/modulation"})
    return trajectory, summary


def figures(data, trajectory, summary):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    for ax, stage in zip(axes, ("core", "scale"), strict=True):
        for family in ("fixed", "residual_fast", "stable", "global", "radau", "gray"):
            selected = summary.loc[(stage, 0.)]
            points = selected.loc[[family+"_"+label for label in ("d0", "d001", "d005")]]
            ax.plot(points.work, points.disagreement, "-", c=COLORS[family], label=family)
            for marker, (_, row) in zip(("s", "o", "^"), points.iterrows(), strict=True):
                ax.scatter(row.work, row.disagreement, marker=marker, s=38, c=COLORS[family])
        ax.set(xscale="log", yscale="symlog", xlabel="Total work / frame", ylabel="Measured reference disagreement",
               title="N=128/256, 48 trajectories" if stage == "core" else "N=512/1024, 36 trajectories")
        ax.set_yscale("symlog", linthresh=1e-5)
        ax.set_ylim(bottom=0)
        ax.grid(alpha=.18)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Markers: square δ=0 / circle δ=.01 / triangle δ=.05", fontsize=10)
    save(fig, "quality_cost")

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout="constrained")
    core = summary.loc[("core", 0.)]
    families = ["fixed", "residual_fast", "stable", "global", "radau", "gray"]
    for i, label in enumerate(("d0", "d001", "d005")):
        points = core.loc[[f+"_"+label for f in families]]
        axes[i].bar(range(6), points.work/1e6, color=[COLORS[f] for f in families])
        axes[i].set(xticks=range(6), xticklabels=["Fixed K", "Residual", "Stable", "Global", "Radau", "Gray"],
                    ylabel="Million work units / frame", title="Reference budget δ="+{"d0": "0", "d001": ".01", "d005": ".05"}[label])
        axes[i].tick_params(axis="x", rotation=35)
    save(fig, "total_cost")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    for label, style in (("d0", "-"), ("d001", "--"), ("d005", ":")):
        part = data[(data.stage != "mismatch") & (data.method == "gray_"+label)]
        grouped = part.groupby("n")[["coverage", "budget_unmet"]].mean()
        axes[0].plot(grouped.index, grouped.coverage, "o"+style, label=label)
        axes[1].plot(grouped.index, grouped.budget_unmet, "o"+style, label=label)
    axes[0].set(xlabel="N", ylabel="Individually certified bit fraction", ylim=(0, 1.02))
    axes[1].set(xlabel="N", ylabel="Requested bound not met: frame fraction", ylim=(-.02, 1.02))
    for ax in axes:
        ax.legend(frameon=False)
        ax.grid(alpha=.18)
    save(fig, "coverage_failures")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    mismatch = trajectory[trajectory.stage == "mismatch"]
    for family in ("residual_fast", "global", "gray"):
        part = mismatch[mismatch.method == family+"_d001"].groupby("nmse").mean(numeric_only=True)
        axes[0].plot(part.index, part.disagreement, "o-", label=family, c=COLORS[family])
        axes[1].plot(part.index, part.ber, "o-", label=family, c=COLORS[family])
    reference = mismatch[mismatch.method == "gray_d001"].groupby("nmse").mean(numeric_only=True)
    axes[0].plot(reference.index, reference.model_mismatch_disagreement, "s--", c="#444444", label="estimated vs true-model LMMSE")
    axes[0].set(xlabel="Receiver CSI NMSE", ylabel="Reference disagreement")
    axes[1].set(xlabel="Receiver CSI NMSE", ylabel="Actual BER")
    for ax in axes:
        ax.legend(frameon=False, fontsize=7)
        ax.grid(alpha=.18)
    save(fig, "csi_mismatch")


def report():
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
    data = load_confirmation()
    trajectory, summary = paired_statistics(data)
    figures(data, trajectory, summary)
    traces_path = DOC/"mechanism_traces.csv"
    if traces_path.exists():
        traces = pd.read_csv(traces_path)
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
        for ax, seed in zip(axes, (72019, 72023), strict=True):
            rows = traces[(traces.seed == seed) & (traces.method == "gray_d001")]
            ax.plot(rows.work, 1-rows.coverage, "o-", label="Unknown individual bits", c="#8971ad")
            ax.plot(rows.work, rows.bound, "s-", label="Joint disagreement bound", c="#246b66")
            ax.plot(rows.work, rows.disagreement, ".-", label="Observed disagreement", c="#265d8d")
            ax.set(xscale="log", xlabel="Cumulative work", ylabel="Bit fraction", title=f"N=1024, {rows.modulation.iloc[0].upper()}")
            ax.set_yscale("symlog", linthresh=1e-4)
            ax.set_ylim(bottom=0)
            ax.grid(alpha=.18)
        axes[0].legend(frameon=False, fontsize=8)
        save(fig, "joint_bound_vs_unknown")
