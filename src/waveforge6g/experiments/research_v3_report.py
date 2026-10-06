"""R3 figures and paired trajectory statistics, without policy tuning."""

import json
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t as student_t

from ..channels import ChannelRealization
from .research_v2 import write_json
from .research_v3 import DOC, actions, completed, link_work
from .research_v3_adapt import budget_value

COLORS = {"ofdm": "#215b8f", "otfs": "#be7b22", "afdm": "#347864"}
POLICIES = ["fixed", "rx_ofdm", "rx_otfs", "rx_afdm", "joint", "dense_reference"]


def save(fig, name):
    fig.savefig(DOC/(name+".png"), dpi=180, bbox_inches="tight")
    fig.savefig(DOC/(name+".pdf"), bbox_inches="tight")
    plt.close(fig)


def load_grid():
    frames = []
    for path, manifest in completed("grid"):
        frame = pd.read_csv(path/"summary.csv")
        s = manifest["settings"]
        frame["profile"], frame["channel_seed"] = s["profile"], s["seed"]
        frame["source"] = path.name
        frames.append(frame)
    result = pd.concat(frames, ignore_index=True)
    result.to_csv(DOC/"development_grid.csv", index=False)
    return result


def grid_figures(grid):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), layout="constrained")
    for ax, (n, mod) in zip(axes.flat, [(128, "qpsk"), (128, "qam16"), (256, "qpsk"), (256, "qam16")], strict=True):
        cell = grid[(grid.n == n) & (grid.modulation == mod) & (grid.profile == "high_rich") & (grid.snr_db == 8)]
        means = cell.groupby(["action", "waveform", "domain"], as_index=False)[["ber", "total_work"]].mean()
        for (wave, domain), part in means.groupby(["waveform", "domain"]):
            ax.scatter(part.total_work, part.ber, c=COLORS[wave], marker={"time": "o", "symbol": "^", "dense": "s"}[domain],
                       s=38, label=f"{wave.upper()} / {domain}", alpha=.85)
        ordered = means.sort_values("total_work")
        frontier = ordered[ordered.ber < ordered.ber.cummin().shift(fill_value=np.inf)]
        ax.plot(frontier.total_work, frontier.ber, color="#444444", lw=1, zorder=0)
        ax.set(xscale="log", xlabel="Work units / frame (all preparation charged)", ylabel="BER",
               title=f"N={n}, {mod.upper()} | 8 paths, high Doppler, 8 dB")
        ax.grid(alpha=.18)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3, frameon=False, fontsize=8)
    save(fig, "pareto")

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), layout="constrained")
    cells = []
    for ax, (n, mod) in zip(axes.flat, [(128, "qpsk"), (128, "qam16"), (256, "qpsk"), (256, "qam16")], strict=True):
        conditions = [(p, s) for p in ("low_sparse", "high_sparse", "high_rich") for s in (8, 18)]
        image = np.zeros((6, 3))
        for r, (profile, snr) in enumerate(conditions):
            subset = grid[(grid.n == n) & (grid.modulation == mod) & (grid.profile == profile) & (grid.snr_db == snr)]
            means = subset.groupby("action_index").agg(selection=("selection_ber", "mean"), ber=("ber", "mean"), cost=("total_work", "max"))
            for c, level in enumerate(("low", "medium", "high")):
                feasible = means[means.cost <= budget_value(n, level)]
                best = int(feasible.sort_values(["selection", "cost"]).index[0])
                picked = subset[subset.action_index == best].sort_values("channel_seed")
                row = picked.iloc[0]
                tied = []
                for other in feasible.index:
                    if other == best:
                        continue
                    difference = subset[subset.action_index == other].sort_values("channel_seed").ber.to_numpy()-picked.ber.to_numpy()
                    radius = student_t.ppf(1-.05/(2*max(1, len(feasible)-1)), 2)*difference.std(ddof=1)/np.sqrt(3)
                    if difference.mean()-radius <= 0:
                        tied.append(int(other))
                image[r, c] = {"ofdm": 0, "otfs": 1, "afdm": 2}[row.waveform]
                label = f"{row.waveform.upper()} {row.domain}\nL{row.keep} K{row.iterations}"+(" †" if tied else "")
                ax.text(c, r, label, ha="center", va="center", fontsize=8)
                cells.append(dict(n=n, modulation=mod, profile=profile, snr=snr, level=level,
                                  budget=budget_value(n, level), selected_action=row.action, validation_ber=means.loc[best, "ber"],
                                  reserved_work=means.loc[best, "cost"], unresolved_alternatives=len(tied)))
        ax.imshow(image, cmap="Pastel2", vmin=0, vmax=2, alpha=.8, aspect="auto")
        ax.set(xticks=range(3), xticklabels=["Low B", "Medium B", "High B"], yticks=range(6),
               yticklabels=[f"{p.replace('_', ' ')} / {s} dB" for p, s in conditions], title=f"N={n}, {mod.upper()}")
    fig.suptitle("Development selection; † unresolved alternatives (paired channel intervals)\nReceiver cost only; deployed policy overhead is included in confirmation", fontsize=11)
    save(fig, "budget_condition")
    pd.DataFrame(cells).to_csv(DOC/"budget_condition.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), layout="constrained")
    subset = grid[(grid.n == 256) & (grid.modulation == "qpsk") & (grid.profile == "high_rich") & (grid.snr_db == 8)]
    reference = subset[subset.domain == "dense"].set_index(["waveform", "channel_seed"]).ber
    for name, rows in subset[subset.domain == "symbol"].groupby("waveform"):
        gap = np.array([row.ber-reference.loc[name, row.channel_seed] for row in rows.itertuples()])
        axes[0].scatter(rows.discarded_energy, gap, label=name.upper(), c=COLORS[name], s=22)
        axes[1].scatter(rows.normal_residual, gap, label=name.upper(), c=COLORS[name], s=22)
    axes[0].set(xlabel="Discarded channel energy fraction", ylabel="BER − same-wave dense BER", xscale="log")
    axes[1].set(xlabel="Relative normal-equation residual", xscale="log")
    for path, manifest in completed("grid"):
        s = manifest["settings"]
        if (s["n"], s["modulation"], s["profile"], s["snr"], s["seed"]) != (256, "qpsk", "high_rich", 8, 42001):
            continue
        for item in json.loads((path/"convergence.json").read_text()):
            if ":time:L0:K16" in item["action"] or ":symbol:L8:K16" in item["action"]:
                name = item["action"].split(":")[0]
                axes[2].semilogy(item["residual"], c=COLORS[name], ls="-" if ":time:" in item["action"] else "--", label=item["action"])
    axes[2].set(xlabel="PCG iteration", ylabel="Relative residual")
    axes[0].legend(frameon=False, fontsize=8)
    axes[2].legend(frameon=False, fontsize=6)
    for ax in axes:
        ax.grid(alpha=.18)
    save(fig, "truncation_convergence")


def confirmation_statistics():
    tables = [pd.read_csv(path/"trajectory.csv") for path, _ in completed("confirmation")]
    if len(tables) != 60:
        raise RuntimeError(f"confirmation incomplete: {len(tables)}/60")
    paths = pd.concat(tables, ignore_index=True)
    bounded = paths[paths.policy != "dense_reference"]
    assert (bounded.reserved_work <= bounded.budget).all()
    assert (bounded.actual_work <= bounded.reserved_work+1e-7).all()
    trajectory = paths.groupby(["seed", "n", "modulation", "level", "policy"], as_index=False).agg(
        ber=("ber", "mean"), work=("actual_work", "mean"), cpu_s=("cpu_s", "mean"), outage=("outage", "mean"))
    trajectory.to_csv(DOC/"confirmation_trajectories.csv", index=False)
    paths.to_csv(DOC/"confirmation_paths.csv", index=False)
    summary = trajectory.groupby("policy")[["ber", "work", "cpu_s", "outage"]].mean().reindex(POLICIES)
    summary.to_csv(DOC/"policy_summary.csv")
    # Stratified trajectory bootstrap: same seed across all policies in each draw.
    pivot = trajectory.pivot(index=["n", "modulation", "level", "seed"], columns="policy", values="ber")
    rng = np.random.default_rng(63001)
    groups = [group.to_numpy() for _, group in pivot.groupby(level=[0, 1, 2])]
    draws = np.mean([group[rng.integers(0, len(group), (5000, len(group)))].mean(axis=1) for group in groups], axis=0)
    columns = list(pivot.columns)
    comparisons = []
    for baseline in ("fixed", "rx_ofdm", "rx_otfs", "rx_afdm", "dense_reference"):
        values = draws[:, columns.index("joint")]-draws[:, columns.index(baseline)]
        point = float((pivot["joint"]-pivot[baseline]).mean())
        lo, hi = np.quantile(values, [.025, .975])
        # Four policy comparisons form the confirmatory family; reference descriptive.
        slo, shi = np.quantile(values, [.00625, .99375])
        comparisons.append(dict(comparison=f"joint - {baseline}", delta_ber=point, paired95_low=lo, paired95_high=hi,
                                family95_low=slo, family95_high=shi, trajectories=60))
    comparisons = pd.DataFrame(comparisons)
    comparisons.to_csv(DOC/"paired_comparisons.csv", index=False)
    by_budget = []
    for level, part in pivot.groupby(level="level"):
        groups = [g.to_numpy() for _, g in part.groupby(level=[0, 1])]
        samples = np.mean([g[rng.integers(0, len(g), (5000, len(g)))].mean(axis=1) for g in groups], axis=0)
        for baseline in ("fixed", "rx_ofdm", "rx_otfs", "rx_afdm"):
            delta = samples[:, columns.index("joint")]-samples[:, columns.index(baseline)]
            lo, hi = np.quantile(delta, [.025, .975])
            by_budget.append(dict(level=level, baseline=baseline, delta=float((part.joint-part[baseline]).mean()),
                                  paired95_low=lo, paired95_high=hi, trajectories=len(part)))
    pd.DataFrame(by_budget).to_csv(DOC/"budget_paired_comparisons.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    labels = ["Fixed config", "RX / OFDM", "RX / OTFS", "RX / AFDM", "Joint", "Dense ref."]
    for i, name in enumerate(POLICIES):
        j = columns.index(name)
        lo, hi = np.quantile(draws[:, j], [.025, .975])
        mean = summary.loc[name, "ber"]
        axes[0].errorbar(i, mean, yerr=[[max(0, mean-lo)], [max(0, hi-mean)]], fmt="o", c="#215b8f" if name != "joint" else "#b34935", capsize=3)
        axes[1].scatter(summary.loc[name, "work"], mean, s=50, marker="s" if name == "dense_reference" else "o")
        axes[1].annotate(labels[i], (summary.loc[name, "work"], mean), xytext=(5, 3), textcoords="offset points", fontsize=8)
    axes[0].set(xticks=range(6), xticklabels=labels, ylabel="Trajectory-mean BER", title="60 fresh paired trajectories; 95% stratified bootstrap")
    axes[0].tick_params(axis="x", rotation=25)
    axes[1].set(xscale="log", xlabel="Mean work units / frame", ylabel="BER", title="All preparation and policy work charged")
    for ax in axes:
        ax.grid(alpha=.18)
    save(fig, "policy_comparison")
    # Preserve per-stratum comparisons rather than hiding budget dependence.
    trajectory.groupby(["n", "modulation", "level", "policy"])[["ber", "work", "outage"]].mean().to_csv(DOC/"policy_strata.csv")
    changes = []
    for (seed, policy), rows in paths.groupby(["seed", "policy"]):
        rows = rows.sort_values("epoch")
        changes.append(dict(seed=seed, policy=policy, changes=int((rows.action.iloc[1:].to_numpy() != rows.action.iloc[:-1].to_numpy()).sum()),
                            waveform_changes=int((rows.action.str.split(":").str[0].iloc[1:].to_numpy() != rows.action.str.split(":").str[0].iloc[:-1].to_numpy()).sum())))
    pd.DataFrame(changes).to_csv(DOC/"policy_switches.csv", index=False)
    write_json(DOC/"confirmation_integrity.json", {"trajectories": 60, "epochs_each": 24,
               "budget_violations": int((bounded.reserved_work > bounded.budget).sum()),
               "outages": int(bounded.outage.sum()), "inference_unit": "trajectory, paired and stratified"})
    return summary, comparisons


def offline_envelope():
    """Evaluation-only selection-bank envelope; never passed to any policy."""
    rows = []
    for directory, manifest in completed("confirmation"):
        n, mod, seed = manifest["n"], manifest["modulation"], manifest["seed"]
        candidates = actions(n)
        cap = budget_value(n, manifest["level"])
        raw = np.load(directory/"replicas.npz")
        channels = json.loads((directory/"channels.json").read_text())
        paths = pd.read_csv(directory/"trajectory.csv")
        for t, saved in enumerate(channels):
            channel = ChannelRealization.from_dict(saved["channel"])
            overhead = paths[(paths.policy == "joint") & (paths.epoch == t)].policy_work.iloc[0]
            costs = np.array([link_work(SimpleNamespace(name=a.waveform, n_symbols=n, cp_length=n//8, subcarriers=16),
                                       channel, a, mod)["total_work"]+overhead for a in candidates])
            select = raw["selection_errors"][t].mean(axis=0)/raw["bit_count"][t]
            validate = raw["validation_errors"][t].mean(axis=0)/raw["bit_count"][t]
            for restriction in ("joint", "ofdm", "otfs", "afdm"):
                eligible = [i for i, a in enumerate(candidates) if a.domain != "dense" and costs[i] <= cap
                            and (restriction == "joint" or a.waveform == restriction)]
                chosen = min(eligible, key=lambda i: (select[i], costs[i]))
                rows.append(dict(seed=seed, n=n, modulation=mod, level=manifest["level"], epoch=t,
                                 restriction=restriction, action=candidates[chosen].key, ber=validate[chosen],
                                 selected_bank_ber=select[chosen], work=costs[chosen]))
    frame = pd.DataFrame(rows)
    frame.groupby(["seed", "n", "modulation", "level", "restriction"])[["ber", "selected_bank_ber", "work"]].mean().to_csv(DOC/"offline_envelope.csv")


def report():
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "savefig.facecolor": "white"})
    grid = load_grid()
    grid_figures(grid)
    confirmation_statistics()
    offline_envelope()
