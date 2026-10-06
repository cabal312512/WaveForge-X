"""Recompute R2 tables, calibration diagnostics, and publication figures from raw records."""

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t as student_t

from .research_v2 import CFG, DOC, FAMILIES, OUT, sha, write_json


def completed(stage):
    records = []
    for path in sorted((OUT / stage).glob("*/manifest.json")):
        manifest = json.loads(path.read_text())
        if manifest["status"] == "complete":
            if not all(sha(path.parent / file) == digest for file, digest in manifest["files"].items()):
                raise ValueError(f"artifact integrity failed: {path.parent}")
            records.append((path.parent, manifest))
    # Keep one latest source version per exact scenario/seed/settings, preserve all files.
    unique = {}
    for path, manifest in records:
        settings = {k: v for k, v in manifest["settings"].items() if k != "source_hash"}
        key = json.dumps(settings, sort_keys=True)
        if key not in unique or path.stat().st_mtime > unique[key][0].stat().st_mtime:
            unique[key] = (path, manifest)
    return list(unique.values())


def gains(directory, manifest):
    table = pd.read_csv(directory / "trajectory.csv")
    data = np.load(directory / "replicas.npz")
    means = data["validation_loss"].mean(axis=1)
    result = []
    for _, row in table[table.policy == "cadc"].iterrows():
        d = json.loads(row.diagnostics)
        if d["incumbent"] == d["proposed_action"]:
            continue
        t, i, j = int(row.time), d["incumbent"], d["proposed_action"]
        true_estimate = means[t, i] - means[t, j]
        error = d["gain"] - true_estimate
        result.append({"seed": manifest["settings"]["seed"], "family": manifest["settings"]["family"],
                       "time": t, "prediction": d["gain"], "estimated_gain": true_estimate,
                       "error": error, "scale": d["gain_scale"], "radius": d["radius"],
                       "covered": abs(error) <= d["radius"],
                       "wrong_sign": d["gain"] * true_estimate < 0,
                       "probe": d["exploration_flag"]})
    return result


def calibrate():
    calibration = completed("calibrate")
    if len(calibration) != 4:
        raise ValueError("need all four predeclared calibration trajectories")
    samples = pd.DataFrame([row for directory, manifest in calibration for row in gains(directory, manifest)])
    multiplier = max(2., float(np.quantile(abs(samples.error) / samples.scale, .95)))
    samples.to_csv(DOC / "calibration_samples.csv", index=False)
    # Deployable fixed is chosen on development replicas, never confirmation.
    values = []
    for directory, _ in completed("pilot") + calibration:
        data = np.load(directory / "replicas.npz")
        values.append(data["validation_loss"].mean(axis=(0, 1)))
    fixed = int(np.argmin(np.mean(values, axis=0)))
    frozen = {"version": "r2-1", "parameters": {"horizon": 16, "block": 8, "sticky_hold": 15,
                                                "multiplier": multiplier},
              "deployable_fixed_action": fixed, "families": FAMILIES,
              "confirm_seeds": {family: list(range(10001 + 100*i, 10021 + 100*i)) for i, family in enumerate(FAMILIES)},
              "epochs": 120, "selection_k": 16, "validation_k": 16,
              "k64_rule": "first predeclared seed of each family, both independent streams",
              "primary_contrasts": ["cadc-sticky", "cadc-development_selected_fixed"],
              "practical_threshold_per_epoch": .002, "confidence": .975,
              "tuning": "one preset per online method; no hyperparameter search; empirical CADC scale calibrated on four held-out development trajectories",
              "source_inputs": [{"path": str(p), "manifest_sha": sha(p / "manifest.json")} for p, _ in calibration],
              "budget": {"seconds": 14400, "phy_evaluations": 2000000, "bytes": 5*1024**3}}
    if (CFG / "frozen.json").exists():
        old = json.loads((CFG / "frozen.json").read_text())
        if completed("confirm") and old != frozen:
            raise RuntimeError("cannot change protocol after confirmation")
    write_json(CFG / "frozen.json", frozen)
    (DOC / "protocol.md").write_text("# R2 冻结协议\n\n"
        "确认集生成前冻结于 configs/research_v2/frozen.json。80条独立轨迹，4家族各20；T=120，物理长度75ms。新种子是同分布新轨迹，不称OOD。场景名称是设计意图，不预设存在正适应收益。\n\n"
        "K=16选路、独立K=16重评；每家族预指定首条K=64，另12 epoch K=256为探索性精度诊断。固定状态副本不增加在线时间/反馈。包含bits、AWGN随机性，固定完整路径状态及CSI误差。目标保留R1权重，E[BLER]和E[截断PAPR]直接逐副本求均值；不以均值BER替代BLER。\n\n"
        "主比较为CADC−sticky与CADC−开发集所选固定。以独立轨迹为单位，等权四场景；分层trajectory bootstrap 4000次和配对t区间交叉检查，两个主比较使用97.5%区间（Bonferroni家族95%）。工程阈值每epoch0.002，收益上界不超过−0.002才称有实质改善。其余比较、子组、扩展均探索性。固定样本量、不按显著性增加样本。\n\n"
        "所有在线方法冷启动，无预训练奖励样本。每种只用一组公开默认超参数，无无限调参。CADC额外获得四条开发轨迹的标量校准系数，这是明确的开发信息优势；固定动作也只在开发集确定。bootstrap ridge每次只更新执行动作，12个Poisson副本的候选−当前差保留相关性。初始3×8帧付费探索，H=16固定；默认无检测、rho=0是假设敏感性基线，不是真实漂移上界。校准是经验性的，策略分布改变和有限MC均可能使覆盖失效，不给安全保证。\n\n"
        "RQ1假设单帧最优有乐观偏差：以独立重评减选路成本检验；相同均值玩具为反例检查。RQ2假设存在实质适应空间：固定与动态差及重评区间，若未分辨则转向收益边界，不搜确认种子。RQ3假设成本/重置解释旧劣势：严格base+switch账目，消融只作相关比较。RQ4假设CADC优于sticky/部署固定：上述主比较可反驳。RQ5噪声/时延/CSI分离扩展仅做机制示例，不对单条轨迹推断总体。\n\n"
        "四小时/200万PHY/5GiB累计上限，失败调用预留预算也计入。串行运行、每轨迹checkpoint，清单校验后续跑；停止不删除原始数据。E5长轨迹600/2000只提供机制示例，slow_change保持60epoch事件间隔从而增加事件数；fast_change混合轨迹随总长度伸缩，两者不混称相同物理时标。E7优先旧重置消融及小型toy；不追加复杂检测器以追求胜出。反馈是simulator-assisted selected-action feedback；转移成本是synthetic penalty，未测硬件或扣除payload。\n", encoding="utf-8")
    print(json.dumps(frozen["parameters"]), "development fixed:", fixed)


def interval(values, confidence=.975):
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    half = float(student_t.ppf((1+confidence)/2, len(values)-1) * values.std(ddof=1)/np.sqrt(len(values)))
    return mean, mean-half, mean+half


def report():
    frozen = json.loads((CFG / "frozen.json").read_text())
    runs = completed("confirm")
    if len(runs) != 80:
        raise ValueError(f"expected 80 completed confirmation trajectories, found {len(runs)}")
    summaries, references, gain_rows, indices = [], [], [], []
    for directory, manifest in runs:
        settings = manifest["settings"]
        fields = {"family": settings["family"], "seed": settings["seed"], "run": str(directory)}
        summaries.append(pd.read_csv(directory / "summary.csv").assign(**fields))
        references.append(pd.read_csv(directory / "references.csv").assign(**fields))
        gain_rows.extend(gains(directory, manifest))
        indices.append({**fields, "manifest_sha256": sha(directory / "manifest.json")})
    summary, reference, gain = pd.concat(summaries), pd.concat(references), pd.DataFrame(gain_rows)
    for name, table in (("summary", summary), ("references", reference), ("gain_coverage", gain)):
        table.to_csv(DOC / f"{name}.csv", index=False)
    write_json(DOC / "evidence_index.json", indices)
    wide = summary.pivot(index=["family", "seed"], columns="policy", values="expected_objective")
    fixed = ("always_ofdm", "always_otfs", "always_afdm")[frozen["deployable_fixed_action"]]
    contrasts = []
    rng = np.random.default_rng(77119)
    for policy in ("sticky", fixed, "block", "linucb", "uncertainty_aware"):
        difference = wide.cadc - wide[policy]
        mean, low, high = interval(difference)
        groups = [difference.loc[family].to_numpy() for family in FAMILIES]
        boot = np.mean([rng.choice(group, (4000, len(group))).mean(axis=1) for group in groups], axis=0)
        blo, bhi = np.quantile(boot, [.0125, .9875])
        contrasts.append({"contrast": "cadc-"+policy, "mean": mean, "t_low": low, "t_high": high,
                          "bootstrap_low": blo, "bootstrap_high": bhi,
                          "primary": policy in ("sticky", fixed), "n_trajectories": len(difference)})
    contrast = pd.DataFrame(contrasts)
    contrast.to_csv(DOC / "paired_contrasts.csv", index=False)
    means = summary.groupby("policy")[["expected_objective", "ber", "bler", "switches", "switch_cost", "base_loss"]].mean()
    means.to_csv(DOC / "policy_means.csv")
    atlas = reference[reference.k == 16].groupby("family")[["estimated_opportunity", "reevaluated_opportunity", "optimism"]].mean()
    atlas.to_csv(DOC / "adaptation_atlas.csv")
    coverage = gain.groupby("family")[["covered", "wrong_sign", "radius", "error"]].mean()
    coverage.to_csv(DOC / "coverage_by_family.csv")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.facecolor": "white"})
    figure_dir = OUT / "report"
    figure_dir.mkdir(exist_ok=True)
    figure_names = []
    def save(name):
        plt.tight_layout()
        plt.savefig(figure_dir / f"{name}.png", dpi=170)
        plt.savefig(figure_dir / f"{name}.pdf")
        plt.close()
        figure_names.append(name)
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    precision_seeds = reference.loc[reference.k == 64, "seed"].unique()
    paired_precision = reference[reference.seed.isin(precision_seeds)]
    paired_precision.to_csv(DOC / "paired_precision.csv", index=False)
    for label, selected, color, offset in [("All 80 trajectories", reference[reference.k <= 16], "#24566d", -0.6),
                                          ("Same 4 precision trajectories", paired_precision, "#b97b45", .6)]:
        for k, group in selected.groupby("k"):
            by_seed = group.optimism.to_numpy()/120
            mean, lo, hi = interval(by_seed, .95)
            ax.errorbar(k+offset, mean, yerr=[[mean-lo], [hi-mean]], fmt="o", capsize=4,
                        color=color, label=label if k == 1 else None)
    ax.legend(frameon=False, fontsize=8)
    ax.axhline(0, color=".6", lw=.8)
    ax.set(xlabel="Selection replicas K", ylabel="Reevaluation − selection cost / epoch", xticks=[1, 16, 64])
    save("oracle-noise-gap")
    fig, ax = plt.subplots(figsize=(7, 3.5))
    for i, family in enumerate(FAMILIES):
        values = reference[(reference.family == family) & (reference.k == 16)].reevaluated_opportunity/120
        mean, lo, hi = interval(values, .95)
        ax.errorbar(i, mean, yerr=[[mean-lo], [hi-mean]], fmt="o", capsize=4, color="#24566d")
    ax.axhline(0, color=".5", lw=.8)
    ax.axhline(.002, color="#a06b35", lw=.8, ls="--")
    ax.set(xticks=range(4), xticklabels=list(FAMILIES), ylabel="Cross-evaluated fixed − dynamic / epoch")
    save("adaptation-value")
    fig, ax = plt.subplots(figsize=(7.2, 3.5))
    means[["base_loss", "switch_cost"]].plot.bar(stacked=True, ax=ax, color=["#24566d", "#b97b45"], rot=30)
    ax.set(xlabel="", ylabel="Realized cumulative objective")
    ax.legend(loc="upper left", ncol=2, frameon=False, bbox_to_anchor=(0, 1.15))
    save("loss-decomposition")
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    ax.bar(coverage.index, coverage.covered, color="#24566d")
    ax.axhline(.95, color="#a06b35", ls="--", lw=1)
    ax.set(ylim=(0, 1), ylabel="Empirical gain coverage")
    save("gain-coverage")
    fig, ax = plt.subplots(figsize=(6.8, 3.5))
    for policy, row in means.iterrows():
        ax.scatter(row.switches, row.expected_objective, color="#24566d")
        offset = (20, -3) if policy == "always_otfs" else ((20, 13) if policy == "always_afdm" else (3, 4))
        ax.annotate(policy, (row.switches, row.expected_objective), xytext=offset, textcoords="offset points", fontsize=8)
    ax.set(xlabel="Switches per 120 epochs", ylabel="Expected objective estimate / epoch")
    ax.set_ylim(means.expected_objective.min()-.004, means.expected_objective.max()+.004)
    save("switching-tradeoff")
    extension_rows = []
    for directory, manifest in completed("extensions"):
        settings = manifest["settings"]
        frame = pd.read_csv(directory / "summary.csv")
        extension_rows.append(frame.assign(**{k: settings[k] for k in ("family", "seed", "epochs", "noise", "delay", "csi_nmse")}))
    ext = pd.concat(extension_rows) if extension_rows else pd.DataFrame()
    ext.to_csv(DOC / "extensions.csv", index=False)
    if len(ext):
        for name, subset, x in [("noise-delay-factorial", ext[(ext.seed == 21001) & (ext.policy == "cadc")], "delay"),
                                ("CSI-sensitivity", ext[(ext.seed == 21002)], "csi_nmse")]:
            if subset.empty:
                continue
            fig, ax = plt.subplots(figsize=(6.2, 3.3))
            grouping = "noise" if x == "delay" else "policy"
            for label, group in subset.groupby(grouping):
                group = group.sort_values(x)
                ax.plot(group[x], group.expected_objective, "o-", ms=3, label=str(label))
            ax.set(xlabel="Delay (epochs)" if x == "delay" else "Receiver CSI NMSE", ylabel="Expected objective / epoch")
            ax.legend(fontsize=7, ncol=2, frameon=False, title="Context noise scale" if x == "delay" else None)
            save(name)
    budget = json.loads((OUT / "budget.json").read_text())
    reference_means = reference.groupby("k")[["optimism", "estimated_opportunity", "reevaluated_opportunity"]].mean()
    def table_md(frame):
        # Avoid adding optional tabulate dependency.
        data = frame.reset_index()
        return "| " + " | ".join(map(str, data.columns)) + " |\n|" + "---|"*len(data.columns) + "\n" + "\n".join(
            "| " + " | ".join(f"{x:.6f}" if isinstance(x, float) else str(x) for x in row) + " |" for row in data.itertuples(index=False, name=None))
    text = "# WaveForge-X R2 实验报告\n\n"
    text += f"冻结协议：`configs/research_v2/frozen.json`，SHA256 `{sha(CFG / 'frozen.json')}`。确认集80条独立轨迹、4家族各20、每条120 epoch；每epoch0.625ms。全部完成。累计预算计数（含pilot、校准、失败预留与扩展）{budget['phy_evaluations']:,} PHY评估；计时{budget['wall_seconds']:.1f}s。来源见 evidence_index.json、每run manifest、replicas.npz和trajectory.csv。\n\n"
    text += "## 已由数据支持\n\nR1逐行账目与动态规划复算一致。原组合方法没有超过固定OTFS，去掉变化检测的旧消融更好；这是指定配置的结论，不是OTFS普遍最优。\n\n"
    text += "独立重评将选路噪声与评价噪声分离；下表为累计量。K64只有预指定4条，不能与全80条直接作无配对效应比较。\n\n" + table_md(reference_means) + "\n\n"
    text += "同一损失表上的fixed−DP必非负；独立重评两条近似路径的差可为负。它不等于真正A_T，不能用零截断掩盖估计误差。有限K16的参考不是已知真期望。\n\n"
    text += "## 主比较及负结果\n\n差值为CADC减对照，负数更好；97.5%区间用于两个预声明主比较。其他行探索性。\n\n" + table_md(contrast.set_index("contrast")) + "\n\n"
    for _, row in contrast[contrast.primary].iterrows():
        status = "达到预声明工程改善阈值" if max(row.t_high, row.bootstrap_high) < -.002 else "未达到预声明工程改善阈值"
        text += f"- {row.contrast}：{status}；不据此声称新方法普遍胜出。\n"
    text += "\n" + table_md(means) + "\n\n代价分解严格为base_loss+switch_cost。PAPR、BLER、BER与复杂度已包含在base；switch_cost是人为惩罚，不是实测硬件时延。高切换方法可能BER相近而目标更差。探索/预测/reset消融不是互斥可加的因果解释。offline_historical使用旧预训练数据，只作描述，不是冷启动公平主比较。\n\n"
    text += "## 适应空间与校准\n\n" + table_md(atlas) + "\n\n" + table_md(coverage) + "\n\n"
    text += "CADC的Poisson ensemble是经验预测区间，校准轨迹只有4条且与最终门控后的动作分布不同；覆盖率针对有限MC估计的候选收益，不能证明真实条件均值95%覆盖。候选与当前相同的平凡零差不计覆盖。没有分布外覆盖或安全保证。默认24帧付费探测之后可能锁定不再探索；它在快速变化下可失效。\n\n"
    text += "## 探索性支持与未验证\n\nE5的600/2000轨迹和E8噪声×时延、CSI 0/0.01/0.1扩展见extensions.csv；这些是单轨迹配对机制例子，不能冒充20种子确认。零noise仍保留delay；CSI误差在共同tap空间生成、按时间域算子NMSE归一，再应用于三种波形，真传播H不变；不是实际pilot估计器。\n\n"
    text += "尚未验证：通用正适应区域、硬件重配置收益、实用payload反馈、真正分布外泛化、CADC无悔/永不差于固定策略保证、等计算复杂度接收机、MIMO/FEC/分数时延。未全面实现四种检测响应的PHY确认比较；旧hard-reset消融和toy诊断只支持机制分析。没有为寻找新方法赢家修改确认集或AFDM参数。\n\n"
    text += "## 复算\n\n先加载scripts/env.ps1，再执行 `.venv/Scripts/python.exe scripts/research_v2.py report`。完整顺序audit、pilot、calibrate、confirm、extensions、report。已完成artifact仅在源码/配置/RNG/参数key及文件哈希吻合时复用；旧R1 cache不会冒充expected oracle。源码发生变化会生成新run，不覆盖旧结果；预算跨进程累计。\n"
    (DOC / "final_report.md").write_text(text, encoding="utf-8")
    (DOC / "explain_to_owner_zh.md").write_text("# 怎么解释这项研究\n\n一次传输的赢家不一定是真正更好的方案：就像同水平选手各投一次球，挑中命中者不能证明他的命中率更高。我们用一组随机传输选择波形，再用另一组重新打分，避免拿同一次好运同时选人和颁奖。\n\n切换也要付钱。本实验的成本是公开的人为目标惩罚，不是已测得的硬件时间。CADC先用24帧付费了解三种方案，再问：未来16帧的预计优势是否足以覆盖切换成本？证据不足时保持原方案。这个判断可能错，误差区间只是经验校准，并非保证书。\n\n固定OTFS在旧场景表现好并不意味着永远最好；同样，减少切换也不意味着误码率必然降低。请同时看final_report.md中的误码、成本和区间。若切换相对固定策略的好处小到测不清，明智地不切换就是有价值的工程结论。我们保留不利结果，不把噪声造成的赢家轮换写成科研突破。\n", encoding="utf-8")
    html = "<!doctype html><html lang='zh'><meta charset='utf-8'><title>WaveForge-X R2</title><style>body{max-width:1100px;margin:48px auto;padding:0 24px;font:15px system-ui;color:#24323b}h1{font-size:28px}table{border-collapse:collapse;font-size:12px;width:100%}td,th{padding:8px;border-bottom:1px solid #ddd;text-align:right}img{width:100%;max-width:820px;display:block;margin:32px auto}a{color:#24566d}</style><h1>WaveForge-X R2</h1>"
    html += "<p>80 trajectories · 4 scenario families · 120 epochs · independent oracle reevaluation</p>"
    html += contrast.to_html(index=False, float_format=lambda v: f"{v:.6f}") + means.to_html(float_format=lambda v: f"{v:.6f}")
    html += "".join(f"<img src='{name}.png' alt='{name}'>" for name in figure_names)
    html += "<p><a href='../../../docs/research_v2/final_report.md'>研究报告</a></p></html>"
    (figure_dir / "index.html").write_text(html, encoding="utf-8")
    print(contrast.to_string(index=False))
    print(means.to_string())
