"""Remove unnecessary full residual checks from the hard-decision heuristic.

Only this strengthened baseline is re-evaluated on saved inputs; no CertiPHY
parameter or observation is changed. Preserve the original evidence separately.
"""

import json
import zipfile
from dataclasses import asdict

import numpy as np
import pandas as pd

from waveforge6g.channels import ChannelRealization
from waveforge6g.core.modulation import demodulate
from waveforge6g.experiments.research_cache import implementation_hash
from waveforge6g.experiments.research_v2 import ROOT, Budget, sha, write_json
from waveforge6g.experiments.research_v4 import CFG, DOC, OUT
from waveforge6g.receivers.certiphy import CertiPHY, StopRule
from waveforge6g.waveforms import create_waveform

budget = Budget(output_root=OUT)
destination = OUT/"stability_refinement"
destination.mkdir(exist_ok=True)
if (destination/"manifest.json").exists():
    raise RuntimeError("strengthened baseline already measured")
rules = {f"s{s}p{p}": StopRule(method="stable_fast", stable_checks=s, period=p, schedule="periodic")
         for s, p in ((2, 1), (3, 1), (5, 1), (3, 2), (3, 4), (8, 2), (12, 2))}
development = pd.read_csv(DOC/"develop.csv")
calibration = []
for _, row in development.drop_duplicates(["seed", "waveform", "modulation"]).iterrows():
    raw = np.load(OUT/"develop"/f"{row.seed}_{row.waveform}_{row.modulation}"/"raw.npz")
    wave = create_waveform(row.waveform, int(row.n), int(row.n)//8, 16)
    channel = ChannelRealization.from_dict(json.loads(str(raw["estimated_channel_json"])))
    target = demodulate(raw["reference_symbols"], row.modulation)
    budget.reserve(len(rules))
    for name, rule in rules.items():
        result = CertiPHY(wave, channel, 10**(-row.snr/10), row.modulation).solve(raw["received"], rule, max_work=8e6*row.n/128)
        calibration.append(dict(rule=name, seed=row.seed, waveform=row.waveform, modulation=row.modulation,
                                work=result["work"], disagreement=float(np.mean(result["bits"] != target))))
pd.DataFrame(calibration).to_csv(DOC/"stability_calibration.csv", index=False)
grouped = pd.DataFrame(calibration).groupby("rule").agg(work=("work", "mean"), worst=("disagreement", "max"))
chosen = {}
for label, delta in (("d0", 0.), ("d001", .01), ("d005", .05)):
    feasible = grouped[grouped.worst <= delta]
    selected = feasible.sort_values("work").index[0] if len(feasible) else grouped.sort_values(["worst", "work"]).index[0]
    spec = asdict(rules[selected])
    spec["delta"] = delta
    chosen["stable_"+label] = StopRule(**spec)
source_hash = implementation_hash()
protocol = {"source_hash": source_hash, "rules": {name: asdict(rule) for name, rule in chosen.items()},
            "calibration_sha256": sha(DOC/"stability_calibration.csv"),
            "reason": "Hard-decision stability needs transform and decisions, not a full residual matvec at every check.",
            "no_certiphy_retuning": True, "reuse_original_received_samples": True}
write_json(CFG/"stability_baseline_protocol.json", protocol)
with zipfile.ZipFile(OUT/"sources"/(source_hash+".zip"), "w", zipfile.ZIP_DEFLATED) as archive:
    for path in (ROOT/"src/waveforge6g").rglob("*.py"):
        archive.write(path, path.relative_to(ROOT))

rows, outputs = [], {}
for stage in ("core", "scale", "mismatch"):
    for manifest_path in sorted((OUT/stage).glob("*/manifest.json")):
        manifest = json.loads(manifest_path.read_text())
        original = pd.read_csv(manifest_path.parent/"frames.csv")
        for saved in original[original.method.str.startswith("stable_")].itertuples(index=False):
            raw = np.load(manifest_path.parent/f"frame_{saved.epoch}_nmse{saved.nmse:g}.npz")
            wave = create_waveform(saved.waveform, saved.n, saved.n//8, 16)
            channel = ChannelRealization.from_dict(json.loads(str(raw["estimated_channel_json"])))
            budget.reserve(1)
            result = CertiPHY(wave, channel, 10**(-saved.snr/10), saved.modulation).solve(raw["received"], chosen[saved.method],
                                                                                       max_work=8e6*saved.n/128)
            target = demodulate(raw["reference_symbols"], saved.modulation)
            row = saved._asdict()
            row.update(stage=stage, implementation="stable_fast", ber=float(np.mean(result["bits"] != raw["truth"])),
                       disagreement=float(np.mean(result["bits"] != target)), iterations=result["iterations"],
                       checks=result["checks"], restarts=result["restarts"], work=result["work"],
                       elapsed_s=result["elapsed_s"], process_cpu_s=result["process_cpu_s"],
                       status=result["status"], work_parts=json.dumps(result["work_parts"], sort_keys=True))
            rows.append(row)
            outputs[f"{stage}_{saved.seed}_{saved.epoch}_{saved.nmse:g}_{saved.method}"] = result["bits"]
        print("strengthened stability", stage, manifest["seed"], flush=True)
pd.DataFrame(rows).to_csv(destination/"frames.csv", index=False)
np.savez_compressed(destination/"outputs.npz", **outputs)
write_json(destination/"manifest.json", {"source_hash": source_hash, "protocol_sha256": sha(CFG/"stability_baseline_protocol.json"),
           "rows": len(rows), "files": {"frames.csv": sha(destination/"frames.csv"), "outputs.npz": sha(destination/"outputs.npz")}})
budget.save()
