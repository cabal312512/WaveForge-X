"""Six probe permutations on eight existing development trajectories; no R2 rerun."""

import itertools
import json

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

from ..decision.advantage_dwell import AdvantageDwell, ExploreThenCommit
from ..decision.base import DecisionContext
from ..decision.objective import Objective
from ..decision.regret import sequence_losses
from .research_v2_report import completed
from .research_v3 import DOC, ROOT, write_json


def run():
    records = []
    q = json.loads((ROOT/"configs/research_v2/frozen.json").read_text())["parameters"]["multiplier"]
    for directory, manifest in completed("pilot")+completed("calibrate"):
        table = pd.read_csv(directory/"trajectory.csv")
        observed = table[table.policy == "always_ofdm"].sort_values("time")
        data = np.load(directory/"replicas.npz")
        obj = Objective(manifest["settings"]["configuration"]["objective"])
        seed = manifest["settings"]["seed"]
        for order in itertools.permutations(range(3)):
            policies = {"cadc_fixed": AdvantageDwell(obj, seed, multiplier=q, probe_order=order),
                        "explore_then_commit": ExploreThenCommit(seed, probe_order=order)}
            for name, policy in policies.items():
                previous, recent, path = None, 0., []
                for t, (_, row) in enumerate(observed.iterrows()):
                    values = json.loads(row.context)[:5]
                    context = DecisionContext((*values, recent, *(float(previous == a) for a in range(3))), previous, t)
                    selected = policy.select(context).action
                    policy.update(context, selected, -float(data["online_loss"][t, 0, selected]))
                    recent = float(data["online_errors"][t, 0, selected]/data["bit_count"])
                    path.append(selected)
                    previous = selected
                values = sequence_losses(data["validation_loss"].mean(axis=1), path, obj)
                mean_ber = data["validation_errors"].mean(axis=1)/data["bit_count"]
                records.append({"seed": seed, "family": manifest["settings"]["family"], "order": "".join(map(str, order)),
                                "policy": name, "objective": values.mean(),
                                "ber": mean_ber[np.arange(len(path)), path].mean(),
                                "switches": np.count_nonzero(np.diff(path)), "commit_action": path[24],
                                "last_probe_action": order[-1], "afdm_after_probe": np.mean(np.array(path[24:]) == 2),
                                "source": str(directory)})
    frame = pd.DataFrame(records)
    frame.to_csv(DOC/"probe_permutations.csv", index=False)
    # Six orderings of one trajectory are not six independent observations.
    wide = frame.groupby(["seed", "policy"]).objective.mean().unstack()
    delta = wide.cadc_fixed-wide.explore_then_commit
    half = student_t.ppf(.975, len(delta)-1)*delta.std(ddof=1)/np.sqrt(len(delta))
    write_json(DOC/"probe_summary.json", {"trajectories": len(delta), "permutations": 6,
               "mean_cadc_minus_etc": float(delta.mean()), "paired95": [float(delta.mean()-half), float(delta.mean()+half)],
               "multiplier_unchanged_from_r2": q,
               "cadc_afdm_occupancy": float(frame[frame.policy == "cadc_fixed"].afdm_after_probe.mean()),
               "commits_matching_last_probe_fraction": float((frame.commit_action == frame.last_probe_action).mean()),
               "scope": "existing development data sensitivity, not new confirmation or independent 48 trajectories"})
    print(frame.groupby(["order", "policy"])[["objective", "ber", "switches", "afdm_after_probe"]].mean().to_string())
