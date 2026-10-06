"""Auditable causal replay on exogenous cached PHY: original sticky and warm-start checks."""

import json

import numpy as np
import pandas as pd

from waveforge6g.decision import create_selector
from waveforge6g.decision.advantage_dwell import BlockLinUCB
from waveforge6g.decision.base import DecisionContext
from waveforge6g.decision.objective import Objective
from waveforge6g.decision.regret import sequence_losses
from waveforge6g.experiments.expected_loss import role_seed
from waveforge6g.experiments.research_v2 import DOC, sha, write_json
from waveforge6g.experiments.research_v2_report import completed


def replay(directory, manifest, policy):
    settings = manifest["settings"]
    table = pd.read_csv(directory / "trajectory.csv")
    observed = table[table.policy == "always_ofdm"].sort_values("time")
    data = np.load(directory / "replicas.npz")
    previous, recent, actions = None, 0., []
    for t, (_, row) in enumerate(observed.iterrows()):
        context_features = json.loads(row.context)[:5]
        context = DecisionContext((*context_features, recent, *(float(previous == a) for a in range(3))), previous, t)
        policy.observe(context)
        selected = policy.select(context).action
        # Feedback retrieval occurs after commitment and uses only the selected arm.
        policy.update(context, selected, -float(data["online_loss"][t, 0, selected]))
        recent = float(data["online_errors"][t, 0, selected] / data["bit_count"])
        previous = selected
        actions.append(selected)
    objective = Objective(settings["configuration"]["objective"])
    costs = sequence_losses(data["validation_loss"].mean(axis=1), actions, objective)
    return np.array(actions), {"expected_objective": costs.mean(), "switches": np.count_nonzero(np.diff(actions))}


def main():
    results, audits, provenance = [], [], []
    for directory, manifest in completed("confirm"):
        settings = manifest["settings"]
        obj = Objective(settings["configuration"]["objective"])
        seed = role_seed(settings["seed"], "policy", "sticky")
        original_spec = {"type": "linucb", "alpha": .4, "policy": "sticky", "base_threshold": .01, "max_hold": 15}
        reproduced, _ = replay(directory, manifest, create_selector(original_spec, seed, obj))
        recorded = pd.read_csv(directory / "trajectory.csv")
        expected = recorded[recorded.policy == "sticky"].sort_values("time").action.to_numpy()
        np.testing.assert_array_equal(reproduced, expected)
        for name, policy in [("sticky_legacy_hold20", create_selector({**original_spec, "max_hold": 20}, seed, obj)),
                             ("block16_exploratory", BlockLinUCB(16))]:
            _, row = replay(directory, manifest, policy)
            results.append({"seed": settings["seed"], "family": settings["family"], "policy": name, **row})
        for _, row in recorded.iterrows():
            diagnostics = json.loads(row.diagnostics)
            proposed = diagnostics.get("proposed_action", diagnostics.get("candidate_action", row.action))
            audits.append({"seed": settings["seed"], "family": settings["family"], "time": row.time,
                           "policy": row.policy, "proposed_action": proposed, "executed_action": row.action,
                           "gate_reason": diagnostics.get("gate_reason", "suppressed" if diagnostics.get("switch_suppressed") else "accepted"),
                           "exploration": bool(diagnostics.get("exploration_flag", diagnostics.get("forced_exploration", False))),
                           "reset": bool(diagnostics.get("detected_change", False)),
                           "feedback": -row.base_loss})
        provenance.append({"run": str(directory), "sha": sha(directory / "replicas.npz")})
    pd.DataFrame(results).to_csv(DOC / "replayed_baselines.csv", index=False)
    # Per-step canonical proposal is extracted from original diagnostics, not guessed from execution.
    pd.DataFrame(audits).to_csv(DOC / "decision_audit.csv", index=False)
    write_json(DOC / "replay_manifest.json", {"runs": provenance,
               "verification": "replayed sticky hold15 action sequence exactly matches every original trajectory",
               "scope": "exploratory historical hold20 and block16; does not replace frozen primary comparisons",
               "assumption": "external channel and independent additive switching cost; no action-dependent hidden state"})


if __name__ == "__main__":
    main()
