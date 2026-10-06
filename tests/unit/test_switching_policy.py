"""Cost and uncertainty ablations preserve the same deployment objective."""

import numpy as np

from waveforge6g.decision.base import BaseSelector, DecisionContext, DecisionResult
from waveforge6g.decision.objective import Objective
from waveforge6g.decision.switching import SwitchingSelector


class Candidate(BaseSelector):
    def select(self, context):
        return DecisionResult(1, np.array([-0.2, -0.1, -0.5]), 0.1, 0.9)


def test_uncertainty_changes_threshold_at_fixed_predicted_gain():
    context = DecisionContext((1,), 0, 1)
    objective = Objective({"switching_weight": 0})
    sticky = SwitchingSelector(Candidate(), objective, policy="sticky", base_threshold=0.01)
    uncertain = SwitchingSelector(Candidate(), objective, policy="uncertainty", base_threshold=0.01, alpha=0.2)
    assert sticky.select(context).action == 1
    assert uncertain.select(context).action == 0
    assert uncertain.select(context).diagnostics["switching_threshold"] == 0.01 + 0.2 * 0.9


def test_cost_ablation_ignores_transition_in_policy_but_does_not_mutate_objective():
    context = DecisionContext((1,), 0, 1)
    objective = Objective({"switching_weight": 0.5})
    aware = SwitchingSelector(Candidate(), objective, policy="sticky", cost_aware=True)
    ignored = SwitchingSelector(Candidate(), objective, policy="sticky", cost_aware=False)
    assert aware.select(context).action == 0
    assert ignored.select(context).action == 1
    assert objective.switch_cost(0, 1) > 0.1


def test_reset_clears_threshold_rejection_state():
    policy = SwitchingSelector(Candidate(), Objective(), max_hold=2, alpha=1)
    context = DecisionContext((1,), 0, 1)
    assert policy.select(context).action == 0
    policy.reset()
    assert policy.select(context).action == 0
    assert policy.select(context).action == 1
