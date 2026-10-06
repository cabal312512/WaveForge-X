"""Check comparator mathematics against independent exhaustive paths."""

from itertools import product

import numpy as np
import pytest

from waveforge6g.decision.objective import Objective
from waveforge6g.decision.oracle import OracleSelector
from waveforge6g.decision.regret import (
    best_fixed_action,
    cumulative_regret,
    hindsight_optimal_sequence,
    instantaneous_regret,
    sequence_losses,
)


def test_conditional_oracle_uses_learner_predecessor_and_nonnegative_regret():
    objective = Objective({"switching_weight": 1})
    oracle = OracleSelector(objective)
    # OFDM has a slightly worse base loss; staying beats the transition.
    result = oracle.evaluate([0.21, 0.20, 0.4], previous=0)
    assert result.action == 0
    assert result.diagnostics["oracle_loss"] == 0.21
    for action, base in enumerate([0.21, 0.20, 0.4]):
        regret = instantaneous_regret(base + objective.switch_cost(0, action),
                                      result.diagnostics["oracle_loss"])
        assert regret >= 0
    assert oracle.evaluate([0.21, 0.20, 0.4], previous=None).action == 1
    np.testing.assert_allclose(cumulative_regret([0.1, 0, 0.2]), [0.1, 0.1, 0.3])
    with pytest.raises(ValueError, match="shared predecessor"):
        instantaneous_regret(0.1, 0.2)


@pytest.mark.parametrize("initial", [None, 0, 2])
def test_hindsight_dynamic_program_matches_exhaustive_asymmetric_sequences(initial):
    objective = Objective({"switching_matrix": [[0, 0.2, 1], [0.9, 0, 0.7], [0.4, 0.1, 0]],
                           "switching_weight": 0.7})
    table = np.array([[0.2, 0.3, 0.1], [0.4, 0.1, 0.2], [0.1, 0.5, 0.2], [0.6, 0.2, 0.1]])
    optimum = hindsight_optimal_sequence(table, objective, initial)
    exhaustive = []
    for path in product(range(3), repeat=len(table)):
        previous = initial
        total = 0.0
        for time, action in enumerate(path):
            total += table[time, action] + objective.switch_cost(previous, action)
            previous = action
        exhaustive.append(total)
    assert optimum.total_loss == pytest.approx(min(exhaustive), abs=1e-14)
    np.testing.assert_allclose(optimum.losses, sequence_losses(table, optimum.actions, objective, initial))
    assert optimum.total_loss <= best_fixed_action(table, objective, initial).total_loss + 1e-14


def test_greedy_conditional_oracle_is_not_hindsight_sequence_oracle():
    objective = Objective({"switching_weight": 0.5})
    table = np.array([[0.0, 0.1, 0.8], [0.8, 0.0, 0.8], [0.8, 0.0, 0.8]])
    greedy_path, previous = [], None
    for base in table:
        result = OracleSelector(objective).evaluate(base, previous)
        greedy_path.append(result.action)
        previous = result.action
    greedy_total = sequence_losses(table, greedy_path, objective).sum()
    optimum = hindsight_optimal_sequence(table, objective)
    assert optimum.total_loss < greedy_total
    np.testing.assert_array_equal(optimum.actions, [1, 1, 1])


def test_fixed_regret_can_be_negative_and_empty_paths_are_valid():
    objective = Objective({"switching_weight": 0})
    table = np.array([[0, 0.5, 0.8], [0.5, 0, 0.8]])
    selected = sequence_losses(table, [0, 1], objective).sum()
    fixed = best_fixed_action(table, objective)
    assert selected - fixed.total_loss < 0
    assert hindsight_optimal_sequence(np.empty((0, 3)), objective).total_loss == 0
    assert best_fixed_action(np.empty((0, 3)), objective).actions.size == 0
