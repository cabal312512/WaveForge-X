"""Distinct conditional, hindsight dynamic, and best-fixed comparators."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from .base import N_ACTIONS, validate_action
from .objective import Objective


@dataclass(frozen=True)
class SequenceComparator:
    """Offline comparator path and actual losses including its own switches."""

    actions: np.ndarray
    losses: np.ndarray
    total_loss: float


def instantaneous_regret(selected_loss: float, oracle_loss: float) -> float:
    """Nonnegative conditional regret when both losses share the same predecessor."""
    if not np.isfinite(selected_loss) or not np.isfinite(oracle_loss):
        raise ValueError("regret inputs must be finite")
    difference = float(selected_loss - oracle_loss)
    if difference < -1e-12:
        raise ValueError("conditional oracle cannot exceed selected loss; check shared predecessor")
    return max(0.0, difference)


def cumulative_regret(regrets: ArrayLike) -> np.ndarray:
    """Prefix sum of finite nonnegative conditional one-step regrets."""
    values = np.asarray(regrets, dtype=float)
    if values.ndim != 1 or not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError("conditional regrets must be a finite nonnegative vector")
    return np.cumsum(values)


def _loss_table(base_losses: ArrayLike) -> np.ndarray:
    losses = np.asarray(base_losses, dtype=float)
    if losses.ndim != 2 or losses.shape[1] != N_ACTIONS or not np.all(np.isfinite(losses)):
        raise ValueError("base_losses must have shape [time, 3] and finite values")
    if np.any((losses < 0) | (losses > 1)):
        raise ValueError("base_losses must lie in [0,1]")
    return losses


def sequence_losses(
    base_losses: ArrayLike, actions: ArrayLike, objective: Objective, initial_action: int | None = None
) -> np.ndarray:
    """Evaluate a path with its own predecessor at every time; initial choice is free by default."""
    table = _loss_table(base_losses)
    path = np.asarray(actions)
    if path.shape != (len(table),):
        raise ValueError("actions must be a vector matching trajectory length")
    if initial_action is not None:
        validate_action(initial_action)
    losses = np.empty(len(table), dtype=float)
    previous = initial_action
    for time, raw_action in enumerate(path):
        action = validate_action(raw_action)
        losses[time] = table[time, action] + objective.switch_cost(previous, action)
        previous = action
    return losses


def hindsight_optimal_sequence(
    base_losses: ArrayLike, objective: Objective, initial_action: int | None = None
) -> SequenceComparator:
    """Dynamic programming optimum over complete action sequences, O(T*A²).

    V_t(a)=base[t,a]+min_b(V_(t-1)(b)+switch_cost(b,a)). This comparator
    accesses future losses solely after the online run for evaluation. It is
    stronger than a greedy current-step oracle and pays its own switches.
    """
    table = _loss_table(base_losses)
    if initial_action is not None:
        validate_action(initial_action)
    if len(table) == 0:
        return SequenceComparator(np.empty(0, dtype=int), np.empty(0), 0.0)
    costs = np.array([[objective.switch_cost(before, after) for after in range(N_ACTIONS)]
                      for before in range(N_ACTIONS)])
    value = table[0] + np.array([objective.switch_cost(initial_action, action) for action in range(N_ACTIONS)])
    back = np.zeros((len(table), N_ACTIONS), dtype=int)
    for time in range(1, len(table)):
        alternatives = value[:, None] + costs
        back[time] = np.argmin(alternatives, axis=0)
        value = table[time] + np.min(alternatives, axis=0)
    path = np.empty(len(table), dtype=int)
    path[-1] = int(np.argmin(value))
    for time in range(len(table) - 1, 0, -1):
        path[time - 1] = back[time, path[time]]
    losses = sequence_losses(table, path, objective, initial_action)
    return SequenceComparator(path, losses, float(np.sum(losses)))


def best_fixed_action(
    base_losses: ArrayLike, objective: Objective, initial_action: int | None = None
) -> SequenceComparator:
    """Best single action used at every time; permits negative static regret."""
    table = _loss_table(base_losses)
    if initial_action is not None:
        validate_action(initial_action)
    totals = table.sum(axis=0)
    if len(table):
        totals += [objective.switch_cost(initial_action, action) for action in range(N_ACTIONS)]
    action = int(np.argmin(totals))
    path = np.full(len(table), action, dtype=int)
    losses = sequence_losses(table, path, objective, initial_action)
    return SequenceComparator(path, losses, float(np.sum(losses)))
