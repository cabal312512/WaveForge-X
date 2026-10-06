"""Necessary R3 physical, numerical, budget and information-boundary checks."""

import itertools

import numpy as np
import pytest

from waveforge6g.channels import ChannelRealization
from waveforge6g.channels.awgn import complex_awgn
from waveforge6g.core.modulation import demodulate, modulate
from waveforge6g.decision.advantage_dwell import AdvantageDwell, ExploreThenCommit
from waveforge6g.decision.base import DecisionContext
from waveforge6g.decision.objective import Objective
from waveforge6g.experiments.expected_loss import role_seed
from waveforge6g.experiments.research_v3 import evaluate_state
from waveforge6g.receivers.budgeted import (
    BudgetedReceiver,
    ReceiverAction,
    cost_model,
    sparsify,
    symbol_channel,
    time_channel,
)
from waveforge6g.receivers.detectors import detect, effective_channel_operator, materialize_operator
from waveforge6g.waveforms import create_waveform


@pytest.mark.parametrize("name", ["ofdm", "otfs", "afdm"])
def test_full_time_and_symbol_operators_match_original_propagation(name):
    wave = create_waveform(name, 32, 8, 8, c1=.017 if name == "afdm" else None)
    ch = ChannelRealization([0, 3, 7], [1, .4j, -.2], [135, -251, 78], 64000)
    time_matrix = time_channel(wave, ch)
    g = symbol_channel(wave, time_matrix)
    original = materialize_operator(effective_channel_operator(wave, ch))
    np.testing.assert_allclose(g, original, atol=1e-13)
    symbols = np.random.default_rng(3).normal(size=32)+1j
    np.testing.assert_allclose(time_matrix@wave.synthesis(symbols), ch.apply(wave.modulate(symbols))[8:], atol=1e-13)


def test_sparsity_keeps_row_top_energy_and_full_structure():
    matrix = np.random.default_rng(12).normal(size=(9, 9))+1j
    sparse, discarded = sparsify(matrix, 3)
    assert (np.diff(sparse.indptr) == 3).all()
    selected = np.sort(abs(matrix)**2, axis=1)[:, -3:].sum()
    assert np.sum(abs(sparse.data)**2) == pytest.approx(selected)
    assert discarded == pytest.approx(1-selected/np.sum(abs(matrix)**2))
    full, discarded = sparsify(matrix, 9)
    np.testing.assert_array_equal(full.toarray(), matrix)
    assert discarded == pytest.approx(0)


@pytest.mark.parametrize("domain", ["symbol", "time"])
def test_sufficient_iterations_recover_dense_reference(domain):
    wave = create_waveform("afdm", 32, 8, 8)
    ch = ChannelRealization([0, 2], [.8+.1j, .2j], [180, -52], 64000)
    rng = np.random.default_rng(2)
    y = rng.normal(size=32)+1j*rng.normal(size=32)
    receiver = BudgetedReceiver(wave, ch, .2, ReceiverAction("afdm", domain, 32, 128), tolerance=1e-11)
    result, diag = receiver.solve(y)
    reference = detect(wave.analysis(y), effective_channel_operator(wave, ch), .2, solver="dense")
    np.testing.assert_allclose(result, reference, atol=1e-9)
    assert diag["converged"]
    assert diag["iterations"] <= 128


def test_cost_includes_setup_and_caps_iterations():
    wave = create_waveform("ofdm", 128, 16, 16)
    ch = ChannelRealization([0, 3], [1, .5j], [500, -82], 64000)
    a = ReceiverAction("ofdm", "symbol", 4, 2)
    rx = BudgetedReceiver(wave, ch, .01, a)
    _, diag = rx.solve(np.ones(128, complex))
    assert diag["iterations"] <= 2
    costs = rx.cost
    assert costs["construction_work"] > 0 and costs["selection_work"] > 0 and costs["precondition_work"] > 0
    assert costs["total_work"] == pytest.approx(costs["preparation_work"]+costs["frame_work"])
    reused = cost_model(wave, 2, "symbol", 4, 2, reuse_frames=8)
    assert reused["total_work"] == pytest.approx(costs["preparation_work"]/8+costs["frame_work"])


def test_evaluation_uses_full_channel_not_truncated_receiver():
    n = 128
    wave = create_waveform("ofdm", n, 16, 16)
    ch = ChannelRealization([0, 7], [1, .9j], [800, -950], 64000)
    action = ReceiverAction("ofdm", "symbol", 4, 2)
    _, data, _ = evaluate_state(n, "qpsk", ch, 12, 123, 0, 2, [action])
    rng = np.random.default_rng(role_seed(123, "r3_validation", 0, 0))
    bits = rng.integers(0, 2, 256, dtype=np.uint8)
    symbols = modulate(bits, "qpsk")
    n0 = 10**(-1.2)
    noise = complex_awgn(144, n0, rng)
    full_y = (ch.apply(wave.modulate(symbols))+noise)[16:]
    receiver = BudgetedReceiver(wave, ch, n0, action)
    estimate, _ = receiver.solve(full_y)
    assert data["validation_errors"][0, 0] == np.count_nonzero(demodulate(estimate, "qpsk") != bits)
    assert receiver.discarded_energy > .05
    false_received = wave.synthesis(receiver.matrix@symbols)+noise[16:]
    assert np.linalg.norm(full_y-false_received) > 1


def test_probe_commit_is_empirical_for_all_six_orders():
    objective = Objective()
    for order in itertools.permutations(range(3)):
        for learner in (AdvantageDwell(objective, probe=2, probe_order=order), ExploreThenCommit(probe=2, probe_order=order)):
            previous = None
            for t in range(6):
                c = DecisionContext((1., 0., 0., 0., 0.), previous, t)
                action = learner.select(c).action
                learner.update(c, action, [-.1, -.3, -.5][action])
                previous = action
            assert learner.select(DecisionContext(c.features, previous, 6)).action == 0


def test_exact_ties_do_not_always_favor_an_action_number():
    choices = []
    for seed in range(90):
        learner = ExploreThenCommit(seed, probe=1)
        for t in range(3):
            c = DecisionContext((1.,), None, t)
            action = learner.select(c).action
            learner.update(c, action, -.2)
        choices.append(learner.select(DecisionContext((1.,), 2, 3)).action)
    assert set(choices) == {0, 1, 2}
    assert max(np.bincount(choices)) < 55


def test_budget_selector_selected_feedback_and_no_future_input():
    from waveforge6g.decision.compute_budget import BudgetSelector
    inv, target = np.tile(np.eye(9)[None], (3, 1, 1)), np.zeros((3, 9))
    first = BudgetSelector(inv, target, range(3), 81)
    second = BudgetSelector(inv, target, range(3), 81)
    x = np.ones(9)
    for t in range(12):
        a, _ = first.choose(x, np.array([2., 4., 8.]), 5., t)
        b, _ = second.choose(x, np.array([2., 4., 8.]), 5., t)
        assert a == b and a in (0, 1)
        before = first.target.copy()
        first.update(x, a, .1)
        second.update(x, b, .1)
        assert np.array_equal(first.target[2], before[2])
    assert first.choose(x, np.array([2., 4., 8.]), 1., 12)[0] is None
    assert np.array_equal(inv, np.tile(np.eye(9)[None], (3, 1, 1)))


def test_budget_overrides_minimum_dwell():
    from waveforge6g.decision.compute_budget import BudgetSelector
    selector = BudgetSelector(np.tile(np.eye(9)[None], (2, 1, 1)), np.zeros((2, 9)), range(2), 0)
    first, _ = selector.choose(np.ones(9), np.array([1., 1.]), 2, 0)
    costs = np.ones(2)
    costs[first] = 10
    following, _ = selector.choose(np.ones(9), costs, 2, 1)
    assert following != first


def test_online_feedback_is_independent_of_counterfactual_replica_budget():
    channel = ChannelRealization([0, 3], [1, .5j], [500, -82], 64000)
    action = ReceiverAction("ofdm", "time", 0, 2)
    _, small, _ = evaluate_state(128, "qpsk", channel, 8, 910, 0, 2, [action])
    _, larger, _ = evaluate_state(128, "qpsk", channel, 8, 910, 0, 5, [action])
    assert np.array_equal(small["online_errors"], larger["online_errors"])
    assert np.array_equal(small["validation_errors"], larger["validation_errors"][:2])
