"""Independent physical reference, information isolation, and comparator invariants."""

import itertools

import numpy as np
import pytest

from waveforge6g.channels import ChannelRealization
from waveforge6g.channels.awgn import complex_awgn, noise_variance
from waveforge6g.core.modulation import demodulate, modulate
from waveforge6g.decision.advantage_dwell import AdvantageDwell, BlockLinUCB
from waveforge6g.decision.base import DecisionContext
from waveforge6g.decision.objective import Objective
from waveforge6g.decision.regret import hindsight_optimal_sequence, sequence_losses
from waveforge6g.experiments.expected_loss import PreparedPHY, role_seed
from waveforge6g.experiments.research_v2 import configuration, simulate
from waveforge6g.receivers.detectors import detect, effective_channel_operator
from waveforge6g.waveforms import create_waveform


def test_replicas_match_independent_original_link():
    config = configuration("fast_change", 8)
    ch = ChannelRealization([0, 2], [0.7 + .2j, .3 - .1j], [123, -410], 64000)
    prepared = PreparedPHY(config, ch, 8)
    actual = prepared.evaluate(33, 2, "online", 4)
    rng = np.random.default_rng(role_seed(33, "online", 2, 0))
    bits = rng.integers(0, 2, 64, dtype=np.uint8)
    symbols = modulate(bits, "qpsk")
    noise = complex_awgn(40, noise_variance(8), rng)
    for a, name in enumerate(("ofdm", "otfs", "afdm")):
        wave = create_waveform(name, 32, 8, 8)
        estimate = detect(wave.demodulate(ch.apply(wave.modulate(symbols)) + noise),
                          effective_channel_operator(wave, ch), noise_variance(8), solver="dense")
        assert actual["errors"][0, a] == np.count_nonzero(demodulate(estimate, "qpsk") != bits)
        np.testing.assert_allclose(actual["energy"][0, a], np.sum(abs(wave.modulate(symbols))**2))
    smaller = prepared.evaluate(33, 2, "online", 2)
    np.testing.assert_array_equal(actual["errors"][:2], smaller["errors"])
    np.testing.assert_allclose(actual["loss"][:2], smaller["loss"], atol=1e-14)


def test_csi_zero_and_operator_nmse():
    cfg = configuration("near_tie", 8)
    ch = ChannelRealization([0, 2], [1, .3j], [0, 52], 64000)
    first = PreparedPHY(cfg, ch, 12)
    zero = PreparedPHY(cfg, ch, 12, 0, 454)
    np.testing.assert_array_equal(first.evaluate(1, 1, "x", 2)["loss"], zero.evaluate(1, 1, "x", 2)["loss"])
    assert PreparedPHY(cfg, ch, 12, .1, 454).nmse == pytest.approx(.1)
    with pytest.raises(ValueError, match="near-zero"):
        PreparedPHY(cfg, ChannelRealization([0], [0], [0], 64000), 12, .1)


def test_unitary_spectrum_and_lmmse_trace_not_ber_claim():
    rng = np.random.default_rng(90)
    h = rng.normal(size=(8, 8)) + 1j * rng.normal(size=(8, 8))
    u, _ = np.linalg.qr(rng.normal(size=(8, 8)) + 1j * rng.normal(size=(8, 8)))
    v, _ = np.linalg.qr(rng.normal(size=(8, 8)) + 1j * rng.normal(size=(8, 8)))
    transformed = v.conj().T @ h @ u
    np.testing.assert_allclose(np.linalg.svd(h, compute_uv=False), np.linalg.svd(transformed, compute_uv=False))
    def error(a):
        return np.trace(np.linalg.inv(np.eye(8) + a.conj().T @ a / .2)).real
    assert error(h) == pytest.approx(error(transformed))


def test_dp_against_exhaustive_and_equal_means_optimism():
    obj = Objective()
    loss = np.random.default_rng(2).uniform(.1, .8, (5, 3))
    optimum = min(sequence_losses(loss, path, obj).sum() for path in itertools.product(range(3), repeat=5))
    assert hindsight_optimal_sequence(loss, obj).total_loss == pytest.approx(optimum)
    rng = np.random.default_rng(41)
    noisy = rng.binomial(1, .5, (500, 3)).astype(float)
    selected = hindsight_optimal_sequence(noisy, obj)
    expectation = np.full_like(noisy, .5)
    assert sequence_losses(expectation, selected.actions, obj).sum() > selected.total_loss + 50


def test_evaluator_and_policy_order_do_not_change_online_actions():
    cfg = configuration("fast_change", 8)
    a, _, _ = simulate(cfg, 997, {}, 2, 2)
    b, _, _ = simulate(cfg, 997, {}, 4, 4, evaluator_seed=998, reverse_policies=True)
    for key in ("action", "base_loss", "ber", "context"):
        np.testing.assert_array_equal(a.sort_values(["policy", "time"])[key],
                                      b.sort_values(["policy", "time"])[key])


def test_cadc_only_updates_executed_action_and_block_delays_feedback():
    obj = Objective()
    context = DecisionContext((1., .4, .1, .2, 0., 0., 1., 0., 0.), 0, 0)
    gate = AdvantageDwell(obj)
    old = gate.gram.copy()
    gate.update(context, 2, -.2)
    np.testing.assert_array_equal(gate.gram[:, :2], old[:, :2])
    assert np.any(gate.gram[:, 2] != old[:, 2])
    block = BlockLinUCB(2)
    block.select(context)
    block.update(context, 0, -.2)
    assert block.base.counts.sum() == 0
    block.select(DecisionContext(context.features, 0, 2))
    assert block.base.counts.sum() == 1


@pytest.mark.parametrize("options", [{"horizon": 1.5}, {"probe": -1}, {"multiplier": float("nan")}, {"rho": float("inf")}])
def test_cadc_invalid_parameters(options):
    with pytest.raises(ValueError):
        AdvantageDwell(Objective(), **options)


def test_zero_context_noise_does_not_remove_latency():
    cfg = configuration("slow_change", 8, noise=0, delay=3)
    table, _, _ = simulate(cfg, 901, {}, 2, 2)
    assert table.loc[table.time == 6, "source_time"].eq(3).all()
    assert cfg["observation"]["perfect"] is False


def test_hidden_counterfactual_outcomes_do_not_reach_cadc(monkeypatch):
    import waveforge6g.experiments.research_v2 as runner
    cfg = configuration("fast_change", 8)
    executed = {}
    class TracedGate(AdvantageDwell):
        def select(self, context):
            result = super().select(context)
            executed[context.time] = result.action
            return result
    monkeypatch.setattr(runner, "policies", lambda config, seed, parameters, ablations:
                        {"cadc": TracedGate(Objective(config["objective"]), seed)})
    a, _, _ = simulate(cfg, 991, {}, 2, 2)
    original = PreparedPHY.evaluate
    def modified(self, seed, epoch, role, replicas):
        output = original(self, seed, epoch, role, replicas)
        if role == "online":
            for action in range(3):
                if action != executed[epoch]:
                    output["loss"][:, action] = .99
                    output["errors"][:, action] = output["bits"]
        return output
    monkeypatch.setattr(PreparedPHY, "evaluate", modified)
    b, _, _ = simulate(cfg, 991, {}, 2, 2)
    for key in ("action", "selected_feedback", "context"):
        np.testing.assert_array_equal(a[key], b[key])


def test_future_channel_controls_do_not_change_past_decisions():
    from waveforge6g.config_dynamic import parse_dynamic_config
    raw = configuration("near_tie", 8).to_dict()
    raw["dynamic"]["trajectories"] = {"snr_db": {"type": "piecewise_constant", "points": [[0, 12], [5, -5]]}}
    a, _, _ = simulate(parse_dynamic_config(raw), 998, {}, 2, 2)
    raw["dynamic"]["trajectories"]["snr_db"]["points"][1][1] = 25
    b, _, _ = simulate(parse_dynamic_config(raw), 998, {}, 2, 2)
    for key in ("action", "selected_feedback", "context"):
        np.testing.assert_array_equal(a[a.time < 5][key], b[b.time < 5][key])
