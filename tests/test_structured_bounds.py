"""Proof-obligation tests for local geometry, DP and operator lower bounds."""
import itertools
from decimal import Decimal, localcontext

import numpy as np
import pytest

from waveforge6g.channels import ChannelRealization
from waveforge6g.core.modulation import demodulate
from waveforge6g.experiments.reference import make_channel, reference
from waveforge6g.receivers.certiphy import CertiPHY, StopRule
from waveforge6g.receivers.certiphy_deep import (
    DeepRefiner,
    circulant_envelope,
    directed_sum,
    inverse_directions,
    refined_target,
    region_costs,
    saturated_dp,
)
from waveforge6g.waveforms import create_waveform


@pytest.mark.parametrize("mod", ["bpsk", "qpsk", "qam16", "qam64"])
def test_gray_dp_complete_enumeration(mod):
    values = np.array([.03+.27j, -.41+.11j])
    width = len(demodulate(values[:1], mod))
    costs = region_costs(values, mod, np.ones((2, width), bool))
    for target in range(1, 2*width+1):
        exact = min(sum(costs[i, h] for i, h in enumerate(choice))
                    for choice in itertools.product(range(costs.shape[1]), repeat=len(costs)) if sum(choice) >= target)
        lower = saturated_dp(costs, target)
        assert lower <= exact
        assert np.isclose(lower, exact)
        for energy in (max(0., exact*.99), exact*1.01):
            proved, info = refined_target(costs, energy, target)
            if proved:
                assert exact > energy
                if info["reason"] == "dual":
                    assert info["lower"] <= exact+1e-13


def test_directed_sum_decimal_cancellation_and_subnormals():
    values = np.array([1e16, 1., -1e16, np.nextafter(0., 1.)])
    with localcontext() as ctx:
        ctx.prec = 400
        truth = sum(Decimal.from_float(float(x)) for x in values)
        assert Decimal.from_float(directed_sum(values)) <= truth
        assert Decimal.from_float(directed_sum(values, True)) >= truth


@pytest.mark.parametrize("name", ["ofdm", "otfs", "afdm"])
def test_circulant_envelope_is_loewner_lower_for_actual_cpp(name):
    n = 32
    wave = create_waveform(name, n, 4, 8)
    channel = ChannelRealization(np.array([0, 2]), np.array([1., -.45]), np.array([.01, -.02]), 64000)
    receiver = CertiPHY(wave, channel, .001, "qam16")
    weights, _ = circulant_envelope(receiver.matrix, receiver.noise)
    if weights is not None:
        f = np.fft.fft(np.eye(n), axis=0, norm="ortho")
        m = f.conj().T @ np.diag(weights) @ f
        c = receiver.matrix.toarray()
        assert np.linalg.eigvalsh(c.conj().T @ c+.001*np.eye(n)-m).min() >= -1e-11
        assert np.linalg.eigvalsh(m).min() > 0


def test_path_deletion_and_diagonal_are_not_lower_bounds():
    c = np.eye(2)-.9*np.eye(2)
    assert np.linalg.eigvalsh(c.T@c-np.eye(2)).min() < 0
    a = np.array([[1., .9], [.9, 1.]])
    assert np.linalg.eigvalsh(a-np.diag(np.diag(a))).min() < 0


@pytest.mark.parametrize("name", ["ofdm", "otfs", "afdm"])
def test_fast_inverse_directions_match_dense_evaluator(name):
    wave = create_waveform(name, 32, 4, 8)
    if name == "afdm":
        from waveforge6g.waveforms.afdm import AFDM
        wave = AFDM(32, 4, c1=.013, c2=.003)
    weights = np.exp(np.linspace(-3, 3, 32))
    transform = np.stack([np.fft.fft(wave.synthesis(v), norm="ortho") for v in np.eye(32)], axis=1)
    truth = (abs(transform)**2/weights[:, None]).sum(axis=0)
    result = inverse_directions(wave, weights)
    assert np.allclose(result, truth, rtol=1e-11)
    assert np.all(result >= truth-1e-13)


@pytest.mark.parametrize("mode", ["envelope", "dp", "target"])
def test_anytime_refines_same_output_without_truth(mode):
    wave = create_waveform("ofdm", 32, 4, 8)
    channel = make_channel(32, "dominant", 901)
    rng = np.random.default_rng(983)
    y = rng.normal(size=32)+1j*rng.normal(size=32)
    ref = reference(wave, channel, y, .01)
    bits = demodulate(ref["symbols"], "qam16")
    result = CertiPHY(wave, channel, .01, "qam16").solve(
        y, StopRule(method="gray", delta=.05), max_work=8e6,
        refiner=DeepRefiner(mode), keep_trace=True)
    for trace in result["trace"]:
        assert np.mean(trace["output"] != bits) <= trace["disagreement_bound"]+1e-14
    assert result["work"] <= 8e6
    assert not result["floating_point_certified"]


def test_ties_and_unknown_cap_do_not_prove_exclusion():
    values = np.array([0.+0j])
    costs = region_costs(values, "qam64", np.ones((1, 6), bool))
    assert saturated_dp(costs, 1) == 0
    success, _ = refined_target(costs, 0., 1)
    assert not success
    success, info = refined_target(np.tile([0., 1., 1.1], (20, 1)), 12., 15, max_cells=0, exact_only=True)
    assert not success and info["reason"] == "refinement_cap"


def test_reduced_cost_filter_against_independent_complete_choices():
    rng = np.random.default_rng(97101)
    for _ in range(12):
        costs = np.c_[np.zeros(4), rng.uniform(.001, 2, (4, 3))]
        for target in (1, 3, 7, 11):
            optimum = min(sum(costs[i,h] for i,h in enumerate(choice))
                          for choice in itertools.product(range(4), repeat=4) if sum(choice) >= target)
            for factor in (.9, 1., 1.1):
                energy = optimum*factor
                proved, _ = refined_target(costs, energy, target)
                if proved:
                    assert optimum > energy


def test_near_slicer_boundaries_have_zero_lower_flip_cost():
    scale = float(np.sqrt(42))
    values = np.array([np.nextafter(2/scale, -np.inf), 2/scale, np.nextafter(2/scale, np.inf)], complex)
    costs = region_costs(values, "qam64", np.ones((3,6), bool))
    assert np.all(costs[:3, 1] == 0.)


def test_dense_free_refinement_and_no_free_budget_fallback(monkeypatch):
    from scipy.sparse import csr_matrix
    def forbidden(*args, **kwargs):
        raise AssertionError("dense conversion entered deployed receiver")
    monkeypatch.setattr(csr_matrix, "toarray", forbidden)
    wave = create_waveform("ofdm", 128, 16, 16)
    channel = ChannelRealization([0,1], [1., -.5], [.01, -.01], 64000)
    receiver = CertiPHY(wave, channel, .0001, "qam64")
    result = receiver.solve(np.ones(128), StopRule(method="gray"), max_work=160000,
                            refiner=DeepRefiner("spectral"))
    assert result["work"] <= 160000
    assert result["status"] == "work_cap"
    assert not result["met_requested_bound"]
