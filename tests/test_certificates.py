"""Necessary numerical and decision-geometry checks for R4."""

import itertools
from decimal import Decimal, localcontext

import numpy as np
import pytest

from waveforge6g.channels import ChannelRealization
from waveforge6g.channels.awgn import complex_awgn
from waveforge6g.core.modulation import demodulate, modulate
from waveforge6g.experiments.reference import role_seed
from waveforge6g.experiments.reference import make_channel
from waveforge6g.receivers.certiphy import (
    CertiPHY,
    StopRule,
    bit_margins,
    gray_energy_bound,
    physical_spectral_bounds,
    targeted_gray_bound,
)
from waveforge6g.waveforms import create_waveform


@pytest.mark.parametrize("modulation", ["bpsk", "qpsk", "qam16", "qam64"])
def test_bit_regions_and_exact_ties(modulation):
    values = np.array([0., .07+.21j, -1.1+.4j, 2.3-1.7j])
    margins = bit_margins(values, modulation)
    width = margins.shape[1]
    base = demodulate(values, modulation).reshape(-1, width)
    assert margins[0].min() == 0
    for shift in (.03, -.03, .03j, -.03j):
        changed = demodulate(values+shift, modulation).reshape(-1, width) != base
        assert not np.any(changed & (margins > abs(shift)))


def test_gray_packing_does_not_spend_the_same_energy_on_every_bit():
    values = np.full(10, .1+.1j)
    active = np.ones((10, 2), bool)
    assert (bit_margins(values, "qpsk") < .11).all()
    assert gray_energy_bound(values, "qpsk", .11, active, 1) == 1


def test_qam_packing_dominates_exhaustive_small_ball():
    values = np.array([.05+.17j, -.4+.02j])
    bits = demodulate(values, "qam16").reshape(2, 4)
    cuts = np.array([-np.inf, -2/np.sqrt(10), 0, 2/np.sqrt(10), np.inf])
    coordinates = np.r_[values.real, values.imag]
    for radius in (.02, .2, .5, 1.):
        maximum = 0
        for labels in itertools.product(range(4), repeat=4):
            distance = np.array([max(cuts[q]-v, v-cuts[q+1], 0) for q, v in zip(labels, coordinates, strict=True)])
            if distance @ distance <= radius**2:
                gray = np.array(labels) ^ (np.array(labels) >> 1)
                axis = ((gray[:, None] >> np.array([1, 0])) & 1)
                target = np.concatenate((axis[:2], axis[2:]), axis=1)
                maximum = max(maximum, int(np.count_nonzero(target != bits)))
        upper = gray_energy_bound(values, "qam16", radius, np.ones((2, 4), bool), 1)
        assert upper >= maximum


@pytest.mark.parametrize("name", ["ofdm", "otfs", "afdm"])
def test_bounds_and_output_against_independent_dense_reference(name):
    n = 32
    wave = create_waveform(name, n, 4, 8, c1=.017 if name == "afdm" else None)
    channel = ChannelRealization([0, 1, 3], [.8, .35j, -.17], [20, 1600, -701], 64000)
    receiver = CertiPHY(wave, channel, .03, "qam16")
    rng = np.random.default_rng(810)
    y = rng.normal(size=n)+1j*rng.normal(size=n)
    c = receiver.matrix.toarray()
    a = c.conj().T@c+.03*np.eye(n)
    exact = np.linalg.solve(a, c.conj().T@y)
    expected = demodulate(wave.analysis(exact), "qam16")
    alpha, beta = physical_spectral_bounds(channel, .03)
    eigenvalues = np.linalg.eigvalsh(a)
    assert alpha <= eigenvalues[0]+1e-13 and beta >= eigenvalues[-1]-1e-13
    for method in ("global", "krylov", "dual", "radau", "packing", "gray"):
        result = receiver.solve(y, StopRule(method=method, period=1, delta=.02), keep_trace=True)
        for row in result["trace"]:
            assert not np.any((row["output"] != expected) & row["certificate"])
            assert np.mean(row["output"] != expected) <= row["disagreement_bound"]+1e-15
        assert result["work"] == sum(result["work_parts"].values())
        assert result["floating_point_certified"] is False


def test_budget_exhaustion_returns_unknown_without_reference_fallback():
    wave = create_waveform("ofdm", 32, 4, 8)
    channel = ChannelRealization([0, 2], [1., .95j], [1500., -1220.], 64000)
    receiver = CertiPHY(wave, channel, 1e-4, "qpsk")
    full = receiver.solve(np.ones(32), StopRule(method="global", max_iterations=2))
    limited = receiver.solve(np.ones(32), StopRule(method="global"), max_work=full["work"])
    assert limited["work"] <= full["work"]
    assert limited["unknown_fraction"] > 0
    assert limited["status"] == "work_cap"
    assert len(limited["bits"]) == 64


def test_ritz_value_is_not_a_spectral_lower_bound():
    a = np.diag([.001, 1.])
    lanczos_start = np.array([0., 1.])
    ritz = lanczos_start @ a @ lanczos_start
    error = np.array([1., 0.])
    assert np.linalg.norm(a@error)/ritz < np.linalg.norm(error)


def test_decimal_near_boundary_reference_is_not_a_float_zero_ber_claim():
    # A physical single-path problem has A=|g|²+N0. Decimal preserves a received
    # component below normal double-scale additions; at zero the bit stays unknown.
    with localcontext() as context:
        context.prec = 70
        gain, noise = Decimal("0.7"), Decimal("0.01")
        y = Decimal("1e-40")
        exact = gain*y/(gain*gain+noise)
        assert exact > 0 and exact < Decimal("1e-38")
    wave = create_waveform("ofdm", 16, 2, 4)
    receiver = CertiPHY(wave, ChannelRealization([0], [.7], [0], 64000), .01, "bpsk")
    result = receiver.solve(np.zeros(16), StopRule(method="global", max_iterations=4, period=1))
    assert result["unknown_fraction"] == 1 and not result["met_requested_bound"]


def test_targeted_gray_bound_handles_shared_axis_bits():
    values = np.array([.05+.17j, -.4+.02j])
    active = np.ones((2, 4), bool)
    margins = bit_margins(values, "qam16")
    for radius in (.02, .2, .5, 1.):
        full = gray_energy_bound(values, "qam16", radius, active, 1)
        targeted, _ = targeted_gray_bound(values, "qam16", radius, active, 1, margins)
        assert targeted >= full


def test_hard_decision_plateau_can_end_in_a_different_reference_bit():
    wave = create_waveform("afdm", 128, 16, 16)
    channel = make_channel(128, "sparse", 62002)
    rng = np.random.default_rng(role_seed(62002, "r4_payload", 0, "qam16"))
    bits = rng.integers(0, 2, 512, dtype=np.uint8)
    noise = 10**(-2.4)
    y = (channel.apply(wave.modulate(modulate(bits, "qam16")))+complex_awgn(144, noise, rng))[16:]
    receiver = CertiPHY(wave, channel, noise, "qam16")
    result = receiver.solve(y, StopRule(method="stable", stable_checks=3, period=4, schedule="periodic"), keep_trace=True)
    c = receiver.matrix.toarray()
    exact = np.linalg.solve(c.conj().T@c+noise*np.eye(128), c.conj().T@y)
    expected = demodulate(wave.analysis(exact), "qam16")
    assert all(np.array_equal(row["output"], result["bits"]) for row in result["trace"][-4:])
    assert np.count_nonzero(result["bits"] != expected) == 1


def test_true_residual_drift_restarts_without_claiming_strict_float_proof():
    wave = create_waveform("ofdm", 32, 4, 8)
    channel = ChannelRealization([0, 1], [1/np.sqrt(2), -1/np.sqrt(2)], [0, 0], 64000)
    receiver = CertiPHY(wave, channel, 1e-12, "qpsk")
    result = receiver.solve(np.random.default_rng(77).normal(size=32),
                            StopRule(method="gray", period=2, max_iterations=64), keep_trace=True)
    assert result["restarts"] > 0 and result["unknown_fraction"] > 0
    assert max(row["recursive_drift"] for row in result["trace"]) > 0
    assert not result["floating_point_certified"]


def test_large_receiver_never_materializes_a_dense_matrix(monkeypatch):
    from scipy.sparse import csr_matrix

    def forbidden(*args, **kwargs):
        raise AssertionError("dense receiver path")

    monkeypatch.setattr(csr_matrix, "toarray", forbidden)
    wave = create_waveform("ofdm", 512, 64, 16)
    channel = ChannelRealization([0, 7], [1., .2j], [700, -82], 64000)
    result = CertiPHY(wave, channel, .1, "qam64").solve(np.ones(512), StopRule(method="gray", delta=.01, max_iterations=8))
    assert len(result["bits"]) == 3072


def test_small_complex_system_against_independent_decimal_arithmetic():
    from waveforge6g.experiments.precision import decimal_reference

    wave = create_waveform("afdm", 8, 2, 4, c1=.017)
    channel = ChannelRealization([0, 1, 2], [.8, .35j, -.17], [20, 1600, -701], 64000)
    receiver = CertiPHY(wave, channel, .003, "qam16")
    y = np.random.default_rng(89001).normal(size=8)+1j*np.random.default_rng(89002).normal(size=8)
    reference = decimal_reference(receiver.matrix.toarray(), y, .003)
    result = receiver.solve(y, StopRule(method="gray", period=1), keep_trace=True)
    assert np.linalg.norm(result["time_estimate"]-reference) < 1e-11
    assert np.array_equal(result["bits"], demodulate(wave.analysis(reference), "qam16"))


def test_fast_stability_removes_unneeded_residual_work():
    wave = create_waveform("ofdm", 32, 4, 8)
    channel = ChannelRealization([0, 2], [1., .3j], [500, -82], 64000)
    receiver = CertiPHY(wave, channel, .1, "qpsk")
    y = np.random.default_rng(3).normal(size=32)
    old = receiver.solve(y, StopRule(method="stable", period=1))
    fast = receiver.solve(y, StopRule(method="stable_fast", period=1))
    assert np.array_equal(old["bits"], fast["bits"])
    assert old["iterations"] == fast["iterations"]
    assert fast["work"] < old["work"]
