"""Independent complex-adjoint identities and dense/iterative equivalence."""

import numpy as np
import pytest
from scipy.sparse.linalg import LinearOperator, aslinearoperator

from waveforge6g.channels.doubly_selective import ChannelRealization
from waveforge6g.receivers.channel_estimation import pilot_ls
from waveforge6g.receivers.detectors import detect, effective_channel_operator, materialize_operator
from waveforge6g.receivers.equalizers import one_tap_mmse, one_tap_zf
from waveforge6g.waveforms.afdm import AFDM
from waveforge6g.waveforms.ofdm import OFDM
from waveforge6g.waveforms.otfs import OTFS


@pytest.mark.parametrize("waveform", [OFDM(32, 8), OTFS(32, 8, 8), AFDM(32, 8, 0.0129, 0.0031)])
def test_effective_channel_has_true_conjugate_adjoint(waveform):
    rng = np.random.default_rng(89)
    channel = ChannelRealization([0, 2, 5], [1, 0.2+0.3j, -0.1j], [25, -78, 120], 1000)
    operator = effective_channel_operator(waveform, channel)
    x = rng.normal(size=32) + 1j * rng.normal(size=32)
    y = rng.normal(size=32) + 1j * rng.normal(size=32)
    np.testing.assert_allclose(np.vdot(y, operator @ x), np.vdot(operator.H @ y, x), atol=1e-12)
    matrix = materialize_operator(operator)
    np.testing.assert_allclose(operator.H @ y, matrix.conj().T @ y, atol=1e-12)


@pytest.mark.parametrize("method", ["zf", "lmmse"])
def test_complex_dense_and_matrix_free_detection_agree(method):
    rng = np.random.default_rng(63)
    matrix = (rng.normal(size=(30, 24)) + 1j * rng.normal(size=(30, 24))) / np.sqrt(24)
    y = rng.normal(size=30) + 1j * rng.normal(size=30)
    dense = detect(y, matrix, 0.2, method=method, solver="dense")
    iterative = detect(y, aslinearoperator(matrix), 0.2, method=method, solver="iterative", rtol=1e-12)
    np.testing.assert_allclose(iterative, dense, rtol=1e-8, atol=1e-8)
    residual = matrix.conj().T @ (matrix @ dense - y)
    if method == "lmmse":
        residual += 0.2 * dense
    np.testing.assert_allclose(residual, 0, atol=1e-12)


def test_iterative_failure_and_dense_allocation_guard_are_visible():
    matrix = np.diag(np.linspace(0.1, 2, 32))
    with pytest.raises(RuntimeError, match="failed to converge"):
        detect(np.ones(32), matrix, 0.01, solver="iterative", rtol=1e-14, maxiter=1)
    huge_identity = LinearOperator((10_000, 10_000), matvec=lambda x: x, rmatvec=lambda x: x)
    with pytest.raises(ValueError, match="exceeds"):
        materialize_operator(huge_identity)


def test_one_tap_matches_dense_and_null_conventions():
    y = np.array([1+2j, -0.7j, 0.4])
    h = np.array([1-1j, -0.4+0.3j, 2])
    np.testing.assert_allclose(one_tap_zf(y, h), detect(y, np.diag(h), 0, "zf"))
    np.testing.assert_allclose(one_tap_mmse(y, h, 0.2), detect(y, np.diag(h), 0.2))
    with pytest.raises(np.linalg.LinAlgError):
        one_tap_zf([1], [0])
    np.testing.assert_array_equal(one_tap_mmse([1], [0], 0), [0])


def test_pilot_ls_primitive():
    pilots = np.array([1+1j, 1-1j, -1+1j]) / np.sqrt(2)
    gains = np.array([1, 2j, 0.3-0.1j])
    np.testing.assert_allclose(pilot_ls(pilots * gains, pilots), gains)
    with pytest.raises(ValueError, match="nonzero"):
        pilot_ls([1], [0])
