"""Dense and matrix-free ZF/LMMSE receivers with exact complex adjoints.

LSMR solves ||Hs-y||² + N0||s||² with damp=sqrt(N0), equivalent to the
unit-prior-variance LMMSE normal equations. No iterative result is accepted
when the solver reports iteration-limit or conditioning failure.

Reference: scipy.sparse.linalg.lsmr documentation,
https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.lsmr.html
"""

from typing import Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.sparse.linalg import LinearOperator, aslinearoperator, lsmr


class WaveformLike(Protocol):
    """Unitary useful-block waveform with a possibly phase-weighted prefix."""

    n_symbols: int
    cp_length: int
    prefix_phases: NDArray[np.complex128]

    def synthesis(self, symbols: ArrayLike) -> NDArray[np.complex128]: ...

    def analysis(self, signal: ArrayLike) -> NDArray[np.complex128]: ...

    def modulate(self, symbols: ArrayLike) -> NDArray[np.complex128]: ...

    def demodulate(self, signal: ArrayLike) -> NDArray[np.complex128]: ...


class ChannelLike(Protocol):
    """Finite-block linear time-domain channel with its true conjugate adjoint."""

    def apply(self, signal: ArrayLike) -> NDArray[np.complex128]: ...

    def adjoint(self, signal: ArrayLike) -> NDArray[np.complex128]: ...


def effective_channel_operator(waveform: WaveformLike, channel: ChannelLike) -> LinearOperator:
    """Return the [N,N] symbol-domain operator A=U* R H P U.

    U is useful-block synthesis, P prepends the waveform prefix, H is the
    causal finite-block channel, and R drops CP samples. The reverse operation
    is U* P* H* R* U. Critically, P* folds the conjugate-weighted prefix into
    the last useful samples; merely dropping CP again is not an adjoint.
    """
    n, cp = waveform.n_symbols, waveform.cp_length
    prefix_phases = np.asarray(waveform.prefix_phases, dtype=np.complex128)
    if not 0 <= cp <= n or prefix_phases.shape != (cp,):
        raise ValueError("Waveform prefix must have cp_length phases with 0 <= CP <= N")

    def matvec(symbols: NDArray[np.complex128]) -> NDArray[np.complex128]:
        received = channel.apply(waveform.modulate(np.asarray(symbols).reshape(n)))
        return waveform.analysis(received[cp:])

    def rmatvec(symbols: NDArray[np.complex128]) -> NDArray[np.complex128]:
        padded = np.zeros(n + cp, dtype=np.complex128)
        padded[cp:] = waveform.synthesis(np.asarray(symbols).reshape(n))
        channel_adjoint = channel.adjoint(padded)
        useful = np.asarray(channel_adjoint[cp:], dtype=np.complex128).copy()
        if cp:
            useful[-cp:] += prefix_phases.conj() * channel_adjoint[:cp]
        return waveform.analysis(useful)

    return LinearOperator((n, n), matvec=matvec, rmatvec=rmatvec, dtype=np.complex128)


def materialize_operator(
    operator: LinearOperator | NDArray[np.complex128], max_size: int = 1024
) -> NDArray[np.complex128]:
    """Build a dense [rows,columns] matrix only within the explicit size guard.

    Memory is 16*rows*columns bytes for complex128, plus one work vector;
    no dense identity matrix is allocated. Callers increasing max_size accept
    this memory cost explicitly. Dense solving needs additional work memory.
    """
    linear = aslinearoperator(operator)
    if max_size < 1 or max(linear.shape) > max_size:
        raise ValueError(f"Dense operator shape {linear.shape} exceeds max_size={max_size}")
    matrix = np.empty(linear.shape, dtype=np.complex128)
    basis = np.zeros(linear.shape[1], dtype=np.complex128)
    for column in range(linear.shape[1]):
        basis[column] = 1
        matrix[:, column] = linear.matvec(basis)
        basis[column] = 0
    return matrix


def detect(
    y: ArrayLike,
    channel: LinearOperator | NDArray[np.complex128],
    noise_variance: float,
    method: str = "lmmse",
    solver: str = "auto",
    dense_threshold: int = 128,
    rtol: float = 1e-8,
    maxiter: int | None = None,
) -> NDArray[np.complex128]:
    """Estimate unit-energy symbols from y=Hs+w using ZF or LMMSE.

    ``auto`` uses dense algebra if max(H.shape)<=dense_threshold, otherwise
    LSMR. Dense construction has a 1024-dimension guard. LSMR's default
    iteration budget is 4*min(H.shape); failure raises RuntimeError. ZF uses
    least squares (minimum-norm solution if a dense channel is rank deficient).
    Returned estimates are soft complex symbols, not hard decisions.
    """
    received = np.asarray(y, dtype=np.complex128)
    linear = aslinearoperator(channel)
    if received.shape != (linear.shape[0],) or not np.all(np.isfinite(received)):
        raise ValueError("y must be a finite vector matching channel's row dimension")
    if min(linear.shape) < 1:
        raise ValueError("channel dimensions must be positive")
    if not np.isfinite(noise_variance) or noise_variance < 0:
        raise ValueError("noise_variance must be finite and nonnegative")
    if method.lower() not in ("zf", "lmmse", "mmse"):
        raise ValueError("method must be 'zf' or 'lmmse'")
    if solver not in ("auto", "dense", "matrix_free", "iterative", "lsmr"):
        raise ValueError("solver must be 'auto', 'dense', 'iterative', 'matrix_free', or 'lsmr'")
    if not isinstance(dense_threshold, int) or not 1 <= dense_threshold <= 1024:
        raise ValueError("dense_threshold must be an integer in [1, 1024]")
    if not np.isfinite(rtol) or not 0 < rtol < 1:
        raise ValueError("rtol must lie in (0, 1)")
    if maxiter is not None and (not isinstance(maxiter, int) or maxiter < 1):
        raise ValueError("maxiter must be a positive integer or None")
    regularization = noise_variance if method.lower() in ("lmmse", "mmse") else 0.0
    use_dense = solver == "dense" or (solver == "auto" and max(linear.shape) <= dense_threshold)
    if use_dense:
        matrix = materialize_operator(linear)
        if not np.all(np.isfinite(matrix)):
            raise ValueError("channel matrix contains non-finite entries")
        if regularization == 0:
            return np.linalg.lstsq(matrix, received, rcond=None)[0]
        gram = matrix.conj().T @ matrix
        gram.flat[:: gram.shape[0] + 1] += regularization
        return np.linalg.solve(gram, matrix.conj().T @ received)
    result = lsmr(
        linear,
        received,
        damp=np.sqrt(regularization),
        atol=rtol,
        btol=rtol,
        maxiter=maxiter if maxiter is not None else 4 * min(linear.shape),
    )
    estimate, status, iterations = result[:3]
    if status not in (0, 1, 2, 4, 5) or not np.all(np.isfinite(estimate)):
        raise RuntimeError(
            f"LSMR failed to converge reliably (status={status}, iterations={iterations}); "
            "increase maxiter, relax rtol, use LMMSE regularization, or inspect conditioning"
        )
    return np.asarray(estimate, dtype=np.complex128)
