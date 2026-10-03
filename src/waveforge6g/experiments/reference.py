import hashlib
import json
import numpy as np
from scipy.linalg import cho_factor,cho_solve
from scipy.sparse.linalg import LinearOperator,cg
from ..channels import ChannelRealization
from ..receivers.budgeted import time_channel
from ..receivers.certiphy import physical_spectral_bounds

def role_seed(seed: int, role: str, *keys: object) -> int:
    """Stable, order-independent streams, including across Python processes."""
    raw = json.dumps([int(seed), role, *keys], separators=(",", ":")).encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "little")

def make_channel(n, condition, seed):
    rng = np.random.default_rng(role_seed(seed, "r4_channel", n, condition))
    p = 8 if condition == "rich" else 3
    delays = np.sort(rng.choice(np.arange(n//8-1), p, replace=False))
    delays[0] = 0
    powers = np.array([.9, .06, .04]) if condition == "dominant" else 10**(-np.linspace(0, 5, p)/10)
    powers /= powers.sum()
    gains = np.sqrt(powers)*np.exp(2j*np.pi*rng.random(p))
    if condition != "dominant":
        gains *= (.5+np.abs(rng.normal(size=p)))
        gains /= np.linalg.norm(gains)
    fd = rng.uniform(-2400 if condition != "dominant" else -5, 2400 if condition != "dominant" else 5, p)
    return ChannelRealization(delays, gains, fd, 64000)

def reference(wave, channel, y, noise):
    """Evaluator only. Dense small reference; matrix-free reference for large N.

    Residual enclosure marks ambiguous reference bits. Ordinary floating results
    are not relabeled exact mathematics; Decimal adversarial tests are separate.
    """
    c = time_channel(wave, channel)
    b = c.conj().T @ y
    n = wave.n_symbols
    if n <= 256:
        a = (c.conj().T @ c).toarray()+noise*np.eye(n)
        factor = cho_factor(a, lower=True)
        solution = cho_solve(factor, b)
        # Extended-precision residual and a correction using the same factor.
        wide_c = c.toarray().astype(np.clongdouble)
        wide_y = y.astype(np.clongdouble)
        residual = wide_c.conj().T @ (wide_y-wide_c @ solution.astype(np.clongdouble))-noise*solution
        solution += cho_solve(factor, np.asarray(residual, complex))
        residual = wide_c.conj().T @ (wide_y-wide_c @ solution.astype(np.clongdouble))-noise*solution
    else:
        op = LinearOperator((n, n), matvec=lambda x: c.conj().T @ (c @ x)+noise*x, dtype=complex)
        diag = np.asarray(abs(c).power(2).sum(axis=0)).ravel()+noise
        pre = LinearOperator((n, n), matvec=lambda x: x/diag, dtype=complex)
        solution, info = cg(op, b, M=pre, rtol=2e-14, atol=0, maxiter=4*n)
        if info != 0:
            raise RuntimeError("reference did not converge")
        residual = b-op@solution
    alpha, beta = physical_spectral_bounds(channel, noise)
    uncertainty = float(np.linalg.norm(residual))/alpha+128*np.finfo(float).eps*(1+np.linalg.norm(b)+beta*np.linalg.norm(solution))/alpha
    symbols = wave.analysis(solution)
    return {"solution": solution, "symbols": symbols,
            "uncertainty": uncertainty, "condition_upper": beta/alpha}

