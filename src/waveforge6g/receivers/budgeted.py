"""Representation-dependent row sparsification and explicitly capped PCG.

The actual received signal is NEVER generated with the truncated receiver matrix.
Work units are a transparent arithmetic model, not CPU cycles or hardware latency.
"""

import time
from dataclasses import dataclass

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.sparse import coo_matrix, csr_matrix


@dataclass(frozen=True)
class ReceiverAction:
    waveform: str
    domain: str
    keep: int
    iterations: int

    @property
    def key(self):
        return f"{self.waveform}:{self.domain}:L{self.keep}:K{self.iterations}"


def transform_work(waveform):
    n = waveform.n_symbols
    if waveform.name == "otfs":
        k = waveform.subcarriers
        return float(5*n*(2*np.log2(k)+np.log2(n/k)))
    return float(5*n*np.log2(n) + (12*n if waveform.name == "afdm" else 0))


def time_channel(waveform, channel):
    """Exact R H P in useful time samples, O(N P); includes the actual CPP phases."""
    n, cp = waveform.n_symbols, waveform.cp_length
    if channel.max_delay > cp:
        raise ValueError("guard must cover all integer delays")
    rows = np.arange(n)
    values, cols, row_list = [], [], []
    phases = waveform.prefix_phases
    for delay, gain, doppler in zip(channel.delays, channel.gains, channel.dopplers_hz, strict=True):
        source = rows-int(delay)
        prefix = source < 0
        multiplier = np.ones(n, dtype=complex)
        multiplier[prefix] = phases[cp+source[prefix]]
        values.extend(gain*np.exp(2j*np.pi*doppler*(cp+rows)/channel.sample_rate_hz)*multiplier)
        cols.extend(source % n)
        row_list.extend(rows)
    matrix = coo_matrix((values, (row_list, cols)), shape=(n, n)).tocsr()
    matrix.eliminate_zeros()
    return matrix


def symbol_channel(waveform, time_matrix):
    """Construct full G by basis applications; this entire work is charged."""
    n = waveform.n_symbols
    result = np.empty((n, n), complex)
    basis = np.zeros(n, complex)
    for j in range(n):
        basis[j] = 1
        result[:, j] = waveform.analysis(time_matrix @ waveform.synthesis(basis))
        basis[j] = 0
    return result


def sparsify(matrix, keep):
    """Keep largest |G_ij| per row, stable column order for coefficient ties."""
    matrix = np.asarray(matrix, complex)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or not np.all(np.isfinite(matrix)):
        raise ValueError("finite square matrix required")
    n = len(matrix)
    if isinstance(keep, bool) or not isinstance(keep, int) or not 1 <= keep <= n:
        raise ValueError("keep must be an integer in [1,N]")
    indices = np.argsort(-np.abs(matrix)**2, axis=1, kind="stable")[:, :keep]
    rows = np.repeat(np.arange(n), keep)
    sparse = csr_matrix((matrix[rows, indices.ravel()], (rows, indices.ravel())), shape=matrix.shape)
    sparse.eliminate_zeros()
    total = float(np.sum(abs(matrix)**2))
    retained = float(np.sum(abs(sparse.data)**2))
    return sparse, 0.0 if total == 0 else max(0., 1-retained/total)


def cost_model(waveform, n_paths, domain, keep, iterations, reuse_frames=1):
    """Charge setup on EVERY state/action change, amortize only real reuse frames.

    FFT=5Nlog2N; complex MAC=8; phase generation=48/path/sample;
    stable-sort comparison budget=N² ceil(log2N); dense Cholesky=8N³/3.
    All are disclosed work-unit conventions, not measured instruction counts.
    """
    n = waveform.n_symbols
    if domain not in ("symbol", "time", "dense") or reuse_frames < 1 or iterations < 0:
        raise ValueError("invalid cost specification")
    f = transform_work(waveform)
    paths = min(n_paths, n)
    configuration_work = 96*n+48*waveform.cp_length if waveform.name == "afdm" else 0
    construction_time = n*n_paths*(48+2*np.ceil(np.log2(max(2, n_paths))))+configuration_work
    # Worst-case storage/sparsity, even if actual duplicate taps cancel.
    nnz = n*(paths if domain == "time" else min(keep, n))
    construction_symbol = n*(2*f+8*n*paths) if domain != "time" else 0
    selection = n*n*(6+np.ceil(np.log2(n))) if domain == "symbol" else 0
    if domain == "dense":
        preparation = construction_time+construction_symbol+(8+8/3)*n**3+2*n
        per_frame = 2*f+24*n*n+8*n
        precondition = 0
        memory = 64*n*n+192*n
    else:
        precondition = 6*nnz+n
        preparation = construction_time+construction_symbol+selection+precondition
        per_frame = 2*f+8*nnz+12*n+iterations*(16*nnz+40*n+8)
        memory = 40*nnz+192*n+8*(n+1)+(32*n*n if domain == "symbol" else 0)
    return {"construction_work": float(construction_time+construction_symbol),
            "selection_work": float(selection), "precondition_work": float(precondition),
            "preparation_work": float(preparation), "frame_work": float(per_frame),
            "total_work": float(preparation/reuse_frames+per_frame),
            "peak_array_bytes_model": int(memory), "reuse_frames": reuse_frames}


class BudgetedReceiver:
    """Diagonal-preconditioned CG on (Aᴴ A + N0 I)x=Aᴴy.

    Stop at relative normal-equation residual <= tolerance, or at K, returning
    the current iterate (a legitimate budget-limited estimate) with diagnostics.
    Discarded interference is NOT added to N0 or assumed white.
    """

    def __init__(self, waveform, channel, noise_variance, action, full_time=None,
                 full_symbol=None, tolerance=1e-6):
        if (not np.isfinite(noise_variance) or noise_variance <= 0
                or not np.isfinite(tolerance) or not 0 < tolerance < 1
                or isinstance(action.iterations, bool) or not isinstance(action.iterations, int)
                or action.iterations < (0 if action.domain == "dense" else 1) or action.waveform != waveform.name):
            raise ValueError("invalid receiver settings")
        self.waveform, self.action = waveform, action
        self.variance, self.tolerance = float(noise_variance), tolerance
        start = time.perf_counter()
        matrix_time = time_channel(waveform, channel) if full_time is None else full_time
        self.discarded_energy = 0.
        self._factor = None
        if action.domain == "time":
            self.matrix = matrix_time
        else:
            full = symbol_channel(waveform, matrix_time) if full_symbol is None else full_symbol
            if action.domain == "dense":
                self.matrix = full
                self._factor = cho_factor(full.conj().T@full+noise_variance*np.eye(len(full)), lower=True)
            elif action.domain == "symbol":
                self.matrix, self.discarded_energy = sparsify(full, action.keep)
            else:
                raise ValueError("unknown receiver domain")
        self.adjoint = self.matrix.conj().T
        if action.domain != "dense":
            self.adjoint = self.adjoint.tocsr()
            self.diagonal = np.asarray(abs(self.matrix).power(2).sum(axis=0)).ravel()+noise_variance
        self.setup_cpu_s = time.perf_counter()-start
        self.cost = cost_model(waveform, channel.n_paths, action.domain, action.keep, action.iterations)

    @property
    def array_bytes(self):
        """Retained numerical buffers; excludes Python objects and BLAS workspace."""
        if self.action.domain == "dense":
            return self.matrix.nbytes+self.adjoint.nbytes+self._factor[0].nbytes
        return sum(x.data.nbytes+x.indices.nbytes+x.indptr.nbytes for x in (self.matrix, self.adjoint))+self.diagonal.nbytes

    def solve(self, received_useful):
        start = time.perf_counter()
        y = np.asarray(received_useful, complex)
        if y.shape != (self.waveform.n_symbols,) or not np.all(np.isfinite(y)):
            raise ValueError("finite useful time-domain received vector required")
        if self.action.domain != "time":
            y = self.waveform.analysis(y)
        b = self.adjoint @ y
        if self.action.domain == "dense":
            x = cho_solve(self._factor, b)
            residual = np.linalg.norm(self.adjoint@(self.matrix@x)+self.variance*x-b)/max(np.linalg.norm(b), 1e-30)
            return x, {"iterations": 0, "relative_normal_residual": float(residual), "converged": True,
                       "solve_cpu_s": time.perf_counter()-start, "residual_history": [float(residual)]}
        x = np.zeros_like(b)
        r = b.copy()
        norm_b = np.linalg.norm(b)
        z = r/self.diagonal
        p = z.copy()
        rho = float(np.vdot(r, z).real)
        history = [1. if norm_b else 0.]
        count = 0
        for _ in range(self.action.iterations):
            if history[-1] <= self.tolerance:
                break
            q = self.adjoint@(self.matrix@p)+self.variance*p
            denominator = float(np.vdot(p, q).real)
            if denominator <= 0 or not np.isfinite(denominator):
                raise FloatingPointError("PCG lost positive definiteness")
            alpha = rho/denominator
            x += alpha*p
            r -= alpha*q
            count += 1
            history.append(float(np.linalg.norm(r)/max(norm_b, 1e-30)))
            z = r/self.diagonal
            next_rho = float(np.vdot(r, z).real)
            p = z + (next_rho/rho)*p
            rho = next_rho
        if self.action.domain == "time":
            x = self.waveform.analysis(x)
        return x, {"iterations": count, "relative_normal_residual": history[-1],
                   "converged": history[-1] <= self.tolerance,
                   "solve_cpu_s": time.perf_counter()-start, "residual_history": history}
