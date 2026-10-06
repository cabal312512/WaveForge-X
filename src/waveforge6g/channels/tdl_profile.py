"""TR 38.901 profile-based SISO channel; finite-SOS and fractional FIR approximation.

This is not a full 3GPP conformance implementation. Common causal filter latency
is included in the guard, not silently removed or charged to one waveform only.
"""
import json
from pathlib import Path
import numpy as np
from scipy.sparse import coo_matrix


def fractional_kernel(delay, half=16):
    if not np.isfinite(delay) or delay < 0 or half < 0:
        raise ValueError("nonnegative finite delay and FIR half width required")
    base = int(np.floor(delay))
    fraction = delay-base
    if half == 0:
        if fraction != 0:
            raise ValueError("fractional delay needs FIR support")
        return np.array([base]), np.array([1.])
    offsets = np.arange(2*half+1)
    h = np.sinc(offsets-half-fraction)*np.kaiser(2*half+1, 8.6)
    # Exact integer limit; avoid numerical sinc leakage.
    if fraction == 0:
        h[:] = 0
        h[half] = 1
    h /= np.linalg.norm(h)
    keep = h != 0
    return base+offsets[keep], h[keep]


class TDLProfile:
    def __init__(self, profile, seed, delay_spread_s=100e-9, doppler_hz=5.,
                 sample_rate_hz=7.68e6, half=16, sinusoids=32, table_path=None):
        table_path = table_path or Path(__file__).resolve().parents[3]/'configs/research_v5/tdl_profiles.json'
        table = json.loads(Path(table_path).read_text())['profiles'][profile]
        self.profile, self.seed = profile, int(seed)
        self.sample_rate_hz, self.doppler_hz = float(sample_rate_hz), float(doppler_hz)
        self.half, self.sinusoids = half, sinusoids
        self.delays = np.array([r['delay'] for r in table])*delay_spread_s*sample_rate_hz
        power = 10**(np.array([r['power_db'] for r in table])/10)
        self.power = power/power.sum()
        self.los = np.array([r['fading'] == 'LOS path' for r in table])
        self.n_paths = len(table)
        rng = np.random.default_rng(seed)
        # Gaussian SOS coefficients give proper complex Gaussian tap marginals;
        # randomized angular quadrature approximates the classical Jakes PSD.
        self.angles = 2*np.pi*(np.arange(sinusoids)[None, :]+rng.random((self.n_paths, 1)))/sinusoids
        self.amplitudes = (rng.normal(size=self.angles.shape)+1j*rng.normal(size=self.angles.shape))/np.sqrt(2*sinusoids)
        self.los_phase = rng.uniform(0, 2*np.pi, self.n_paths)
        self.kernels = [fractional_kernel(d, half) for d in self.delays]
        self.max_delay = max(int(d.max()) for d, h in self.kernels)

    def coefficients(self, indices):
        times = np.asarray(indices)/self.sample_rate_hz
        result = np.empty((self.n_paths, len(times)), complex)
        for p in range(self.n_paths):
            if self.los[p]:
                result[p] = np.exp(1j*(self.los_phase[p]+2*np.pi*.7*self.doppler_hz*times))
            else:
                result[p] = self.amplitudes[p] @ np.exp(2j*np.pi*self.doppler_hz*np.cos(self.angles[p])[:, None]*times)
        return np.sqrt(self.power)[:, None]*result

    def apply(self, transmitted, start_index=0):
        x = np.asarray(transmitted, complex)
        gains = self.coefficients(start_index+np.arange(len(x)))
        y = np.zeros_like(x)
        for p, (delays, h) in enumerate(self.kernels):
            for d, weight in zip(delays, h):
                if d < len(x):
                    y[d:] += gains[p, d:]*weight*x[:len(x)-d]
        return y

    def matrix(self, wave, start_index=0):
        n, cp = wave.n_symbols, wave.cp_length
        if cp < self.max_delay:
            raise ValueError('guard shorter than causal FIR support')
        rows = np.arange(n)
        gains = self.coefficients(start_index+cp+rows)
        rr, cc, vv = [], [], []
        for p, (delays, h) in enumerate(self.kernels):
            for d, weight in zip(delays, h):
                source = rows-d
                phase = np.ones(n, complex)
                negative = source < 0
                phase[negative] = wave.prefix_phases[cp+source[negative]]
                rr.append(rows); cc.append(source % n); vv.append(gains[p]*weight*phase)
        c = coo_matrix((np.concatenate(vv), (np.concatenate(rr), np.concatenate(cc))), shape=(n,n)).tocsr()
        c.eliminate_zeros()
        return c

    def preparation_work(self, n):
        entries = n*sum(len(d) for d, h in self.kernels)
        # complex exponential 32 real-equivalent units, multiply/add 8;
        # COO accumulation/sort conservatively charged at log2 contributions.
        return 40*n*self.n_paths*self.sinusoids + entries*(20+2*np.log2(max(2,entries))) + 64*self.n_paths*(2*self.half+1)
