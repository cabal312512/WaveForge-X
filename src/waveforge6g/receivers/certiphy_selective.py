"""R5 integration: unchanged PCG/Deep kernels, causal gate and isolated ablations."""
import time
import numpy as np
from .certiphy import CertiPHY
from .certiphy_deep import DeepRefiner, circulant_envelope
from .budgeted import transform_work
from ..core.modulation import bits_per_symbol


class ProfileCertiPHY(CertiPHY):
    """Sparse fractional/SOS adapter. No integer-unitary-path spectral assumption."""
    def __init__(self, wave, channel, noise, modulation, start_index=0):
        if not np.isfinite(noise) or noise <= 0:
            raise ValueError('positive finite noise required')
        start, cpu = time.perf_counter(), time.process_time()
        self.wave, self.channel, self.noise = wave, channel, float(noise)
        self.n, self.width = wave.n_symbols, bits_per_symbol(modulation)
        self.modulation = modulation
        self.matrix = channel.matrix(wave, start_index)
        self.adjoint = self.matrix.conj().T.tocsr()
        absolute = abs(self.matrix)
        self.diagonal = np.asarray(absolute.power(2).sum(axis=0)).ravel()+noise
        self.alpha = float(noise)
        self.beta = float(noise+np.max(absolute.sum(axis=0))*np.max(absolute.sum(axis=1)))
        self.mu = (1-1e-12)*self.alpha/float(self.diagonal.max())
        self.nnz = self.matrix.nnz
        self.transform = transform_work(wave)
        self.step_work = 16*self.nnz+40*self.n+8
        self.normal_work = 16*self.nnz+12*self.n
        self.decode_work = 4*self.n*self.width
        self.setup_work = channel.preparation_work(self.n)+32*self.nnz+20*self.n
        self.setup_elapsed = time.perf_counter()-start
        self.setup_cpu = time.process_time()-cpu


class SpectralAblation(DeepRefiner):
    def __init__(self, directions=False):
        super().__init__('spectral')
        self.directions = directions

    def prepare(self, receiver, ledger):
        if self.directions:
            return super().prepare(receiver, ledger)
        self.receiver = receiver
        start = time.perf_counter()
        cost = 100*receiver.nnz+5*receiver.n*np.log2(receiver.n)+80*receiver.n
        if ledger.fits(cost):
            ledger.add('deep_envelope_preparation', cost)
            self.weights, details = circulant_envelope(receiver.matrix, receiver.noise)
            self.metrics.update(details)
        self.metrics['preparation_seconds'] += time.perf_counter()-start

    def radii(self, energy, radius, guard, ledger, residual):
        result = super().radii(energy, radius, guard, ledger, residual)
        if self.directions or self.current_eta is None:
            return result
        # A-energy only: eta / alpha is a valid Euclidean ball.
        return np.full(self.receiver.n, min(radius, np.sqrt(self.current_eta/self.receiver.alpha)))

    def bound(self, symbols, active, target, radius, energy, guard, ledger, margins):
        return int(active.sum())


class SelectiveRefiner(DeepRefiner):
    def __init__(self, doppler_limit=.01, minimum_predicted_steps=12.):
        super().__init__('spectral')
        self.doppler_limit, self.minimum_predicted_steps = doppler_limit, minimum_predicted_steps
        self.enabled = False

    def prepare(self, receiver, ledger):
        self.receiver = receiver
        gate_work = 80.
        if not ledger.fits(gate_work):
            self.metrics['gate_cap'] = True
            return
        ledger.add('selective_gate', gate_work)
        c = receiver.channel
        variation = c.doppler_hz*receiver.n/c.sample_rate_hz
        # Deliberately cheap predictor, not an estimate of a certified saving.
        predicted_steps = .5*np.sqrt(receiver.beta/receiver.alpha)
        preparation = 100*receiver.nnz+25*receiver.n*np.log2(receiver.n)+120*receiver.n
        predicted_saving = .2*predicted_steps*receiver.step_work
        self.enabled = bool(variation <= self.doppler_limit and predicted_steps >= self.minimum_predicted_steps
                            and predicted_saving >= preparation)
        self.metrics.update(gate_enabled=self.enabled, variation=variation,
                            predicted_steps=predicted_steps, predicted_saving=predicted_saving,
                            predicted_preparation=preparation)
        if self.enabled:
            super().prepare(receiver, ledger)

    def radii(self, energy, radius, guard, ledger, residual):
        if not self.enabled:
            return np.full(self.receiver.n, radius)
        return super().radii(energy, radius, guard, ledger, residual)

    def bound(self, symbols, active, target, radius, energy, guard, ledger, margins):
        if not self.enabled:
            return int(active.sum())
        return super().bound(symbols, active, target, radius, energy, guard, ledger, margins)
