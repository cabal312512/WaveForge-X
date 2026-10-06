"""Fixed-state replicated PHY evaluation; no evaluator data enters selectors."""

import hashlib
import json
from functools import lru_cache

import numpy as np

from ..analysis.papr import oversample
from ..channels import ChannelRealization
from ..channels.awgn import complex_awgn, noise_variance
from ..core.modulation import bits_per_symbol, demodulate, modulate
from ..decision.base import ACTIONS
from ..decision.objective import Objective
from ..waveforms import create_waveform
from .link import operation_proxy


def role_seed(seed: int, role: str, *keys: object) -> int:
    """Stable, order-independent streams, including across Python processes."""
    raw = json.dumps([int(seed), role, *keys], separators=(",", ":")).encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "little")


@lru_cache(maxsize=32)
def transforms(n, cp, subcarriers, c1, c2):
    matrices = []
    for name in ACTIONS:
        wave = create_waveform(name, n, cp, subcarriers, c1, c2)
        prefix = np.column_stack([wave.modulate(x) for x in np.eye(n)])
        matrices.append((prefix, prefix[cp:].conj().T))
    return matrices


def mismatch(channel, nmse, seed):
    """Common physical tap error, normalized in finite time-domain operator norm."""
    if not np.isfinite(nmse) or nmse < 0:
        raise ValueError("NMSE must be finite and nonnegative")
    if nmse == 0:
        return channel
    rng = np.random.default_rng(seed)
    error = (rng.normal(size=channel.n_paths) + 1j * rng.normal(size=channel.n_paths)) / np.sqrt(2)
    # Exact operator normalization is performed by PreparedPHY for its frame size.
    return ChannelRealization(channel.delays, error, channel.dopplers_hz, channel.sample_rate_hz)


class PreparedPHY:
    """Reuse dense LMMSE factors across bits/noise replicas of one frozen state.

    Uses existing waveform/channel operations, not a second propagation model.
    Actual online time advances once per state, never once per evaluator replica.
    """

    def __init__(self, config, channel, snr_db, csi_nmse=0.0, csi_seed=0):
        self.system = s = config["system"]
        self.n, self.cp = n, cp = s["frame_size"], s["cp_length"]
        receiver = config["receiver"]
        if n > 128 or receiver["detector"] != "lmmse":
            raise ValueError("replicated evaluator supports dense LMMSE with N<=128 only")
        self.variance = noise_variance(snr_db)
        self.objective = Objective(config["objective"])
        self.channel = channel
        self.matrices = []
        estimated = channel
        self.nmse = 0.0
        if csi_nmse:
            error_channel = mismatch(channel, csi_nmse, csi_seed)
            identity = np.eye(n + cp)
            true_h = np.column_stack([channel.apply(x) for x in identity])
            error_h = np.column_stack([error_channel.apply(x) for x in identity])
            norm = np.linalg.norm(true_h)**2
            if norm < 1e-24:
                raise ValueError("relative CSI NMSE undefined for near-zero channel")
            scale = np.sqrt(csi_nmse * norm / np.linalg.norm(error_h)**2)
            estimated = ChannelRealization(channel.delays, channel.gains + scale * error_channel.gains,
                                           channel.dopplers_hz, channel.sample_rate_hz)
            self.nmse = float(np.linalg.norm(scale * error_h)**2 / norm)
        for prefix, analysis in transforms(n, cp, s["subcarriers"], s["c1"], s["c2"]):
            propagated = np.column_stack([channel.apply(x) for x in prefix.T])
            effective = analysis @ propagated[cp:]
            hhat = effective if estimated is channel else analysis @ np.column_stack(
                [estimated.apply(x)[cp:] for x in prefix.T])
            filt = np.linalg.solve(hhat.conj().T @ hhat + self.variance * np.eye(n), hhat.conj().T)
            self.matrices.append((prefix, analysis, effective, filt))

    def evaluate(self, seed, epoch, role, replicas):
        """Arrays [replica,action]; prefix-stable replica keys and paired CRN."""
        if not isinstance(replicas, int) or replicas < 1:
            raise ValueError("replicas must be positive")
        s, n, cp = self.system, self.n, self.cp
        width = n * bits_per_symbol(s["modulation"])
        bits, noise = [], []
        for k in range(replicas):
            rng = np.random.default_rng(role_seed(seed, role, epoch, k))
            bits.append(rng.integers(0, 2, width, dtype=np.uint8))
            noise.append(complex_awgn(n + cp, self.variance, rng))
        bits = np.asarray(bits)
        symbols = modulate(bits.ravel(), s["modulation"]).reshape(replicas, n).T
        noise = np.asarray(noise).T
        errors = np.empty((replicas, 3), dtype=np.int32)
        papr = np.empty((replicas, 3))
        energy = np.empty((replicas, 3))
        losses = np.empty((replicas, 3))
        for a, (prefix, analysis, effective, filt) in enumerate(self.matrices):
            tx = prefix @ symbols
            estimates = filt @ (effective @ symbols + analysis @ noise[cp:])
            received_bits = demodulate(estimates.T.ravel(), s["modulation"]).reshape(replicas, width)
            errors[:, a] = np.count_nonzero(received_bits != bits, axis=1)
            complexity = operation_proxy(ACTIONS[a], n, s["subcarriers"], self.channel.n_paths, "dense")
            for k in range(replicas):
                power = abs(oversample(tx[cp:, k], 4))**2
                papr[k, a] = 10 * np.log10(power.max() / power.mean())
            energy[:, a] = np.sum(abs(tx)**2, axis=0)
            obj = self.objective
            losses[:, a] = (obj.weights["ber"] * errors[:, a] / width
                            + obj.weights["bler"] * (errors[:, a] > 0)
                            + obj.weights["papr"] * np.clip(papr[:, a] / obj.papr_reference_db, 0, 1)
                            + obj.weights["complexity"] * min(complexity / obj.complexity_reference, 1)
                            ) / obj.total_weight
        return {"loss": losses, "errors": errors, "bits": width, "papr": papr, "energy": energy}
