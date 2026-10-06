"""Shared physical simulator for causal decision experiments and oracle audits."""

from __future__ import annotations

import time

import numpy as np

from ..analysis.papr import oversample
from ..channels import ChannelRealization
from ..channels.awgn import complex_awgn, noise_variance
from ..config import ExperimentConfig
from ..core.modulation import bits_per_symbol, demodulate, modulate
from ..receivers.detectors import detect, effective_channel_operator
from ..waveforms import create_waveform


def operation_proxy(waveform: str, n: int, subcarriers: int, n_paths: int,
                     solver: str) -> float:
    """Deterministic relative arithmetic model, not FLOPs or measured runtime.

    A complex FFT stage is assigned 5 N log2(N) units; a chirp multiplication
    6N. Dense detection uses N^3 units, plus N effective-operator applications.
    Iterative detection uses a fixed planning budget of 4N applications; actual
    solver iteration counts and hardware timings are not inferred from this.
    """
    transform = 5 * n * np.log2(n)
    if waveform == "otfs":
        transform = 5 * n * (2 * np.log2(subcarriers) + np.log2(n / subcarriers))
    elif waveform == "afdm":
        transform += 12 * n
    operator = 2 * transform + 8 * n_paths * n
    return float(2 * transform + (n ** 3 + n * operator if solver == "dense" else 4 * n * operator))


def evaluate_waveforms(config: ExperimentConfig, channel: ChannelRealization,
                        snr_db: float, seed: int, epoch: int) -> list[dict]:
    """Run all three links with identical information bits, channel and time noise.

    Called only AFTER all online selectors have committed their epoch actions.
    Results belong to the evaluator and are never passed wholesale to selectors.
    """
    system, receiver = config["system"], config["receiver"]
    n, cp = system["frame_size"], system["cp_length"]
    rng = np.random.default_rng(np.random.SeedSequence([seed, epoch, 9173]))
    bits = rng.integers(0, 2, n * bits_per_symbol(system["modulation"]), dtype=np.uint8)
    symbols = modulate(bits, system["modulation"])
    noise = complex_awgn(n + cp, noise_variance(snr_db), rng)
    metrics = []
    dense = receiver["solver"] == "dense" or (receiver["solver"] == "auto" and
                                               n <= receiver["dense_threshold"])
    for name in ("ofdm", "otfs", "afdm"):
        waveform = create_waveform(name, n, cp, system["subcarriers"], system["c1"], system["c2"])
        start = time.perf_counter()
        tx = waveform.modulate(symbols)
        received = channel.apply(tx) + noise
        observation = waveform.demodulate(received)
        estimate = detect(observation, effective_channel_operator(waveform, channel), noise_variance(snr_db),
                          method=receiver["detector"], solver=receiver["solver"],
                          dense_threshold=receiver["dense_threshold"], rtol=receiver["rtol"],
                          maxiter=receiver["maxiter"])
        elapsed = time.perf_counter() - start
        errors = int(np.count_nonzero(demodulate(estimate, system["modulation"]) != bits))
        samples = oversample(tx[cp:], 4)
        power = np.abs(samples) ** 2
        metrics.append({"waveform": name, "ber": errors / len(bits), "bler": float(errors > 0),
                        "bit_errors": errors, "n_bits": len(bits),
                        "papr_db": float(10 * np.log10(power.max() / power.mean())),
                        "complexity_proxy": operation_proxy(name, n, system["subcarriers"],
                                                             channel.n_paths, "dense" if dense else "iterative"),
                        "spectral_efficiency": bits_per_symbol(system["modulation"]) * n / (n + cp),
                        "runtime_s": elapsed,
                        "evm_rms": float(np.sqrt(np.mean(np.abs(estimate - symbols) ** 2) /
                                                  np.mean(np.abs(symbols) ** 2)))})
    return metrics
