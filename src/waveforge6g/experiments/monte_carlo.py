"""Paired, auditable Monte Carlo simulation with common random numbers."""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from ..analysis.papr import oversample
from ..analysis.statistics import frame_mean_interval, wilson_interval
from ..channels import ChannelRealization, realize_channel
from ..channels.awgn import complex_awgn, noise_variance
from ..config import ConfigError, ExperimentConfig
from ..core.modulation import bits_per_symbol, demodulate, modulate
from ..receivers.detectors import detect, effective_channel_operator
from ..receivers.equalizers import one_tap_mmse, one_tap_zf
from ..reproducibility import stable_digest, trial_rng
from ..waveforms import create_waveform
from .sweep import sweep_points


@dataclass(frozen=True)
class FairComparisonContext:
    """Same bits, channel and unit-noise array reused by every paired candidate."""

    seed: int
    trial: int
    point: dict[str, Any]
    bits: np.ndarray
    channel: ChannelRealization
    unit_noise: np.ndarray

    @classmethod
    def create(cls, config: ExperimentConfig, point: dict[str, Any],
               trial: int) -> FairComparisonContext:
        seed, system = config["experiment"]["seed"], config["system"]
        n = int(point["frame_size"])
        # SNR is deliberately part of point: independent noise/bits across sweep cells.
        bits = trial_rng(seed, point, trial, 0).integers(
            0, 2, n * bits_per_symbol(system["modulation"]), dtype=np.uint8)
        channel = realize_channel(config["channel"], trial_rng(seed, point, trial, 1),
                                  system["sample_rate_hz"], point["max_doppler_hz"])
        if np.max(channel.delays, initial=0) > system["cp_length"]:
            raise ConfigError("system.cp_length must cover every realized channel delay")
        noise = complex_awgn(n + system["cp_length"], 1.0,
                             trial_rng(seed, point, trial, 2))
        bits.setflags(write=False)
        noise.setflags(write=False)
        return cls(seed, trial, dict(point), bits, channel, noise)

    def audit(self) -> dict[str, Any]:
        """Small fingerprints establish shared inputs without storing large arrays."""
        return {
            "bits_sha256": hashlib.sha256(self.bits.tobytes()).hexdigest(),
            "noise_sha256": hashlib.sha256(self.unit_noise.tobytes()).hexdigest(),
            "channel_sha256": stable_digest(self.channel.to_dict()),
        }


class MonteCarloRunner:
    """One shared frame loop; early stop only after ALL candidates meet target."""

    def __init__(self, config: ExperimentConfig):
        self.config = config

    def run(self) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
        """Return aggregate metrics, per-frame measurements and channel examples."""
        config = self.config
        system, receiver, mc = (config[key] for key in ("system", "receiver", "monte_carlo"))
        kind = config["experiment"]["kind"]
        records, examples = [], []
        for point in sweep_points(config):
            n, cp = int(point["frame_size"]), system["cp_length"]
            candidates = [create_waveform(name, n, cp, system["subcarriers"],
                                          system["c1"], system["c2"])
                          for name in system["waveforms"]]
            detectors = receiver["detectors"] if kind != "papr" else ["none"]
            accumulated = {(w.name, d): 0 for w in candidates for d in detectors}
            variance = noise_variance(point["snr_db"])
            iterator = tqdm(range(mc["max_frames"]), disable=not mc["progress"],
                            desc=f"{kind} N={n} Es/N0={point['snr_db']:g}", leave=False)
            for frame in iterator:
                context = FairComparisonContext.create(config, point, frame)
                if frame == 0:
                    examples.append({"point": point, "channel": context.channel.to_dict(),
                                     "constellations": []})
                symbols = modulate(context.bits, system["modulation"])
                audit = context.audit()
                for waveform in candidates:
                    start = time.perf_counter()
                    tx = waveform.modulate(symbols)
                    tx_seconds = time.perf_counter() - start
                    useful = oversample(tx[cp:], config["plotting"]["papr_oversampling"])
                    power = np.abs(useful) ** 2
                    papr = float(10 * np.log10(power.max() / power.mean()))
                    for detector in detectors:
                        errors, symbol_errors, evm = 0, 0, 0.0
                        start = time.perf_counter()
                        if kind != "papr":
                            received = context.channel.apply(tx) + np.sqrt(variance) * context.unit_noise
                            observation = waveform.demodulate(received)
                            if detector.startswith("one_tap"):
                                impulse = np.zeros(n, dtype=complex)
                                np.add.at(impulse, context.channel.delays, context.channel.gains)
                                response = np.fft.fft(impulse)
                                estimated = (one_tap_zf(observation, response) if detector ==
                                             "one_tap_zf" else
                                             one_tap_mmse(observation, response, variance))
                            else:
                                operator = effective_channel_operator(waveform, context.channel)
                                estimated = detect(observation, operator, variance, method=detector,
                                                   solver=receiver["solver"],
                                                   dense_threshold=receiver["dense_threshold"],
                                                   rtol=receiver["rtol"], maxiter=receiver["maxiter"])
                            decided = demodulate(estimated, system["modulation"])
                            incorrect = decided != context.bits
                            errors = int(np.count_nonzero(incorrect))
                            symbol_errors = int(np.count_nonzero(incorrect.reshape(n, -1).any(axis=1)))
                            evm = float(np.mean(np.abs(estimated - symbols) ** 2) /
                                        np.mean(np.abs(symbols) ** 2))
                            if frame == 0:
                                examples[-1]["constellations"].append({
                                    "waveform": waveform.name, "detector": detector,
                                    "estimated": np.column_stack((estimated.real, estimated.imag)).tolist(),
                                    "reference": np.column_stack((symbols.real, symbols.imag)).tolist(),
                                })
                        elapsed = time.perf_counter() - start + tx_seconds
                        accumulated[waveform.name, detector] += errors
                        delays = context.channel.delays
                        gain_power = np.abs(context.channel.gains) ** 2
                        delay = delays / system["sample_rate_hz"]
                        mean_delay = np.average(delay, weights=gain_power)
                        rms_delay = np.sqrt(np.average((delay - mean_delay) ** 2, weights=gain_power))
                        records.append({
                            **point, "point_id": stable_digest(point)[:16], "trial": frame,
                            "waveform": waveform.name, "detector": detector,
                            "modulation": system["modulation"], "bit_errors": errors,
                            "symbol_errors": symbol_errors, "block_errors": int(errors > 0),
                            "n_bits": len(context.bits), "n_symbols": n, "evm_squared": evm,
                            "papr_db": papr, "runtime_s": elapsed,
                            "transmit_power": float(np.mean(np.abs(tx) ** 2)),
                            "realized_max_doppler_hz": float(np.max(np.abs(context.channel.dopplers_hz))),
                            "delay_spread_s": float(rms_delay),
                            "delay_span_s": float((delays.max() - delays.min()) / system["sample_rate_hz"]),
                            "configuration_hash": config.config_hash, **audit,
                        })
                if kind == "ber" and mc["min_bit_errors"] > 0 and frame + 1 >= mc["min_frames"]:
                    if all(value >= mc["min_bit_errors"] for value in accumulated.values()):
                        break
        trials = pd.DataFrame.from_records(records)
        return self._aggregate(trials), trials, examples

    def _aggregate(self, trials: pd.DataFrame) -> pd.DataFrame:
        rows = []
        config, system, receiver = self.config, self.config["system"], self.config["receiver"]
        for _, group in trials.groupby(["point_id", "waveform", "detector"], sort=False):
            first = group.iloc[0]
            errors, n_bits = int(group.bit_errors.sum()), int(group.n_bits.sum())
            frames, n = len(group), int(first.frame_size)
            ber = errors / n_bits
            lo, hi = wilson_interval(errors, n_bits)
            # Independent frames, potentially dependent bits within each fading frame.
            frame_rates = group.bit_errors / group.n_bits
            flo, fhi = frame_mean_interval(frame_rates)
            blo, bhi = wilson_interval(int(group.block_errors.sum()), frames)
            dense = receiver["solver"] == "dense" or (
                receiver["solver"] == "auto" and n <= receiver["dense_threshold"])
            one_tap = str(first.detector).startswith("one_tap")
            # Algorithmic working arrays, excludes Python/BLAS/allocator overhead.
            memory = 16 * ((4 * n * n + 12 * n) if dense and not one_tap else 24 * n)
            row = {key: first[key] for key in
                   ("point_id", "waveform", "detector", "modulation", "snr_db", "frame_size",
                    "max_doppler_hz", "configuration_hash")}
            row.update({
                "n_frames": frames, "n_bits": n_bits, "bit_errors": errors,
                "n_symbols": int(group.n_symbols.sum()), "symbol_errors": int(group.symbol_errors.sum()),
                "block_errors": int(group.block_errors.sum()), "ber": ber,
                "ser": float(group.symbol_errors.sum() / group.n_symbols.sum()),
                "bler": float(group.block_errors.mean()), "ber_iid_ci_low": lo, "ber_iid_ci_high": hi,
                "ber_frame_ci_low": flo,
                "ber_frame_ci_high": fhi,
                "bler_ci_low": blo, "bler_ci_high": bhi,
                "papr_mean_db": float(group.papr_db.mean()),
                "evm_rms": float(np.sqrt(group.evm_squared.mean())),
                "runtime_mean_s": float(group.runtime_s.mean()),
                "runtime_std_s": float(group.runtime_s.std(ddof=0)),
                "memory_estimate_bytes": memory,
                "spectral_efficiency_bps_hz": bits_per_symbol(system["modulation"]) * n / (n + system["cp_length"]),
                "transmit_power_mean": float(group.transmit_power.mean()),
                "max_doppler_hz": float(group.realized_max_doppler_hz.max()),
                "delay_spread_s": float(group.delay_spread_s.mean()),
                "delay_span_s": float(group.delay_span_s.mean()),
                "frame_duration_s": (n + system["cp_length"]) / system["sample_rate_hz"],
                "bandwidth_hz": system["sample_rate_hz"],
                "solver": "one_tap" if one_tap else ("dense" if dense else "iterative"),
                "stopping_reason": "max_frames" if frames == config["monte_carlo"]["max_frames"] else "all_error_targets",
            })
            if config["experiment"]["kind"] == "papr":
                for key in ("ber", "ser", "bler", "evm_rms", "ber_iid_ci_low", "ber_iid_ci_high",
                            "ber_frame_ci_low", "ber_frame_ci_high", "bler_ci_low", "bler_ci_high"):
                    row[key] = float("nan")
            rows.append(row)
        return pd.DataFrame(rows)
