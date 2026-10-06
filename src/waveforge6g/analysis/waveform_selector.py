"""Transparent benchmark oracle and small nearest-region waveform selector."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

FEATURES = ("snr_db", "max_doppler_hz", "delay_span_s", "modulation_bits")
WAVEFORMS = ("ofdm", "otfs", "afdm")


def rule_based_selector(snr_db: float, max_doppler_hz: float, delay_span_s: float,
                        modulation_bits: int, *, sample_rate_hz: float,
                        frame_size: int) -> str:
    """Explicit heuristic, NOT an oracle or a proven ordering of waveform quality.

    Doppler is normalized by OFDM subcarrier spacing fs/N. Thresholds are
    illustrative, dimensionless rules for comparison with the empirical oracle.
    """
    values = (snr_db, max_doppler_hz, delay_span_s, sample_rate_hz, frame_size)
    if not all(np.isfinite(values)) or min(max_doppler_hz, delay_span_s) < 0:
        raise ValueError("selector inputs must be finite and Doppler/delay nonnegative")
    if sample_rate_hz <= 0 or frame_size < 1 or modulation_bits not in (1, 2, 4, 6):
        raise ValueError("invalid sampling rate, frame size or modulation bits")
    normalized = max_doppler_hz * frame_size / sample_rate_hz
    if normalized < 0.05 or snr_db < 0:
        return "ofdm"
    if normalized < 0.35 or (delay_span_s * sample_rate_hz >= 4 and modulation_bits >= 4):
        return "otfs"
    return "afdm"


def objective_scores(group: pd.DataFrame, objective: str = "ber",
                     complexity_weight: float = 0.05, papr_weight: float = 0.02) -> np.ndarray:
    """Smaller is better; utility uses BER + scaled measured time + scaled PAPR."""
    if objective not in ("ber", "utility"):
        raise ValueError("objective must be ber or utility")
    score = group.ber.to_numpy(float).copy()
    if not np.all(np.isfinite(score)) or np.any((score < 0) | (score > 1)):
        raise ValueError("selector requires measured finite BER in [0, 1]")
    if objective == "utility":
        if not np.all(np.isfinite([complexity_weight, papr_weight])) or min(
            complexity_weight, papr_weight
        ) < 0:
            raise ValueError("penalty weights must be finite and nonnegative")
        runtime = group.runtime_mean_s.to_numpy(float)
        papr = group.papr_mean_db.to_numpy(float)
        if not np.all(np.isfinite(runtime)) or np.any(runtime < 0):
            raise ValueError("runtime measurements must be finite and nonnegative")
        if not np.all(np.isfinite(papr)) or np.any(papr < 0):
            raise ValueError("PAPR measurements must be finite and nonnegative")
        score += complexity_weight * runtime / max(runtime.max(), 1e-15)
        score += papr_weight * papr / max(papr.max(), 1e-15)
    return score


def benchmark_oracle(group: pd.DataFrame, objective: str = "ber",
                     complexity_weight: float = 0.05, papr_weight: float = 0.02) -> str:
    """Hindsight minimum measured objective, tie broken OFDM/OTFS/AFDM.

    This finite-sample oracle does not assert statistical superiority. Each
    condition must contain every waveform and exactly one common detector.
    """
    if len(group) != 3 or set(group.waveform) != set(WAVEFORMS) or group.detector.nunique() != 1:
        raise ValueError("oracle needs all three waveforms with one shared detector per condition")
    ordered = group.set_index("waveform").loc[list(WAVEFORMS)].reset_index()
    scores = objective_scores(ordered, objective, complexity_weight, papr_weight)
    return WAVEFORMS[int(np.argmin(scores))]


@dataclass
class NearestRegionSelector:
    """Dependency-free nearest measured operating region with saved normalization."""

    features: list[list[float]]
    labels: list[str]
    center: list[float]
    scale: list[float]
    metadata: dict

    def __post_init__(self) -> None:
        """Reject malformed portable models before distance calculations."""
        try:
            features = np.asarray(self.features, dtype=float)
            center = np.asarray(self.center, dtype=float)
            scale = np.asarray(self.scale, dtype=float)
        except (TypeError, ValueError) as error:
            raise ValueError("selector model arrays must contain numbers") from error
        if features.ndim != 2 or features.shape[1] != 4 or features.shape[0] == 0:
            raise ValueError("selector features must have shape [nonzero regions, 4]")
        if not np.all(np.isfinite(features)):
            raise ValueError("selector features must be finite")
        if np.any(features[:, 1:3] < 0) or not np.all(np.isin(features[:, 3], [1, 2, 4, 6])):
            raise ValueError("selector features require nonnegative Doppler/delay and supported bits")
        if not isinstance(self.labels, list) or len(self.labels) != len(features):
            raise ValueError("selector labels must match the number of feature rows")
        if any(label not in WAVEFORMS for label in self.labels):
            raise ValueError("selector labels must be ofdm, otfs or afdm")
        if center.shape != (4,) or not np.all(np.isfinite(center)):
            raise ValueError("selector center must contain four finite values")
        if scale.shape != (4,) or not np.all(np.isfinite(scale)) or np.any(scale <= 0):
            raise ValueError("selector scale must contain four finite positive values")
        if not isinstance(self.metadata, dict):
            raise ValueError("selector metadata must be a mapping")
        self.features = features.tolist()
        self.center = center.tolist()
        self.scale = scale.tolist()

    def predict(self, conditions: np.ndarray) -> list[str]:
        """Predict from rows [Es/N0 dB, maximum Doppler Hz, delay span s, bits/symbol]."""
        query = np.atleast_2d(np.asarray(conditions, dtype=float))
        if query.ndim != 2 or query.shape[1] != 4 or not np.all(np.isfinite(query)):
            raise ValueError("conditions must have four finite feature columns")
        if np.any(query[:, 1:3] < 0) or not np.all(np.isin(query[:, 3], [1, 2, 4, 6])):
            raise ValueError("conditions require nonnegative Doppler/delay and supported bits")
        distances = ((query[:, None, :] - np.asarray(self.features)[None, :, :]) /
                     np.asarray(self.scale)) ** 2
        indices = distances.sum(axis=2).argmin(axis=1)
        return [self.labels[index] for index in indices]

    def save(self, path: str | Path) -> None:
        """Save portable JSON; no executable pickles."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(asdict(self), indent=2, allow_nan=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> NearestRegionSelector:
        """Read and validate JSON model data without executing serialized code."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        expected = {"features", "labels", "center", "scale", "metadata"}
        if not isinstance(data, dict) or set(data) != expected:
            raise ValueError("selector JSON must contain features, labels, center, scale and metadata")
        return cls(**data)


def train_selector(directories: list[Path], objective: str = "ber",
                   complexity_weight: float = 0.05,
                   papr_weight: float = 0.02) -> NearestRegionSelector:
    """Fit to measured complete paired benchmark points; report held-out regions.

    Split entire condition cells, never different waveform rows from one cell.
    For fewer than five conditions no generalization score is reported.
    """
    from ..core.modulation import bits_per_symbol
    tables = []
    for directory in directories:
        table = pd.read_csv(directory / "metrics.csv")
        table["source"] = str(directory.resolve())
        tables.append(table)
    data = pd.concat(tables, ignore_index=True)
    features, labels, sources = [], [], []
    signatures = set()
    for _, group in data.groupby(["source", "point_id"], sort=True):
        label = benchmark_oracle(group, objective, complexity_weight, papr_weight)
        row = group.iloc[0]
        # N, sample rate and detector define a model domain, not hidden features.
        signatures.add((int(row.frame_size), float(row.bandwidth_hz), str(row.detector)))
        features.append([float(row.snr_db), float(row.max_doppler_hz),
                         float(row.delay_span_s), bits_per_symbol(row.modulation)])
        labels.append(label)
        sources.append(str(row.configuration_hash))
    if len(signatures) != 1:
        raise ValueError("train separate selectors for each frame size, sample rate and detector")
    x = np.asarray(features)
    if len(x) == 0:
        raise ValueError("no complete benchmark conditions")
    # Duplicate physical conditions across reruns would leak into region holdout.
    if len(np.unique(x, axis=0)) != len(x):
        raise ValueError("duplicate operating conditions: pool raw counts before training")
    center = x.mean(axis=0)
    scale = np.where(x.std(axis=0) > 0, x.std(axis=0), 1)
    validation = {"held_out_regions": 0, "accuracy": None}
    if len(x) >= 5:
        held_out = np.arange(0, len(x), 5)
        training = np.setdiff1d(np.arange(len(x)), held_out)
        fit_scale = np.where(x[training].std(axis=0) > 0, x[training].std(axis=0), 1)
        model = NearestRegionSelector(x[training].tolist(), [labels[i] for i in training],
                                      x[training].mean(axis=0).tolist(), fit_scale.tolist(), {})
        predictions = model.predict(x[held_out])
        validation = {"held_out_regions": len(held_out), "accuracy": float(np.mean(
            np.asarray(predictions) == np.asarray(labels)[held_out]))}
    metadata = {
        "features": FEATURES, "objective": objective, "complexity_weight": complexity_weight,
        "papr_weight": papr_weight, "domain": list(next(iter(signatures))),
        "configuration_hashes": sorted(set(sources)), "validation": validation,
        "training_regions": len(x), "feature_min": x.min(axis=0).tolist(),
        "feature_max": x.max(axis=0).tolist(),
        "interpretation": "Nearest finite-sample benchmark winner; not proof of superiority. "
                          "Runtime utility is hardware-dependent. Outside-domain predictions are extrapolation.",
    }
    return NearestRegionSelector(features, labels, center.tolist(), scale.tolist(), metadata)
