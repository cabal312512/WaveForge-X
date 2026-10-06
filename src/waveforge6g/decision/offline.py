"""Train-only NumPy multinomial logistic classifier with strict provenance.

Class probabilities predict the empirical best base-loss action. A companion
ridge fit estimates per-action mean base rewards for switching comparisons;
classifier probabilities are never mislabeled as reward predictions.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ..reproducibility import project_root
from .base import BaseSelector, DecisionContext, DecisionResult
from .objective import Objective


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=-1, keepdims=True)


def _domain(config) -> dict:
    system = config["system"]
    return {"system": {key: system.get(key) for key in (
        "frame_size", "cp_length", "subcarriers", "sample_rate_hz", "modulation", "c1", "c2")},
        "detector": config["receiver"]["detector"]}


def _expand_directories(directories: list[Path]) -> list[Path]:
    expanded = []
    for raw in directories:
        path = Path(raw)
        manifest = path / "runs.json"
        if manifest.exists():
            for row in json.loads(manifest.read_text(encoding="utf-8")):
                child = Path(row["directory"])
                expanded.append(child if child.is_absolute() else path / child)
        else:
            expanded.append(path)
    return list(dict.fromkeys(path.resolve() for path in expanded))


def train_offline(directories: list[Path], output: Path) -> dict:
    """Fit deterministic softmax/ridge models only from split=train run dirs.

    Input runs must contain paired oracle_outcomes.csv and context_features in
    trajectory.csv. Only features [:5] are used (intercept and four channel
    estimates); past policy BER and previous actions are deliberately omitted.
    No validation/test data enter normalization, coefficient fitting, or labels.
    """
    features, targets, labels, provenance = [], [], [], []
    domain = objective_settings = None
    seen_epochs = set()
    for directory in _expand_directories(directories):
        config = yaml.safe_load((directory / "config.yaml").read_text(encoding="utf-8"))
        if config["experiment"].get("split") != "train":
            raise ValueError(f"offline training requires split=train: {directory}")
        this_domain = _domain(config)
        objective = Objective(config.get("objective"))
        if domain is None:
            domain, objective_settings = this_domain, objective.to_dict()
        if this_domain != domain or objective.to_dict() != objective_settings:
            raise ValueError("offline training runs must share physical domain and objective")
        trajectory = pd.read_csv(directory / "trajectory.csv")
        outcomes = pd.read_csv(directory / "oracle_outcomes.csv")
        if not (trajectory["split"] == "train").all():
            raise ValueError("trajectory rows must all be split=train")
        if trajectory.empty or outcomes.empty:
            raise ValueError("offline training runs must contain measured samples")
        identity = str(config["dynamic"]["scenario"])
        for (seed, time), group in trajectory.groupby(["seed", "time"], sort=True):
            signature = (identity, int(seed), int(time), str(group.iloc[0].configuration_hash))
            if signature in seen_epochs:
                raise ValueError("duplicate training epochs would overweight a trajectory")
            seen_epochs.add(signature)
            vectors = np.array([json.loads(value)[:5] for value in group.context_features], dtype=float)
            if vectors.shape[1:] != (5,) or not np.all(np.isfinite(vectors)):
                raise ValueError("training contexts require five finite observable features")
            if not np.allclose(vectors, vectors[0], rtol=0, atol=0):
                raise ValueError("channel context features differ across policies on a paired epoch")
            paired = outcomes[(outcomes.seed == seed) & (outcomes.time == time)].sort_values("action")
            if len(paired) != 3 or paired.action.tolist() != [0, 1, 2]:
                raise ValueError("training labels require exactly three paired simulator outcomes")
            losses = paired.base_loss.to_numpy(float)
            if not np.all(np.isfinite(losses)) or np.any((losses < 0) | (losses > 1)):
                raise ValueError("training base losses must be finite and in [0,1]")
            features.append(vectors[0])
            targets.append(-losses)
            labels.append(int(np.argmin(losses)))
        digest = hashlib.sha256()
        for filename in ("config.yaml", "trajectory.csv", "oracle_outcomes.csv"):
            digest.update((directory / filename).read_bytes())
        provenance.append({"directory": str(directory), "scenario": identity,
                           "seeds": sorted(int(x) for x in trajectory.seed.unique()),
                           "configuration_hashes": sorted(str(x) for x in trajectory.configuration_hash.unique()),
                           "sha256": digest.hexdigest()})
    if not features:
        raise ValueError("no offline training observations supplied")
    x, y = np.asarray(features), np.asarray(labels)
    center, scale = x.mean(axis=0), x.std(axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    design = np.column_stack((np.ones(len(x)), (x - center) / scale))
    onehot = np.eye(3)[y]
    weights = np.zeros((design.shape[1], 3))
    l2 = 0.01
    learning_rate = 1 / (0.5 * np.linalg.norm(design, ord=2)**2 / len(x) + l2)
    # Fixed iteration budget avoids any implicit test/validation tuning.
    for _ in range(1000):
        gradient = design.T @ (_softmax(design @ weights) - onehot) / len(x)
        gradient[1:] += l2 * weights[1:]
        weights -= learning_rate * gradient
    penalty = np.eye(design.shape[1]) * l2
    penalty[0, 0] = 0
    reward_weights = np.linalg.solve(design.T @ design / len(x) + penalty,
                                     design.T @ np.asarray(targets) / len(x))
    model = {"format": "waveforge_x_offline_v1", "feature_count": 5,
             "feature_indices": list(range(5)), "center": center.tolist(), "scale": scale.tolist(),
             "coefficients": weights.tolist(), "reward_coefficients": reward_weights.tolist(),
             "training": provenance, "training_seeds": sorted({seed for row in provenance for seed in row["seeds"]}),
             "domain": domain, "objective": objective_settings, "n_samples": len(x),
             "training_accuracy": float(np.mean(np.argmax(design @ weights, axis=1) == y)),
             "hyperparameters": {"iterations": 1000, "l2": l2, "tuning": "fixed_before_test"}}
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(model, indent=2, allow_nan=False), encoding="utf-8")
    return model


class OfflineSelector(BaseSelector):
    """Frozen training-only softmax action model; online updates do not refit."""

    def __init__(self, model: str | Path, objective: Objective | None = None) -> None:
        self.path = Path(model)
        if not self.path.is_absolute():
            self.path = project_root() / self.path
        model_bytes = self.path.read_bytes()
        self.model = json.loads(model_bytes)
        self.model_sha256 = hashlib.sha256(model_bytes).hexdigest()
        if self.model.get("format") != "waveforge_x_offline_v1":
            raise ValueError("unsupported offline model format")
        self.center, self.scale = np.array(self.model["center"]), np.array(self.model["scale"])
        self.coefficients = np.array(self.model["coefficients"])
        self.reward_coefficients = np.array(self.model["reward_coefficients"])
        if self.center.shape != (5,) or self.scale.shape != (5,) or np.any(self.scale <= 0):
            raise ValueError("offline model requires five feature normalization values")
        if self.coefficients.shape != (6, 3) or self.reward_coefficients.shape != (6, 3):
            raise ValueError("offline model coefficients require shape [6,3]")
        if not all(np.all(np.isfinite(x)) for x in (self.center, self.scale, self.coefficients, self.reward_coefficients)):
            raise ValueError("offline model coefficients must be finite")
        self.objective = objective if objective is not None else Objective(self.model["objective"])
        self._validate_objective(self.objective)

    def _validate_objective(self, objective: Objective) -> None:
        trained = Objective(self.model["objective"])
        keys = ("ber_weight", "bler_weight", "papr_weight", "complexity_weight",
                "papr_reference_db", "complexity_reference")
        if any(trained.config[key] != objective.config[key] for key in keys):
            raise ValueError("offline base objective differs from training domain")
        self.reward_rescale = trained.total_weight / objective.total_weight

    def validate_domain(self, config, seed: int) -> None:
        if _domain(config) != self.model["domain"]:
            raise ValueError("offline physical domain differs from training domain")
        if config["experiment"]["split"] in ("validation", "test") and seed in self.model["training_seeds"]:
            raise ValueError("offline evaluation seed overlaps training seeds")
        self._validate_objective(Objective(config["objective"]))

    def select(self, context: DecisionContext) -> DecisionResult:
        if len(context.features) < 5:
            raise ValueError("offline selector requires at least five context features")
        x = np.concatenate(([1.0], (np.array(context.features[:5]) - self.center) / self.scale))
        probabilities = _softmax(x @ self.coefficients)
        entropy = float(-np.sum(probabilities * np.log(np.maximum(probabilities, 1e-300))) / np.log(3))
        scores = np.clip(x @ self.reward_coefficients * self.reward_rescale, -1, 0)
        return DecisionResult(int(np.argmax(probabilities)), scores, float(probabilities.max()),
                              float(np.clip(entropy, 0, 1)), {"probabilities": probabilities.tolist(),
                                                           "model_sha256": self.model_sha256})

    def state_dict(self) -> dict:
        return {"selector": type(self).__name__, "model": str(self.path),
                "training_seeds": self.model["training_seeds"], "n_samples": self.model["n_samples"]}


def check_offline_protocol(spec: dict, config, seed: int) -> None:
    """Runner-side protocol check even when the policy is wrapped."""
    if spec.get("type") == "offline":
        OfflineSelector(spec["model"], Objective(config["objective"])).validate_domain(config, seed)
