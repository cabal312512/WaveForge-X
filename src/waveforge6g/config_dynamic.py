"""Configuration for causal, multi-seed WaveForge-X decision experiments."""

from __future__ import annotations

import copy
import itertools
import re
from pathlib import Path
from typing import Any

import yaml

from .channels.nonstationary import SCENARIO_NAMES, scenario_config
from .channels.observation import ObservationModel
from .config import ConfigError, ExperimentConfig, _number, parse_config
from .decision.objective import DEFAULT_OBJECTIVE, Objective

_DYNAMIC_FIELDS = {
    "scenario",
    "epochs",
    "change_interval",
    "snr_db",
    "max_doppler_hz",
    "delay_samples",
    "k_factor_db",
    "correlation",
    "path_count",
    "fractional_doppler_severity",
    "trajectories",
    "cluster_dynamics",
    "dynamics",
}
_OBSERVATION_FIELDS = {
    "snr_std",
    "doppler_relative_error",
    "delay_spread_relative_error",
    "delay_frames",
    "perfect",
    "doppler_std_hz",
    "delay_spread_std_s",
    "biases",
    "snr_bias_db",
    "doppler_bias_hz",
    "delay_spread_bias_s",
    "snr_variation_scale_db",
    "doppler_variation_scale_hz",
    "delay_variation_scale_s",
    "reference_bandwidth_hz",
    "sample_rate_hz",
    "coherence_cap_s",
}
_CLUSTER_FIELDS = {
    "birth_probability",
    "death_probability",
    "cluster_birth_probability",
    "cluster_death_probability",
    "lifetime",
    "power_ramp_frames",
    "max_paths",
    "cluster_size",
    "delay_drift_std",
    "doppler_drift_std",
    "power_drift_std",
}
_TRAJECTORY_FIELDS = {
    "constant": {"value", "start"},
    "linear": {"start", "end", "slope"},
    "sinusoidal": {"offset", "amplitude", "period", "phase"},
    "random_walk": {"initial", "start", "std", "step_std"},
    "piecewise_constant": {"points", "values", "breakpoints"},
    "piecewise_linear": {"points", "values", "breakpoints"},
    "abrupt": {"time", "change_at", "before", "after"},
    "smooth": {"start_time", "end_time", "before", "after"},
}


def _keys(values: Any, allowed: set, location: str) -> None:
    if not isinstance(values, dict):
        raise ConfigError(f"{location}: expected a mapping")
    if set(values) - allowed:
        raise ConfigError(f"{location}: unknown fields {sorted(set(values) - allowed)}")


def _validate_trajectory(spec: Any, location: str) -> None:
    if not isinstance(spec, dict):
        _number(spec, location)
        return
    kind = spec.get("type", spec.get("kind", "constant"))
    if not isinstance(kind, str) or kind not in _TRAJECTORY_FIELDS:
        raise ConfigError(f"{location}: unknown trajectory type {kind!r}")
    _keys(spec, _TRAJECTORY_FIELDS[kind] | {"type", "kind", "min", "max"}, location)
    if "type" in spec and "kind" in spec and spec["type"] != spec["kind"]:
        raise ConfigError(f"{location}: type and kind disagree")
    for key, value in spec.items():
        if key not in {"type", "kind", "points", "values", "breakpoints"}:
            _number(
                value,
                f"{location}.{key}",
                0 if key in {"std", "step_std"} else None,
                key in {"time", "change_at"},
            )
    if spec.get("min", float("-inf")) > spec.get("max", float("inf")):
        raise ConfigError(f"{location}: min cannot exceed max")
    if "period" in spec and spec["period"] <= 0:
        raise ConfigError(f"{location}.period must be positive")
    if kind == "smooth" and "start_time" in spec and "end_time" in spec:
        if spec["end_time"] <= spec["start_time"]:
            raise ConfigError(f"{location}: smooth transition must have positive duration")
    if kind.startswith("piecewise"):
        if "points" in spec:
            if "values" in spec or "breakpoints" in spec:
                raise ConfigError(f"{location}: choose points or values/breakpoints, not both")
            points = spec["points"]
            if not isinstance(points, list) or not points:
                raise ConfigError(f"{location}.points must be nonempty")
            previous = -1
            for index, point in enumerate(points):
                if not isinstance(point, list) or len(point) != 2:
                    raise ConfigError(f"{location}.points must contain [integer epoch, value]")
                _number(point[0], f"{location}.points[{index}].time", 0, True)
                _number(point[1], f"{location}.points[{index}].value")
                if point[0] <= previous or (index == 0 and point[0] != 0):
                    raise ConfigError(f"{location}: points must start at 0 and increase strictly")
                previous = point[0]
        else:
            values, breaks = spec.get("values"), spec.get("breakpoints", [])
            if (
                not isinstance(values, list)
                or not isinstance(breaks, list)
                or len(values) != len(breaks) + 1
            ):
                raise ConfigError(
                    f"{location}: piecewise values require one more value than breakpoints"
                )
            previous = 0
            for value in values:
                _number(value, f"{location}.values")
            for epoch in breaks:
                _number(epoch, f"{location}.breakpoints", 1, True)
                if epoch <= previous:
                    raise ConfigError(f"{location}: breakpoints must increase strictly")
                previous = epoch


def _validate_dynamic(values: dict, system: dict) -> None:
    if values["scenario"] not in SCENARIO_NAMES:
        raise ConfigError("dynamic.scenario: unknown synthetic scenario")
    for key in ("epochs", "change_interval"):
        _number(values[key], f"dynamic.{key}", 1, True)
    parameters = {
        "snr_db",
        "max_doppler_hz",
        "delay_samples",
        "k_factor_db",
        "correlation",
        "path_count",
        "fractional_doppler_severity",
        "delay_spread_s",
    }
    for key in parameters & set(values):
        _number(
            values[key],
            f"dynamic.{key}",
            None if key in {"snr_db", "k_factor_db"} else 0,
            key == "path_count",
        )
    for key in ("correlation", "fractional_doppler_severity"):
        if key in values and not 0 <= values[key] <= 1:
            raise ConfigError(f"dynamic.{key} must lie in [0,1]")
    if values.get("max_doppler_hz", 0) >= system["sample_rate_hz"] / 2:
        raise ConfigError("dynamic.max_doppler_hz must be below sample-rate Nyquist")
    if "path_count" in values and values["path_count"] < 1:
        raise ConfigError("dynamic.path_count must be positive")
    if "cluster_dynamics" in values and "dynamics" in values:
        raise ConfigError("dynamic: specify cluster_dynamics once, not also dynamics")
    cluster = values.get("cluster_dynamics", values.get("dynamics", {}))
    _keys(cluster, _CLUSTER_FIELDS, "dynamic.cluster_dynamics")
    for key, value in cluster.items():
        location = f"dynamic.cluster_dynamics.{key}"
        if key == "lifetime":
            if value is not None:
                if isinstance(value, list):
                    if len(value) != 2:
                        raise ConfigError(f"{location} requires [min,max]")
                    for bound in value:
                        _number(bound, location, 1, True)
                    if value[0] > value[1]:
                        raise ConfigError(f"{location}: minimum exceeds maximum")
                else:
                    _number(value, location, 1, True)
        else:
            integer = key in {"power_ramp_frames", "max_paths", "cluster_size"}
            _number(value, location, 1 if key in {"max_paths", "cluster_size"} else 0, integer)
            if key.endswith("probability") and value > 1:
                raise ConfigError(f"{location} must lie in [0,1]")
    if values.get("path_count", 3) > cluster.get("max_paths", 10):
        raise ConfigError("dynamic.path_count exceeds cluster max_paths")
    trajectories = values.get("trajectories", {})
    _keys(trajectories, parameters, "dynamic.trajectories")
    for key, spec in trajectories.items():
        _validate_trajectory(spec, f"dynamic.trajectories.{key}")
    resolved = scenario_config(values)
    for key, spec in resolved["trajectories"].items():
        if key in {"snr_db", "k_factor_db"}:
            continue
        if isinstance(spec, dict):
            kind = spec.get("type", spec.get("kind", "constant"))
            candidates = [
                spec[name]
                for name in ("value", "initial", "start", "end", "before", "after")
                if name in spec
            ]
            if kind == "linear" and "slope" in spec:
                candidates.append(spec.get("start", 0) + (values["epochs"] - 1) * spec["slope"])
            elif kind == "sinusoidal":
                candidates.extend(
                    [
                        spec.get("offset", 0) - abs(spec.get("amplitude", 1)),
                        spec.get("offset", 0) + abs(spec.get("amplitude", 1)),
                    ]
                )
            elif kind.startswith("piecewise"):
                candidates.extend([point[1] for point in spec.get("points", [])])
                candidates.extend(spec.get("values", []))
            candidates = [
                max(spec.get("min", float("-inf")), min(value, spec.get("max", float("inf"))))
                for value in candidates
            ]
        else:
            candidates = [spec]
        minimum = 1 if key == "path_count" else 0
        maximum = (
            1
            if key in {"correlation", "fractional_doppler_severity"}
            else resolved["cluster_dynamics"]["max_paths"]
            if key == "path_count"
            else float("inf")
        )
        if any(value < minimum or value > maximum for value in candidates):
            raise ConfigError(f"dynamic.trajectories.{key}: values outside physical range")
        if key == "max_doppler_hz" and any(
            value >= system["sample_rate_hz"] / 2 for value in candidates
        ):
            raise ConfigError("dynamic.trajectories.max_doppler_hz must stay below Nyquist")


DYNAMIC_DEFAULTS = {
    "experiment": {"name": "adaptive", "seeds": [11, 22, 33], "split": "test"},
    "system": {
        "frame_size": 32,
        "cp_length": 8,
        "subcarriers": 8,
        "sample_rate_hz": 64000.0,
        "modulation": "qpsk",
        "c1": None,
        "c2": None,
    },
    "receiver": {
        "detector": "lmmse",
        "solver": "auto",
        "dense_threshold": 64,
        "rtol": 1e-8,
        "maxiter": None,
    },
    "dynamic": {"scenario": "mixed_extreme_mobility", "epochs": 120, "change_interval": 30},
    "observation": {
        "snr_std": 1.0,
        "doppler_relative_error": 0.1,
        "delay_spread_relative_error": 0.1,
        "delay_frames": 1,
    },
    "objective": {
        "ber_weight": 1.0,
        "bler_weight": 0.2,
        "papr_weight": 0.05,
        "complexity_weight": 0.05,
        "switching_weight": 0.1,
        "papr_reference_db": 12.0,
        "complexity_reference": 100000.0,
    },
    "selectors": [{"name": "always_ofdm", "type": "static", "action": 0}],
    "sweep": {},
    "cache": {"enabled": True},
}


def parse_dynamic_config(raw: dict[str, Any]) -> ExperimentConfig:
    """Validate research settings and reuse physical-layer configuration checks."""
    if not isinstance(raw, dict):
        raise ConfigError("dynamic configuration must be a mapping")
    if set(raw) - set(DYNAMIC_DEFAULTS):
        raise ConfigError(f"unknown dynamic sections: {sorted(set(raw) - set(DYNAMIC_DEFAULTS))}")
    data = copy.deepcopy(DYNAMIC_DEFAULTS)
    for section, values in raw.items():
        if section == "selectors":
            data[section] = copy.deepcopy(values)
        else:
            allowed = (
                _DYNAMIC_FIELDS
                if section == "dynamic"
                else _OBSERVATION_FIELDS
                if section == "observation"
                else set(DEFAULT_OBJECTIVE)
                if section == "objective"
                else set(values)
                if section == "sweep" and isinstance(values, dict)
                else set(DYNAMIC_DEFAULTS[section])
            )
            _keys(values, allowed, section)
            data[section].update(copy.deepcopy(values))
    e, d, o, r = (data[key] for key in ("experiment", "dynamic", "observation", "receiver"))
    base = parse_config(
        {
            "experiment": {"name": e["name"]},
            "system": data["system"],
            "receiver": {k: v for k, v in r.items() if k != "detector"},
        }
    )
    data["system"] = {k: v for k, v in base["system"].items() if k != "waveforms"}
    if r["detector"] not in ("zf", "lmmse"):
        raise ConfigError("receiver.detector: dynamic links require zf or lmmse")
    if e["split"] not in ("train", "validation", "test"):
        raise ConfigError("experiment.split must be train, validation or test")
    if not isinstance(e["seeds"], list) or not e["seeds"]:
        raise ConfigError("experiment.seeds: expected nonempty list")
    for seed in e["seeds"]:
        _number(seed, "experiment.seeds", 0, True)
    if len(set(e["seeds"])) != len(e["seeds"]):
        raise ConfigError("experiment.seeds: duplicate seeds are not independent replicates")
    _validate_dynamic(d, data["system"])
    for field, value in o.items():
        if field == "perfect":
            if not isinstance(value, bool):
                raise ConfigError("observation.perfect must be boolean")
        elif field == "biases":
            _keys(value, {"snr_db", "doppler_hz", "delay_spread_s"}, "observation.biases")
            for name, bias in value.items():
                _number(bias, f"observation.biases.{name}")
        else:
            _number(
                value,
                f"observation.{field}",
                None if "bias" in field else 0,
                field == "delay_frames",
            )
    o.setdefault(
        "reference_bandwidth_hz", o.get("sample_rate_hz", data["system"]["sample_rate_hz"])
    )
    try:
        ObservationModel(o, seed=0)
    except (TypeError, ValueError) as error:
        raise ConfigError(f"observation: {error}") from error
    for field, value in data["objective"].items():
        if field == "switching_matrix":
            if not isinstance(value, list) or len(value) != 3:
                raise ConfigError("objective.switching_matrix must be a 3 by 3 list")
            for row in value:
                if not isinstance(row, list) or len(row) != 3:
                    raise ConfigError("objective.switching_matrix must be a 3 by 3 list")
                for entry in row:
                    _number(entry, "objective.switching_matrix", 0)
        else:
            _number(value, f"objective.{field}", 0)
    try:
        data["objective"] = Objective(data["objective"]).to_dict()
    except ValueError as error:
        raise ConfigError(f"objective: {error}") from error
    if not isinstance(data["cache"]["enabled"], bool):
        raise ConfigError("cache.enabled must be boolean")
    if not isinstance(data["selectors"], list) or not data["selectors"]:
        raise ConfigError("selectors: expected nonempty list of selector mappings")
    names = set()
    from .decision import validate_selector_spec

    for selector in data["selectors"]:
        if (
            not isinstance(selector, dict)
            or not isinstance(selector.get("name"), str)
            or "type" not in selector
        ):
            raise ConfigError("each selector requires name and type")
        if not re.fullmatch(r"[A-Za-z0-9_-]+", selector["name"]):
            raise ConfigError("selector names must use letters, numbers, underscores or hyphens")
        if selector["name"] in names:
            raise ConfigError("selector names must be unique")
        try:
            validate_selector_spec(selector)
        except (TypeError, ValueError) as error:
            raise ConfigError(f"selectors.{selector['name']}: {error}") from error
        names.add(selector["name"])
    for key, values in data["sweep"].items():
        if key not in {
            "objective.switching_weight",
            "observation.snr_std",
            "observation.error_scale",
            "dynamic.change_interval",
        }:
            raise ConfigError(f"unsupported research sweep {key}")
        if not isinstance(values, list) or not values:
            raise ConfigError(f"sweep.{key}: expected nonempty list")
        for value in values:
            _number(
                value,
                f"sweep.{key}",
                2 if key.endswith("change_interval") else 0,
                key.endswith("change_interval"),
            )
        if len(set(values)) != len(values):
            raise ConfigError(f"sweep.{key}: duplicate conditions are not independent replicates")
    if "observation.snr_std" in data["sweep"] and "observation.error_scale" in data["sweep"]:
        raise ConfigError(
            "cannot sweep SNR error both directly and through observation.error_scale"
        )
    return ExperimentConfig(data)


def load_dynamic_config(path: str | Path) -> ExperimentConfig:
    """Load UTF-8 YAML for the research command."""
    try:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8-sig"))
    except yaml.YAMLError as error:
        raise ConfigError(f"invalid dynamic YAML: {error}") from error
    return parse_dynamic_config(raw)


def research_variants(config: ExperimentConfig) -> list[tuple[dict, ExperimentConfig]]:
    """Cartesian sweeps with concrete resolved settings recorded for every run."""
    keys = sorted(config["sweep"])
    variants = []
    for values in itertools.product(*(config["sweep"][key] for key in keys)):
        raw = config.to_dict()
        label = dict(zip(keys, values, strict=True))
        for key, value in label.items():
            section, field = key.split(".")
            if key == "observation.error_scale":
                # Zero is the explicitly designated perfect-current-context upper bound.
                # Positive levels retain the experiment's declared latency and biases.
                raw["observation"]["perfect"] = value == 0
                for noise in (
                    "snr_std",
                    "doppler_relative_error",
                    "delay_spread_relative_error",
                    "doppler_std_hz",
                    "delay_spread_std_s",
                ):
                    if noise in config["observation"]:
                        raw["observation"][noise] = config["observation"][noise] * value
            else:
                raw[section][field] = value
        raw["sweep"] = {}
        variants.append((label, parse_dynamic_config(raw)))
    return variants
