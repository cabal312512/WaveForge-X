"""Strict YAML configuration, canonical serialization and provenance hashing."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .channels.profiles import SYNTHETIC_PROFILES


class ConfigError(ValueError):
    """A configuration field is invalid; messages include its location."""


DEFAULTS: dict[str, dict[str, Any]] = {
    "experiment": {"name": "experiment", "seed": 42, "kind": "ber"},
    "system": {
        "waveforms": ["ofdm", "otfs", "afdm"], "modulation": "qpsk",
        "frame_size": 64, "cp_length": 8, "subcarriers": 8,
        "sample_rate_hz": 64000.0, "c1": None, "c2": None,
    },
    "channel": {
        "model": "awgn", "snr_db": 10.0, "max_doppler_hz": None,
        "profile": None, "paths": None, "normalize_power": True, "k_factor_db": 6.0,
    },
    "receiver": {
        "detectors": ["lmmse"], "solver": "auto", "dense_threshold": 128,
        "rtol": 1e-9, "maxiter": None,
    },
    "monte_carlo": {
        "max_frames": 100, "min_frames": 10, "min_bit_errors": 200, "progress": True,
    },
    "sweep": {},
    "plotting": {"enabled": True, "papr_oversampling": 4},
}


def _number(value: Any, location: str, minimum: float | None = None,
            integer: bool = False) -> None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ConfigError(f"{location}: expected {'integer' if integer else 'number'}")
    if integer and not isinstance(value, int):
        raise ConfigError(f"{location}: expected integer, not floating-point value")
    if isinstance(value, float) and not math.isfinite(value):
        raise ConfigError(f"{location}: expected finite {'integer' if integer else 'number'}")
    if minimum is not None and value < minimum:
        raise ConfigError(f"{location}: must be >= {minimum}")


def _choice(value: Any, choices: tuple[str, ...], location: str) -> None:
    if value not in choices:
        raise ConfigError(f"{location}: expected one of {', '.join(choices)}; got {value!r}")


@dataclass(frozen=True)
class ExperimentConfig:
    """Validated resolved configuration. Use to_dict() for a defensive copy."""

    data: dict[str, Any]

    @property
    def config_hash(self) -> str:
        """SHA256 over canonical JSON, including defaults and all sweep settings."""
        payload = json.dumps(self.data, sort_keys=True, separators=(",", ":"), allow_nan=False)
        return hashlib.sha256(payload.encode()).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self.data)

    def __getitem__(self, key: str) -> Any:
        return copy.deepcopy(self.data[key])


def parse_config(raw: dict[str, Any]) -> ExperimentConfig:
    """Resolve defaults and reject invalid fields before starting an experiment."""
    if not isinstance(raw, dict):
        raise ConfigError("configuration: expected a YAML mapping")
    if any(not isinstance(key, str) for key in raw):
        raise ConfigError("configuration: section names must be strings")
    extra = set(raw) - set(DEFAULTS)
    if extra:
        raise ConfigError(f"configuration: unknown sections {sorted(extra)}")
    data = copy.deepcopy(DEFAULTS)
    for section, values in raw.items():
        if not isinstance(values, dict):
            raise ConfigError(f"{section}: expected a mapping")
        if any(not isinstance(key, str) for key in values):
            raise ConfigError(f"{section}: field names must be strings")
        allowed = set(DEFAULTS[section]) if section != "sweep" else {
            "snr_db", "max_doppler_hz", "frame_size",
        }
        if set(values) - allowed:
            raise ConfigError(f"{section}: unknown fields {sorted(set(values) - allowed)}")
        data[section].update(copy.deepcopy(values))
    e, s, c, r, m = (data[key] for key in
                      ("experiment", "system", "channel", "receiver", "monte_carlo"))
    if not isinstance(e["name"], str) or not re.fullmatch(r"[A-Za-z0-9_-]+", e["name"]):
        raise ConfigError("experiment.name: use letters, numbers, underscores or hyphens")
    _number(e["seed"], "experiment.seed", 0, True)
    _choice(e["kind"], ("ber", "papr", "complexity"), "experiment.kind")
    _choice(s["modulation"], ("bpsk", "qpsk", "16qam", "64qam", "qam16", "qam64"),
            "system.modulation")
    s["modulation"] = {"16qam": "qam16", "64qam": "qam64"}.get(
        s["modulation"], s["modulation"])
    for field, options in (("waveforms", ("ofdm", "otfs", "afdm")),):
        if not isinstance(s[field], list) or not s[field]:
            raise ConfigError(f"system.{field}: expected a nonempty list")
        for value in s[field]:
            _choice(value, options, f"system.{field}")
        if len(set(s[field])) != len(s[field]):
            raise ConfigError(f"system.{field}: duplicates are not allowed")
    for field, minimum in (("frame_size", 2), ("cp_length", 0), ("subcarriers", 1)):
        _number(s[field], f"system.{field}", minimum, True)
    _number(s["sample_rate_hz"], "system.sample_rate_hz", 1e-12)
    for field in ("c1", "c2"):
        if s[field] is not None:
            _number(s[field], f"system.{field}")
    _choice(c["model"], ("awgn", "flat_rayleigh", "flat_rician", "static_tdl",
                          "doubly_selective"), "channel.model")
    _number(c["snr_db"], "channel.snr_db")
    _number(c["k_factor_db"], "channel.k_factor_db")
    if c["max_doppler_hz"] is not None:
        _number(c["max_doppler_hz"], "channel.max_doppler_hz", 0)
    if not isinstance(c["normalize_power"], bool):
        raise ConfigError("channel.normalize_power: expected boolean")
    if c["profile"] is not None:
        _choice(c["profile"], ("pedestrian", "urban_vehicle", "high_speed_train",
                              "extreme_mobility"), "channel.profile")
    if c["profile"] is not None and c["paths"] is not None:
        raise ConfigError("channel: specify either profile or paths, not both")
    if c["paths"] is not None:
        if not isinstance(c["paths"], list) or not c["paths"]:
            raise ConfigError("channel.paths: expected a nonempty list")
        for index, path in enumerate(c["paths"]):
            loc = f"channel.paths[{index}]"
            if not isinstance(path, dict) or "delay_samples" not in path:
                raise ConfigError(f"{loc}: mapping with delay_samples required")
            if set(path) - {"delay_samples", "power_db", "doppler_hz", "gain"}:
                raise ConfigError(f"{loc}: unknown path field")
            _number(path["delay_samples"], f"{loc}.delay_samples", 0, True)
            for field in ("power_db", "doppler_hz"):
                if field in path:
                    _number(path[field], f"{loc}.{field}")
            if "gain" in path:
                if not isinstance(path["gain"], list) or len(path["gain"]) != 2:
                    raise ConfigError(f"{loc}.gain: expected [real, imaginary]")
                for value in path["gain"]:
                    _number(value, f"{loc}.gain")
    if not isinstance(r["detectors"], list) or not r["detectors"]:
        raise ConfigError("receiver.detectors: expected nonempty list")
    for value in r["detectors"]:
        _choice(value, ("zf", "lmmse", "one_tap_zf", "one_tap_mmse"), "receiver.detectors")
    if len(set(r["detectors"])) != len(r["detectors"]):
        raise ConfigError("receiver.detectors: duplicates are not allowed")
    if any(value.startswith("one_tap") for value in r["detectors"]):
        if s["waveforms"] != ["ofdm"] or c["model"] == "doubly_selective":
            raise ConfigError("one_tap detectors require only OFDM and a static channel")
    _choice(r["solver"], ("auto", "dense", "iterative", "matrix_free", "lsmr"),
            "receiver.solver")
    if r["solver"] in ("iterative", "lsmr"):
        r["solver"] = "matrix_free"
    _number(r["dense_threshold"], "receiver.dense_threshold", 2, True)
    if r["dense_threshold"] > 1024:
        raise ConfigError("receiver.dense_threshold: must be <= 1024")
    _number(r["rtol"], "receiver.rtol", 1e-15)
    if r["rtol"] >= 1:
        raise ConfigError("receiver.rtol: must be < 1")
    if r["maxiter"] is not None:
        _number(r["maxiter"], "receiver.maxiter", 1, True)
    for field, minimum in (("max_frames", 1), ("min_frames", 1), ("min_bit_errors", 0)):
        _number(m[field], f"monte_carlo.{field}", minimum, True)
    if m["min_frames"] > m["max_frames"]:
        raise ConfigError("monte_carlo.min_frames: cannot exceed max_frames")
    for loc, value in (("monte_carlo.progress", m["progress"]),
                       ("plotting.enabled", data["plotting"]["enabled"])):
        if not isinstance(value, bool):
            raise ConfigError(f"{loc}: expected boolean")
    _number(data["plotting"]["papr_oversampling"], "plotting.papr_oversampling", 1, True)
    for key, values in data["sweep"].items():
        if not isinstance(values, list) or not values:
            raise ConfigError(f"sweep.{key}: expected a nonempty list")
        for value in values:
            _number(value, f"sweep.{key}", 2 if key == "frame_size" else
                    (0 if key == "max_doppler_hz" else None), key == "frame_size")
        if len(set(values)) != len(values):
            raise ConfigError(f"sweep.{key}: duplicates are not allowed")
    for size in data["sweep"].get("frame_size", [s["frame_size"]]):
        if s["cp_length"] >= size:
            raise ConfigError("system.cp_length: must be smaller than every frame_size")
        if "otfs" in s["waveforms"] and size % s["subcarriers"]:
            raise ConfigError("system.subcarriers: must divide every frame_size for OTFS")
        if r["solver"] == "dense" and size > 1024:
            raise ConfigError("system.frame_size: dense solver supports at most 1024 symbols")
    if c["model"] in ("static_tdl", "doubly_selective"):
        if c["paths"] is not None:
            maximum_delay = max(path["delay_samples"] for path in c["paths"])
        elif c["profile"] is not None:
            maximum_delay = max(SYNTHETIC_PROFILES[c["profile"]]["delays"])
        else:
            maximum_delay = max(SYNTHETIC_PROFILES["urban_vehicle"]["delays"])
        if maximum_delay > s["cp_length"]:
            raise ConfigError("system.cp_length: must cover the largest configured channel delay")
    if c["model"] == "static_tdl" and c["paths"] is not None:
        if any(path.get("doppler_hz", 0.0) != 0 for path in c["paths"]):
            raise ConfigError("channel.paths: static_tdl requires zero Doppler on every path")
    if c["model"] in ("doubly_selective", "flat_rayleigh", "flat_rician"):
        limits = list(data["sweep"].get("max_doppler_hz", []))
        if c["max_doppler_hz"] is not None:
            limits.append(c["max_doppler_hz"])
        elif not limits and c["profile"] is not None:
            limits.append(SYNTHETIC_PROFILES[c["profile"]]["max_doppler_hz"])
        elif not limits and c["model"] == "doubly_selective" and c["paths"] is None:
            limits.append(SYNTHETIC_PROFILES["urban_vehicle"]["max_doppler_hz"])
        if not limits and c["paths"] is not None:
            limits.extend(abs(path.get("doppler_hz", 0.0)) for path in c["paths"])
        if any(value >= s["sample_rate_hz"] / 2 for value in limits):
            raise ConfigError("channel Doppler: must be strictly below sample_rate_hz / 2")
        if any(value > 0 for value in limits) and any(
            value.startswith("one_tap") for value in r["detectors"]
        ):
            raise ConfigError("one_tap detectors require zero Doppler on every channel path")
    return ExperimentConfig(data)


def load_config(path: str | Path) -> ExperimentConfig:
    """Read UTF-8 YAML and return a validated resolved configuration."""
    try:
        with Path(path).open(encoding="utf-8-sig") as stream:
            raw = yaml.safe_load(stream)
    except yaml.YAMLError as error:
        raise ConfigError(f"{path}: invalid YAML: {error}") from error
    return parse_config(raw)
