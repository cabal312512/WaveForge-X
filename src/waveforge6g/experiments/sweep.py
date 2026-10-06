"""Deterministic Cartesian experimental design."""

from itertools import product
from typing import Any

from ..config import ExperimentConfig


def sweep_points(config: ExperimentConfig) -> list[dict[str, Any]]:
    """Return canonical physical points independent of YAML key ordering."""
    system, channel, sweep = config["system"], config["channel"], config["sweep"]
    axes = {
        "snr_db": sweep.get("snr_db", [channel["snr_db"]]),
        "max_doppler_hz": sweep.get("max_doppler_hz", [channel["max_doppler_hz"]]),
        "frame_size": sweep.get("frame_size", [system["frame_size"]]),
    }
    return [dict(zip(axes, values, strict=True)) for values in product(*axes.values())]
