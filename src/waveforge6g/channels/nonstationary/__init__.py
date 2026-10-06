"""Synthetic nonstationary propagation with causal, reproducible evolution."""

from .engine import NonStationaryChannel
from .scenarios import SCENARIO_NAMES, scenario_config
from .state import TrueChannelState
from .trajectory import ScalarTrajectory

__all__ = [
    "NonStationaryChannel",
    "SCENARIO_NAMES",
    "ScalarTrajectory",
    "TrueChannelState",
    "scenario_config",
]
