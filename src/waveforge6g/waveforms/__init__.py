"""Unitary OFDM, OTFS, and AFDM waveform implementations."""

from .afdm import AFDM
from .base import Waveform
from .ofdm import OFDM
from .otfs import OTFS

__all__ = ["AFDM", "OFDM", "OTFS", "Waveform", "create_waveform"]


def create_waveform(
    name: str,
    n_symbols: int,
    cp_length: int,
    subcarriers: int,
    c1: float | None = None,
    c2: float | None = None,
) -> Waveform:
    """Construct a waveform using the common frame resource budget."""
    normalized = name.lower()
    if normalized == "ofdm":
        return OFDM(n_symbols, cp_length)
    if normalized == "otfs":
        return OTFS(n_symbols, cp_length, subcarriers)
    if normalized == "afdm":
        return AFDM(n_symbols, cp_length, c1, c2)
    raise ValueError(f"unknown waveform {name!r}; expected ofdm, otfs, or afdm")
