"""Shared, noise-free channel realizations and synthetic profile factory."""

from .doubly_selective import ChannelRealization
from .tapped_delay_line import realize_channel

__all__ = ["ChannelRealization", "realize_channel"]
