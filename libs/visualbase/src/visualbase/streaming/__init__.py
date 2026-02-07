"""Streaming module for real-time video distribution.

This module provides:
- ProxyFanout: Distribute frames from a single source to multiple FIFO outputs
"""

from visualbase.streaming.fanout import ProxyFanout, ProxyConfig

__all__ = [
    "ProxyFanout",
    "ProxyConfig",
]
