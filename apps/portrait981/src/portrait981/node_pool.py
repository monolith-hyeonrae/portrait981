"""ComfyUI node pool — round-robin selection with health tracking."""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Node is considered unhealthy for this duration after a failure.
_COOLDOWN_SEC = 60.0


@dataclass
class _NodeState:
    url: str
    healthy: bool = True
    last_failure: float = 0.0
    in_flight: int = 0


class NodePool:
    """Round-robin ComfyUI node selector with failure cooldown.

    Usage::

        pool = NodePool(["http://gpu1:8188", "http://gpu2:8188"])
        url = pool.acquire()      # next healthy node
        try:
            result = generate(url)
            pool.release(url)
        except Exception:
            pool.mark_unhealthy(url)
            pool.release(url)
    """

    def __init__(self, urls: list[str]) -> None:
        if not urls:
            raise ValueError("At least one ComfyUI URL required")
        self._nodes = [_NodeState(url=u) for u in urls]
        self._lock = threading.Lock()
        self._index = 0

    @property
    def urls(self) -> list[str]:
        return [n.url for n in self._nodes]

    def acquire(self) -> str:
        """Select the next healthy node (round-robin). Recovers cooled-down nodes."""
        with self._lock:
            now = time.monotonic()
            n = len(self._nodes)

            # Try to find a healthy node starting from current index
            for _ in range(n):
                node = self._nodes[self._index % n]
                self._index += 1

                # Recover from cooldown
                if not node.healthy and (now - node.last_failure) >= _COOLDOWN_SEC:
                    node.healthy = True
                    logger.info("Node recovered: %s", node.url)

                if node.healthy:
                    node.in_flight += 1
                    return node.url

            # All nodes unhealthy — force-recover the least recently failed
            fallback = min(self._nodes, key=lambda n: n.last_failure)
            fallback.healthy = True
            fallback.in_flight += 1
            logger.warning("All nodes unhealthy, force-recovering: %s", fallback.url)
            return fallback.url

    def release(self, url: str) -> None:
        """Release a node after use."""
        with self._lock:
            for node in self._nodes:
                if node.url == url:
                    node.in_flight = max(0, node.in_flight - 1)
                    return

    def mark_unhealthy(self, url: str) -> None:
        """Mark a node as unhealthy after a failure."""
        with self._lock:
            for node in self._nodes:
                if node.url == url:
                    node.healthy = False
                    node.last_failure = time.monotonic()
                    logger.warning("Node marked unhealthy: %s", url)
                    return

    def status(self) -> list[dict]:
        """Return node pool status for diagnostics."""
        with self._lock:
            now = time.monotonic()
            return [
                {
                    "url": n.url,
                    "healthy": n.healthy,
                    "in_flight": n.in_flight,
                    "cooldown_remaining": max(
                        0, _COOLDOWN_SEC - (now - n.last_failure)
                    )
                    if not n.healthy
                    else 0,
                }
                for n in self._nodes
            ]
