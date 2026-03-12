"""Tests for ComfyUI node pool."""

from __future__ import annotations

import time
from unittest.mock import patch

from portrait981.node_pool import NodePool, _COOLDOWN_SEC


class TestNodePool:
    def test_round_robin(self):
        pool = NodePool(["http://a:8188", "http://b:8188"])
        first = pool.acquire()
        pool.release(first)
        second = pool.acquire()
        pool.release(second)
        assert first != second

    def test_single_node(self):
        pool = NodePool(["http://a:8188"])
        url = pool.acquire()
        assert url == "http://a:8188"
        pool.release(url)

    def test_empty_raises(self):
        import pytest
        with pytest.raises(ValueError):
            NodePool([])

    def test_mark_unhealthy_skips_node(self):
        pool = NodePool(["http://a:8188", "http://b:8188"])
        pool.mark_unhealthy("http://a:8188")
        # Should only get b
        urls = set()
        for _ in range(4):
            url = pool.acquire()
            urls.add(url)
            pool.release(url)
        assert urls == {"http://b:8188"}

    def test_cooldown_recovery(self):
        pool = NodePool(["http://a:8188", "http://b:8188"])
        pool.mark_unhealthy("http://a:8188")

        # Simulate cooldown elapsed
        for node in pool._nodes:
            if node.url == "http://a:8188":
                node.last_failure = time.monotonic() - _COOLDOWN_SEC - 1

        urls = set()
        for _ in range(4):
            url = pool.acquire()
            urls.add(url)
            pool.release(url)
        assert "http://a:8188" in urls

    def test_all_unhealthy_force_recovery(self):
        pool = NodePool(["http://a:8188", "http://b:8188"])
        pool.mark_unhealthy("http://a:8188")
        pool.mark_unhealthy("http://b:8188")
        # Should still return a node
        url = pool.acquire()
        assert url in ("http://a:8188", "http://b:8188")
        pool.release(url)

    def test_status(self):
        pool = NodePool(["http://a:8188", "http://b:8188"])
        pool.mark_unhealthy("http://b:8188")
        st = pool.status()
        assert len(st) == 2
        assert st[0]["healthy"] is True
        assert st[1]["healthy"] is False
        assert st[1]["cooldown_remaining"] > 0

    def test_in_flight_tracking(self):
        pool = NodePool(["http://a:8188"])
        url = pool.acquire()
        st = pool.status()
        assert st[0]["in_flight"] == 1
        pool.release(url)
        st = pool.status()
        assert st[0]["in_flight"] == 0
