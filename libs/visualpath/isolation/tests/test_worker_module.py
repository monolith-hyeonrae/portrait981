"""Tests for WorkerModule attribute forwarding.

Verifies that WorkerModule correctly proxies capabilities, optional_depends,
and stateful from the original module.
"""

import pytest
from unittest.mock import MagicMock, patch

from visualpath.core.capabilities import Capability, ModuleCapabilities
from visualpath.process.worker_module import WorkerModule


def _make_mock_worker():
    """Create a mock BaseWorker."""
    worker = MagicMock()
    worker.is_running = False
    worker.worker_info = MagicMock(
        isolation_level="PROCESS", pid=12345, venv_path=None
    )
    return worker


class TestWorkerModuleAttributeForwarding:
    """Tests for proxied attributes on WorkerModule."""

    def test_default_capabilities(self):
        """WorkerModule without explicit capabilities returns default."""
        wm = WorkerModule(name="a", worker=_make_mock_worker())
        caps = wm.capabilities
        assert caps.flags == Capability.NONE
        assert caps.max_batch_size == 1

    def test_capabilities_forwarded(self):
        """WorkerModule forwards capabilities from original module."""
        caps = ModuleCapabilities(
            flags=Capability.BATCHING | Capability.GPU,
            max_batch_size=8,
            gpu_memory_mb=512,
        )
        wm = WorkerModule(
            name="a",
            worker=_make_mock_worker(),
            capabilities=caps,
        )
        assert Capability.BATCHING in wm.capabilities.flags
        assert Capability.GPU in wm.capabilities.flags
        assert wm.capabilities.max_batch_size == 8
        assert wm.capabilities.gpu_memory_mb == 512

    def test_optional_depends_default(self):
        """WorkerModule defaults optional_depends to empty list."""
        wm = WorkerModule(name="a", worker=_make_mock_worker())
        assert wm.optional_depends == []

    def test_optional_depends_forwarded(self):
        """WorkerModule forwards optional_depends from original module."""
        wm = WorkerModule(
            name="a",
            worker=_make_mock_worker(),
            optional_depends=["b", "c"],
        )
        assert wm.optional_depends == ["b", "c"]

    def test_stateful_default(self):
        """WorkerModule defaults stateful to False."""
        wm = WorkerModule(name="a", worker=_make_mock_worker())
        assert wm.stateful is False

    def test_stateful_forwarded(self):
        """WorkerModule forwards stateful from original module."""
        wm = WorkerModule(
            name="a",
            worker=_make_mock_worker(),
            stateful=True,
        )
        assert wm.stateful is True

    def test_depends_forwarded(self):
        """WorkerModule forwards depends as before."""
        wm = WorkerModule(
            name="a",
            worker=_make_mock_worker(),
            depends=["x", "y"],
        )
        assert wm.depends == ["x", "y"]

    def test_all_attributes_together(self):
        """WorkerModule correctly forwards all attributes simultaneously."""
        caps = ModuleCapabilities(
            flags=Capability.BATCHING | Capability.STATEFUL,
            max_batch_size=4,
        )
        wm = WorkerModule(
            name="fusion",
            worker=_make_mock_worker(),
            depends=["face.detect"],
            optional_depends=["face.expression"],
            stateful=True,
            capabilities=caps,
        )
        assert wm.name == "fusion"
        assert wm.depends == ["face.detect"]
        assert wm.optional_depends == ["face.expression"]
        assert wm.stateful is True
        assert Capability.BATCHING in wm.capabilities.flags
        assert wm.capabilities.max_batch_size == 4


class TestWorkerBackendWrapModules:
    """Tests that WorkerBackend._wrap_modules copies attributes from originals."""

    def test_wrap_copies_capabilities(self):
        """_wrap_modules should copy capabilities to WorkerModule."""
        from visualpath.core.module import Module
        from visualpath.core.observation import Observation
        from visualpath.core.isolation import IsolationConfig, IsolationLevel

        class BatchModule(Module):
            depends = ["upstream"]
            optional_depends = ["extra"]
            stateful = True

            @property
            def name(self):
                return "batch_mod"

            @property
            def capabilities(self):
                return ModuleCapabilities(
                    flags=Capability.BATCHING | Capability.GPU,
                    max_batch_size=16,
                )

            def process(self, frame, deps=None):
                return None

        module = BatchModule()
        isolation = IsolationConfig(default_level=IsolationLevel.PROCESS)

        from visualpath.backends.worker.backend import WorkerBackend

        backend = WorkerBackend()

        # Mock WorkerLauncher.create to avoid actual subprocess
        mock_worker = _make_mock_worker()
        with patch(
            "visualpath.process.launcher.WorkerLauncher.create",
            return_value=mock_worker,
        ):
            wrapped = backend._wrap_modules([module], isolation)

        assert len(wrapped) == 1
        wm = wrapped[0]
        assert isinstance(wm, WorkerModule)
        assert wm.depends == ["upstream"]
        assert wm.optional_depends == ["extra"]
        assert wm.stateful is True
        assert Capability.BATCHING in wm.capabilities.flags
        assert Capability.GPU in wm.capabilities.flags
        assert wm.capabilities.max_batch_size == 16
