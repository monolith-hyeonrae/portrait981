"""Tests for momentscan pipeline integration."""

from unittest.mock import Mock, patch

import pytest

from vpx.sdk import Observation


class TestHighlevelAPIBackend:
    """Tests for high-level API backend parameter."""

    def test_run_accepts_backend_parameter(self):
        """Test that ms.run() accepts backend parameter."""
        from momentscan.main import run

        # Just verify the function signature accepts backend
        import inspect
        sig = inspect.signature(run)
        assert "backend" in sig.parameters

    def test_main_run_uses_flowgraph_and_backend(self):
        """Test ms.run delegates to MomentscanApp.run()."""
        from visualpath.backends.base import PipelineResult

        mock_engine = Mock()
        mock_engine.execute.return_value = PipelineResult(triggers=[], frame_count=2)
        mock_engine.name = "SimpleBackend"

        with patch("visualpath.runner.get_backend", return_value=mock_engine), \
             patch("visualpath.runner._open_video_source", return_value=(iter([]), None)):
            from momentscan.main import run
            result = run("fake.mp4", analyzers=["mock.dummy"], fps=5)

            assert result.frame_count == 2
            assert result.actual_backend == "SimpleBackend"


class TestResourceConflictIsolation:
    """Tests for resource conflict detection and auto subprocess isolation."""

    @staticmethod
    def _make_module(name, resource_groups=frozenset(), gpu=False):
        """Create a mock Module with capabilities."""
        from visualpath.core.capabilities import Capability, ModuleCapabilities

        flags = frozenset({Capability.GPU}) if gpu else frozenset()
        caps = ModuleCapabilities(
            resource_groups=resource_groups,
            flags=flags,
        )
        mod = Mock()
        mod.name = name
        mod.capabilities = caps
        return mod

    def test_no_conflict_single_group(self):
        """No conflict when all modules are in the same resource group."""
        from visualpath.core.compat import build_conflict_isolation

        modules = [
            self._make_module("face.detect", frozenset({"onnxruntime"})),
            self._make_module("face.expression", frozenset({"onnxruntime"})),
        ]
        config = build_conflict_isolation(modules)
        assert config is None

    def test_no_conflict_no_resource_groups(self):
        """No conflict when modules don't declare resource groups."""
        from visualpath.core.compat import build_conflict_isolation

        modules = [
            self._make_module("frame.quality"),
            self._make_module("mock.dummy"),
        ]
        config = build_conflict_isolation(modules)
        assert config is None

    def test_conflict_onnxruntime_vs_torch(self):
        """Conflict detected when onnxruntime + torch groups are both active."""
        from visualpath.core.compat import build_conflict_isolation
        from visualpath.core.isolation import IsolationLevel

        modules = [
            self._make_module("face.detect", frozenset({"onnxruntime"})),
            self._make_module("body.pose", frozenset({"torch"})),
        ]
        config = build_conflict_isolation(modules)
        assert config is not None
        # Both groups have 1 module — minority is whichever min() picks
        isolated_names = {n for n, lv in config.overrides.items() if lv == IsolationLevel.PROCESS}
        assert len(isolated_names) == 1

    def test_conflict_multiple_onnxruntime_vs_single_torch(self):
        """Multiple onnxruntime modules vs single torch -> torch isolated."""
        from visualpath.core.compat import build_conflict_isolation
        from visualpath.core.isolation import IsolationLevel

        modules = [
            self._make_module("face.detect", frozenset({"onnxruntime"})),
            self._make_module("face.expression", frozenset({"onnxruntime"})),
            self._make_module("body.pose", frozenset({"torch"})),
        ]
        config = build_conflict_isolation(modules)
        assert config is not None
        # torch has 1 module (minority) -> body.pose isolated
        assert config.get_level("body.pose") == IsolationLevel.PROCESS

    def test_no_conflict_without_zmq(self):
        """Returns None when pyzmq is unavailable."""
        from visualpath.core.compat import build_conflict_isolation
        import importlib

        modules = [
            self._make_module("face.detect", frozenset({"onnxruntime"})),
            self._make_module("body.pose", frozenset({"torch"})),
        ]
        with patch("builtins.__import__", side_effect=lambda name, *a, **kw: (_ for _ in ()).throw(ImportError("no zmq")) if name == "zmq" else importlib.__import__(name, *a, **kw)):
            config = build_conflict_isolation(modules)
            assert config is None

    def test_distributed_config_gpu_modules(self):
        """GPU modules get PROCESS isolation in distributed config."""
        from visualpath.core.compat import build_distributed_config
        from visualpath.core.isolation import IsolationLevel

        modules = [
            self._make_module("face.detect", gpu=True),
            self._make_module("face.classify", gpu=False),
        ]
        config = build_distributed_config(modules)
        assert config.get_level("face.detect") == IsolationLevel.PROCESS
        assert config.get_level("face.classify") == IsolationLevel.INLINE

    def test_distributed_config_with_venv(self):
        """GPU modules with venv_paths get VENV isolation."""
        from visualpath.core.compat import build_distributed_config
        from visualpath.core.isolation import IsolationLevel

        modules = [
            self._make_module("face.detect", gpu=True),
            self._make_module("body.pose", gpu=True),
        ]
        config = build_distributed_config(
            modules, venv_paths={"face.detect": "/opt/venv-face"}
        )
        assert config.get_level("face.detect") == IsolationLevel.VENV
        assert config.get_level("body.pose") == IsolationLevel.PROCESS
        assert config.venv_paths["face.detect"] == "/opt/venv-face"
