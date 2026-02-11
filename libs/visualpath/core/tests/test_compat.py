"""Tests for module compatibility checking."""

import pytest

from visualpath.core.capabilities import Capability, ModuleCapabilities
from unittest.mock import patch

from visualpath.core.compat import (
    check_compatibility,
    CompatibilityReport,
    build_conflict_isolation,
    build_distributed_config,
)
from visualpath.core.isolation import IsolationConfig, IsolationLevel
from visualpath.core.module import Module


def _make_module(name, flags=Capability.NONE, gpu_mb=0, groups=frozenset()):
    """Helper to create a test module with given capabilities."""

    class _Mod(Module):
        @property
        def name(self_):
            return name

        @property
        def capabilities(self_):
            return ModuleCapabilities(
                flags=flags,
                gpu_memory_mb=gpu_mb,
                resource_groups=groups,
            )

        def process(self_, frame, deps=None):
            return None

    return _Mod()


class TestCheckCompatibility:
    """Test check_compatibility()."""

    def test_empty_modules(self):
        report = check_compatibility([])
        assert report.valid is True
        assert report.warnings == []
        assert report.estimated_gpu_mb == 0

    def test_no_conflicts(self):
        """Modules in the same resource group -> no conflict."""
        mods = [
            _make_module("a", groups=frozenset({"onnxruntime"})),
            _make_module("b", groups=frozenset({"onnxruntime"})),
        ]
        report = check_compatibility(mods)
        assert report.valid is True
        assert report.resource_conflicts == {}

    def test_resource_group_conflict(self):
        """Modules in different resource groups -> warning."""
        mods = [
            _make_module("face", flags=Capability.GPU, gpu_mb=512, groups=frozenset({"onnxruntime"})),
            _make_module("pose", flags=Capability.GPU, gpu_mb=384, groups=frozenset({"torch"})),
        ]
        report = check_compatibility(mods)
        assert report.valid is True  # still valid (warnings only)
        assert len(report.resource_conflicts) == 2
        assert "onnxruntime" in report.resource_conflicts
        assert "torch" in report.resource_conflicts
        assert any("conflict" in w.lower() for w in report.warnings)

    def test_gpu_memory_estimation(self):
        mods = [
            _make_module("a", flags=Capability.GPU, gpu_mb=512),
            _make_module("b", flags=Capability.GPU, gpu_mb=256),
            _make_module("c", gpu_mb=0),
        ]
        report = check_compatibility(mods)
        assert report.estimated_gpu_mb == 768

    def test_zero_copy_warning(self):
        mods = [
            _make_module("a", flags=Capability.NEEDS_ZERO_COPY),
        ]
        report = check_compatibility(mods)
        assert any("zero-copy" in w.lower() for w in report.warnings)

    def test_no_capabilities_declared(self):
        """Modules with no capabilities are fine."""

        class Plain(Module):
            @property
            def name(self_):
                return "plain"

            def process(self_, frame, deps=None):
                return None

        report = check_compatibility([Plain()])
        assert report.valid is True
        assert report.estimated_gpu_mb == 0

    def test_three_groups_conflict(self):
        mods = [
            _make_module("a", groups=frozenset({"onnxruntime"})),
            _make_module("b", groups=frozenset({"torch"})),
            _make_module("c", groups=frozenset({"mediapipe"})),
        ]
        report = check_compatibility(mods)
        assert len(report.resource_conflicts) == 3

    def test_mixed_gpu_and_non_gpu(self):
        mods = [
            _make_module("gpu_mod", flags=Capability.GPU, gpu_mb=512, groups=frozenset({"torch"})),
            _make_module("cpu_mod", flags=Capability.DETERMINISTIC),
        ]
        report = check_compatibility(mods)
        assert report.estimated_gpu_mb == 512
        # No conflict (only 1 group active)
        assert report.resource_conflicts == {}


class TestBuildConflictIsolation:
    """Test build_conflict_isolation()."""

    def test_no_conflicts_returns_none(self):
        mods = [
            _make_module("a", groups=frozenset({"onnxruntime"})),
            _make_module("b", groups=frozenset({"onnxruntime"})),
        ]
        assert build_conflict_isolation(mods) is None

    def test_no_resource_groups_returns_none(self):
        mods = [_make_module("a"), _make_module("b")]
        assert build_conflict_isolation(mods) is None

    def test_conflict_isolates_minority(self):
        mods = [
            _make_module("face1", groups=frozenset({"onnxruntime"})),
            _make_module("face2", groups=frozenset({"onnxruntime"})),
            _make_module("pose", groups=frozenset({"torch"})),
        ]
        config = build_conflict_isolation(mods)
        assert config is not None
        # torch group (1 module) is minority
        assert config.get_level("pose") == IsolationLevel.PROCESS
        assert config.get_level("face1") == IsolationLevel.INLINE

    def test_returns_none_without_zmq(self):
        import importlib

        mods = [
            _make_module("a", groups=frozenset({"onnxruntime"})),
            _make_module("b", groups=frozenset({"torch"})),
        ]
        with patch(
            "builtins.__import__",
            side_effect=lambda name, *a, **kw: (
                (_ for _ in ()).throw(ImportError("no zmq"))
                if name == "zmq"
                else importlib.__import__(name, *a, **kw)
            ),
        ):
            assert build_conflict_isolation(mods) is None

    def test_empty_modules(self):
        assert build_conflict_isolation([]) is None


class TestBuildDistributedConfig:
    """Test build_distributed_config()."""

    def test_gpu_modules_get_process(self):
        mods = [
            _make_module("face", flags=Capability.GPU),
            _make_module("quality"),
        ]
        config = build_distributed_config(mods)
        assert config.get_level("face") == IsolationLevel.PROCESS
        assert config.get_level("quality") == IsolationLevel.INLINE

    def test_venv_paths_override(self):
        mods = [
            _make_module("face", flags=Capability.GPU),
            _make_module("pose", flags=Capability.GPU),
        ]
        config = build_distributed_config(
            mods, venv_paths={"face": "/opt/venv-face"}
        )
        assert config.get_level("face") == IsolationLevel.VENV
        assert config.get_level("pose") == IsolationLevel.PROCESS
        assert config.venv_paths["face"] == "/opt/venv-face"

    def test_no_gpu_modules(self):
        mods = [_make_module("a"), _make_module("b")]
        config = build_distributed_config(mods)
        assert config.default_level == IsolationLevel.INLINE
        assert config.overrides == {}

    def test_empty_modules(self):
        config = build_distributed_config([])
        assert config.default_level == IsolationLevel.INLINE
        assert config.overrides == {}


class TestCompatibilityReport:
    """Test CompatibilityReport dataclass."""

    def test_defaults(self):
        r = CompatibilityReport()
        assert r.valid is True
        assert r.warnings == []
        assert r.errors == []
        assert r.resource_conflicts == {}
        assert r.estimated_gpu_mb == 0
