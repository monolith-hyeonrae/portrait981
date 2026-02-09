"""Tests for Phase 3: Execution profiles (lite/platform)."""

import pytest

from visualpath.core.capabilities import Capability, ModuleCapabilities
from visualpath.core.isolation import IsolationLevel, IsolationConfig
from visualpath.core.module import Module
from visualpath.core.profile import (
    ExecutionProfile,
    ProfileName,
    resolve_profile,
)


def _make_module(name, flags=Capability.NONE, groups=frozenset()):
    class _Mod(Module):
        @property
        def name(self_):
            return name

        @property
        def capabilities(self_):
            return ModuleCapabilities(flags=flags, resource_groups=groups)

        def process(self_, frame, deps=None):
            return None

    return _Mod()


class TestProfileName:
    def test_values(self):
        assert ProfileName.LITE == "lite"
        assert ProfileName.PLATFORM == "platform"


class TestExecutionProfile:
    def test_lite_preset(self):
        p = ExecutionProfile.lite()
        assert p.name == ProfileName.LITE
        assert p.isolation_default == IsolationLevel.INLINE
        assert p.enable_observability is False
        assert p.backend == "simple"

    def test_platform_preset(self):
        p = ExecutionProfile.platform()
        assert p.name == ProfileName.PLATFORM
        assert p.isolation_default == IsolationLevel.PROCESS
        assert p.enable_observability is True
        assert p.backend == "pathway"

    def test_from_name_lite(self):
        p = ExecutionProfile.from_name("lite")
        assert p.name == ProfileName.LITE

    def test_from_name_platform(self):
        p = ExecutionProfile.from_name("platform")
        assert p.name == ProfileName.PLATFORM

    def test_from_name_unknown(self):
        with pytest.raises(ValueError, match="Unknown profile"):
            ExecutionProfile.from_name("unknown")

    def test_frozen(self):
        p = ExecutionProfile.lite()
        with pytest.raises(AttributeError):
            p.backend = "pathway"  # type: ignore[misc]


class TestResolveProfile:
    def test_lite_all_inline(self):
        """Lite profile keeps all modules INLINE."""
        mods = [
            _make_module("a", flags=Capability.GPU, groups=frozenset({"onnxruntime"})),
            _make_module("b", flags=Capability.GPU, groups=frozenset({"torch"})),
            _make_module("c"),
        ]
        config = resolve_profile(ExecutionProfile.lite(), mods)
        assert config.default_level == IsolationLevel.INLINE
        assert config.overrides == {}

    def test_platform_gpu_modules_isolated(self):
        """Platform profile isolates GPU modules to PROCESS."""
        mods = [
            _make_module("face", flags=Capability.GPU, groups=frozenset({"onnxruntime"})),
            _make_module("pose", flags=Capability.GPU, groups=frozenset({"torch"})),
            _make_module("quality", flags=Capability.DETERMINISTIC),
        ]
        config = resolve_profile(ExecutionProfile.platform(), mods)
        assert config.default_level == IsolationLevel.PROCESS
        assert "face" in config.overrides
        assert "pose" in config.overrides
        assert config.overrides["face"] == IsolationLevel.PROCESS
        assert "quality" not in config.overrides

    def test_platform_no_gpu_modules(self):
        """Platform profile with no GPU modules -> no overrides."""
        mods = [_make_module("cpu_only")]
        config = resolve_profile(ExecutionProfile.platform(), mods)
        assert config.overrides == {}

    def test_empty_modules(self):
        config = resolve_profile(ExecutionProfile.lite(), [])
        assert config.default_level == IsolationLevel.INLINE
        assert config.overrides == {}

    def test_returns_isolation_config(self):
        config = resolve_profile(ExecutionProfile.lite(), [])
        assert isinstance(config, IsolationConfig)
