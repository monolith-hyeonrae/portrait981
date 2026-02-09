"""Tests for module capability declarations."""

import pytest

from visualpath.core.capabilities import Capability, ModuleCapabilities
from visualpath.core.module import Module


class TestCapabilityFlag:
    """Test Capability enum."""

    def test_none_is_zero(self):
        assert Capability.NONE.value == 0

    def test_bitwise_or(self):
        flags = Capability.GPU | Capability.STATEFUL
        assert Capability.GPU in flags
        assert Capability.STATEFUL in flags
        assert Capability.BATCHING not in flags

    def test_all_flags_distinct(self):
        all_flags = [f for f in Capability if f != Capability.NONE]
        values = [f.value for f in all_flags]
        assert len(values) == len(set(values))

    def test_flag_membership(self):
        flags = Capability.GPU | Capability.THREAD_SAFE | Capability.DETERMINISTIC
        assert Capability.GPU in flags
        assert Capability.THREAD_SAFE in flags
        assert Capability.DETERMINISTIC in flags
        assert Capability.STATEFUL not in flags


class TestModuleCapabilities:
    """Test ModuleCapabilities dataclass."""

    def test_defaults(self):
        caps = ModuleCapabilities()
        assert caps.flags == Capability.NONE
        assert caps.gpu_memory_mb == 0
        assert caps.init_time_sec == 0.0
        assert caps.max_batch_size == 1
        assert caps.resource_groups == frozenset()
        assert caps.required_extras == frozenset()

    def test_frozen(self):
        caps = ModuleCapabilities(flags=Capability.GPU)
        with pytest.raises(AttributeError):
            caps.flags = Capability.NONE  # type: ignore[misc]

    def test_gpu_module(self):
        caps = ModuleCapabilities(
            flags=Capability.GPU | Capability.STATEFUL,
            gpu_memory_mb=512,
            resource_groups=frozenset({"onnxruntime"}),
        )
        assert Capability.GPU in caps.flags
        assert Capability.STATEFUL in caps.flags
        assert caps.gpu_memory_mb == 512
        assert "onnxruntime" in caps.resource_groups

    def test_multiple_resource_groups(self):
        caps = ModuleCapabilities(
            resource_groups=frozenset({"onnxruntime", "tensorrt"}),
        )
        assert "onnxruntime" in caps.resource_groups
        assert "tensorrt" in caps.resource_groups


class TestModuleCapabilitiesProperty:
    """Test Module.capabilities default."""

    def test_default_capabilities(self):
        """Module.capabilities returns empty ModuleCapabilities by default."""

        class SimpleModule(Module):
            @property
            def name(self) -> str:
                return "test.simple"

            def process(self, frame, deps=None):
                return None

        mod = SimpleModule()
        caps = mod.capabilities
        assert isinstance(caps, ModuleCapabilities)
        assert caps.flags == Capability.NONE
        assert caps.gpu_memory_mb == 0
        assert caps.resource_groups == frozenset()

    def test_override_capabilities(self):
        """Subclass can override capabilities."""

        class GpuModule(Module):
            @property
            def name(self) -> str:
                return "test.gpu"

            @property
            def capabilities(self) -> ModuleCapabilities:
                return ModuleCapabilities(
                    flags=Capability.GPU | Capability.STATEFUL,
                    gpu_memory_mb=1024,
                    resource_groups=frozenset({"torch"}),
                )

            def process(self, frame, deps=None):
                return None

        mod = GpuModule()
        assert Capability.GPU in mod.capabilities.flags
        assert mod.capabilities.gpu_memory_mb == 1024
        assert "torch" in mod.capabilities.resource_groups
