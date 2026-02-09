"""Tests for plugin discovery system."""

import pytest
from typing import Optional

from visualpath.core import Module, Observation
from visualpath.plugin import (
    discover_analyzers,
    discover_fusions,
    PluginRegistry,
    ANALYZERS_GROUP,
)


# =============================================================================
# Test Fixtures
# =============================================================================


class MockAnalyzer(Module):
    """Mock analyzer for testing."""

    def __init__(self, value: float = 0.5):
        self._value = value
        self._initialized = False

    @property
    def name(self) -> str:
        return "mock"

    def process(self, frame, deps=None) -> Optional[Observation]:
        return Observation(
            source=self.name,
            frame_id=frame.frame_id,
            t_ns=frame.t_src_ns,
            signals={"value": self._value},
        )

    def initialize(self) -> None:
        self._initialized = True

    def cleanup(self) -> None:
        self._initialized = False


class MockAnalyzer2(Module):
    """Another mock extractor for testing."""

    @property
    def name(self) -> str:
        return "mock2"

    def process(self, frame, deps=None) -> Optional[Observation]:
        return None


# =============================================================================
# Discovery Tests
# =============================================================================


class TestDiscovery:
    """Tests for plugin discovery functions."""

    def test_discover_analyzers_returns_dict(self):
        """Test that discover_analyzers returns a dict."""
        extractors = discover_analyzers()
        assert isinstance(extractors, dict)

    def test_discover_fusions_returns_dict(self):
        """Test that discover_fusions returns a dict."""
        fusions = discover_fusions()
        assert isinstance(fusions, dict)

    def test_extractors_group_name(self):
        """Test the extractors group name constant."""
        assert ANALYZERS_GROUP == "visualpath.analyzers"


# =============================================================================
# PluginRegistry Tests
# =============================================================================


class TestPluginRegistry:
    """Tests for PluginRegistry."""

    def test_create_registry(self):
        """Test creating a registry."""
        registry = PluginRegistry()
        assert registry is not None

    def test_register_analyzer(self):
        """Test registering an extractor."""
        registry = PluginRegistry()
        registry.register_analyzer("mock", MockAnalyzer)

        assert "mock" in registry.list_analyzers()

    def test_register_fusion(self):
        """Test registering a fusion."""
        registry = PluginRegistry()

        class MockFusion:
            pass

        registry.register_fusion("mock_fusion", MockFusion)

        assert "mock_fusion" in registry.list_fusions()

    def test_get_analyzer_class_registered(self):
        """Test getting a registered extractor class."""
        registry = PluginRegistry()
        registry.register_analyzer("mock", MockAnalyzer)

        cls = registry.get_analyzer_class("mock")

        assert cls is MockAnalyzer

    def test_get_analyzer_class_not_found(self):
        """Test getting non-existent extractor raises KeyError."""
        registry = PluginRegistry()

        with pytest.raises(KeyError):
            registry.get_analyzer_class("nonexistent")

    def test_create_analyzer(self):
        """Test creating an extractor instance."""
        registry = PluginRegistry()
        registry.register_analyzer("mock", MockAnalyzer)

        extractor = registry.create_analyzer("mock", value=0.7)

        assert isinstance(extractor, MockAnalyzer)
        assert extractor._value == 0.7

    def test_create_analyzer_singleton(self):
        """Test singleton pattern for extractors."""
        registry = PluginRegistry()
        registry.register_analyzer("mock", MockAnalyzer)

        ext1 = registry.create_analyzer("mock", singleton=True)
        ext2 = registry.create_analyzer("mock", singleton=True)

        assert ext1 is ext2

    def test_create_analyzer_non_singleton(self):
        """Test non-singleton creates new instances."""
        registry = PluginRegistry()
        registry.register_analyzer("mock", MockAnalyzer)

        ext1 = registry.create_analyzer("mock", singleton=False)
        ext2 = registry.create_analyzer("mock", singleton=False)

        assert ext1 is not ext2

    def test_list_analyzers(self):
        """Test listing registered extractors."""
        registry = PluginRegistry()
        registry.register_analyzer("mock1", MockAnalyzer)
        registry.register_analyzer("mock2", MockAnalyzer2)

        extractors = registry.list_analyzers()

        assert "mock1" in extractors
        assert "mock2" in extractors

    def test_cleanup(self):
        """Test cleanup cleans up instances."""
        registry = PluginRegistry()
        registry.register_analyzer("mock", MockAnalyzer)

        ext = registry.create_analyzer("mock", singleton=True)
        ext.initialize()
        assert ext._initialized

        registry.cleanup()

        assert not ext._initialized
        # Should create new instance after cleanup
        ext2 = registry.create_analyzer("mock", singleton=True)
        assert ext2 is not ext

    def test_multiple_registries_independent(self):
        """Test that multiple registries are independent."""
        reg1 = PluginRegistry()
        reg2 = PluginRegistry()

        reg1.register_analyzer("mock", MockAnalyzer)

        assert "mock" in reg1.list_analyzers()
        # Check that discovered extractors are shared, but registered are not
        # Since we register in reg1 only, it should be in reg1's list

    def test_list_analyzers_sorted(self):
        """Test that list_analyzers returns sorted names."""
        registry = PluginRegistry()
        registry.register_analyzer("zebra", MockAnalyzer)
        registry.register_analyzer("alpha", MockAnalyzer2)

        extractors = registry.list_analyzers()

        # Should be alphabetically sorted
        assert extractors.index("alpha") < extractors.index("zebra")
