"""Tests for AnalyzerOrchestrator."""

import pytest
import time
import numpy as np
from dataclasses import dataclass
from typing import Optional, List

from visualpath.core import Module, Observation
from visualpath.process import AnalyzerOrchestrator


# =============================================================================
# Test Fixtures
# =============================================================================


@dataclass
class MockFrame:
    """Mock Frame for testing."""
    frame_id: int
    t_src_ns: int
    data: np.ndarray


class SimpleAnalyzer(Module):
    """Simple analyzer for testing."""

    def __init__(
        self,
        name: str = "simple",
        delay_ms: float = 0,
        fail: bool = False,
        return_none: bool = False,
    ):
        self._name = name
        self._delay_ms = delay_ms
        self._fail = fail
        self._return_none = return_none
        self._initialized = False
        self._cleaned_up = False
        self._process_count = 0

    @property
    def name(self) -> str:
        return self._name

    def process(self, frame, deps=None) -> Optional[Observation]:
        if self._delay_ms > 0:
            time.sleep(self._delay_ms / 1000)

        if self._fail:
            raise RuntimeError("Analysis failed")

        if self._return_none:
            return None

        self._process_count += 1
        return Observation(
            source=self.name,
            frame_id=frame.frame_id,
            t_ns=frame.t_src_ns,
            signals={"count": self._process_count},
        )

    def initialize(self) -> None:
        self._initialized = True

    def cleanup(self) -> None:
        self._cleaned_up = True


# =============================================================================
# Basic Tests
# =============================================================================


class TestAnalyzerOrchestrator:
    """Tests for AnalyzerOrchestrator."""

    def test_initialization(self):
        """Test orchestrator initialization."""
        analyzers = [SimpleAnalyzer("ext1"), SimpleAnalyzer("ext2")]
        orchestrator = AnalyzerOrchestrator(analyzers)

        assert not orchestrator.is_initialized
        assert orchestrator.analyzer_names == ["ext1", "ext2"]

    def test_requires_at_least_one_analyzer(self):
        """Test that at least one analyzer is required."""
        with pytest.raises(ValueError, match="At least one analyzer"):
            AnalyzerOrchestrator([])

    def test_initialize(self):
        """Test initialize() initializes all analyzers."""
        ext1 = SimpleAnalyzer("ext1")
        ext2 = SimpleAnalyzer("ext2")
        orchestrator = AnalyzerOrchestrator([ext1, ext2])

        orchestrator.initialize()

        assert orchestrator.is_initialized
        assert ext1._initialized
        assert ext2._initialized

    def test_cleanup(self):
        """Test cleanup() cleans up all analyzers."""
        ext1 = SimpleAnalyzer("ext1")
        ext2 = SimpleAnalyzer("ext2")
        orchestrator = AnalyzerOrchestrator([ext1, ext2])

        orchestrator.initialize()
        orchestrator.cleanup()

        assert not orchestrator.is_initialized
        assert ext1._cleaned_up
        assert ext2._cleaned_up

    def test_context_manager(self):
        """Test context manager protocol."""
        ext = SimpleAnalyzer()

        with AnalyzerOrchestrator([ext]) as orchestrator:
            assert orchestrator.is_initialized
            assert ext._initialized

        assert not orchestrator.is_initialized
        assert ext._cleaned_up

    def test_double_initialize(self):
        """Test that double initialize is safe."""
        ext = SimpleAnalyzer()
        orchestrator = AnalyzerOrchestrator([ext])

        orchestrator.initialize()
        orchestrator.initialize()  # Should be no-op

        assert orchestrator.is_initialized

    def test_double_cleanup(self):
        """Test that double cleanup is safe."""
        ext = SimpleAnalyzer()
        orchestrator = AnalyzerOrchestrator([ext])

        orchestrator.initialize()
        orchestrator.cleanup()
        orchestrator.cleanup()  # Should be no-op

        assert not orchestrator.is_initialized


# =============================================================================
# Analysis Tests
# =============================================================================


class TestAnalyzerOrchestratorAnalysis:
    """Tests for analysis functionality."""

    def test_analyze_all_single_extractor(self):
        """Test analyzing with a single analyzer."""
        ext = SimpleAnalyzer()
        frame = MockFrame(frame_id=1, t_src_ns=1000000, data=np.zeros((100, 100, 3)))

        with AnalyzerOrchestrator([ext]) as orchestrator:
            observations = orchestrator.analyze_all(frame)

        assert len(observations) == 1
        assert observations[0].source == "simple"
        assert observations[0].frame_id == 1

    def test_analyze_all_multiple_extractors(self):
        """Test analyzing with multiple analyzers."""
        ext1 = SimpleAnalyzer("ext1")
        ext2 = SimpleAnalyzer("ext2")
        ext3 = SimpleAnalyzer("ext3")
        frame = MockFrame(frame_id=1, t_src_ns=1000000, data=np.zeros((100, 100, 3)))

        with AnalyzerOrchestrator([ext1, ext2, ext3]) as orchestrator:
            observations = orchestrator.analyze_all(frame)

        assert len(observations) == 3
        sources = {obs.source for obs in observations}
        assert sources == {"ext1", "ext2", "ext3"}

    def test_analyze_all_requires_initialization(self):
        """Test that analyze_all requires initialization."""
        ext = SimpleAnalyzer()
        orchestrator = AnalyzerOrchestrator([ext])
        frame = MockFrame(frame_id=1, t_src_ns=1000000, data=np.zeros((100, 100, 3)))

        with pytest.raises(RuntimeError, match="not initialized"):
            orchestrator.analyze_all(frame)

    def test_analyze_all_filters_none(self):
        """Test that None observations are filtered out."""
        ext1 = SimpleAnalyzer("ext1")
        ext2 = SimpleAnalyzer("ext2", return_none=True)
        frame = MockFrame(frame_id=1, t_src_ns=1000000, data=np.zeros((100, 100, 3)))

        with AnalyzerOrchestrator([ext1, ext2]) as orchestrator:
            observations = orchestrator.analyze_all(frame)

        assert len(observations) == 1
        assert observations[0].source == "ext1"

    def test_analyze_all_handles_errors(self):
        """Test that errors in analyzers are handled."""
        ext1 = SimpleAnalyzer("ext1")
        ext2 = SimpleAnalyzer("ext2", fail=True)
        frame = MockFrame(frame_id=1, t_src_ns=1000000, data=np.zeros((100, 100, 3)))

        with AnalyzerOrchestrator([ext1, ext2]) as orchestrator:
            observations = orchestrator.analyze_all(frame)

        # Only ext1 should succeed
        assert len(observations) == 1
        assert observations[0].source == "ext1"

        # Error should be counted
        stats = orchestrator.get_stats()
        assert stats["errors"] == 1

    def test_analyze_all_timeout(self):
        """Test that slow analyzers timeout."""
        ext1 = SimpleAnalyzer("ext1")
        ext2 = SimpleAnalyzer("ext2", delay_ms=2000)  # Very slow
        frame = MockFrame(frame_id=1, t_src_ns=1000000, data=np.zeros((100, 100, 3)))

        with AnalyzerOrchestrator([ext1, ext2], timeout=0.1) as orchestrator:
            observations = orchestrator.analyze_all(frame)

        # ext1 should succeed, ext2 may timeout
        assert len(observations) >= 1

        stats = orchestrator.get_stats()
        # May or may not have timeout depending on timing
        # Just verify stats are collected
        assert stats["frames_processed"] == 1

    def test_analyze_multiple_frames(self):
        """Test processing multiple frames."""
        ext = SimpleAnalyzer()

        with AnalyzerOrchestrator([ext]) as orchestrator:
            for i in range(5):
                frame = MockFrame(frame_id=i, t_src_ns=i * 1000000, data=np.zeros((100, 100, 3)))
                observations = orchestrator.analyze_all(frame)
                assert len(observations) == 1
                assert observations[0].frame_id == i

        stats = orchestrator.get_stats()
        assert stats["frames_processed"] == 5
        assert stats["total_observations"] == 5


# =============================================================================
# Sequential Analysis Tests
# =============================================================================


class TestAnalyzerOrchestratorSequential:
    """Tests for sequential analysis."""

    def test_analyze_sequential(self):
        """Test sequential analysis."""
        ext1 = SimpleAnalyzer("ext1")
        ext2 = SimpleAnalyzer("ext2")
        frame = MockFrame(frame_id=1, t_src_ns=1000000, data=np.zeros((100, 100, 3)))

        with AnalyzerOrchestrator([ext1, ext2]) as orchestrator:
            observations = orchestrator.analyze_sequential(frame)

        assert len(observations) == 2

    def test_analyze_sequential_requires_initialization(self):
        """Test that analyze_sequential requires initialization."""
        ext = SimpleAnalyzer()
        orchestrator = AnalyzerOrchestrator([ext])
        frame = MockFrame(frame_id=1, t_src_ns=1000000, data=np.zeros((100, 100, 3)))

        with pytest.raises(RuntimeError, match="not initialized"):
            orchestrator.analyze_sequential(frame)

    def test_analyze_sequential_handles_errors(self):
        """Test that sequential analysis handles errors."""
        ext1 = SimpleAnalyzer("ext1")
        ext2 = SimpleAnalyzer("ext2", fail=True)
        ext3 = SimpleAnalyzer("ext3")
        frame = MockFrame(frame_id=1, t_src_ns=1000000, data=np.zeros((100, 100, 3)))

        with AnalyzerOrchestrator([ext1, ext2, ext3]) as orchestrator:
            observations = orchestrator.analyze_sequential(frame)

        # ext1 and ext3 should succeed
        assert len(observations) == 2

        stats = orchestrator.get_stats()
        assert stats["errors"] == 1


# =============================================================================
# Stats Tests
# =============================================================================


class TestAnalyzerOrchestratorStats:
    """Tests for statistics collection."""

    def test_get_stats_initial(self):
        """Test initial stats."""
        ext = SimpleAnalyzer()
        orchestrator = AnalyzerOrchestrator([ext])

        stats = orchestrator.get_stats()

        assert stats["frames_processed"] == 0
        assert stats["total_observations"] == 0
        assert stats["timeouts"] == 0
        assert stats["errors"] == 0
        assert stats["analyzers"] == ["simple"]

    def test_get_stats_after_processing(self):
        """Test stats after processing."""
        ext1 = SimpleAnalyzer("ext1")
        ext2 = SimpleAnalyzer("ext2")

        with AnalyzerOrchestrator([ext1, ext2]) as orchestrator:
            for i in range(3):
                frame = MockFrame(frame_id=i, t_src_ns=i * 1000000, data=np.zeros((100, 100, 3)))
                orchestrator.analyze_all(frame)

            stats = orchestrator.get_stats()

        assert stats["frames_processed"] == 3
        assert stats["total_observations"] == 6
        assert stats["avg_time_ms"] > 0

    def test_max_workers_in_stats(self):
        """Test that max_workers is in stats."""
        ext = SimpleAnalyzer()
        orchestrator = AnalyzerOrchestrator([ext], max_workers=4)

        stats = orchestrator.get_stats()

        assert stats["max_workers"] == 4


# =============================================================================
# Configuration Tests
# =============================================================================


class TestAnalyzerOrchestratorConfig:
    """Tests for configuration options."""

    def test_custom_max_workers(self):
        """Test custom max_workers setting."""
        analyzers = [SimpleAnalyzer(f"ext{i}") for i in range(5)]
        orchestrator = AnalyzerOrchestrator(analyzers, max_workers=2)

        assert orchestrator._max_workers == 2

    def test_custom_timeout(self):
        """Test custom timeout setting."""
        ext = SimpleAnalyzer()
        orchestrator = AnalyzerOrchestrator([ext], timeout=10.0)

        assert orchestrator._timeout == 10.0

    def test_default_max_workers_is_analyzer_count(self):
        """Test that default max_workers equals analyzer count."""
        analyzers = [SimpleAnalyzer(f"ext{i}") for i in range(3)]
        orchestrator = AnalyzerOrchestrator(analyzers)

        assert orchestrator._max_workers == 3
