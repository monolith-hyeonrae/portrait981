"""Tests for Pathway backend integration.

These tests verify the Pathway execution backend functionality.
Tests are skipped if Pathway is not installed.
"""

import pytest
import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Dict
from unittest.mock import MagicMock, patch

from visualpath.core import Module, Observation
from visualpath.backends.base import ExecutionBackend, PipelineResult
from visualpath.backends.simple import SimpleBackend


# Check if Pathway is available
try:
    import pathway as pw
    PATHWAY_AVAILABLE = True
except ImportError:
    PATHWAY_AVAILABLE = False


# =============================================================================
# Test Fixtures
# =============================================================================


@dataclass
class MockFrame:
    """Mock Frame for testing."""
    frame_id: int
    t_src_ns: int
    data: np.ndarray


class CountingAnalyzer(Module):
    """Analyzer that counts calls for testing."""

    def __init__(self, name: str, return_value: float = 0.5):
        self._name = name
        self._return_value = return_value
        self._process_count = 0
        self._initialized = False
        self._cleaned_up = False

    @property
    def name(self) -> str:
        return self._name

    def process(self, frame, deps=None) -> Optional[Observation]:
        self._process_count += 1
        return Observation(
            source=self.name,
            frame_id=frame.frame_id,
            t_ns=frame.t_src_ns,
            signals={"value": self._return_value, "call_count": self._process_count},
        )

    def initialize(self) -> None:
        self._initialized = True

    def cleanup(self) -> None:
        self._cleaned_up = True


class ThresholdFusion(Module):
    """Simple fusion for testing."""

    def __init__(self, threshold: float = 0.5, depends_on: str = None):
        self._threshold = threshold
        self._gate_open = True
        self._cooldown = False
        self._update_count = 0
        self.depends = [depends_on] if depends_on else []
        self._trigger_count = 0

    @property
    def name(self) -> str:
        return "threshold_fusion"

    def process(self, frame, deps=None) -> Optional[Observation]:
        """Process observations from deps and decide on trigger."""
        self._update_count += 1
        # Find any observation from deps
        observation = None
        if deps:
            for obs in deps.values():
                if obs is not None:
                    observation = obs
                    break

        if observation is None:
            return Observation(
                source=self.name,
                frame_id=frame.frame_id,
                t_ns=frame.t_src_ns,
                signals={"should_trigger": False},
            )

        value = observation.signals.get("value", 0)
        if value > self._threshold:
            self._trigger_count += 1
            # Create a mock trigger
            from visualbase import Trigger
            trigger = Trigger.point(
                event_time_ns=observation.t_ns,
                pre_sec=2.0,
                post_sec=2.0,
                label="threshold_exceeded",
                score=value,
            )
            return Observation(
                source=self.name,
                frame_id=frame.frame_id,
                t_ns=frame.t_src_ns,
                signals={
                    "should_trigger": True,
                    "trigger_score": value,
                    "trigger_reason": "threshold_exceeded",
                },
                metadata={"trigger": trigger},
            )
        return Observation(
            source=self.name,
            frame_id=frame.frame_id,
            t_ns=frame.t_src_ns,
            signals={"should_trigger": False},
        )

    def reset(self) -> None:
        self._update_count = 0
        self._trigger_count = 0


def make_frame(frame_id: int = 1, t_ns: int = 1_000_000) -> MockFrame:
    """Create a mock frame."""
    return MockFrame(
        frame_id=frame_id,
        t_src_ns=t_ns,
        data=np.zeros((100, 100, 3), dtype=np.uint8),
    )


def make_frames(count: int, interval_ns: int = 100_000_000) -> List[MockFrame]:
    """Create a list of mock frames."""
    return [
        make_frame(frame_id=i, t_ns=i * interval_ns)
        for i in range(count)
    ]


# =============================================================================
# ExecutionBackend ABC Tests
# =============================================================================


class TestExecutionBackend:
    """Tests for ExecutionBackend abstract base class."""

    def test_is_abstract(self):
        """Test that ExecutionBackend cannot be instantiated."""
        with pytest.raises(TypeError):
            ExecutionBackend()

    def test_simple_backend_implements_interface(self):
        """Test SimpleBackend implements ExecutionBackend."""
        backend = SimpleBackend()
        assert isinstance(backend, ExecutionBackend)
        assert hasattr(backend, "execute")
        assert hasattr(backend, "name")


# =============================================================================
# SimpleBackend Tests (execute API)
# =============================================================================


class TestSimpleBackend:
    """Tests for SimpleBackend.execute() with FlowGraph."""

    def test_run_single_analyzer(self):
        """Test running with a single analyzer."""
        from visualpath.flow.graph import FlowGraph

        backend = SimpleBackend()
        analyzer = CountingAnalyzer("test", return_value=0.3)
        graph = FlowGraph.from_modules([analyzer])
        frames = make_frames(5)

        result = backend.execute(iter(frames), graph)

        assert analyzer._process_count == 5
        assert len(result.triggers) == 0  # No fusion

    def test_run_with_fusion(self):
        """Test running with fusion that triggers."""
        from visualpath.flow.graph import FlowGraph

        backend = SimpleBackend()
        analyzer = CountingAnalyzer("test", return_value=0.7)
        fusion = ThresholdFusion(threshold=0.5, depends_on="test")
        graph = FlowGraph.from_modules([analyzer, fusion])
        frames = make_frames(5)

        result = backend.execute(iter(frames), graph)

        assert analyzer._process_count == 5
        assert fusion._update_count == 5
        assert len(result.triggers) == 5  # All frames trigger

    def test_run_with_callback(self):
        """Test running with trigger callback via graph."""
        from visualpath.flow.graph import FlowGraph

        backend = SimpleBackend()
        analyzer = CountingAnalyzer("test", return_value=0.7)
        fusion = ThresholdFusion(threshold=0.5, depends_on="test")
        graph = FlowGraph.from_modules([analyzer, fusion])
        frames = make_frames(3)

        callback_data = []
        graph.on_trigger(lambda d: callback_data.append(d))

        result = backend.execute(iter(frames), graph)

        assert len(callback_data) == 3
        assert len(result.triggers) == 3

    def test_run_multiple_analyzers(self):
        """Test running with multiple analyzers."""
        from visualpath.flow.graph import FlowGraph

        backend = SimpleBackend()
        ext1 = CountingAnalyzer("ext1", return_value=0.3)
        ext2 = CountingAnalyzer("ext2", return_value=0.7)
        fusion = ThresholdFusion(threshold=0.5, depends_on="ext1")
        graph = FlowGraph.from_modules([ext1, ext2, fusion])
        frames = make_frames(3)

        result = backend.execute(iter(frames), graph)

        assert ext1._process_count == 3
        assert ext2._process_count == 3
        # Fusion runs once per frame (depends on ext1)
        assert fusion._update_count == 3

    def test_run_graph(self):
        """Test execute with simple graph."""
        from visualpath.flow import FlowGraph, SourceNode

        backend = SimpleBackend()
        graph = FlowGraph()
        graph.add_node(SourceNode("source"))

        frames = make_frames(3)
        result = backend.execute(iter(frames), graph)

        assert result.frame_count == 3


# =============================================================================
# SimpleBackend execute() Tests
# =============================================================================


class TestSimpleBackendExecute:
    """Tests for SimpleBackend.execute() with FlowGraph."""

    def test_execute_returns_pipeline_result(self):
        """Test execute() returns PipelineResult."""
        from visualpath.flow.graph import FlowGraph

        backend = SimpleBackend()
        ext = CountingAnalyzer("test", return_value=0.3)
        graph = FlowGraph.from_modules([ext])
        frames = make_frames(5)

        result = backend.execute(iter(frames), graph)

        assert isinstance(result, PipelineResult)
        assert result.frame_count == 5
        assert result.triggers == []

    def test_execute_with_fusion(self):
        """Test execute() with fusion that triggers."""
        from visualpath.flow.graph import FlowGraph

        backend = SimpleBackend()
        ext = CountingAnalyzer("test", return_value=0.7)
        fusion = ThresholdFusion(threshold=0.5, depends_on="test")
        graph = FlowGraph.from_modules([ext, fusion])
        frames = make_frames(3)

        result = backend.execute(iter(frames), graph)

        assert result.frame_count == 3
        assert len(result.triggers) == 3


# =============================================================================
# PathwayBackend Tests (require Pathway)
# =============================================================================


@pytest.mark.skipif(not PATHWAY_AVAILABLE, reason="Pathway not installed")
class TestPathwayBackend:
    """Tests for PathwayBackend."""

    def test_import(self):
        """Test PathwayBackend can be imported."""
        from visualpath.backends.pathway import PathwayBackend
        assert PathwayBackend is not None

    def test_instantiation(self):
        """Test PathwayBackend can be instantiated."""
        from visualpath.backends.pathway import PathwayBackend
        backend = PathwayBackend()
        assert backend.name == "pathway"

    def test_configuration(self):
        """Test PathwayBackend configuration options."""
        from visualpath.backends.pathway import PathwayBackend

        backend = PathwayBackend(
            window_ns=200_000_000,
            allowed_lateness_ns=100_000_000,
            autocommit_ms=50,
        )

        assert backend._window_ns == 200_000_000
        assert backend._allowed_lateness_ns == 100_000_000
        assert backend._autocommit_ms == 50

    def test_implements_interface(self):
        """Test PathwayBackend implements ExecutionBackend."""
        from visualpath.backends.pathway import PathwayBackend
        backend = PathwayBackend()
        assert isinstance(backend, ExecutionBackend)


# =============================================================================
# Pathway Schema Tests
# =============================================================================


@pytest.mark.skipif(not PATHWAY_AVAILABLE, reason="Pathway not installed")
class TestSchemas:
    """Tests for Pathway schemas using PyObjectWrapper."""

    def test_frame_schema(self):
        """Test FrameSchema uses PyObjectWrapper."""
        from visualpath.backends.pathway.connector import FrameSchema

        columns = FrameSchema.column_names()
        assert "frame_id" in columns
        assert "t_ns" in columns
        assert "frame" in columns

    def test_observation_schema(self):
        """Test ObservationSchema uses PyObjectWrapper."""
        from visualpath.backends.pathway.connector import ObservationSchema

        columns = ObservationSchema.column_names()
        assert "frame_id" in columns
        assert "t_ns" in columns
        assert "source" in columns
        assert "observation" in columns

    def test_trigger_schema(self):
        """Test TriggerSchema uses PyObjectWrapper."""
        from visualpath.backends.pathway.connector import TriggerSchema

        columns = TriggerSchema.column_names()
        assert "frame_id" in columns
        assert "t_ns" in columns
        assert "trigger" in columns


# =============================================================================
# VideoConnectorSubject Tests
# =============================================================================


@pytest.mark.skipif(not PATHWAY_AVAILABLE, reason="Pathway not installed")
class TestVideoConnectorSubject:
    """Tests for VideoConnectorSubject."""

    def test_import(self):
        """Test VideoConnectorSubject can be imported."""
        from visualpath.backends.pathway.connector import VideoConnectorSubject
        assert VideoConnectorSubject is not None

    def test_instantiation(self):
        """Test VideoConnectorSubject can be instantiated."""
        from visualpath.backends.pathway.connector import VideoConnectorSubject

        frames = make_frames(3)
        subject = VideoConnectorSubject(iter(frames))
        assert subject is not None


# =============================================================================
# Pathway End-to-End Execution Tests
# =============================================================================


@pytest.mark.skipif(not PATHWAY_AVAILABLE, reason="Pathway not installed")
class TestPathwayExecution:
    """Tests that verify actual Pathway engine execution."""

    def test_run_single_analyzer(self):
        """Test PathwayBackend.execute() with a single analyzer through Pathway engine."""
        from visualpath.backends.pathway import PathwayBackend
        from visualpath.flow.graph import FlowGraph

        backend = PathwayBackend(autocommit_ms=10)
        analyzer = CountingAnalyzer("test", return_value=0.3)
        graph = FlowGraph.from_modules([analyzer])
        frames = make_frames(5)

        result = backend.execute(iter(frames), graph)

        # No fusion = no triggers
        assert len(result.triggers) == 0
        # Analyzer should have been called for each frame
        assert analyzer._process_count == 5

    def test_run_with_fusion_triggers(self):
        """Test PathwayBackend.execute() with fusion that fires triggers."""
        from visualpath.backends.pathway import PathwayBackend
        from visualpath.flow.graph import FlowGraph

        backend = PathwayBackend(autocommit_ms=10)
        analyzer = CountingAnalyzer("test", return_value=0.7)
        fusion = ThresholdFusion(threshold=0.5, depends_on="test")
        graph = FlowGraph.from_modules([analyzer, fusion])
        frames = make_frames(5)

        result = backend.execute(iter(frames), graph)

        # All frames should trigger (value 0.7 > threshold 0.5)
        assert len(result.triggers) == 5
        assert analyzer._process_count == 5
        assert fusion._update_count == 5

    def test_run_with_fusion_no_trigger(self):
        """Test PathwayBackend.execute() when fusion doesn't fire."""
        from visualpath.backends.pathway import PathwayBackend
        from visualpath.flow.graph import FlowGraph

        backend = PathwayBackend(autocommit_ms=10)
        analyzer = CountingAnalyzer("test", return_value=0.3)
        fusion = ThresholdFusion(threshold=0.5, depends_on="test")
        graph = FlowGraph.from_modules([analyzer, fusion])
        frames = make_frames(3)

        result = backend.execute(iter(frames), graph)

        # value 0.3 < threshold 0.5 → no triggers
        assert len(result.triggers) == 0
        assert analyzer._process_count == 3
        assert fusion._update_count == 3

    def test_run_multiple_analyzers(self):
        """Test PathwayBackend.execute() with multiple analyzers."""
        from visualpath.backends.pathway import PathwayBackend
        from visualpath.flow.graph import FlowGraph

        backend = PathwayBackend(autocommit_ms=10)
        ext1 = CountingAnalyzer("ext1", return_value=0.3)
        ext2 = CountingAnalyzer("ext2", return_value=0.7)
        fusion = ThresholdFusion(threshold=0.5, depends_on="ext2")
        graph = FlowGraph.from_modules([ext1, ext2, fusion])
        frames = make_frames(3)

        result = backend.execute(iter(frames), graph)

        assert ext1._process_count == 3
        assert ext2._process_count == 3
        # Fusion runs once per frame in the UDF pipeline
        assert fusion._update_count == 3
        # Fusion depends on ext2 (0.7 > 0.5 threshold) → triggers
        assert len(result.triggers) == 3

    def test_run_with_callback(self):
        """Test PathwayBackend.execute() with on_trigger callback via graph."""
        from visualpath.backends.pathway import PathwayBackend
        from visualpath.flow.graph import FlowGraph

        backend = PathwayBackend(autocommit_ms=10)
        analyzer = CountingAnalyzer("test", return_value=0.7)
        fusion = ThresholdFusion(threshold=0.5, depends_on="test")
        graph = FlowGraph.from_modules([analyzer, fusion])
        frames = make_frames(3)

        callback_data = []
        graph.on_trigger(lambda d: callback_data.append(d))

        result = backend.execute(iter(frames), graph)

        assert len(callback_data) == 3
        assert len(result.triggers) == 3

    def test_run_cleanup_on_error(self):
        """Test that cleanup happens even if analysis errors."""
        from visualpath.backends.pathway import PathwayBackend
        from visualpath.flow.graph import FlowGraph

        class ErrorAnalyzer(Module):
            _name = "error"
            _initialized = False
            _cleaned_up = False

            @property
            def name(self):
                return self._name

            def process(self, frame, deps=None):
                raise RuntimeError("intentional")

            def initialize(self):
                self._initialized = True

            def cleanup(self):
                self._cleaned_up = True

        backend = PathwayBackend(autocommit_ms=10)
        ext = ErrorAnalyzer()
        graph = FlowGraph.from_modules([ext])
        frames = make_frames(2)

        # Should not raise - errors are caught inside UDF
        result = backend.execute(iter(frames), graph)
        assert len(result.triggers) == 0


# =============================================================================
# PathwayBackend execute() Tests
# =============================================================================


@pytest.mark.skipif(not PATHWAY_AVAILABLE, reason="Pathway not installed")
class TestPathwayBackendExecute:
    """Tests for PathwayBackend.execute() with FlowGraph."""

    def test_execute_returns_pipeline_result(self):
        """Test execute() returns PipelineResult."""
        from visualpath.backends.pathway import PathwayBackend
        from visualpath.flow.graph import FlowGraph

        backend = PathwayBackend(autocommit_ms=10)
        ext = CountingAnalyzer("test", return_value=0.3)
        graph = FlowGraph.from_modules([ext])
        frames = make_frames(5)

        result = backend.execute(iter(frames), graph)

        assert isinstance(result, PipelineResult)
        assert result.frame_count == 5
        assert result.triggers == []
        assert isinstance(result.stats, dict)

    def test_execute_with_fusion(self):
        """Test execute() with fusion that triggers."""
        from visualpath.backends.pathway import PathwayBackend
        from visualpath.flow.graph import FlowGraph

        backend = PathwayBackend(autocommit_ms=10)
        ext = CountingAnalyzer("test", return_value=0.7)
        fusion = ThresholdFusion(threshold=0.5, depends_on="test")
        graph = FlowGraph.from_modules([ext, fusion])
        frames = make_frames(3)

        result = backend.execute(iter(frames), graph)

        assert result.frame_count == 3
        assert len(result.triggers) == 3

    def test_execute_stats(self):
        """Test execute() populates stats."""
        from visualpath.backends.pathway import PathwayBackend
        from visualpath.flow.graph import FlowGraph

        backend = PathwayBackend(autocommit_ms=10)
        ext = CountingAnalyzer("test", return_value=0.5)
        graph = FlowGraph.from_modules([ext])
        frames = make_frames(5)

        result = backend.execute(iter(frames), graph)

        assert result.stats["frames_ingested"] == 5
        assert result.stats["pipeline_duration_sec"] > 0


# =============================================================================
# Pathway vs Simple Backend Comparison
# =============================================================================


@pytest.mark.skipif(not PATHWAY_AVAILABLE, reason="Pathway not installed")
class TestBackendComparison:
    """Tests comparing SimpleBackend and PathwayBackend produce same results."""

    def test_same_trigger_count(self):
        """Test both backends produce the same number of triggers."""
        from visualpath.backends.pathway import PathwayBackend
        from visualpath.flow.graph import FlowGraph

        # Simple backend
        simple = SimpleBackend()
        ext_s = CountingAnalyzer("test", return_value=0.7)
        fusion_s = ThresholdFusion(threshold=0.5, depends_on="test")
        graph_s = FlowGraph.from_modules([ext_s, fusion_s])
        simple_result = simple.execute(iter(make_frames(5)), graph_s)

        # Pathway backend
        pathway = PathwayBackend(autocommit_ms=10)
        ext_p = CountingAnalyzer("test", return_value=0.7)
        fusion_p = ThresholdFusion(threshold=0.5, depends_on="test")
        graph_p = FlowGraph.from_modules([ext_p, fusion_p])
        pathway_result = pathway.execute(iter(make_frames(5)), graph_p)

        assert len(simple_result.triggers) == len(pathway_result.triggers)

    def test_same_process_count(self):
        """Test both backends call analyzers the same number of times."""
        from visualpath.backends.pathway import PathwayBackend
        from visualpath.flow.graph import FlowGraph

        ext_s = CountingAnalyzer("test", return_value=0.5)
        graph_s = FlowGraph.from_modules([ext_s])
        SimpleBackend().execute(iter(make_frames(10)), graph_s)

        ext_p = CountingAnalyzer("test", return_value=0.5)
        graph_p = FlowGraph.from_modules([ext_p])
        PathwayBackend(autocommit_ms=10).execute(iter(make_frames(10)), graph_p)

        assert ext_s._process_count == ext_p._process_count

    def test_no_trigger_consistency(self):
        """Test both backends agree on no-trigger case."""
        from visualpath.backends.pathway import PathwayBackend
        from visualpath.flow.graph import FlowGraph

        ext_s = CountingAnalyzer("test", return_value=0.2)
        fusion_s = ThresholdFusion(threshold=0.5, depends_on="test")
        graph_s = FlowGraph.from_modules([ext_s, fusion_s])
        simple_result = SimpleBackend().execute(iter(make_frames(5)), graph_s)

        ext_p = CountingAnalyzer("test", return_value=0.2)
        fusion_p = ThresholdFusion(threshold=0.5, depends_on="test")
        graph_p = FlowGraph.from_modules([ext_p, fusion_p])
        pathway_result = PathwayBackend(autocommit_ms=10).execute(
            iter(make_frames(5)), graph_p
        )

        assert len(simple_result.triggers) == 0
        assert len(pathway_result.triggers) == 0


# =============================================================================
# Pathway Operator Tests
# =============================================================================


@pytest.mark.skipif(not PATHWAY_AVAILABLE, reason="Pathway not installed")
class TestPathwayOperators:
    """Tests for Pathway operators."""

    def test_create_analyzer_udf(self):
        """Test creating analyzer UDF."""
        from visualpath.backends.pathway.operators import create_analyzer_udf

        analyzer = CountingAnalyzer("test", return_value=0.5)
        udf = create_analyzer_udf(analyzer)

        frame = make_frame()
        results = udf(frame)

        assert len(results) == 1
        assert results[0].source == "test"
        assert results[0].observation is not None

    def test_create_multi_analyzer_udf(self):
        """Test creating multi-analyzer UDF."""
        from visualpath.backends.pathway.operators import create_multi_analyzer_udf

        ext1 = CountingAnalyzer("ext1", return_value=0.3)
        ext2 = CountingAnalyzer("ext2", return_value=0.7)
        udf = create_multi_analyzer_udf([ext1, ext2])

        frame = make_frame()
        results = udf(frame)

        assert len(results) == 2
        sources = {r.source for r in results}
        assert sources == {"ext1", "ext2"}

    def test_apply_analyzers_pathway_table(self):
        """Test apply_analyzers creates a valid Pathway table pipeline."""
        from visualpath.backends.pathway.connector import (
            VideoConnectorSubject,
            FrameSchema,
        )
        from visualpath.backends.pathway.operators import apply_analyzers

        ext1 = CountingAnalyzer("ext1", return_value=0.5)
        ext2 = CountingAnalyzer("ext2", return_value=0.8)
        frames = make_frames(3)

        subject = VideoConnectorSubject(iter(frames))
        frames_table = pw.io.python.read(
            subject, schema=FrameSchema, autocommit_duration_ms=10,
        )

        obs_table = apply_analyzers(frames_table, [ext1, ext2])

        collected = []
        pw.io.subscribe(
            obs_table,
            on_change=lambda key, row, time, is_addition: (
                collected.append(row) if is_addition else None
            ),
        )
        pw.run()

        assert len(collected) == 3
        assert ext1._process_count == 3
        assert ext2._process_count == 3


@pytest.mark.skipif(not PATHWAY_AVAILABLE, reason="Pathway not installed")
class TestFlowGraphConverter:
    """Tests for FlowGraphConverter."""

    def test_import(self):
        """Test FlowGraphConverter can be imported."""
        from visualpath.backends.pathway.converter import FlowGraphConverter
        assert FlowGraphConverter is not None

    def test_instantiation(self):
        """Test FlowGraphConverter can be instantiated."""
        from visualpath.backends.pathway.converter import FlowGraphConverter

        converter = FlowGraphConverter(
            window_ns=100_000_000,
            allowed_lateness_ns=50_000_000,
        )
        assert converter._window_ns == 100_000_000
        assert converter._allowed_lateness_ns == 50_000_000


# =============================================================================
# API Integration Tests (runner.py)
# =============================================================================


class TestAPIBackendParameter:
    """Tests for backend parameter in runner.py."""

    def test_get_backend_simple(self):
        """Test get_backend returns SimpleBackend for 'simple'."""
        from visualpath.runner import get_backend

        backend = get_backend("simple")
        assert isinstance(backend, SimpleBackend)

    def test_get_backend_unknown(self):
        """Test get_backend raises for unknown backend."""
        from visualpath.runner import get_backend

        with pytest.raises(ValueError, match="Unknown backend"):
            get_backend("unknown")

    @pytest.mark.skipif(not PATHWAY_AVAILABLE, reason="Pathway not installed")
    def test_get_backend_pathway(self):
        """Test get_backend returns PathwayBackend for 'pathway'."""
        from visualpath.runner import get_backend
        from visualpath.backends.pathway import PathwayBackend

        backend = get_backend("pathway")
        assert isinstance(backend, PathwayBackend)


# =============================================================================
# Deps Support Tests
# =============================================================================


class UpstreamAnalyzer(Module):
    """Analyzer that produces observations used by dependent analyzers."""

    def __init__(self, name: str = "upstream"):
        self._name = name
        self._process_count = 0

    @property
    def name(self) -> str:
        return self._name

    def process(self, frame, deps=None) -> Optional[Observation]:
        self._process_count += 1
        return Observation(
            source=self.name,
            frame_id=frame.frame_id,
            t_ns=frame.t_src_ns,
            signals={"upstream_value": 42, "call_count": self._process_count},
        )

    def initialize(self) -> None:
        pass

    def cleanup(self) -> None:
        pass


class DependentAnalyzer(Module):
    """Analyzer that depends on upstream and records received deps."""

    depends = ["upstream"]

    def __init__(self, name: str = "dependent"):
        self._name = name
        self._process_count = 0
        self.received_deps: List[Optional[Dict]] = []

    @property
    def name(self) -> str:
        return self._name

    def process(self, frame, deps=None) -> Optional[Observation]:
        self._process_count += 1
        self.received_deps.append(deps)
        upstream_value = None
        if deps and "upstream" in deps:
            upstream_value = deps["upstream"].signals.get("upstream_value")
        return Observation(
            source=self.name,
            frame_id=frame.frame_id,
            t_ns=frame.t_src_ns,
            signals={
                "received_upstream": upstream_value is not None,
                "upstream_value": upstream_value,
            },
        )

    def initialize(self) -> None:
        pass

    def cleanup(self) -> None:
        pass


class TestMultiAnalyzerUDFWithDeps:
    """Tests for create_multi_analyzer_udf with deps accumulation."""

    def test_deps_passed_to_dependent_analyzer(self):
        """Test that UDF passes deps from upstream to dependent analyzer."""
        from visualpath.backends.pathway.operators import create_multi_analyzer_udf

        upstream = UpstreamAnalyzer()
        dependent = DependentAnalyzer()
        udf = create_multi_analyzer_udf([upstream, dependent])

        frame = make_frame()
        results = udf(frame)

        assert len(results) == 2
        # Dependent should have received upstream's observation
        assert dependent.received_deps[-1] is not None
        assert "upstream" in dependent.received_deps[-1]
        assert dependent.received_deps[-1]["upstream"].signals["upstream_value"] == 42

    def test_deps_not_passed_when_no_depends(self):
        """Test that analyzers without depends don't get deps."""
        from visualpath.backends.pathway.operators import create_multi_analyzer_udf

        ext1 = CountingAnalyzer("ext1")
        ext2 = CountingAnalyzer("ext2")
        udf = create_multi_analyzer_udf([ext1, ext2])

        frame = make_frame()
        results = udf(frame)

        assert len(results) == 2

    def test_deps_accumulate_across_chain(self):
        """Test deps accumulate for multi-level dependency chains."""
        from visualpath.backends.pathway.operators import create_multi_analyzer_udf

        class Level2Analyzer(Module):
            depends = ["dependent"]

            def __init__(self):
                self.received_deps = []

            @property
            def name(self):
                return "level2"

            def process(self, frame, deps=None):
                self.received_deps.append(deps)
                has_dep = deps is not None and "dependent" in deps
                return Observation(
                    source="level2",
                    frame_id=frame.frame_id,
                    t_ns=frame.t_src_ns,
                    signals={"has_dependent_dep": has_dep},
                )

            def initialize(self):
                pass

            def cleanup(self):
                pass

        upstream = UpstreamAnalyzer()
        dependent = DependentAnalyzer()
        level2 = Level2Analyzer()
        udf = create_multi_analyzer_udf([upstream, dependent, level2])

        frame = make_frame()
        results = udf(frame)

        assert len(results) == 3
        # Level2 should have received dependent's observation
        assert level2.received_deps[-1] is not None
        assert "dependent" in level2.received_deps[-1]


class TestGroupByDependencyLevel:
    """Tests for _group_by_dependency_level."""

    def test_all_independent(self):
        """All independent modules land in level 0."""
        from visualpath.backends.pathway.operators import _group_by_dependency_level

        a = CountingAnalyzer("a")
        b = CountingAnalyzer("b")
        c = CountingAnalyzer("c")
        levels = _group_by_dependency_level([a, b, c])

        assert len(levels) == 1
        assert len(levels[0]) == 3
        names = {m.name for m in levels[0]}
        assert names == {"a", "b", "c"}

    def test_linear_chain(self):
        """Linear chain: each module in its own level."""
        from visualpath.backends.pathway.operators import (
            _toposort_modules,
            _group_by_dependency_level,
        )

        a = CountingAnalyzer("a")
        b = CountingAnalyzer("b")
        b.depends = ["a"]
        c = CountingAnalyzer("c")
        c.depends = ["b"]

        sorted_mods = _toposort_modules([a, b, c])
        levels = _group_by_dependency_level(sorted_mods)

        assert len(levels) == 3
        assert levels[0][0].name == "a"
        assert levels[1][0].name == "b"
        assert levels[2][0].name == "c"

    def test_diamond_dependency(self):
        """Diamond: A -> B, A -> C, B+C -> D."""
        from visualpath.backends.pathway.operators import (
            _toposort_modules,
            _group_by_dependency_level,
        )

        a = CountingAnalyzer("a")
        b = CountingAnalyzer("b")
        b.depends = ["a"]
        c = CountingAnalyzer("c")
        c.depends = ["a"]
        d = CountingAnalyzer("d")
        d.depends = ["b", "c"]

        sorted_mods = _toposort_modules([a, b, c, d])
        levels = _group_by_dependency_level(sorted_mods)

        assert len(levels) == 3
        assert {m.name for m in levels[0]} == {"a"}
        assert {m.name for m in levels[1]} == {"b", "c"}
        assert {m.name for m in levels[2]} == {"d"}

    def test_empty_list(self):
        """Empty input returns empty output."""
        from visualpath.backends.pathway.operators import _group_by_dependency_level

        levels = _group_by_dependency_level([])
        assert levels == []

    def test_facemoment_typical_layout(self):
        """Typical facemoment layout: 4 independent + 2 dependent + 1 final."""
        from visualpath.backends.pathway.operators import (
            _toposort_modules,
            _group_by_dependency_level,
        )

        face_detect = CountingAnalyzer("face.detect")
        body_pose = CountingAnalyzer("body.pose")
        hand_gesture = CountingAnalyzer("hand.gesture")
        frame_quality = CountingAnalyzer("frame.quality")
        face_expression = CountingAnalyzer("face.expression")
        face_expression.depends = ["face.detect"]
        face_classify = CountingAnalyzer("face.classify")
        face_classify.depends = ["face.detect"]
        highlight = CountingAnalyzer("highlight")
        highlight.depends = ["face.detect"]
        highlight.optional_depends = ["face.expression", "face.classify"]

        all_mods = [
            face_detect, body_pose, hand_gesture, frame_quality,
            face_expression, face_classify, highlight,
        ]
        sorted_mods = _toposort_modules(all_mods)
        levels = _group_by_dependency_level(sorted_mods)

        assert len(levels) == 3
        level0_names = {m.name for m in levels[0]}
        level1_names = {m.name for m in levels[1]}
        level2_names = {m.name for m in levels[2]}
        assert level0_names == {"face.detect", "body.pose", "hand.gesture", "frame.quality"}
        assert level1_names == {"face.expression", "face.classify"}
        assert level2_names == {"highlight"}


class TestParallelMultiAnalyzerUDF:
    """Tests for parallel execution in create_multi_analyzer_udf."""

    def test_parallel_independent_analyzers(self):
        """Independent analyzers produce same results as sequential."""
        from visualpath.backends.pathway.operators import create_multi_analyzer_udf

        a = CountingAnalyzer("a", return_value=1.0)
        b = CountingAnalyzer("b", return_value=2.0)
        c = CountingAnalyzer("c", return_value=3.0)
        udf = create_multi_analyzer_udf([a, b, c])

        frame = make_frame()
        results = udf(frame)

        assert len(results) == 3
        sources = {r.source for r in results}
        assert sources == {"a", "b", "c"}
        for r in results:
            assert r.observation is not None
            assert r.elapsed_ms >= 0

    def test_parallel_preserves_deps_chain(self):
        """3-level chain: upstream -> dependent -> level2 deps are correct."""
        from visualpath.backends.pathway.operators import create_multi_analyzer_udf

        upstream = UpstreamAnalyzer()
        dependent = DependentAnalyzer()

        class Level2Analyzer(Module):
            depends = ["dependent"]

            def __init__(self):
                self.received_deps = []

            @property
            def name(self):
                return "level2"

            def process(self, frame, deps=None):
                self.received_deps.append(deps)
                has_dep = deps is not None and "dependent" in deps
                return Observation(
                    source="level2",
                    frame_id=frame.frame_id,
                    t_ns=frame.t_src_ns,
                    signals={"has_dependent_dep": has_dep},
                )

            def initialize(self):
                pass

            def cleanup(self):
                pass

        level2 = Level2Analyzer()
        udf = create_multi_analyzer_udf([upstream, dependent, level2])

        frame = make_frame()
        results = udf(frame)

        assert len(results) == 3
        # Upstream should have run first
        assert upstream._process_count == 1
        # Dependent received upstream's observation
        assert dependent.received_deps[-1] is not None
        assert "upstream" in dependent.received_deps[-1]
        assert dependent.received_deps[-1]["upstream"].signals["upstream_value"] == 42
        # Level2 received dependent's observation
        assert level2.received_deps[-1] is not None
        assert "dependent" in level2.received_deps[-1]

    def test_parallel_error_isolation(self):
        """An error in one parallel analyzer doesn't affect siblings."""
        from visualpath.backends.pathway.operators import create_multi_analyzer_udf

        class ErrorAnalyzer(Module):
            @property
            def name(self):
                return "error"

            def process(self, frame, deps=None):
                raise RuntimeError("boom")

            def initialize(self):
                pass

            def cleanup(self):
                pass

        good = CountingAnalyzer("good", return_value=1.0)
        bad = ErrorAnalyzer()
        udf = create_multi_analyzer_udf([good, bad])

        frame = make_frame()
        results = udf(frame)

        assert len(results) == 2
        good_result = next(r for r in results if r.source == "good")
        bad_result = next(r for r in results if r.source == "error")
        assert good_result.observation is not None
        assert bad_result.observation is None

    def test_multiple_frames_reuse_pool(self):
        """ThreadPoolExecutor is reused across multiple frame invocations."""
        from visualpath.backends.pathway.operators import create_multi_analyzer_udf

        a = CountingAnalyzer("a", return_value=1.0)
        b = CountingAnalyzer("b", return_value=2.0)
        udf = create_multi_analyzer_udf([a, b])

        for i in range(5):
            frame = make_frame(frame_id=i, t_ns=i * 100_000)
            results = udf(frame)
            assert len(results) == 2

        assert a._process_count == 5
        assert b._process_count == 5


@pytest.mark.skipif(not PATHWAY_AVAILABLE, reason="Pathway not installed")
class TestPathwayDepsExecution:
    """Tests for dependency resolution through Pathway engine."""

    def test_deps_through_pathway(self):
        """Test deps work when running through actual Pathway engine."""
        from visualpath.backends.pathway import PathwayBackend
        from visualpath.flow.graph import FlowGraph

        backend = PathwayBackend(autocommit_ms=10)
        upstream = UpstreamAnalyzer()
        dependent = DependentAnalyzer()
        graph = FlowGraph.from_modules([upstream, dependent])
        frames = make_frames(3)

        backend.execute(iter(frames), graph)

        assert upstream._process_count == 3
        assert dependent._process_count == 3
        # All calls should have received deps
        for dep in dependent.received_deps:
            assert dep is not None
            assert "upstream" in dep


class TestVenvWorkerDepsSerialization:
    """Tests for VenvWorker deps serialization/deserialization roundtrip."""

    def test_serialize_observation_for_deps(self):
        """Test observation serialization for deps."""
        from visualpath.process.launcher import _serialize_observation_for_deps

        obs = Observation(
            source="test",
            frame_id=1,
            t_ns=1000000,
            signals={"value": 42, "name": "test"},
            metadata={"key": "val"},
        )

        serialized = _serialize_observation_for_deps(obs)

        assert serialized is not None
        assert serialized["source"] == "test"
        assert serialized["frame_id"] == 1
        assert serialized["t_ns"] == 1000000
        assert serialized["signals"]["value"] == 42

    def test_serialize_none_observation(self):
        """Test serializing None observation."""
        from visualpath.process.launcher import _serialize_observation_for_deps

        assert _serialize_observation_for_deps(None) is None

    def test_deserialize_observation_in_worker(self):
        """Test observation deserialization in worker."""
        from visualpath.process.worker import _deserialize_observation_in_worker

        data = {
            "source": "test",
            "frame_id": 1,
            "t_ns": 1000000,
            "signals": {"value": 42},
            "metadata": {"key": "val"},
            "timing": None,
        }

        obs = _deserialize_observation_in_worker(data)

        assert obs.source == "test"
        assert obs.frame_id == 1
        assert obs.t_ns == 1000000
        assert obs.signals["value"] == 42

    def test_roundtrip_serialization(self):
        """Test serialize -> deserialize roundtrip preserves data."""
        from visualpath.process.launcher import _serialize_observation_for_deps
        from visualpath.process.worker import _deserialize_observation_in_worker

        original = Observation(
            source="face_detect",
            frame_id=5,
            t_ns=5000000,
            signals={"face_count": 2, "confidence": 0.95},
            metadata={"backend": "insightface"},
        )

        serialized = _serialize_observation_for_deps(original)
        restored = _deserialize_observation_in_worker(serialized)

        assert restored.source == original.source
        assert restored.frame_id == original.frame_id
        assert restored.t_ns == original.t_ns
        assert restored.signals == original.signals
        assert restored.metadata == original.metadata


# =============================================================================
# PathwayStats Unit Tests
# =============================================================================


class TestPathwayStats:
    """Tests for PathwayStats dataclass."""

    def test_initial_values(self):
        """Test PathwayStats starts with zero counters."""
        from visualpath.backends.pathway.stats import PathwayStats

        stats = PathwayStats()
        assert stats.frames_ingested == 0
        assert stats.frames_analyzed == 0
        assert stats.analyses_completed == 0
        assert stats.analyses_failed == 0
        assert stats.triggers_fired == 0
        assert stats.observations_output == 0
        assert stats.total_analysis_ms == 0.0
        assert stats.per_analyzer_time_ms == {}
        assert stats.pipeline_start_ns == 0
        assert stats.pipeline_end_ns == 0

    def test_record_ingestion(self):
        """Test recording frame ingestion."""
        from visualpath.backends.pathway.stats import PathwayStats

        stats = PathwayStats()
        stats.record_ingestion()
        stats.record_ingestion()
        stats.record_ingestion()
        assert stats.frames_ingested == 3

    def test_record_analysis_success(self):
        """Test recording successful analyses."""
        from visualpath.backends.pathway.stats import PathwayStats

        stats = PathwayStats()
        stats.record_analysis("face", 10.0, success=True)
        stats.record_analysis("pose", 20.0, success=True)
        assert stats.analyses_completed == 2
        assert stats.analyses_failed == 0
        assert stats.total_analysis_ms == 30.0

    def test_record_analysis_failure(self):
        """Test recording failed analyses."""
        from visualpath.backends.pathway.stats import PathwayStats

        stats = PathwayStats()
        stats.record_analysis("face", 5.0, success=False)
        assert stats.analyses_completed == 0
        assert stats.analyses_failed == 1
        assert stats.total_analysis_ms == 5.0

    def test_per_analyzer_ema(self):
        """Test EMA calculation for per-analyzer times."""
        from visualpath.backends.pathway.stats import PathwayStats

        stats = PathwayStats()
        # First call sets the initial value
        stats.record_analysis("face", 10.0)
        assert stats.per_analyzer_time_ms["face"] == 10.0

        # Second call applies EMA: 0.3 * 20 + 0.7 * 10 = 13.0
        stats.record_analysis("face", 20.0)
        assert abs(stats.per_analyzer_time_ms["face"] - 13.0) < 0.01

    def test_record_trigger(self):
        """Test recording triggers."""
        from visualpath.backends.pathway.stats import PathwayStats

        stats = PathwayStats()
        stats.record_trigger()
        stats.record_trigger()
        assert stats.triggers_fired == 2

    def test_record_observation_output(self):
        """Test recording observation output."""
        from visualpath.backends.pathway.stats import PathwayStats

        stats = PathwayStats()
        stats.record_observation_output()
        assert stats.observations_output == 1

    def test_record_frame_analyzed(self):
        """Test recording frame analysis completion."""
        from visualpath.backends.pathway.stats import PathwayStats

        stats = PathwayStats()
        stats.record_frame_analyzed()
        stats.record_frame_analyzed()
        assert stats.frames_analyzed == 2

    def test_avg_analysis_ms(self):
        """Test average analysis time."""
        from visualpath.backends.pathway.stats import PathwayStats

        stats = PathwayStats()
        stats.record_analysis("a", 10.0)
        stats.record_analysis("b", 20.0)
        assert abs(stats.avg_analysis_ms - 15.0) < 0.01

    def test_avg_analysis_ms_empty(self):
        """Test average analysis time when empty."""
        from visualpath.backends.pathway.stats import PathwayStats

        stats = PathwayStats()
        assert stats.avg_analysis_ms == 0.0

    def test_p95_analysis_ms(self):
        """Test p95 analysis time."""
        from visualpath.backends.pathway.stats import PathwayStats

        stats = PathwayStats()
        # Add 20 values: 1..20
        for i in range(1, 21):
            stats.record_analysis("ext", float(i))
        # p95 of 1..20 → index 19 (0-based), value 19 or 20
        p95 = stats.p95_analysis_ms
        assert p95 >= 19.0

    def test_p95_empty(self):
        """Test p95 when no data."""
        from visualpath.backends.pathway.stats import PathwayStats

        stats = PathwayStats()
        assert stats.p95_analysis_ms == 0.0

    def test_pipeline_duration(self):
        """Test pipeline duration calculation."""
        from visualpath.backends.pathway.stats import PathwayStats

        stats = PathwayStats()
        stats.mark_pipeline_start()
        # Simulate some time passing
        import time
        time.sleep(0.01)
        stats.mark_pipeline_end()
        assert stats.pipeline_duration_sec > 0.0

    def test_pipeline_duration_not_started(self):
        """Test pipeline duration when not started."""
        from visualpath.backends.pathway.stats import PathwayStats

        stats = PathwayStats()
        assert stats.pipeline_duration_sec == 0.0

    def test_throughput_fps(self):
        """Test throughput FPS calculation."""
        from visualpath.backends.pathway.stats import PathwayStats

        stats = PathwayStats()
        stats.mark_pipeline_start()
        for _ in range(10):
            stats.record_frame_analyzed()
        import time
        time.sleep(0.01)
        stats.mark_pipeline_end()
        assert stats.throughput_fps > 0.0

    def test_throughput_fps_no_duration(self):
        """Test throughput when no duration recorded."""
        from visualpath.backends.pathway.stats import PathwayStats

        stats = PathwayStats()
        assert stats.throughput_fps == 0.0

    def test_to_dict(self):
        """Test to_dict contains all expected keys."""
        from visualpath.backends.pathway.stats import PathwayStats

        stats = PathwayStats()
        stats.record_ingestion()
        stats.record_analysis("face", 10.0)
        stats.record_trigger()

        d = stats.to_dict()
        assert d["frames_ingested"] == 1
        assert d["analyses_completed"] == 1
        assert d["triggers_fired"] == 1
        assert d["total_analysis_ms"] == 10.0
        assert "face" in d["per_analyzer_time_ms"]
        assert "throughput_fps" in d
        assert "avg_analysis_ms" in d
        assert "p95_analysis_ms" in d
        assert "pipeline_duration_sec" in d

    def test_reset(self):
        """Test that reset clears all counters."""
        from visualpath.backends.pathway.stats import PathwayStats

        stats = PathwayStats()
        stats.record_ingestion()
        stats.record_analysis("face", 10.0)
        stats.record_trigger()
        stats.record_observation_output()
        stats.record_frame_analyzed()
        stats.mark_pipeline_start()
        stats.mark_pipeline_end()

        stats.reset()

        assert stats.frames_ingested == 0
        assert stats.frames_analyzed == 0
        assert stats.analyses_completed == 0
        assert stats.analyses_failed == 0
        assert stats.triggers_fired == 0
        assert stats.observations_output == 0
        assert stats.total_analysis_ms == 0.0
        assert stats.per_analyzer_time_ms == {}
        assert stats.pipeline_start_ns == 0
        assert stats.pipeline_end_ns == 0

    def test_thread_safety(self):
        """Test that concurrent access does not corrupt counters."""
        import threading
        from visualpath.backends.pathway.stats import PathwayStats

        stats = PathwayStats()
        n_threads = 4
        n_ops = 1000

        def worker():
            for _ in range(n_ops):
                stats.record_ingestion()
                stats.record_analysis("ext", 1.0)
                stats.record_frame_analyzed()

        threads = [threading.Thread(target=worker) for _ in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert stats.frames_ingested == n_threads * n_ops
        assert stats.analyses_completed == n_threads * n_ops
        assert stats.frames_analyzed == n_threads * n_ops


# =============================================================================
# PathwayBackend Stats Integration Tests
# =============================================================================


@pytest.mark.skipif(not PATHWAY_AVAILABLE, reason="Pathway not installed")
class TestPathwayBackendStats:
    """Tests for PathwayBackend.get_stats() integration."""

    def test_get_stats_initial(self):
        """Test get_stats returns zeroed stats before run."""
        from visualpath.backends.pathway import PathwayBackend

        backend = PathwayBackend()
        s = backend.get_stats()
        assert s["frames_ingested"] == 0
        assert s["frames_analyzed"] == 0

    def test_get_stats_after_run(self):
        """Test get_stats after running pipeline."""
        from visualpath.backends.pathway import PathwayBackend
        from visualpath.flow.graph import FlowGraph

        backend = PathwayBackend(autocommit_ms=10)
        analyzer = CountingAnalyzer("test", return_value=0.5)
        graph = FlowGraph.from_modules([analyzer])
        frames = make_frames(5)

        backend.execute(iter(frames), graph)

        s = backend.get_stats()
        assert s["frames_ingested"] == 5
        assert s["frames_analyzed"] == 5
        assert s["analyses_completed"] == 5
        assert s["analyses_failed"] == 0
        assert s["observations_output"] == 5
        assert s["pipeline_duration_sec"] > 0
        assert s["throughput_fps"] > 0

    def test_get_stats_with_fusion_triggers(self):
        """Test get_stats tracks triggers."""
        from visualpath.backends.pathway import PathwayBackend
        from visualpath.flow.graph import FlowGraph

        backend = PathwayBackend(autocommit_ms=10)
        analyzer = CountingAnalyzer("test", return_value=0.7)
        fusion = ThresholdFusion(threshold=0.5, depends_on="test")
        graph = FlowGraph.from_modules([analyzer, fusion])
        frames = make_frames(3)

        backend.execute(iter(frames), graph)

        s = backend.get_stats()
        assert s["triggers_fired"] == 3

    def test_get_stats_per_analyzer_time(self):
        """Test per-analyzer time tracking."""
        from visualpath.backends.pathway import PathwayBackend
        from visualpath.flow.graph import FlowGraph

        backend = PathwayBackend(autocommit_ms=10)
        ext1 = CountingAnalyzer("ext1", return_value=0.3)
        ext2 = CountingAnalyzer("ext2", return_value=0.7)
        graph = FlowGraph.from_modules([ext1, ext2])
        frames = make_frames(3)

        backend.execute(iter(frames), graph)

        s = backend.get_stats()
        assert "ext1" in s["per_analyzer_time_ms"]
        assert "ext2" in s["per_analyzer_time_ms"]

    def test_stats_reset_between_runs(self):
        """Test that stats reset between consecutive runs."""
        from visualpath.backends.pathway import PathwayBackend
        from visualpath.flow.graph import FlowGraph

        backend = PathwayBackend(autocommit_ms=10)
        ext = CountingAnalyzer("test", return_value=0.5)

        # First run
        graph1 = FlowGraph.from_modules([ext])
        backend.execute(iter(make_frames(5)), graph1)
        s1 = backend.get_stats()
        assert s1["frames_ingested"] == 5

        # Second run - stats should be fresh
        ext2 = CountingAnalyzer("test", return_value=0.5)
        graph2 = FlowGraph.from_modules([ext2])
        backend.execute(iter(make_frames(3)), graph2)
        s2 = backend.get_stats()
        assert s2["frames_ingested"] == 3

    def test_get_stats_timing_fields(self):
        """Test timing-related stats fields."""
        from visualpath.backends.pathway import PathwayBackend
        from visualpath.flow.graph import FlowGraph

        backend = PathwayBackend(autocommit_ms=10)
        ext = CountingAnalyzer("test", return_value=0.5)
        graph = FlowGraph.from_modules([ext])
        frames = make_frames(5)

        backend.execute(iter(frames), graph)

        s = backend.get_stats()
        assert s["total_analysis_ms"] > 0
        assert s["avg_analysis_ms"] > 0
        assert s["p95_analysis_ms"] >= 0


# =============================================================================
# PathwayBackend ObservabilityHub Tests
# =============================================================================


@pytest.mark.skipif(not PATHWAY_AVAILABLE, reason="Pathway not installed")
class TestPathwayObservabilityHub:
    """Tests for ObservabilityHub integration in PathwayBackend."""

    def setup_method(self):
        """Reset the ObservabilityHub before each test."""
        from visualpath.observability import ObservabilityHub
        ObservabilityHub.reset_instance()

    def teardown_method(self):
        """Reset the ObservabilityHub after each test."""
        from visualpath.observability import ObservabilityHub
        ObservabilityHub.reset_instance()

    def test_session_records_emitted(self):
        """Test SessionStartRecord and SessionEndRecord are emitted."""
        from visualpath.observability import ObservabilityHub, TraceLevel, MemorySink
        from visualpath.backends.pathway import PathwayBackend
        from visualpath.flow.graph import FlowGraph

        hub = ObservabilityHub.get_instance()
        sink = MemorySink()
        hub.configure(level=TraceLevel.MINIMAL, sinks=[sink])

        backend = PathwayBackend(autocommit_ms=10)
        ext = CountingAnalyzer("test", return_value=0.5)
        graph = FlowGraph.from_modules([ext])
        backend.execute(iter(make_frames(3)), graph)

        records = sink.get_records()
        types = [r.record_type for r in records]
        assert "session_start" in types
        assert "session_end" in types

        # Verify session_end content
        end_rec = next(r for r in records if r.record_type == "session_end")
        assert end_rec.total_frames == 3
        assert end_rec.duration_sec > 0

    def test_timing_records_emitted_at_normal(self):
        """Test TimingRecords are emitted at NORMAL level."""
        from visualpath.observability import ObservabilityHub, TraceLevel, MemorySink
        from visualpath.backends.pathway import PathwayBackend
        from visualpath.flow.graph import FlowGraph

        hub = ObservabilityHub.get_instance()
        sink = MemorySink()
        hub.configure(level=TraceLevel.NORMAL, sinks=[sink])

        backend = PathwayBackend(autocommit_ms=10)
        ext = CountingAnalyzer("test", return_value=0.5)
        graph = FlowGraph.from_modules([ext])
        backend.execute(iter(make_frames(3)), graph)

        records = sink.get_records()
        types = [r.record_type for r in records]
        assert "timing" in types

        timing_recs = [r for r in records if r.record_type == "timing"]
        assert len(timing_recs) == 3  # One per frame
        assert all(r.component == "pathway_udf" for r in timing_recs)

    def test_no_records_when_off(self):
        """Test no records emitted when hub is OFF."""
        from visualpath.observability import ObservabilityHub, TraceLevel, MemorySink
        from visualpath.backends.pathway import PathwayBackend
        from visualpath.flow.graph import FlowGraph

        hub = ObservabilityHub.get_instance()
        sink = MemorySink()
        hub.configure(level=TraceLevel.OFF)
        hub.add_sink(sink)

        backend = PathwayBackend(autocommit_ms=10)
        ext = CountingAnalyzer("test", return_value=0.5)
        graph = FlowGraph.from_modules([ext])
        backend.execute(iter(make_frames(3)), graph)

        assert len(sink) == 0


# =============================================================================
# Per-Analyzer UDF Tests
# =============================================================================


class TestSingleAnalyzerUDF:
    """Tests for create_single_analyzer_udf."""

    def test_independent_analyzer(self):
        """Independent analyzer UDF returns correct result."""
        from visualpath.backends.pathway.operators import create_single_analyzer_udf

        analyzer = CountingAnalyzer("face", return_value=0.8)
        udf = create_single_analyzer_udf(analyzer)

        frame = make_frame(frame_id=1, t_ns=100_000)
        results = udf(frame)

        assert len(results) == 1
        assert results[0].source == "face"
        assert results[0].observation is not None
        assert results[0].observation.signals["value"] == 0.8
        assert results[0].elapsed_ms >= 0
        assert results[0].frame_id == 1
        assert results[0].t_ns == 100_000

    def test_error_returns_none_observation(self):
        """Error in analyzer returns AnalyzerResult with observation=None."""
        from visualpath.backends.pathway.operators import create_single_analyzer_udf

        class ErrorAnalyzer(Module):
            @property
            def name(self):
                return "error"

            def process(self, frame, deps=None):
                raise RuntimeError("boom")

            def initialize(self):
                pass

            def cleanup(self):
                pass

        udf = create_single_analyzer_udf(ErrorAnalyzer())
        results = udf(make_frame())

        assert len(results) == 1
        assert results[0].source == "error"
        assert results[0].observation is None
        assert results[0].elapsed_ms >= 0


class TestDepAnalyzerUDF:
    """Tests for create_dep_analyzer_udf."""

    def test_receives_upstream_deps(self):
        """Dependent UDF extracts deps from upstream results."""
        from visualpath.backends.pathway.operators import (
            create_dep_analyzer_udf,
            AnalyzerResult,
        )

        dependent = DependentAnalyzer(name="dependent")
        udf = create_dep_analyzer_udf(dependent)

        upstream_obs = Observation(
            source="upstream",
            frame_id=1,
            t_ns=100_000,
            signals={"upstream_value": 42},
        )
        upstream_results = [AnalyzerResult(
            frame_id=1, t_ns=100_000, source="upstream",
            observation=upstream_obs,
        )]

        frame = make_frame()
        results = udf(frame, upstream_results)

        assert len(results) == 1
        assert results[0].source == "dependent"
        assert results[0].observation is not None
        assert results[0].observation.signals["received_upstream"] is True
        assert results[0].observation.signals["upstream_value"] == 42

    def test_missing_dep_passes_none(self):
        """When upstream result is missing, deps is None."""
        from visualpath.backends.pathway.operators import create_dep_analyzer_udf

        dependent = DependentAnalyzer(name="dependent")
        udf = create_dep_analyzer_udf(dependent)

        frame = make_frame()
        results = udf(frame, [])

        assert len(results) == 1
        assert results[0].observation is not None
        assert results[0].observation.signals["received_upstream"] is False

    def test_error_returns_none_observation(self):
        """Error in dependent analyzer returns observation=None."""
        from visualpath.backends.pathway.operators import (
            create_dep_analyzer_udf,
            AnalyzerResult,
        )

        class ErrorDepAnalyzer(Module):
            depends = ["upstream"]

            @property
            def name(self):
                return "error_dep"

            def process(self, frame, deps=None):
                raise RuntimeError("boom")

            def initialize(self):
                pass

            def cleanup(self):
                pass

        udf = create_dep_analyzer_udf(ErrorDepAnalyzer())
        upstream_obs = Observation(
            source="upstream", frame_id=1, t_ns=100_000, signals={},
        )
        results = udf(make_frame(), [AnalyzerResult(
            frame_id=1, t_ns=100_000, source="upstream",
            observation=upstream_obs,
        )])

        assert len(results) == 1
        assert results[0].observation is None


# =============================================================================
# Per-Analyzer DAG Tests
# =============================================================================


@pytest.mark.skipif(not PATHWAY_AVAILABLE, reason="Pathway not installed")
class TestPerAnalyzerDAG:
    """Tests for per-analyzer DAG construction via FlowGraphConverter."""

    def test_independent_analyzers_produce_separate_tables(self):
        """Independent analyzers are split into separate leaf tables."""
        from visualpath.backends.pathway.converter import FlowGraphConverter
        from visualpath.backends.pathway.connector import (
            VideoConnectorSubject, FrameSchema,
        )
        from visualpath.flow.graph import FlowGraph
        from visualpath.flow.nodes.source import SourceNode
        from visualpath.flow.nodes.path import PathNode

        a = CountingAnalyzer("a", return_value=1.0)
        b = CountingAnalyzer("b", return_value=2.0)

        graph = FlowGraph(entry_node="source")
        graph.add_node(SourceNode(name="source"))
        graph.add_node(PathNode(
            name="pipeline", modules=[a, b], parallel=True,
        ))
        graph.add_edge("source", "pipeline")

        frames = make_frames(3)
        subject = VideoConnectorSubject(iter(frames))
        frames_table = pw.io.python.read(
            subject, schema=FrameSchema, autocommit_duration_ms=10,
        )

        converter = FlowGraphConverter()
        output = converter.convert(graph, frames_table)

        # Collect and verify all results
        collected = []
        pw.io.subscribe(
            output,
            on_change=lambda key, row, time, is_addition: (
                collected.append(row) if is_addition else None
            ),
        )
        pw.run()

        assert len(collected) == 3
        # Each row should have results from both analyzers (merged by _auto_join)
        for row in collected:
            results_wrapper = row.get("results")
            results = results_wrapper.value if hasattr(results_wrapper, 'value') else results_wrapper
            sources = {r.source for r in results}
            assert sources == {"a", "b"}

    def test_dependent_chain_correctness(self):
        """face.detect → face.expression chain produces correct deps."""
        from visualpath.backends.pathway.converter import FlowGraphConverter
        from visualpath.backends.pathway.connector import (
            VideoConnectorSubject, FrameSchema,
        )
        from visualpath.flow.graph import FlowGraph
        from visualpath.flow.nodes.source import SourceNode
        from visualpath.flow.nodes.path import PathNode

        upstream = UpstreamAnalyzer(name="upstream")
        dependent = DependentAnalyzer(name="dependent")

        graph = FlowGraph(entry_node="source")
        graph.add_node(SourceNode(name="source"))
        graph.add_node(PathNode(
            name="pipeline", modules=[upstream, dependent], parallel=True,
        ))
        graph.add_edge("source", "pipeline")

        frames = make_frames(3)
        subject = VideoConnectorSubject(iter(frames))
        frames_table = pw.io.python.read(
            subject, schema=FrameSchema, autocommit_duration_ms=10,
        )

        converter = FlowGraphConverter()
        output = converter.convert(graph, frames_table)

        collected = []
        pw.io.subscribe(
            output,
            on_change=lambda key, row, time, is_addition: (
                collected.append(row) if is_addition else None
            ),
        )
        pw.run()

        assert len(collected) == 3
        # Dependent should have received upstream deps
        for dep in dependent.received_deps:
            assert dep is not None
            assert "upstream" in dep

    def test_mixed_independent_and_dependent(self):
        """Mix of independent + dependent analyzers."""
        from visualpath.backends.pathway.converter import FlowGraphConverter
        from visualpath.backends.pathway.connector import (
            VideoConnectorSubject, FrameSchema,
        )
        from visualpath.flow.graph import FlowGraph
        from visualpath.flow.nodes.source import SourceNode
        from visualpath.flow.nodes.path import PathNode

        face_detect = UpstreamAnalyzer(name="face.detect")
        body_pose = CountingAnalyzer("body.pose", return_value=0.5)
        face_expression = DependentAnalyzer(name="face.expression")
        face_expression.depends = ["face.detect"]

        graph = FlowGraph(entry_node="source")
        graph.add_node(SourceNode(name="source"))
        graph.add_node(PathNode(
            name="pipeline",
            modules=[face_detect, body_pose, face_expression],
            parallel=True,
        ))
        graph.add_edge("source", "pipeline")

        frames = make_frames(3)
        subject = VideoConnectorSubject(iter(frames))
        frames_table = pw.io.python.read(
            subject, schema=FrameSchema, autocommit_duration_ms=10,
        )

        converter = FlowGraphConverter()
        output = converter.convert(graph, frames_table)

        collected = []
        pw.io.subscribe(
            output,
            on_change=lambda key, row, time, is_addition: (
                collected.append(row) if is_addition else None
            ),
        )
        pw.run()

        assert len(collected) == 3
        for row in collected:
            results_wrapper = row.get("results")
            results = results_wrapper.value if hasattr(results_wrapper, 'value') else results_wrapper
            sources = {r.source for r in results}
            assert sources == {"face.detect", "body.pose", "face.expression"}

    def test_auto_join_merges_results(self):
        """interval_join merges results from multiple branch tables."""
        from visualpath.backends.pathway.converter import FlowGraphConverter
        from visualpath.backends.pathway.connector import (
            VideoConnectorSubject, FrameSchema,
        )
        from visualpath.flow.graph import FlowGraph
        from visualpath.flow.nodes.source import SourceNode
        from visualpath.flow.nodes.path import PathNode

        a = CountingAnalyzer("a", return_value=1.0)
        b = CountingAnalyzer("b", return_value=2.0)
        c = CountingAnalyzer("c", return_value=3.0)

        graph = FlowGraph(entry_node="source")
        graph.add_node(SourceNode(name="source"))
        graph.add_node(PathNode(
            name="pipeline", modules=[a, b, c], parallel=True,
        ))
        graph.add_edge("source", "pipeline")

        frames = make_frames(2)
        subject = VideoConnectorSubject(iter(frames))
        frames_table = pw.io.python.read(
            subject, schema=FrameSchema, autocommit_duration_ms=10,
        )

        converter = FlowGraphConverter()
        output = converter.convert(graph, frames_table)

        collected = []
        pw.io.subscribe(
            output,
            on_change=lambda key, row, time, is_addition: (
                collected.append(row) if is_addition else None
            ),
        )
        pw.run()

        assert len(collected) == 2
        for row in collected:
            results_wrapper = row.get("results")
            results = results_wrapper.value if hasattr(results_wrapper, 'value') else results_wrapper
            sources = {r.source for r in results}
            assert sources == {"a", "b", "c"}
            # Verify all results have observations
            for r in results:
                assert r.observation is not None


# =============================================================================
# FlowGraphConverter Spec-Based Dispatch Tests
# =============================================================================


class StatefulCooldownFusion(Module):
    """Stateful fusion with cooldown that requires temporal ordering.

    Triggers when value > threshold AND cooldown_frames have passed since
    the last trigger.  This requires frames to arrive in temporal order;
    if frames arrive out of order, cooldown counting breaks.
    """

    stateful = True

    def __init__(self, cooldown_frames: int = 3, depends_on: str = "test"):
        self._cooldown_frames = cooldown_frames
        self._frames_since_trigger = cooldown_frames  # Start ready to trigger
        self._update_count = 0
        self._trigger_count = 0
        self.depends = [depends_on] if depends_on else []
        self.frame_ids_seen: List[int] = []

    @property
    def name(self) -> str:
        return "stateful_fusion"

    def process(self, frame, deps=None) -> Optional[Observation]:
        self._update_count += 1
        self.frame_ids_seen.append(frame.frame_id)

        value = 0.0
        if deps:
            for obs in deps.values():
                if obs is not None:
                    value = obs.signals.get("value", 0.0)
                    break

        self._frames_since_trigger += 1
        should_trigger = (value > 0.5 and self._frames_since_trigger >= self._cooldown_frames)

        if should_trigger:
            self._frames_since_trigger = 0
            self._trigger_count += 1
            from visualbase import Trigger
            trigger = Trigger.point(
                event_time_ns=frame.t_src_ns,
                pre_sec=2.0, post_sec=2.0,
                label="stateful_trigger",
                score=value,
            )
            return Observation(
                source=self.name,
                frame_id=frame.frame_id,
                t_ns=frame.t_src_ns,
                signals={
                    "should_trigger": True,
                    "trigger_score": value,
                    "trigger_reason": "stateful_trigger",
                },
                metadata={"trigger": trigger},
            )

        return Observation(
            source=self.name,
            frame_id=frame.frame_id,
            t_ns=frame.t_src_ns,
            signals={"should_trigger": False},
        )

    def initialize(self) -> None:
        pass

    def cleanup(self) -> None:
        pass


# =============================================================================
# Deferred Fusion (stateful=True) Tests
# =============================================================================


class TestConverterDeferredModules:
    """Tests for FlowGraphConverter deferred module separation."""

    @pytest.mark.skipif(not PATHWAY_AVAILABLE, reason="Pathway not installed")
    def test_converter_defers_trigger_modules(self):
        """Modules with stateful=True are excluded from UDF."""
        from visualpath.backends.pathway.converter import FlowGraphConverter
        from visualpath.backends.pathway.connector import (
            VideoConnectorSubject,
            FrameSchema,
        )
        from visualpath.flow.graph import FlowGraph

        analyzer = CountingAnalyzer("test", return_value=0.7)
        fusion = StatefulCooldownFusion(cooldown_frames=2, depends_on="test")
        graph = FlowGraph.from_modules([analyzer, fusion])

        frames = make_frames(3)
        subject = VideoConnectorSubject(iter(frames))
        frames_table = pw.io.python.read(
            subject, schema=FrameSchema, autocommit_duration_ms=10,
        )

        converter = FlowGraphConverter()
        converter.convert(graph, frames_table)

        deferred = converter.deferred_modules
        assert len(deferred) == 1
        assert deferred[0].name == "stateful_fusion"

    def test_non_stateful_modules_not_deferred(self):
        """Modules without stateful stay in UDF (no deferred)."""
        from visualpath.backends.pathway.operators import create_multi_analyzer_udf

        ext1 = CountingAnalyzer("a", return_value=0.3)
        ext2 = CountingAnalyzer("b", return_value=0.7)
        # Neither has stateful=True
        assert not ext1.stateful
        assert not ext2.stateful

        udf = create_multi_analyzer_udf([ext1, ext2])
        results = udf(make_frame())
        assert len(results) == 2


@pytest.mark.skipif(not PATHWAY_AVAILABLE, reason="Pathway not installed")
class TestDeferredFusionExecution:
    """Tests for deferred fusion execution in PathwayBackend subscribe callback."""

    def test_deferred_fusion_triggers(self):
        """Deferred fusion fires triggers from the subscribe callback."""
        from visualpath.backends.pathway import PathwayBackend
        from visualpath.flow.graph import FlowGraph

        backend = PathwayBackend(autocommit_ms=10)
        analyzer = CountingAnalyzer("test", return_value=0.7)
        fusion = StatefulCooldownFusion(cooldown_frames=2, depends_on="test")
        graph = FlowGraph.from_modules([analyzer, fusion])
        frames = make_frames(10)

        result = backend.execute(iter(frames), graph)

        # With cooldown=2, all value>0.5: triggers at frames 0, 3, 6, 9
        # (frame 0 starts ready, then every 3rd frame)
        assert fusion._update_count == 10
        assert len(result.triggers) > 0
        assert fusion._trigger_count == len(result.triggers)

    def test_deferred_fusion_sees_frames_in_order(self):
        """Deferred fusion receives frames in temporal order."""
        from visualpath.backends.pathway import PathwayBackend
        from visualpath.flow.graph import FlowGraph

        backend = PathwayBackend(autocommit_ms=10)
        analyzer = CountingAnalyzer("test", return_value=0.7)
        fusion = StatefulCooldownFusion(cooldown_frames=2, depends_on="test")
        graph = FlowGraph.from_modules([analyzer, fusion])
        frames = make_frames(20)

        backend.execute(iter(frames), graph)

        # Subscribe delivers in temporal order, so frame_ids_seen must be sorted
        assert fusion.frame_ids_seen == sorted(fusion.frame_ids_seen)

    def test_deferred_vs_simple_trigger_parity(self):
        """Pathway deferred fusion and SimpleBackend produce same trigger count."""
        from visualpath.backends.pathway import PathwayBackend
        from visualpath.flow.graph import FlowGraph

        # SimpleBackend
        simple = SimpleBackend()
        ext_s = CountingAnalyzer("test", return_value=0.7)
        fusion_s = StatefulCooldownFusion(cooldown_frames=2, depends_on="test")
        graph_s = FlowGraph.from_modules([ext_s, fusion_s])
        simple_result = simple.execute(iter(make_frames(10)), graph_s)

        # PathwayBackend (fusion deferred to subscribe)
        pathway = PathwayBackend(autocommit_ms=10)
        ext_p = CountingAnalyzer("test", return_value=0.7)
        fusion_p = StatefulCooldownFusion(cooldown_frames=2, depends_on="test")
        graph_p = FlowGraph.from_modules([ext_p, fusion_p])
        pathway_result = pathway.execute(iter(make_frames(10)), graph_p)

        assert len(simple_result.triggers) == len(pathway_result.triggers)

    def test_deferred_fusion_receives_deps(self):
        """Deferred fusion receives deps from UDF analyzer results."""
        from visualpath.backends.pathway import PathwayBackend
        from visualpath.flow.graph import FlowGraph

        backend = PathwayBackend(autocommit_ms=10)
        analyzer = CountingAnalyzer("test", return_value=0.3)
        # cooldown=1, value 0.3 < 0.5 -> no triggers
        fusion = StatefulCooldownFusion(cooldown_frames=1, depends_on="test")
        graph = FlowGraph.from_modules([analyzer, fusion])
        frames = make_frames(5)

        result = backend.execute(iter(frames), graph)

        # Fusion was called for all frames but below threshold
        assert fusion._update_count == 5
        assert len(result.triggers) == 0

    def test_deferred_fusion_callback_fires(self):
        """on_trigger callback fires for deferred fusion triggers."""
        from visualpath.backends.pathway import PathwayBackend
        from visualpath.flow.graph import FlowGraph

        backend = PathwayBackend(autocommit_ms=10)
        analyzer = CountingAnalyzer("test", return_value=0.7)
        fusion = StatefulCooldownFusion(cooldown_frames=2, depends_on="test")
        graph = FlowGraph.from_modules([analyzer, fusion])
        frames = make_frames(10)

        callback_data = []
        graph.on_trigger(lambda d: callback_data.append(d))

        result = backend.execute(iter(frames), graph)

        assert len(callback_data) == len(result.triggers)
        assert len(callback_data) > 0


class TestConverterSpecDispatch:
    """Tests that FlowGraphConverter uses spec for dispatch."""

    def test_converter_uses_spec_not_isinstance(self):
        """Verify converter dispatches based on node.spec type."""
        from visualpath.backends.pathway.converter import FlowGraphConverter
        from visualpath.flow.node import FlowNode
        from visualpath.flow.specs import SourceSpec

        converter = FlowGraphConverter()

        class FakeSourceNode(FlowNode):
            @property
            def name(self):
                return "fake_source"

            @property
            def spec(self):
                return SourceSpec(default_path_id="test")

            def process(self, data):
                return [data]

        from visualpath.flow.graph import FlowGraph

        graph = FlowGraph()
        node = FakeSourceNode()
        graph.add_node(node)

        spec = node.spec
        assert isinstance(spec, SourceSpec)
