"""Tests for SimpleBackend execute() interface.

Tests verify:
- SimpleBackend.execute() with FlowGraph
- PipelineResult structure
"""

import pytest
import numpy as np
from dataclasses import dataclass
from typing import Optional, List

from visualpath.core import Module, Observation
from visualpath.backends.base import ExecutionBackend, PipelineResult


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
    """Analyzer that counts calls."""

    def __init__(self, name: str, value: float = 0.5):
        self._name = name
        self._value = value
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
            signals={"value": self._value, "count": self._process_count},
        )

    def initialize(self) -> None:
        self._initialized = True

    def cleanup(self) -> None:
        self._cleaned_up = True


class ThresholdFusion(Module):
    """Simple fusion for testing."""

    def __init__(self, threshold: float = 0.5, depends_on: str = None):
        self._threshold = threshold
        self._update_count = 0
        self.depends = [depends_on] if depends_on else []

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
            from visualbase import Trigger
            trigger = Trigger.point(
                event_time_ns=observation.t_ns,
                pre_sec=2.0,
                post_sec=2.0,
                label="threshold",
                score=value,
            )
            return Observation(
                source=self.name,
                frame_id=frame.frame_id,
                t_ns=frame.t_src_ns,
                signals={
                    "should_trigger": True,
                    "trigger_score": value,
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


def make_frame(frame_id: int = 1, t_ns: int = 1_000_000) -> MockFrame:
    return MockFrame(
        frame_id=frame_id,
        t_src_ns=t_ns,
        data=np.zeros((100, 100, 3), dtype=np.uint8),
    )


def make_frames(count: int, interval_ns: int = 100_000_000) -> List[MockFrame]:
    return [make_frame(frame_id=i, t_ns=i * interval_ns) for i in range(count)]


# =============================================================================
# SimpleBackend execute() Tests
# =============================================================================


class TestSimpleBackendExecute:
    """Tests for SimpleBackend.execute() with FlowGraph."""

    def test_execute_single_analyzer(self):
        """Test execute() with a single analyzer."""
        from visualpath.backends.simple import SimpleBackend
        from visualpath.flow.graph import FlowGraph

        backend = SimpleBackend()
        ext = CountingAnalyzer("test", value=0.3)
        graph = FlowGraph.from_modules([ext])
        frames = make_frames(5)

        result = backend.execute(iter(frames), graph)

        assert isinstance(result, PipelineResult)
        assert result.frame_count == 5
        assert ext._process_count == 5
        assert len(result.triggers) == 0  # No fusion

    def test_execute_with_fusion(self):
        """Test execute() with fusion that triggers."""
        from visualpath.backends.simple import SimpleBackend
        from visualpath.flow.graph import FlowGraph

        backend = SimpleBackend()
        ext = CountingAnalyzer("test", value=0.7)
        fusion = ThresholdFusion(threshold=0.5, depends_on="test")
        graph = FlowGraph.from_modules([ext, fusion])
        frames = make_frames(5)

        result = backend.execute(iter(frames), graph)

        assert result.frame_count == 5
        assert ext._process_count == 5
        assert len(result.triggers) == 5  # All frames trigger

    def test_execute_no_trigger(self):
        """Test execute() when fusion doesn't fire."""
        from visualpath.backends.simple import SimpleBackend
        from visualpath.flow.graph import FlowGraph

        backend = SimpleBackend()
        ext = CountingAnalyzer("test", value=0.3)
        fusion = ThresholdFusion(threshold=0.5, depends_on="test")
        graph = FlowGraph.from_modules([ext, fusion])
        frames = make_frames(3)

        result = backend.execute(iter(frames), graph)

        assert result.frame_count == 3
        assert len(result.triggers) == 0

    def test_execute_multiple_analyzers(self):
        """Test execute() with multiple analyzers."""
        from visualpath.backends.simple import SimpleBackend
        from visualpath.flow.graph import FlowGraph

        backend = SimpleBackend()
        ext1 = CountingAnalyzer("ext1", value=0.3)
        ext2 = CountingAnalyzer("ext2", value=0.7)
        fusion = ThresholdFusion(threshold=0.5, depends_on="ext1")
        graph = FlowGraph.from_modules([ext1, ext2, fusion])
        frames = make_frames(3)

        result = backend.execute(iter(frames), graph)

        assert ext1._process_count == 3
        assert ext2._process_count == 3
        assert result.frame_count == 3

    def test_execute_empty_frames(self):
        """Test execute() with empty frame iterator."""
        from visualpath.backends.simple import SimpleBackend
        from visualpath.flow.graph import FlowGraph

        backend = SimpleBackend()
        ext = CountingAnalyzer("test")
        graph = FlowGraph.from_modules([ext])

        result = backend.execute(iter([]), graph)

        assert result.frame_count == 0
        assert result.triggers == []

    def test_execute_with_flowgraph_builder(self):
        """Test execute() with a FlowGraphBuilder-built graph."""
        from visualpath.backends.simple import SimpleBackend
        from visualpath.flow import FlowGraphBuilder

        backend = SimpleBackend()
        ext = CountingAnalyzer("ext1", value=0.7)
        fusion = ThresholdFusion(threshold=0.5, depends_on="ext1")

        graph = (FlowGraphBuilder()
            .source("frames")
            .sample(every_nth=2)
            .path("main", modules=[ext, fusion])
            .build())

        triggered = []
        graph.on_trigger(lambda d: triggered.append(d))

        frames = make_frames(6)
        result = backend.execute(iter(frames), graph)

        # Only every 2nd frame is processed
        assert ext._process_count == 3
        assert len(triggered) == 3

    def test_execute_with_callback(self):
        """Test execute() with on_trigger callback registered on graph."""
        from visualpath.backends.simple import SimpleBackend
        from visualpath.flow.graph import FlowGraph

        backend = SimpleBackend()
        ext = CountingAnalyzer("test", value=0.7)
        fusion = ThresholdFusion(threshold=0.5, depends_on="test")
        graph = FlowGraph.from_modules([ext, fusion])
        frames = make_frames(3)

        callback_data = []
        graph.on_trigger(lambda d: callback_data.append(d))

        result = backend.execute(iter(frames), graph)

        assert len(callback_data) == 3
        assert len(result.triggers) == 3


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
        from visualpath.backends.simple import SimpleBackend

        backend = SimpleBackend()
        assert isinstance(backend, ExecutionBackend)
        assert hasattr(backend, "execute")
        assert hasattr(backend, "name")


# =============================================================================
# PipelineResult Tests
# =============================================================================


class TestPipelineResult:
    """Tests for PipelineResult dataclass."""

    def test_default_values(self):
        result = PipelineResult()
        assert result.triggers == []
        assert result.frame_count == 0
        assert result.stats == {}

    def test_with_values(self):
        result = PipelineResult(triggers=["t1"], frame_count=10, stats={"key": "val"})
        assert result.triggers == ["t1"]
        assert result.frame_count == 10
        assert result.stats == {"key": "val"}


# =============================================================================
# Parallel Module Execution Tests
# =============================================================================


class TestParallelModuleExecution:
    """Tests for parallel module dispatch in SimpleInterpreter."""

    def test_level_sort_independent_modules(self):
        """Independent modules are grouped into one level."""
        from visualpath.backends.simple.interpreter import SimpleInterpreter

        a = CountingAnalyzer("a")
        b = CountingAnalyzer("b")
        c = CountingAnalyzer("c")

        interp = SimpleInterpreter()
        levels = interp._level_sort_modules((a, b, c))

        assert len(levels) == 1
        assert len(levels[0]) == 3

    def test_level_sort_with_dependencies(self):
        """Modules with deps are sorted into correct levels."""
        from visualpath.backends.simple.interpreter import SimpleInterpreter

        a = CountingAnalyzer("a")
        b = CountingAnalyzer("b")
        b.depends = ["a"]
        c = CountingAnalyzer("c")
        c.depends = ["b"]

        interp = SimpleInterpreter()
        levels = interp._level_sort_modules((a, b, c))

        assert len(levels) == 3
        assert levels[0][0].name == "a"
        assert levels[1][0].name == "b"
        assert levels[2][0].name == "c"

    def test_level_sort_diamond_deps(self):
        """Diamond dependency: A -> B,C -> D. B and C should be in same level."""
        from visualpath.backends.simple.interpreter import SimpleInterpreter

        a = CountingAnalyzer("a")
        b = CountingAnalyzer("b")
        b.depends = ["a"]
        c = CountingAnalyzer("c")
        c.depends = ["a"]
        d = CountingAnalyzer("d")
        d.depends = ["b", "c"]

        interp = SimpleInterpreter()
        levels = interp._level_sort_modules((a, b, c, d))

        assert len(levels) == 3
        assert levels[0][0].name == "a"
        assert {m.name for m in levels[1]} == {"b", "c"}
        assert levels[2][0].name == "d"

    def test_parallel_execution_produces_same_results(self):
        """Parallel execution produces same observations as sequential."""
        from visualpath.backends.simple.interpreter import SimpleInterpreter
        from visualpath.flow.nodes.path import PathNode
        from visualpath.flow.node import FlowData

        a = CountingAnalyzer("a", value=0.3)
        b = CountingAnalyzer("b", value=0.7)

        # Sequential
        node_seq = PathNode(name="seq", modules=[a, b], parallel=False)
        interp_seq = SimpleInterpreter()
        frame = make_frame()
        data = FlowData(frame=frame, timestamp_ns=frame.t_src_ns)
        out_seq = interp_seq.interpret(node_seq, data)

        # Reset counters
        a._process_count = 0
        b._process_count = 0

        # Parallel
        node_par = PathNode(name="par", modules=[a, b], parallel=True)
        interp_par = SimpleInterpreter()
        data2 = FlowData(frame=frame, timestamp_ns=frame.t_src_ns)
        out_par = interp_par.interpret(node_par, data2)

        assert len(out_seq) == len(out_par) == 1
        seq_obs = out_seq[0].observations
        par_obs = out_par[0].observations
        assert len(seq_obs) == len(par_obs) == 2
        assert {o.source for o in seq_obs} == {o.source for o in par_obs}

    def test_parallel_with_deps_produces_correct_results(self):
        """Parallel execution respects dependencies across levels."""
        from visualpath.backends.simple.interpreter import SimpleInterpreter
        from visualpath.flow.nodes.path import PathNode
        from visualpath.flow.node import FlowData

        class DepModule(Module):
            def __init__(self, name, dep_name=None):
                self._name = name
                self.depends = [dep_name] if dep_name else []
                self.received_deps = None

            @property
            def name(self):
                return self._name

            def process(self, frame, deps=None):
                self.received_deps = deps
                val = 0.0
                if deps:
                    for obs in deps.values():
                        val += obs.signals.get("value", 0)
                return Observation(
                    source=self.name,
                    frame_id=frame.frame_id,
                    t_ns=frame.t_src_ns,
                    signals={"value": val + 1.0},
                )

        a = DepModule("a")
        b = DepModule("b", dep_name="a")

        node = PathNode(name="test", modules=[a, b], parallel=True)
        interp = SimpleInterpreter()
        frame = make_frame()
        data = FlowData(frame=frame, timestamp_ns=frame.t_src_ns)
        out = interp.interpret(node, data)

        obs_map = {o.source: o for o in out[0].observations}
        # a: value=1.0 (no deps)
        assert obs_map["a"].signals["value"] == 1.0
        # b: value=2.0 (a's value + 1.0)
        assert obs_map["b"].signals["value"] == 2.0
        # b received deps from a
        assert b.received_deps is not None
        assert "a" in b.received_deps

    def test_parallel_execution_all_modules_called(self):
        """All modules in parallel execution produce observations."""
        from visualpath.backends.simple import SimpleBackend
        from visualpath.flow.graph import FlowGraph
        from visualpath.flow.nodes.source import SourceNode
        from visualpath.flow.nodes.path import PathNode

        a = CountingAnalyzer("a")
        b = CountingAnalyzer("b")
        c = CountingAnalyzer("c")

        graph = FlowGraph(entry_node="source")
        graph.add_node(SourceNode(name="source"))
        graph.add_node(PathNode(name="pipe", modules=[a, b, c], parallel=True))
        graph.add_edge("source", "pipe")

        backend = SimpleBackend()
        result = backend.execute(iter(make_frames(3)), graph)

        assert a._process_count == 3
        assert b._process_count == 3
        assert c._process_count == 3
        assert result.frame_count == 3
