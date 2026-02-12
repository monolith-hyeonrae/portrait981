"""Tests for Module batch processing API.

Tests verify:
- Module.process_batch() default implementation (sequential fallback)
- SimpleInterpreter batch dispatch (BATCHING vs non-BATCHING modules)
- SimpleBackend batch_size parameter
- GraphExecutor batch BFS routing
"""

import numpy as np
import pytest
from dataclasses import dataclass
from typing import Dict, List, Optional

from visualpath.core import Module, Observation
from visualpath.core.capabilities import Capability, ModuleCapabilities
from visualpath.backends.base import PipelineResult


# =============================================================================
# Fixtures
# =============================================================================


@dataclass
class MockFrame:
    """Mock Frame for testing."""
    frame_id: int
    t_src_ns: int
    data: np.ndarray


def make_frame(frame_id: int = 0, t_ns: int = 0) -> MockFrame:
    return MockFrame(
        frame_id=frame_id,
        t_src_ns=t_ns,
        data=np.zeros((100, 100, 3), dtype=np.uint8),
    )


def make_frames(count: int, interval_ns: int = 100_000_000) -> List[MockFrame]:
    return [make_frame(frame_id=i, t_ns=i * interval_ns) for i in range(count)]


class SimpleModule(Module):
    """Module without BATCHING capability."""

    def __init__(self, name: str = "simple"):
        self._name = name
        self.process_count = 0

    @property
    def name(self) -> str:
        return self._name

    def process(self, frame, deps=None) -> Optional[Observation]:
        self.process_count += 1
        return Observation(
            source=self.name,
            frame_id=frame.frame_id,
            t_ns=frame.t_src_ns,
            signals={"value": 1.0, "count": self.process_count},
        )


class BatchModule(Module):
    """Module with BATCHING capability and custom process_batch()."""

    def __init__(self, name: str = "batch", max_batch_size: int = 8):
        self._name = name
        self._max_batch_size = max_batch_size
        self.process_count = 0
        self.batch_calls = 0
        self.batch_sizes: List[int] = []

    @property
    def name(self) -> str:
        return self._name

    @property
    def capabilities(self) -> ModuleCapabilities:
        return ModuleCapabilities(
            flags=Capability.BATCHING,
            max_batch_size=self._max_batch_size,
        )

    def process(self, frame, deps=None) -> Optional[Observation]:
        self.process_count += 1
        return Observation(
            source=self.name,
            frame_id=frame.frame_id,
            t_ns=frame.t_src_ns,
            signals={"value": 1.0, "batched": False},
        )

    def process_batch(self, frames, deps_list) -> List[Optional[Observation]]:
        self.batch_calls += 1
        self.batch_sizes.append(len(frames))
        results = []
        for frame, deps in zip(frames, deps_list):
            self.process_count += 1
            results.append(Observation(
                source=self.name,
                frame_id=frame.frame_id,
                t_ns=frame.t_src_ns,
                signals={"value": 1.0, "batched": True},
            ))
        return results


class DepBatchModule(Module):
    """Batch module that depends on another module."""

    def __init__(self, dep_name: str):
        self._dep_name = dep_name
        self.depends = [dep_name]
        self.batch_calls = 0
        self.received_deps: List = []

    @property
    def name(self) -> str:
        return "dep_batch"

    @property
    def capabilities(self) -> ModuleCapabilities:
        return ModuleCapabilities(flags=Capability.BATCHING, max_batch_size=8)

    def process(self, frame, deps=None):
        dep_val = deps.get(self._dep_name).signals["value"] if deps else 0
        return Observation(
            source=self.name,
            frame_id=frame.frame_id,
            t_ns=frame.t_src_ns,
            signals={"value": dep_val + 1.0, "batched": False},
        )

    def process_batch(self, frames, deps_list):
        self.batch_calls += 1
        results = []
        for frame, deps in zip(frames, deps_list):
            self.received_deps.append(deps)
            dep_val = deps.get(self._dep_name).signals["value"] if deps else 0
            results.append(Observation(
                source=self.name,
                frame_id=frame.frame_id,
                t_ns=frame.t_src_ns,
                signals={"value": dep_val + 1.0, "batched": True},
            ))
        return results


# =============================================================================
# Module.process_batch() Default Implementation
# =============================================================================


class TestProcessBatchDefault:
    """Test default process_batch() behavior (sequential fallback)."""

    def test_default_calls_process_sequentially(self):
        """Default process_batch() calls process() for each frame."""
        module = SimpleModule()
        frames = make_frames(4)
        deps_list = [None] * 4

        results = module.process_batch(frames, deps_list)

        assert len(results) == 4
        assert module.process_count == 4
        for i, obs in enumerate(results):
            assert obs is not None
            assert obs.frame_id == i
            assert obs.source == "simple"

    def test_default_with_deps(self):
        """Default process_batch() passes deps correctly."""
        class DependentModule(Module):
            depends = ["upstream"]

            @property
            def name(self):
                return "dependent"

            def process(self, frame, deps=None):
                val = deps["upstream"].signals["x"] if deps else 0
                return Observation(
                    source=self.name,
                    frame_id=frame.frame_id,
                    t_ns=frame.t_src_ns,
                    signals={"result": val * 2},
                )

        module = DependentModule()
        frames = make_frames(3)
        deps_list = [
            {"upstream": Observation(source="upstream", frame_id=i, t_ns=0, signals={"x": float(i)})}
            for i in range(3)
        ]

        results = module.process_batch(frames, deps_list)

        assert len(results) == 3
        assert results[0].signals["result"] == 0.0
        assert results[1].signals["result"] == 2.0
        assert results[2].signals["result"] == 4.0

    def test_empty_batch(self):
        """process_batch() with empty list returns empty list."""
        module = SimpleModule()
        results = module.process_batch([], [])
        assert results == []


# =============================================================================
# Custom process_batch() Override
# =============================================================================


class TestProcessBatchCustom:
    """Test custom process_batch() implementations."""

    def test_batch_module_uses_process_batch(self):
        """BatchModule's process_batch() is called and tracks calls."""
        module = BatchModule()
        frames = make_frames(4)
        deps_list = [None] * 4

        results = module.process_batch(frames, deps_list)

        assert len(results) == 4
        assert module.batch_calls == 1
        assert module.batch_sizes == [4]
        for obs in results:
            assert obs.signals["batched"] is True

    def test_batch_results_match_sequential(self):
        """process_batch() results match process() for stateless modules."""
        module = SimpleModule()
        frames = make_frames(4)
        deps_list = [None] * 4

        # Sequential
        sequential = [module.process(f, d) for f, d in zip(frames, deps_list)]
        module.process_count = 0

        # Batch (uses default sequential fallback)
        batch = module.process_batch(frames, deps_list)

        assert len(batch) == len(sequential)
        for s, b in zip(sequential, batch):
            assert s.source == b.source
            assert s.frame_id == b.frame_id
            assert s.signals["value"] == b.signals["value"]


# =============================================================================
# SimpleInterpreter Batch Dispatch
# =============================================================================


class TestInterpreterBatchDispatch:
    """Test _dispatch_module_batch() routing."""

    def test_dispatch_batch_uses_process_batch_for_batching_module(self):
        """Modules with BATCHING use process_batch()."""
        from visualpath.backends.simple.interpreter import SimpleInterpreter

        interp = SimpleInterpreter()
        module = BatchModule()
        frames = make_frames(4)
        deps_per_frame = [{} for _ in range(4)]

        results = interp._dispatch_module_batch(module, frames, deps_per_frame)

        assert len(results) == 4
        assert module.batch_calls == 1
        for obs in results:
            assert obs.signals["batched"] is True

    def test_dispatch_batch_falls_back_for_non_batching_module(self):
        """Modules without BATCHING use sequential process()."""
        from visualpath.backends.simple.interpreter import SimpleInterpreter

        interp = SimpleInterpreter()
        module = SimpleModule()
        frames = make_frames(4)
        deps_per_frame = [{} for _ in range(4)]

        results = interp._dispatch_module_batch(module, frames, deps_per_frame)

        assert len(results) == 4
        assert module.process_count == 4
        for obs in results:
            assert "batched" not in obs.signals

    def test_interpret_modules_batch_with_deps(self):
        """Batch interpret correctly propagates deps between modules."""
        from visualpath.backends.simple.interpreter import SimpleInterpreter
        from visualpath.flow.nodes.path import PathNode
        from visualpath.flow.node import FlowData

        upstream = BatchModule(name="upstream")
        downstream = DepBatchModule(dep_name="upstream")

        node = PathNode(name="test", modules=[upstream, downstream])
        interp = SimpleInterpreter()

        datas = [
            FlowData(frame=make_frame(i), timestamp_ns=i * 100)
            for i in range(3)
        ]

        outputs = interp.interpret_modules_batch(node, node.spec, datas)

        assert len(outputs) == 3
        # Each output should have observations from both modules
        for i, output_list in enumerate(outputs):
            assert len(output_list) == 1
            obs_map = {o.source: o for o in output_list[0].observations}
            assert "upstream" in obs_map
            assert "dep_batch" in obs_map
            # downstream value = upstream value + 1.0
            assert obs_map["dep_batch"].signals["value"] == 2.0
            assert obs_map["dep_batch"].signals["batched"] is True

        # Both modules should have been called via process_batch
        assert upstream.batch_calls == 1
        assert downstream.batch_calls == 1


# =============================================================================
# SimpleBackend Batch Processing
# =============================================================================


class TestSimpleBackendBatch:
    """Test SimpleBackend with batch_size > 1."""

    def test_batch_size_default_is_one(self):
        """Default batch_size is 1 (backward compatible)."""
        from visualpath.backends.simple import SimpleBackend
        backend = SimpleBackend()
        assert backend._batch_size == 1

    def test_batch_size_configurable(self):
        """batch_size parameter is stored correctly."""
        from visualpath.backends.simple import SimpleBackend
        backend = SimpleBackend(batch_size=8)
        assert backend._batch_size == 8

    def test_batch_size_minimum_one(self):
        """batch_size is clamped to minimum 1."""
        from visualpath.backends.simple import SimpleBackend
        backend = SimpleBackend(batch_size=0)
        assert backend._batch_size == 1

    def test_batch_execute_produces_same_triggers(self):
        """Batch execution produces same triggers as frame-by-frame."""
        from visualpath.backends.simple import SimpleBackend
        from visualpath.flow.graph import FlowGraph
        from visualbase import Trigger

        class TriggerModule(Module):
            depends = ["analyzer"]
            @property
            def name(self):
                return "trigger"
            def process(self, frame, deps=None):
                obs = deps.get("analyzer") if deps else None
                val = obs.signals.get("value", 0) if obs else 0
                if val > 0.5:
                    trigger = Trigger.point(
                        event_time_ns=frame.t_src_ns,
                        pre_sec=1.0, post_sec=1.0,
                        label="test",
                    )
                    return Observation(
                        source=self.name,
                        frame_id=frame.frame_id,
                        t_ns=frame.t_src_ns,
                        signals={"should_trigger": True, "trigger_score": val},
                        metadata={"trigger": trigger},
                    )
                return Observation(
                    source=self.name,
                    frame_id=frame.frame_id,
                    t_ns=frame.t_src_ns,
                    signals={"should_trigger": False},
                )

        class HighValueAnalyzer(Module):
            @property
            def name(self):
                return "analyzer"
            def process(self, frame, deps=None):
                return Observation(
                    source=self.name,
                    frame_id=frame.frame_id,
                    t_ns=frame.t_src_ns,
                    signals={"value": 0.9},
                )

        frames = make_frames(6)

        # Frame-by-frame
        backend_seq = SimpleBackend(batch_size=1)
        graph_seq = FlowGraph.from_modules([HighValueAnalyzer(), TriggerModule()])
        result_seq = backend_seq.execute(iter(frames), graph_seq)

        # Batch
        backend_batch = SimpleBackend(batch_size=3)
        graph_batch = FlowGraph.from_modules([HighValueAnalyzer(), TriggerModule()])
        result_batch = backend_batch.execute(iter(frames), graph_batch)

        assert result_seq.frame_count == result_batch.frame_count == 6
        assert len(result_seq.triggers) == len(result_batch.triggers) == 6

    def test_batch_execute_with_batching_module(self):
        """Batch execution calls process_batch() for BATCHING modules."""
        from visualpath.backends.simple import SimpleBackend
        from visualpath.flow.graph import FlowGraph

        module = BatchModule(name="batch.mod")
        graph = FlowGraph.from_modules([module])
        frames = make_frames(6)

        backend = SimpleBackend(batch_size=4)
        result = backend.execute(iter(frames), graph)

        assert result.frame_count == 6
        assert module.process_count == 6
        # Should have been called via process_batch
        assert module.batch_calls >= 1


# =============================================================================
# GraphExecutor Batch Processing
# =============================================================================


class TestGraphExecutorBatch:
    """Test GraphExecutor.process_batch() with batch BFS."""

    def test_process_batch_single_module(self):
        """process_batch() processes all frames."""
        from visualpath.backends.simple.executor import GraphExecutor
        from visualpath.flow.graph import FlowGraph

        module = SimpleModule(name="test")
        graph = FlowGraph.from_modules([module])
        frames = make_frames(4)

        executor = GraphExecutor(graph, batch_size=4)
        with executor:
            results = executor.process_batch(frames)

        assert len(results) == 4
        for i, result_list in enumerate(results):
            assert len(result_list) >= 1
            obs = result_list[0].observations
            assert any(o.source == "test" and o.frame_id == i for o in obs)

    def test_process_batch_with_deps(self):
        """process_batch() handles module dependencies correctly."""
        from visualpath.backends.simple.executor import GraphExecutor
        from visualpath.flow.graph import FlowGraph

        upstream = BatchModule(name="up")
        downstream = DepBatchModule(dep_name="up")
        graph = FlowGraph.from_modules([upstream, downstream])
        frames = make_frames(3)

        executor = GraphExecutor(graph, batch_size=4)
        with executor:
            results = executor.process_batch(frames)

        assert len(results) == 3
        for result_list in results:
            obs_map = {o.source: o for o in result_list[0].observations}
            assert "up" in obs_map
            assert "dep_batch" in obs_map

    def test_process_batch_falls_back_for_batch_size_one(self):
        """With batch_size=1, process_batch calls process() per frame."""
        from visualpath.backends.simple.executor import GraphExecutor
        from visualpath.flow.graph import FlowGraph

        module = SimpleModule(name="test")
        graph = FlowGraph.from_modules([module])
        frames = make_frames(3)

        executor = GraphExecutor(graph, batch_size=1)
        with executor:
            results = executor.process_batch(frames)

        assert len(results) == 3
        assert module.process_count == 3
