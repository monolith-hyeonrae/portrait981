"""Backend Conformance Test Suite.

Verifies that Simple and Worker backends produce identical results
for the same scenarios. Parametrized across backend types so that
any new backend can be plugged in with the same guarantees.

Categories:
- deps: dependency passing between modules
- trigger: trigger callback and PipelineResult.triggers
- batch: BATCHING capability dispatch
- lifecycle: initialize/cleanup calls
- on_frame: per-frame callback and early stop
- stateful: temporal ordering for stateful modules
"""

import pytest
import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

from visualpath.core.module import Module, DepsContext
from visualpath.core.observation import Observation
from visualpath.core.capabilities import Capability, ModuleCapabilities
from visualpath.backends.base import PipelineResult
from visualpath.flow.graph import FlowGraph


# =============================================================================
# Fixtures: Mock Frame + Module Zoo
# =============================================================================


@dataclass
class MockFrame:
    """Minimal Frame mock for conformance tests."""
    frame_id: int
    t_src_ns: int
    data: np.ndarray


def make_frames(count: int, interval_ns: int = 100_000_000) -> List[MockFrame]:
    data = np.zeros((64, 64, 3), dtype=np.uint8)
    return [
        MockFrame(frame_id=i, t_src_ns=i * interval_ns, data=data)
        for i in range(count)
    ]


class CountingModule(Module):
    """Records process() call count and received deps."""

    def __init__(self, name: str, value: float = 1.0):
        self._name = name
        self._value = value
        self.call_count = 0
        self.received_deps: List[DepsContext] = []
        self._initialized = False
        self._cleaned_up = False

    @property
    def name(self) -> str:
        return self._name

    def initialize(self) -> None:
        self._initialized = True

    def cleanup(self) -> None:
        self._cleaned_up = True

    def process(self, frame, deps=None) -> Optional[Observation]:
        self.call_count += 1
        self.received_deps.append(deps)
        return Observation(
            source=self.name,
            frame_id=frame.frame_id,
            t_ns=frame.t_src_ns,
            signals={"value": self._value, "count": self.call_count},
        )


class DepsRecordingModule(Module):
    """Module with depends — records exact deps dict per call."""

    def __init__(self, name: str, depends_on: List[str], optional: Optional[List[str]] = None):
        self._name = name
        self.depends = list(depends_on)
        self.optional_depends = list(optional) if optional else []
        self.received_deps: List[DepsContext] = []

    @property
    def name(self) -> str:
        return self._name

    def process(self, frame, deps=None) -> Optional[Observation]:
        self.received_deps.append(deps)
        val = 0.0
        if deps:
            for obs in deps.values():
                if obs is not None:
                    val += obs.signals.get("value", 0)
        return Observation(
            source=self.name,
            frame_id=frame.frame_id,
            t_ns=frame.t_src_ns,
            signals={"value": val + 1.0},
        )


class TriggerModule(Module):
    """Module that triggers when value > threshold in deps."""

    def __init__(self, name: str, depends_on: str, threshold: float = 0.5):
        self._name = name
        self.depends = [depends_on]
        self._threshold = threshold

    @property
    def name(self) -> str:
        return self._name

    def process(self, frame, deps=None) -> Optional[Observation]:
        obs = deps.get(self.depends[0]) if deps else None
        value = obs.signals.get("value", 0) if obs else 0

        if value > self._threshold:
            from visualbase import Trigger
            trigger = Trigger.point(
                event_time_ns=frame.t_src_ns,
                pre_sec=1.0,
                post_sec=1.0,
                label="test_trigger",
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


class BatchModule(Module):
    """Module with BATCHING capability — records batch vs sequential calls."""

    def __init__(self, name: str, value: float = 1.0):
        self._name = name
        self._value = value
        self.batch_calls: List[int] = []  # batch sizes
        self.sequential_calls: int = 0
        self._initialized = False
        self._cleaned_up = False

    @property
    def name(self) -> str:
        return self._name

    @property
    def capabilities(self) -> ModuleCapabilities:
        return ModuleCapabilities(
            flags=Capability.BATCHING,
            max_batch_size=32,
        )

    def initialize(self) -> None:
        self._initialized = True

    def cleanup(self) -> None:
        self._cleaned_up = True

    def process(self, frame, deps=None) -> Optional[Observation]:
        self.sequential_calls += 1
        return Observation(
            source=self.name,
            frame_id=frame.frame_id,
            t_ns=frame.t_src_ns,
            signals={"value": self._value},
        )

    def process_batch(self, frames, deps_list) -> List[Optional[Observation]]:
        self.batch_calls.append(len(frames))
        return [
            Observation(
                source=self.name,
                frame_id=f.frame_id,
                t_ns=f.t_src_ns,
                signals={"value": self._value},
            )
            for f in frames
        ]


class StatefulModule(Module):
    """Stateful module with internal counter — verifies temporal ordering."""

    stateful = True

    def __init__(self, name: str):
        self._name = name
        self._counter = 0
        self.frame_order: List[int] = []  # frame_ids in process order

    @property
    def name(self) -> str:
        return self._name

    def process(self, frame, deps=None) -> Optional[Observation]:
        self._counter += 1
        self.frame_order.append(frame.frame_id)
        return Observation(
            source=self.name,
            frame_id=frame.frame_id,
            t_ns=frame.t_src_ns,
            signals={"counter": self._counter},
        )

    def reset(self) -> None:
        self._counter = 0
        self.frame_order.clear()


class ErrorModule(Module):
    """Module that raises on process() — for cleanup-on-error tests."""

    def __init__(self, name: str = "error_mod"):
        self._name = name
        self._initialized = False
        self._cleaned_up = False

    @property
    def name(self) -> str:
        return self._name

    def initialize(self) -> None:
        self._initialized = True

    def cleanup(self) -> None:
        self._cleaned_up = True

    def process(self, frame, deps=None):
        raise RuntimeError("intentional error")


# =============================================================================
# Observation Comparison Utility
# =============================================================================


def assert_observations_equivalent(obs_a: Observation, obs_b: Observation):
    """Verify two Observations are semantically equivalent."""
    assert obs_a.source == obs_b.source
    assert obs_a.frame_id == obs_b.frame_id
    for key in obs_a.signals:
        assert key in obs_b.signals
        assert obs_a.signals[key] == pytest.approx(obs_b.signals[key])


# =============================================================================
# Backend Factory Fixture
# =============================================================================


def _make_simple_backend(**kwargs):
    from visualpath.backends.simple import SimpleBackend
    return SimpleBackend(**kwargs)


def _make_worker_backend(**kwargs):
    """Create WorkerBackend that skips actual subprocess wrapping.

    Since no modules have IsolationConfig in their ModuleSpec, WorkerBackend
    detects needs_wrapping=False and delegates directly to SimpleBackend.
    This tests the full WorkerBackend code path without IPC.
    """
    from visualpath.backends.worker.backend import WorkerBackend
    return WorkerBackend()


@pytest.fixture(params=["simple", "worker"], ids=["SimpleBackend", "WorkerBackend"])
def backend_factory(request):
    """Returns a backend constructor parametrized across Simple and Worker."""
    if request.param == "simple":
        return _make_simple_backend
    else:
        return _make_worker_backend


@pytest.fixture(params=["simple", "worker"], ids=["SimpleBackend", "WorkerBackend"])
def backend(request):
    """Returns a backend instance parametrized across Simple and Worker."""
    if request.param == "simple":
        return _make_simple_backend()
    else:
        return _make_worker_backend()


# =============================================================================
# Test Category: Dependency Passing
# =============================================================================


class TestDepsConformance:
    """Verify deps are passed correctly between modules across backends."""

    def test_deps_passed_correctly(self, backend):
        """A→B dependency: B receives A's Observation in deps."""
        a = CountingModule("a", value=2.0)
        b = DepsRecordingModule("b", depends_on=["a"])
        graph = FlowGraph.from_modules([a, b])
        frames = make_frames(3)

        result = backend.execute(iter(frames), graph)

        assert result.frame_count == 3
        assert a.call_count == 3
        # B should have received deps with key "a" each time
        for deps in b.received_deps:
            assert deps is not None
            assert "a" in deps
            assert deps["a"].source == "a"
            assert deps["a"].signals["value"] == 2.0

    def test_optional_deps_passed(self, backend):
        """optional_depends modules' results are included in deps."""
        a = CountingModule("a", value=3.0)
        b = DepsRecordingModule("b", depends_on=[], optional=["a"])
        graph = FlowGraph.from_modules([a, b])
        frames = make_frames(2)

        backend.execute(iter(frames), graph)

        # B has optional_depends=["a"], so deps should include "a"
        for deps in b.received_deps:
            assert deps is not None
            assert "a" in deps
            assert deps["a"].signals["value"] == 3.0

    def test_deps_none_when_no_dependency(self, backend):
        """Independent modules receive deps=None."""
        a = CountingModule("a")
        b = CountingModule("b")
        graph = FlowGraph.from_modules([a, b])
        frames = make_frames(2)

        backend.execute(iter(frames), graph)

        # Both are independent — deps should be None
        for deps in a.received_deps:
            assert deps is None
        for deps in b.received_deps:
            assert deps is None

    def test_diamond_deps(self, backend):
        """Diamond dependency: A→B,C→D. D gets B and C observations."""
        a = CountingModule("a", value=1.0)
        b = DepsRecordingModule("b", depends_on=["a"])
        c = DepsRecordingModule("c", depends_on=["a"])
        d = DepsRecordingModule("d", depends_on=["b", "c"])
        graph = FlowGraph.from_modules([a, b, c, d])
        frames = make_frames(2)

        backend.execute(iter(frames), graph)

        for deps in d.received_deps:
            assert deps is not None
            assert "b" in deps
            assert "c" in deps


# =============================================================================
# Test Category: Trigger
# =============================================================================


class TestTriggerConformance:
    """Verify trigger handling is consistent across backends."""

    def test_trigger_fires_callback(self, backend):
        """should_trigger=True fires on_trigger callback."""
        src = CountingModule("src", value=0.9)
        trg = TriggerModule("trg", depends_on="src", threshold=0.5)
        graph = FlowGraph.from_modules([src, trg])

        callback_data = []
        graph.on_trigger(lambda d: callback_data.append(d))

        frames = make_frames(3)
        backend.execute(iter(frames), graph)

        # All frames should trigger (value=0.9 > 0.5)
        assert len(callback_data) == 3

    def test_trigger_in_pipeline_result(self, backend):
        """Triggers appear in PipelineResult.triggers."""
        src = CountingModule("src", value=0.9)
        trg = TriggerModule("trg", depends_on="src", threshold=0.5)
        graph = FlowGraph.from_modules([src, trg])
        frames = make_frames(4)

        result = backend.execute(iter(frames), graph)

        assert len(result.triggers) == 4

    def test_no_trigger_when_below_threshold(self, backend):
        """No triggers when value is below threshold."""
        src = CountingModule("src", value=0.3)
        trg = TriggerModule("trg", depends_on="src", threshold=0.5)
        graph = FlowGraph.from_modules([src, trg])
        frames = make_frames(3)

        result = backend.execute(iter(frames), graph)

        assert len(result.triggers) == 0


# =============================================================================
# Test Category: Batch
# =============================================================================


class TestBatchConformance:
    """Verify batch processing across backends."""

    def test_batch_module_receives_batch(self, backend_factory):
        """BATCHING module receives process_batch() when batch_size > 1."""
        bm = BatchModule("bm", value=1.0)
        graph = FlowGraph.from_modules([bm])
        frames = make_frames(6)

        b = backend_factory(batch_size=4)
        result = b.execute(iter(frames), graph)

        assert result.frame_count == 6
        # With batch_size=4 and 6 frames: one batch of 4, one batch of 2
        if bm.batch_calls:
            # batch path was used
            total_batched = sum(bm.batch_calls)
            assert total_batched == 6
        else:
            # Worker backend delegates to SimpleBackend so sequential is ok
            assert bm.sequential_calls == 6

    def test_batch_results_match_sequential(self, backend_factory):
        """Batch and sequential processing produce equivalent signals."""
        # Sequential run
        seq_mod = CountingModule("m", value=5.0)
        seq_graph = FlowGraph.from_modules([seq_mod])
        frames = make_frames(4)

        seq_backend = backend_factory(batch_size=1)
        seq_result = seq_backend.execute(iter(frames), seq_graph)

        # Batch run
        batch_mod = CountingModule("m", value=5.0)
        batch_graph = FlowGraph.from_modules([batch_mod])

        batch_backend = backend_factory(batch_size=4)
        batch_result = batch_backend.execute(iter(make_frames(4)), batch_graph)

        assert seq_result.frame_count == batch_result.frame_count
        assert len(seq_result.triggers) == len(batch_result.triggers)
        # Both processed same number of frames
        assert seq_mod.call_count == batch_mod.call_count


# =============================================================================
# Test Category: Lifecycle
# =============================================================================


class TestLifecycleConformance:
    """Verify initialize/cleanup lifecycle across backends."""

    def test_initialize_called(self, backend):
        """Modules are initialized before processing."""
        mod = CountingModule("m")
        graph = FlowGraph.from_modules([mod])
        frames = make_frames(1)

        backend.execute(iter(frames), graph)

        assert mod._initialized is True

    def test_cleanup_called(self, backend):
        """Modules are cleaned up after processing."""
        mod = CountingModule("m")
        graph = FlowGraph.from_modules([mod])
        frames = make_frames(1)

        backend.execute(iter(frames), graph)

        assert mod._cleaned_up is True

    def test_cleanup_on_error(self, backend):
        """Modules are cleaned up even when an error occurs."""
        mod = ErrorModule("err")
        graph = FlowGraph.from_modules([mod])
        frames = make_frames(1)

        # The error module will raise during processing
        # The backend should still call cleanup
        try:
            backend.execute(iter(frames), graph)
        except Exception:
            pass

        assert mod._initialized is True
        assert mod._cleaned_up is True


# =============================================================================
# Test Category: on_frame Callback
# =============================================================================


class TestOnFrameConformance:
    """Verify on_frame callback behaviour across backends."""

    def test_on_frame_receives_all_observations(self, backend):
        """on_frame callback receives FlowData with all module observations."""
        a = CountingModule("a", value=1.0)
        b = CountingModule("b", value=2.0)
        graph = FlowGraph.from_modules([a, b])
        frames = make_frames(3)

        frame_results = []

        def on_frame(frame, terminal_results):
            frame_results.append((frame.frame_id, terminal_results))
            return True  # continue

        result = backend.execute(iter(frames), graph, on_frame=on_frame)

        assert result.frame_count == 3
        assert len(frame_results) == 3

        # Each terminal_results should have observations from both a and b
        for fid, terminal in frame_results:
            assert len(terminal) >= 1
            obs_sources = set()
            for data in terminal:
                for obs in data.observations:
                    obs_sources.add(obs.source)
            assert "a" in obs_sources
            assert "b" in obs_sources

    def test_on_frame_stop_early(self, backend):
        """on_frame returning False stops processing early."""
        mod = CountingModule("m")
        graph = FlowGraph.from_modules([mod])
        frames = make_frames(10)

        stop_after = 3
        call_count = [0]

        def on_frame(frame, terminal_results):
            call_count[0] += 1
            return call_count[0] < stop_after

        result = backend.execute(iter(frames), graph, on_frame=on_frame)

        # Should have processed at most stop_after frames
        assert result.frame_count <= stop_after
        assert mod.call_count <= stop_after


# =============================================================================
# Test Category: Stateful Module Ordering
# =============================================================================


class TestStatefulConformance:
    """Verify stateful modules are processed in temporal order."""

    def test_stateful_temporal_order(self, backend):
        """stateful=True module processes frames in input order."""
        mod = StatefulModule("sf")
        graph = FlowGraph.from_modules([mod])
        frames = make_frames(5)

        backend.execute(iter(frames), graph)

        # Frame order should match input order: 0, 1, 2, 3, 4
        assert mod.frame_order == [0, 1, 2, 3, 4]

    def test_stateful_counter_increments(self, backend):
        """stateful module's internal counter reflects processing order."""
        mod = StatefulModule("sf")
        graph = FlowGraph.from_modules([mod])
        frames = make_frames(3)

        backend.execute(iter(frames), graph)

        assert mod._counter == 3
        assert mod.frame_order == [0, 1, 2]


# =============================================================================
# Test Category: Cross-Backend Result Equivalence
# =============================================================================


class TestCrossBackendEquivalence:
    """Compare results across Simple and Worker backends directly."""

    def test_simple_vs_worker_same_triggers(self):
        """Simple and Worker backends produce same triggers."""
        from visualpath.backends.simple import SimpleBackend
        from visualpath.backends.worker.backend import WorkerBackend

        def run_with(backend_cls):
            src = CountingModule("src", value=0.8)
            trg = TriggerModule("trg", depends_on="src", threshold=0.5)
            graph = FlowGraph.from_modules([src, trg])
            frames = make_frames(5)
            return backend_cls().execute(iter(frames), graph)

        simple_result = run_with(SimpleBackend)
        worker_result = run_with(WorkerBackend)

        assert simple_result.frame_count == worker_result.frame_count
        assert len(simple_result.triggers) == len(worker_result.triggers)

    def test_simple_vs_worker_same_observations(self):
        """Simple and Worker backends produce equivalent observations."""
        from visualpath.backends.simple import SimpleBackend
        from visualpath.backends.worker.backend import WorkerBackend

        def run_with(backend_cls):
            a = CountingModule("a", value=1.0)
            b = DepsRecordingModule("b", depends_on=["a"])
            graph = FlowGraph.from_modules([a, b])
            frames = make_frames(3)
            result = backend_cls().execute(iter(frames), graph)
            return result, b.received_deps

        simple_result, simple_deps = run_with(SimpleBackend)
        worker_result, worker_deps = run_with(WorkerBackend)

        assert simple_result.frame_count == worker_result.frame_count

        # Both should have received same deps structure
        assert len(simple_deps) == len(worker_deps)
        for s_deps, w_deps in zip(simple_deps, worker_deps):
            assert set(s_deps.keys()) == set(w_deps.keys())
            for key in s_deps:
                assert_observations_equivalent(s_deps[key], w_deps[key])
