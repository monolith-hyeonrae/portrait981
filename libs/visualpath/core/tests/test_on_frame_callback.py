"""Tests for on_frame callback in SimpleBackend.execute().

Verifies:
- on_frame is called with (Frame, List[FlowData]) after each frame
- Returning False from on_frame stops processing early
- on_frame=None preserves existing behavior
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional

from visualpath.core import Module, Observation
from visualpath.backends.simple import SimpleBackend
from visualpath.flow.graph import FlowGraph
from visualpath.flow.nodes.source import SourceNode
from visualpath.flow.nodes.path import PathNode
from visualpath.flow.node import FlowData


@dataclass
class MockFrame:
    """Mock Frame for testing."""
    frame_id: int
    t_src_ns: int
    data: np.ndarray


class EchoAnalyzer(Module):
    """Simple analyzer that echoes frame data."""

    def __init__(self, name: str = "echo"):
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def process(self, frame, deps=None) -> Optional[Observation]:
        return Observation(
            source=self.name,
            frame_id=frame.frame_id,
            t_ns=frame.t_src_ns,
            signals={"frame_id": frame.frame_id},
        )


def _make_frames(count: int):
    """Create mock frames."""
    return [
        MockFrame(
            frame_id=i,
            t_src_ns=i * 100_000_000,
            data=np.zeros((100, 100, 3), dtype=np.uint8),
        )
        for i in range(count)
    ]


def _make_graph(modules):
    """Build a simple FlowGraph."""
    graph = FlowGraph(entry_node="source")
    graph.add_node(SourceNode(name="source"))
    graph.add_node(PathNode(name="pipeline", modules=modules))
    graph.add_edge("source", "pipeline")
    return graph


class TestOnFrameCallback:
    """Test on_frame callback in SimpleBackend."""

    def test_on_frame_called_for_each_frame(self):
        """on_frame should be called once per frame."""
        analyzer = EchoAnalyzer()
        graph = _make_graph([analyzer])
        frames = _make_frames(5)
        calls = []

        def on_frame(frame, terminal_results):
            calls.append((frame.frame_id, len(terminal_results)))
            return True

        backend = SimpleBackend()
        result = backend.execute(iter(frames), graph, on_frame=on_frame)

        assert result.frame_count == 5
        assert len(calls) == 5
        assert [c[0] for c in calls] == [0, 1, 2, 3, 4]

    def test_on_frame_receives_flow_data(self):
        """on_frame should receive FlowData with observations."""
        analyzer = EchoAnalyzer()
        graph = _make_graph([analyzer])
        frames = _make_frames(1)
        received = []

        def on_frame(frame, terminal_results):
            received.append(terminal_results)
            return True

        backend = SimpleBackend()
        backend.execute(iter(frames), graph, on_frame=on_frame)

        assert len(received) == 1
        assert len(received[0]) > 0
        fd = received[0][0]
        assert isinstance(fd, FlowData)
        assert len(fd.observations) > 0
        assert fd.observations[0].source == "echo"

    def test_on_frame_false_stops_early(self):
        """Returning False from on_frame should stop processing."""
        analyzer = EchoAnalyzer()
        graph = _make_graph([analyzer])
        frames = _make_frames(10)
        call_count = [0]

        def on_frame(frame, terminal_results):
            call_count[0] += 1
            return call_count[0] < 3  # Stop after 3 frames

        backend = SimpleBackend()
        result = backend.execute(iter(frames), graph, on_frame=on_frame)

        assert result.frame_count == 3
        assert call_count[0] == 3

    def test_on_frame_none_is_default(self):
        """on_frame=None should work the same as no callback."""
        analyzer = EchoAnalyzer()
        graph = _make_graph([analyzer])
        frames = _make_frames(5)

        backend = SimpleBackend()
        result = backend.execute(iter(frames), graph, on_frame=None)

        assert result.frame_count == 5

    def test_on_frame_none_default_behavior(self):
        """Not passing on_frame at all should work."""
        analyzer = EchoAnalyzer()
        graph = _make_graph([analyzer])
        frames = _make_frames(3)

        backend = SimpleBackend()
        result = backend.execute(iter(frames), graph)

        assert result.frame_count == 3

    def test_on_frame_receives_frame_object(self):
        """on_frame should receive the actual Frame object."""
        analyzer = EchoAnalyzer()
        graph = _make_graph([analyzer])
        frames = _make_frames(2)
        received_frames = []

        def on_frame(frame, terminal_results):
            received_frames.append(frame)
            return True

        backend = SimpleBackend()
        backend.execute(iter(frames), graph, on_frame=on_frame)

        assert len(received_frames) == 2
        assert received_frames[0].frame_id == 0
        assert received_frames[1].frame_id == 1

    def test_triggers_still_collected_with_on_frame(self):
        """Trigger collection should work alongside on_frame."""
        from visualbase import Trigger

        class TriggerModule(Module):
            @property
            def name(self):
                return "trigger"

            def process(self, frame, deps=None):
                trigger = Trigger.point(
                    event_time_ns=frame.t_src_ns,
                    pre_sec=1.0,
                    post_sec=1.0,
                    label="test",
                )
                return Observation(
                    source=self.name,
                    frame_id=frame.frame_id,
                    t_ns=frame.t_src_ns,
                    signals={"should_trigger": True},
                    metadata={"trigger": trigger},
                )

        graph = _make_graph([TriggerModule()])
        frames = _make_frames(3)

        def on_frame(frame, terminal_results):
            return True

        backend = SimpleBackend()
        result = backend.execute(iter(frames), graph, on_frame=on_frame)

        assert result.frame_count == 3
        assert len(result.triggers) == 3
