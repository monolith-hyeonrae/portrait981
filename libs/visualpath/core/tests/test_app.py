"""Tests for vp.App convention layer."""

from dataclasses import dataclass
from typing import Optional
from unittest.mock import patch, Mock, MagicMock

import numpy as np
import pytest

from visualpath.app import App
from visualpath.core import Module, Observation
from visualpath.backends.base import PipelineResult
from visualpath.runner import ProcessResult


# =============================================================================
# Test Fixtures
# =============================================================================


class EchoAnalyzer(Module):
    """Simple analyzer for testing."""

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
            signals={"value": 1.0},
        )


@dataclass
class MockFrame:
    frame_id: int
    t_src_ns: int
    data: np.ndarray


def _make_frames(count):
    return [
        MockFrame(i, i * 100_000_000, np.zeros((10, 10, 3), dtype=np.uint8))
        for i in range(count)
    ]


def _mock_engine(frame_count=5):
    engine = Mock()
    engine.execute.return_value = PipelineResult(
        triggers=[], frame_count=frame_count,
    )
    engine.name = "SimpleBackend"
    return engine


# =============================================================================
# Tests
# =============================================================================


class TestApp:
    """Tests for vp.App class."""

    def test_default_app_run(self):
        """App().run(video, modules=[...]) completes without error."""
        engine = _mock_engine()

        with patch("visualpath.runner.get_backend", return_value=engine), \
             patch("visualpath.runner._open_video_source", return_value=(iter(_make_frames(5)), None)):
            app = App()
            result = app.run("video.mp4", modules=[EchoAnalyzer()])

            assert isinstance(result, ProcessResult)
            assert result.frame_count == 5
            assert result.actual_backend == "SimpleBackend"

    def test_class_attributes(self):
        """Subclass class attributes serve as defaults."""

        class MyApp(App):
            modules = [EchoAnalyzer("custom")]
            fps = 5
            backend = "simple"

        engine = _mock_engine(3)

        with patch("visualpath.runner.get_backend", return_value=engine) as mock_gb, \
             patch("visualpath.runner._open_video_source", return_value=(iter(_make_frames(3)), None)) as mock_open:
            app = MyApp()
            result = app.run("test.mp4")

            # Class defaults used
            mock_gb.assert_called_once_with("simple", batch_size=1)
            mock_open.assert_called_once_with("test.mp4", 5)
            assert result.frame_count == 3

    def test_configure_modules_hook(self):
        """Subclass can override configure_modules."""
        extra = EchoAnalyzer("extra")

        class MyApp(App):
            def configure_modules(self, modules):
                resolved = super().configure_modules(modules)
                resolved.append(extra)
                return resolved

        engine = _mock_engine()

        with patch("visualpath.runner.get_backend", return_value=engine), \
             patch("visualpath.runner._open_video_source", return_value=(iter(_make_frames(1)), None)):
            app = MyApp()
            result = app.run("v.mp4", modules=[EchoAnalyzer()])
            assert isinstance(result, ProcessResult)

    def test_configure_graph_hook(self):
        """Subclass can override configure_graph."""
        from visualpath.flow.graph import FlowGraph
        from visualpath.flow.nodes.source import SourceNode
        from visualpath.flow.nodes.path import PathNode

        class MyApp(App):
            def configure_graph(self, modules, *, isolation=None):
                graph = FlowGraph(entry_node="src")
                graph.add_node(SourceNode(name="src"))
                graph.add_node(PathNode(name="pipe", modules=modules))
                graph.add_edge("src", "pipe")
                return graph

        engine = _mock_engine()

        with patch("visualpath.runner.get_backend", return_value=engine), \
             patch("visualpath.runner._open_video_source", return_value=(iter(_make_frames(1)), None)):
            app = MyApp()
            result = app.run("v.mp4", modules=[EchoAnalyzer()])
            assert isinstance(result, ProcessResult)

    def test_on_frame_hook(self):
        """Subclass on_frame hook is called; False stops processing."""
        calls = []

        class MyApp(App):
            def on_frame(self, frame, results):
                calls.append(frame.frame_id)
                return len(calls) < 2  # stop after 2

        frames = _make_frames(5)
        engine = Mock()
        # Simulate execute calling on_frame
        def fake_execute(frames_iter, graph, on_frame=None):
            count = 0
            for f in frames_iter:
                count += 1
                if on_frame and not on_frame(f, []):
                    break
            return PipelineResult(triggers=[], frame_count=count)

        engine.execute.side_effect = fake_execute
        engine.name = "SimpleBackend"

        with patch("visualpath.runner.get_backend", return_value=engine), \
             patch("visualpath.runner._open_video_source", return_value=(iter(frames), None)):
            app = MyApp()
            result = app.run("v.mp4", modules=[EchoAnalyzer()])
            assert len(calls) == 2

    def test_on_trigger_hook(self):
        """Subclass on_trigger hook is registered on graph."""
        triggers_received = []

        class MyApp(App):
            def on_trigger(self, trigger):
                triggers_received.append(trigger)

        engine = _mock_engine()

        with patch("visualpath.runner.get_backend", return_value=engine), \
             patch("visualpath.runner._open_video_source", return_value=(iter(_make_frames(1)), None)):
            app = MyApp()
            result = app.run("v.mp4", modules=[EchoAnalyzer()])
            # Just verify it doesn't error — trigger adapter was registered on graph
            assert isinstance(result, ProcessResult)

    def test_after_run_hook(self):
        """Subclass can transform result in after_run."""

        @dataclass
        class CustomResult:
            count: int

        class MyApp(App):
            def after_run(self, result):
                return CustomResult(count=result.frame_count)

        engine = _mock_engine(7)

        with patch("visualpath.runner.get_backend", return_value=engine), \
             patch("visualpath.runner._open_video_source", return_value=(iter(_make_frames(7)), None)):
            app = MyApp()
            result = app.run("v.mp4", modules=[EchoAnalyzer()])
            assert isinstance(result, CustomResult)
            assert result.count == 7

    def test_run_overrides_defaults(self):
        """Explicit run() args override class defaults."""

        class MyApp(App):
            fps = 10
            backend = "pathway"

        engine = _mock_engine()

        with patch("visualpath.runner.get_backend", return_value=engine) as mock_gb, \
             patch("visualpath.runner._open_video_source", return_value=(iter(_make_frames(1)), None)) as mock_open:
            app = MyApp()
            app.run("v.mp4", modules=[EchoAnalyzer()], fps=5, backend="simple")

            mock_gb.assert_called_once_with("simple", batch_size=1)
            mock_open.assert_called_once_with("v.mp4", 5)

    def test_explicit_callback_and_hook_merge(self):
        """Both on_frame hook and explicit callback are called."""
        hook_calls = []
        cb_calls = []

        class MyApp(App):
            def on_frame(self, frame, results):
                hook_calls.append(frame.frame_id)
                return True

        def explicit_cb(frame, results):
            cb_calls.append(frame.frame_id)
            return True

        frames = _make_frames(3)
        engine = Mock()

        def fake_execute(frames_iter, graph, on_frame=None):
            count = 0
            for f in frames_iter:
                count += 1
                if on_frame:
                    on_frame(f, [])
            return PipelineResult(triggers=[], frame_count=count)

        engine.execute.side_effect = fake_execute
        engine.name = "SimpleBackend"

        with patch("visualpath.runner.get_backend", return_value=engine), \
             patch("visualpath.runner._open_video_source", return_value=(iter(frames), None)):
            app = MyApp()
            app.run("v.mp4", modules=[EchoAnalyzer()], on_frame=explicit_cb)

            assert hook_calls == [0, 1, 2]
            assert cb_calls == [0, 1, 2]

    def test_video_attribute(self):
        """self.video is available in hooks."""
        captured = {}

        class MyApp(App):
            def after_run(self, result):
                captured["video"] = self.video
                return result

        engine = _mock_engine()

        with patch("visualpath.runner.get_backend", return_value=engine), \
             patch("visualpath.runner._open_video_source", return_value=(iter(_make_frames(1)), None)):
            app = MyApp()
            app.run("my_video.mp4", modules=[EchoAnalyzer()])
            assert captured["video"] == "my_video.mp4"

    def test_app_importable(self):
        """vp.App is accessible from the visualpath namespace."""
        import visualpath as vp
        assert hasattr(vp, "App")
        assert vp.App is App

    def test_duration_sec_computed(self):
        """duration_sec is computed from frame_count / fps."""
        engine = _mock_engine(frame_count=50)

        with patch("visualpath.runner.get_backend", return_value=engine), \
             patch("visualpath.runner._open_video_source", return_value=(iter(_make_frames(50)), None)):
            app = App()
            result = app.run("v.mp4", modules=[EchoAnalyzer()], fps=10)
            assert result.duration_sec == 5.0

    def test_zero_frames_duration(self):
        """duration_sec is 0.0 when no frames processed."""
        engine = _mock_engine(frame_count=0)

        with patch("visualpath.runner.get_backend", return_value=engine), \
             patch("visualpath.runner._open_video_source", return_value=(iter([]), None)):
            app = App()
            result = app.run("v.mp4", modules=[EchoAnalyzer()])
            assert result.duration_sec == 0.0

    def test_cleanup_called_on_success(self):
        """Cleanup function is called after successful execution."""
        cleanup = Mock()
        engine = _mock_engine()

        with patch("visualpath.runner.get_backend", return_value=engine), \
             patch("visualpath.runner._open_video_source", return_value=(iter(_make_frames(1)), cleanup)):
            app = App()
            app.run("v.mp4", modules=[EchoAnalyzer()])
            cleanup.assert_called_once()

    def test_cleanup_called_on_error(self):
        """Cleanup function is called even when execution fails."""
        cleanup = Mock()
        engine = Mock()
        engine.execute.side_effect = RuntimeError("boom")

        with patch("visualpath.runner.get_backend", return_value=engine), \
             patch("visualpath.runner._open_video_source", return_value=(iter(_make_frames(1)), cleanup)):
            app = App()
            with pytest.raises(RuntimeError, match="boom"):
                app.run("v.mp4", modules=[EchoAnalyzer()])
            cleanup.assert_called_once()


class TestAppLifecycle:
    """Tests for setup/teardown lifecycle hooks."""

    def test_setup_hook_called(self):
        """setup() is called with self.video accessible."""
        captured = {}

        class MyApp(App):
            def setup(self):
                captured["video"] = self.video

        engine = _mock_engine()

        with patch("visualpath.runner.get_backend", return_value=engine), \
             patch("visualpath.runner._open_video_source", return_value=(iter(_make_frames(1)), None)):
            app = MyApp()
            app.run("my_video.mp4", modules=[EchoAnalyzer()])
            assert captured["video"] == "my_video.mp4"

    def test_teardown_hook_called(self):
        """teardown() is called on successful execution."""
        teardown_called = []

        class MyApp(App):
            def teardown(self):
                teardown_called.append(True)

        engine = _mock_engine()

        with patch("visualpath.runner.get_backend", return_value=engine), \
             patch("visualpath.runner._open_video_source", return_value=(iter(_make_frames(1)), None)):
            app = MyApp()
            app.run("v.mp4", modules=[EchoAnalyzer()])
            assert len(teardown_called) == 1

    def test_teardown_called_on_error(self):
        """teardown() is called even when execution raises."""
        teardown_called = []

        class MyApp(App):
            def teardown(self):
                teardown_called.append(True)

        engine = Mock()
        engine.execute.side_effect = RuntimeError("boom")

        with patch("visualpath.runner.get_backend", return_value=engine), \
             patch("visualpath.runner._open_video_source", return_value=(iter(_make_frames(1)), None)):
            app = MyApp()
            with pytest.raises(RuntimeError, match="boom"):
                app.run("v.mp4", modules=[EchoAnalyzer()])
            assert len(teardown_called) == 1

    def test_teardown_called_when_after_run_fails(self):
        """teardown() is called even when after_run raises."""
        teardown_called = []

        class MyApp(App):
            def after_run(self, result):
                raise ValueError("after_run failed")

            def teardown(self):
                teardown_called.append(True)

        engine = _mock_engine()

        with patch("visualpath.runner.get_backend", return_value=engine), \
             patch("visualpath.runner._open_video_source", return_value=(iter(_make_frames(1)), None)):
            app = MyApp()
            with pytest.raises(ValueError, match="after_run failed"):
                app.run("v.mp4", modules=[EchoAnalyzer()])
            assert len(teardown_called) == 1

    def test_lifecycle_order(self):
        """Lifecycle order: setup → configure_modules → configure_graph → after_run → teardown."""
        order = []

        class MyApp(App):
            def setup(self):
                order.append("setup")

            def configure_modules(self, modules):
                order.append("configure_modules")
                return super().configure_modules(modules)

            def configure_graph(self, modules, *, isolation=None):
                order.append("configure_graph")
                return super().configure_graph(modules, isolation=isolation)

            def after_run(self, result):
                order.append("after_run")
                return result

            def teardown(self):
                order.append("teardown")

        engine = _mock_engine()

        with patch("visualpath.runner.get_backend", return_value=engine), \
             patch("visualpath.runner._open_video_source", return_value=(iter(_make_frames(1)), None)):
            app = MyApp()
            app.run("v.mp4", modules=[EchoAnalyzer()])
            assert order == ["setup", "configure_modules", "configure_graph", "after_run", "teardown"]

    def test_default_setup_teardown_are_noop(self):
        """Default App with no overrides works fine (setup/teardown are no-ops)."""
        engine = _mock_engine()

        with patch("visualpath.runner.get_backend", return_value=engine), \
             patch("visualpath.runner._open_video_source", return_value=(iter(_make_frames(1)), None)):
            app = App()
            result = app.run("v.mp4", modules=[EchoAnalyzer()])
            assert isinstance(result, ProcessResult)
