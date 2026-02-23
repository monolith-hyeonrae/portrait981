"""Tests for momentscan high-level API."""

import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock, Mock

import numpy as np
import pytest

from visualpath.runner import ProcessResult as VPProcessResult
from visualpath.backends.base import PipelineResult


def _mock_engine(frame_count=0, backend_name="SimpleBackend"):
    engine = Mock()
    engine.execute.return_value = PipelineResult(
        triggers=[], frame_count=frame_count,
    )
    engine.name = backend_name
    return engine


def _mock_video_source():
    """Return (frames_iter, cleanup_fn) for _open_video_source mock."""
    return iter([]), None


class TestRunFunction:
    """Tests for ms.run() function."""

    def test_import(self):
        """Test that run can be imported."""
        import momentscan as ms
        assert hasattr(ms, "run")
        assert hasattr(ms, "Result")

    def test_run_returns_result(self):
        """Test run returns Result."""
        import momentscan as ms

        engine = _mock_engine()

        with patch("visualpath.runner.get_backend", return_value=engine), \
             patch("visualpath.runner._open_video_source", return_value=_mock_video_source()):
            result = ms.run("fake_video.mp4", analyzers=["mock.dummy"])

            assert isinstance(result, ms.Result)
            assert hasattr(result, "highlights")
            assert hasattr(result, "frame_count")
            assert hasattr(result, "duration_sec")

    def test_run_with_output_dir(self):
        """Test run with output directory for highlight export."""
        import momentscan as ms

        with tempfile.TemporaryDirectory() as tmpdir:
            engine = _mock_engine()

            with patch("visualpath.runner.get_backend", return_value=engine), \
                 patch("visualpath.runner._open_video_source", return_value=_mock_video_source()):
                result = ms.run(
                    "fake_video.mp4",
                    output_dir=tmpdir,
                    analyzers=["mock.dummy"],
                )
                assert isinstance(result, ms.Result)

    def test_run_with_specific_analyzers(self):
        """Test run with specific analyzer list."""
        import momentscan as ms

        engine = _mock_engine()

        with patch("visualpath.runner.get_backend", return_value=engine), \
             patch("visualpath.runner._open_video_source", return_value=_mock_video_source()):
            result = ms.run("fake_video.mp4", analyzers=["frame.quality"])
            assert isinstance(result, ms.Result)


class TestConfigureModules:
    """Tests for MomentscanApp.configure_modules()."""

    def test_default_analyzers(self):
        """Test default analyzer list."""
        from momentscan.main import MomentscanApp

        app = MomentscanApp()
        modules = app.configure_modules([])
        names = [m.name for m in modules]
        assert "face.detect" in names
        assert "portrait.score" in names
        assert "hand.gesture" not in names
        assert "face.classify" in names  # auto-injected

    def test_face_classifier_auto_inject(self):
        """Test FaceClassifier is auto-injected when face is used."""
        from momentscan.main import MomentscanApp

        app = MomentscanApp()
        modules = app.configure_modules(["face.detect"])
        names = [m.name for m in modules]
        assert "face.classify" in names

    def test_face_detect_triggers_classifier(self):
        """Test FaceClassifier is auto-injected when face_detect is used."""
        from momentscan.main import MomentscanApp

        app = MomentscanApp()
        modules = app.configure_modules(["face.detect", "face.expression"])
        names = [m.name for m in modules]
        assert "face.classify" in names

    def test_no_classifier_without_face(self):
        """Test FaceClassifier is not injected when face is not used."""
        from momentscan.main import MomentscanApp

        app = MomentscanApp()
        modules = app.configure_modules(["frame.quality", "mock.dummy"])
        names = [m.name for m in modules]
        assert "face.classify" not in names

    def test_no_duplicate_classifier(self):
        """Test FaceClassifier is not duplicated if already included."""
        from momentscan.main import MomentscanApp

        app = MomentscanApp()
        modules = app.configure_modules(["face.detect", "face.classify"])
        names = [m.name for m in modules]
        assert names.count("face.classify") == 1


class TestMomentscanApp:
    """Tests for MomentscanApp class."""

    def test_import(self):
        """MomentscanApp is importable from momentscan."""
        import momentscan as ms
        assert hasattr(ms, "MomentscanApp")

    def test_class_defaults(self):
        """MomentscanApp has correct class defaults."""
        from momentscan.main import MomentscanApp
        assert MomentscanApp.fps == 10
        assert MomentscanApp.backend == "simple"

    def test_configure_modules_auto_inject(self):
        """MomentscanApp auto-injects face.classify."""
        from momentscan.main import MomentscanApp

        app = MomentscanApp()
        resolved = app.configure_modules(["face.detect"])
        names = [m.name for m in resolved]
        assert "face.classify" in names

    def test_configure_modules_no_inject_without_face(self):
        """No face.classify injection when face.detect is absent."""
        from momentscan.main import MomentscanApp

        app = MomentscanApp()
        resolved = app.configure_modules(["mock.dummy"])
        names = [m.name for m in resolved]
        assert "face.classify" not in names

    def test_configure_modules_default_analyzers(self):
        """Empty modules list falls back to default analyzers."""
        from momentscan.main import MomentscanApp

        app = MomentscanApp()
        resolved = app.configure_modules([])
        names = [m.name for m in resolved]
        assert "face.detect" in names
        assert "face.expression" in names
        assert "portrait.score" in names
        assert "hand.gesture" not in names
        assert "face.classify" in names

    def test_after_run_returns_ms_result(self):
        """after_run wraps ProcessResult into ms.Result with batch highlights."""
        from momentscan.main import MomentscanApp, Result

        app = MomentscanApp()
        app._frame_records = []  # simulate empty video
        vp_result = VPProcessResult(
            triggers=[], frame_count=10, duration_sec=1.0,
            actual_backend="SimpleBackend", stats={},
        )
        result = app.after_run(vp_result)
        assert isinstance(result, Result)
        assert result.frame_count == 10
        assert result.highlights == []

    def test_after_run_with_output_dir(self, tmp_path):
        """after_run exports highlight results when output_dir is set."""
        from momentscan.main import MomentscanApp, Result

        app = MomentscanApp(output_dir=str(tmp_path))
        app._frame_records = []  # simulate empty video
        vp_result = VPProcessResult(
            triggers=[], frame_count=5, duration_sec=0.5,
            actual_backend="SimpleBackend", stats={},
        )
        result = app.after_run(vp_result)
        assert isinstance(result, Result)
        # highlight dir created even if empty
        assert (tmp_path / "highlight" / "windows.json").exists()

    def test_run_end_to_end(self):
        """MomentscanApp.run() returns ms.Result."""
        from momentscan.main import MomentscanApp, Result

        engine = _mock_engine(frame_count=3)

        with patch("visualpath.runner.get_backend", return_value=engine), \
             patch("visualpath.runner._open_video_source", return_value=_mock_video_source()):
            app = MomentscanApp(analyzers=["mock.dummy"])
            result = app.run("v.mp4", modules=["mock.dummy"])
            assert isinstance(result, Result)
            assert result.frame_count == 3

    def test_setup_resets_frame_records(self):
        """setup() resets accumulated frame records."""
        from momentscan.main import MomentscanApp

        app = MomentscanApp()
        app._frame_records = ["dummy"]
        app.setup()
        assert app._frame_records == []

    def test_teardown_clears_frame_records(self):
        """teardown() clears frame records."""
        from momentscan.main import MomentscanApp

        app = MomentscanApp()
        app._frame_records = ["dummy"]
        app.teardown()
        assert app._frame_records == []

    def test_on_frame_accumulates_records(self):
        """on_frame() converts results to FrameRecord and accumulates."""
        from momentscan.main import MomentscanApp

        app = MomentscanApp()
        app.setup()

        # Mock frame and results
        frame = Mock()
        frame.frame_id = 1
        frame.t_src_ns = 1_000_000_000

        mock_obs = Mock()
        mock_obs.source = "face.detect"
        mock_obs.signals = {"face_count": 1}
        mock_obs.data = None  # no FaceDetectOutput → record with defaults

        mock_flow_data = Mock()
        mock_flow_data.observations = [mock_obs]

        result = app.on_frame(frame, [mock_flow_data])
        assert result is True
        assert len(app._frame_records) == 1
        assert app._frame_records[0].frame_idx == 1
        app.teardown()

    def test_on_frame_returns_false_when_interrupted(self):
        """on_frame() returns False after SIGINT for graceful shutdown."""
        from momentscan.main import MomentscanApp

        app = MomentscanApp()
        app.setup()

        # Simulate SIGINT
        app._interrupted = True
        result = app.on_frame(Mock(), [])
        assert result is False
        app.teardown()

    def test_teardown_restores_sigint_handler(self):
        """teardown() restores original SIGINT handler."""
        import signal
        from momentscan.main import MomentscanApp

        original = signal.getsignal(signal.SIGINT)
        app = MomentscanApp()
        app.setup()
        # After setup, handler should be changed
        assert signal.getsignal(signal.SIGINT) != original
        app.teardown()
        # After teardown, handler should be restored
        assert signal.getsignal(signal.SIGINT) == original


class TestConfiguration:
    """Tests for configuration constants."""

    def test_default_values(self):
        """Test default configuration values."""
        import momentscan as ms

        assert ms.DEFAULT_FPS == 10


class TestExports:
    """Tests for __all__ exports."""

    def test_all_exports_exist(self):
        """Test all items in __all__ are actually exported."""
        import momentscan as ms

        for name in ms.__all__:
            assert hasattr(ms, name), f"Missing export: {name}"

    def test_core_exports(self):
        """Test core classes are exported."""
        import momentscan as ms

        assert callable(ms.run)
        assert ms.Result is not None
        assert ms.MomentscanApp is not None
