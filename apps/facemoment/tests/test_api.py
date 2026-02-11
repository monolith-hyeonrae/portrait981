"""Tests for facemoment high-level API."""

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
    """Tests for fm.run() function."""

    def test_import(self):
        """Test that run can be imported."""
        import facemoment as fm
        assert hasattr(fm, "run")
        assert hasattr(fm, "Result")

    def test_run_returns_result(self):
        """Test run returns Result."""
        import facemoment as fm

        engine = _mock_engine()

        with patch("visualpath.runner.get_backend", return_value=engine), \
             patch("visualpath.runner._open_video_source", return_value=_mock_video_source()):
            result = fm.run("fake_video.mp4", analyzers=["mock.dummy"])

            assert isinstance(result, fm.Result)
            assert hasattr(result, "triggers")
            assert hasattr(result, "frame_count")
            assert hasattr(result, "duration_sec")
            assert hasattr(result, "clips_extracted")

    def test_run_with_callback(self):
        """Test run with on_trigger callback."""
        import facemoment as fm

        engine = _mock_engine()

        with patch("visualpath.runner.get_backend", return_value=engine), \
             patch("visualpath.runner._open_video_source", return_value=_mock_video_source()):
            cb = lambda t: None
            result = fm.run(
                "fake_video.mp4",
                analyzers=["mock.dummy"],
                on_trigger=cb,
            )
            assert isinstance(result, fm.Result)

    def test_run_with_output_dir(self):
        """Test run with output directory for clips."""
        import facemoment as fm

        with tempfile.TemporaryDirectory() as tmpdir:
            engine = _mock_engine()
            mock_vb = MagicMock()

            with patch("visualpath.runner.get_backend", return_value=engine), \
                 patch("visualpath.runner._open_video_source", return_value=_mock_video_source()), \
                 patch("visualbase.VisualBase", return_value=mock_vb), \
                 patch("visualbase.FileSource"):

                result = fm.run(
                    "fake_video.mp4",
                    output_dir=tmpdir,
                    analyzers=["mock.dummy"],
                )
                assert result.clips_extracted == 0
                mock_vb.connect.assert_called_once()
                mock_vb.disconnect.assert_called_once()

    def test_run_with_specific_analyzers(self):
        """Test run with specific analyzer list."""
        import facemoment as fm

        engine = _mock_engine()

        with patch("visualpath.runner.get_backend", return_value=engine), \
             patch("visualpath.runner._open_video_source", return_value=_mock_video_source()):
            result = fm.run("fake_video.mp4", analyzers=["frame.quality"])
            assert isinstance(result, fm.Result)


class TestConfigureModules:
    """Tests for FacemomentApp.configure_modules()."""

    def test_default_analyzers(self):
        """Test default analyzer list."""
        from facemoment.main import FacemomentApp
        from facemoment.algorithm.analyzers.highlight import HighlightFusion

        app = FacemomentApp()
        modules = app.configure_modules([])
        names = [m.name for m in modules if not isinstance(m, HighlightFusion)]
        assert "face.detect" in names
        assert "body.pose" in names
        assert "hand.gesture" in names
        assert "face.classify" in names  # auto-injected
        assert any(isinstance(m, HighlightFusion) for m in modules)

    def test_face_classifier_auto_inject(self):
        """Test FaceClassifier is auto-injected when face is used."""
        from facemoment.main import FacemomentApp

        app = FacemomentApp()
        modules = app.configure_modules(["face.detect"])
        names = [m.name for m in modules]
        assert "face.classify" in names

    def test_face_detect_triggers_classifier(self):
        """Test FaceClassifier is auto-injected when face_detect is used."""
        from facemoment.main import FacemomentApp

        app = FacemomentApp()
        modules = app.configure_modules(["face.detect", "face.expression"])
        names = [m.name for m in modules]
        assert "face.classify" in names

    def test_no_classifier_without_face(self):
        """Test FaceClassifier is not injected when face is not used."""
        from facemoment.main import FacemomentApp

        app = FacemomentApp()
        modules = app.configure_modules(["frame.quality", "mock.dummy"])
        names = [m.name for m in modules]
        assert "face.classify" not in names

    def test_no_duplicate_classifier(self):
        """Test FaceClassifier is not duplicated if already included."""
        from facemoment.main import FacemomentApp

        app = FacemomentApp()
        modules = app.configure_modules(["face.detect", "face.classify"])
        names = [m.name for m in modules]
        assert names.count("face.classify") == 1

    def test_custom_cooldown(self):
        """Test custom cooldown parameter."""
        from facemoment.main import FacemomentApp
        from facemoment.algorithm.analyzers.highlight import HighlightFusion

        app = FacemomentApp(cooldown=5.0)
        modules = app.configure_modules(["mock.dummy"])
        fusion = next(m for m in modules if isinstance(m, HighlightFusion))
        assert fusion._cooldown_ns == int(5.0 * 1e9)


class TestFacemomentApp:
    """Tests for FacemomentApp class."""

    def test_import(self):
        """FacemomentApp is importable from facemoment."""
        import facemoment as fm
        assert hasattr(fm, "FacemomentApp")

    def test_class_defaults(self):
        """FacemomentApp has correct class defaults."""
        from facemoment.main import FacemomentApp
        assert FacemomentApp.fps == 10
        assert FacemomentApp.backend == "pathway"

    def test_configure_modules_auto_inject(self):
        """FacemomentApp auto-injects face.classify."""
        from facemoment.main import FacemomentApp
        from facemoment.algorithm.analyzers.highlight import HighlightFusion

        app = FacemomentApp()
        resolved = app.configure_modules(["face.detect"])
        names = [m.name for m in resolved]
        assert "face.classify" in names
        assert any(isinstance(m, HighlightFusion) for m in resolved)

    def test_configure_modules_no_inject_without_face(self):
        """No face.classify injection when face.detect is absent."""
        from facemoment.main import FacemomentApp

        app = FacemomentApp()
        resolved = app.configure_modules(["mock.dummy"])
        names = [m.name for m in resolved]
        assert "face.classify" not in names

    def test_configure_modules_default_analyzers(self):
        """Empty modules list falls back to default 4 analyzers."""
        from facemoment.main import FacemomentApp
        from facemoment.algorithm.analyzers.highlight import HighlightFusion

        app = FacemomentApp()
        resolved = app.configure_modules([])
        names = [m.name for m in resolved if not isinstance(m, HighlightFusion)]
        assert "face.detect" in names
        assert "face.expression" in names
        assert "body.pose" in names
        assert "hand.gesture" in names
        assert "face.classify" in names

    def test_after_run_returns_fm_result(self):
        """after_run wraps ProcessResult into fm.Result."""
        from facemoment.main import FacemomentApp, Result

        app = FacemomentApp()
        app.video = "test.mp4"
        vp_result = VPProcessResult(
            triggers=[], frame_count=10, duration_sec=1.0,
            actual_backend="SimpleBackend", stats={},
        )
        result = app.after_run(vp_result)
        assert isinstance(result, Result)
        assert result.frame_count == 10
        assert result.clips_extracted == 0

    def test_after_run_with_output_dir(self):
        """after_run uses _clips_extracted count."""
        from facemoment.main import FacemomentApp, Result

        app = FacemomentApp(output_dir="/tmp/clips")
        app.video = "test.mp4"
        app._clips_extracted = 3
        vp_result = VPProcessResult(
            triggers=[], frame_count=5, duration_sec=0.5,
            actual_backend="SimpleBackend", stats={},
        )
        result = app.after_run(vp_result)
        assert result.clips_extracted == 3

    def test_custom_cooldown(self):
        """Custom cooldown is passed to HighlightFusion."""
        from facemoment.main import FacemomentApp
        from facemoment.algorithm.analyzers.highlight import HighlightFusion

        app = FacemomentApp(cooldown=5.0)
        resolved = app.configure_modules(["mock.dummy"])
        fusion = next(m for m in resolved if isinstance(m, HighlightFusion))
        assert fusion._cooldown_ns == int(5.0 * 1e9)

    def test_run_end_to_end(self):
        """FacemomentApp.run() returns fm.Result."""
        from facemoment.main import FacemomentApp, Result

        engine = _mock_engine(frame_count=3)

        with patch("visualpath.runner.get_backend", return_value=engine), \
             patch("visualpath.runner._open_video_source", return_value=_mock_video_source()):
            app = FacemomentApp(analyzers=["mock.dummy"])
            result = app.run("v.mp4", modules=["mock.dummy"])
            assert isinstance(result, Result)
            assert result.frame_count == 3


class TestConfiguration:
    """Tests for configuration constants."""

    def test_default_values(self):
        """Test default configuration values."""
        import facemoment as fm

        assert fm.DEFAULT_FPS == 10
        assert fm.DEFAULT_COOLDOWN == 2.0


class TestFacemomentAppLifecycle:
    """Tests for FacemomentApp setup/teardown lifecycle."""

    def test_setup_creates_visualbase(self):
        """setup() creates VisualBase when output_dir is set."""
        from facemoment.main import FacemomentApp

        mock_vb = MagicMock()
        engine = _mock_engine()

        with patch("visualpath.runner.get_backend", return_value=engine), \
             patch("visualpath.runner._open_video_source", return_value=_mock_video_source()), \
             patch("visualbase.VisualBase", return_value=mock_vb) as mock_cls, \
             patch("visualbase.FileSource") as mock_fs:
            app = FacemomentApp(output_dir="/tmp/clips", analyzers=["mock.dummy"])
            app.run("video.mp4", modules=["mock.dummy"])

            mock_cls.assert_called_once_with(clip_output_dir=Path("/tmp/clips"))
            mock_vb.connect.assert_called_once()
            mock_vb.disconnect.assert_called_once()

    def test_teardown_disconnects(self):
        """teardown() disconnects VisualBase."""
        from facemoment.main import FacemomentApp

        mock_vb = MagicMock()
        engine = Mock()
        engine.execute.side_effect = RuntimeError("boom")

        with patch("visualpath.runner.get_backend", return_value=engine), \
             patch("visualpath.runner._open_video_source", return_value=_mock_video_source()), \
             patch("visualbase.VisualBase", return_value=mock_vb), \
             patch("visualbase.FileSource"):
            app = FacemomentApp(output_dir="/tmp/clips", analyzers=["mock.dummy"])
            with pytest.raises(RuntimeError, match="boom"):
                app.run("video.mp4", modules=["mock.dummy"])
            mock_vb.disconnect.assert_called_once()

    def test_on_trigger_extracts_clip(self):
        """on_trigger() calls vb.trigger() and increments clip count."""
        from facemoment.main import FacemomentApp

        mock_vb = MagicMock()
        mock_trigger_result = MagicMock()
        mock_trigger_result.success = True
        mock_vb.trigger.return_value = mock_trigger_result

        app = FacemomentApp(output_dir="/tmp/clips")
        app._vb = mock_vb
        app._clips_extracted = 0

        trigger = MagicMock()
        app.on_trigger(trigger)

        mock_vb.trigger.assert_called_once_with(trigger)
        assert app._clips_extracted == 1


class TestExports:
    """Tests for __all__ exports."""

    def test_all_exports_exist(self):
        """Test all items in __all__ are actually exported."""
        import facemoment as fm

        for name in fm.__all__:
            assert hasattr(fm, name), f"Missing export: {name}"

    def test_core_exports(self):
        """Test core classes are exported."""
        import facemoment as fm

        assert callable(fm.run)
        assert fm.Result is not None
        assert fm.FacemomentApp is not None
