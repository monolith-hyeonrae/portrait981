"""Tests for facemoment high-level API."""

import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock, Mock

import pytest


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
        from visualpath.backends.base import PipelineResult

        mock_engine = MagicMock()
        mock_engine.execute.return_value = PipelineResult(triggers=[], frame_count=0)
        mock_engine.name = "SimpleBackend"

        with patch("facemoment.main.build_graph") as mock_bg, \
             patch("facemoment.main._get_backend", return_value=mock_engine), \
             patch("facemoment.cli.utils.create_video_stream") as mock_cvs:
            mock_cvs.return_value = (MagicMock(), MagicMock(), iter([]))
            mock_bg.return_value = MagicMock()

            result = fm.run("fake_video.mp4", analyzers=["mock.dummy"])

            assert isinstance(result, fm.Result)
            assert hasattr(result, "triggers")
            assert hasattr(result, "frame_count")
            assert hasattr(result, "duration_sec")
            assert hasattr(result, "clips_extracted")

    def test_run_with_callback(self):
        """Test run with on_trigger callback."""
        import facemoment as fm
        from visualpath.backends.base import PipelineResult

        mock_engine = MagicMock()
        mock_engine.execute.return_value = PipelineResult(triggers=[], frame_count=0)
        mock_engine.name = "SimpleBackend"

        with patch("facemoment.main.build_graph") as mock_bg, \
             patch("facemoment.main._get_backend", return_value=mock_engine), \
             patch("facemoment.cli.utils.create_video_stream") as mock_cvs:
            mock_cvs.return_value = (MagicMock(), MagicMock(), iter([]))
            mock_bg.return_value = MagicMock()

            cb = lambda t: None
            result = fm.run(
                "fake_video.mp4",
                analyzers=["mock.dummy"],
                on_trigger=cb,
            )
            assert isinstance(result, fm.Result)
            # on_trigger should be passed to build_graph
            _, kwargs = mock_bg.call_args
            assert kwargs.get("on_trigger") is cb

    def test_run_with_output_dir(self):
        """Test run with output directory for clips."""
        import facemoment as fm
        from visualpath.backends.base import PipelineResult

        with tempfile.TemporaryDirectory() as tmpdir:
            mock_engine = MagicMock()
            mock_engine.execute.return_value = PipelineResult(triggers=[], frame_count=0)
            mock_engine.name = "SimpleBackend"

            with patch("facemoment.main.build_graph") as mock_bg, \
                 patch("facemoment.main._get_backend", return_value=mock_engine), \
                 patch("facemoment.cli.utils.create_video_stream") as mock_cvs, \
                 patch("facemoment.main._extract_clips", return_value=0) as mock_clips:
                mock_cvs.return_value = (MagicMock(), MagicMock(), iter([]))
                mock_bg.return_value = MagicMock()

                result = fm.run(
                    "fake_video.mp4",
                    output_dir=tmpdir,
                    analyzers=["mock.dummy"],
                )
                assert result.clips_extracted == 0
                mock_clips.assert_called_once()

    def test_run_with_specific_analyzers(self):
        """Test run with specific analyzer list."""
        import facemoment as fm
        from visualpath.backends.base import PipelineResult

        mock_engine = MagicMock()
        mock_engine.execute.return_value = PipelineResult(triggers=[], frame_count=0)
        mock_engine.name = "SimpleBackend"

        with patch("facemoment.main.build_graph") as mock_bg, \
             patch("facemoment.main._get_backend", return_value=mock_engine), \
             patch("facemoment.cli.utils.create_video_stream") as mock_cvs:
            mock_cvs.return_value = (MagicMock(), MagicMock(), iter([]))
            mock_bg.return_value = MagicMock()

            result = fm.run("fake_video.mp4", analyzers=["frame.quality"])
            assert isinstance(result, fm.Result)


class TestBuildModules:
    """Tests for build_modules() function."""

    def test_default_analyzers(self):
        """Test default analyzer list."""
        from facemoment.main import build_modules
        from facemoment.moment_detector.fusion import HighlightFusion

        modules = build_modules()
        names = [m for m in modules if isinstance(m, str)]
        assert "face.detect" in names
        assert "body.pose" in names
        assert "hand.gesture" in names
        assert "face.classify" in names  # auto-injected
        assert any(isinstance(m, HighlightFusion) for m in modules)

    def test_face_classifier_auto_inject(self):
        """Test FaceClassifier is auto-injected when face is used."""
        from facemoment.main import build_modules

        modules = build_modules(["face.detect"])
        names = [m for m in modules if isinstance(m, str)]
        assert "face.classify" in names

    def test_face_detect_triggers_classifier(self):
        """Test FaceClassifier is auto-injected when face_detect is used."""
        from facemoment.main import build_modules

        modules = build_modules(["face.detect", "face.expression"])
        names = [m for m in modules if isinstance(m, str)]
        assert "face.classify" in names

    def test_no_classifier_without_face(self):
        """Test FaceClassifier is not injected when face is not used."""
        from facemoment.main import build_modules

        modules = build_modules(["frame.quality", "mock.dummy"])
        names = [m for m in modules if isinstance(m, str)]
        assert "face.classify" not in names

    def test_no_duplicate_classifier(self):
        """Test FaceClassifier is not duplicated if already included."""
        from facemoment.main import build_modules

        modules = build_modules(["face.detect", "face.classify"])
        names = [m for m in modules if isinstance(m, str)]
        assert names.count("face.classify") == 1

    def test_custom_cooldown(self):
        """Test custom cooldown parameter."""
        from facemoment.main import build_modules
        from facemoment.moment_detector.fusion import HighlightFusion

        modules = build_modules(["mock.dummy"], cooldown=5.0)
        fusion = next(m for m in modules if isinstance(m, HighlightFusion))
        assert fusion._cooldown_ns == int(5.0 * 1e9)


class TestConfiguration:
    """Tests for configuration constants."""

    def test_default_values(self):
        """Test default configuration values."""
        import facemoment as fm

        assert fm.DEFAULT_FPS == 10
        assert fm.DEFAULT_COOLDOWN == 2.0


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
        assert callable(fm.build_modules)
        assert fm.Result is not None
        assert fm.MomentDetector is not None
        assert callable(fm.visualize)
