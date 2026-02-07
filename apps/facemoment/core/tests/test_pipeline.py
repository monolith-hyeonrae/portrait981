"""Tests for the facemoment pipeline module."""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from dataclasses import dataclass

import numpy as np
import pytest

from facemoment.pipeline import (
    ExtractorConfig,
    FusionConfig,
    PipelineConfig,
    PipelineOrchestrator,
    PipelineStats,
    create_default_config,
)
from visualpath.extractors.base import Observation, FaceObservation
from visualpath.core import IsolationLevel

from helpers import create_test_video


class TestExtractorConfig:
    """Tests for ExtractorConfig dataclass."""

    def test_basic_config(self):
        """Test basic config creation."""
        config = ExtractorConfig(name="face")
        assert config.name == "face"
        assert config.venv_path is None
        assert config.isolation == IsolationLevel.INLINE
        assert config.kwargs == {}

    def test_venv_path_sets_venv_isolation(self):
        """Test that venv_path implies VENV isolation."""
        config = ExtractorConfig(name="face", venv_path="/opt/venv-face")
        assert config.venv_path == "/opt/venv-face"
        assert config.isolation == IsolationLevel.VENV

    def test_explicit_isolation_override(self):
        """Test explicit isolation level."""
        config = ExtractorConfig(
            name="quality",
            isolation=IsolationLevel.THREAD,
        )
        assert config.isolation == IsolationLevel.THREAD

    def test_effective_isolation(self):
        """Test effective_isolation property."""
        config1 = ExtractorConfig(name="face")
        assert config1.effective_isolation == IsolationLevel.INLINE

        config2 = ExtractorConfig(name="face", venv_path="/opt/venv")
        assert config2.effective_isolation == IsolationLevel.VENV

    def test_kwargs_passed(self):
        """Test kwargs are stored."""
        config = ExtractorConfig(
            name="face",
            kwargs={"device": "cuda:0", "batch_size": 4},
        )
        assert config.kwargs == {"device": "cuda:0", "batch_size": 4}


class TestFusionConfig:
    """Tests for FusionConfig dataclass."""

    def test_default_values(self):
        """Test default fusion config."""
        config = FusionConfig()
        assert config.name == "highlight"
        assert config.cooldown_sec == 2.0
        assert config.kwargs == {}

    def test_custom_values(self):
        """Test custom fusion config."""
        config = FusionConfig(
            name="custom",
            cooldown_sec=3.0,
            kwargs={"threshold": 0.8},
        )
        assert config.name == "custom"
        assert config.cooldown_sec == 3.0
        assert config.kwargs == {"threshold": 0.8}


class TestPipelineConfig:
    """Tests for PipelineConfig dataclass."""

    def test_default_config(self):
        """Test default pipeline config."""
        config = PipelineConfig()
        assert config.extractors == []
        assert config.fusion.name == "highlight"
        assert config.clip_output_dir == "./clips"
        assert config.fps == 10

    def test_from_dict(self):
        """Test creating config from dictionary."""
        data = {
            "extractors": [
                {"name": "face", "venv_path": "/opt/venv-face"},
                {"name": "quality", "isolation": "inline"},
            ],
            "fusion": {"name": "highlight", "cooldown_sec": 3.0},
            "clip_output_dir": "/tmp/clips",
            "fps": 15,
        }

        config = PipelineConfig.from_dict(data)

        assert len(config.extractors) == 2
        assert config.extractors[0].name == "face"
        assert config.extractors[0].venv_path == "/opt/venv-face"
        assert config.extractors[1].name == "quality"
        assert config.extractors[1].isolation == IsolationLevel.INLINE
        assert config.fusion.cooldown_sec == 3.0
        assert config.clip_output_dir == "/tmp/clips"
        assert config.fps == 15

    def test_to_dict(self):
        """Test serializing config to dictionary."""
        config = PipelineConfig(
            extractors=[
                ExtractorConfig(name="face", venv_path="/opt/venv-face"),
            ],
            fusion=FusionConfig(cooldown_sec=2.5),
            clip_output_dir="./output",
            fps=20,
        )

        data = config.to_dict()

        assert data["extractors"][0]["name"] == "face"
        assert data["extractors"][0]["venv_path"] == "/opt/venv-face"
        assert data["fusion"]["cooldown_sec"] == 2.5
        assert data["clip_output_dir"] == "./output"
        assert data["fps"] == 20


class TestCreateDefaultConfig:
    """Tests for create_default_config helper."""

    def test_default_inline_mode(self):
        """Test default config with no venv paths."""
        config = create_default_config()

        # Should have face, pose, quality (no gesture without venv)
        assert len(config.extractors) == 3
        names = [e.name for e in config.extractors]
        assert "face" in names
        assert "pose" in names
        assert "quality" in names

        # All should be inline
        for ext in config.extractors:
            assert ext.effective_isolation == IsolationLevel.INLINE

    def test_with_venv_paths(self):
        """Test config with venv paths."""
        config = create_default_config(
            venv_face="/opt/venv-face",
            venv_pose="/opt/venv-pose",
        )

        face_config = next(e for e in config.extractors if e.name == "face")
        pose_config = next(e for e in config.extractors if e.name == "pose")
        quality_config = next(e for e in config.extractors if e.name == "quality")

        assert face_config.effective_isolation == IsolationLevel.VENV
        assert face_config.venv_path == "/opt/venv-face"
        assert pose_config.effective_isolation == IsolationLevel.VENV
        assert quality_config.effective_isolation == IsolationLevel.INLINE

    def test_with_gesture_venv(self):
        """Test gesture extractor is added when venv provided."""
        config = create_default_config(
            venv_gesture="/opt/venv-gesture",
        )

        names = [e.name for e in config.extractors]
        assert "gesture" in names

    def test_without_gesture_venv(self):
        """Test gesture extractor is not added without venv."""
        config = create_default_config()

        names = [e.name for e in config.extractors]
        assert "gesture" not in names


class TestPipelineStats:
    """Tests for PipelineStats dataclass."""

    def test_default_values(self):
        """Test default stats."""
        stats = PipelineStats()
        assert stats.frames_processed == 0
        assert stats.triggers_fired == 0
        assert stats.clips_extracted == 0
        assert stats.processing_time_sec == 0.0
        assert stats.worker_stats == {}

    def test_custom_values(self):
        """Test stats with values."""
        stats = PipelineStats(
            frames_processed=100,
            triggers_fired=5,
            clips_extracted=4,
            processing_time_sec=10.5,
            avg_frame_time_ms=105.0,
            worker_stats={"face": {"frames": 100, "total_ms": 5000.0}},
        )
        assert stats.frames_processed == 100
        assert stats.worker_stats["face"]["frames"] == 100


class TestPipelineOrchestrator:
    """Tests for PipelineOrchestrator."""

    def test_init_requires_configs(self):
        """Test that at least one extractor config is required."""
        with pytest.raises(ValueError, match="At least one extractor"):
            PipelineOrchestrator(extractor_configs=[])

    def test_from_config(self):
        """Test creating orchestrator from PipelineConfig."""
        config = PipelineConfig(
            extractors=[ExtractorConfig(name="dummy")],
            clip_output_dir="/tmp/clips",
        )

        orchestrator = PipelineOrchestrator.from_config(config)
        assert orchestrator.clip_output_dir == Path("/tmp/clips")

    def test_callbacks(self):
        """Test callback setters."""
        orchestrator = PipelineOrchestrator(
            extractor_configs=[ExtractorConfig(name="dummy")],
        )

        on_frame = Mock()
        on_obs = Mock()
        on_trigger = Mock()

        orchestrator.set_on_frame(on_frame)
        orchestrator.set_on_observations(on_obs)
        orchestrator.set_on_trigger(on_trigger)

        assert orchestrator._on_frame is on_frame
        assert orchestrator._on_observations is on_obs
        assert orchestrator._on_trigger is on_trigger

    def test_initial_state(self):
        """Test initial orchestrator state."""
        orchestrator = PipelineOrchestrator(
            extractor_configs=[ExtractorConfig(name="dummy")],
        )

        assert not orchestrator.is_initialized
        assert orchestrator.worker_names == []


class TestPipelineOrchestratorIntegration:
    """Integration tests for PipelineOrchestrator with inline workers."""

    @pytest.fixture
    def test_video(self, tmp_path):
        """Create a test video."""
        video_path = tmp_path / "test.mp4"
        create_test_video(video_path, num_frames=90, fps=30)  # 3 seconds
        return video_path

    @pytest.fixture
    def output_dir(self, tmp_path):
        """Create output directory."""
        output = tmp_path / "clips"
        output.mkdir()
        return output

    def test_run_with_dummy_extractor(self, test_video, output_dir):
        """Test running pipeline with dummy extractor delegates to FlowGraph."""
        from visualpath.backends.base import PipelineResult

        config = PipelineConfig(
            extractors=[ExtractorConfig(name="dummy")],
            fusion=FusionConfig(cooldown_sec=1.0),
            clip_output_dir=str(output_dir),
            fps=10,
            backend="simple",
        )

        orchestrator = PipelineOrchestrator.from_config(config)

        mock_engine = Mock()
        mock_engine.execute.return_value = PipelineResult(triggers=[], frame_count=10)
        mock_engine.name = "SimpleBackend"

        # Orchestrator imports build_graph and _get_backend locally in run()
        with patch("facemoment.main.build_graph") as mock_bg, \
             patch("facemoment.main._get_backend", return_value=mock_engine), \
             patch("facemoment.pipeline.orchestrator.VisualBase") as mock_vb_cls, \
             patch("facemoment.pipeline.orchestrator.FileSource"):
            mock_bg.return_value = Mock()
            mock_vb = Mock()
            mock_vb.get_stream.return_value = iter([])
            mock_vb_cls.return_value = mock_vb

            clips = orchestrator.run(str(test_video), fps=10)

            # Verify stats
            stats = orchestrator.get_stats()
            assert stats.frames_processed == 10
            assert stats.processing_time_sec > 0

    def test_run_stream_deprecated(self, test_video, output_dir):
        """Test stream mode is deprecated and returns empty iterator."""
        config = PipelineConfig(
            extractors=[ExtractorConfig(name="dummy")],
            clip_output_dir=str(output_dir),
        )

        orchestrator = PipelineOrchestrator.from_config(config)

        with pytest.warns(DeprecationWarning):
            frame_count = 0
            for frame, observations, result in orchestrator.run_stream(
                str(test_video), fps=10
            ):
                frame_count += 1

            # run_stream is deprecated and returns empty iterator
            assert frame_count == 0

    def test_get_stats_after_run(self, test_video, output_dir):
        """Test statistics collection."""
        from visualpath.backends.base import PipelineResult

        config = PipelineConfig(
            extractors=[ExtractorConfig(name="dummy")],
            clip_output_dir=str(output_dir),
        )

        orchestrator = PipelineOrchestrator.from_config(config)

        mock_engine = Mock()
        mock_engine.execute.return_value = PipelineResult(triggers=[], frame_count=30)
        mock_engine.name = "SimpleBackend"

        with patch("facemoment.main.build_graph") as mock_bg, \
             patch("facemoment.main._get_backend", return_value=mock_engine), \
             patch("facemoment.pipeline.orchestrator.VisualBase") as mock_vb_cls, \
             patch("facemoment.pipeline.orchestrator.FileSource"):
            mock_bg.return_value = Mock()
            mock_vb = Mock()
            mock_vb.get_stream.return_value = iter([])
            mock_vb_cls.return_value = mock_vb

            orchestrator.run(str(test_video), fps=10)

            stats = orchestrator.get_stats()

            assert stats.frames_processed == 30
            assert stats.processing_time_sec > 0
            assert stats.avg_frame_time_ms > 0


class TestYAMLConfig:
    """Tests for YAML configuration loading."""

    def test_from_yaml_missing_pyyaml(self, tmp_path):
        """Test error when PyYAML is not installed."""
        # This test is hard to implement due to import caching
        # Skip this test - the actual behavior is tested in from_yaml_valid
        pytest.skip("Cannot reliably test missing pyyaml due to import caching")

    def test_from_yaml_file_not_found(self):
        """Test error when YAML file doesn't exist."""
        with pytest.raises(FileNotFoundError):
            PipelineConfig.from_yaml("/nonexistent/path.yaml")

    def test_from_yaml_valid(self, tmp_path):
        """Test loading valid YAML config."""
        yaml_content = """
extractors:
  - name: face
    venv_path: /opt/venv-face
    isolation: venv
  - name: quality
    isolation: inline

fusion:
  name: highlight
  cooldown_sec: 2.5

clip_output_dir: ./output
fps: 15
"""
        yaml_path = tmp_path / "config.yaml"
        yaml_path.write_text(yaml_content)

        try:
            import yaml
            config = PipelineConfig.from_yaml(str(yaml_path))

            assert len(config.extractors) == 2
            assert config.extractors[0].name == "face"
            assert config.extractors[0].venv_path == "/opt/venv-face"
            assert config.fusion.cooldown_sec == 2.5
            assert config.fps == 15
        except ImportError:
            pytest.skip("PyYAML not installed")


class TestOrchestratorDelegation:
    """Tests for PipelineOrchestrator delegating to the unified FlowGraph path."""

    def test_orchestrator_delegates_to_build_graph(self):
        """Test that orchestrator.run() delegates to build_graph."""
        from visualpath.backends.base import PipelineResult

        orchestrator = PipelineOrchestrator(
            extractor_configs=[ExtractorConfig(name="dummy")],
        )

        mock_engine = Mock()
        mock_engine.execute.return_value = PipelineResult(triggers=[], frame_count=5)
        mock_engine.name = "SimpleBackend"

        with patch("facemoment.main.build_graph") as mock_bg, \
             patch("facemoment.main._get_backend", return_value=mock_engine), \
             patch("facemoment.pipeline.orchestrator.VisualBase") as mock_vb_cls, \
             patch("facemoment.pipeline.orchestrator.FileSource"):
            mock_bg.return_value = Mock()
            mock_vb = Mock()
            mock_vb.get_stream.return_value = iter([])
            mock_vb_cls.return_value = mock_vb

            with tempfile.NamedTemporaryFile(suffix=".mp4") as tmp:
                orchestrator.run(tmp.name, fps=10)

            stats = orchestrator.get_stats()
            assert stats.frames_processed == 5
