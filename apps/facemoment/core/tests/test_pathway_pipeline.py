"""Tests for the facemoment Pathway pipeline integration (Phase 17)."""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import numpy as np
import pytest

from facemoment.pipeline.pathway_pipeline import (
    FacemomentPipeline,
    PATHWAY_AVAILABLE,
    _CUDA_GROUPS,
)
from visualpath.analyzers.base import Observation, FaceObservation

from helpers import create_test_video, create_mock_frame


class TestFacemomentPipeline:
    """Tests for FacemomentPipeline class."""

    def test_init_default_analyzers(self):
        """Test default analyzer configuration."""
        pipeline = FacemomentPipeline()
        assert pipeline._analyzer_names == ["face", "pose", "gesture"]

    def test_init_custom_analyzers(self):
        """Test custom analyzer configuration."""
        pipeline = FacemomentPipeline(analyzers=["face", "quality"])
        assert pipeline._analyzer_names == ["face", "quality"]

    def test_init_fusion_config(self):
        """Test fusion configuration."""
        pipeline = FacemomentPipeline(
            fusion_config={"cooldown_sec": 3.0, "main_only": False}
        )
        assert pipeline._fusion_config["cooldown_sec"] == 3.0
        assert pipeline._fusion_config["main_only"] is False

    def test_fusion_config_used_in_run(self):
        """Test fusion config is passed through to build_modules."""
        from facemoment.main import build_modules
        from facemoment.moment_detector.fusion import HighlightFusion

        modules = build_modules(["dummy"], cooldown=5.0)
        fusion = next(m for m in modules if isinstance(m, HighlightFusion))
        assert fusion._cooldown_ns == int(5.0 * 1e9)


class TestFacemomentPipelineExecution:
    """Tests for FacemomentPipeline execution via FlowGraph delegation."""

    def test_run_delegates_to_flowgraph(self):
        """Test that run delegates to the unified FlowGraph path."""
        from visualpath.backends.base import PipelineResult

        pipeline = FacemomentPipeline(analyzers=["dummy"])

        mock_engine = Mock()
        mock_engine.execute.return_value = PipelineResult(triggers=[], frame_count=5)
        mock_engine.name = "SimpleBackend"

        with patch("facemoment.main.build_graph") as mock_bg, \
             patch("facemoment.main._get_backend", return_value=mock_engine):
            mock_bg.return_value = Mock()

            frames = [create_mock_frame(i, i * 100000) for i in range(5)]
            triggers = pipeline.run(frames)

            assert isinstance(triggers, list)
            mock_bg.assert_called_once()
            mock_engine.execute.assert_called_once()

    def test_on_trigger_callback(self):
        """Test on_trigger callback is passed through."""
        from visualpath.backends.base import PipelineResult

        pipeline = FacemomentPipeline(analyzers=["dummy"])

        mock_engine = Mock()
        mock_engine.execute.return_value = PipelineResult(triggers=[], frame_count=0)
        mock_engine.name = "SimpleBackend"

        cb = Mock()

        with patch("facemoment.main.build_graph") as mock_bg, \
             patch("facemoment.main._get_backend", return_value=mock_engine):
            mock_bg.return_value = Mock()

            frames = [create_mock_frame(0, 0)]
            pipeline.run(frames, on_trigger=cb)

            _, kwargs = mock_bg.call_args
            assert kwargs.get("on_trigger") is cb


class TestPathwayAvailability:
    """Tests for Pathway availability check."""

    def test_pathway_available_flag(self):
        """Test that PATHWAY_AVAILABLE reflects actual availability."""
        # Just verify the flag exists and is a boolean
        assert isinstance(PATHWAY_AVAILABLE, bool)

    def test_run_with_pathway_unavailable(self):
        """Test that run works when Pathway is not available."""
        from visualpath.backends.base import PipelineResult

        with patch("facemoment.pipeline.pathway_pipeline.PATHWAY_AVAILABLE", False):
            pipeline = FacemomentPipeline(analyzers=["dummy"])

            mock_engine = Mock()
            mock_engine.execute.return_value = PipelineResult(triggers=[], frame_count=0)
            mock_engine.name = "SimpleBackend"

            with patch("facemoment.main.build_graph") as mock_bg, \
                 patch("facemoment.main._get_backend", return_value=mock_engine):
                mock_bg.return_value = Mock()

                frames = [create_mock_frame(0, 0)]
                triggers = pipeline.run(frames)
                assert isinstance(triggers, list)


class TestHighlevelAPIBackend:
    """Tests for high-level API backend parameter."""

    def test_run_accepts_backend_parameter(self):
        """Test that fm.run() accepts backend parameter."""
        from facemoment.main import run

        # Just verify the function signature accepts backend
        import inspect
        sig = inspect.signature(run)
        assert "backend" in sig.parameters

    def test_main_run_uses_flowgraph_and_backend(self):
        """Test fm.run creates FlowGraph via build_graph and uses backend.execute."""
        from visualpath.backends.base import PipelineResult

        mock_engine = Mock()
        mock_engine.execute.return_value = PipelineResult(triggers=[], frame_count=2)
        mock_engine.name = "SimpleBackend"

        mock_stream = [Mock(), Mock()]
        mock_vb = Mock()

        with patch("facemoment.main.build_graph") as mock_bg, \
             patch("facemoment.main._get_backend", return_value=mock_engine), \
             patch("facemoment.cli.utils.create_video_stream", return_value=(mock_vb, Mock(), mock_stream)):
            mock_bg.return_value = Mock()

            from facemoment.main import run
            result = run("fake.mp4", analyzers=["dummy"], fps=5, cooldown=1.5)

            # build_graph was called
            mock_bg.assert_called_once()
            # backend.execute was called
            mock_engine.execute.assert_called_once()
            assert result.frame_count == 2
            assert result.actual_backend == "SimpleBackend"

    def test_main_run_passes_on_trigger(self):
        """Test fm.run passes on_trigger to build_graph."""
        from visualpath.backends.base import PipelineResult

        mock_engine = Mock()
        mock_engine.execute.return_value = PipelineResult(triggers=[], frame_count=0)
        mock_engine.name = "SimpleBackend"
        cb = lambda t: None

        with patch("facemoment.main.build_graph") as mock_bg, \
             patch("facemoment.main._get_backend", return_value=mock_engine), \
             patch("facemoment.cli.utils.create_video_stream", return_value=(Mock(), Mock(), [Mock()])):
            mock_bg.return_value = Mock()

            from facemoment.main import run
            run("test.mp4", analyzers=["dummy"], on_trigger=cb)

            _, kwargs = mock_bg.call_args
            assert kwargs["on_trigger"] is cb


class TestPipelineConfigBackend:
    """Tests for PipelineConfig backend field."""

    def test_config_default_backend(self):
        """Test default backend in PipelineConfig."""
        from facemoment.pipeline.config import PipelineConfig

        config = PipelineConfig()
        assert config.backend == "pathway"

    def test_config_custom_backend(self):
        """Test custom backend in PipelineConfig."""
        from facemoment.pipeline.config import PipelineConfig

        config = PipelineConfig(backend="simple")
        assert config.backend == "simple"

    def test_config_from_dict_with_backend(self):
        """Test loading backend from dict."""
        from facemoment.pipeline.config import PipelineConfig

        data = {
            "analyzers": [{"name": "dummy"}],
            "backend": "simple",
        }
        config = PipelineConfig.from_dict(data)
        assert config.backend == "simple"

    def test_config_to_dict_includes_backend(self):
        """Test backend is included in to_dict."""
        from facemoment.pipeline.config import PipelineConfig, AnalyzerConfig

        config = PipelineConfig(
            analyzers=[AnalyzerConfig(name="dummy")],
            backend="pathway",
        )
        data = config.to_dict()
        assert data["backend"] == "pathway"

    def test_create_default_config_backend(self):
        """Test create_default_config accepts backend."""
        from facemoment.pipeline.config import create_default_config

        config = create_default_config(backend="simple")
        assert config.backend == "simple"


class TestOrchestratorBackend:
    """Tests for PipelineOrchestrator backend support."""

    def test_orchestrator_default_backend(self):
        """Test default backend in orchestrator."""
        from facemoment.pipeline import PipelineOrchestrator, AnalyzerConfig

        orchestrator = PipelineOrchestrator(
            analyzer_configs=[AnalyzerConfig(name="dummy")]
        )
        assert orchestrator._backend == "pathway"

    def test_orchestrator_custom_backend(self):
        """Test custom backend in orchestrator."""
        from facemoment.pipeline import PipelineOrchestrator, AnalyzerConfig

        orchestrator = PipelineOrchestrator(
            analyzer_configs=[AnalyzerConfig(name="dummy")],
            backend="simple",
        )
        assert orchestrator._backend == "simple"

    def test_from_config_passes_backend(self):
        """Test that from_config passes backend to orchestrator."""
        from facemoment.pipeline import (
            PipelineOrchestrator,
            PipelineConfig,
            AnalyzerConfig,
        )

        config = PipelineConfig(
            analyzers=[AnalyzerConfig(name="dummy")],
            backend="simple",
        )
        orchestrator = PipelineOrchestrator.from_config(config)
        assert orchestrator._backend == "simple"


class TestPipelineDelegation:
    """Tests for FacemomentPipeline delegating to the unified FlowGraph path."""

    def test_pipeline_delegates_build_modules(self):
        """Test that pipeline uses build_modules for module construction."""
        from facemoment.main import build_modules

        modules = build_modules(["face"])
        names = [m for m in modules if isinstance(m, str)]
        assert "face_classifier" in names  # auto-injected

    def test_pipeline_actual_backend_recorded(self):
        """Test that the actual backend name is recorded after run."""
        from visualpath.backends.base import PipelineResult

        pipeline = FacemomentPipeline(analyzers=["dummy"])

        mock_engine = Mock()
        mock_engine.execute.return_value = PipelineResult(triggers=[], frame_count=0)
        mock_engine.name = "TestBackend"

        with patch("facemoment.main.build_graph") as mock_bg, \
             patch("facemoment.main._get_backend", return_value=mock_engine):
            mock_bg.return_value = Mock()

            frames = [create_mock_frame(0, 0)]
            pipeline.run(frames)

            assert pipeline.actual_backend == "TestBackend"


class TestHighlightFusionMergedSignals:
    """Tests for HighlightFusion reading from merged signals."""

    def test_update_main_face_id_from_merged_signals(self):
        """Test that fusion reads main_face_id from merged signals."""
        from facemoment.moment_detector.fusion import HighlightFusion

        fusion = HighlightFusion(main_only=True)

        # Create observation with main_face_id in signals
        obs = Observation(
            source="merged",
            frame_id=1,
            t_ns=1000000,
            signals={"main_face_id": 42, "face_count": 1},
            faces=[
                FaceObservation(
                    face_id=42, bbox=(0.1, 0.1, 0.3, 0.3),
                    confidence=0.9, yaw=0.0, pitch=0.0, expression=0.8,
                )
            ],
            metadata={},
        )

        # Call update (which calls _update_main_face_id internally)
        fusion.update(obs)

        # main_face_id should be set from signals
        assert fusion._main_face_id == 42

    def test_explicit_classifier_obs_takes_priority(self):
        """Test that explicit classifier_obs takes priority over signals."""
        from facemoment.moment_detector.fusion import HighlightFusion

        fusion = HighlightFusion(main_only=True)

        # Create mock classifier observation
        mock_main_face = Mock()
        mock_main_face.face = Mock()
        mock_main_face.face.face_id = 99

        mock_data = Mock()
        mock_data.main_face = mock_main_face

        classifier_obs = Observation(
            source="face_classifier",
            frame_id=1,
            t_ns=1000000,
            signals={},
            faces=[],
            metadata={},
            data=mock_data,
        )

        # Create main observation with different main_face_id in signals
        obs = Observation(
            source="merged",
            frame_id=1,
            t_ns=1000000,
            signals={"main_face_id": 42},  # Different ID
            faces=[],
            metadata={},
        )

        # Call update with explicit classifier_obs
        fusion.update(obs, classifier_obs=classifier_obs)

        # Explicit classifier_obs should take priority
        assert fusion._main_face_id == 99


class TestCudaConflictDetection:
    """Tests for CUDA conflict detection and auto subprocess isolation."""

    def test_cuda_groups_defined(self):
        """Test that CUDA groups contain expected analyzers."""
        assert "onnxruntime" in _CUDA_GROUPS
        assert "torch" in _CUDA_GROUPS
        assert "face" in _CUDA_GROUPS["onnxruntime"]
        assert "pose" in _CUDA_GROUPS["torch"]

    def test_no_conflict_single_group(self):
        """No conflict when all analyzers are in the same CUDA group."""
        isolated = FacemomentPipeline._detect_cuda_conflicts(["face", "expression"])
        assert isolated == set()

    def test_no_conflict_no_cuda_analyzers(self):
        """No conflict when analyzers don't belong to any CUDA group."""
        isolated = FacemomentPipeline._detect_cuda_conflicts(["quality", "dummy"])
        assert isolated == set()

    def test_conflict_face_and_pose(self):
        """Conflict detected when face (onnxruntime) + pose (torch) are both active."""
        isolated = FacemomentPipeline._detect_cuda_conflicts(["face", "pose"])
        # torch group (pose) is minority → should be isolated
        assert isolated == {"pose"}

    def test_conflict_multiple_onnxruntime_vs_pose(self):
        """Multiple onnxruntime analyzers vs single torch → torch isolated."""
        isolated = FacemomentPipeline._detect_cuda_conflicts(
            ["face", "face_detect", "expression", "pose"]
        )
        # onnxruntime has 3 analyzers, torch has 1 → torch (pose) is minority
        assert isolated == {"pose"}

    def test_no_conflict_without_zmq(self):
        """Returns empty set when pyzmq is unavailable."""
        with patch.dict("sys.modules", {"zmq": None}):
            # Force ImportError for zmq
            import importlib
            with patch("builtins.__import__", side_effect=lambda name, *a, **kw: (_ for _ in ()).throw(ImportError("no zmq")) if name == "zmq" else importlib.__import__(name, *a, **kw)):
                isolated = FacemomentPipeline._detect_cuda_conflicts(["face", "pose"])
                assert isolated == set()

    def test_pipeline_init_has_workers_dict(self):
        """Pipeline initializes with empty workers dict."""
        pipeline = FacemomentPipeline(analyzers=["face"])
        assert pipeline._workers == {}
        assert pipeline.workers == {}

    def test_workers_property(self):
        """Workers property returns the _workers dict."""
        pipeline = FacemomentPipeline(analyzers=["face"])
        pipeline._workers = {"pose": Mock()}
        assert "pose" in pipeline.workers

    def test_isolation_config_built_for_conflicts(self):
        """IsolationConfig is built when CUDA conflicts detected."""
        from facemoment.main import _build_isolation_config

        config = _build_isolation_config(["face", "pose"])
        if config is not None:
            # pose should be isolated (minority group)
            from visualpath.core.isolation import IsolationLevel
            assert config.get_level("pose") == IsolationLevel.PROCESS

    def test_no_isolation_config_without_conflicts(self):
        """No IsolationConfig when no CUDA conflicts."""
        from facemoment.main import _build_isolation_config

        config = _build_isolation_config(["dummy"])
        assert config is None

    def test_cleanup_resets_state(self):
        """cleanup() resets initialized state."""
        pipeline = FacemomentPipeline(analyzers=["face"])
        pipeline._initialized = True
        pipeline.cleanup()
        assert not pipeline._initialized

    def test_initialize_loads_analyzers(self):
        """initialize() loads analyzers and fusion."""
        pipeline = FacemomentPipeline(analyzers=["dummy"])
        pipeline.initialize()
        assert pipeline._initialized
        assert len(pipeline.analyzers) > 0
        assert pipeline.fusion is not None
