"""Tests for momentscan pipeline integration."""

from unittest.mock import Mock, patch

import pytest

from vpx.sdk import Observation
from vpx.face_detect.types import FaceObservation
from vpx.face_detect.output import FaceDetectOutput


class TestHighlevelAPIBackend:
    """Tests for high-level API backend parameter."""

    def test_run_accepts_backend_parameter(self):
        """Test that ms.run() accepts backend parameter."""
        from momentscan.main import run

        # Just verify the function signature accepts backend
        import inspect
        sig = inspect.signature(run)
        assert "backend" in sig.parameters

    def test_main_run_uses_flowgraph_and_backend(self):
        """Test ms.run delegates to MomentscanApp.run()."""
        from visualpath.backends.base import PipelineResult

        mock_engine = Mock()
        mock_engine.execute.return_value = PipelineResult(triggers=[], frame_count=2)
        mock_engine.name = "SimpleBackend"

        with patch("visualpath.runner.get_backend", return_value=mock_engine), \
             patch("visualpath.runner._open_video_source", return_value=(iter([]), None)):
            from momentscan.main import run
            result = run("fake.mp4", analyzers=["mock.dummy"], fps=5, cooldown=1.5)

            assert result.frame_count == 2
            assert result.actual_backend == "SimpleBackend"

    def test_main_run_passes_on_trigger(self):
        """Test ms.run passes on_trigger through App."""
        from visualpath.backends.base import PipelineResult

        mock_engine = Mock()
        mock_engine.execute.return_value = PipelineResult(triggers=[], frame_count=0)
        mock_engine.name = "SimpleBackend"
        cb = lambda t: None

        with patch("visualpath.runner.get_backend", return_value=mock_engine), \
             patch("visualpath.runner._open_video_source", return_value=(iter([]), None)):
            from momentscan.main import run
            # Should not raise — on_trigger is passed through App
            run("test.mp4", analyzers=["mock.dummy"], on_trigger=cb)


class TestHighlightFusionMergedSignals:
    """Tests for HighlightFusion reading from merged signals."""

    def test_update_main_face_id_from_merged_signals(self):
        """Test that fusion reads main_face_id from merged signals."""
        from momentscan.algorithm.analyzers.highlight import HighlightFusion

        fusion = HighlightFusion(main_only=True)

        # Create observation with main_face_id in signals
        obs = Observation(
            source="merged",
            frame_id=1,
            t_ns=1000000,
            signals={"main_face_id": 42, "face_count": 1},
            data=FaceDetectOutput(faces=[
                FaceObservation(
                    face_id=42, bbox=(0.1, 0.1, 0.3, 0.3),
                    confidence=0.9, yaw=0.0, pitch=0.0, expression=0.8,
                )
            ]),
            metadata={},
        )

        # Call update (which calls _update_main_face_id internally)
        fusion.update(obs)

        # main_face_id should be set from signals
        assert fusion._main_face_id == 42

    def test_explicit_classifier_obs_takes_priority(self):
        """Test that explicit classifier_obs takes priority over signals."""
        from momentscan.algorithm.analyzers.highlight import HighlightFusion

        fusion = HighlightFusion(main_only=True)

        # Create mock classifier observation
        mock_main_face = Mock()
        mock_main_face.face = Mock()
        mock_main_face.face.face_id = 99

        mock_data = Mock()
        mock_data.main_face = mock_main_face

        classifier_obs = Observation(
            source="face.classify",
            frame_id=1,
            t_ns=1000000,
            signals={},
            metadata={},
            data=mock_data,
        )

        # Create main observation with different main_face_id in signals
        obs = Observation(
            source="merged",
            frame_id=1,
            t_ns=1000000,
            signals={"main_face_id": 42},  # Different ID
            metadata={},
        )

        # Call update with explicit classifier_obs
        fusion.update(obs, classifier_obs=classifier_obs)

        # Explicit classifier_obs should take priority
        assert fusion._main_face_id == 99


class TestResourceConflictIsolation:
    """Tests for resource conflict detection and auto subprocess isolation."""

    @staticmethod
    def _make_module(name, resource_groups=frozenset(), gpu=False):
        """Create a mock Module with capabilities."""
        from visualpath.core.capabilities import Capability, ModuleCapabilities

        flags = frozenset({Capability.GPU}) if gpu else frozenset()
        caps = ModuleCapabilities(
            resource_groups=resource_groups,
            flags=flags,
        )
        mod = Mock()
        mod.name = name
        mod.capabilities = caps
        return mod

    def test_no_conflict_single_group(self):
        """No conflict when all modules are in the same resource group."""
        from visualpath.core.compat import build_conflict_isolation

        modules = [
            self._make_module("face.detect", frozenset({"onnxruntime"})),
            self._make_module("face.expression", frozenset({"onnxruntime"})),
        ]
        config = build_conflict_isolation(modules)
        assert config is None

    def test_no_conflict_no_resource_groups(self):
        """No conflict when modules don't declare resource groups."""
        from visualpath.core.compat import build_conflict_isolation

        modules = [
            self._make_module("frame.quality"),
            self._make_module("mock.dummy"),
        ]
        config = build_conflict_isolation(modules)
        assert config is None

    def test_conflict_onnxruntime_vs_torch(self):
        """Conflict detected when onnxruntime + torch groups are both active."""
        from visualpath.core.compat import build_conflict_isolation
        from visualpath.core.isolation import IsolationLevel

        modules = [
            self._make_module("face.detect", frozenset({"onnxruntime"})),
            self._make_module("body.pose", frozenset({"torch"})),
        ]
        config = build_conflict_isolation(modules)
        assert config is not None
        # Both groups have 1 module — minority is whichever min() picks
        isolated_names = {n for n, lv in config.overrides.items() if lv == IsolationLevel.PROCESS}
        assert len(isolated_names) == 1

    def test_conflict_multiple_onnxruntime_vs_single_torch(self):
        """Multiple onnxruntime modules vs single torch -> torch isolated."""
        from visualpath.core.compat import build_conflict_isolation
        from visualpath.core.isolation import IsolationLevel

        modules = [
            self._make_module("face.detect", frozenset({"onnxruntime"})),
            self._make_module("face.expression", frozenset({"onnxruntime"})),
            self._make_module("body.pose", frozenset({"torch"})),
        ]
        config = build_conflict_isolation(modules)
        assert config is not None
        # torch has 1 module (minority) -> body.pose isolated
        assert config.get_level("body.pose") == IsolationLevel.PROCESS

    def test_no_conflict_without_zmq(self):
        """Returns None when pyzmq is unavailable."""
        from visualpath.core.compat import build_conflict_isolation
        import importlib

        modules = [
            self._make_module("face.detect", frozenset({"onnxruntime"})),
            self._make_module("body.pose", frozenset({"torch"})),
        ]
        with patch("builtins.__import__", side_effect=lambda name, *a, **kw: (_ for _ in ()).throw(ImportError("no zmq")) if name == "zmq" else importlib.__import__(name, *a, **kw)):
            config = build_conflict_isolation(modules)
            assert config is None

    def test_distributed_config_gpu_modules(self):
        """GPU modules get PROCESS isolation in distributed config."""
        from visualpath.core.compat import build_distributed_config
        from visualpath.core.isolation import IsolationLevel

        modules = [
            self._make_module("face.detect", gpu=True),
            self._make_module("face.classify", gpu=False),
        ]
        config = build_distributed_config(modules)
        assert config.get_level("face.detect") == IsolationLevel.PROCESS
        assert config.get_level("face.classify") == IsolationLevel.INLINE

    def test_distributed_config_with_venv(self):
        """GPU modules with venv_paths get VENV isolation."""
        from visualpath.core.compat import build_distributed_config
        from visualpath.core.isolation import IsolationLevel

        modules = [
            self._make_module("face.detect", gpu=True),
            self._make_module("body.pose", gpu=True),
        ]
        config = build_distributed_config(
            modules, venv_paths={"face.detect": "/opt/venv-face"}
        )
        assert config.get_level("face.detect") == IsolationLevel.VENV
        assert config.get_level("body.pose") == IsolationLevel.PROCESS
        assert config.venv_paths["face.detect"] == "/opt/venv-face"
