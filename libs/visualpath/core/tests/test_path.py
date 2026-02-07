"""Tests for Path and PathOrchestrator."""

import pytest
import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

from visualpath.core import (
    Module,
    Observation,
    Path,
    PathConfig,
    PathOrchestrator,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@dataclass
class MockFrame:
    """Mock Frame for testing."""
    frame_id: int
    t_src_ns: int
    data: np.ndarray


class CountingExtractor(Module):
    """Extractor that counts calls for testing."""

    def __init__(self, name: str, return_value: float = 0.5):
        self._name = name
        self._return_value = return_value
        self._process_count = 0
        self._initialized = False
        self._cleaned_up = False

    @property
    def name(self) -> str:
        return self._name

    def process(self, frame, deps=None) -> Optional[Observation]:
        self._process_count += 1
        return Observation(
            source=self.name,
            frame_id=frame.frame_id,
            t_ns=frame.t_src_ns,
            signals={"value": self._return_value, "call_count": self._process_count},
        )

    def initialize(self) -> None:
        self._initialized = True

    def cleanup(self) -> None:
        self._cleaned_up = True


class ThresholdFusion(Module):
    """Simple fusion for testing."""

    depends = []

    def __init__(self, threshold: float = 0.5):
        self._threshold = threshold
        self._gate_open = True
        self._cooldown = False
        self._update_count = 0

    @property
    def name(self) -> str:
        return "threshold_fusion"

    def process(self, frame, deps=None) -> Optional[Observation]:
        """Process observations from deps and decide on trigger."""
        self._update_count += 1
        # Find any observation from deps
        observation = None
        if deps:
            for obs in deps.values():
                if obs is not None:
                    observation = obs
                    break

        if observation is None:
            return Observation(
                source=self.name,
                frame_id=frame.frame_id,
                t_ns=frame.t_src_ns,
                signals={"should_trigger": False},
            )

        value = observation.signals.get("value", 0)
        if value > self._threshold:
            return Observation(
                source=self.name,
                frame_id=frame.frame_id,
                t_ns=frame.t_src_ns,
                signals={
                    "should_trigger": True,
                    "trigger_score": value,
                    "trigger_reason": "threshold_exceeded",
                },
            )
        return Observation(
            source=self.name,
            frame_id=frame.frame_id,
            t_ns=frame.t_src_ns,
            signals={"should_trigger": False},
        )

    def reset(self) -> None:
        self._update_count = 0


# =============================================================================
# PathConfig Tests
# =============================================================================


class TestPathConfig:
    """Tests for PathConfig dataclass."""

    def test_basic_config(self):
        """Test creating a basic path config."""
        from visualpath.core.isolation import IsolationLevel

        config = PathConfig(
            name="human",
            modules=["face", "pose"],
        )

        assert config.name == "human"
        assert config.modules == ["face", "pose"]
        assert config.default_isolation == IsolationLevel.INLINE

    def test_config_with_isolation(self):
        """Test config with isolation level."""
        from visualpath.core.isolation import IsolationLevel

        config = PathConfig(
            name="scene",
            modules=["object", "ocr"],
            default_isolation=IsolationLevel.VENV,
        )

        assert config.default_isolation == IsolationLevel.VENV

    def test_config_with_module_overrides(self):
        """Test config with per-module overrides."""
        config = PathConfig(
            name="human",
            modules=["face", "pose"],
            module_config={
                "face": {"confidence_threshold": 0.8},
                "pose": {"max_persons": 5},
            },
        )

        assert config.module_config["face"]["confidence_threshold"] == 0.8
        assert config.module_config["pose"]["max_persons"] == 5


# =============================================================================
# Path Tests
# =============================================================================


class TestPath:
    """Tests for Path class."""

    def test_basic_path_creation(self):
        """Test creating a basic path."""
        e1 = CountingExtractor("ext1")
        e2 = CountingExtractor("ext2")

        path = Path(
            name="test",
            modules=[e1, e2],
        )

        assert path.name == "test"
        assert len(path.extractors) == 2
        assert path.fusion is None

    def test_path_with_fusion(self):
        """Test path with fusion module."""
        e1 = CountingExtractor("ext1")
        fusion = ThresholdFusion()

        path = Path(
            name="test",
            modules=[e1],
            fusion=fusion,
        )

        assert path.fusion is fusion

    def test_path_lifecycle(self):
        """Test path initialize/cleanup lifecycle."""
        e1 = CountingExtractor("ext1")
        e2 = CountingExtractor("ext2")

        path = Path(name="test", modules=[e1, e2])

        assert not e1._initialized
        assert not e2._initialized

        path.initialize()

        assert e1._initialized
        assert e2._initialized

        path.cleanup()

        assert e1._cleaned_up
        assert e2._cleaned_up

    def test_path_context_manager(self):
        """Test path as context manager."""
        e1 = CountingExtractor("ext1")
        e2 = CountingExtractor("ext2")

        path = Path(name="test", modules=[e1, e2])

        with path as p:
            assert e1._initialized
            assert e2._initialized
            assert p is path

        assert e1._cleaned_up
        assert e2._cleaned_up

    def test_extract_all_sequential(self):
        """Test extracting from all modules sequentially."""
        e1 = CountingExtractor("ext1", return_value=0.3)
        e2 = CountingExtractor("ext2", return_value=0.7)

        path = Path(name="test", modules=[e1, e2])
        frame = MockFrame(frame_id=1, t_src_ns=1000000, data=np.zeros((100, 100, 3)))

        with path:
            observations = path.extract_all(frame)

        assert len(observations) == 2
        assert observations[0].source == "ext1"
        assert observations[0].signals["value"] == 0.3
        assert observations[1].source == "ext2"
        assert observations[1].signals["value"] == 0.7

    def test_extract_all_multiple(self):
        """Test extracting from multiple modules."""
        e1 = CountingExtractor("ext1", return_value=0.3)
        e2 = CountingExtractor("ext2", return_value=0.7)
        e3 = CountingExtractor("ext3", return_value=0.5)

        path = Path(name="test", modules=[e1, e2, e3])
        frame = MockFrame(frame_id=1, t_src_ns=1000000, data=np.zeros((100, 100, 3)))

        with path:
            observations = path.extract_all(frame)

        assert len(observations) == 3
        sources = {obs.source for obs in observations}
        assert sources == {"ext1", "ext2", "ext3"}

    def test_extract_all_not_initialized_raises(self):
        """Test that extract_all raises when not initialized."""
        e1 = CountingExtractor("ext1")
        path = Path(name="test", modules=[e1])
        frame = MockFrame(frame_id=1, t_src_ns=1000000, data=np.zeros((100, 100, 3)))

        with pytest.raises(RuntimeError, match="not initialized"):
            path.extract_all(frame)

    def test_process_with_fusion(self):
        """Test processing through modules and fusion."""
        e1 = CountingExtractor("ext1", return_value=0.7)  # Above threshold
        e2 = CountingExtractor("ext2", return_value=0.3)  # Below threshold
        fusion = ThresholdFusion(threshold=0.5)

        path = Path(name="test", modules=[e1, e2], fusion=fusion)
        frame = MockFrame(frame_id=1, t_src_ns=1000000, data=np.zeros((100, 100, 3)))

        with path:
            results = path.process(frame)

        # Fusion is called once with all observations aggregated
        assert len(results) == 1
        # ThresholdFusion picks first non-None obs (ext1, value=0.7 > 0.5)
        assert results[0].should_trigger

    def test_process_no_fusion(self):
        """Test processing without fusion returns empty."""
        e1 = CountingExtractor("ext1")
        path = Path(name="test", modules=[e1])
        frame = MockFrame(frame_id=1, t_src_ns=1000000, data=np.zeros((100, 100, 3)))

        with path:
            results = path.process(frame)

        assert results == []


# =============================================================================
# PathOrchestrator Tests
# =============================================================================


class TestPathOrchestrator:
    """Tests for PathOrchestrator class."""

    def test_orchestrator_creation(self):
        """Test creating an orchestrator."""
        path1 = Path(name="path1", modules=[CountingExtractor("ext1")])
        path2 = Path(name="path2", modules=[CountingExtractor("ext2")])

        orchestrator = PathOrchestrator([path1, path2])

        assert len(orchestrator.paths) == 2

    def test_orchestrator_lifecycle(self):
        """Test orchestrator initialize/cleanup lifecycle."""
        e1 = CountingExtractor("ext1")
        e2 = CountingExtractor("ext2")
        path1 = Path(name="path1", modules=[e1])
        path2 = Path(name="path2", modules=[e2])

        orchestrator = PathOrchestrator([path1, path2])

        assert not e1._initialized
        assert not e2._initialized

        orchestrator.initialize()

        assert e1._initialized
        assert e2._initialized

        orchestrator.cleanup()

        assert e1._cleaned_up
        assert e2._cleaned_up

    def test_orchestrator_context_manager(self):
        """Test orchestrator as context manager."""
        e1 = CountingExtractor("ext1")
        e2 = CountingExtractor("ext2")
        path1 = Path(name="path1", modules=[e1])
        path2 = Path(name="path2", modules=[e2])

        orchestrator = PathOrchestrator([path1, path2])

        with orchestrator as o:
            assert e1._initialized
            assert e2._initialized
            assert o is orchestrator

        assert e1._cleaned_up
        assert e2._cleaned_up

    def test_process_all_sequential(self):
        """Test processing all paths sequentially."""
        e1 = CountingExtractor("ext1", return_value=0.7)
        e2 = CountingExtractor("ext2", return_value=0.3)
        fusion1 = ThresholdFusion(threshold=0.5)
        fusion2 = ThresholdFusion(threshold=0.5)

        path1 = Path(name="path1", modules=[e1], fusion=fusion1)
        path2 = Path(name="path2", modules=[e2], fusion=fusion2)

        orchestrator = PathOrchestrator([path1, path2], parallel=False)
        frame = MockFrame(frame_id=1, t_src_ns=1000000, data=np.zeros((100, 100, 3)))

        with orchestrator:
            results = orchestrator.process_all(frame)

        assert "path1" in results
        assert "path2" in results
        assert len(results["path1"]) == 1
        assert len(results["path2"]) == 1
        assert results["path1"][0].should_trigger  # 0.7 > 0.5
        assert not results["path2"][0].should_trigger  # 0.3 < 0.5

    def test_process_all_parallel(self):
        """Test processing all paths in parallel."""
        e1 = CountingExtractor("ext1", return_value=0.7)
        e2 = CountingExtractor("ext2", return_value=0.8)
        fusion1 = ThresholdFusion(threshold=0.5)
        fusion2 = ThresholdFusion(threshold=0.5)

        path1 = Path(name="path1", modules=[e1], fusion=fusion1)
        path2 = Path(name="path2", modules=[e2], fusion=fusion2)

        orchestrator = PathOrchestrator([path1, path2], parallel=True)
        frame = MockFrame(frame_id=1, t_src_ns=1000000, data=np.zeros((100, 100, 3)))

        with orchestrator:
            results = orchestrator.process_all(frame)

        assert "path1" in results
        assert "path2" in results
        assert results["path1"][0].should_trigger
        assert results["path2"][0].should_trigger

    def test_process_all_not_initialized_raises(self):
        """Test that process_all raises when not initialized."""
        path = Path(name="path1", modules=[CountingExtractor("ext1")])
        orchestrator = PathOrchestrator([path])
        frame = MockFrame(frame_id=1, t_src_ns=1000000, data=np.zeros((100, 100, 3)))

        with pytest.raises(RuntimeError, match="not initialized"):
            orchestrator.process_all(frame)

    def test_extract_all(self):
        """Test extracting observations from all paths."""
        e1 = CountingExtractor("ext1")
        e2 = CountingExtractor("ext2")
        e3 = CountingExtractor("ext3")

        path1 = Path(name="path1", modules=[e1, e2])
        path2 = Path(name="path2", modules=[e3])

        orchestrator = PathOrchestrator([path1, path2])
        frame = MockFrame(frame_id=1, t_src_ns=1000000, data=np.zeros((100, 100, 3)))

        with orchestrator:
            observations = orchestrator.extract_all(frame)

        assert "path1" in observations
        assert "path2" in observations
        assert len(observations["path1"]) == 2
        assert len(observations["path2"]) == 1

    def test_multiple_frames(self):
        """Test processing multiple frames."""
        e1 = CountingExtractor("ext1")
        path = Path(name="path1", modules=[e1])
        orchestrator = PathOrchestrator([path])

        frames = [
            MockFrame(frame_id=i, t_src_ns=i * 1000000, data=np.zeros((100, 100, 3)))
            for i in range(5)
        ]

        with orchestrator:
            for frame in frames:
                orchestrator.extract_all(frame)

        assert e1._process_count == 5
