"""Tests for the high-level API."""

import numpy as np
import pytest

from visualbase import Frame

import visualpath as vp
from visualpath.api import (
    _analyzer_registry,
    _fusion_registry,
    FunctionAnalyzer,
    FunctionFusion,
)


@pytest.fixture(autouse=True)
def clear_registry():
    """Clear registries before each test."""
    _analyzer_registry.clear()
    _fusion_registry.clear()
    yield
    _analyzer_registry.clear()
    _fusion_registry.clear()


def make_frame(frame_id: int = 0, t_ns: int = 0) -> Frame:
    """Create a test frame."""
    return Frame.from_array(
        np.zeros((480, 640, 3), dtype=np.uint8),
        frame_id=frame_id,
        t_src_ns=t_ns,
    )


class TestAnalyzerDecorator:
    """Tests for @vp.analyzer decorator."""

    def test_basic_analyzer(self):
        """Test simple analyzer creation."""
        @vp.analyzer("test")
        def my_analyzer(frame):
            return {"score": 0.5}

        assert isinstance(my_analyzer, FunctionAnalyzer)
        assert my_analyzer.name == "test"

        # Should be registered
        assert "test" in _analyzer_registry

    def test_analyzer_process(self):
        """Test analyzer process."""
        @vp.analyzer("brightness")
        def check_brightness(frame):
            return {"brightness": 128.0, "valid": 1.0}

        frame = make_frame()
        obs = check_brightness.process(frame)

        assert obs is not None
        assert obs.source == "brightness"
        assert obs.signals["brightness"] == 128.0
        assert obs.signals["valid"] == 1.0

    def test_analyzer_returns_none(self):
        """Test analyzer that returns None."""
        @vp.analyzer("conditional")
        def conditional_analyzer(frame):
            if frame.frame_id < 5:
                return None
            return {"passed": 1.0}

        frame = make_frame(frame_id=0)
        assert conditional_analyzer.process(frame) is None

        frame = make_frame(frame_id=10)
        obs = conditional_analyzer.process(frame)
        assert obs is not None
        assert obs.signals["passed"] == 1.0

    def test_analyzer_with_init_cleanup(self):
        """Test analyzer with init and cleanup functions."""
        state = {"initialized": False, "cleaned": False}

        def init():
            state["initialized"] = True

        def cleanup():
            state["cleaned"] = True

        @vp.analyzer("stateful", init=init, cleanup=cleanup)
        def stateful_analyzer(frame):
            return {"ready": 1.0 if state["initialized"] else 0.0}

        assert not state["initialized"]

        stateful_analyzer.initialize()
        assert state["initialized"]

        stateful_analyzer.cleanup()
        assert state["cleaned"]

    def test_analyzer_context_manager(self):
        """Test analyzer as context manager."""
        calls = []

        @vp.analyzer(
            "ctx",
            init=lambda: calls.append("init"),
            cleanup=lambda: calls.append("cleanup"),
        )
        def ctx_analyzer(frame):
            return {"ok": 1.0}

        with ctx_analyzer:
            assert "init" in calls
            assert "cleanup" not in calls

        assert "cleanup" in calls

    def test_analyzer_non_scalar_data(self):
        """Test analyzer with non-scalar data."""
        @vp.analyzer("objects")
        def object_detector(frame):
            return {
                "count": 2.0,
                "boxes": [[10, 20, 30, 40], [50, 60, 70, 80]],
            }

        frame = make_frame()
        obs = object_detector.process(frame)

        assert obs.signals["count"] == 2.0
        assert obs.data["boxes"] == [[10, 20, 30, 40], [50, 60, 70, 80]]


class TestFusionDecorator:
    """Tests for @vp.fusion decorator."""

    def test_basic_fusion(self):
        """Test simple fusion creation."""
        @vp.fusion(sources=["face"])
        def smile_detector(face):
            if face.get("happy", 0) > 0.5:
                return vp.trigger("smile")

        assert isinstance(smile_detector, FunctionFusion)
        assert smile_detector.name == "smile_detector"
        assert "smile_detector" in _fusion_registry

    def test_fusion_with_name(self):
        """Test fusion with custom name."""
        @vp.fusion(sources=["face"], name="my_fusion")
        def detector(face):
            pass

        assert detector.name == "my_fusion"

    def test_fusion_triggers(self):
        """Test fusion triggering."""
        @vp.fusion(sources=["face"], cooldown=0.1)
        def happy_trigger(face):
            if face.get("happy", 0) > 0.5:
                return vp.trigger("happy", score=face["happy"])

        # Create observation
        obs = vp.Observation(
            source="face",
            frame_id=0,
            t_ns=0,
            signals={"happy": 0.8},
        )

        frame = make_frame(frame_id=0, t_ns=0)
        result = happy_trigger.process(frame, {"face": obs})

        assert result.should_trigger
        assert result.trigger is not None
        assert result.trigger_reason == "happy"
        assert result.trigger_score == 0.8

    def test_fusion_no_trigger(self):
        """Test fusion not triggering."""
        @vp.fusion(sources=["face"])
        def happy_trigger(face):
            if face.get("happy", 0) > 0.5:
                return vp.trigger("happy")

        obs = vp.Observation(
            source="face",
            frame_id=0,
            t_ns=0,
            signals={"happy": 0.2},  # Below threshold
        )

        frame = make_frame(frame_id=0, t_ns=0)
        result = happy_trigger.process(frame, {"face": obs})
        assert not result.should_trigger

    def test_fusion_cooldown(self):
        """Test fusion cooldown."""
        @vp.fusion(sources=["face"], cooldown=1.0)
        def always_trigger(face):
            return vp.trigger("test")

        # First trigger
        obs1 = vp.Observation(source="face", frame_id=0, t_ns=0, signals={})
        frame1 = make_frame(frame_id=0, t_ns=0)
        result1 = always_trigger.process(frame1, {"face": obs1})
        assert result1.should_trigger

        # During cooldown
        obs2 = vp.Observation(source="face", frame_id=1, t_ns=int(0.5e9), signals={})
        frame2 = make_frame(frame_id=1, t_ns=int(0.5e9))
        result2 = always_trigger.process(frame2, {"face": obs2})
        assert not result2.should_trigger  # In cooldown

        # After cooldown
        obs3 = vp.Observation(source="face", frame_id=2, t_ns=int(1.5e9), signals={})
        frame3 = make_frame(frame_id=2, t_ns=int(1.5e9))
        result3 = always_trigger.process(frame3, {"face": obs3})
        assert result3.should_trigger

    def test_fusion_multiple_sources(self):
        """Test fusion with multiple sources."""
        @vp.fusion(sources=["face", "pose"])
        def interaction(face, pose):
            if face.get("happy", 0) > 0.5 and pose.get("wave", 0) > 0.5:
                return vp.trigger("greeting")

        # Only face observation - should not trigger
        obs_face = vp.Observation(
            source="face", frame_id=0, t_ns=0,
            signals={"happy": 0.8},
        )
        frame1 = make_frame(frame_id=0, t_ns=0)
        result1 = interaction.process(frame1, {"face": obs_face})
        assert not result1.should_trigger  # Missing pose

        # Add pose observation (face already stored from previous call)
        obs_pose = vp.Observation(
            source="pose", frame_id=0, t_ns=0,
            signals={"wave": 0.9},
        )
        result2 = interaction.process(frame1, {"pose": obs_pose})
        assert result2.should_trigger


class TestTriggerSpec:
    """Tests for trigger() helper."""

    def test_simple_trigger(self):
        """Test simple trigger creation."""
        t = vp.trigger("smile")
        assert t.reason == "smile"
        assert t.score == 1.0
        assert t.metadata == {}

    def test_trigger_with_score(self):
        """Test trigger with custom score."""
        t = vp.trigger("wave", score=0.75)
        assert t.reason == "wave"
        assert t.score == 0.75

    def test_trigger_with_metadata(self):
        """Test trigger with metadata."""
        t = vp.trigger("face", score=0.9, face_id=5, emotion="happy")
        assert t.reason == "face"
        assert t.score == 0.9
        assert t.metadata == {"face_id": 5, "emotion": "happy"}


class TestRegistry:
    """Tests for analyzer/fusion registry."""

    def test_get_analyzer(self):
        """Test getting registered analyzer."""
        @vp.analyzer("test_ext")
        def my_ext(frame):
            return {"x": 1.0}

        ext = vp.get_analyzer("test_ext")
        assert ext is not None
        assert ext.name == "test_ext"

    def test_get_unknown_analyzer(self):
        """Test getting unknown analyzer."""
        ext = vp.get_analyzer("nonexistent")
        assert ext is None

    def test_get_fusion(self):
        """Test getting registered fusion."""
        @vp.fusion(sources=["a"], name="test_fus")
        def my_fus(a):
            pass

        fus = vp.get_fusion("test_fus")
        assert fus is not None
        assert fus.name == "test_fus"

    def test_list_analyzers(self):
        """Test listing analyzers."""
        @vp.analyzer("ext1")
        def ext1(frame):
            return {}

        @vp.analyzer("ext2")
        def ext2(frame):
            return {}

        names = vp.list_analyzers()
        assert "ext1" in names
        assert "ext2" in names

    def test_list_fusions(self):
        """Test listing fusions."""
        @vp.fusion(sources=["a"], name="fus1")
        def fus1(a):
            pass

        @vp.fusion(sources=["b"], name="fus2")
        def fus2(b):
            pass

        names = vp.list_fusions()
        assert "fus1" in names
        assert "fus2" in names


class TestProcessResult:
    """Tests for ProcessResult dataclass."""

    def test_default_values(self):
        """Test ProcessResult defaults."""
        result = vp.ProcessResult()
        assert result.triggers == []
        assert result.frame_count == 0
        assert result.duration_sec == 0.0


class TestAPIUsability:
    """Tests demonstrating API usability."""

    def test_readme_example(self):
        """Test example that would go in README."""
        # Define analyzer
        @vp.analyzer("quality")
        def check_quality(frame):
            brightness = float(frame.data.mean())
            return {"brightness": brightness, "is_bright": brightness > 128}

        # Define fusion
        @vp.fusion(sources=["quality"], cooldown=0.5)
        def brightness_spike(quality):
            if quality.get("is_bright") and quality.get("brightness", 0) > 200:
                return vp.trigger("bright_frame", score=quality["brightness"] / 255)

        # Test the analyzer
        frame = Frame.from_array(
            np.full((480, 640, 3), 220, dtype=np.uint8),  # Bright frame
            frame_id=0,
            t_src_ns=0,
        )

        obs = check_quality.process(frame)
        assert obs.signals["brightness"] == 220.0
        assert obs.signals["is_bright"] == 1.0

        # Test the fusion
        result = brightness_spike.process(frame, {"quality": obs})
        assert result.should_trigger
        assert result.trigger_reason == "bright_frame"
        assert 0.8 < result.trigger_score < 0.9  # 220/255 ≈ 0.86

    def test_minimal_analyzer(self):
        """Test minimal analyzer definition."""
        @vp.analyzer("simple")
        def simple(frame):
            return {"value": 1.0}

        # That's it - 3 lines to define an analyzer
        obs = simple.process(make_frame())
        assert obs.signals["value"] == 1.0

    def test_minimal_fusion(self):
        """Test minimal fusion definition."""
        @vp.fusion(sources=["simple"])
        def always_fire(simple):
            return vp.trigger("test")

        # That's it - 3 lines to define a fusion
        obs = vp.Observation(source="simple", frame_id=0, t_ns=0, signals={})
        frame = make_frame(frame_id=0, t_ns=0)
        result = always_fire.process(frame, {"simple": obs})
        assert result.should_trigger
