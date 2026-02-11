"""Tests for DebugFrameHandler on_frame callback.

Verifies:
- Observation extraction from FlowData
- Gate/cooldown state parsing from fusion metadata
- Trigger counting
- Return value behavior (True=continue, False=stop)
"""

from unittest.mock import Mock, patch, MagicMock
import numpy as np
import pytest

from visualpath.flow.node import FlowData
from vpx.sdk import Observation
from helpers import create_mock_frame


def _make_observation(source, signals=None, metadata=None, data=None):
    """Create a test observation."""
    return Observation(
        source=source,
        frame_id=0,
        t_ns=0,
        signals=signals or {},
        metadata=metadata or {},
        data=data,
    )


def _make_flow_data(observations, frame=None):
    """Create a FlowData with the given observations."""
    obs_list = list(observations.values()) if isinstance(observations, dict) else observations
    result_obs = [o for o in obs_list if getattr(o, 'should_trigger', False)]
    return FlowData(
        frame=frame,
        observations=obs_list,
        results=result_obs,
    )


@pytest.fixture
def handler():
    """Create a DebugFrameHandler with mocked cv2."""
    with patch("facemoment.cli.debug_handler.cv2") as mock_cv2:
        mock_cv2.waitKey.return_value = 0  # No key pressed
        from facemoment.cli.debug_handler import DebugFrameHandler
        h = DebugFrameHandler(show_window=False)
        # Mock the visualizer to avoid needing real cv2
        h.visualizer = MagicMock()
        h.visualizer.create_debug_view.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
        h.scorer = MagicMock()
        h.scorer.score.return_value = None
        yield h


class TestObservationExtraction:
    """Test that DebugFrameHandler extracts observations correctly."""

    def test_extracts_observations_from_flow_data(self, handler):
        """Handler should extract observations from FlowData."""
        face_obs = _make_observation("face.detect", signals={"face_count": 2})
        pose_obs = _make_observation("body.pose", signals={"person_count": 1})

        frame = create_mock_frame()
        fd = _make_flow_data({"face.detect": face_obs, "body.pose": pose_obs}, frame=frame)

        result = handler(frame, [fd])

        assert result is True
        handler.visualizer.create_debug_view.assert_called_once()
        call_kwargs = handler.visualizer.create_debug_view.call_args[1]
        assert call_kwargs["face_obs"] is face_obs
        assert call_kwargs["pose_obs"] is pose_obs

    def test_empty_terminal_results(self, handler):
        """Handler should handle empty terminal results gracefully."""
        frame = create_mock_frame()
        result = handler(frame, [])
        assert result is True

    def test_no_flow_data_continues(self, handler):
        """Handler should return True when flow data is None/empty."""
        frame = create_mock_frame()
        result = handler(frame, [])
        assert result is True


class TestGateCooldownState:
    """Test gate/cooldown state parsing from fusion metadata."""

    def test_cooldown_state(self, handler):
        """state='cooldown' should set in_cooldown=True."""
        fusion_obs = _make_observation(
            "highlight",
            signals={"should_trigger": False},
            metadata={"state": "cooldown"},
        )
        frame = create_mock_frame()
        fd = _make_flow_data([fusion_obs], frame=frame)

        handler(frame, [fd])

        call_kwargs = handler.visualizer.create_debug_view.call_args[1]
        assert call_kwargs["in_cooldown"] is True
        assert call_kwargs["is_gate_open"] is False

    def test_gate_closed_state(self, handler):
        """state='gate_closed' should set is_gate_open=False."""
        fusion_obs = _make_observation(
            "highlight",
            signals={"should_trigger": False},
            metadata={"state": "gate_closed"},
        )
        frame = create_mock_frame()
        fd = _make_flow_data([fusion_obs], frame=frame)

        handler(frame, [fd])

        call_kwargs = handler.visualizer.create_debug_view.call_args[1]
        assert call_kwargs["is_gate_open"] is False
        assert call_kwargs["in_cooldown"] is False

    def test_monitoring_gate_open(self, handler):
        """state='monitoring' with gate_open=True should set is_gate_open=True."""
        fusion_obs = _make_observation(
            "highlight",
            signals={"should_trigger": False},
            metadata={"state": "monitoring", "gate_open": True},
        )
        frame = create_mock_frame()
        fd = _make_flow_data([fusion_obs], frame=frame)

        handler(frame, [fd])

        call_kwargs = handler.visualizer.create_debug_view.call_args[1]
        assert call_kwargs["is_gate_open"] is True

    def test_trigger_fired_implies_gate_open(self, handler):
        """When should_trigger=True, gate should be considered open."""
        from visualbase import Trigger

        trigger = Trigger.point(
            event_time_ns=0, pre_sec=1.0, post_sec=1.0, label="test",
        )
        fusion_obs = _make_observation(
            "highlight",
            signals={"should_trigger": True},
            metadata={"trigger": trigger},
        )
        frame = create_mock_frame()
        fd = _make_flow_data([fusion_obs], frame=frame)

        handler(frame, [fd])

        call_kwargs = handler.visualizer.create_debug_view.call_args[1]
        assert call_kwargs["is_gate_open"] is True


class TestTriggerCounting:
    """Test trigger count tracking."""

    def test_counts_triggers(self, handler):
        """Handler should count observations with should_trigger=True."""
        from visualbase import Trigger

        trigger = Trigger.point(
            event_time_ns=0, pre_sec=1.0, post_sec=1.0, label="test",
        )
        fusion_obs = _make_observation(
            "highlight",
            signals={"should_trigger": True},
            metadata={"trigger": trigger},
        )
        frame = create_mock_frame()
        fd = _make_flow_data([fusion_obs], frame=frame)

        handler(frame, [fd])
        handler(frame, [fd])

        assert handler.trigger_count == 2

    def test_no_trigger_no_count(self, handler):
        """Non-trigger observations should not increment trigger count."""
        obs = _make_observation(
            "face.detect",
            signals={"should_trigger": False},
        )
        frame = create_mock_frame()
        fd = _make_flow_data([obs], frame=frame)

        handler(frame, [fd])

        assert handler.trigger_count == 0


class TestFrameCount:
    """Test frame count tracking."""

    def test_increments_frame_count(self, handler):
        """Handler should increment frame_count on each call."""
        obs = _make_observation("face.detect")
        frame = create_mock_frame()
        fd = _make_flow_data([obs], frame=frame)

        handler(frame, [fd])
        handler(frame, [fd])
        handler(frame, [fd])

        assert handler.frame_count == 3
