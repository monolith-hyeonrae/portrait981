"""Tests for render_marks() renderer."""

import numpy as np
import pytest

from vpx.viz.renderer import render_marks
from vpx.sdk.marks import (
    BBoxMark,
    KeypointsMark,
    BarMark,
    LabelMark,
    DrawStyle,
)


@pytest.fixture
def blank_frame():
    """640x480 black frame."""
    return np.zeros((480, 640, 3), dtype=np.uint8)


class TestRenderMarks:
    def test_empty_marks_returns_same_frame(self, blank_frame):
        result = render_marks(blank_frame, [])
        # Should return the same object (not a copy) when no marks
        assert result is blank_frame

    def test_does_not_modify_input(self, blank_frame):
        marks = [BBoxMark(x=0.1, y=0.1, w=0.3, h=0.3, color=(0, 255, 0))]
        render_marks(blank_frame, marks)
        # Input frame should still be all zeros
        assert np.all(blank_frame == 0)

    def test_render_bbox_mark_modifies_frame(self, blank_frame):
        marks = [BBoxMark(x=0.1, y=0.1, w=0.3, h=0.3, color=(0, 255, 0))]
        result = render_marks(blank_frame, marks)
        assert not np.array_equal(result, blank_frame)

    def test_render_bbox_with_label(self, blank_frame):
        marks = [BBoxMark(x=0.1, y=0.1, w=0.3, h=0.3, label="ID:1", color=(0, 255, 0))]
        result = render_marks(blank_frame, marks)
        assert not np.array_equal(result, blank_frame)

    def test_render_keypoints_mark(self, blank_frame):
        points = (
            (320.0, 100.0, 0.9),
            (310.0, 90.0, 0.85),
            (330.0, 90.0, 0.85),
        )
        connections = ((0, 1), (0, 2))
        marks = [KeypointsMark(points=points, connections=connections)]
        result = render_marks(blank_frame, marks)
        assert not np.array_equal(result, blank_frame)

    def test_render_keypoints_normalized(self, blank_frame):
        points = (
            (0.5, 0.2, 0.9),
            (0.4, 0.3, 0.85),
        )
        marks = [KeypointsMark(points=points, connections=((0, 1),), normalized=True)]
        result = render_marks(blank_frame, marks)
        assert not np.array_equal(result, blank_frame)

    def test_render_keypoints_low_confidence_skipped(self, blank_frame):
        points = (
            (320.0, 100.0, 0.1),  # Below min_confidence
            (310.0, 90.0, 0.1),
        )
        marks = [KeypointsMark(points=points, connections=((0, 1),), min_confidence=0.3)]
        result = render_marks(blank_frame, marks)
        # Low-confidence points should be skipped, frame unchanged
        assert np.array_equal(result, blank_frame)

    def test_render_bar_mark(self, blank_frame):
        marks = [BarMark(x=0.1, y=0.5, w=0.3, value=0.7, color=(0, 255, 255))]
        result = render_marks(blank_frame, marks)
        assert not np.array_equal(result, blank_frame)

    def test_render_bar_mark_zero_value(self, blank_frame):
        marks = [BarMark(x=0.1, y=0.5, w=0.3, value=0.0)]
        result = render_marks(blank_frame, marks)
        # Background is drawn even for zero value
        assert not np.array_equal(result, blank_frame)

    def test_render_label_mark(self, blank_frame):
        marks = [LabelMark(text="hello", x=0.5, y=0.5, color=(255, 255, 255))]
        result = render_marks(blank_frame, marks)
        assert not np.array_equal(result, blank_frame)

    def test_render_label_with_background(self, blank_frame):
        marks = [LabelMark(text="test", x=0.5, y=0.5, background=(40, 40, 40))]
        result = render_marks(blank_frame, marks)
        assert not np.array_equal(result, blank_frame)

    def test_render_style_overrides_color(self, blank_frame):
        marks = [BBoxMark(x=0.1, y=0.1, w=0.3, h=0.3, color=(0, 255, 0))]
        style = DrawStyle(color=(0, 0, 255))

        result_default = render_marks(blank_frame, marks)
        result_styled = render_marks(blank_frame, marks, style=style)

        # Styled result should differ from default (different color)
        assert not np.array_equal(result_default, result_styled)

    def test_render_style_overrides_thickness(self, blank_frame):
        marks = [BBoxMark(x=0.1, y=0.1, w=0.3, h=0.3, color=(0, 255, 0), thickness=1)]
        style = DrawStyle(thickness=5)

        result_thin = render_marks(blank_frame, marks)
        result_thick = render_marks(blank_frame, marks, style=style)

        # Different thickness should produce different results
        assert not np.array_equal(result_thin, result_thick)

    def test_render_multiple_mark_types(self, blank_frame):
        marks = [
            BBoxMark(x=0.1, y=0.1, w=0.2, h=0.2, color=(0, 255, 0)),
            LabelMark(text="face", x=0.1, y=0.08, color=(255, 255, 255)),
            BarMark(x=0.1, y=0.35, w=0.2, value=0.8),
        ]
        result = render_marks(blank_frame, marks)
        assert not np.array_equal(result, blank_frame)

    def test_render_keypoints_with_point_colors(self, blank_frame):
        points = ((320.0, 100.0, 0.9), (310.0, 90.0, 0.9))
        point_colors = {0: (255, 255, 255), 1: (0, 255, 0)}
        marks = [KeypointsMark(points=points, point_colors=point_colors)]
        result = render_marks(blank_frame, marks)
        assert not np.array_equal(result, blank_frame)
