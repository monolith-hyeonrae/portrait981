"""Tests for vpx-sdk mark types."""

import pytest

from vpx.sdk.marks import (
    DrawStyle,
    BBoxMark,
    KeypointsMark,
    BarMark,
    LabelMark,
    Mark,
)


class TestMarkTypes:
    """Test that all mark types exist and can be instantiated."""

    def test_mark_types_exist(self):
        """All 4 mark types should be importable."""
        assert BBoxMark is not None
        assert KeypointsMark is not None
        assert BarMark is not None
        assert LabelMark is not None

    def test_bbox_mark_frozen(self):
        """BBoxMark should be immutable."""
        mark = BBoxMark(x=0.1, y=0.2, w=0.3, h=0.4)
        with pytest.raises(AttributeError):
            mark.x = 0.5  # type: ignore[misc]

    def test_keypoints_mark_frozen(self):
        """KeypointsMark should be immutable."""
        mark = KeypointsMark(points=((0.5, 0.5, 0.9),))
        with pytest.raises(AttributeError):
            mark.points = ()  # type: ignore[misc]

    def test_bar_mark_frozen(self):
        """BarMark should be immutable."""
        mark = BarMark(x=0.1, y=0.2, w=0.3, value=0.5)
        with pytest.raises(AttributeError):
            mark.value = 0.8  # type: ignore[misc]

    def test_label_mark_frozen(self):
        """LabelMark should be immutable."""
        mark = LabelMark(text="hello", x=0.1, y=0.2)
        with pytest.raises(AttributeError):
            mark.text = "bye"  # type: ignore[misc]

    def test_bbox_mark_defaults(self):
        """BBoxMark defaults."""
        mark = BBoxMark(x=0.1, y=0.2, w=0.3, h=0.4)
        assert mark.label == ""
        assert mark.color is None
        assert mark.thickness == 2
        assert mark.confidence == 1.0

    def test_keypoints_mark_defaults(self):
        """KeypointsMark defaults."""
        mark = KeypointsMark(points=())
        assert mark.connections == ()
        assert mark.normalized is False
        assert mark.line_color == (255, 200, 100)
        assert mark.point_colors is None
        assert mark.point_radius == 3
        assert mark.min_confidence == 0.3

    def test_bar_mark_defaults(self):
        """BarMark defaults."""
        mark = BarMark(x=0.0, y=0.0, w=0.1, value=0.5)
        assert mark.color == (0, 255, 255)
        assert mark.height_px == 6

    def test_label_mark_defaults(self):
        """LabelMark defaults."""
        mark = LabelMark(text="test", x=0.0, y=0.0)
        assert mark.normalized is True
        assert mark.color == (255, 255, 255)
        assert mark.background is None
        assert mark.font_scale == 0.45

    def test_bbox_mark_equality(self):
        """Frozen marks support == comparison."""
        a = BBoxMark(x=0.1, y=0.2, w=0.3, h=0.4, label="ID:1")
        b = BBoxMark(x=0.1, y=0.2, w=0.3, h=0.4, label="ID:1")
        assert a == b

    def test_bbox_mark_inequality(self):
        """Different marks are not equal."""
        a = BBoxMark(x=0.1, y=0.2, w=0.3, h=0.4, label="ID:1")
        b = BBoxMark(x=0.1, y=0.2, w=0.3, h=0.4, label="ID:2")
        assert a != b


class TestDrawStyle:
    """Test DrawStyle."""

    def test_draw_style_defaults(self):
        """DrawStyle has sensible defaults."""
        style = DrawStyle()
        assert style.color is None
        assert style.thickness is None
        assert style.font_scale is None
        assert style.show_labels is True
        assert style.show_confidence is True
        assert style.extra is None

    def test_draw_style_frozen(self):
        """DrawStyle should be immutable."""
        style = DrawStyle()
        with pytest.raises(AttributeError):
            style.color = (255, 0, 0)  # type: ignore[misc]

    def test_draw_style_custom(self):
        """DrawStyle with custom values."""
        style = DrawStyle(
            color=(0, 255, 0),
            thickness=3,
            show_labels=False,
            extra={"role_colors": {"main": (0, 255, 0)}},
        )
        assert style.color == (0, 255, 0)
        assert style.thickness == 3
        assert style.show_labels is False
        assert style.extra["role_colors"]["main"] == (0, 255, 0)
