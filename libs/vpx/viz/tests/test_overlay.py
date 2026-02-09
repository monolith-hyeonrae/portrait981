"""Tests for vpx-viz overlay module."""

import numpy as np
import pytest
from dataclasses import dataclass, field
from typing import Any, Dict


class TestTextOverlay:
    def test_draw_modifies_frame(self):
        from vpx.viz.overlay import TextOverlay

        overlay = TextOverlay()
        frame = np.zeros((100, 200, 3), dtype=np.uint8)
        original = frame.copy()

        @dataclass
        class FakeObs:
            source: str = "mock.test"
            signals: Dict[str, Any] = field(
                default_factory=lambda: {"score": 0.95, "count": 3}
            )

        result = overlay.draw(frame, FakeObs())

        # Frame should be modified (text drawn)
        assert not np.array_equal(result, original)

    def test_draw_returns_frame(self):
        from vpx.viz.overlay import TextOverlay

        overlay = TextOverlay()
        frame = np.zeros((100, 200, 3), dtype=np.uint8)

        @dataclass
        class FakeObs:
            source: str = "test"
            signals: Dict[str, Any] = field(default_factory=dict)

        result = overlay.draw(frame, FakeObs())
        assert isinstance(result, np.ndarray)
        assert result.shape == (100, 200, 3)

    def test_custom_params(self):
        from vpx.viz.overlay import TextOverlay

        overlay = TextOverlay(
            y_offset=50,
            font_scale=0.8,
            color=(0, 255, 0),
            thickness=2,
        )
        assert overlay._y_offset == 50
        assert overlay._font_scale == 0.8

    def test_line_height(self):
        from vpx.viz.overlay import TextOverlay

        overlay = TextOverlay(font_scale=0.45)
        assert overlay.line_height == 20

        overlay2 = TextOverlay(font_scale=0.9)
        assert overlay2.line_height == 40

    def test_float_formatting(self):
        """Float signals use 3 decimal places."""
        from vpx.viz.overlay import TextOverlay

        overlay = TextOverlay()
        frame = np.zeros((200, 400, 3), dtype=np.uint8)

        @dataclass
        class FakeObs:
            source: str = "test"
            signals: Dict[str, Any] = field(
                default_factory=lambda: {"score": 0.123456}
            )

        # Should not raise
        result = overlay.draw(frame, FakeObs())
        assert isinstance(result, np.ndarray)


class TestOverlayProtocol:
    def test_text_overlay_is_overlay(self):
        from vpx.viz.overlay import Overlay, TextOverlay

        overlay = TextOverlay()
        assert isinstance(overlay, Overlay)

    def test_custom_overlay_protocol(self):
        from vpx.viz.overlay import Overlay

        class CustomOverlay:
            def draw(self, frame, obs):
                return frame

        assert isinstance(CustomOverlay(), Overlay)
