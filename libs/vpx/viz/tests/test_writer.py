"""Tests for vpx-viz VideoSaver."""

import numpy as np
import pytest
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict


@dataclass
class _FakeObs:
    source: str = "mock.test"
    signals: Dict[str, Any] = field(
        default_factory=lambda: {"score": 0.5}
    )


class TestVideoSaver:
    def test_creates_file(self, tmp_path):
        from vpx.viz.writer import VideoSaver

        output = str(tmp_path / "out.avi")
        saver = VideoSaver(output, fps=10.0, width=320, height=240, codec="MJPG")

        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        saver.update(frame, {"mock.test": _FakeObs()})
        saver.update(frame, {"mock.test": _FakeObs()})
        saver.close()

        assert Path(output).exists()
        assert Path(output).stat().st_size > 0

    def test_handles_frame_with_data_attr(self, tmp_path):
        from vpx.viz.writer import VideoSaver

        output = str(tmp_path / "out.avi")
        saver = VideoSaver(output, fps=10.0, width=320, height=240, codec="MJPG")

        @dataclass
        class FakeFrame:
            data: np.ndarray = field(
                default_factory=lambda: np.zeros((240, 320, 3), dtype=np.uint8)
            )

        saver.update(FakeFrame(), {"mock.test": _FakeObs()})
        saver.close()

        assert Path(output).exists()

    def test_empty_observations(self, tmp_path):
        from vpx.viz.writer import VideoSaver

        output = str(tmp_path / "out.avi")
        saver = VideoSaver(output, fps=10.0, width=320, height=240, codec="MJPG")

        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        saver.update(frame, {})
        saver.close()

        assert Path(output).exists()
