"""Timeline panel — draws emotion graph, baseline, and trigger markers.

Renders a time-series graph at the bottom of the canvas showing:
- Raw happy value (yellow line)
- Baseline (gray dots)
- Spike area (green/red fill between baseline and current)
- Gate background (green/red tint)
- Trigger markers (red vertical lines)
"""

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from vpx.sdk import Observation
from momentscan.visualize.components import (
    COLOR_DARK_BGR,
    COLOR_WHITE_BGR,
    COLOR_GRAY_BGR,
    COLOR_RED_BGR,
    COLOR_GREEN_BGR,
    COLOR_HAPPY_BGR,
    FONT,
)


class TimelinePanel:
    """Time-series graph for the bottom panel.

    Tracks history of:
    - Raw happy values
    - Baseline (slow EWMA)
    - Spike (recent - baseline)
    - Gate open/closed state
    - Trigger timestamps
    """

    def __init__(self, max_history: int = 300):
        self._max_history = max_history
        self._happy_history: List[float] = []
        self._baseline_history: List[float] = []
        self._spike_history: List[float] = []
        self._gate_history: List[bool] = []
        self._trigger_times: List[int] = []
        self._anchor_times: List[int] = []
        # Embedding tracks
        self._face_identity_history: List[float] = []
        self._face_identity_history_q: List[float] = []  # alias kept for compat
        self._head_aesthetic_history: List[float] = []

    def reset(self) -> None:
        self._happy_history.clear()
        self._baseline_history.clear()
        self._spike_history.clear()
        self._gate_history.clear()
        self._trigger_times.clear()
        self._anchor_times.clear()
        self._face_identity_history.clear()
        self._face_identity_history_q.clear()
        self._head_aesthetic_history.clear()

    @property
    def trigger_count(self) -> int:
        return len(self._trigger_times)

    def update(
        self,
        face_obs: Optional[Observation],
        fusion_result: Optional[Observation],
        is_gate_open: bool,
        *,
        expression_obs: Optional[Observation] = None,
        embed_stats: Optional[Dict[str, float]] = None,
    ) -> None:
        """Record one frame's data into the history."""
        happy = 0.0
        if expression_obs is not None:
            happy = expression_obs.signals.get("expression_happy", 0)
        elif face_obs is not None:
            happy = face_obs.signals.get("expression_happy", 0)

        self._happy_history.append(happy)
        self._gate_history.append(is_gate_open)

        # Adaptive summary from fusion result
        adaptive_summary = None
        if fusion_result and fusion_result.metadata:
            adaptive_summary = fusion_result.metadata.get("adaptive_summary")

        if adaptive_summary and adaptive_summary.get("states"):
            first_state = next(iter(adaptive_summary["states"].values()), None)
            if first_state:
                self._baseline_history.append(first_state["baseline"])
                self._spike_history.append(first_state["spike"])
            else:
                self._baseline_history.append(happy)
                self._spike_history.append(0)
        else:
            self._baseline_history.append(happy)
            self._spike_history.append(0)

        # Embedding tracks
        if embed_stats:
            self._face_identity_history.append(embed_stats.get("face_identity", 0.0))
            self._face_identity_history_q.append(embed_stats.get("face_identity", 0.0))
            self._head_aesthetic_history.append(embed_stats.get("head_aesthetic", 0.0))
        else:
            self._face_identity_history.append(0.0)
            self._face_identity_history_q.append(0.0)
            self._head_aesthetic_history.append(0.0)

        # Anchor update marker
        if embed_stats and embed_stats.get("anchor_updated", 0.0) > 0:
            self._anchor_times.append(len(self._happy_history))

        # Trigger marker
        if fusion_result and fusion_result.should_trigger:
            self._trigger_times.append(len(self._happy_history))

        # Trim
        if len(self._happy_history) > self._max_history:
            self._happy_history.pop(0)
            self._baseline_history.pop(0)
            self._spike_history.pop(0)
            self._gate_history.pop(0)
            self._face_identity_history.pop(0)
            self._face_identity_history_q.pop(0)
            self._head_aesthetic_history.pop(0)
            self._trigger_times = [t - 1 for t in self._trigger_times if t > 1]
            self._anchor_times = [t - 1 for t in self._anchor_times if t > 1]

    def draw(
        self,
        canvas: np.ndarray,
        region: Tuple[int, int, int, int],
        threshold: float = 0.12,
    ) -> None:
        """Draw the timeline graph onto the canvas.

        Args:
            canvas: Full canvas to draw on.
            region: (x1, y1, x2, y2) for this panel.
            threshold: Spike threshold for coloring.
        """
        rx1, ry1, rx2, ry2 = region
        margin = 8
        x = rx1 + margin
        y = ry1 + margin
        width = rx2 - rx1 - margin * 2
        height = ry2 - ry1 - margin * 2

        if len(self._happy_history) < 2:
            cv2.putText(
                canvas, "Waiting for data...",
                (x + 10, y + height // 2),
                FONT, 0.35, COLOR_GRAY_BGR, 1,
            )
            return

        # Gate background
        for i, gate in enumerate(self._gate_history):
            px = x + int(i * width / self._max_history)
            cv2.line(
                canvas, (px, y), (px, y + height),
                (40, 60, 40) if gate else (60, 40, 40), 1,
            )

        # Threshold line
        th_y = y + height - int(threshold * height * 2)
        cv2.line(canvas, (x, th_y), (x + width, th_y), (100, 100, 100), 1)

        # Baseline (gray dots)
        if self._baseline_history:
            for i in range(0, len(self._baseline_history), 3):
                px = x + int(i * width / self._max_history)
                py = y + height - int(self._baseline_history[i] * height)
                cv2.circle(canvas, (px, py), 1, COLOR_GRAY_BGR, -1)

        # Raw happy line (yellow)
        pts = [
            (
                x + int(i * width / self._max_history),
                y + height - int(v * height),
            )
            for i, v in enumerate(self._happy_history)
        ]
        for i in range(1, len(pts)):
            cv2.line(canvas, pts[i - 1], pts[i], COLOR_HAPPY_BGR, 1)

        # Spike area
        if self._spike_history and self._baseline_history:
            for i in range(len(self._spike_history)):
                if i >= len(self._baseline_history):
                    break
                px = x + int(i * width / self._max_history)
                base_y = y + height - int(self._baseline_history[i] * height)
                spike = self._spike_history[i]
                if spike > 0:
                    spike_h = int(spike * height)
                    color = COLOR_RED_BGR if spike > threshold else (0, 180, 0)
                    cv2.line(canvas, (px, base_y), (px, base_y - spike_h), color, 1)

        # Embedding tracks
        # face_identity (green line, upper zone = good portrait)
        COLOR_ID = (0, 200, 100)  # green-ish
        if self._face_identity_history:
            pts_q = [
                (
                    x + int(i * width / self._max_history),
                    y + height - int(v * height),
                )
                for i, v in enumerate(self._face_identity_history)
            ]
            for i in range(1, len(pts_q)):
                if self._face_identity_history[i] > 0 or self._face_identity_history[i - 1] > 0:
                    cv2.line(canvas, pts_q[i - 1], pts_q[i], COLOR_ID, 1)

        # head_aesthetic (orange line, CLIP portrait quality [0,1])
        COLOR_AES = (0, 140, 255)  # orange BGR
        if self._head_aesthetic_history:
            pts_aes = [
                (
                    x + int(i * width / self._max_history),
                    y + height - int(v * height),
                )
                for i, v in enumerate(self._head_aesthetic_history)
            ]
            for i in range(1, len(pts_aes)):
                if self._head_aesthetic_history[i] > 0 or self._head_aesthetic_history[i - 1] > 0:
                    cv2.line(canvas, pts_aes[i - 1], pts_aes[i], COLOR_AES, 1)

        # Trigger markers
        for t in self._trigger_times:
            if t < len(self._happy_history):
                px = x + int(t * width / self._max_history)
                cv2.line(canvas, (px, y), (px, y + height), COLOR_RED_BGR, 2)

        # Anchor update markers (yellow inverted triangle at top)
        COLOR_ANCHOR = (0, 255, 255)  # yellow BGR
        for t in self._anchor_times:
            if t < len(self._happy_history):
                px = x + int(t * width / self._max_history)
                cv2.drawMarker(
                    canvas, (px, y + 4), COLOR_ANCHOR,
                    cv2.MARKER_TRIANGLE_DOWN, 6, 1,
                )

        # Legend
        cv2.putText(canvas, "Happy", (x + 5, y + 12), FONT, 0.3, COLOR_HAPPY_BGR, 1)
        cv2.putText(canvas, "ID", (x + 45, y + 12), FONT, 0.3, COLOR_ID, 1)
        cv2.putText(canvas, "Aes", (x + 60, y + 12), FONT, 0.3, COLOR_AES, 1)
        cv2.putText(canvas, "base", (x + 85, y + 12), FONT, 0.3, COLOR_GRAY_BGR, 1)
        cv2.putText(
            canvas, f"th={threshold:.2f}", (x + width - 50, y + 12),
            FONT, 0.25, COLOR_GRAY_BGR, 1,
        )

        # Frame info on right side
        cv2.putText(
            canvas, f"{len(self._happy_history)} frames",
            (x + width - 60, y + height - 4),
            FONT, 0.25, COLOR_GRAY_BGR, 1,
        )
