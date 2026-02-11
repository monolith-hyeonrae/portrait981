"""Video panel — draws annotations directly on the video frame.

Uses Module.annotate() to produce declarative marks, then
render_marks() to draw them. App-specific overlays (ROI, trigger
flash) are drawn directly.
"""

from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from vpx.sdk import Observation
from vpx.viz.renderer import render_marks
from vpx.sdk.marks import DrawStyle
from facemoment.visualize.components import (
    COLOR_RED_BGR,
    FONT,
    DebugLayer,
    LayerState,
)


def _build_default_modules() -> Dict[str, object]:
    """Create lightweight module instances for annotate() only.

    These instances are not initialized (no backends loaded) — they
    exist solely so we can call module.annotate(obs).
    """
    modules: Dict[str, object] = {}
    try:
        from vpx.face_detect.analyzer import FaceDetectionAnalyzer
        modules["face.detect"] = FaceDetectionAnalyzer(face_backend=None)
    except ImportError:
        pass
    try:
        from vpx.face_expression.analyzer import ExpressionAnalyzer
        modules["face.expression"] = ExpressionAnalyzer(expression_backend=None)
    except ImportError:
        pass
    try:
        from vpx.body_pose.analyzer import PoseAnalyzer
        modules["body.pose"] = PoseAnalyzer(pose_backend=None)
    except ImportError:
        pass
    try:
        from vpx.hand_gesture.analyzer import GestureAnalyzer
        modules["hand.gesture"] = GestureAnalyzer(hand_backend=None)
    except ImportError:
        pass
    try:
        from facemoment.algorithm.analyzers.face_classifier import FaceClassifierAnalyzer
        modules["face.classify"] = FaceClassifierAnalyzer()
    except ImportError:
        pass
    return modules


# Layer mapping: module name -> DebugLayer
_LAYER_MAP = {
    "body.pose": DebugLayer.POSE,
    "hand.gesture": DebugLayer.GESTURE,
    "face.detect": DebugLayer.FACE,
    "face.expression": DebugLayer.FACE,
    "face.classify": DebugLayer.FACE,
}


class VideoPanel:
    """Draws spatial annotations on the video frame.

    Delegates to Module.annotate() for mark generation and
    render_marks() for rendering. Only ROI and trigger flash
    are drawn directly (app-specific overlays).

    Args:
        modules: Dict of module name -> Module instance (for annotate).
                 If None, creates lightweight default instances.
    """

    def __init__(self, modules: Optional[Dict[str, object]] = None):
        self._modules = modules if modules is not None else _build_default_modules()

    def draw(
        self,
        image: np.ndarray,
        observations: Optional[Dict[str, Observation]] = None,
        *,
        roi: Optional[Tuple[float, float, float, float]] = None,
        layers: Optional[LayerState] = None,
        fusion_result: Optional[Observation] = None,
        styles: Optional[Dict[str, DrawStyle]] = None,
    ) -> np.ndarray:
        """Draw all video annotations.

        Args:
            image: Video frame (will be copied).
            observations: Dict of module name -> Observation.
            roi: ROI boundary in normalized coordinates (x1, y1, x2, y2).
            layers: Layer visibility state. None means all layers enabled.
            fusion_result: Fusion result (for trigger flash).
            styles: Per-module DrawStyle overrides.

        Returns:
            Annotated image.
        """
        output = image.copy()

        if roi is not None and (layers is None or layers[DebugLayer.ROI]):
            self._draw_roi(output, roi)

        # Draw order: background (pose, gesture) -> foreground (face)
        draw_order = [
            "body.pose", "hand.gesture",
            "face.detect", "face.expression", "face.classify",
        ]

        for name in draw_order:
            obs = observations.get(name) if observations else None
            if obs is None:
                continue

            # Skip face.detect bbox when classifier is present (it draws its own)
            if name == "face.detect" and observations and observations.get("face.classify"):
                continue

            # Check layer visibility
            layer = _LAYER_MAP.get(name)
            if layer and layers and not layers[layer]:
                continue

            module = self._modules.get(name)
            if module:
                marks = module.annotate(obs)
                style = styles.get(name) if styles else None
                output = render_marks(output, marks, style=style)

        if fusion_result is not None and fusion_result.should_trigger:
            if layers is None or layers[DebugLayer.TRIGGER]:
                self._draw_trigger_flash(output, fusion_result)

        return output

    # --- ROI ---

    def _draw_roi(self, image: np.ndarray, roi: Tuple[float, float, float, float]) -> None:
        """Draw subtle ROI boundary."""
        h, w = image.shape[:2]
        x1, y1, x2, y2 = roi
        px1, py1 = int(x1 * w), int(y1 * h)
        px2, py2 = int(x2 * w), int(y2 * h)
        cv2.rectangle(image, (px1, py1), (px2, py2), (80, 80, 80), 1)

    # --- Trigger flash ---

    def _draw_trigger_flash(self, image: np.ndarray, result: Observation) -> None:
        """Draw trigger flash border and text."""
        h, w = image.shape[:2]
        cv2.rectangle(image, (0, 0), (w - 1, h - 1), COLOR_RED_BGR, 10)
        cv2.putText(
            image, f"TRIGGER: {result.trigger_reason}",
            (w // 2 - 100, h // 2),
            FONT, 1.0, COLOR_RED_BGR, 3,
        )
