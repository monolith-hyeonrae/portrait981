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
from momentscan.visualize.components import (
    COLOR_RED_BGR,
    COLOR_GREEN_BGR,
    COLOR_WHITE_BGR,
    FONT,
    DebugLayer,
    LayerState,
    draw_horizontal_bar,
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
        from momentscan.algorithm.analyzers.face_classifier import FaceClassifierAnalyzer
        modules["face.classify"] = FaceClassifierAnalyzer()
    except ImportError:
        pass
    try:
        from vpx.face_au.analyzer import FaceAUAnalyzer
        modules["face.au"] = FaceAUAnalyzer(au_backend=None)
    except ImportError:
        pass
    try:
        from vpx.head_pose.analyzer import HeadPoseAnalyzer
        modules["head.pose"] = HeadPoseAnalyzer(pose_backend=None)
    except ImportError:
        pass
    try:
        from vpx.portrait_score.analyzer import PortraitScoreAnalyzer
        modules["portrait.score"] = PortraitScoreAnalyzer()
    except ImportError:
        pass
    try:
        from momentscan.algorithm.analyzers.face_quality import FaceQualityAnalyzer
        modules["face.quality"] = FaceQualityAnalyzer()
    except ImportError:
        pass
    return modules


# Layer mapping: module name -> DebugLayer
_LAYER_MAP = {
    "face.detect": DebugLayer.FACE,
    "face.expression": DebugLayer.FACE,
    "face.au": DebugLayer.FACE,
    "head.pose": DebugLayer.FACE,
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
        embed_stats: Optional[Dict[str, float]] = None,
    ) -> np.ndarray:
        """Draw all video annotations.

        Args:
            image: Video frame (will be copied).
            observations: Dict of module name -> Observation.
            roi: ROI boundary in normalized coordinates (x1, y1, x2, y2).
            layers: Layer visibility state. None means all layers enabled.
            fusion_result: Fusion result (for trigger flash).
            styles: Per-module DrawStyle overrides.
            embed_stats: Embedding stats from EmbedTracker.

        Returns:
            Annotated image.
        """
        output = image.copy()

        if roi is not None and (layers is None or layers[DebugLayer.ROI]):
            self._draw_roi(output, roi)

        # Draw order: background (portrait.score, face.quality) -> foreground (face)
        draw_order = [
            "portrait.score", "face.quality",
            "face.detect", "face.expression",
            "face.au", "head.pose",
            "face.classify",
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

        # Embedding quality indicator on main face
        if embed_stats and (layers is None or layers[DebugLayer.FACE]):
            self._draw_embed_quality(output, observations, embed_stats)

        if fusion_result is not None and fusion_result.should_trigger:
            if layers is None or layers[DebugLayer.TRIGGER]:
                self._draw_trigger_flash(output, fusion_result)

        return output

    # --- Embedding quality indicator ---

    def _draw_embed_quality(
        self,
        image: np.ndarray,
        observations: Optional[Dict[str, Observation]],
        embed_stats: Dict[str, float],
    ) -> None:
        """Draw shot quality indicators next to the main face bbox.

        Shows small horizontal bars below the face bbox:
        - face_identity: Green=high anchor similarity, Red=low
        - head_blur: Cyan bar (head crop sharpness, Laplacian variance)
        - scene_bg_separation: Green bar (head sharper than background ratio)
        """
        identity = embed_stats.get("face_identity", 0.0)
        head_blur = embed_stats.get("head_blur", 0.0)
        bg_sep = embed_stats.get("scene_bg_separation", 0.0)

        if identity <= 0 and head_blur <= 0 and bg_sep <= 0:
            return

        # Find main face bbox from face.detect
        face_obs = observations.get("face.detect") if observations else None
        if face_obs is None:
            return
        data = getattr(face_obs, "data", None)
        if data is None:
            return
        faces = getattr(data, "faces", None)
        if not faces:
            return

        face = max(faces, key=lambda f: getattr(f, "area_ratio", 0.0))
        h, w = image.shape[:2]
        bx, by, bw, bh = face.bbox
        px = int(bx * w)
        py = int((by + bh) * h) + 4  # Just below face bbox
        bar_w = max(30, int(bw * w))

        # Anchor update flash
        if embed_stats.get("anchor_updated", 0.0) > 0:
            anchor_y = int(by * h) - 8
            cv2.putText(
                image, "ANCHOR", (px, max(12, anchor_y)),
                FONT, 0.45, (0, 255, 255), 2,  # bright yellow BGR
            )

        # Identity bar (green-red gradient)
        if identity > 0:
            qr = int(255 * (1 - identity))
            qg = int(255 * identity)
            q_color = (0, qg, qr)  # BGR
            draw_horizontal_bar(image, px, py, bar_w, 5, identity, q_color)
            cv2.putText(
                image, f"ID:{identity:.2f}", (px + bar_w + 3, py + 5),
                FONT, 0.3, q_color, 1,
            )
            py += 8

        # Head blur bar (cyan) — Laplacian variance, scaled to 500
        if head_blur > 0:
            blr_color = (200, 200, 0)  # cyan-ish BGR
            draw_horizontal_bar(image, px, py, bar_w, 4, min(1.0, head_blur / 500.0), blr_color)
            cv2.putText(
                image, f"BLR:{head_blur:.0f}", (px + bar_w + 3, py + 4),
                FONT, 0.25, blr_color, 1,
            )
            py += 7

        # BG separation bar (green) — ratio head_blur/scene_blur, scaled to 3.0
        if bg_sep > 0:
            sep_color = (0, 200, 100)  # green-ish BGR
            draw_horizontal_bar(image, px, py, bar_w, 4, min(1.0, bg_sep / 3.0), sep_color)
            cv2.putText(
                image, f"SEP:{bg_sep:.2f}", (px + bar_w + 3, py + 4),
                FONT, 0.25, sep_color, 1,
            )

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
