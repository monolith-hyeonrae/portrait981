"""Video panel — draws annotations directly on the video frame.

Uses Module.annotate() to produce declarative marks, then
render_marks() to draw them. App-specific overlays (ROI, trigger
flash) are drawn directly.
"""

from dataclasses import replace
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from vpx.sdk import Observation
from vpx.sdk.marks import AxisMark, BarMark, BBoxMark, DrawStyle, KeypointsMark, LabelMark
from vpx.viz.renderer import render_marks
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
        from vpx.face_parse.analyzer import FaceParseAnalyzer
        modules["face.parse"] = FaceParseAnalyzer(parse_backend=None)
    except ImportError:
        pass
    try:
        from momentscan.algorithm.analyzers.face_quality import FaceQualityAnalyzer
        modules["face.quality"] = FaceQualityAnalyzer()
    except ImportError:
        pass
    try:
        from momentscan.algorithm.analyzers.face_gate import FaceGateAnalyzer
        modules["face.gate"] = FaceGateAnalyzer()
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
    "face.gate": DebugLayer.FACE,
    "face.parse": DebugLayer.FACE,
}


# seg_face: contour only (no fill). Classes matching _SEG_GROUPS["face"].
_FACE_CLASSES: frozenset[int] = frozenset({1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14})

# Highlighted sub-regions: color fill (BGR)
_HIGHLIGHT_GROUPS: dict[str, tuple[frozenset[int], tuple[int, int, int]]] = {
    "mouth": (frozenset({11, 12, 13}), (50, 50, 255)),    # red
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

        # Portrait clip overlay (ROI-style darkened outside)
        if observations and (layers is None or layers[DebugLayer.FACE]):
            self._draw_portrait_clip(output, observations)

        # Face parse mask overlay (background — rendered below marks)
        if (observations and observations.get("face.parse")
                and (layers is None or layers[DebugLayer.FACE])):
            self._draw_face_parse_overlay(output, observations)

        # Draw order: background (portrait.score, face.quality) -> foreground (face)
        draw_order = [
            "portrait.score", "face.quality", "face.parse",
            "face.detect", "face.expression",
            "face.au", "head.pose",
            "face.classify", "face.gate",
        ]

        active_bboxes = self._get_active_bboxes(observations)
        emphasized_bboxes = (
            self._get_emphasized_bboxes(active_bboxes, observations, roi)
            if active_bboxes is not None else None
        )

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

                # portrait.score BBoxMark drawn separately as ROI-style overlay
                if name == "portrait.score":
                    marks = [m for m in marks if not isinstance(m, BBoxMark)]

                # 3-level mark treatment (skip face.classify — has own role colors)
                # emphasized: thick + bright | filtered: thin + bright | non-selected: thin + dim
                if active_bboxes is not None and name != "face.classify":
                    styled = []
                    for m in marks:
                        if emphasized_bboxes and self._is_near_bboxes(m, emphasized_bboxes):
                            styled.append(m)  # full emphasis
                        elif self._is_near_bboxes(m, active_bboxes):
                            styled.append(self._thin_mark(m))  # gate/ROI filtered
                        else:
                            styled.append(self._dim_mark(m))  # non-selected
                    marks = styled

                style = styles.get(name) if styles else None
                output = render_marks(output, marks, style=style)

        # Embedding quality indicator on main face
        if embed_stats and (layers is None or layers[DebugLayer.FACE]):
            self._draw_embed_quality(output, observations, embed_stats)

        if fusion_result is not None and fusion_result.should_trigger:
            if layers is None or layers[DebugLayer.TRIGGER]:
                self._draw_trigger_flash(output, fusion_result)

        return output

    # --- Face dimming helpers ---

    def _get_active_bboxes(
        self, observations: Optional[Dict[str, Observation]],
    ) -> Optional[List[Tuple[float, float, float, float]]]:
        """Extract main+passenger face bboxes from face.classify."""
        if not observations:
            return None
        classify_obs = observations.get("face.classify")
        if not classify_obs:
            return None
        data = getattr(classify_obs, "data", None)
        if data is None:
            return None
        faces = getattr(data, "faces", None)
        if not faces:
            return None
        active = []
        for cf in faces:
            if getattr(cf, "role", "") in ("main", "passenger"):
                face = getattr(cf, "face", None)
                if face is not None:
                    active.append(getattr(face, "bbox", (0, 0, 0, 0)))
        return active if active else None

    def _get_emphasized_bboxes(
        self,
        active_bboxes: List[Tuple[float, float, float, float]],
        observations: Optional[Dict[str, Observation]],
        roi: Optional[Tuple[float, float, float, float]],
    ) -> List[Tuple[float, float, float, float]]:
        """Filter active bboxes to only those passing gate AND inside ROI.

        Returns the subset of active_bboxes that should get full emphasis
        (thick lines, bright colors). Others are "filtered" (thin lines only).
        """
        # Collect gate-passed face bboxes
        gate_passed_set: Optional[set] = None
        gate_obs = observations.get("face.gate") if observations else None
        if gate_obs is not None:
            gate_data = getattr(gate_obs, "data", None)
            if gate_data is not None:
                results = getattr(gate_data, "results", [])
                if results:
                    gate_passed_set = set()
                    for r in results:
                        if r.gate_passed:
                            gate_passed_set.add(r.face_bbox)

        emphasized = []
        for bbox in active_bboxes:
            # Gate check: if gate data exists, bbox must be in passed set
            if gate_passed_set is not None and bbox not in gate_passed_set:
                continue
            # ROI check: bbox center must be inside ROI
            if roi is not None:
                rx1, ry1, rx2, ry2 = roi
                cx = bbox[0] + bbox[2] / 2
                cy = bbox[1] + bbox[3] / 2
                if not (rx1 <= cx <= rx2 and ry1 <= cy <= ry2):
                    continue
            emphasized.append(bbox)
        return emphasized

    def _is_near_bboxes(
        self,
        mark: object,
        bboxes: List[Tuple[float, float, float, float]],
        margin: float = 0.05,
    ) -> bool:
        """Check if a mark's position falls within any of the given bboxes."""
        if isinstance(mark, BBoxMark):
            mx = mark.x + mark.w / 2
            my = mark.y + mark.h / 2
        elif isinstance(mark, BarMark):
            mx, my = mark.x, mark.y
        elif isinstance(mark, LabelMark):
            mx, my = mark.x, mark.y
        elif isinstance(mark, AxisMark):
            mx, my = mark.cx, mark.cy
        elif isinstance(mark, KeypointsMark):
            return True  # body keypoints — always active
        else:
            return True  # unknown type — treat as active

        for bx, by, bw, bh in bboxes:
            if (bx - margin <= mx <= bx + bw + margin
                    and by - margin <= my <= by + bh + margin):
                return True
        return False

    def _thin_mark(self, mark: object) -> object:
        """Return a copy with reduced thickness only (keep original colors)."""
        if isinstance(mark, BBoxMark):
            return replace(mark, thickness=1)
        elif isinstance(mark, AxisMark):
            return replace(mark, thickness=1)
        return mark

    def _dim_mark(self, mark: object) -> object:
        """Return a dimmed copy of the mark (40% brightness, thinner lines)."""
        def dim_color(c: Optional[Tuple[int, int, int]]) -> Tuple[int, int, int]:
            if c is None:
                return (50, 50, 50)
            return (int(c[0] * 0.4), int(c[1] * 0.4), int(c[2] * 0.4))

        if isinstance(mark, BBoxMark):
            return replace(mark, color=dim_color(mark.color), thickness=1)
        elif isinstance(mark, BarMark):
            return replace(mark, color=dim_color(mark.color))
        elif isinstance(mark, LabelMark):
            return replace(mark, color=dim_color(mark.color))
        elif isinstance(mark, AxisMark):
            return replace(mark, thickness=1)
        return mark

    # --- Face parse mask overlay ---

    def _draw_face_parse_overlay(
        self,
        image: np.ndarray,
        observations: Dict[str, Observation],
    ) -> None:
        """Draw face.parse results as per-class colored overlay + contour.

        Uses class_map (19-class) when available for semantic coloring,
        falls back to binary face_mask (single cyan) otherwise.
        """
        parse_obs = observations.get("face.parse")
        if parse_obs is None or parse_obs.data is None:
            return

        h, w = image.shape[:2]
        alpha = 0.35
        group_ratios: dict[str, float] = {}  # name → coverage ratio
        legend_anchor: Optional[Tuple[int, int]] = None  # (x, y) for first face

        for i, result in enumerate(parse_obs.data.results):
            if result.crop_box == (0, 0, 0, 0):
                continue
            bx, by, bw, bh = result.crop_box
            if bw <= 0 or bh <= 0:
                continue

            # Frame boundary clipping
            x1, y1 = max(0, bx), max(0, by)
            x2, y2 = min(w, bx + bw), min(h, by + bh)
            if x2 <= x1 or y2 <= y1:
                continue

            mx1, my1 = x1 - bx, y1 - by
            mx2, my2 = mx1 + (x2 - x1), my1 + (y2 - y1)

            roi = image[y1:y2, x1:x2]

            # Branch: class_map available → per-class coloring
            has_class_map = (
                result.class_map is not None
                and result.class_map.size > 1
            )

            if has_class_map:
                class_map_resized = cv2.resize(
                    result.class_map, (bw, bh),
                    interpolation=cv2.INTER_NEAREST,
                )
                region_map = class_map_resized[my1:my2, mx1:mx2]

                # Color fill: mouth sub-region
                color_overlay = np.zeros_like(roi)
                for name, (classes, color) in _HIGHLIGHT_GROUPS.items():
                    group_mask = np.isin(region_map, list(classes))
                    if group_mask.any():
                        color_overlay[group_mask] = color

                fill_mask = color_overlay.any(axis=2)
                if fill_mask.any():
                    roi[fill_mask] = (
                        roi[fill_mask].astype(np.float32) * (1 - alpha)
                        + color_overlay[fill_mask].astype(np.float32) * alpha
                    ).astype(np.uint8)

                # Contour: face region boundary
                face_binary = np.isin(region_map, list(_FACE_CLASSES)).astype(np.uint8) * 255
                contours, _ = cv2.findContours(
                    face_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
                )
                for c in contours:
                    c[:, :, 0] += x1
                    c[:, :, 1] += y1
                    cv2.drawContours(image, [c], -1, (0, 255, 0), 1)

                # Record legend anchor from the first face crop
                if i == 0:
                    legend_anchor = (min(w - 1, bx + bw + 4), max(0, by))

            else:
                # Fallback: binary face_mask → single cyan
                face_mask_resized = cv2.resize(
                    result.face_mask, (bw, bh),
                    interpolation=cv2.INTER_NEAREST,
                )
                face_region = face_mask_resized[my1:my2, mx1:mx2]
                face_pixels = face_region > 0
                if face_pixels.any():
                    face_color = np.array([255, 255, 0], dtype=np.uint8)
                    roi[face_pixels] = (
                        roi[face_pixels].astype(np.float32) * (1 - alpha)
                        + face_color * alpha
                    ).astype(np.uint8)

                contours, _ = cv2.findContours(
                    face_region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
                )
                for c in contours:
                    c[:, :, 0] += x1
                    c[:, :, 1] += y1
                    cv2.drawContours(image, [c], -1, (0, 255, 255), 1)

        # Read coverage ratios from face.quality observation
        fq_obs = observations.get("face.quality")
        fq_data = getattr(fq_obs, "data", None) if fq_obs else None
        if fq_data is not None:
            group_ratios["face"] = getattr(fq_data, "seg_face", 0.0)
            group_ratios["mouth"] = getattr(fq_data, "seg_mouth", 0.0)

        # Legend — groups with coverage, next to first face crop
        if group_ratios and legend_anchor is not None:
            self._draw_parse_legend(image, legend_anchor, group_ratios)

    def _draw_parse_legend(
        self,
        image: np.ndarray,
        anchor: Tuple[int, int],
        group_ratios: dict[str, float],
    ) -> None:
        """Draw small color legend with coverage ratios for parse groups."""
        h, w = image.shape[:2]

        # face contour + highlight groups
        entries: list[tuple[str, tuple[int, int, int], float]] = []
        if "face" in group_ratios:
            entries.append(("face", (0, 255, 0), group_ratios["face"]))
        for name, (_, color) in _HIGHLIGHT_GROUPS.items():
            if name in group_ratios:
                entries.append((name, color, group_ratios[name]))

        if not entries:
            return

        lx, ly = anchor
        row_h = 14
        total_h = len(entries) * row_h + 4
        box_w = 90

        # Clamp to frame bounds
        if lx + box_w > w:
            lx = w - box_w - 2
        if ly + total_h > h:
            ly = h - total_h - 2
        lx = max(0, lx)
        ly = max(0, ly)

        # Semi-transparent background
        overlay_region = image[ly:ly + total_h, lx:lx + box_w].copy()
        cv2.rectangle(image, (lx, ly), (lx + box_w, ly + total_h), (0, 0, 0), -1)
        image[ly:ly + total_h, lx:lx + box_w] = (
            overlay_region.astype(np.float32) * 0.4
            + image[ly:ly + total_h, lx:lx + box_w].astype(np.float32) * 0.6
        ).astype(np.uint8)

        # Draw entries
        cy = ly + 2
        for name, color, ratio in entries:
            cv2.rectangle(image, (lx + 2, cy + 1), (lx + 12, cy + 11), color, -1)
            label = f"{name} {ratio:.0%}"
            cv2.putText(
                image, label, (lx + 15, cy + 10),
                FONT, 0.32, (220, 220, 220), 1,
            )
            cy += row_h

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
        py = int((by + bh) * h) + 100  # Zone 3: below mark-based overlays
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
        """Draw ROI as a simple line rectangle."""
        h, w = image.shape[:2]
        x1, y1, x2, y2 = roi
        px1, py1 = int(x1 * w), int(y1 * h)
        px2, py2 = int(x2 * w), int(y2 * h)
        cv2.rectangle(image, (px1, py1), (px2, py2), (60, 60, 60), 1)

    # --- Portrait clip (ROI-style) ---

    def _draw_portrait_clip(
        self, image: np.ndarray, observations: Dict[str, Observation],
    ) -> None:
        """Draw portrait.score crop box with darkened outside."""
        obs = observations.get("portrait.score")
        if obs is None or obs.data is None:
            return
        data = obs.data
        box = getattr(data, "portrait_crop_box", None)
        if box is None:
            return
        img_w, img_h = data.image_size if data.image_size else (1, 1)
        x, y, w_box, h_box = box
        h, w = image.shape[:2]
        px1 = int(x / img_w * w)
        py1 = int(y / img_h * h)
        px2 = int((x + w_box) / img_w * w)
        py2 = int((y + h_box) / img_h * h)

        # Dim outside portrait clip
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[:py1, :] = 1
        mask[py2:, :] = 1
        mask[py1:py2, :px1] = 1
        mask[py1:py2, px2:] = 1
        image[mask == 1] = (image[mask == 1] * 0.4).astype(np.uint8)

        # Thin border + label
        color = (200, 200, 255)
        cv2.rectangle(image, (px1, py1), (px2, py2), color, 1)
        label = f"portrait clip={data.head_aesthetic:.2f}"
        cv2.putText(image, label, (px1, py1 - 4), FONT, 0.35, color, 1)

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
