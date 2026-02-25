"""Face quality analyzer — face crop blur and exposure assessment.

FaceQualityAnalyzer: depends on face.detect, optional_depends on face.parse.
Computes quality metrics from a tight face crop using a 3-level mask fallback:
  1. face.parse (BiSeNet) — pixel-precise face skin mask
  2. Landmark ellipse (5-point) — geometric approximation
  3. Center patch (50%) — last resort

Metrics:
  - head_blur: Laplacian variance within mask (face sharpness)
  - head_exposure: mean brightness within mask
  - head_contrast: CV = std/mean (skin-tone invariant exposure quality)
  - clipped_ratio: overexposed (>250) pixel ratio
  - crushed_ratio: underexposed (<5) pixel ratio
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional, Dict, List

import cv2
import numpy as np

if TYPE_CHECKING:
    from visualbase import Frame

from vpx.sdk import (
    Module,
    Observation,
    ProcessingStep,
    processing_step,
    get_processing_steps,
    Capability,
    ModuleCapabilities,
)
from vpx.sdk.crop import face_crop, BBoxSmoother
from momentscan.algorithm.analyzers.face_quality.output import FaceQualityOutput, FaceQualityResult

logger = logging.getLogger(__name__)

# Minimum mask coverage fraction to accept (5% of crop area)
_MIN_MASK_COVERAGE = 0.05


# ── Mask construction helpers ──


def _align_parse_to_crop(
    mask: np.ndarray,
    parse_crop_box: tuple,
    head_box: tuple,
    crop_shape: tuple[int, int],
) -> Optional[np.ndarray]:
    """Transform an arbitrary mask/class_map from parse crop space to head crop space.

    Both parse_crop_box and head_box are in original image pixel coordinates.
    The head crop image is resized to crop_shape, so we must scale coordinates
    from original image space into crop_shape space.

    Args:
        mask: 2D array (H, W) in parse crop coordinates (e.g. face_mask or class_map).
        parse_crop_box: (x, y, w, h) of the parse crop in image pixels.
        head_box: (x, y, w, h) of the face.quality head crop in image pixels.
        crop_shape: (height, width) of the face.quality crop (resized).

    Returns:
        Aligned array (crop_shape) with same dtype, or None if no overlap.
    """
    px, py, pw, ph = parse_crop_box
    hx, hy, hw, hh = head_box

    if pw <= 0 or ph <= 0 or hw <= 0 or hh <= 0:
        return None

    # Compute overlap region in image coordinates
    ox1 = max(px, hx)
    oy1 = max(py, hy)
    ox2 = min(px + pw, hx + hw)
    oy2 = min(py + ph, hy + hh)

    if ox2 <= ox1 or oy2 <= oy1:
        return None

    # Scale factors: original image coords → crop image coords
    sx = crop_shape[1] / hw
    sy = crop_shape[0] / hh

    # Resize parse mask from model output size to actual parse crop size
    parse_h, parse_w = mask.shape[:2]
    if (parse_h, parse_w) != (ph, pw):
        resized_parse = cv2.resize(
            mask, (pw, ph), interpolation=cv2.INTER_NEAREST
        )
    else:
        resized_parse = mask

    # Extract overlap region from resized parse mask (parse crop coords)
    src_x1 = ox1 - px
    src_y1 = oy1 - py
    src_w = ox2 - ox1
    src_h = oy2 - oy1
    overlap = resized_parse[src_y1:src_y1 + src_h, src_x1:src_x1 + src_w]

    # Destination in crop_shape coords (scaled from image coords)
    dst_x1 = int(round((ox1 - hx) * sx))
    dst_y1 = int(round((oy1 - hy) * sy))
    dst_x2 = int(round((ox2 - hx) * sx))
    dst_y2 = int(round((oy2 - hy) * sy))

    # Clip to crop bounds
    dst_x1 = max(0, min(dst_x1, crop_shape[1]))
    dst_y1 = max(0, min(dst_y1, crop_shape[0]))
    dst_x2 = max(0, min(dst_x2, crop_shape[1]))
    dst_y2 = max(0, min(dst_y2, crop_shape[0]))

    if dst_x2 <= dst_x1 or dst_y2 <= dst_y1:
        return None

    # Resize overlap to match scaled destination size
    dst_w = dst_x2 - dst_x1
    dst_h = dst_y2 - dst_y1
    if overlap.shape[1] != dst_w or overlap.shape[0] != dst_h:
        overlap = cv2.resize(overlap, (dst_w, dst_h), interpolation=cv2.INTER_NEAREST)

    result = np.zeros(crop_shape, dtype=mask.dtype)
    result[dst_y1:dst_y2, dst_x1:dst_x2] = overlap

    return result


def _transform_parse_mask(
    parse_result, head_box: tuple, crop_shape: tuple[int, int]
) -> Optional[np.ndarray]:
    """Transform face.parse binary mask into face.quality crop coordinate space.

    Thin wrapper around _align_parse_to_crop for backward compatibility.
    """
    return _align_parse_to_crop(
        parse_result.face_mask, parse_result.crop_box, head_box, crop_shape,
    )


_SEG_GROUPS: dict[str, frozenset[int]] = {
    "face": frozenset({1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14}),  # skin+brow+eye+glasses+ear+nose+mouth+lip+neck
    "eye": frozenset({4, 5}),
    "mouth": frozenset({11, 12, 13}),
    "hair": frozenset({17}),
}
def _compute_semantic_ratios(
    class_map: np.ndarray,
    region: tuple[int, int, int, int] | None = None,
) -> dict[str, float]:
    """Compute per-group pixel ratios from a class_map.

    Args:
        class_map: 2D uint8 array with class indices [0..18].
        region: (x, y, w, h) sub-region in class_map coordinates.
                If None, use the full map.

    Returns:
        Dict with keys: seg_face, seg_eye, seg_mouth, seg_hair, eye_pixel_ratio.
    """
    if region is not None:
        rx, ry, rw, rh = region
        mh, mw = class_map.shape[:2]
        rx = max(0, rx)
        ry = max(0, ry)
        rw = min(rw, mw - rx)
        rh = min(rh, mh - ry)
        if rw <= 0 or rh <= 0:
            return {f"seg_{k}": 0.0 for k in _SEG_GROUPS} | {"eye_pixel_ratio": 0.0}
        sub = class_map[ry:ry + rh, rx:rx + rw]
    else:
        sub = class_map

    total = sub.size
    if total == 0:
        return {f"seg_{k}": 0.0 for k in _SEG_GROUPS} | {"eye_pixel_ratio": 0.0}

    ratios: dict[str, float] = {}
    for name, classes in _SEG_GROUPS.items():
        count = sum(int(np.count_nonzero(sub == c)) for c in classes)
        ratios[f"seg_{name}"] = count / total

    ratios["eye_pixel_ratio"] = ratios["seg_eye"]
    return ratios


def _landmark_ellipse_mask(
    gray: np.ndarray,
    landmarks: np.ndarray,
    head_box: tuple,
    crop_shape: tuple[int, int],
) -> np.ndarray:
    """Create an ellipse mask from 5-point face landmarks.

    5 points: [left_eye, right_eye, nose, left_mouth, right_mouth].
    Ellipse: center = midpoint of eyes+mouth, axes = (IOD*0.85, eye_mouth*1.3).

    Args:
        gray: Grayscale crop image (for shape reference).
        landmarks: (5, 2) array of landmark pixel coordinates in image space.
        head_box: face.quality crop box (x, y, w, h) in image pixels.
        crop_shape: (height, width) of the crop.

    Returns:
        Binary mask (crop_shape) uint8.
    """
    hx, hy, _, _ = head_box

    # Transform landmarks from image coords to crop coords
    pts = landmarks[:, :2].astype(np.float64)
    pts[:, 0] -= hx
    pts[:, 1] -= hy

    # Key points
    left_eye = pts[0]
    right_eye = pts[1]
    left_mouth = pts[3]
    right_mouth = pts[4]

    # Inter-ocular distance
    iod = np.linalg.norm(right_eye - left_eye)
    if iod < 1.0:
        return np.zeros(crop_shape, dtype=np.uint8)

    # Eye-mouth vertical distance
    eye_mid = (left_eye + right_eye) / 2
    mouth_mid = (left_mouth + right_mouth) / 2
    eye_mouth_dist = np.linalg.norm(mouth_mid - eye_mid)

    # Ellipse center: midpoint between eye center and mouth center
    center = ((eye_mid + mouth_mid) / 2).astype(int)

    # Ellipse axes
    ax_x = int(iod * 0.85)
    ax_y = int(eye_mouth_dist * 1.3)

    # Angle from eye line
    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]
    angle = float(np.degrees(np.arctan2(dy, dx)))

    mask = np.zeros(crop_shape, dtype=np.uint8)
    cv2.ellipse(mask, tuple(center), (ax_x, ax_y), angle, 0, 360, 255, -1)

    return mask


def _center_patch_mask(crop_shape: tuple[int, int]) -> np.ndarray:
    """Create a center 50% rectangular patch mask."""
    h, w = crop_shape
    y1, y2 = h // 4, h * 3 // 4
    x1, x2 = w // 4, w * 3 // 4
    mask = np.zeros(crop_shape, dtype=np.uint8)
    mask[y1:y2, x1:x2] = 255
    return mask


def _compute_face_mask(
    head_gray: np.ndarray,
    parse_result,
    landmarks: Optional[np.ndarray],
    head_box: tuple,
) -> tuple[np.ndarray, str, float]:
    """3-level fallback mask: parsing → landmark ellipse → center patch.

    Returns:
        (mask, method, parsing_coverage) — binary mask (H, W) uint8,
        method name string, and face.parse coverage ratio (0 if not attempted).
    """
    crop_shape = head_gray.shape[:2]
    crop_area = crop_shape[0] * crop_shape[1]
    min_pixels = int(crop_area * _MIN_MASK_COVERAGE)
    parsing_coverage = 0.0  # 0 = parsing 미시도

    # Level 1: Face parsing (BiSeNet)
    if parse_result is not None:
        mask = _transform_parse_mask(parse_result, head_box, crop_shape)
        if mask is not None:
            parsing_coverage = float(np.count_nonzero(mask)) / crop_area if crop_area > 0 else 0.0
            if np.count_nonzero(mask) > min_pixels:
                return mask, "parsing", parsing_coverage
            # parsing 시도했지만 coverage 부족 → coverage 기록 후 fallback

    # Level 2: Landmark ellipse (5-point)
    if landmarks is not None and len(landmarks) >= 5:
        mask = _landmark_ellipse_mask(head_gray, landmarks, head_box, crop_shape)
        if np.count_nonzero(mask) > min_pixels:
            return mask, "landmark", parsing_coverage

    # Level 3: Center patch (50%)
    return _center_patch_mask(crop_shape), "center_patch", parsing_coverage


# ── Metric functions ──


def _laplacian_variance_masked(gray: np.ndarray, mask: np.ndarray) -> float:
    """Laplacian variance within mask region — sharpness metric."""
    if gray.size == 0:
        return 0.0
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    masked_pixels = lap[mask > 0]
    if masked_pixels.size == 0:
        return 0.0
    return float(masked_pixels.var())


def _local_contrast(gray: np.ndarray, mask: np.ndarray) -> float:
    """CV = std/mean. Skin-tone invariant exposure quality metric."""
    pixels = gray[mask > 0].astype(np.float64)
    if pixels.size == 0:
        return 0.0
    mean = pixels.mean()
    if mean < 1.0:
        return 0.0
    return float(pixels.std() / mean)


def _exposure_stats(gray: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    """Compute clipped (>250) and crushed (<5) pixel ratios within mask."""
    pixels = gray[mask > 0]
    n = max(pixels.size, 1)
    clipped = float(np.count_nonzero(pixels > 250) / n)
    crushed = float(np.count_nonzero(pixels < 5) / n)
    return clipped, crushed


# ── DetectedFace matching ──


def _find_detected_face(face_obs, detected_faces, img_w: int, img_h: int):
    """Match FaceObservation (normalized bbox) to DetectedFace (pixel bbox).

    Uses center-distance matching with 10% threshold.
    """
    if not detected_faces:
        return None

    nx, ny, nw, nh = face_obs.bbox
    px_cx = (nx + nw / 2) * img_w
    px_cy = (ny + nh / 2) * img_h

    best, best_dist = None, float("inf")
    for df in detected_faces:
        dx, dy, dw, dh = df.bbox
        d = abs(dx + dw / 2 - px_cx) + abs(dy + dh / 2 - px_cy)
        if d < best_dist:
            best, best_dist = df, d

    threshold = max(img_w, img_h) * 0.1
    return best if best_dist < threshold else None


class FaceQualityAnalyzer(Module):
    """Analyzer that assesses blur and exposure of the face region.

    Uses a 3-level mask fallback (face.parse → landmark ellipse → center patch)
    to measure quality on face-only pixels, excluding background/hair/clothing.

    depends: ["face.detect"]
    optional_depends: ["face.parse"]
    """

    depends = ["face.detect"]
    optional_depends = ["face.parse"]

    def __init__(
        self,
        face_expand: float = 1.1,
        smooth_alpha: float = 0.3,
        crop_ratio: str = "4:5",
    ):
        self._face_expand = face_expand
        self._crop_ratio = crop_ratio
        self._head_smoother = BBoxSmoother(alpha=smooth_alpha)
        self._initialized = False
        self._step_timings: Optional[Dict[str, float]] = None

    @property
    def name(self) -> str:
        return "face.quality"

    @property
    def capabilities(self) -> ModuleCapabilities:
        return ModuleCapabilities(
            flags=Capability.STATEFUL,
            init_time_sec=0.01,
        )

    @property
    def processing_steps(self) -> List[ProcessingStep]:
        return get_processing_steps(self)

    def initialize(self) -> None:
        if self._initialized:
            return
        self._initialized = True
        logger.info("FaceQualityAnalyzer initialized")

    def cleanup(self) -> None:
        self._head_smoother.reset()
        self._initialized = False
        logger.info("FaceQualityAnalyzer cleaned up")

    def reset(self) -> None:
        self._head_smoother.reset()

    @processing_step(
        name="face_quality",
        description="Compute face crop quality metrics (blur, exposure, contrast) with mask fallback",
        backend="OpenCV",
        input_type="Frame + FaceDetectOutput + FaceParseOutput(optional)",
        output_type="FaceQualityOutput",
    )
    def _compute_quality(
        self,
        image: np.ndarray,
        face_data,
        parse_data=None,
        detected_faces=None,
    ) -> FaceQualityOutput:
        """Compute quality metrics for all detected face crops."""
        if not face_data or not face_data.faces:
            return FaceQualityOutput()

        img_w, img_h = face_data.image_size

        # Build face.parse result map: face_id → FaceParseResult
        parse_map: Dict[int, object] = {}
        if parse_data is not None:
            for pr in getattr(parse_data, "results", []):
                parse_map[pr.face_id] = pr

        face_results: List[FaceQualityResult] = []

        # Find the main (largest) face for backward compat fields + smoothing
        main_face = max(face_data.faces, key=lambda f: f.area_ratio)

        for face in face_data.faces:
            nx, ny, nw, nh = face.bbox
            px = int(nx * img_w)
            py = int(ny * img_h)
            pw = int(nw * img_w)
            ph = int(nh * img_h)

            # Only apply temporal smoothing for main face
            if face is main_face:
                box = self._head_smoother.update((px, py, pw, ph))
            else:
                box = (px, py, pw, ph)

            head_img, head_box = face_crop(
                image, box, expand=self._face_expand, crop_ratio=self._crop_ratio,
            )

            head_gray = cv2.cvtColor(head_img, cv2.COLOR_BGR2GRAY)

            # Match to DetectedFace for landmarks
            detected = _find_detected_face(
                face, detected_faces or [], img_w, img_h,
            )
            landmarks = getattr(detected, "landmarks", None) if detected else None

            # face.parse result for this face
            face_id = getattr(face, "face_id", 0)
            parse_result = parse_map.get(face_id)

            # Semantic ratios from class_map (face bbox region in head crop)
            semantic_ratios: dict[str, float] = {}
            if parse_result is not None:
                crop_shape = head_gray.shape[:2]
                hx, hy, hw, hh = head_box
                # Scale face bbox from image coords to crop_shape coords
                sx = crop_shape[1] / hw if hw > 0 else 1.0
                sy = crop_shape[0] / hh if hh > 0 else 1.0
                face_region = (
                    int((px - hx) * sx), int((py - hy) * sy),
                    int(pw * sx), int(ph * sy),
                )

                class_map = getattr(parse_result, "class_map", None)
                if class_map is not None and class_map.size > 1:
                    aligned_cmap = _align_parse_to_crop(
                        class_map, parse_result.crop_box, head_box, crop_shape,
                    )
                    if aligned_cmap is not None:
                        semantic_ratios = _compute_semantic_ratios(aligned_cmap, region=face_region)

            # 3-level mask fallback
            mask, method, parsing_cov = _compute_face_mask(
                head_gray, parse_result, landmarks, head_box,
            )

            # Compute metrics within mask
            head_blur = _laplacian_variance_masked(head_gray, mask)
            head_exposure = float(head_gray[mask > 0].mean()) if np.any(mask > 0) else 0.0
            head_contrast = _local_contrast(head_gray, mask)
            clipped, crushed = _exposure_stats(head_gray, mask)

            face_results.append(FaceQualityResult(
                face_id=face_id,
                head_blur=head_blur,
                head_exposure=head_exposure,
                head_crop_box=head_box,
                mask_method=method,
                head_contrast=head_contrast,
                clipped_ratio=clipped,
                crushed_ratio=crushed,
                parsing_coverage=parsing_cov,
                seg_face=semantic_ratios.get("seg_face", 0.0),
                seg_eye=semantic_ratios.get("seg_eye", 0.0),
                seg_mouth=semantic_ratios.get("seg_mouth", 0.0),
                seg_hair=semantic_ratios.get("seg_hair", 0.0),
                eye_pixel_ratio=semantic_ratios.get("eye_pixel_ratio", 0.0),
            ))

        # Main face result for backward compat
        main_result = next(
            (r for r in face_results if r.face_id == getattr(main_face, "face_id", 0)),
            face_results[0] if face_results else None,
        )

        return FaceQualityOutput(
            face_results=face_results,
            head_crop_box=main_result.head_crop_box if main_result else None,
            image_size=(img_w, img_h),
            head_blur=main_result.head_blur if main_result else 0.0,
            head_exposure=main_result.head_exposure if main_result else 0.0,
            mask_method=main_result.mask_method if main_result else "",
            head_contrast=main_result.head_contrast if main_result else 0.0,
            clipped_ratio=main_result.clipped_ratio if main_result else 0.0,
            crushed_ratio=main_result.crushed_ratio if main_result else 0.0,
            parsing_coverage=main_result.parsing_coverage if main_result else 0.0,
            seg_face=main_result.seg_face if main_result else 0.0,
            seg_eye=main_result.seg_eye if main_result else 0.0,
            seg_mouth=main_result.seg_mouth if main_result else 0.0,
            seg_hair=main_result.seg_hair if main_result else 0.0,
            eye_pixel_ratio=main_result.eye_pixel_ratio if main_result else 0.0,
        )

    def process(
        self,
        frame: "Frame",
        deps: Optional[Dict[str, Observation]] = None,
    ) -> Optional[Observation]:
        if not self._initialized:
            raise RuntimeError("Analyzer not initialized — call initialize() first")

        face_obs = deps.get("face.detect") if deps else None
        if face_obs is None:
            logger.debug("FaceQualityAnalyzer: no face.detect dependency")
            return None

        face_data = face_obs.data
        image = frame.data

        # Optional: face.parse data
        parse_obs = deps.get("face.parse") if deps else None
        parse_data = getattr(parse_obs, "data", None) if parse_obs else None

        # Get detected_faces for landmark access
        detected_faces = getattr(face_data, "detected_faces", []) or []

        self._step_timings = {}
        output = self._compute_quality(
            image, face_data, parse_data, detected_faces,
        )
        timing = self._step_timings.copy() if self._step_timings else None
        self._step_timings = None

        h, w = image.shape[:2]
        output.image_size = (w, h)

        has_quality = output.head_blur > 0 or output.head_exposure > 0

        return Observation(
            source=self.name,
            frame_id=frame.frame_id,
            t_ns=frame.t_src_ns,
            signals={
                "has_quality": has_quality,
            },
            data=output,
            metadata={
                "_metrics": {
                    "head_blur": output.head_blur,
                    "mask_method": output.mask_method,
                },
            },
            timing=timing,
        )

    def annotate(self, obs):
        """Return marks for head crop region."""
        if obs is None or obs.data is None:
            return []

        from vpx.sdk.marks import BBoxMark

        data = obs.data
        marks = []

        img_w, img_h = data.image_size if data.image_size else (1, 1)

        if data.head_crop_box is not None:
            x, y, w, h = data.head_crop_box
            method_tag = f" [{data.mask_method}]" if data.mask_method else ""
            marks.append(BBoxMark(
                x=x / img_w, y=y / img_h,
                w=w / img_w, h=h / img_h,
                label=f"head q={data.head_blur:.0f}{method_tag}",
                color=(255, 255, 0),
                thickness=1,
            ))

        return marks
