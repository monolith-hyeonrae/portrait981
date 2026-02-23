"""Face quality analyzer — face crop blur and exposure assessment.

FaceQualityAnalyzer: depends on face.detect only.
Computes quality metrics from a tight face crop:
  - head_blur: Laplacian variance — face sharpness metric
  - head_exposure: mean brightness in face region
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
from momentscan.algorithm.analyzers.face_quality.output import FaceQualityOutput

logger = logging.getLogger(__name__)


def _laplacian_variance(gray: np.ndarray) -> float:
    """Laplacian variance — sharpness metric."""
    if gray.size == 0:
        return 0.0
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return float(lap.var())


class FaceQualityAnalyzer(Module):
    """Analyzer that assesses blur and exposure of the face crop region.

    Computes quality metrics within a tight face crop derived from face detection.
    Uses OpenCV only — no ML dependencies.

    depends: ["face.detect"]
    """

    depends = ["face.detect"]
    optional_depends: list = []

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
        description="Compute face crop quality metrics (blur, exposure)",
        backend="OpenCV",
        input_type="Frame + FaceDetectOutput",
        output_type="FaceQualityOutput",
    )
    def _compute_quality(
        self, image: np.ndarray, face_data
    ) -> FaceQualityOutput:
        """Compute quality metrics for face crop."""
        if not face_data or not face_data.faces:
            return FaceQualityOutput()

        face = max(face_data.faces, key=lambda f: f.area_ratio)
        img_w, img_h = face_data.image_size

        # Pixel bbox for head crop
        nx, ny, nw, nh = face.bbox
        px = int(nx * img_w)
        py = int(ny * img_h)
        pw = int(nw * img_w)
        ph = int(nh * img_h)

        # Head crop (smoothed bbox) — tight, for blur/exposure
        smoothed_head = self._head_smoother.update((px, py, pw, ph))
        head_img, head_box = face_crop(
            image, smoothed_head, expand=self._face_expand, crop_ratio=self._crop_ratio,
        )

        # Head blur (Laplacian variance)
        head_gray = cv2.cvtColor(head_img, cv2.COLOR_BGR2GRAY)
        head_blur = _laplacian_variance(head_gray)

        # Head exposure (mean brightness)
        head_exposure = float(head_gray.mean())

        return FaceQualityOutput(
            head_crop_box=head_box,
            image_size=(img_w, img_h),
            head_blur=head_blur,
            head_exposure=head_exposure,
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

        self._step_timings = {}
        output = self._compute_quality(image, face_data)
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
            marks.append(BBoxMark(
                x=x / img_w, y=y / img_h,
                w=w / img_w, h=h / img_h,
                label=f"head q={data.head_blur:.0f}",
                color=(255, 255, 0),
                thickness=1,
            ))

        return marks
