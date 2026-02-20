"""Shot quality analyzer — portrait crop quality assessment.

ShotQualityAnalyzer: depends on face.detect only.
Computes quality metrics using two crops derived from face detection:

Tight crop (face_expand=1.1, 4:5):
  - head_blur: Laplacian variance — face sharpness metric
  - head_exposure: mean brightness in face region

Portrait crop (aesthetic_expand=2.5, 4:5):
  - head_aesthetic: LAION Aesthetic score [0,1] — wider crop including
    shoulders so the model sees portrait composition context

두 crop을 분리하는 이유:
  - blur/exposure는 얼굴 영역에 집중해야 정확 (tight crop)
  - LAION aesthetic는 구도·배경·피사체 배치를 평가하므로 어깨/배경 포함 필요
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
from vpx.vision_embed.types import ShotQualityOutput
from vpx.vision_embed.crop import CropRatio, face_crop, BBoxSmoother

logger = logging.getLogger(__name__)

def _laplacian_variance(gray: np.ndarray) -> float:
    """Laplacian variance — sharpness metric."""
    if gray.size == 0:
        return 0.0
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return float(lap.var())


class ShotQualityAnalyzer(Module):
    """Analyzer that assesses quality of the head-shot portrait crop region.

    Computes quality metrics within the head-shot (tight face crop) derived
    from face detection.

    Layer 1 (always, OpenCV only):
      head_blur, head_exposure

    Layer 2 (optional, requires open-clip-torch + LAION weights):
      head_aesthetic

    depends: ["face.detect"]
    """

    depends = ["face.detect"]
    optional_depends: list = []

    def __init__(
        self,
        face_expand: float = 1.1,
        aesthetic_expand: float = 2.5,
        smooth_alpha: float = 0.3,
        crop_ratio: CropRatio = "4:5",
        enable_aesthetic: bool = True,
        device: str = "cpu",
    ):
        self._face_expand = face_expand          # blur/exposure: tight face crop
        self._aesthetic_expand = aesthetic_expand  # aesthetic: portrait with shoulders
        self._crop_ratio = crop_ratio
        self._device = device
        self._enable_aesthetic = enable_aesthetic

        self._head_smoother = BBoxSmoother(alpha=smooth_alpha)
        self._aesthetic: Optional[object] = None
        self._initialized = False
        self._step_timings: Optional[Dict[str, float]] = None

    @property
    def name(self) -> str:
        return "shot.quality"

    @property
    def capabilities(self) -> ModuleCapabilities:
        return ModuleCapabilities(
            flags=Capability.STATEFUL,
            init_time_sec=0.1,
            required_extras=frozenset({"vpx-vision-embed"}),
        )

    @property
    def processing_steps(self) -> List[ProcessingStep]:
        return get_processing_steps(self)

    def initialize(self) -> None:
        if self._initialized:
            return

        if self._enable_aesthetic:
            try:
                from vpx.sdk.paths import get_models_dir
                from vpx.vision_embed.backends.laion import LAIONAestheticBackend
                aesthetic = LAIONAestheticBackend(
                    models_dir=get_models_dir(), device=self._device
                )
                aesthetic.initialize()
                self._aesthetic = aesthetic
                if aesthetic.available:
                    logger.info("ShotQualityAnalyzer: LAION aesthetic scoring enabled")
                else:
                    logger.info(
                        "ShotQualityAnalyzer: LAION weights not found, aesthetic scoring disabled"
                    )
            except Exception as e:
                logger.debug("LAION aesthetic backend unavailable: %s", e)
                self._aesthetic = None

        self._initialized = True
        logger.info("ShotQualityAnalyzer initialized")

    def cleanup(self) -> None:
        if self._aesthetic is not None:
            self._aesthetic.cleanup()
            self._aesthetic = None
        self._head_smoother.reset()
        self._initialized = False
        logger.info("ShotQualityAnalyzer cleaned up")

    def reset(self) -> None:
        self._head_smoother.reset()

    @processing_step(
        name="shot_quality",
        description="Compute head-crop quality metrics (blur, exposure, aesthetic)",
        backend="OpenCV",
        input_type="Frame + FaceDetectOutput",
        output_type="ShotQualityOutput",
    )
    def _compute_quality(
        self, image: np.ndarray, face_data
    ) -> ShotQualityOutput:
        """Compute quality metrics for head crop."""
        if not face_data or not face_data.faces:
            return ShotQualityOutput()

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

        # ── Layer 1: CV-based quality (tight crop) ──
        # Head blur (Laplacian variance)
        head_gray = cv2.cvtColor(head_img, cv2.COLOR_BGR2GRAY)
        head_blur = _laplacian_variance(head_gray)

        # Head exposure (mean brightness)
        head_exposure = float(head_gray.mean())

        # ── Layer 2: LAION aesthetic (wider portrait crop with shoulders) ──
        head_aesthetic = 0.0
        if self._aesthetic is not None and getattr(self._aesthetic, "available", False):
            portrait_img, _ = face_crop(
                image, smoothed_head,
                expand=self._aesthetic_expand,
                crop_ratio=self._crop_ratio,
            )
            head_aesthetic = self._aesthetic.score(portrait_img)

        return ShotQualityOutput(
            head_crop_box=head_box,
            image_size=(img_w, img_h),
            head_blur=head_blur,
            head_exposure=head_exposure,
            head_aesthetic=head_aesthetic,
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
            logger.debug("ShotQualityAnalyzer: no face.detect dependency")
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
                "has_aesthetic": output.head_aesthetic > 0,
            },
            data=output,
            metadata={
                "_metrics": {
                    "head_blur": output.head_blur,
                    "aesthetic_active": output.head_aesthetic > 0,
                }
            },
            timing=timing,
        )

    def annotate(self, obs):
        """Return marks for head/scene crop regions."""
        if obs is None or obs.data is None:
            return []

        from vpx.sdk.marks import BBoxMark, LabelMark

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

        if data.head_aesthetic > 0:
            marks.append(LabelMark(
                text=f"aes: {data.head_aesthetic:.2f}",
                x=0.01, y=0.06,
                color=(200, 200, 255),
                background=(30, 30, 30),
                font_scale=0.4,
            ))

        return marks
