"""Portrait score analyzer — CLIP portrait quality scoring.

PortraitScoreAnalyzer: depends on face.detect only.
Computes CLIP-based portrait quality from a portrait crop (head-to-shoulders):
  - head_aesthetic: CLIP portrait quality score [0,1]
    포트레이트 품질 텍스트 프롬프트와 cosine similarity로 계산.
    얼굴 표정/각도 변화에 민감.

blur/exposure는 face.quality analyzer가 담당 (momentscan 내부 모듈).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional, Dict, List

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
from vpx.portrait_score.types import PortraitScoreOutput
from vpx.sdk.crop import CropRatio, face_crop, BBoxSmoother

logger = logging.getLogger(__name__)


class PortraitScoreAnalyzer(Module):
    """Analyzer that scores portrait quality using CLIP embeddings.

    Computes CLIP portrait quality from a portrait crop (head-to-shoulders)
    derived from face detection.

    Requires open-clip-torch for scoring. When unavailable, returns zero scores.

    depends: ["face.detect"]
    """

    depends = ["face.detect"]
    optional_depends: list = []

    def __init__(
        self,
        aesthetic_expand: float = 2.2,
        aesthetic_y_shift: float = 0.3,
        smooth_alpha: float = 0.3,
        enable_aesthetic: bool = True,
        enable_caption: bool = False,
        clip_stride: int = 1,
        device: str = "auto",
    ):
        self._aesthetic_expand = aesthetic_expand
        self._aesthetic_y_shift = aesthetic_y_shift
        self._device = device
        self._enable_aesthetic = enable_aesthetic
        self._enable_caption = enable_caption
        self._clip_stride = max(1, clip_stride)

        self._head_smoother = BBoxSmoother(alpha=smooth_alpha)
        self._aesthetic: Optional[object] = None
        self._initialized = False
        self._step_timings: Optional[Dict[str, float]] = None
        self._clip_frame_counter: int = 0
        self._clip_cache_score: float = 0.0

    @property
    def name(self) -> str:
        return "portrait.score"

    @property
    def capabilities(self) -> ModuleCapabilities:
        return ModuleCapabilities(
            flags=Capability.STATEFUL,
            init_time_sec=0.1,
            required_extras=frozenset({"vpx-portrait-score"}),
        )

    @property
    def processing_steps(self) -> List[ProcessingStep]:
        return get_processing_steps(self)

    def initialize(self) -> None:
        if self._initialized:
            return

        if self._enable_aesthetic:
            try:
                from vpx.portrait_score.backends.clip_portrait import CLIPPortraitScorer
                scorer = CLIPPortraitScorer(
                    device=self._device,
                    enable_caption=self._enable_caption,
                )
                scorer.initialize()
                self._aesthetic = scorer
                if scorer.available:
                    logger.info("PortraitScoreAnalyzer: CLIP portrait scoring enabled")
                else:
                    logger.info(
                        "PortraitScoreAnalyzer: open-clip not available, portrait scoring disabled"
                    )
            except Exception as e:
                logger.debug("CLIP portrait scorer unavailable: %s", e)
                self._aesthetic = None

        self._initialized = True
        logger.info("PortraitScoreAnalyzer initialized")

    def cleanup(self) -> None:
        if self._aesthetic is not None:
            self._aesthetic.cleanup()
            self._aesthetic = None
        self._head_smoother.reset()
        self._initialized = False
        logger.info("PortraitScoreAnalyzer cleaned up")

    def reset(self) -> None:
        self._head_smoother.reset()
        self._clip_frame_counter = 0
        self._clip_cache_score = 0.0
        if self._aesthetic is not None:
            self._aesthetic._embed_ema = None

    @processing_step(
        name="portrait_score",
        description="Compute CLIP portrait quality score",
        backend="CLIP",
        input_type="Frame + FaceDetectOutput",
        output_type="PortraitScoreOutput",
    )
    def _compute_score(
        self, image: np.ndarray, face_data
    ) -> PortraitScoreOutput:
        """Compute CLIP portrait score."""
        if not face_data or not face_data.faces:
            return PortraitScoreOutput()

        face = max(face_data.faces, key=lambda f: f.area_ratio)
        img_w, img_h = face_data.image_size

        # Pixel bbox
        nx, ny, nw, nh = face.bbox
        px = int(nx * img_w)
        py = int(ny * img_h)
        pw = int(nw * img_w)
        ph = int(nh * img_h)

        smoothed_head = self._head_smoother.update((px, py, pw, ph))

        # CLIP portrait quality (portrait crop: 정수리~어깨)
        head_aesthetic = 0.0
        portrait_box = None
        if self._aesthetic is not None and getattr(self._aesthetic, "available", False):
            portrait_img, portrait_box = face_crop(
                image, smoothed_head,
                expand=self._aesthetic_expand,
                crop_ratio="1:1",
                y_shift=self._aesthetic_y_shift,
            )
            self._clip_frame_counter += 1
            if self._clip_frame_counter >= self._clip_stride:
                self._clip_frame_counter = 0
                head_aesthetic = self._aesthetic.score(portrait_img)
                self._clip_cache_score = head_aesthetic
            else:
                head_aesthetic = self._clip_cache_score

        return PortraitScoreOutput(
            portrait_crop_box=portrait_box,
            image_size=(img_w, img_h),
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
            logger.debug("PortraitScoreAnalyzer: no face.detect dependency")
            return None

        face_data = face_obs.data
        image = frame.data

        self._step_timings = {}
        output = self._compute_score(image, face_data)
        timing = self._step_timings.copy() if self._step_timings else None
        self._step_timings = None

        h, w = image.shape[:2]
        output.image_size = (w, h)

        # CLIP breakdown + axes + caption → metadata for debug overlay
        clip_breakdown = None
        clip_axes = None
        clip_caption = None
        if self._aesthetic is not None:
            if hasattr(self._aesthetic, "last_breakdown"):
                clip_breakdown = self._aesthetic.last_breakdown
            if hasattr(self._aesthetic, "last_axes"):
                clip_axes = self._aesthetic.last_axes
            if hasattr(self._aesthetic, "last_caption"):
                clip_caption = self._aesthetic.last_caption

        # Axis signals
        axis_signals: Dict[str, bool] = {}
        if clip_axes:
            for ax in clip_axes:
                if ax.active:
                    axis_signals[f"clip_axis_{ax.name}"] = True
            axis_signals["clip_quirky"] = any(
                ax.active and ax.action == "quirky" for ax in clip_axes
            )
            axis_signals["clip_select"] = any(
                ax.active and ax.action == "select" for ax in clip_axes
            )

        return Observation(
            source=self.name,
            frame_id=frame.frame_id,
            t_ns=frame.t_src_ns,
            signals={
                "has_aesthetic": output.head_aesthetic > 0,
                **axis_signals,
            },
            data=output,
            metadata={
                "_metrics": {
                    "aesthetic_active": output.head_aesthetic > 0,
                },
                "_clip_breakdown": clip_breakdown,
                "_clip_axes": clip_axes,
                "_clip_caption": clip_caption,
            },
            timing=timing,
        )

    def annotate(self, obs):
        """Return marks for portrait crop region."""
        if obs is None or obs.data is None:
            return []

        from vpx.sdk.marks import BBoxMark, LabelMark

        data = obs.data
        marks = []

        img_w, img_h = data.image_size if data.image_size else (1, 1)

        if data.portrait_crop_box is not None:
            x, y, w, h = data.portrait_crop_box
            marks.append(BBoxMark(
                x=x / img_w, y=y / img_h,
                w=w / img_w, h=h / img_h,
                label=f"portrait clip={data.head_aesthetic:.2f}",
                color=(200, 200, 255),
                thickness=1,
            ))
        elif data.head_aesthetic > 0:
            marks.append(LabelMark(
                text=f"clip: {data.head_aesthetic:.2f}",
                x=0.01, y=0.06,
                color=(200, 200, 255),
                background=(30, 30, 30),
                font_scale=0.4,
            ))

        # CoCa caption
        caption = obs.metadata.get("_clip_caption") if obs.metadata else None
        if caption:
            marks.append(LabelMark(
                text=f"CoCa: {caption}",
                x=0.01, y=0.92,
                color=(200, 255, 255),
                background=(20, 20, 20),
                font_scale=0.35,
            ))

        # CLIP score summary
        bd = obs.metadata.get("_clip_breakdown") if obs.metadata else None
        if bd is not None:
            marks.append(LabelMark(
                text=f"CLIP score={bd.score:.2f} (diff={bd.raw_diff:+.3f})",
                x=0.01, y=0.95,
                color=(255, 255, 200),
                background=(20, 20, 20),
                font_scale=0.35,
            ))

        # CLIP axes
        _AXIS_COLORS = {
            "disney_smile": (150, 255, 150),
            "charisma": (255, 180, 255),
            "playful_cute": (150, 255, 255),
            "wild_roar": (100, 255, 100),
        }
        clip_axes = obs.metadata.get("_clip_axes") if obs.metadata else None
        if clip_axes:
            from vpx.sdk.marks import BarMark
            y_apos = 0.02
            for ax in clip_axes:
                color = _AXIS_COLORS.get(ax.name, (200, 200, 200))
                active_marker = " *" if ax.active else ""
                marks.append(LabelMark(
                    text=f"{ax.name}: {ax.score:.2f}{active_marker}",
                    x=0.01, y=y_apos,
                    color=color,
                    background=(20, 20, 20),
                    font_scale=0.35,
                ))
                marks.append(BarMark(
                    x=0.22, y=y_apos,
                    value=ax.score,
                    w=0.15,
                    color=color,
                ))
                y_apos += 0.025

        return marks
