"""Vision embedding analyzer - depends on face_detect, optionally body_pose."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Dict, List
import logging

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
from vpx.vision_embed.types import EmbedOutput
from vpx.vision_embed.backends.base import EmbeddingBackend
from vpx.vision_embed.crop import face_crop, body_crop, BBoxSmoother

logger = logging.getLogger(__name__)


class VisionEmbedAnalyzer(Module):
    """Analyzer for vision embedding extraction.

    Extracts DINOv2 (or other backbone) embeddings from face and body crops.
    Depends on face_detect for face bounding boxes, optionally uses body_pose
    for upper-body crops.

    depends: ["face.detect"]
    optional_depends: ["body.pose"]

    Outputs:
        - signals: has_face_embed, has_body_embed
        - data: EmbedOutput with e_face (384,) and e_body (384,)

    Example:
        >>> graph = (FlowGraphBuilder()
        ...     .source()
        ...     .path("detect", modules=[FaceDetectionAnalyzer()])
        ...     .path("embed", modules=[VisionEmbedAnalyzer()])
        ...     .build())
    """

    depends = ["face.detect"]
    optional_depends = ["body.pose"]

    def __init__(
        self,
        backend: Optional[EmbeddingBackend] = None,
        device: str = "cuda:0",
        face_expand: float = 1.5,
        smooth_alpha: float = 0.3,
    ):
        self._device = device
        self._backend = backend
        self._face_expand = face_expand
        self._initialized = False

        # Bbox smoothing state
        self._face_smoother = BBoxSmoother(alpha=smooth_alpha)
        self._body_smoother = BBoxSmoother(alpha=smooth_alpha)

        # Step timing tracking
        self._step_timings: Optional[Dict[str, float]] = None

    @property
    def name(self) -> str:
        return "vision.embed"

    @property
    def capabilities(self) -> ModuleCapabilities:
        return ModuleCapabilities(
            flags=Capability.GPU | Capability.STATEFUL,
            gpu_memory_mb=512,
            init_time_sec=5.0,
            resource_groups=frozenset({"torch"}),
            required_extras=frozenset({"vpx-vision-embed"}),
        )

    @property
    def processing_steps(self) -> List[ProcessingStep]:
        return get_processing_steps(self)

    def initialize(self) -> None:
        if self._initialized:
            return

        if self._backend is None:
            from vpx.vision_embed.backends.dinov2 import DINOv2Backend
            self._backend = DINOv2Backend()

        self._backend.initialize(self._device)
        self._initialized = True
        logger.info("VisionEmbedAnalyzer initialized")

    def cleanup(self) -> None:
        if self._backend is not None:
            self._backend.cleanup()
        self._face_smoother.reset()
        self._body_smoother.reset()
        self._initialized = False
        logger.info("VisionEmbedAnalyzer cleaned up")

    def reset(self) -> None:
        self._face_smoother.reset()
        self._body_smoother.reset()

    # ========== Processing Steps ==========

    @processing_step(
        name="face_embed",
        description="Extract face crop and compute embedding",
        backend="DINOv2",
        input_type="Frame + FaceDetectOutput",
        output_type="Optional[np.ndarray] (384,)",
    )
    def _embed_face(
        self, image, face_data
    ) -> tuple[Optional["np.ndarray"], Optional[tuple[int, int, int, int]]]:
        """Extract face crop from detection and compute embedding."""
        import numpy as np

        if not face_data or not face_data.faces:
            return None, None

        # Select face with highest area_ratio (most prominent)
        face = max(face_data.faces, key=lambda f: f.area_ratio)
        img_w, img_h = face_data.image_size

        # Convert normalized bbox to pixel coords
        nx, ny, nw, nh = face.bbox
        px = int(nx * img_w)
        py = int(ny * img_h)
        pw = int(nw * img_w)
        ph = int(nh * img_h)

        # Smooth the bbox
        smoothed = self._face_smoother.update((px, py, pw, ph))

        # Extract crop
        crop, actual_box = face_crop(
            image, smoothed, expand=self._face_expand
        )

        # Compute embedding
        embedding = self._backend.embed(crop)
        return embedding, actual_box

    @processing_step(
        name="body_embed",
        description="Extract body crop and compute embedding",
        backend="DINOv2",
        input_type="Frame + PoseOutput",
        output_type="Optional[np.ndarray] (384,)",
        depends_on=["face_embed"],
    )
    def _embed_body(
        self, image, pose_data
    ) -> tuple[Optional["np.ndarray"], Optional[tuple[int, int, int, int]]]:
        """Extract upper-body crop from pose and compute embedding."""
        if pose_data is None:
            return None, None

        # Get keypoints from pose data
        keypoints = None
        if hasattr(pose_data, "keypoints") and pose_data.keypoints is not None:
            keypoints = pose_data.keypoints
        elif hasattr(pose_data, "poses") and pose_data.poses:
            # Use first detected pose
            first_pose = pose_data.poses[0]
            if hasattr(first_pose, "keypoints"):
                keypoints = first_pose.keypoints

        if keypoints is None:
            return None, None

        crop, actual_box = body_crop(image, keypoints)

        # Smooth the bbox
        smoothed = self._body_smoother.update(actual_box)
        # Re-crop with smoothed box is not worth the cost; just record smoothed box

        embedding = self._backend.embed(crop)
        return embedding, smoothed

    # ========== Main process method ==========

    def process(
        self,
        frame: Frame,
        deps: Optional[Dict[str, Observation]] = None,
    ) -> Optional[Observation]:
        if self._backend is None:
            raise RuntimeError("Analyzer not initialized")

        # Get face_detect dependency
        face_obs = deps.get("face.detect") if deps else None
        if face_obs is None:
            logger.debug("VisionEmbedAnalyzer: no face.detect dependency")
            return None

        face_data = face_obs.data

        # Get optional body.pose dependency
        pose_obs = deps.get("body.pose") if deps else None
        pose_data = pose_obs.data if pose_obs else None

        image = frame.data

        # Enable step timing
        self._step_timings = {}

        # Face embedding
        e_face, face_box = self._embed_face(image, face_data)

        # Body embedding (optional)
        e_body, body_box = self._embed_body(image, pose_data)

        # Collect timing
        timing = self._step_timings.copy() if self._step_timings else None
        self._step_timings = None

        h, w = image.shape[:2]
        output = EmbedOutput(
            e_face=e_face,
            e_body=e_body,
            face_crop_box=face_box,
            body_crop_box=body_box,
            image_size=(w, h),
        )

        return Observation(
            source=self.name,
            frame_id=frame.frame_id,
            t_ns=frame.t_src_ns,
            signals={
                "has_face_embed": e_face is not None,
                "has_body_embed": e_body is not None,
            },
            data=output,
            metadata={
                "_metrics": {
                    "face_embed_computed": e_face is not None,
                    "body_embed_computed": e_body is not None,
                }
            },
            timing=timing,
        )

    # ========== Visualization ==========

    def annotate(self, obs):
        """Return marks for face/body crop regions and embedding status."""
        if obs is None or obs.data is None:
            return []

        from vpx.sdk.marks import BBoxMark, LabelMark

        data = obs.data
        marks = []

        img_w, img_h = data.image_size if data.image_size else (1, 1)

        # Face crop box (cyan, dashed-style thinner line)
        if data.face_crop_box is not None:
            x, y, w, h = data.face_crop_box
            marks.append(BBoxMark(
                x=x / img_w, y=y / img_h,
                w=w / img_w, h=h / img_h,
                label="face_emb",
                color=(255, 255, 0),  # cyan
                thickness=1,
            ))

        # Body crop box (green)
        if data.body_crop_box is not None:
            x, y, w, h = data.body_crop_box
            marks.append(BBoxMark(
                x=x / img_w, y=y / img_h,
                w=w / img_w, h=h / img_h,
                label="body_emb",
                color=(0, 200, 0),  # green
                thickness=1,
            ))

        # Status label
        parts = []
        if obs.signals.get("has_face_embed"):
            parts.append("face")
        if obs.signals.get("has_body_embed"):
            parts.append("body")
        if parts:
            marks.append(LabelMark(
                text=f"emb: {'+'.join(parts)}",
                x=0.01, y=0.06,
                color=(255, 255, 0),
                background=(30, 30, 30),
                font_scale=0.4,
            ))

        return marks
