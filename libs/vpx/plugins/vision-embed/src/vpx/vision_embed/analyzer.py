"""Face and body embedding analyzers — split from VisionEmbedAnalyzer.

FaceEmbedAnalyzer: depends on face.detect only.
BodyEmbedAnalyzer: depends on face.detect, optionally body.pose.
Both share DINOv2Backend, crop utilities, and BBoxSmoother.
"""

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
from vpx.vision_embed.types import FaceEmbedOutput, BodyEmbedOutput
from vpx.vision_embed.backends.base import EmbeddingBackend
from vpx.vision_embed.crop import face_crop, body_crop, BBoxSmoother

logger = logging.getLogger(__name__)


class FaceEmbedAnalyzer(Module):
    """Analyzer for face embedding extraction.

    Extracts DINOv2 (or other backbone) embeddings from face crops.
    Depends on face_detect for face bounding boxes.

    depends: ["face.detect"]

    Outputs:
        - signals: has_face_embed
        - data: FaceEmbedOutput with e_face (384,)

    Example:
        >>> graph = (FlowGraphBuilder()
        ...     .source()
        ...     .path("detect", modules=[FaceDetectionAnalyzer()])
        ...     .path("embed", modules=[FaceEmbedAnalyzer()])
        ...     .build())
    """

    depends = ["face.detect"]

    def __init__(
        self,
        backend: Optional[EmbeddingBackend] = None,
        device: str = "cuda:0",
        face_expand: float = 1.1,
        smooth_alpha: float = 0.3,
    ):
        self._device = device
        self._backend = backend
        self._owns_backend = backend is not None
        self._face_expand = face_expand
        self._initialized = False

        self._face_smoother = BBoxSmoother(alpha=smooth_alpha)
        self._step_timings: Optional[Dict[str, float]] = None

    @property
    def name(self) -> str:
        return "face.embed"

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
            self._backend = DINOv2Backend.get_shared(self._device)
            self._owns_backend = False
        elif not self._initialized:
            self._backend.initialize(self._device)
        self._initialized = True
        logger.info("FaceEmbedAnalyzer initialized")

    def cleanup(self) -> None:
        if not self._owns_backend and self._backend is not None:
            from vpx.vision_embed.backends.dinov2 import DINOv2Backend
            DINOv2Backend.release_shared(self._device)
        elif self._owns_backend and self._backend is not None:
            self._backend.cleanup()
        self._backend = None
        self._face_smoother.reset()
        self._initialized = False
        logger.info("FaceEmbedAnalyzer cleaned up")

    def reset(self) -> None:
        self._face_smoother.reset()

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

        face = max(face_data.faces, key=lambda f: f.area_ratio)
        img_w, img_h = face_data.image_size

        nx, ny, nw, nh = face.bbox
        px = int(nx * img_w)
        py = int(ny * img_h)
        pw = int(nw * img_w)
        ph = int(nh * img_h)

        smoothed = self._face_smoother.update((px, py, pw, ph))

        crop, actual_box = face_crop(
            image, smoothed, expand=self._face_expand
        )

        embedding = self._backend.embed(crop)
        return embedding, actual_box

    def process(
        self,
        frame: Frame,
        deps: Optional[Dict[str, Observation]] = None,
    ) -> Optional[Observation]:
        if self._backend is None:
            raise RuntimeError("Analyzer not initialized")

        face_obs = deps.get("face.detect") if deps else None
        if face_obs is None:
            logger.debug("FaceEmbedAnalyzer: no face.detect dependency")
            return None

        face_data = face_obs.data
        image = frame.data

        self._step_timings = {}

        e_face, face_box = self._embed_face(image, face_data)

        timing = self._step_timings.copy() if self._step_timings else None
        self._step_timings = None

        h, w = image.shape[:2]
        output = FaceEmbedOutput(
            e_face=e_face,
            face_crop_box=face_box,
            image_size=(w, h),
        )

        return Observation(
            source=self.name,
            frame_id=frame.frame_id,
            t_ns=frame.t_src_ns,
            signals={
                "has_face_embed": e_face is not None,
            },
            data=output,
            metadata={
                "_metrics": {
                    "face_embed_computed": e_face is not None,
                }
            },
            timing=timing,
        )

    def annotate(self, obs):
        """Return marks for face crop region and embedding status."""
        if obs is None or obs.data is None:
            return []

        from vpx.sdk.marks import BBoxMark, LabelMark

        data = obs.data
        marks = []

        img_w, img_h = data.image_size if data.image_size else (1, 1)

        if data.face_crop_box is not None:
            x, y, w, h = data.face_crop_box
            marks.append(BBoxMark(
                x=x / img_w, y=y / img_h,
                w=w / img_w, h=h / img_h,
                label="face_emb",
                color=(255, 255, 0),  # cyan
                thickness=1,
            ))

        if obs.signals.get("has_face_embed"):
            marks.append(LabelMark(
                text="emb: face",
                x=0.01, y=0.06,
                color=(255, 255, 0),
                background=(30, 30, 30),
                font_scale=0.4,
            ))

        return marks


class BodyEmbedAnalyzer(Module):
    """Analyzer for body embedding extraction.

    Extracts DINOv2 (or other backbone) embeddings from upper-body crops.
    Depends on face_detect for face bounding boxes (used for pose matching),
    optionally uses body_pose for upper-body crops.

    depends: ["face.detect"]
    optional_depends: ["body.pose"]

    Outputs:
        - signals: has_body_embed
        - data: BodyEmbedOutput with e_body (384,)
    """

    depends = ["face.detect", "face.embed"]
    optional_depends = ["body.pose"]

    def __init__(
        self,
        backend: Optional[EmbeddingBackend] = None,
        device: str = "cuda:0",
        smooth_alpha: float = 0.3,
    ):
        self._device = device
        self._backend = backend
        self._owns_backend = backend is not None
        self._initialized = False

        self._body_smoother = BBoxSmoother(alpha=smooth_alpha)
        self._step_timings: Optional[Dict[str, float]] = None

    @property
    def name(self) -> str:
        return "body.embed"

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
            self._backend = DINOv2Backend.get_shared(self._device)
            self._owns_backend = False
        elif not self._initialized:
            self._backend.initialize(self._device)
        self._initialized = True
        logger.info("BodyEmbedAnalyzer initialized")

    def cleanup(self) -> None:
        if not self._owns_backend and self._backend is not None:
            from vpx.vision_embed.backends.dinov2 import DINOv2Backend
            DINOv2Backend.release_shared(self._device)
        elif self._owns_backend and self._backend is not None:
            self._backend.cleanup()
        self._backend = None
        self._body_smoother.reset()
        self._initialized = False
        logger.info("BodyEmbedAnalyzer cleaned up")

    def reset(self) -> None:
        self._body_smoother.reset()

    @staticmethod
    def _match_pose_to_face(kp_list: list, face_data) -> object | None:
        """주 얼굴 bbox 중심과 nose 키포인트로 pose를 매칭."""
        import math

        if not kp_list:
            return None
        if len(kp_list) == 1 or face_data is None:
            return kp_list[0]

        faces = getattr(face_data, "faces", None)
        if not faces:
            return kp_list[0]
        face = max(faces, key=lambda f: getattr(f, "area_ratio", 0.0))
        bbox = getattr(face, "bbox", None)
        if bbox is None or len(bbox) < 4:
            return kp_list[0]
        fx = bbox[0] + bbox[2] / 2.0
        fy = bbox[1] + bbox[3] / 2.0

        best, best_dist = kp_list[0], float("inf")
        for person in kp_list:
            kpts = person.get("keypoints", []) if isinstance(person, dict) else getattr(person, "keypoints", [])
            img_size = person.get("image_size", (1, 1)) if isinstance(person, dict) else getattr(person, "image_size", (1, 1))
            iw = float(img_size[0]) if img_size[0] > 0 else 1.0
            ih = float(img_size[1]) if img_size[1] > 0 else 1.0
            if kpts and len(kpts) > 0:
                pt = kpts[0]  # nose
                if hasattr(pt, '__len__') and len(pt) >= 2 and pt[0] > 0 and pt[1] > 0:
                    d = math.hypot(pt[0] / iw - fx, pt[1] / ih - fy)
                    if d < best_dist:
                        best_dist = d
                        best = person
        return best

    @processing_step(
        name="body_embed",
        description="Extract body crop and compute embedding",
        backend="DINOv2",
        input_type="Frame + PoseOutput",
        output_type="Optional[np.ndarray] (384,)",
    )
    def _embed_body(
        self, image, pose_data, face_data=None,
    ) -> tuple[Optional["np.ndarray"], Optional[tuple[int, int, int, int]]]:
        """Extract upper-body crop from pose and compute embedding.

        When face_data is available, matches the pose whose nose keypoint
        is closest to the main face bbox center (filters out bystanders).
        """
        import numpy as np

        if pose_data is None:
            return None, None

        kp_list = getattr(pose_data, "keypoints", None)
        if not kp_list or not isinstance(kp_list, list):
            return None, None

        person = self._match_pose_to_face(kp_list, face_data)
        if person is None:
            return None, None

        keypoints = None
        if isinstance(person, dict) and "keypoints" in person:
            keypoints = np.asarray(person["keypoints"], dtype=np.float32)
        elif hasattr(person, "keypoints"):
            keypoints = np.asarray(person.keypoints, dtype=np.float32)

        if keypoints is None or len(keypoints) < 13:
            return None, None

        crop, actual_box = body_crop(image, keypoints)

        smoothed = self._body_smoother.update(actual_box)

        embedding = self._backend.embed(crop)
        return embedding, smoothed

    def process(
        self,
        frame: Frame,
        deps: Optional[Dict[str, Observation]] = None,
    ) -> Optional[Observation]:
        if self._backend is None:
            raise RuntimeError("Analyzer not initialized")

        face_obs = deps.get("face.detect") if deps else None
        if face_obs is None:
            logger.debug("BodyEmbedAnalyzer: no face.detect dependency")
            return None

        face_data = face_obs.data

        pose_obs = deps.get("body.pose") if deps else None
        pose_data = pose_obs.data if pose_obs else None

        image = frame.data

        self._step_timings = {}

        e_body, body_box = self._embed_body(image, pose_data, face_data)

        timing = self._step_timings.copy() if self._step_timings else None
        self._step_timings = None

        h, w = image.shape[:2]
        output = BodyEmbedOutput(
            e_body=e_body,
            body_crop_box=body_box,
            image_size=(w, h),
        )

        return Observation(
            source=self.name,
            frame_id=frame.frame_id,
            t_ns=frame.t_src_ns,
            signals={
                "has_body_embed": e_body is not None,
            },
            data=output,
            metadata={
                "_metrics": {
                    "body_embed_computed": e_body is not None,
                }
            },
            timing=timing,
        )

    def annotate(self, obs):
        """Return marks for body crop region and embedding status."""
        if obs is None or obs.data is None:
            return []

        from vpx.sdk.marks import BBoxMark, LabelMark

        data = obs.data
        marks = []

        img_w, img_h = data.image_size if data.image_size else (1, 1)

        if data.body_crop_box is not None:
            x, y, w, h = data.body_crop_box
            marks.append(BBoxMark(
                x=x / img_w, y=y / img_h,
                w=w / img_w, h=h / img_h,
                label="body_emb",
                color=(0, 200, 0),  # green
                thickness=1,
            ))

        if obs.signals.get("has_body_embed"):
            marks.append(LabelMark(
                text="emb: body",
                x=0.01, y=0.09,
                color=(0, 200, 0),
                background=(30, 30, 30),
                font_scale=0.4,
            ))

        return marks
