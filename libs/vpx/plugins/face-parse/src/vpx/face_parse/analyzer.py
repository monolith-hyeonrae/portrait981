"""Face parsing analyzer — pixel-precise face segmentation.

FaceParseAnalyzer: depends on face.detect.
Uses BiSeNet (CelebAMask-HQ, ResNet18 backbone) to produce
per-face binary masks separating skin from background/hair/clothing.

depends: ["face.detect"]
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Dict, List, Optional

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
from vpx.face_detect.output import FaceDetectOutput
from vpx.face_parse.backends.base import FaceParseBackend
from vpx.face_parse.output import FaceParseOutput, FaceParseResult

logger = logging.getLogger(__name__)


class FaceParseAnalyzer(Module):
    """Analyzer for pixel-precise face segmentation using BiSeNet.

    Produces binary face masks (skin region) for each detected face,
    enabling downstream analyzers (face.quality) to compute metrics
    on face-only pixels instead of the full bounding box crop.

    depends: ["face.detect"]
    """

    depends = ["face.detect"]

    def __init__(
        self,
        parse_backend: Optional[FaceParseBackend] = None,
        device: str = "cuda:0",
    ):
        self._device = device
        self._parse_backend = parse_backend
        self._initialized = False
        self._step_timings: Optional[Dict[str, float]] = None

    @property
    def name(self) -> str:
        return "face.parse"

    @property
    def capabilities(self) -> ModuleCapabilities:
        return ModuleCapabilities(
            flags=Capability.GPU | Capability.DETERMINISTIC,
            gpu_memory_mb=200,
            init_time_sec=1.0,
            required_extras=frozenset({"vpx-face-parse"}),
        )

    @property
    def processing_steps(self) -> List[ProcessingStep]:
        return get_processing_steps(self)

    def initialize(self) -> None:
        if self._initialized:
            return

        if self._parse_backend is None:
            try:
                from vpx.face_parse.backends.bisenet import BiSeNetBackend
                self._parse_backend = BiSeNetBackend()
                self._parse_backend.initialize(self._device)
                logger.info("FaceParseAnalyzer using BiSeNetBackend")
            except (ImportError, FileNotFoundError) as e:
                logger.warning("BiSeNet backend not available: %s", e)
            except Exception as e:
                logger.warning("Failed to initialize BiSeNet: %s", e)

        self._initialized = True

    def cleanup(self) -> None:
        if self._parse_backend is not None:
            self._parse_backend.cleanup()
        self._initialized = False
        logger.info("FaceParseAnalyzer cleaned up")

    @processing_step(
        name="face_parse",
        description="Segment face skin regions per detected face",
        backend="BiSeNet",
        input_type="Frame + List[DetectedFace]",
        output_type="List[FaceParseResult]",
    )
    def _segment_faces(
        self, image, detected_faces
    ) -> List[FaceParseResult]:
        if self._parse_backend is None:
            return []
        return self._parse_backend.segment(image, detected_faces)

    def process(
        self,
        frame: "Frame",
        deps: Optional[Dict[str, Observation]] = None,
    ) -> Optional[Observation]:
        face_obs = deps.get("face.detect") if deps else None
        if face_obs is None:
            return None

        face_data: FaceDetectOutput = face_obs.data
        if not face_data:
            return None

        detected_faces = face_data.detected_faces
        if not detected_faces:
            return Observation(
                source=self.name,
                frame_id=frame.frame_id,
                t_ns=frame.t_src_ns,
                signals={"faces_parsed": 0},
                data=FaceParseOutput(),
            )

        self._step_timings = {}
        results = self._segment_faces(frame.data, detected_faces)
        timing = self._step_timings.copy() if self._step_timings else None
        self._step_timings = None

        # Build normalized bboxes from crop_box
        image_h, image_w = frame.data.shape[:2]
        face_bboxes = []
        for r in results:
            bx, by, bw, bh = r.crop_box
            if (bw, bh) != (0, 0):
                face_bboxes.append((
                    bx / image_w, by / image_h,
                    bw / image_w, bh / image_h,
                ))
            else:
                face_bboxes.append((0.0, 0.0, 0.0, 0.0))

        return Observation(
            source=self.name,
            frame_id=frame.frame_id,
            t_ns=frame.t_src_ns,
            signals={"faces_parsed": len(results)},
            data=FaceParseOutput(results=results, face_bboxes=face_bboxes),
            metadata={"_metrics": {"faces_parsed": len(results)}},
            timing=timing,
        )

    def annotate(self, obs):
        """Return crop box BBoxMark + face_id LabelMark for each parsed face."""
        if obs is None or obs.data is None:
            return []

        from vpx.sdk.marks import BBoxMark, LabelMark

        marks = []
        for i, result in enumerate(obs.data.results):
            if i >= len(obs.data.face_bboxes):
                break
            bbox = obs.data.face_bboxes[i]
            if bbox == (0.0, 0.0, 0.0, 0.0):
                continue
            marks.append(BBoxMark(
                x=bbox[0], y=bbox[1], w=bbox[2], h=bbox[3],
                label=f"parse:{result.face_id}",
                color=(0, 255, 255),
                thickness=1,
            ))

        return marks
