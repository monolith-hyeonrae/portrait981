"""Face AU detection analyzer - depends on face.detect."""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional
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
from vpx.face_detect.output import FaceDetectOutput
from vpx.face_au.backends.base import FaceAUBackend, FaceAUResult
from vpx.face_au.output import FaceAUOutput

logger = logging.getLogger(__name__)


class FaceAUAnalyzer(Module):
    """Analyzer for face Action Unit detection using LibreFace.

    Detects 12 DISFA Action Units (AU1-AU26) with intensity estimation.
    Key AUs for portrait pivot: AU12 (smile), AU25 (lips part), AU26 (jaw drop).

    depends: ["face.detect"]
    """

    depends = ["face.detect"]

    def __init__(
        self,
        au_backend: Optional[FaceAUBackend] = None,
        device: str = "cuda:0",
    ):
        self._device = device
        self._au_backend = au_backend
        self._initialized = False
        self._step_timings: Optional[Dict[str, float]] = None

    @property
    def name(self) -> str:
        return "face.au"

    @property
    def capabilities(self) -> ModuleCapabilities:
        return ModuleCapabilities(
            flags=Capability.GPU,
            gpu_memory_mb=256,
            init_time_sec=1.5,
            resource_groups=frozenset({"onnxruntime"}),
            required_extras=frozenset({"vpx-face-au"}),
        )

    @property
    def processing_steps(self) -> List[ProcessingStep]:
        return get_processing_steps(self)

    def initialize(self) -> None:
        if self._initialized:
            return

        if self._au_backend is None:
            try:
                from vpx.face_au.backends.libreface import LibreFaceBackend
                self._au_backend = LibreFaceBackend()
                self._au_backend.initialize(self._device)
                logger.info("FaceAUAnalyzer using LibreFaceBackend")
            except (ImportError, FileNotFoundError) as e:
                logger.warning("LibreFace backend not available: %s", e)
            except Exception as e:
                logger.warning("Failed to initialize LibreFace: %s", e)

        self._initialized = True

    def cleanup(self) -> None:
        if self._au_backend is not None:
            self._au_backend.cleanup()
        logger.info("FaceAUAnalyzer cleaned up")

    @processing_step(
        name="au_detection",
        description="Action Unit intensity detection per face",
        backend="LibreFace",
        input_type="Frame + List[DetectedFace]",
        output_type="List[FaceAUResult]",
    )
    def _detect_aus(self, image, detected_faces) -> List[FaceAUResult]:
        if self._au_backend is None:
            return []
        return self._au_backend.analyze(image, detected_faces)

    def process(
        self,
        frame: Frame,
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
                signals={},
                data=FaceAUOutput(),
                metadata={"_metrics": {"faces_analyzed": 0}},
            )

        # Extract normalized bboxes for overlay positioning
        image_h, image_w = frame.data.shape[:2]
        face_bboxes = []
        for df in detected_faces:
            bx, by, bw, bh = df.bbox
            face_bboxes.append((
                bx / image_w, by / image_h,
                bw / image_w, bh / image_h,
            ))

        self._step_timings = {}
        au_results = self._detect_aus(frame.data, detected_faces)
        timing = self._step_timings.copy() if self._step_timings else None
        self._step_timings = None

        # Build output
        all_intensities = []
        all_presence = []
        signals: Dict[str, float] = {}

        for i, result in enumerate(au_results):
            all_intensities.append(result.au_intensities)
            all_presence.append(result.au_presence)

        # Aggregate signals from first (main) face
        if all_intensities:
            main_au = all_intensities[0]
            for au_name, intensity in main_au.items():
                signals[f"au_{au_name.lower()}"] = intensity

        return Observation(
            source=self.name,
            frame_id=frame.frame_id,
            t_ns=frame.t_src_ns,
            signals=signals,
            data=FaceAUOutput(
                au_intensities=all_intensities,
                au_presence=all_presence,
                face_bboxes=face_bboxes,
            ),
            metadata={"_metrics": {"faces_analyzed": len(au_results)}},
            timing=timing,
        )

    def annotate(self, obs):
        """Return BarMarks for key AU intensities below expression bars."""
        if obs is None or obs.data is None:
            return []
        from vpx.sdk.marks import BarMark

        marks = []
        data: FaceAUOutput = obs.data
        for i, bbox in enumerate(data.face_bboxes):
            if i >= len(data.au_intensities):
                break
            au = data.au_intensities[i]
            # Zone 2: AU bars (below expression bars)
            bar_y = bbox[1] + bbox[3] + 0.066
            bar_w = bbox[2]
            key_aus = [
                ("AU12", (200, 0, 200), "smile"),
                ("AU25", (255, 0, 180), "lips"),
                ("AU26", (255, 0, 140), "jaw"),
            ]
            for j, (au_name, color, label) in enumerate(key_aus):
                intensity = au.get(au_name, 0.0)
                value = min(1.0, intensity / 5.0)
                marks.append(BarMark(
                    x=bbox[0], y=bar_y + j * 0.013,
                    w=bar_w, value=value, color=color, label=label,
                ))
        return marks
