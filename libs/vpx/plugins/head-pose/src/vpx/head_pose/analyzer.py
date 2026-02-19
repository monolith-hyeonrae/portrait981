"""Head pose estimation analyzer - depends on face.detect."""

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
from vpx.head_pose.backends.base import HeadPoseBackend
from vpx.head_pose.output import HeadPoseOutput
from vpx.head_pose.types import HeadPoseEstimate

logger = logging.getLogger(__name__)


class HeadPoseAnalyzer(Module):
    """Analyzer for precise head pose estimation using 6DRepNet.

    Outputs yaw/pitch/roll in degrees with ~3.5 degree MAE accuracy,
    significantly better than geometric 5-point landmark estimation.

    depends: ["face.detect"]
    """

    depends = ["face.detect"]

    def __init__(
        self,
        pose_backend: Optional[HeadPoseBackend] = None,
        device: str = "cuda:0",
    ):
        self._device = device
        self._pose_backend = pose_backend
        self._initialized = False
        self._step_timings: Optional[Dict[str, float]] = None

    @property
    def name(self) -> str:
        return "head.pose"

    @property
    def capabilities(self) -> ModuleCapabilities:
        return ModuleCapabilities(
            flags=Capability.GPU,
            gpu_memory_mb=512,
            init_time_sec=2.0,
            resource_groups=frozenset({"onnxruntime"}),
            required_extras=frozenset({"vpx-head-pose"}),
        )

    @property
    def processing_steps(self) -> List[ProcessingStep]:
        return get_processing_steps(self)

    def initialize(self) -> None:
        if self._initialized:
            return

        if self._pose_backend is None:
            try:
                from vpx.head_pose.backends.repnet6d import RepNet6DBackend
                self._pose_backend = RepNet6DBackend()
                self._pose_backend.initialize(self._device)
                logger.info("HeadPoseAnalyzer using RepNet6DBackend")
            except (ImportError, FileNotFoundError) as e:
                logger.warning("6DRepNet backend not available: %s", e)
            except Exception as e:
                logger.warning("Failed to initialize 6DRepNet: %s", e)

        self._initialized = True

    def cleanup(self) -> None:
        if self._pose_backend is not None:
            self._pose_backend.cleanup()
        logger.info("HeadPoseAnalyzer cleaned up")

    @processing_step(
        name="head_pose",
        description="Head pose estimation per face (yaw/pitch/roll)",
        backend="6DRepNet",
        input_type="Frame + List[DetectedFace]",
        output_type="List[HeadPoseEstimate]",
    )
    def _estimate_poses(self, image, detected_faces) -> List[HeadPoseEstimate]:
        if self._pose_backend is None:
            return []
        return self._pose_backend.estimate(image, detected_faces)

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
                data=HeadPoseOutput(),
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
        estimates = self._estimate_poses(frame.data, detected_faces)
        timing = self._step_timings.copy() if self._step_timings else None
        self._step_timings = None

        # Aggregate signals from first (main) face
        signals: Dict[str, float] = {}
        if estimates:
            main = estimates[0]
            signals["head_yaw"] = main.yaw
            signals["head_pitch"] = main.pitch
            signals["head_roll"] = main.roll

        return Observation(
            source=self.name,
            frame_id=frame.frame_id,
            t_ns=frame.t_src_ns,
            signals=signals,
            data=HeadPoseOutput(
                estimates=estimates,
                face_bboxes=face_bboxes,
            ),
            metadata={"_metrics": {"faces_analyzed": len(estimates)}},
            timing=timing,
        )

    def annotate(self, obs):
        """Return AxisMark (3D axes) + LabelMark (Y/P/R text) per face."""
        if obs is None or obs.data is None:
            return []
        from vpx.sdk.marks import AxisMark, LabelMark

        marks = []
        data: HeadPoseOutput = obs.data
        for i, bbox in enumerate(data.face_bboxes):
            if i >= len(data.estimates):
                break
            est = data.estimates[i]
            # 3D axis from face center (6DRepNet-style)
            cx = bbox[0] + bbox[2] / 2
            cy = bbox[1] + bbox[3] / 2
            axis_size = bbox[2] * 0.8  # proportional to face width
            marks.append(AxisMark(
                cx=cx, cy=cy,
                yaw=est.yaw, pitch=est.pitch, roll=est.roll,
                size=axis_size, thickness=2,
            ))
            # Text label above bbox
            text = f"Y:{est.yaw:.0f} P:{est.pitch:.0f} R:{est.roll:.0f}"
            marks.append(LabelMark(
                text=text,
                x=bbox[0],
                y=bbox[1] - 0.02,
                color=(255, 255, 0),  # cyan BGR
                font_scale=0.35,
            ))
        return marks
