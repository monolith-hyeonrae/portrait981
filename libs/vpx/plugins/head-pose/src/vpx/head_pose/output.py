"""Output type for head pose analyzer."""

from dataclasses import dataclass, field
from typing import List, Tuple

from vpx.head_pose.types import HeadPoseEstimate


@dataclass
class HeadPoseOutput:
    """Output from HeadPoseAnalyzer.

    Attributes:
        estimates: Per-face head pose estimates (yaw, pitch, roll in degrees).
        face_bboxes: Per-face normalized bounding boxes (x, y, w, h) for overlay positioning.
    """

    estimates: List[HeadPoseEstimate] = field(default_factory=list)
    face_bboxes: List[Tuple[float, float, float, float]] = field(default_factory=list)


__all__ = ["HeadPoseOutput"]
