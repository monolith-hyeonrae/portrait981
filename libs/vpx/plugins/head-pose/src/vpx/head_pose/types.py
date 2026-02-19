"""Head pose estimation types."""

from dataclasses import dataclass


@dataclass
class HeadPoseEstimate:
    """Head pose angles for a single face.

    All angles in degrees. Coordinate system:
    - yaw: left(-) / right(+) head turn
    - pitch: down(-) / up(+) head tilt
    - roll: counter-clockwise(-) / clockwise(+) head roll
    """

    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0


__all__ = ["HeadPoseEstimate"]
