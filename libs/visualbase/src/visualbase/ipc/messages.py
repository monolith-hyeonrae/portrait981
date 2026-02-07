"""OBS and TRIG message parsing and serialization.

Message formats follow the spec from plan.md:

OBS (B→C):
    OBS src=face frame=1234 t_ns=1234567890 faces=2 \
      f0=id:0,conf:0.95,x:100,y:100,w:200,h:200,expr:0.8 \
      f1=id:1,conf:0.92,x:400,y:100,w:180,h:180,expr:0.3

TRIG (C→A):
    TRIG label=PORTRAIT_HIGHLIGHT t_start_ns=1234567890 t_end_ns=1239567890 \
      faces=2 score=0.85 reason=expr_spike
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import re
import logging

logger = logging.getLogger(__name__)


@dataclass
class FaceData:
    """Single face observation data."""

    id: int
    conf: float
    x: float  # Normalized (0-1) or pixel coordinates
    y: float
    w: float
    h: float
    expr: float = 0.0  # Expression intensity
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0

    def to_string(self) -> str:
        """Serialize to message format."""
        parts = [
            f"id:{self.id}",
            f"conf:{self.conf:.3f}",
            f"x:{self.x:.4f}",
            f"y:{self.y:.4f}",
            f"w:{self.w:.4f}",
            f"h:{self.h:.4f}",
            f"expr:{self.expr:.3f}",
        ]
        if self.yaw != 0 or self.pitch != 0:
            parts.extend([f"yaw:{self.yaw:.1f}", f"pitch:{self.pitch:.1f}"])
        return ",".join(parts)

    @classmethod
    def from_string(cls, s: str) -> "FaceData":
        """Parse from message format."""
        parts = dict(p.split(":") for p in s.split(","))
        return cls(
            id=int(parts["id"]),
            conf=float(parts["conf"]),
            x=float(parts["x"]),
            y=float(parts["y"]),
            w=float(parts["w"]),
            h=float(parts["h"]),
            expr=float(parts.get("expr", 0)),
            yaw=float(parts.get("yaw", 0)),
            pitch=float(parts.get("pitch", 0)),
            roll=float(parts.get("roll", 0)),
        )


@dataclass
class PoseData:
    """Single pose observation data."""

    id: int
    conf: float
    hand_raised: bool = False
    hand_wave: bool = False
    wave_count: int = 0

    def to_string(self) -> str:
        """Serialize to message format."""
        parts = [
            f"id:{self.id}",
            f"conf:{self.conf:.3f}",
            f"hr:{int(self.hand_raised)}",
            f"hw:{int(self.hand_wave)}",
        ]
        if self.wave_count > 0:
            parts.append(f"wc:{self.wave_count}")
        return ",".join(parts)

    @classmethod
    def from_string(cls, s: str) -> "PoseData":
        """Parse from message format."""
        parts = dict(p.split(":") for p in s.split(","))
        return cls(
            id=int(parts["id"]),
            conf=float(parts["conf"]),
            hand_raised=bool(int(parts.get("hr", 0))),
            hand_wave=bool(int(parts.get("hw", 0))),
            wave_count=int(parts.get("wc", 0)),
        )


@dataclass
class QualityData:
    """Quality observation data."""

    blur: float  # Higher = sharper (laplacian variance)
    brightness: float  # 0-255
    contrast: float  # 0-1
    gate_open: bool = True

    def to_string(self) -> str:
        """Serialize to message format."""
        return (
            f"blur:{self.blur:.1f},bright:{self.brightness:.1f},"
            f"contrast:{self.contrast:.3f},gate:{int(self.gate_open)}"
        )

    @classmethod
    def from_string(cls, s: str) -> "QualityData":
        """Parse from message format."""
        parts = dict(p.split(":") for p in s.split(","))
        return cls(
            blur=float(parts["blur"]),
            brightness=float(parts["bright"]),
            contrast=float(parts["contrast"]),
            gate_open=bool(int(parts.get("gate", 1))),
        )


@dataclass
class FaceOBS:
    """Face observation message."""

    frame_id: int
    t_ns: int
    faces: List[FaceData] = field(default_factory=list)

    def to_message(self) -> str:
        """Serialize to OBS message format."""
        parts = [
            "OBS",
            "src=face",
            f"frame={self.frame_id}",
            f"t_ns={self.t_ns}",
            f"faces={len(self.faces)}",
        ]
        for i, face in enumerate(self.faces):
            parts.append(f"f{i}={face.to_string()}")
        return " ".join(parts)


@dataclass
class PoseOBS:
    """Pose observation message."""

    frame_id: int
    t_ns: int
    poses: List[PoseData] = field(default_factory=list)

    def to_message(self) -> str:
        """Serialize to OBS message format."""
        parts = [
            "OBS",
            "src=pose",
            f"frame={self.frame_id}",
            f"t_ns={self.t_ns}",
            f"poses={len(self.poses)}",
        ]
        for i, pose in enumerate(self.poses):
            parts.append(f"p{i}={pose.to_string()}")
        return " ".join(parts)


@dataclass
class QualityOBS:
    """Quality observation message."""

    frame_id: int
    t_ns: int
    quality: QualityData = field(default_factory=lambda: QualityData(0, 0, 0))

    def to_message(self) -> str:
        """Serialize to OBS message format."""
        return (
            f"OBS src=quality frame={self.frame_id} t_ns={self.t_ns} "
            f"{self.quality.to_string()}"
        )


@dataclass
class OBSMessage:
    """Parsed OBS message (generic container)."""

    src: str  # "face", "pose", "quality"
    frame_id: int
    t_ns: int
    faces: List[FaceData] = field(default_factory=list)
    poses: List[PoseData] = field(default_factory=list)
    quality: Optional[QualityData] = None
    raw: str = ""


@dataclass
class TRIGMessage:
    """TRIG message for clip extraction trigger."""

    label: str
    t_start_ns: int
    t_end_ns: int
    faces: int = 0
    score: float = 0.0
    reason: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_message(self) -> str:
        """Serialize to TRIG message format."""
        parts = [
            "TRIG",
            f"label={self.label}",
            f"t_start_ns={self.t_start_ns}",
            f"t_end_ns={self.t_end_ns}",
            f"faces={self.faces}",
            f"score={self.score:.3f}",
            f"reason={self.reason}",
        ]
        return " ".join(parts)


def parse_obs_message(message: str) -> Optional[OBSMessage]:
    """Parse an OBS message string.

    Args:
        message: Raw OBS message string.

    Returns:
        Parsed OBSMessage or None on error.
    """
    if not message.startswith("OBS "):
        return None

    try:
        # Parse key=value pairs
        parts = message.split()
        data = {}
        for part in parts[1:]:  # Skip "OBS"
            if "=" in part:
                key, value = part.split("=", 1)
                data[key] = value

        src = data.get("src", "")
        frame_id = int(data.get("frame", 0))
        t_ns = int(data.get("t_ns", 0))

        obs = OBSMessage(
            src=src,
            frame_id=frame_id,
            t_ns=t_ns,
            raw=message,
        )

        if src == "face":
            # Parse face data
            num_faces = int(data.get("faces", 0))
            for i in range(num_faces):
                face_str = data.get(f"f{i}", "")
                if face_str:
                    obs.faces.append(FaceData.from_string(face_str))

        elif src == "pose":
            # Parse pose data
            num_poses = int(data.get("poses", 0))
            for i in range(num_poses):
                pose_str = data.get(f"p{i}", "")
                if pose_str:
                    obs.poses.append(PoseData.from_string(pose_str))

        elif src == "quality":
            # Parse quality data (inline, not in separate fields)
            # Quality data is embedded directly in the message
            quality_parts = {}
            for part in parts:
                if ":" in part:
                    for kv in part.split(","):
                        if ":" in kv:
                            k, v = kv.split(":", 1)
                            quality_parts[k] = v

            if quality_parts:
                obs.quality = QualityData(
                    blur=float(quality_parts.get("blur", 0)),
                    brightness=float(quality_parts.get("bright", 0)),
                    contrast=float(quality_parts.get("contrast", 0)),
                    gate_open=bool(int(quality_parts.get("gate", 1))),
                )

        return obs

    except Exception as e:
        logger.warning(f"Failed to parse OBS message: {e}")
        return None


def parse_trig_message(message: str) -> Optional[TRIGMessage]:
    """Parse a TRIG message string.

    Args:
        message: Raw TRIG message string.

    Returns:
        Parsed TRIGMessage or None on error.
    """
    if not message.startswith("TRIG "):
        return None

    try:
        # Parse key=value pairs
        parts = message.split()
        data = {}
        for part in parts[1:]:  # Skip "TRIG"
            if "=" in part:
                key, value = part.split("=", 1)
                data[key] = value

        return TRIGMessage(
            label=data.get("label", ""),
            t_start_ns=int(data.get("t_start_ns", 0)),
            t_end_ns=int(data.get("t_end_ns", 0)),
            faces=int(data.get("faces", 0)),
            score=float(data.get("score", 0)),
            reason=data.get("reason", ""),
        )

    except Exception as e:
        logger.warning(f"Failed to parse TRIG message: {e}")
        return None
