"""Coordinate system definitions.

Every spatial value in the pipeline must be tagged with its coordinate space
and ROI context. This prevents the recurring class of bugs where a value
computed in one ROI's pixel space is interpreted in another.

Usage:
    from visualbase.core.coordinate import Coord, Coord3D, Space, Space3D

    # 2D: always (value, space, roi)
    nose = Coord((0.45, 0.62), Space.NORM, roi="global")
    seg_pt = Coord((128, 256), Space.PIXEL, roi="portrait")

    # 3D: always (value, space, convention)
    sh_dir = Coord3D((0.3, -0.2, 0.8), Space3D.IMAGE_XYZ, convention="DPR")
    pose = Coord3D((15.0, -5.0, 2.0), Space3D.IMAGE_XYZ, convention="6DRepNet")
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Tuple, Union


class Space(Enum):
    """2D coordinate unit."""
    PIXEL = "pixel"   # (0,0)=top-left, unit=pixels
    NORM = "norm"     # (0,0)=top-left, (1,1)=bottom-right


class Space3D(Enum):
    """3D coordinate convention."""
    IMAGE_XYZ = "image_xyz"   # Image-aligned: X+=right, Y+=depth, Z+=up
    FACE_XYZ = "face_xyz"     # Face-aligned: mirror of image (SfSNet convention)


@dataclass(frozen=True, slots=True)
class Coord:
    """2D coordinate tagged with space and ROI.

    Attributes:
        value: Coordinate data (tuple, ndarray, float, etc.)
        space: PIXEL or NORM
        roi: Which ROI this coordinate is relative to ("global", "portrait", "face", etc.)
    """
    value: Any
    space: Space
    roi: str = "global"

    def __repr__(self) -> str:
        return f"Coord({self.value}, {self.space.value}, roi={self.roi!r})"


@dataclass(frozen=True, slots=True)
class Coord3D:
    """3D coordinate tagged with space and model convention.

    Attributes:
        value: 3D data (tuple, ndarray)
        space: IMAGE_XYZ or FACE_XYZ
        convention: Model-specific sign convention ("6DRepNet", "DPR", etc.)
    """
    value: Any
    space: Space3D
    convention: str = ""

    def __repr__(self) -> str:
        conv = f", {self.convention}" if self.convention else ""
        return f"Coord3D({self.value}, {self.space.value}{conv})"
