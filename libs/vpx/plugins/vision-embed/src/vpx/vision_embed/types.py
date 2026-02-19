"""Output types for face/body embedding analyzers."""

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class FaceEmbedOutput:
    """Output from FaceEmbedAnalyzer.

    Attributes:
        e_face: L2-normalized face embedding (384,) or None if no face.
        face_crop_box: Face crop bounding box (x, y, w, h) in pixels or None.
        image_size: (width, height) of the source frame in pixels.
    """

    e_face: Optional[np.ndarray] = None  # (384,) L2-normalized
    face_crop_box: Optional[tuple[int, int, int, int]] = None
    image_size: Optional[tuple[int, int]] = None  # (w, h)


@dataclass
class BodyEmbedOutput:
    """Output from BodyEmbedAnalyzer.

    Attributes:
        e_body: L2-normalized body embedding (384,) or None if no body pose.
        body_crop_box: Body crop bounding box (x, y, w, h) in pixels or None.
        image_size: (width, height) of the source frame in pixels.
    """

    e_body: Optional[np.ndarray] = None  # (384,) L2-normalized
    body_crop_box: Optional[tuple[int, int, int, int]] = None
    image_size: Optional[tuple[int, int]] = None  # (w, h)


__all__ = ["FaceEmbedOutput", "BodyEmbedOutput"]
