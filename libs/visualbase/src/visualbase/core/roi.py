"""ROI (Region of Interest) management.

Defines ROI specifications and crops. Each ROI is a named, formally defined
region derived from a source frame. Analyzers reference ROIs by name instead
of computing crops independently.

Hierarchy (portrait981):
    Global ⊃ Body ⊃ Portrait ⊃ Head ⊃ Face

Usage:
    from visualbase.core.roi import ROISpec, ROICrop

    # Define specs (typically once at pipeline setup)
    PORTRAIT = ROISpec("portrait", expand=1.3, size=(512, 512))
    FACE = ROISpec("face", expand=1.0, size=(224, 224))

    # Create crop from detected bbox
    crop = ROICrop.from_bbox(PORTRAIT, frame_data, bbox=(100, 50, 300, 350), frame_id=0)

    # Coordinate conversion
    frame_xy = crop.to_frame(128, 256)    # ROI pixel → frame pixel
    roi_xy = crop.from_frame(200, 150)    # frame pixel → ROI pixel
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np


@dataclass(frozen=True, slots=True)
class ROISpec:
    """Formal ROI definition. Referenced by name.

    Attributes:
        name: Unique identifier ("face", "portrait", "head", "body", "global").
        expand: Expansion ratio around source bbox (1.0 = tight, 1.3 = 30% margin).
        size: Target (width, height) after resize. None = no resize (keep crop size).
        aspect_ratio: Enforce aspect ratio (w/h). None = free. 1.0 = square, 0.8 = 4:5.
    """
    name: str
    expand: float = 1.0
    size: Optional[Tuple[int, int]] = None
    aspect_ratio: Optional[float] = None


@dataclass(frozen=True, slots=True)
class ROICrop:
    """A realized ROI crop from a specific frame.

    Holds the cropped image and the coordinate mapping back to the source frame.
    Analyzers should use to_frame()/from_frame() instead of manual calculation.

    Attributes:
        spec: The ROISpec this crop was created from.
        image: Cropped (and optionally resized) BGR image.
        crop_box: Actual crop region in source frame pixel coordinates (x1, y1, x2, y2).
        frame_id: Source frame identifier for synchronization.
    """
    spec: ROISpec
    image: np.ndarray
    crop_box: Tuple[int, int, int, int]  # (x1, y1, x2, y2) in frame pixels
    frame_id: int

    @classmethod
    def from_bbox(
        cls,
        spec: ROISpec,
        frame_data: np.ndarray,
        bbox: Tuple[int, int, int, int],
        frame_id: int = 0,
    ) -> ROICrop:
        """Create ROICrop from a detection bbox.

        Applies expansion, aspect ratio enforcement, and optional resize.

        Args:
            spec: ROI specification.
            frame_data: Source frame BGR image.
            bbox: Detection bbox (x1, y1, x2, y2) in frame pixels.
            frame_id: Frame identifier.
        """
        x1, y1, x2, y2 = bbox
        h, w = frame_data.shape[:2]

        # Center and dimensions
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        bw = x2 - x1
        bh = y2 - y1

        # Expand
        bw *= spec.expand
        bh *= spec.expand

        # Enforce aspect ratio
        if spec.aspect_ratio is not None:
            target_ratio = spec.aspect_ratio
            current_ratio = bw / bh if bh > 0 else 1.0
            if current_ratio > target_ratio:
                bh = bw / target_ratio
            else:
                bw = bh * target_ratio

        # Compute crop box (clamp to frame)
        cx1 = max(0, int(cx - bw / 2))
        cy1 = max(0, int(cy - bh / 2))
        cx2 = min(w, int(cx + bw / 2))
        cy2 = min(h, int(cy + bh / 2))

        # Crop
        cropped = frame_data[cy1:cy2, cx1:cx2]

        # Resize
        if spec.size is not None and cropped.size > 0:
            cropped = cv2.resize(cropped, spec.size)

        return cls(
            spec=spec,
            image=cropped,
            crop_box=(cx1, cy1, cx2, cy2),
            frame_id=frame_id,
        )

    def to_frame(self, roi_x: float, roi_y: float) -> Tuple[float, float]:
        """Convert ROI pixel coordinate → source frame pixel coordinate."""
        x1, y1, x2, y2 = self.crop_box
        roi_h, roi_w = self.image.shape[:2]
        if roi_w == 0 or roi_h == 0:
            return float(x1), float(y1)
        frame_x = x1 + roi_x / roi_w * (x2 - x1)
        frame_y = y1 + roi_y / roi_h * (y2 - y1)
        return frame_x, frame_y

    def from_frame(self, frame_x: float, frame_y: float) -> Tuple[float, float]:
        """Convert source frame pixel coordinate → ROI pixel coordinate."""
        x1, y1, x2, y2 = self.crop_box
        roi_h, roi_w = self.image.shape[:2]
        if (x2 - x1) == 0 or (y2 - y1) == 0:
            return 0.0, 0.0
        roi_x = (frame_x - x1) / (x2 - x1) * roi_w
        roi_y = (frame_y - y1) / (y2 - y1) * roi_h
        return roi_x, roi_y

    @property
    def width(self) -> int:
        return self.image.shape[1] if self.image.size > 0 else 0

    @property
    def height(self) -> int:
        return self.image.shape[0] if self.image.size > 0 else 0
