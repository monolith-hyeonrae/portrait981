"""Crop utilities for head/scene regions and bbox smoothing."""

from __future__ import annotations

from typing import Literal, Optional

import cv2
import numpy as np

# 지원 크롭 비율 — "1:1" 정방형 / "4:5" 포트레이트 세로형
CropRatio = Literal["1:1", "4:5"]


def face_crop(
    image: np.ndarray,
    bbox: tuple[int, int, int, int],
    expand: float = 1.5,
    output_size: int = 224,
    crop_ratio: CropRatio = "1:1",
    y_shift: float = 0.0,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Extract an expanded face crop from the image.

    Args:
        image: Full frame (H, W, 3) BGR.
        bbox: Face bounding box (x, y, w, h) in pixels.
        expand: Expansion factor around the face center.
        output_size: Output width in pixels.
        crop_ratio: Aspect ratio of the crop window and output image.
            "1:1" → square (output_size × output_size).
            "4:5" → portrait (output_size × output_size×5//4).
        y_shift: Vertical shift of crop center as a fraction of bbox height.
            Positive = downward (e.g., 0.3 shifts by 30% of face height).
            Useful for portrait crops that should cover head-to-shoulders
            rather than being centered on the nose.

    Returns:
        Tuple of (crop, actual_box):
          - crop: Resized image (out_h, output_size, 3).
          - actual_box: Clamped bounding box (x, y, w, h) used for the crop.
    """
    h, w = image.shape[:2]
    bx, by, bw, bh = bbox

    # Expand around center (optionally shifted downward)
    cx = bx + bw / 2
    cy = by + bh / 2 + y_shift * bh
    side = max(bw, bh) * expand
    half = side / 2

    x1 = max(0, int(cx - half))
    x2 = min(w, int(cx + half))

    if crop_ratio == "4:5":
        # height = side × 5/4  →  half_h = half × 5/4
        half_h = half * 5 / 4
        out_h = output_size * 5 // 4
    else:
        half_h = half
        out_h = output_size

    y1 = max(0, int(cy - half_h))
    y2 = min(h, int(cy + half_h))

    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        crop = np.zeros((out_h, output_size, 3), dtype=np.uint8)
    else:
        crop = cv2.resize(crop, (output_size, out_h))

    actual_box = (x1, y1, x2 - x1, y2 - y1)
    return crop, actual_box


class BBoxSmoother:
    """Exponential moving average smoother for bounding boxes.

    Smooths bounding box coordinates over time to reduce jitter
    in face/body tracking.

    Args:
        alpha: EMA smoothing factor in (0, 1]. Lower = smoother.
    """

    def __init__(self, alpha: float = 0.3) -> None:
        self._alpha = alpha
        self._state: Optional[tuple[float, float, float, float]] = None

    def update(
        self, bbox: tuple[int, int, int, int]
    ) -> tuple[int, int, int, int]:
        """Update smoother with new bbox and return smoothed result.

        Args:
            bbox: Raw bounding box (x, y, w, h) in pixels.

        Returns:
            Smoothed bounding box (x, y, w, h) as integers.
        """
        raw = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))

        if self._state is None:
            self._state = raw
        else:
            a = self._alpha
            self._state = tuple(
                a * r + (1 - a) * s for r, s in zip(raw, self._state)
            )

        return (
            int(round(self._state[0])),
            int(round(self._state[1])),
            int(round(self._state[2])),
            int(round(self._state[3])),
        )

    def reset(self) -> None:
        """Reset smoother state."""
        self._state = None


__all__ = ["CropRatio", "face_crop", "BBoxSmoother"]
