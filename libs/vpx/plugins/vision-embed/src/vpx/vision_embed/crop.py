"""Crop utilities for face/body regions and bbox smoothing."""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np


def face_crop(
    image: np.ndarray,
    bbox: tuple[int, int, int, int],
    expand: float = 1.5,
    output_size: int = 224,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Extract an expanded face crop from the image.

    Args:
        image: Full frame (H, W, 3) BGR.
        bbox: Face bounding box (x, y, w, h) in pixels.
        expand: Expansion factor around the face center.
        output_size: Output crop size (square).

    Returns:
        Tuple of (crop, actual_box):
          - crop: Resized image (output_size, output_size, 3).
          - actual_box: Clamped bounding box (x, y, w, h) used for the crop.
    """
    h, w = image.shape[:2]
    bx, by, bw, bh = bbox

    # Expand around center
    cx = bx + bw / 2
    cy = by + bh / 2
    side = max(bw, bh) * expand
    half = side / 2

    x1 = max(0, int(cx - half))
    y1 = max(0, int(cy - half))
    x2 = min(w, int(cx + half))
    y2 = min(h, int(cy + half))

    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        crop = np.zeros((output_size, output_size, 3), dtype=np.uint8)
    else:
        crop = cv2.resize(crop, (output_size, output_size))

    actual_box = (x1, y1, x2 - x1, y2 - y1)
    return crop, actual_box


def body_crop(
    image: np.ndarray,
    keypoints: np.ndarray,
    output_size: int = 224,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Extract an upper-body crop (face included) based on pose keypoints.

    Uses nose, eyes, ears, shoulders, and hips
    (COCO keypoint indices 0-6, 11, 12) to define the region.

    Args:
        image: Full frame (H, W, 3) BGR.
        keypoints: Keypoints array of shape (N, 2) or (N, 3) with x, y[, conf].
        output_size: Output crop size (square).

    Returns:
        Tuple of (crop, actual_box):
          - crop: Resized image (output_size, output_size, 3).
          - actual_box: Bounding box (x, y, w, h) used for the crop.
    """
    h, w = image.shape[:2]

    # COCO: 0=nose, 1/2=eyes, 3/4=ears, 5/6=shoulders, 11/12=hips
    upper_body_indices = [0, 1, 2, 3, 4, 5, 6, 11, 12]
    pts = []
    for idx in upper_body_indices:
        if idx < len(keypoints):
            pt = keypoints[idx]
            x, y = float(pt[0]), float(pt[1])
            if x > 0 and y > 0:
                pts.append((x, y))

    if len(pts) < 2:
        # Fallback: center crop
        side = min(h, w) // 2
        cx, cy = w // 2, h // 2
        x1 = max(0, cx - side // 2)
        y1 = max(0, cy - side // 2)
        crop = image[y1:y1 + side, x1:x1 + side]
        crop = cv2.resize(crop, (output_size, output_size))
        return crop, (x1, y1, side, side)

    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    # Add padding (20% on each side)
    pw = (x_max - x_min) * 0.2
    ph = (y_max - y_min) * 0.2

    x1 = max(0, int(x_min - pw))
    y1 = max(0, int(y_min - ph))
    x2 = min(w, int(x_max + pw))
    y2 = min(h, int(y_max + ph))

    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        crop = np.zeros((output_size, output_size, 3), dtype=np.uint8)
    else:
        crop = cv2.resize(crop, (output_size, output_size))

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


__all__ = ["face_crop", "body_crop", "BBoxSmoother"]
