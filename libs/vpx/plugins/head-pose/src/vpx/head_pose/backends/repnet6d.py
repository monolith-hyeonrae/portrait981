"""6DRepNet ONNX backend for head pose estimation.

6DRepNet (IEEE FG 2022, RepVGG-B1g2) predicts a 3x3 rotation matrix,
which is then converted to Euler angles (yaw, pitch, roll).

ONNX model:
  - sixdrepnet.onnx: [1,3,224,224] -> [1,3,3] (rotation matrix)

Preprocessing: resize 224 -> ImageNet normalize.
Accuracy: MAE ~3.47 degrees on AFLW2000.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import List, Optional

import numpy as np

from vpx.face_detect.backends.base import DetectedFace
from vpx.head_pose.types import HeadPoseEstimate

logger = logging.getLogger(__name__)

# ImageNet normalization constants
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def rotation_matrix_to_euler(R: np.ndarray) -> tuple[float, float, float]:
    """Convert 3x3 rotation matrix to Euler angles (yaw, pitch, roll) in degrees.

    Uses the convention: R = Rz(roll) @ Ry(yaw) @ Rx(pitch).
    """
    sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)

    if sy > 1e-6:
        pitch = math.atan2(R[2, 1], R[2, 2])
        yaw = math.atan2(-R[2, 0], sy)
        roll = math.atan2(R[1, 0], R[0, 0])
    else:
        pitch = math.atan2(-R[1, 2], R[1, 1])
        yaw = math.atan2(-R[2, 0], sy)
        roll = 0.0

    return (
        math.degrees(yaw),
        math.degrees(pitch),
        math.degrees(roll),
    )


class RepNet6DBackend:
    """6DRepNet ONNX backend for head pose estimation.

    Model loaded from ``get_models_dir() / "6drepnet" / "sixdrepnet.onnx"``.
    """

    def __init__(self, models_dir: Optional[Path] = None):
        self._models_dir = models_dir
        self._session = None
        self._initialized = False

    def initialize(self, device: str = "cuda:0") -> None:
        if self._initialized:
            return

        import onnxruntime as ort

        if self._models_dir is None:
            from vpx.sdk.paths import get_models_dir
            self._models_dir = get_models_dir()

        model_path = self._models_dir / "6drepnet" / "sixdrepnet.onnx"
        if not model_path.exists():
            raise FileNotFoundError(
                f"6DRepNet ONNX model not found at {model_path}. "
                "Export from PyTorch checkpoint using scripts/export_onnx.py."
            )

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if "cpu" in device.lower():
            providers = ["CPUExecutionProvider"]

        self._session = ort.InferenceSession(
            str(model_path), providers=providers,
        )
        self._initialized = True
        logger.info("6DRepNet backend initialized from %s", model_path)

    def estimate(
        self, image: np.ndarray, faces: List[DetectedFace]
    ) -> List[HeadPoseEstimate]:
        if not self._initialized:
            raise RuntimeError("Backend not initialized. Call initialize() first.")
        if not faces:
            return []

        import cv2

        results = []
        for face in faces:
            crop = self._extract_crop(image, face)
            if crop is None:
                results.append(HeadPoseEstimate())
                continue

            try:
                tensor = self._preprocess(crop)
                rot_matrix = self._predict(tensor)
                yaw, pitch, roll = rotation_matrix_to_euler(rot_matrix)
                results.append(HeadPoseEstimate(yaw=yaw, pitch=pitch, roll=roll))
            except Exception as e:
                logger.warning("6DRepNet prediction failed: %s", e)
                results.append(HeadPoseEstimate())

        return results

    def _extract_crop(
        self, image: np.ndarray, face: DetectedFace
    ) -> Optional[np.ndarray]:
        """Extract face crop with padding."""
        import cv2

        x, y, w, h = face.bbox
        pad = int(max(w, h) * 0.1)
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(image.shape[1], x + w + pad)
        y2 = min(image.shape[0], y + h + pad)

        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            return None

        return cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

    def _preprocess(self, face_rgb: np.ndarray) -> np.ndarray:
        """Resize to 224x224, ImageNet normalize, NCHW tensor."""
        import cv2

        resized = cv2.resize(face_rgb, (224, 224))
        img = resized.astype(np.float32) / 255.0
        img = (img - IMAGENET_MEAN) / IMAGENET_STD
        img = np.transpose(img, (2, 0, 1))[np.newaxis, ...]
        return img.astype(np.float32)

    def _predict(self, tensor: np.ndarray) -> np.ndarray:
        """Run inference and return 3x3 rotation matrix.

        Returns:
            3x3 rotation matrix as numpy array.
        """
        input_name = self._session.get_inputs()[0].name
        output = self._session.run(None, {input_name: tensor})[0]

        # Output shape: [1, 3, 3] -> [3, 3]
        rot_matrix = output[0]
        return rot_matrix

    def cleanup(self) -> None:
        self._session = None
        self._initialized = False
        logger.info("6DRepNet backend cleaned up")
