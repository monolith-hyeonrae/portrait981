"""LibreFace AU detection backend (2-stage ONNX: encoder + intensity head).

LibreFace (WACV 2024, ResNet-18) detects 12 DISFA Action Units.
Two ONNX models:
  - LibreFace_AU_Encoder.onnx: [1,3,224,224] -> [1,512,1,1]
  - LibreFace_AU_Intensity.onnx: [1,512] -> [1,12]

Preprocessing: resize 256 -> center crop 224 -> ImageNet normalize.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import numpy as np

from vpx.face_detect.backends.base import DetectedFace
from vpx.face_au.backends.base import AU_NAMES, FaceAUResult

logger = logging.getLogger(__name__)

# ImageNet normalization constants
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class LibreFaceBackend:
    """LibreFace 2-stage ONNX backend for AU intensity estimation.

    Models are loaded from ``get_models_dir() / "libreface" /``.
    """

    def __init__(self, models_dir: Optional[Path] = None):
        self._models_dir = models_dir
        self._encoder_session = None
        self._intensity_session = None
        self._initialized = False

    def initialize(self, device: str = "cuda:0") -> None:
        if self._initialized:
            return

        import onnxruntime as ort

        if self._models_dir is None:
            from vpx.sdk.paths import get_models_dir
            self._models_dir = get_models_dir()

        model_dir = self._models_dir / "libreface"
        encoder_path = model_dir / "LibreFace_AU_Encoder.onnx"
        intensity_path = model_dir / "LibreFace_AU_Intensity.onnx"

        if not encoder_path.exists() or not intensity_path.exists():
            raise FileNotFoundError(
                f"LibreFace ONNX models not found in {model_dir}. "
                "Download from NuGet: libreface 2.0.0 package."
            )

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if "cpu" in device.lower():
            providers = ["CPUExecutionProvider"]

        self._encoder_session = ort.InferenceSession(
            str(encoder_path), providers=providers,
        )
        self._intensity_session = ort.InferenceSession(
            str(intensity_path), providers=providers,
        )
        self._initialized = True
        logger.info("LibreFace backend initialized from %s", model_dir)

    def analyze(
        self, image: np.ndarray, faces: List[DetectedFace]
    ) -> List[FaceAUResult]:
        if not self._initialized:
            raise RuntimeError("Backend not initialized. Call initialize() first.")
        if not faces:
            return []

        import cv2

        results = []
        for face in faces:
            crop = self._extract_crop(image, face)
            if crop is None:
                results.append(FaceAUResult())
                continue

            try:
                tensor = self._preprocess(crop)
                intensities = self._predict(tensor)
                au_dict = {
                    AU_NAMES[i]: float(intensities[i])
                    for i in range(len(AU_NAMES))
                }
                au_presence = {k: v >= 1.0 for k, v in au_dict.items()}
                results.append(FaceAUResult(
                    au_intensities=au_dict,
                    au_presence=au_presence,
                ))
            except Exception as e:
                logger.warning("LibreFace AU prediction failed: %s", e)
                results.append(FaceAUResult())

        return results

    def _extract_crop(
        self, image: np.ndarray, face: DetectedFace
    ) -> Optional[np.ndarray]:
        """Extract face crop with padding from pixel-coord bbox."""
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

        # BGR -> RGB
        return cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

    def _preprocess(self, face_rgb: np.ndarray) -> np.ndarray:
        """Resize 256 -> center crop 224 -> ImageNet normalize -> NCHW tensor."""
        import cv2

        # Resize to 256x256
        resized = cv2.resize(face_rgb, (256, 256))

        # Center crop 224x224
        offset = (256 - 224) // 2
        cropped = resized[offset:offset + 224, offset:offset + 224]

        # Float32 normalize [0, 1]
        img = cropped.astype(np.float32) / 255.0

        # ImageNet normalize
        img = (img - IMAGENET_MEAN) / IMAGENET_STD

        # HWC -> NCHW
        img = np.transpose(img, (2, 0, 1))[np.newaxis, ...]
        return img.astype(np.float32)

    def _predict(self, tensor: np.ndarray) -> np.ndarray:
        """Run 2-stage inference: encoder -> squeeze -> intensity head.

        Returns:
            1D array of 12 AU intensities (0-5 scale).
        """
        # Stage 1: Encoder [1,3,224,224] -> [1,512,1,1]
        enc_input = self._encoder_session.get_inputs()[0].name
        enc_out = self._encoder_session.run(None, {enc_input: tensor})[0]

        # Squeeze spatial dims: [1,512,1,1] -> [1,512]
        enc_out = enc_out.reshape(1, -1)

        # Stage 2: Intensity head [1,512] -> [1,12]
        int_input = self._intensity_session.get_inputs()[0].name
        intensities = self._intensity_session.run(None, {int_input: enc_out})[0]

        # Clamp to [0, 5]
        intensities = np.clip(intensities[0], 0.0, 5.0)
        return intensities

    def cleanup(self) -> None:
        self._encoder_session = None
        self._intensity_session = None
        self._initialized = False
        logger.info("LibreFace backend cleaned up")
