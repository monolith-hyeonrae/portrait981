"""BiSeNet face parsing backend via ONNX Runtime.

CelebAMask-HQ 19-class segmentation model (ResNet18 backbone).
Input: [1, 3, 512, 512] RGB, ImageNet normalize.
Output: [1, 19, 512, 512] → argmax → class indices.
"""

from __future__ import annotations

import logging
from typing import List

import cv2
import numpy as np

from vpx.sdk.crop import face_crop
from vpx.sdk.paths import get_models_dir
from vpx.face_parse.output import FaceParseResult

logger = logging.getLogger(__name__)


class BiSeNetBackend:
    """BiSeNet face parsing via ONNX Runtime."""

    INPUT_SIZE = 512
    # Skin, l_brow, r_brow, l_eye, r_eye, eye_g, nose, mouth, u_lip, l_lip
    FACE_CLASSES = frozenset({1, 2, 3, 4, 5, 6, 10, 11, 12, 13})
    CLASS_LABELS: dict[int, str] = {
        0: "background", 1: "skin", 2: "l_brow", 3: "r_brow",
        4: "l_eye", 5: "r_eye", 6: "eye_g", 7: "l_ear", 8: "r_ear",
        9: "ear_r", 10: "nose", 11: "mouth", 12: "u_lip", 13: "l_lip",
        14: "neck", 15: "neck_l", 16: "cloth", 17: "hair", 18: "hat",
    }
    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    EXPAND_RATIO = 1.3
    CROP_RATIO = "1:1"

    def __init__(self) -> None:
        self._session = None
        self._input_name: str = ""

    def initialize(self, device: str = "cuda:0") -> None:
        """Load BiSeNet ONNX model."""
        import onnxruntime as ort

        model_path = get_models_dir() / "bisenet" / "resnet18.onnx"
        if not model_path.exists():
            raise FileNotFoundError(
                f"BiSeNet model not found: {model_path}. "
                "Download from github.com/yakhyo/face-parsing/releases"
            )

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self._session = ort.InferenceSession(str(model_path), providers=providers)
        self._input_name = self._session.get_inputs()[0].name
        logger.info("BiSeNet loaded from %s", model_path)

    def segment(
        self, image: np.ndarray, detected_faces: list
    ) -> List[FaceParseResult]:
        """Segment face skin regions for each detected face."""
        if self._session is None:
            return []

        results = []
        for face in detected_faces:
            crop, crop_box = self._crop_face(image, face.bbox)
            if crop.size == 0:
                continue

            input_tensor = self._preprocess(crop)
            logits = self._session.run(None, {self._input_name: input_tensor})[0]

            # argmax → class map
            class_map = np.argmax(logits[0], axis=0).astype(np.uint8)

            # Binary face mask
            face_mask = np.zeros_like(class_map, dtype=np.uint8)
            for cls_id in self.FACE_CLASSES:
                face_mask[class_map == cls_id] = 255

            results.append(FaceParseResult(
                face_id=getattr(face, "face_id", 0),
                face_mask=face_mask,
                crop_box=crop_box,
                class_map=class_map,
            ))

        return results

    def cleanup(self) -> None:
        """Release ONNX session."""
        self._session = None

    def _crop_face(
        self, image: np.ndarray, bbox: tuple
    ) -> tuple[np.ndarray, tuple[int, int, int, int]]:
        """Crop face region with expansion (4:5 portrait ratio).

        Uses the shared face_crop utility to match face.quality head crop
        dimensions, ensuring the crown/top of head is included for better
        segmentation.

        Args:
            image: BGR image (H, W, 3).
            bbox: (x, y, w, h) in pixels.

        Returns:
            (crop, (x, y, w, h)) — crop image and pixel-coordinate crop box.
        """
        crop, actual_box = face_crop(
            image, bbox,
            expand=self.EXPAND_RATIO,
            output_size=self.INPUT_SIZE,
            crop_ratio=self.CROP_RATIO,
        )
        return crop, actual_box

    def _preprocess(self, crop: np.ndarray) -> np.ndarray:
        """Resize to 512x512, normalize, NCHW float32."""
        resized = cv2.resize(crop, (self.INPUT_SIZE, self.INPUT_SIZE))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        normalized = (rgb - self.MEAN) / self.STD
        # HWC → CHW → NCHW
        return normalized.transpose(2, 0, 1)[np.newaxis]


__all__ = ["BiSeNetBackend"]
