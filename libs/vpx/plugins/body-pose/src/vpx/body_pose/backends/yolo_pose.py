"""Pose estimation backend implementations."""

from pathlib import Path
from typing import List, Optional
import logging

import numpy as np

from vpx.body_pose.backends.base import PoseKeypoints
from vpx.body_pose.types import COCO_KEYPOINT_NAMES

logger = logging.getLogger(__name__)


class YOLOPoseBackend:
    """Pose estimation backend using YOLOv8-Pose.

    YOLOv8-Pose provides fast and accurate pose estimation with
    17 COCO keypoints for body tracking and gesture detection.

    Args:
        model_name: Model variant (default: "yolov8m-pose.pt" for balanced speed/accuracy).
        conf_threshold: Detection confidence threshold.
        iou_threshold: NMS IoU threshold.

    Example:
        >>> backend = YOLOPoseBackend()
        >>> backend.initialize("cuda:0")
        >>> poses = backend.detect(image)
        >>> backend.cleanup()
    """

    def __init__(
        self,
        model_name: str = "yolov8m-pose.pt",
        conf_threshold: float = 0.5,
        iou_threshold: float = 0.7,
        models_dir: Optional[Path] = None,
    ):
        self._model_name = model_name
        self._conf_threshold = conf_threshold
        self._iou_threshold = iou_threshold
        self._models_dir = models_dir
        self._model: Optional[object] = None
        self._initialized = False

    def initialize(self, device: str = "cuda:0") -> None:
        """Initialize YOLOv8-Pose model."""
        if self._initialized:
            return  # Already initialized

        try:
            from ultralytics import YOLO

            models_dir = self._models_dir
            if models_dir is None:
                from vpx.sdk.paths import get_models_dir
                models_dir = get_models_dir()
            # ultralytics downloads bare filenames to CWD.
            # 1) Set weights_dir so fallback download goes to models_dir
            # 2) Pass absolute path so YOLO never checks CWD
            models_dir.mkdir(parents=True, exist_ok=True)
            try:
                from ultralytics.utils import SETTINGS
                SETTINGS["weights_dir"] = str(models_dir)
            except Exception:
                pass
            model_path = str(models_dir / self._model_name)
            self._model = YOLO(model_path)

            # Resolve device with platform-aware fallback
            self._device, self._actual_provider = self._resolve_device(device)

            self._initialized = True
            logger.info("YOLOv8-Pose initialized with model %s (device=%s)",
                        self._model_name, self._actual_provider)

        except ImportError:
            raise ImportError(
                "ultralytics is required for YOLOPoseBackend. "
                "Install with: pip install ultralytics"
            )
        except Exception as e:
            logger.error(f"Failed to initialize YOLOv8-Pose: {e}")
            raise

    def detect(self, image: np.ndarray) -> List[PoseKeypoints]:
        """Detect poses using YOLOv8-Pose.

        Args:
            image: BGR image as numpy array (H, W, 3).

        Returns:
            List of detected poses with COCO 17 keypoints.
        """
        if not self._initialized or self._model is None:
            raise RuntimeError("Backend not initialized. Call initialize() first.")

        # Run inference
        results = self._model(
            image,
            device=self._device,
            conf=self._conf_threshold,
            iou=self._iou_threshold,
            verbose=False,
        )

        poses = []

        for result in results:
            if result.keypoints is None:
                continue

            keypoints_data = result.keypoints.data.cpu().numpy()
            boxes = result.boxes

            for i, kpts in enumerate(keypoints_data):
                # kpts shape: (17, 3) - x, y, confidence
                if kpts.shape[0] != 17:
                    continue

                # Get bounding box if available
                bbox = None
                conf = 1.0
                if boxes is not None and i < len(boxes):
                    box = boxes[i]
                    xyxy = box.xyxy[0].cpu().numpy()
                    bbox = (
                        int(xyxy[0]),
                        int(xyxy[1]),
                        int(xyxy[2] - xyxy[0]),
                        int(xyxy[3] - xyxy[1]),
                    )
                    conf = float(box.conf[0])

                poses.append(
                    PoseKeypoints(
                        keypoints=kpts.astype(np.float32),
                        keypoint_names=COCO_KEYPOINT_NAMES,
                        person_id=i,
                        bbox=bbox,
                        confidence=conf,
                    )
                )

        return poses

    @staticmethod
    def _resolve_device(device: str) -> tuple:
        """Resolve requested device to an available one.

        YOLO accepts: int (cuda device id), "cpu", "mps".
        Falls back gracefully: cuda → mps (macOS) → cpu.

        Returns:
            (device_for_yolo, provider_name) tuple.
        """
        import torch

        if device.startswith("cuda"):
            if torch.cuda.is_available():
                device_id = device.split(":")[-1] if ":" in device else "0"
                return int(device_id), f"CUDA:{device_id}"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                logger.info("CUDA unavailable, using MPS acceleration (macOS)")
                return "mps", "MPS"
            logger.warning("CUDA unavailable, falling back to CPU")
            return "cpu", "CPU (CUDA unavailable)"

        if device == "mps":
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps", "MPS"
            logger.warning("MPS unavailable, falling back to CPU")
            return "cpu", "CPU (MPS unavailable)"

        return "cpu", "CPU"

    def cleanup(self) -> None:
        """Release YOLOv8-Pose resources."""
        self._model = None
        self._initialized = False
        logger.info("YOLOv8-Pose cleaned up")

    @staticmethod
    def get_keypoint_index(name: str) -> int:
        """Get index of a keypoint by name.

        Args:
            name: Keypoint name (e.g., "left_wrist").

        Returns:
            Index in the keypoints array.

        Raises:
            ValueError: If keypoint name is not found.
        """
        try:
            return COCO_KEYPOINT_NAMES.index(name)
        except ValueError:
            raise ValueError(
                f"Unknown keypoint: {name}. "
                f"Valid names: {COCO_KEYPOINT_NAMES}"
            )
