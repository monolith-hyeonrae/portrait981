"""NIMA aesthetic quality backend (MobileNetV2, optional).

MobileNetV2 + NIMA head (Linear 1280→10 + Softmax) 기반 미학적 품질 평가.
AVA 데이터셋으로 학습된 pretrained weights가 필요하며,
weights 파일이 없을 경우 score=0.0으로 graceful degradation.

Model weights: {models_dir}/nima/nima-mobilenetv2.pth
Score: Σ(i × p_i for i in 1..10) / 9  →  [0, 1] (normalized mean aesthetic score)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def _build_nima_model():
    """MobileNetV2 + NIMA head 모델 빌드."""
    import torch.nn as nn
    from torchvision.models import mobilenet_v2

    class NIMANet(nn.Module):
        def __init__(self):
            super().__init__()
            base = mobilenet_v2(weights=None)
            self.features = base.features
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.dropout = nn.Dropout(p=0.75)
            self.fc = nn.Linear(1280, 10)
            self.softmax = nn.Softmax(dim=1)

        def forward(self, x):
            x = self.features(x)
            x = self.pool(x)
            x = x.flatten(1)
            x = self.dropout(x)
            x = self.fc(x)
            return self.softmax(x)

    return NIMANet()


class NIMABackend:
    """NIMA aesthetic quality scorer for portrait crops.

    Uses a MobileNetV2 backbone fine-tuned on AVA dataset.
    Returns normalized aesthetic score in [0, 1].

    Requires:
        - torch, torchvision installed
        - pretrained weights at models_dir / "nima" / "nima-mobilenetv2.pth"

    When weights are not found, score() returns 0.0 (graceful degradation).
    """

    def __init__(self, models_dir: Optional[Path] = None, device: str = "cpu"):
        self._device = device
        self._models_dir = models_dir
        self._model = None
        self._transform = None
        self._available = False

    def initialize(self, models_dir: Optional[Path] = None, device: Optional[str] = None) -> bool:
        """모델 로드. weights가 없으면 False 반환 (skip, not error)."""
        if device:
            self._device = device
        if models_dir:
            self._models_dir = models_dir

        weights_path = None
        if self._models_dir is not None:
            weights_path = Path(self._models_dir) / "nima" / "nima-mobilenetv2.pth"

        if weights_path is None or not weights_path.exists():
            logger.info(
                "NIMA weights not found at %s — aesthetic scoring disabled",
                weights_path,
            )
            return False

        try:
            import torch
            model = _build_nima_model()
            state = torch.load(weights_path, map_location=self._device)
            model.load_state_dict(state)
            model.eval()
            model.to(self._device)
            self._model = model
            self._available = True
            logger.info("NIMABackend loaded from %s", weights_path)
            return True
        except Exception as e:
            logger.warning("NIMABackend load failed: %s — aesthetic scoring disabled", e)
            return False

    @property
    def available(self) -> bool:
        return self._available

    def score(self, image_bgr: np.ndarray) -> float:
        """BGR image → aesthetic score [0, 1].

        Returns 0.0 if model not loaded.
        """
        if not self._available or self._model is None:
            return 0.0

        try:
            import torch
            from torchvision import transforms

            if self._transform is None:
                self._transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                    ),
                ])

            # BGR → RGB
            import cv2
            rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            tensor = self._transform(rgb).unsqueeze(0).to(self._device)

            with torch.no_grad():
                probs = self._model(tensor).squeeze(0).cpu().numpy()

            # Mean aesthetic score: Σ(i × p_i) for i in 1..10 → normalize to [0,1]
            scores = np.arange(1, 11, dtype=np.float32)
            mean_score = float(np.dot(probs, scores))
            return (mean_score - 1.0) / 9.0  # [1,10] → [0,1]

        except Exception as e:
            logger.debug("NIMABackend.score() failed: %s", e)
            return 0.0

    def cleanup(self) -> None:
        self._model = None
        self._available = False


__all__ = ["NIMABackend"]
