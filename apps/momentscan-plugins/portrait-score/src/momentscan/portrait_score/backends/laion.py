"""LAION Aesthetic Predictor backend.

Architecture:
  CLIP ViT-B/32 (open_clip) → encode_image() → L2 normalize
  → Linear(512, 1) → raw score [~1, ~10] → normalize to [0, 1]

Weights:
  sa_0_4_vit_b_32_linear.pth (~2KB, MIT license)
  Download from: https://github.com/christophschuhmann/improved-aesthetic-predictor

Usage:
  backend = LAIONAestheticBackend(models_dir=get_models_dir())
  backend.initialize()
  if backend.available:
      score = backend.score(image_bgr)  # float in [0, 1]
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_WEIGHTS_SUBDIR = "laion"
_WEIGHTS_FILENAME = "sa_0_4_vit_b_32_linear.pth"
_CLIP_MODEL = "ViT-B-32"
_CLIP_PRETRAINED = "openai"
_EMBED_DIM = 512

# Raw LAION scores ≈ [1, 10]; map to [0, 1]
_SCORE_MIN = 1.0
_SCORE_MAX = 10.0


class LAIONAestheticBackend:
    """LAION Aesthetic Predictor: CLIP ViT-B/32 → Linear(512,1) → [0,1]."""

    def __init__(
        self,
        models_dir: Optional[Path] = None,
        device: str = "cpu",
    ):
        if models_dir is None:
            models_dir = Path.home() / ".portrait981" / "models"
        self._weights_path = Path(models_dir) / _WEIGHTS_SUBDIR / _WEIGHTS_FILENAME
        self._device = device

        self._clip_model = None
        self._preprocess = None
        self._linear = None
        self.available: bool = False

    def initialize(self) -> bool:
        """Load CLIP model and Linear head weights.

        Returns True if successfully loaded, False otherwise.
        """
        if not self._weights_path.exists():
            logger.warning(
                "LAIONAestheticBackend: weights not found at %s — "
                "download from https://github.com/christophschuhmann/improved-aesthetic-predictor",
                self._weights_path,
            )
            return False

        try:
            import torch
            import torch.nn as nn
            import open_clip

            # Load CLIP ViT-B/32
            model, _, preprocess = open_clip.create_model_and_transforms(
                _CLIP_MODEL, pretrained=_CLIP_PRETRAINED
            )
            model = model.to(self._device)
            model.eval()

            # Load Linear(512, 1) head
            linear = nn.Linear(_EMBED_DIM, 1)
            state = torch.load(
                self._weights_path, map_location=self._device, weights_only=True
            )
            linear.load_state_dict(state)
            linear = linear.to(self._device)
            linear.eval()

            self._clip_model = model
            self._preprocess = preprocess
            self._linear = linear
            self.available = True
            logger.info(
                "LAIONAestheticBackend: loaded (device=%s, weights=%s)",
                self._device, self._weights_path.name,
            )
            return True

        except Exception as e:
            logger.warning("LAIONAestheticBackend: initialization failed — %s", e)
            self.available = False
            return False

    def score(self, image_bgr: np.ndarray) -> float:
        """Return aesthetic score in [0, 1] for a BGR image.

        Scores head crop only (Option A: 1 CLIP call per frame).
        """
        if not self.available:
            return 0.0

        import torch
        from PIL import Image

        try:
            # BGR → RGB → PIL
            rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)

            with torch.no_grad():
                tensor = self._preprocess(pil_img).unsqueeze(0).to(self._device)
                embedding = self._clip_model.encode_image(tensor)
                # L2 normalize
                embedding = embedding / embedding.norm(dim=-1, keepdim=True)
                raw = float(self._linear(embedding.float()).squeeze().item())

            # Map ~[1, 10] → [0, 1]
            normalized = (raw - _SCORE_MIN) / (_SCORE_MAX - _SCORE_MIN)
            return float(max(0.0, min(1.0, normalized)))

        except Exception as e:
            logger.debug("LAIONAestheticBackend.score failed: %s", e)
            return 0.0

    def cleanup(self) -> None:
        self._clip_model = None
        self._preprocess = None
        self._linear = None
        self.available = False
