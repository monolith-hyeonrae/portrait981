"""DINOv2 ViT-S/14 embedding backend."""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class DINOv2Backend:
    """DINOv2 ViT-S/14 embedding backend.

    Loads dinov2_vits14 via torch.hub and produces 384-dim
    L2-normalized embeddings from 224x224 image crops.
    """

    def __init__(self) -> None:
        self._model = None
        self._transform = None
        self._device: Optional[str] = None

    @property
    def embed_dim(self) -> int:
        return 384

    def initialize(self, device: str) -> None:
        import torch
        from torchvision import transforms

        # Device selection
        if device.startswith("cuda") and torch.cuda.is_available():
            self._device = device
        elif torch.backends.mps.is_available():
            self._device = "mps"
        else:
            self._device = "cpu"

        self._model = torch.hub.load(
            "facebookresearch/dinov2", "dinov2_vits14", pretrained=True
        )
        self._model = self._model.to(self._device)
        self._model.eval()

        # ImageNet normalization + resize to 224x224
        self._transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        logger.info(
            "DINOv2Backend initialized on %s (embed_dim=%d)",
            self._device, self.embed_dim,
        )

    def embed(self, image: np.ndarray) -> np.ndarray:
        """Compute L2-normalized embedding from a BGR image crop.

        Args:
            image: BGR image (H, W, 3) uint8.

        Returns:
            L2-normalized float32 vector of shape (384,).
        """
        import torch

        # BGR -> RGB
        rgb = image[:, :, ::-1].copy()

        tensor = self._transform(rgb).unsqueeze(0).to(self._device)

        with torch.no_grad():
            features = self._model(tensor)  # (1, 384)

        vec = features.squeeze(0).cpu().numpy().astype(np.float32)

        # L2 normalize
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm

        return vec

    def cleanup(self) -> None:
        if self._model is not None:
            del self._model
            self._model = None
        self._transform = None
        self._device = None
        logger.info("DINOv2Backend cleaned up")


__all__ = ["DINOv2Backend"]
