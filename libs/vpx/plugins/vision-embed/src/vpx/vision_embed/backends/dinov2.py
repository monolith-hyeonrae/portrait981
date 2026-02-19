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

    Supports device-based singleton sharing via get_shared()/release_shared().
    """

    # Device-based singleton registry
    _shared_instances: dict[str, DINOv2Backend] = {}
    _ref_counts: dict[str, int] = {}

    def __init__(self) -> None:
        self._model = None
        self._transform = None
        self._device: Optional[str] = None

    @property
    def embed_dim(self) -> int:
        return 384

    @classmethod
    def get_shared(cls, device: str = "cuda:0") -> DINOv2Backend:
        """Return a shared instance for the given device. Creates on first call."""
        if device not in cls._shared_instances:
            backend = cls()
            backend.initialize(device)
            cls._shared_instances[device] = backend
            cls._ref_counts[device] = 0
        cls._ref_counts[device] += 1
        return cls._shared_instances[device]

    @classmethod
    def release_shared(cls, device: str) -> None:
        """Decrement reference count. Cleanup when count reaches zero."""
        if device in cls._ref_counts:
            cls._ref_counts[device] -= 1
            if cls._ref_counts[device] <= 0:
                if device in cls._shared_instances:
                    cls._shared_instances[device].cleanup()
                    del cls._shared_instances[device]
                del cls._ref_counts[device]

    @classmethod
    def _reset_shared(cls) -> None:
        """Reset all shared state. For testing only."""
        cls._shared_instances.clear()
        cls._ref_counts.clear()

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

    def embed_batch(self, images: list[np.ndarray]) -> list[np.ndarray]:
        """Batch-embed multiple crops in a single forward pass.

        Args:
            images: List of BGR images (H, W, 3) uint8.

        Returns:
            List of L2-normalized float32 vectors of shape (384,).
        """
        if not images:
            return []
        if len(images) == 1:
            return [self.embed(images[0])]

        import torch

        tensors = [self._transform(img[:, :, ::-1].copy()) for img in images]
        batch = torch.stack(tensors).to(self._device)
        with torch.no_grad():
            features = self._model(batch)  # (N, 384)
        vecs = features.cpu().numpy().astype(np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        vecs = vecs / norms
        return [vecs[i] for i in range(len(vecs))]

    def cleanup(self) -> None:
        if self._model is not None:
            del self._model
            self._model = None
        self._transform = None
        self._device = None
        logger.info("DINOv2Backend cleaned up")


__all__ = ["DINOv2Backend"]
