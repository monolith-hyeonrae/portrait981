"""Backend protocol for vision embedding models."""

from __future__ import annotations

from typing import Protocol

import numpy as np


class EmbeddingBackend(Protocol):
    """Protocol for vision embedding backends.

    Implementations should be swappable without changing analyzer logic.
    Examples: DINOv2, SigLIP, CLIP.

    All embed() implementations must return L2-normalized vectors.
    """

    def initialize(self, device: str) -> None:
        """Initialize the backend and load model to device."""
        ...

    def embed(self, image: np.ndarray) -> np.ndarray:
        """Compute embedding for a single image crop.

        Args:
            image: BGR image as numpy array (H, W, 3).

        Returns:
            L2-normalized embedding vector (e.g. 384-dim for DINOv2 ViT-S/14).
        """
        ...

    def embed_batch(self, images: list[np.ndarray]) -> list[np.ndarray]:
        """Batch-embed multiple crops.

        Default: sequential fallback calling embed() per image.
        Backends with GPU batching should override for efficiency.

        Args:
            images: List of BGR images (H, W, 3).

        Returns:
            List of L2-normalized embedding vectors.
        """
        return [self.embed(img) for img in images]

    @property
    def embed_dim(self) -> int:
        """Embedding dimension (e.g. 384 for DINOv2 ViT-S/14)."""
        ...

    def cleanup(self) -> None:
        """Release resources and unload model."""
        ...


__all__ = ["EmbeddingBackend"]
