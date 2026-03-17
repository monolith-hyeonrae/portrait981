"""Binding strategy protocol and registry.

A :class:`BindingStrategy` combines multi-observer outputs into per-bucket scores.
Different strategies (centroid matching, tree-based classifiers, etc.) implement
the same interface so they can be swapped or compared transparently.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class BindingStrategy(Protocol):
    """Observer output binding strategy."""

    def fit(self, vectors: dict[str, np.ndarray], **kwargs: object) -> None:
        """Build profiles / train model from category vectors.

        Args:
            vectors: category_name -> ``(N, D)`` signal vector matrix.
            **kwargs: strategy-specific parameters.
        """
        ...

    def predict(self, frame_vec: np.ndarray) -> dict[str, float]:
        """Score a single frame against all categories.

        Args:
            frame_vec: ``(D,)`` normalized signal vector.

        Returns:
            Dict of category_name -> score (higher is better).
        """
        ...
