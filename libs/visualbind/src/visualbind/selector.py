"""Frame selection based on strategy scores.

Replaces peak detection with direct concept-based selection:
  score every frame → quality gate → top-K per category.

No delta computation, no EMA smoothing, no peak finding.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .strategies import BindingStrategy

logger = logging.getLogger(__name__)


@dataclass
class SelectedFrame:
    """선택된 프레임."""
    index: int
    category: str
    score: float
    scores: dict = field(default_factory=dict)


@dataclass
class SelectionResult:
    """프레임 선택 결과."""
    frames: list[SelectedFrame] = field(default_factory=list)
    frame_count: int = 0
    per_category: dict[str, list[SelectedFrame]] = field(default_factory=dict)


def select_frames(
    vectors: np.ndarray,
    strategy: BindingStrategy,
    *,
    top_k: int = 5,
    min_score: float = 0.0,
    gate_mask: Optional[np.ndarray] = None,
    categories: Optional[list[str]] = None,
) -> SelectionResult:
    """Score all frames and select top-K per category.

    Args:
        vectors: ``(N, D)`` signal matrix.
        strategy: Trained BindingStrategy (e.g. TreeStrategy).
        top_k: Number of top frames to select per category.
        min_score: Minimum score threshold.
        gate_mask: ``(N,)`` boolean mask. True = frame passes quality gate.
            None = all frames pass.
        categories: Category names to select for.
            None = use all categories from strategy predictions.

    Returns:
        SelectionResult with selected frames per category.
    """
    n_frames = len(vectors)
    if n_frames == 0:
        return SelectionResult()

    # Score all frames
    all_scores: list[dict[str, float]] = []
    for i in range(n_frames):
        scores = strategy.predict(vectors[i])
        all_scores.append(scores)

    # Discover categories if not provided
    if categories is None:
        categories = sorted({k for scores in all_scores for k in scores})

    # Apply quality gate
    if gate_mask is None:
        gate_mask = np.ones(n_frames, dtype=bool)

    # Select top-K per category
    result = SelectionResult(frame_count=n_frames)

    for cat in categories:
        # Get scores for this category, respecting gate
        cat_scores = []
        for i in range(n_frames):
            if gate_mask[i] and cat in all_scores[i]:
                score = all_scores[i][cat]
                if score >= min_score:
                    cat_scores.append((score, i))

        # Sort by score descending, take top-K
        cat_scores.sort(reverse=True)
        selected = []
        for score, idx in cat_scores[:top_k]:
            sf = SelectedFrame(
                index=idx,
                category=cat,
                score=score,
                scores=all_scores[idx],
            )
            selected.append(sf)
            result.frames.append(sf)

        result.per_category[cat] = selected
        logger.info("  %s: %d frames selected (top score=%.3f)",
                    cat, len(selected),
                    selected[0].score if selected else 0.0)

    return result
