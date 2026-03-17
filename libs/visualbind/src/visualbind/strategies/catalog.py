"""Fisher-weighted centroid matching strategy (from catalog_scoring).

This is the original VisualBind strategy: compute per-category Fisher ratio
importance weights, then match frames via weighted Euclidean distance
transformed to similarity.
"""

from __future__ import annotations

import logging
from math import sqrt
from typing import List, Optional

import numpy as np

from ..profile import CategoryProfile
from ..signals import _NDIM

logger = logging.getLogger(__name__)


def compute_importance_weights(
    category_vectors: dict[str, np.ndarray],
    epsilon: float = 1e-8,
) -> dict[str, np.ndarray]:
    """Compute per-category pairwise Fisher ratio importance weights.

    For each category, computes the pairwise Fisher ratio against every other
    category, then averages.  This avoids the "global inter-variance" problem
    where one dominant category's variance affects all others.

    ::

        fisher_ij(d) = (mean_i(d) - mean_j(d))^2 / (var_i(d) + var_j(d) + eps)
        fisher_i(d) = mean_j(fisher_ij(d))  (leave-one-out average)

    A sqrt transform compresses dynamic range before normalization to prevent
    dominant AU/emotion dimensions from overwhelming pose/CLIP dimensions.

    Args:
        category_vectors: category name -> ``(N, D)`` signal vector matrix.
        epsilon: denominator stabilization constant.

    Returns:
        category name -> ``(D,)`` importance weights (sum=1).
    """
    if not category_vectors:
        return {}

    cat_names = sorted(category_vectors.keys())
    n_cats = len(cat_names)

    # Per-category mean and variance
    cat_means = np.zeros((n_cats, _NDIM), dtype=np.float64)
    cat_vars = np.zeros((n_cats, _NDIM), dtype=np.float64)

    for ci, name in enumerate(cat_names):
        vecs = category_vectors[name]  # (N, D)
        if len(vecs) == 0:
            continue
        cat_means[ci] = vecs.mean(axis=0)
        cat_vars[ci] = vecs.var(axis=0) if len(vecs) > 1 else np.zeros(_NDIM)

    result = {}
    for ci, name in enumerate(cat_names):
        fisher_sum = np.zeros(_NDIM, dtype=np.float64)
        n_pairs = 0
        for cj in range(n_cats):
            if cj == ci:
                continue
            inter_sq = (cat_means[ci] - cat_means[cj]) ** 2
            pooled_var = cat_vars[ci] + cat_vars[cj]
            fisher_sum += inter_sq / (pooled_var + epsilon)
            n_pairs += 1

        if n_pairs > 0:
            fisher = fisher_sum / n_pairs
        else:
            fisher = np.ones(_NDIM, dtype=np.float64)

        # sqrt transform: dynamic range compression
        fisher = np.sqrt(fisher)

        # Normalize to sum=1
        total = fisher.sum()
        if total > 0:
            weights = fisher / total
        else:
            weights = np.ones(_NDIM, dtype=np.float64) / _NDIM
        result[name] = weights

    return result


def match_category(
    frame_vec: np.ndarray,
    profiles: List[CategoryProfile],
) -> tuple[float, str]:
    """Match a frame signal vector to the closest category.

    Weighted Euclidean distance -> similarity::

        d = sqrt(sum(w_i * (x_i - mu_i)^2))
        sim = 1.0 / (1.0 + d)

    Args:
        frame_vec: ``(D,)`` normalized signal vector.
        profiles: category profile list.

    Returns:
        ``(similarity_score, category_name)``.  ``(0.0, "")`` if no profiles.
    """
    if not profiles:
        return 0.0, ""

    best_sim = -1.0
    best_name = ""

    for profile in profiles:
        diff = frame_vec - profile.mean_signals
        weighted_sq = profile.importance_weights * (diff ** 2)
        d = sqrt(float(weighted_sq.sum()))
        sim = 1.0 / (1.0 + d)

        if sim > best_sim:
            best_sim = sim
            best_name = profile.name

    return best_sim, best_name


class CatalogStrategy:
    """Fisher-weighted centroid matching strategy.

    Implements :class:`~visualbind.strategies.BindingStrategy`.

    Can be initialized with pre-built profiles (for inference) or fitted
    from category vectors (which builds profiles internally).
    """

    def __init__(self, profiles: Optional[List[CategoryProfile]] = None) -> None:
        self._profiles: List[CategoryProfile] = list(profiles) if profiles else []

    @property
    def profiles(self) -> List[CategoryProfile]:
        """Access the built profiles."""
        return self._profiles

    def fit(self, vectors: dict[str, np.ndarray], **kwargs: object) -> None:
        """Build CategoryProfile list from category vectors.

        Args:
            vectors: category_name -> ``(N, D)`` normalized signal vectors.
        """
        weights = compute_importance_weights(vectors)
        self._profiles = []
        for name in sorted(vectors.keys()):
            vecs = vectors[name]
            if len(vecs) == 0:
                continue
            self._profiles.append(CategoryProfile(
                name=name,
                mean_signals=vecs.mean(axis=0),
                importance_weights=weights[name],
                n_refs=len(vecs),
            ))
        logger.info("CatalogStrategy fitted with %d categories", len(self._profiles))

    def predict(self, frame_vec: np.ndarray) -> dict[str, float]:
        """Score a frame against all category profiles.

        Args:
            frame_vec: ``(D,)`` normalized signal vector.

        Returns:
            Dict of category_name -> similarity score.
        """
        scores: dict[str, float] = {}
        for profile in self._profiles:
            diff = frame_vec - profile.mean_signals
            weighted_sq = profile.importance_weights * (diff ** 2)
            d = sqrt(float(weighted_sq.sum()))
            scores[profile.name] = 1.0 / (1.0 + d)
        return scores
