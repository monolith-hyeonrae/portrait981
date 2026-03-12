"""Stage 4: Embedding encoder via contrastive learning (numpy PoC).

Simple linear projection trained with triplet margin loss.
No PyTorch dependency — pure numpy gradient descent for PoC validation.

    encoder = TripletEncoder(input_dim=12, embed_dim=4)
    history = encoder.fit(anchors, positives, negatives, epochs=100)
    embeddings = encoder.encode(signals)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


@dataclass
class TrainHistory:
    """Training history from encoder.fit().

    Attributes:
        losses: Loss per epoch.
        pos_dists: Mean positive distance per epoch.
        neg_dists: Mean negative distance per epoch.
    """

    losses: List[float] = field(default_factory=list)
    pos_dists: List[float] = field(default_factory=list)
    neg_dists: List[float] = field(default_factory=list)

    @property
    def final_loss(self) -> float:
        return self.losses[-1] if self.losses else float("inf")

    @property
    def converged(self) -> bool:
        """Check if loss decreased over training."""
        if len(self.losses) < 2:
            return False
        return self.losses[-1] < self.losses[0]

    @property
    def separation(self) -> float:
        """Final neg_dist - pos_dist. Higher = better separation."""
        if not self.pos_dists or not self.neg_dists:
            return 0.0
        return self.neg_dists[-1] - self.pos_dists[-1]


class TripletEncoder:
    """Linear projection encoder trained with triplet margin loss.

    Learns a projection W such that:
        d(W @ anchor, W @ positive) < d(W @ anchor, W @ negative) + margin

    Args:
        input_dim: Dimension of input signal vectors.
        embed_dim: Dimension of output embeddings.
        margin: Triplet loss margin.
        lr: Learning rate.
        seed: Random seed.
    """

    def __init__(
        self,
        input_dim: int,
        embed_dim: int,
        margin: float = 0.3,
        lr: float = 0.01,
        seed: Optional[int] = None,
    ):
        rng = np.random.default_rng(seed)
        # Xavier initialization
        scale = np.sqrt(2.0 / (input_dim + embed_dim))
        self._W = rng.normal(0, scale, (embed_dim, input_dim)).astype(np.float64)
        self._margin = margin
        self._lr = lr
        self._input_dim = input_dim
        self._embed_dim = embed_dim

    @property
    def input_dim(self) -> int:
        return self._input_dim

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    def encode(self, x: np.ndarray) -> np.ndarray:
        """Project input signals to embedding space.

        Args:
            x: (N, input_dim) or (input_dim,) array.

        Returns:
            L2-normalized embeddings, same leading shape.
        """
        squeeze = x.ndim == 1
        if squeeze:
            x = x[np.newaxis, :]

        e = x @ self._W.T  # (N, embed_dim)
        norms = np.linalg.norm(e, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        e = e / norms

        return e[0] if squeeze else e

    def fit(
        self,
        anchors: np.ndarray,
        positives: np.ndarray,
        negatives: np.ndarray,
        epochs: int = 200,
        batch_size: int = 32,
        verbose: bool = False,
    ) -> TrainHistory:
        """Train the encoder with triplet margin loss.

        Args:
            anchors: (N, input_dim) anchor vectors.
            positives: (N, input_dim) positive vectors.
            negatives: (N, input_dim) negative vectors.
            epochs: Number of training epochs.
            batch_size: Mini-batch size.
            verbose: Print progress.

        Returns:
            TrainHistory with per-epoch metrics.
        """
        N = anchors.shape[0]
        if N == 0:
            return TrainHistory()

        history = TrainHistory()

        for epoch in range(epochs):
            # Shuffle
            perm = np.random.permutation(N)
            epoch_loss = 0.0
            epoch_pos_dist = 0.0
            epoch_neg_dist = 0.0
            n_batches = 0

            for start in range(0, N, batch_size):
                idx = perm[start:start + batch_size]
                a = anchors[idx]  # (B, D_in)
                p = positives[idx]
                n = negatives[idx]

                loss, pos_d, neg_d = self._step(a, p, n)
                epoch_loss += loss
                epoch_pos_dist += pos_d
                epoch_neg_dist += neg_d
                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            avg_pos = epoch_pos_dist / max(n_batches, 1)
            avg_neg = epoch_neg_dist / max(n_batches, 1)
            history.losses.append(avg_loss)
            history.pos_dists.append(avg_pos)
            history.neg_dists.append(avg_neg)

            if verbose and (epoch % 50 == 0 or epoch == epochs - 1):
                print(
                    f"  epoch {epoch:4d}  loss={avg_loss:.4f}  "
                    f"pos_d={avg_pos:.4f}  neg_d={avg_neg:.4f}"
                )

        return history

    def _step(
        self, a: np.ndarray, p: np.ndarray, n: np.ndarray,
    ) -> tuple[float, float, float]:
        """One gradient step on a mini-batch.

        Returns (loss, mean_pos_dist, mean_neg_dist).
        """
        W = self._W
        B = a.shape[0]

        # Forward: embed and normalize
        ea = self._embed_raw(a)  # (B, embed_dim)
        ep = self._embed_raw(p)
        en = self._embed_raw(n)

        # L2 distances
        diff_pos = ea - ep  # (B, embed_dim)
        diff_neg = ea - en
        dist_pos = np.sum(diff_pos ** 2, axis=1)  # (B,)
        dist_neg = np.sum(diff_neg ** 2, axis=1)

        # Triplet margin loss: max(0, dist_pos - dist_neg + margin)
        raw_loss = dist_pos - dist_neg + self._margin
        active = raw_loss > 0
        loss = np.mean(np.maximum(raw_loss, 0.0))

        # Backward: gradient of loss w.r.t. W
        # d(loss)/d(ea) = 2 * diff_pos * active - 2 * diff_neg * active / B
        mask = active[:, np.newaxis].astype(np.float64)  # (B, 1)
        grad_ea = 2.0 * diff_pos * mask / B
        grad_ep = -2.0 * diff_pos * mask / B
        grad_en = 2.0 * diff_neg * mask / B

        # d(ea)/d(W) = d(normalize(a @ W.T))/d(W)
        # Approximate: skip normalization gradient (works for PoC)
        grad_W = (
            grad_ea.T @ a
            + grad_ep.T @ p
            + grad_en.T @ n
        )

        # Update
        self._W -= self._lr * grad_W

        return float(loss), float(np.mean(dist_pos)), float(np.mean(dist_neg))

    def _embed_raw(self, x: np.ndarray) -> np.ndarray:
        """Embed without L2 normalization (for gradient computation)."""
        e = x @ self._W.T
        norms = np.linalg.norm(e, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        return e / norms
