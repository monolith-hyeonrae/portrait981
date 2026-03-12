"""Stage 3: Contrastive pair construction from agreement scores.

Given a sequence of HintFrames with agreement scores, construct
(anchor, positive, negative) triplets for contrastive learning.

    miner = PairMiner(positive_threshold=0.6, negative_threshold=0.3)
    pairs = miner.mine(frames, agreements)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

from visualbind.types import AgreementResult, HintFrame


@dataclass(frozen=True)
class ContrastivePair:
    """A (anchor, positive, negative) triplet for contrastive learning.

    Attributes:
        anchor: Flat signal vector of the anchor frame.
        positive: Flat signal vector of a similar (high-agreement) frame.
        negative: Flat signal vector of a dissimilar (low-agreement) frame.
        anchor_agreement: Agreement score of the anchor frame.
        positive_agreement: Agreement score of the positive frame.
        negative_agreement: Agreement score of the negative frame.
    """

    anchor: tuple[float, ...]
    positive: tuple[float, ...]
    negative: tuple[float, ...]
    anchor_agreement: float = 0.0
    positive_agreement: float = 0.0
    negative_agreement: float = 0.0


@dataclass
class PairMiningResult:
    """Result from pair mining.

    Attributes:
        pairs: Contrastive triplets.
        n_high: Number of high-agreement frames.
        n_low: Number of low-agreement frames.
        n_ambiguous: Number of ambiguous (dropped) frames.
    """

    pairs: List[ContrastivePair] = field(default_factory=list)
    n_high: int = 0
    n_low: int = 0
    n_ambiguous: int = 0

    def as_arrays(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Convert pairs to numpy arrays (anchors, positives, negatives).

        Returns:
            Tuple of (N, D) arrays where N=len(pairs), D=signal dimension.
        """
        if not self.pairs:
            return np.empty((0, 0)), np.empty((0, 0)), np.empty((0, 0))
        anchors = np.array([p.anchor for p in self.pairs], dtype=np.float32)
        positives = np.array([p.positive for p in self.pairs], dtype=np.float32)
        negatives = np.array([p.negative for p in self.pairs], dtype=np.float32)
        return anchors, positives, negatives


class PairMiner:
    """Mines contrastive pairs from agreement-scored frames.

    Frames are split into three groups by agreement score:
    - high (>= positive_threshold): candidates for anchors and positives
    - low (< negative_threshold): candidates for negatives
    - ambiguous (between thresholds): dropped

    Triplets are formed by pairing high-agreement frames with each other
    (as anchor-positive) and with low-agreement frames (as negatives).

    Args:
        positive_threshold: Minimum agreement for positive pairs.
        negative_threshold: Maximum agreement for negative frames.
        max_pairs_per_anchor: Maximum negative pairs per anchor-positive pair.
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        positive_threshold: float = 0.5,
        negative_threshold: float = 0.2,
        max_pairs_per_anchor: int = 3,
        seed: Optional[int] = None,
    ):
        self._pos_thresh = positive_threshold
        self._neg_thresh = negative_threshold
        self._max_pairs = max_pairs_per_anchor
        self._rng = np.random.default_rng(seed)

    def mine(
        self,
        frames: Sequence[HintFrame],
        agreements: Sequence[AgreementResult],
    ) -> PairMiningResult:
        """Mine contrastive triplets from frames and their agreement scores.

        Args:
            frames: Sequence of HintFrames (must have same length as agreements).
            agreements: Corresponding agreement results.

        Returns:
            PairMiningResult with triplets.
        """
        if len(frames) != len(agreements):
            raise ValueError(
                f"frames ({len(frames)}) and agreements ({len(agreements)}) "
                f"must have the same length"
            )

        # Partition frames by agreement score
        high_indices: List[int] = []
        low_indices: List[int] = []
        n_ambiguous = 0

        for i, agr in enumerate(agreements):
            if agr.score >= self._pos_thresh:
                high_indices.append(i)
            elif agr.score < self._neg_thresh:
                low_indices.append(i)
            else:
                n_ambiguous += 1

        # Get flat vectors
        vectors = [f.flat_vector() for f in frames]
        scores = [a.score for a in agreements]

        # Mine pairs: for each pair of high-agreement frames,
        # sample negatives from low-agreement frames
        pairs: List[ContrastivePair] = []

        if len(high_indices) < 2 or len(low_indices) == 0:
            return PairMiningResult(
                pairs=[],
                n_high=len(high_indices),
                n_low=len(low_indices),
                n_ambiguous=n_ambiguous,
            )

        for i, anchor_idx in enumerate(high_indices):
            # Pick positives from other high-agreement frames
            other_high = [j for j in high_indices if j != anchor_idx]
            if not other_high:
                continue

            # Select one positive (closest agreement score)
            pos_idx = min(
                other_high,
                key=lambda j: abs(scores[j] - scores[anchor_idx]),
            )

            # Sample negatives
            n_neg = min(self._max_pairs, len(low_indices))
            neg_indices = self._rng.choice(
                low_indices, size=n_neg, replace=False,
            )

            for neg_idx in neg_indices:
                pairs.append(ContrastivePair(
                    anchor=vectors[anchor_idx],
                    positive=vectors[pos_idx],
                    negative=vectors[neg_idx],
                    anchor_agreement=scores[anchor_idx],
                    positive_agreement=scores[pos_idx],
                    negative_agreement=scores[neg_idx],
                ))

        return PairMiningResult(
            pairs=pairs,
            n_high=len(high_indices),
            n_low=len(low_indices),
            n_ambiguous=n_ambiguous,
        )
