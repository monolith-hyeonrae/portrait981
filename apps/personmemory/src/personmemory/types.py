"""Momentbank data types.

identity_memory.md 설계를 따르는 데이터 구조 정의.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field


@dataclass
class NodeMeta:
    """Memory node의 버킷 분포 + 품질 통계."""

    yaw_bins: dict[str, int] = field(default_factory=dict)
    pitch_bins: dict[str, int] = field(default_factory=dict)
    expression_bins: dict[str, int] = field(default_factory=dict)
    quality_best: float = 0.0
    quality_mean: float = 0.0
    hit_count: int = 0
    last_updated_ms: float = 0.0

    def update(self, meta: dict, quality: float) -> None:
        """Update histogram counts and quality stats.

        Args:
            meta: Bucket metadata, e.g. {"yaw": "[-5,5]", "expression": "smile"}.
            quality: Quality score of the observation.
        """
        # Increment bin counts from meta
        if "yaw" in meta:
            self.yaw_bins[meta["yaw"]] = self.yaw_bins.get(meta["yaw"], 0) + 1
        if "pitch" in meta:
            self.pitch_bins[meta["pitch"]] = self.pitch_bins.get(meta["pitch"], 0) + 1
        if "expression" in meta:
            self.expression_bins[meta["expression"]] = (
                self.expression_bins.get(meta["expression"], 0) + 1
            )

        # Update quality stats
        self.quality_best = max(self.quality_best, quality)
        # Running average: mean = (mean * count + new) / (count + 1)
        total = self.quality_mean * self.hit_count + quality
        self.hit_count += 1
        self.quality_mean = total / self.hit_count

    @classmethod
    def from_single(cls, meta: dict, quality: float) -> NodeMeta:
        """Create NodeMeta from a single observation."""
        node_meta = cls()
        node_meta.update(meta, quality)
        return node_meta

    def coverage_score(self, target_buckets: dict[str, str]) -> float:
        """Score how well this node's histogram covers the target buckets.

        Args:
            target_buckets: e.g. {"yaw": "[-5,5]", "expression": "smile"}

        Returns:
            Score in [0, 1] — fraction of target buckets matched.
        """
        if not target_buckets:
            return 0.0

        matched = 0
        total = len(target_buckets)

        for key, bucket_val in target_buckets.items():
            bins = getattr(self, f"{key}_bins", {})
            if bins.get(bucket_val, 0) > 0:
                matched += 1

        return matched / total


@dataclass
class MemoryNode:
    """인물의 특정 상태 영역을 대표하는 메모리 노드.

    예: "정면 나", "어두운 조명 나", "옆얼굴 나".
    """

    node_id: int
    vec_id: np.ndarray  # Face-ID embedding, L2 normalized
    rep_images: list[str] = field(default_factory=list)  # up to 3 paths, quality sorted
    meta_hist: NodeMeta = field(default_factory=NodeMeta)

    # Quality values for each rep_image (parallel list)
    _rep_qualities: list[float] = field(default_factory=list)

    def update_rep_images(self, image_path: str, quality: float, max_images: int = 3) -> None:
        """Keep top-k images by quality.

        Args:
            image_path: Path to the new image.
            quality: Quality score of the new image.
            max_images: Maximum number of representative images.
        """
        self.rep_images.append(image_path)
        self._rep_qualities.append(quality)

        if len(self.rep_images) > max_images:
            # Find index of lowest quality
            min_idx = min(range(len(self._rep_qualities)), key=lambda i: self._rep_qualities[i])
            # Only remove if the new image is better
            if self._rep_qualities[min_idx] < quality or min_idx < len(self.rep_images) - 1:
                self.rep_images.pop(min_idx)
                self._rep_qualities.pop(min_idx)
            else:
                # New image is the worst — remove it (the last one)
                self.rep_images.pop()
                self._rep_qualities.pop()

        # Sort both lists by quality descending
        if len(self.rep_images) > 1:
            paired = sorted(
                zip(self._rep_qualities, self.rep_images),
                key=lambda x: -x[0],
            )
            self._rep_qualities = [q for q, _ in paired]
            self.rep_images = [p for _, p in paired]


@dataclass
class MatchResult:
    """Identity match result from MemoryBank.match()."""

    stable_score: float  # cosine sim to best node
    best_node_id: int
    top3: list[tuple[int, float]]  # [(node_id, sim), ...]


@dataclass
class RefQuery:
    """Query for reference image selection (bucket-based matching)."""

    target_buckets: dict[str, str] = field(default_factory=dict)
    # e.g. {"yaw": "[-5,5]", "expression": "smile"}


@dataclass
class RefSelection:
    """Reference image selection result for ComfyUI."""

    anchor_refs: list[str]  # image paths
    coverage_refs: list[str]
    challenge_refs: list[str]
    paths_for_comfy: list[str]  # all paths combined for ComfyUI
