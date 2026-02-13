"""MemoryBank — Temporal identity memory bank.

identity_memory.md 설계를 따르는 핵심 구현.
인물별 Face-ID 임베딩을 시간 경과에 따라 누적/관리하는 메모리 뱅크.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np

from momentbank.types import (
    MemoryNode,
    NodeMeta,
    MatchResult,
    RefQuery,
    RefSelection,
)


class MemoryBank:
    """인물별 다중 centroid 메모리 뱅크.

    단일 prototype(medoid)의 한계를 극복하여 상태별 다중 centroid를 유지.

    Args:
        person_id: Person identifier.
        k_max: Maximum number of memory nodes per person.
        alpha: EMA update coefficient (lower = more stable).
        tau_merge: Merge threshold — cosine sim >= this triggers EMA merge.
        tau_new: New node threshold — cosine sim < this creates new node.
        tau_close: Close threshold — nodes with cos > this get merged in compact.
        q_update_min: Minimum quality to update bank (below = match only).
        q_new_min: Minimum quality to create a new node.
        temperature: Softmax temperature for select_refs.
        top_p: Number of top nodes to select in select_refs.
        anchor_min_weight: Minimum weight for anchor nodes in select_refs.
    """

    def __init__(
        self,
        person_id: int = 0,
        k_max: int = 10,
        alpha: float = 0.1,
        tau_merge: float = 0.5,
        tau_new: float = 0.3,
        tau_close: float = 0.8,
        q_update_min: float = 0.3,
        q_new_min: float = 0.5,
        temperature: float = 0.1,
        top_p: int = 3,
        anchor_min_weight: float = 0.15,
    ):
        self.person_id = person_id
        self.k_max = k_max
        self.alpha = alpha
        self.tau_merge = tau_merge
        self.tau_new = tau_new
        self.tau_close = tau_close
        self.q_update_min = q_update_min
        self.q_new_min = q_new_min
        self.temperature = temperature
        self.top_p = top_p
        self.anchor_min_weight = anchor_min_weight

        self.nodes: list[MemoryNode] = []
        self._next_id: int = 0

    def update(self, e_id: np.ndarray, quality: float, meta: dict, image_path: str) -> None:
        """Add observation to memory bank.

        Logic:
        1. Quality gate: if quality < q_update_min, skip
        2. Find nearest node by cosine similarity
        3. If sim >= tau_merge: EMA merge into existing node
        4. If sim < tau_new and quality >= q_new_min: create new node
        5. If k > k_max: evict or merge

        Args:
            e_id: L2-normalized Face-ID embedding.
            quality: Quality score [0, 1].
            meta: Bucket metadata, e.g. {"yaw": "[-5,5]", "expression": "smile"}.
            image_path: Path to the face image.
        """
        # (A) Quality gate
        if quality < self.q_update_min:
            return

        # Empty bank — create first node
        if not self.nodes:
            if quality >= self.q_new_min:
                self._create_node(e_id, quality, meta, image_path)
            return

        # (B) Nearest node search
        sims = [self._cosine_sim(e_id, m.vec_id) for m in self.nodes]
        i_star = int(np.argmax(sims))
        best_sim = sims[i_star]

        # (C) EMA merge (close match)
        if best_sim >= self.tau_merge:
            m = self.nodes[i_star]
            merged_vec = (1 - self.alpha) * m.vec_id + self.alpha * e_id
            m.vec_id = self._l2_normalize(merged_vec)
            m.meta_hist.update(meta, quality)
            m.update_rep_images(image_path, quality)
            return

        # (D) New node creation (distant embedding + high quality)
        if best_sim < self.tau_new and quality >= self.q_new_min:
            self._create_node(e_id, quality, meta, image_path)

            # (E) Bank management
            if len(self.nodes) > self.k_max:
                self._evict_or_merge()

    def match(self, e_id: np.ndarray) -> MatchResult:
        """Check identity stability against bank.

        Args:
            e_id: L2-normalized Face-ID embedding to match.

        Returns:
            MatchResult with stable_score = max cosine sim to any node.
        """
        if not self.nodes:
            return MatchResult(stable_score=0.0, best_node_id=-1, top3=[])

        sims = [(self.nodes[i].node_id, self._cosine_sim(e_id, self.nodes[i].vec_id))
                for i in range(len(self.nodes))]

        # Sort by sim descending
        sorted_sims = sorted(sims, key=lambda x: -x[1])

        return MatchResult(
            stable_score=sorted_sims[0][1],
            best_node_id=sorted_sims[0][0],
            top3=sorted_sims[:3],
        )

    def select_refs(self, query: RefQuery) -> RefSelection:
        """Select reference images for diffusion generation.

        Bucket-based matching (방식 3 from identity_memory.md):
        1. Score each node by meta_hist overlap with query target_buckets
        2. Ensure anchor nodes get minimum weight
        3. Select top-p nodes
        4. Collect rep_images from selected nodes
        5. Split into anchor/coverage/challenge refs

        Args:
            query: RefQuery with target bucket conditions.

        Returns:
            RefSelection with anchor/coverage/challenge refs and paths_for_comfy.
        """
        if not self.nodes:
            return RefSelection(
                anchor_refs=[], coverage_refs=[],
                challenge_refs=[], paths_for_comfy=[],
            )

        # Score each node by bucket coverage
        scores = np.array([
            node.meta_hist.coverage_score(query.target_buckets)
            for node in self.nodes
        ])

        # Softmax weighted selection
        if self.temperature > 0:
            exp_scores = np.exp(scores / self.temperature)
            weights = exp_scores / exp_scores.sum()
        else:
            weights = np.ones(len(self.nodes)) / len(self.nodes)

        # Anchor minimum weight guarantee:
        # Node 0 (first created, typically frontal high-quality) treated as anchor
        if len(self.nodes) > 0:
            weights[0] = max(weights[0], self.anchor_min_weight)
            weights = weights / weights.sum()  # re-normalize

        # Top-p node selection
        top_count = min(self.top_p, len(self.nodes))
        top_indices = np.argsort(weights)[-top_count:][::-1]

        # Collect reference images
        anchor_refs: list[str] = []
        coverage_refs: list[str] = []
        challenge_refs: list[str] = []

        for rank, idx in enumerate(top_indices):
            node = self.nodes[idx]
            if not node.rep_images:
                continue

            if idx == 0:
                # Anchor node — always goes to anchor_refs
                anchor_refs.extend(node.rep_images[:2])
            elif rank < top_count - 1:
                # Mid-ranked nodes — coverage
                coverage_refs.extend(node.rep_images[:2])
            else:
                # Last ranked node — challenge (novel conditions)
                challenge_refs.extend(node.rep_images[:1])

        # Ensure anchor has at least one image from the best quality node
        if not anchor_refs and self.nodes[0].rep_images:
            anchor_refs.append(self.nodes[0].rep_images[0])

        paths_for_comfy = anchor_refs + coverage_refs + challenge_refs

        return RefSelection(
            anchor_refs=anchor_refs,
            coverage_refs=coverage_refs,
            challenge_refs=challenge_refs,
            paths_for_comfy=paths_for_comfy,
        )

    def _create_node(
        self, e_id: np.ndarray, quality: float, meta: dict, image_path: str,
    ) -> MemoryNode:
        """Create and append a new memory node."""
        node = MemoryNode(
            node_id=self._next_node_id(),
            vec_id=e_id.copy(),
            rep_images=[image_path],
            meta_hist=NodeMeta.from_single(meta, quality),
            _rep_qualities=[quality],
        )
        self.nodes.append(node)
        return node

    def _evict_or_merge(self) -> None:
        """Remove or merge nodes when k > k_max.

        Priority: lowest hit_count + lowest quality_best first.
        """
        if len(self.nodes) <= self.k_max:
            return

        # Try compact first (merge very similar nodes)
        self._compact()
        if len(self.nodes) <= self.k_max:
            return

        # Evict: sort by (hit_count, quality_best) ascending, remove first
        self.nodes.sort(key=lambda n: (n.meta_hist.hit_count, n.meta_hist.quality_best))
        self.nodes.pop(0)

    def _compact(self) -> None:
        """Merge nodes that are too similar (cos > tau_close)."""
        for a, b in combinations(self.nodes, 2):
            if self._cosine_sim(a.vec_id, b.vec_id) > self.tau_close:
                merged = self._merge_nodes(a, b)
                self.nodes.remove(a)
                self.nodes.remove(b)
                self.nodes.append(merged)
                break  # one merge at a time

    def _merge_nodes(self, a: MemoryNode, b: MemoryNode) -> MemoryNode:
        """Merge two nodes, keeping the one with higher hit_count as base."""
        # Keep the node with more history as base
        if a.meta_hist.hit_count >= b.meta_hist.hit_count:
            base, other = a, b
        else:
            base, other = b, a

        # Weighted average of embeddings
        total_hits = base.meta_hist.hit_count + other.meta_hist.hit_count
        if total_hits > 0:
            w_base = base.meta_hist.hit_count / total_hits
            w_other = other.meta_hist.hit_count / total_hits
        else:
            w_base = w_other = 0.5

        merged_vec = w_base * base.vec_id + w_other * other.vec_id
        merged_vec = self._l2_normalize(merged_vec)

        # Merge rep_images by quality (top 3)
        all_images = list(zip(base._rep_qualities, base.rep_images)) + \
                     list(zip(other._rep_qualities, other.rep_images))
        all_images.sort(key=lambda x: -x[0])
        top_images = all_images[:3]

        # Merge meta histograms
        merged_meta = NodeMeta(
            yaw_bins={**other.meta_hist.yaw_bins, **base.meta_hist.yaw_bins},
            pitch_bins={**other.meta_hist.pitch_bins, **base.meta_hist.pitch_bins},
            expression_bins={**other.meta_hist.expression_bins, **base.meta_hist.expression_bins},
            quality_best=max(base.meta_hist.quality_best, other.meta_hist.quality_best),
            quality_mean=(
                base.meta_hist.quality_mean * base.meta_hist.hit_count +
                other.meta_hist.quality_mean * other.meta_hist.hit_count
            ) / total_hits if total_hits > 0 else 0.0,
            hit_count=total_hits,
            last_updated_ms=max(base.meta_hist.last_updated_ms, other.meta_hist.last_updated_ms),
        )

        # Properly merge bin counts (sum values for same keys)
        for key in ("yaw_bins", "pitch_bins", "expression_bins"):
            base_bins = getattr(base.meta_hist, key)
            other_bins = getattr(other.meta_hist, key)
            merged_bins = dict(other_bins)
            for k, v in base_bins.items():
                merged_bins[k] = merged_bins.get(k, 0) + v
            setattr(merged_meta, key, merged_bins)

        return MemoryNode(
            node_id=base.node_id,
            vec_id=merged_vec,
            rep_images=[p for _, p in top_images],
            meta_hist=merged_meta,
            _rep_qualities=[q for q, _ in top_images],
        )

    @staticmethod
    def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity between L2-normalized vectors."""
        return float(np.dot(a, b))

    @staticmethod
    def _l2_normalize(v: np.ndarray) -> np.ndarray:
        """L2 normalize a vector."""
        norm = np.linalg.norm(v)
        if norm > 0:
            return v / norm
        return v

    def _next_node_id(self) -> int:
        nid = self._next_id
        self._next_id += 1
        return nid
