"""Tests for MemoryBank core logic."""

import numpy as np
import pytest

from momentbank.bank import MemoryBank
from momentbank.types import MatchResult, RefQuery, RefSelection


class TestUpdate:
    def test_update_creates_first_node(self, random_embedding):
        """Empty bank, update creates node."""
        bank = MemoryBank()
        bank.update(random_embedding, quality=0.8, meta={"yaw": "[-5,5]"}, image_path="img0.jpg")

        assert len(bank.nodes) == 1
        assert bank.nodes[0].node_id == 0
        assert bank.nodes[0].rep_images == ["img0.jpg"]
        assert bank.nodes[0].meta_hist.hit_count == 1
        assert bank.nodes[0].meta_hist.quality_best == 0.8

    def test_update_merges_similar(self, random_embedding, similar_embedding):
        """sim >= tau_merge triggers EMA merge."""
        bank = MemoryBank(tau_merge=0.5)
        bank.update(random_embedding, quality=0.8, meta={"yaw": "[-5,5]"}, image_path="img0.jpg")

        # Similar embedding should merge into existing node
        sim = float(np.dot(random_embedding, similar_embedding))
        assert sim > 0.5  # verify fixture is similar enough

        bank.update(similar_embedding, quality=0.85, meta={"yaw": "[5,15]"}, image_path="img1.jpg")

        assert len(bank.nodes) == 1
        assert bank.nodes[0].meta_hist.hit_count == 2
        assert bank.nodes[0].meta_hist.quality_best == 0.85
        assert len(bank.nodes[0].rep_images) == 2

    def test_update_creates_new_node(self, random_embedding, distant_embedding):
        """sim < tau_new creates new node."""
        bank = MemoryBank(tau_new=0.3)
        bank.update(random_embedding, quality=0.8, meta={"yaw": "[-5,5]"}, image_path="img0.jpg")

        sim = float(np.dot(random_embedding, distant_embedding))
        assert sim < 0.3  # verify fixture is distant enough

        bank.update(distant_embedding, quality=0.7, meta={"yaw": "[30,45]"}, image_path="img1.jpg")

        assert len(bank.nodes) == 2
        assert bank.nodes[1].node_id == 1

    def test_update_quality_gate(self, random_embedding):
        """Low quality skips update."""
        bank = MemoryBank(q_update_min=0.3)
        bank.update(random_embedding, quality=0.1, meta={}, image_path="low_q.jpg")

        assert len(bank.nodes) == 0

    def test_update_between_thresholds(self, make_embedding):
        """tau_new < sim < tau_merge does nothing (no merge, no new node)."""
        # Create two embeddings with moderate similarity
        e1 = make_embedding(seed=10)

        bank = MemoryBank(tau_merge=0.8, tau_new=0.2, q_update_min=0.1)
        bank.update(e1, quality=0.8, meta={}, image_path="img0.jpg")
        assert len(bank.nodes) == 1

        # Create embedding with moderate similarity to e1
        noise = np.random.default_rng(20).standard_normal(512).astype(np.float32) * 0.06
        e2 = e1 + noise
        e2 = e2 / np.linalg.norm(e2)

        sim = float(np.dot(e1, e2))
        # Ensure it's between thresholds
        assert 0.2 < sim < 0.8, f"sim={sim}, need 0.2 < sim < 0.8"

        bank.update(e2, quality=0.8, meta={}, image_path="img1.jpg")

        # No merge (sim < tau_merge) and no new node (sim >= tau_new)
        assert len(bank.nodes) == 1
        assert bank.nodes[0].meta_hist.hit_count == 1  # unchanged

    def test_update_first_node_quality_gate(self, random_embedding):
        """First node also requires q_new_min quality."""
        bank = MemoryBank(q_update_min=0.1, q_new_min=0.5)
        bank.update(random_embedding, quality=0.3, meta={}, image_path="low_q.jpg")
        assert len(bank.nodes) == 0


class TestMatch:
    def test_match_returns_best(self, random_embedding, distant_embedding):
        """Correct stable_score and node_id."""
        bank = MemoryBank(tau_new=0.3)
        bank.update(random_embedding, quality=0.8, meta={}, image_path="img0.jpg")
        bank.update(distant_embedding, quality=0.7, meta={}, image_path="img1.jpg")

        result = bank.match(random_embedding)

        assert isinstance(result, MatchResult)
        assert result.stable_score > 0.9  # should match closely
        assert result.best_node_id == 0
        assert len(result.top3) == 2

    def test_match_empty_bank(self, random_embedding):
        """Empty bank returns zero score."""
        bank = MemoryBank()
        result = bank.match(random_embedding)

        assert result.stable_score == 0.0
        assert result.best_node_id == -1
        assert result.top3 == []

    def test_match_top3_ordering(self, make_embedding):
        """Top3 is ordered by descending similarity."""
        bank = MemoryBank(tau_new=0.1, q_new_min=0.1, q_update_min=0.1)

        # Create 4 distinct nodes
        for i in range(4):
            e = make_embedding(seed=i * 100)
            bank.update(e, quality=0.8, meta={}, image_path=f"img{i}.jpg")

        query = make_embedding(seed=0)  # same as first node
        result = bank.match(query)

        assert len(result.top3) == 3
        # Top3 should be descending
        assert result.top3[0][1] >= result.top3[1][1]
        assert result.top3[1][1] >= result.top3[2][1]


class TestSelectRefs:
    def test_select_refs_bucket_matching(self, make_embedding):
        """Correct refs for query with matching buckets."""
        bank = MemoryBank(tau_new=0.1, q_new_min=0.1, q_update_min=0.1)

        # Node 0: frontal
        bank.update(
            make_embedding(seed=0), quality=0.9,
            meta={"yaw": "[-5,5]", "expression": "neutral"},
            image_path="frontal.jpg",
        )
        # Node 1: side view with smile
        bank.update(
            make_embedding(seed=100), quality=0.8,
            meta={"yaw": "[30,45]", "expression": "smile"},
            image_path="side_smile.jpg",
        )
        # Node 2: another view
        bank.update(
            make_embedding(seed=200), quality=0.7,
            meta={"yaw": "[15,30]", "expression": "neutral"},
            image_path="mid_view.jpg",
        )

        query = RefQuery(target_buckets={"yaw": "[30,45]", "expression": "smile"})
        result = bank.select_refs(query)

        assert isinstance(result, RefSelection)
        assert len(result.paths_for_comfy) > 0
        # Anchor (node 0) should always be present
        assert "frontal.jpg" in result.paths_for_comfy

    def test_select_refs_anchor_minimum(self, make_embedding):
        """Anchor weight guaranteed even when query doesn't match anchor node."""
        bank = MemoryBank(
            tau_new=0.1, q_new_min=0.1, q_update_min=0.1,
            anchor_min_weight=0.15,
        )

        # Node 0 (anchor): frontal, no matching buckets for query
        bank.update(
            make_embedding(seed=0), quality=0.9,
            meta={"yaw": "[-5,5]", "expression": "neutral"},
            image_path="anchor.jpg",
        )
        # Node 1: matches query perfectly
        bank.update(
            make_embedding(seed=100), quality=0.8,
            meta={"yaw": "[30,45]", "expression": "smile"},
            image_path="match.jpg",
        )

        query = RefQuery(target_buckets={"yaw": "[30,45]", "expression": "smile"})
        result = bank.select_refs(query)

        # Anchor image should be in the result
        assert "anchor.jpg" in result.anchor_refs or "anchor.jpg" in result.paths_for_comfy

    def test_select_refs_empty_bank(self):
        """Empty bank returns empty RefSelection."""
        bank = MemoryBank()
        result = bank.select_refs(RefQuery())

        assert result.anchor_refs == []
        assert result.coverage_refs == []
        assert result.challenge_refs == []
        assert result.paths_for_comfy == []


class TestEviction:
    def test_eviction_removes_lowest(self, make_embedding):
        """Correct eviction priority: lowest hit_count + lowest quality_best first."""
        bank = MemoryBank(k_max=3, tau_new=0.1, q_new_min=0.1, q_update_min=0.1)

        # Create 3 nodes with different hit counts
        for i in range(3):
            e = make_embedding(seed=i * 100)
            bank.update(e, quality=0.5 + i * 0.1, meta={}, image_path=f"img{i}.jpg")

        # Manually bump hit_count of first two nodes
        bank.nodes[0].meta_hist.hit_count = 5
        bank.nodes[0].meta_hist.quality_best = 0.9
        bank.nodes[1].meta_hist.hit_count = 3
        bank.nodes[1].meta_hist.quality_best = 0.8

        assert len(bank.nodes) == 3

        # Add 4th node (should trigger eviction)
        e4 = make_embedding(seed=999)
        bank.update(e4, quality=0.8, meta={}, image_path="img3.jpg")

        assert len(bank.nodes) <= 3
        # Node with lowest hit_count (node 2, hit_count=1) should be evicted
        node_ids = [n.node_id for n in bank.nodes]
        assert 0 in node_ids  # high hit_count preserved
        assert 1 in node_ids  # medium hit_count preserved


class TestCompact:
    def test_compact_merges_similar(self, random_embedding, similar_embedding):
        """Nodes with cos > tau_close merged."""
        bank = MemoryBank(tau_close=0.7, tau_new=0.0, q_new_min=0.1, q_update_min=0.1)

        # Force two very similar nodes
        bank._create_node(random_embedding, 0.8, {"yaw": "[-5,5]"}, "img0.jpg")
        bank._create_node(similar_embedding, 0.85, {"yaw": "[-5,5]"}, "img1.jpg")

        sim = float(np.dot(random_embedding, similar_embedding))
        assert sim > 0.7  # verify they're close enough

        assert len(bank.nodes) == 2
        bank._compact()
        assert len(bank.nodes) == 1

        # Merged node should have combined hit counts
        assert bank.nodes[0].meta_hist.hit_count == 2


class TestRepImages:
    def test_rep_images_max_three(self, random_embedding, similar_embedding):
        """Only top 3 by quality kept."""
        bank = MemoryBank(tau_merge=0.5)
        bank.update(random_embedding, quality=0.6, meta={}, image_path="img_low.jpg")

        # Merge in more images with varying quality
        for i, q in enumerate([0.7, 0.8, 0.9, 0.5]):
            noise = np.random.default_rng(i + 50).standard_normal(512).astype(np.float32) * 0.05
            v = random_embedding + noise
            v = v / np.linalg.norm(v)
            bank.update(v, quality=q, meta={}, image_path=f"img_q{q}.jpg")

        node = bank.nodes[0]
        assert len(node.rep_images) <= 3
        # Top 3 qualities should be 0.9, 0.8, 0.7
        assert node._rep_qualities[0] >= node._rep_qualities[-1]
