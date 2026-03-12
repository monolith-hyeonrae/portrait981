"""Tests for TripletEncoder."""

import numpy as np
import pytest

from visualbind import TripletEncoder


class TestTripletEncoder:
    def test_encode_shape(self):
        enc = TripletEncoder(input_dim=10, embed_dim=4, seed=42)
        x = np.random.randn(5, 10).astype(np.float32)
        out = enc.encode(x)
        assert out.shape == (5, 4)

    def test_encode_single(self):
        enc = TripletEncoder(input_dim=10, embed_dim=4, seed=42)
        x = np.random.randn(10).astype(np.float32)
        out = enc.encode(x)
        assert out.shape == (4,)

    def test_encode_normalized(self):
        enc = TripletEncoder(input_dim=10, embed_dim=4, seed=42)
        x = np.random.randn(5, 10).astype(np.float32)
        out = enc.encode(x)
        norms = np.linalg.norm(out, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-6)

    def test_fit_reduces_loss(self):
        """Training should reduce loss on separable data."""
        rng = np.random.default_rng(42)
        dim = 8
        n = 50

        # Create clearly separable data
        anchors = rng.normal(1.0, 0.1, (n, dim)).astype(np.float32)
        positives = rng.normal(1.0, 0.1, (n, dim)).astype(np.float32)
        negatives = rng.normal(-1.0, 0.1, (n, dim)).astype(np.float32)

        enc = TripletEncoder(input_dim=dim, embed_dim=4, margin=0.3, lr=0.01, seed=42)
        history = enc.fit(anchors, positives, negatives, epochs=100)

        assert len(history.losses) == 100
        # Loss should decrease
        assert history.losses[-1] <= history.losses[0]

    def test_fit_empty_data(self):
        enc = TripletEncoder(input_dim=8, embed_dim=4)
        history = enc.fit(
            np.empty((0, 8)),
            np.empty((0, 8)),
            np.empty((0, 8)),
        )
        assert len(history.losses) == 0
        assert history.final_loss == float("inf")

    def test_history_properties(self):
        rng = np.random.default_rng(42)
        dim = 8
        n = 30

        # Use overlapping clusters so loss starts > 0 and can decrease
        anchors = rng.normal(0.5, 0.3, (n, dim)).astype(np.float32)
        positives = rng.normal(0.5, 0.3, (n, dim)).astype(np.float32)
        negatives = rng.normal(-0.5, 0.3, (n, dim)).astype(np.float32)

        enc = TripletEncoder(input_dim=dim, embed_dim=4, margin=0.3, lr=0.01, seed=42)
        history = enc.fit(anchors, positives, negatives, epochs=100)

        assert len(history.pos_dists) == 100
        assert len(history.neg_dists) == 100
        assert history.final_loss >= 0

    def test_deterministic_with_seed(self):
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, (3, 8)).astype(np.float32)

        enc1 = TripletEncoder(input_dim=8, embed_dim=4, seed=99)
        enc2 = TripletEncoder(input_dim=8, embed_dim=4, seed=99)

        out1 = enc1.encode(x)
        out2 = enc2.encode(x)
        np.testing.assert_allclose(out1, out2)


class TestEndToEnd:
    """Verify encoder produces meaningful embeddings from visualbind pipeline."""

    def test_clusters_separate(self):
        """Same-class samples should be closer than cross-class after training."""
        rng = np.random.default_rng(42)
        dim = 6
        n = 40

        # Two clusters
        class_a = rng.normal([1, 1, 0, 0, 0, 0], 0.2, (n, dim)).astype(np.float32)
        class_b = rng.normal([0, 0, 1, 1, 0, 0], 0.2, (n, dim)).astype(np.float32)

        # Build triplets: anchor from A, positive from A, negative from B
        anchors = class_a
        positives = np.roll(class_a, 1, axis=0)  # shifted A
        negatives = class_b

        enc = TripletEncoder(input_dim=dim, embed_dim=3, margin=0.3, lr=0.01, seed=42)
        enc.fit(anchors, positives, negatives, epochs=200)

        # Encode both classes
        emb_a = enc.encode(class_a)
        emb_b = enc.encode(class_b)

        # Mean intra-class distance should be less than inter-class
        intra_dist = np.mean(np.linalg.norm(emb_a - emb_a.mean(axis=0), axis=1))
        inter_dist = np.linalg.norm(emb_a.mean(axis=0) - emb_b.mean(axis=0))

        assert inter_dist > intra_dist
