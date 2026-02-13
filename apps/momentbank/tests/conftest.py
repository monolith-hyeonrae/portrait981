"""Shared fixtures for momentbank tests.

All embeddings are synthetic — NO ML models needed.
"""

import numpy as np
import pytest


@pytest.fixture
def random_embedding():
    """Generate random L2-normalized 512D vector."""
    rng = np.random.default_rng(42)
    v = rng.standard_normal(512).astype(np.float32)
    return v / np.linalg.norm(v)


@pytest.fixture
def similar_embedding(random_embedding):
    """Generate embedding similar to random_embedding (cos > 0.9)."""
    rng = np.random.default_rng(99)
    noise = rng.standard_normal(512).astype(np.float32) * 0.02
    v = random_embedding + noise
    return v / np.linalg.norm(v)


@pytest.fixture
def distant_embedding():
    """Generate embedding far from typical (cos < 0.3 from random_embedding)."""
    rng = np.random.default_rng(777)
    v = rng.standard_normal(512).astype(np.float32)
    return v / np.linalg.norm(v)


@pytest.fixture
def make_embedding():
    """Factory fixture for generating deterministic L2-normalized embeddings."""
    def _make(seed: int = 0, dim: int = 512) -> np.ndarray:
        rng = np.random.default_rng(seed)
        v = rng.standard_normal(dim).astype(np.float32)
        return v / np.linalg.norm(v)
    return _make
