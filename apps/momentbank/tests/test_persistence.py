"""Tests for MemoryBank persistence (JSON save/load)."""

import json
import numpy as np
import pytest

from momentbank.bank import MemoryBank
from momentbank.persistence import save_bank, load_bank


class TestSaveLoadRoundtrip:
    def test_save_load_roundtrip(self, tmp_path, random_embedding, distant_embedding):
        """Save then load, verify identical."""
        bank = MemoryBank(person_id=1, k_max=5)
        bank.update(random_embedding, quality=0.8, meta={"yaw": "[-5,5]"}, image_path="img0.jpg")
        bank.update(distant_embedding, quality=0.7, meta={"yaw": "[30,45]"}, image_path="img1.jpg")

        path = tmp_path / "memory_bank.json"
        save_bank(bank, path)

        loaded = load_bank(path)

        assert loaded.person_id == bank.person_id
        assert loaded.k_max == bank.k_max
        assert len(loaded.nodes) == len(bank.nodes)
        assert loaded._next_id == bank._next_id

        for orig, restored in zip(bank.nodes, loaded.nodes):
            assert orig.node_id == restored.node_id
            np.testing.assert_allclose(orig.vec_id, restored.vec_id, atol=1e-5)
            assert orig.rep_images == restored.rep_images
            assert orig.meta_hist.hit_count == restored.meta_hist.hit_count
            assert orig.meta_hist.quality_best == pytest.approx(restored.meta_hist.quality_best)

    def test_roundtrip_preserves_config(self, tmp_path, random_embedding):
        """Config parameters survive round-trip."""
        bank = MemoryBank(
            person_id=2, k_max=7, alpha=0.2, tau_merge=0.6,
            tau_new=0.25, tau_close=0.85, q_update_min=0.4,
            q_new_min=0.6, temperature=0.2, top_p=5,
            anchor_min_weight=0.2,
        )
        bank.update(random_embedding, quality=0.9, meta={}, image_path="img.jpg")

        path = tmp_path / "bank.json"
        save_bank(bank, path)
        loaded = load_bank(path)

        assert loaded.k_max == 7
        assert loaded.alpha == 0.2
        assert loaded.tau_merge == 0.6
        assert loaded.tau_new == 0.25
        assert loaded.tau_close == 0.85
        assert loaded.temperature == 0.2
        assert loaded.top_p == 5


class TestNumpySerialization:
    def test_numpy_serialization(self, tmp_path, random_embedding):
        """vec_id survives JSON conversion."""
        bank = MemoryBank()
        bank.update(random_embedding, quality=0.8, meta={}, image_path="img.jpg")

        path = tmp_path / "bank.json"
        save_bank(bank, path)

        # Verify JSON contains list (not numpy)
        with open(path) as f:
            data = json.load(f)
        vec_data = data["nodes"][0]["vec_id"]
        assert isinstance(vec_data, list)
        assert len(vec_data) == 512

        # Verify loaded back as numpy
        loaded = load_bank(path)
        assert isinstance(loaded.nodes[0].vec_id, np.ndarray)
        assert loaded.nodes[0].vec_id.dtype == np.float32
        np.testing.assert_allclose(
            loaded.nodes[0].vec_id, random_embedding, atol=1e-5,
        )


class TestVersionMetadata:
    def test_version_metadata(self, tmp_path, random_embedding):
        """_version field present."""
        bank = MemoryBank()
        bank.update(random_embedding, quality=0.8, meta={}, image_path="img.jpg")

        path = tmp_path / "bank.json"
        save_bank(bank, path)

        with open(path) as f:
            data = json.load(f)

        assert "_version" in data
        assert data["_version"]["app"] == "momentbank"
        assert data["_version"]["app_version"] == "0.1.0"
        assert data["_version"]["embed_model"] == "arcface-r100"
        assert data["_version"]["gate_version"] == "v1"
        assert "created_at" in data["_version"]


class TestLoadErrors:
    def test_load_nonexistent(self, tmp_path):
        """Graceful error handling for missing file."""
        with pytest.raises(FileNotFoundError):
            load_bank(tmp_path / "nonexistent.json")
