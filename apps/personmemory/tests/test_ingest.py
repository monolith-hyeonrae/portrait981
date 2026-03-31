"""Tests for personmemory.ingest bridge."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pytest

from personmemory.ingest import (
    ingest_collection, save_selected_frames, lookup_frames,
    _yaw_to_bin, _expression_bin, IngestResult,
)
from personmemory.paths import get_bank_path, _shard, list_member_ids
from personmemory.persistence import load_bank


# --- Lightweight stubs (avoid importing momentscan) ---


@dataclass
class FakeCollectionRecord:
    frame_idx: int
    timestamp_ms: float = 0.0
    e_id: Optional[np.ndarray] = None
    head_yaw: float = 0.0
    smile_intensity: float = 0.0
    em_surprise: float = 0.0
    catalog_primary: str = ""


@dataclass
class FakeSelectedFrame:
    frame_idx: int
    timestamp_ms: float = 0.0
    cell_key: str = "frontal|neutral"
    quality_score: float = 0.7
    face_crop_box: Optional[tuple] = None
    image_size: Optional[tuple] = None
    pose_name: str = "frontal"
    pivot_name: str = "neutral"
    set_type: str = "grid"
    cell_score: float = 0.0
    catalog_sim: float = 0.0
    pose_fit: float = 0.0
    stable_score: float = 0.0


@dataclass
class FakePersonCollection:
    person_id: int
    grid: Dict[str, List[FakeSelectedFrame]] = field(default_factory=dict)
    catalog_mode: bool = False

    def all_frames(self) -> List[FakeSelectedFrame]:
        return [f for frames in self.grid.values() for f in frames]


@dataclass
class FakeCollectionResult:
    persons: Dict[int, FakePersonCollection] = field(default_factory=dict)
    frame_count: int = 0


def _make_embedding(seed: int = 42) -> np.ndarray:
    rng = np.random.RandomState(seed)
    v = rng.randn(512).astype(np.float32)
    return v / np.linalg.norm(v)


@pytest.fixture()
def patch_bank_base(tmp_path, monkeypatch):
    """Redirect bank base dir to tmp_path for test isolation."""
    monkeypatch.setattr(
        "personmemory.paths.get_home_dir",
        lambda: tmp_path,
    )
    # Also patch the import inside ingest.py
    monkeypatch.setattr(
        "personmemory.ingest.get_bank_path",
        lambda mid: tmp_path / "personmemory" / _shard(mid) / mid / "memory_bank.json",
    )
    return tmp_path


# --- Tests ---


class TestYawBin:
    def test_frontal(self):
        assert _yaw_to_bin(0.0) == "[-5,5]"

    def test_slight_left(self):
        assert _yaw_to_bin(-15.0) == "[-30,-5]"

    def test_slight_right(self):
        assert _yaw_to_bin(20.0) == "[5,30]"

    def test_extreme_left(self):
        assert _yaw_to_bin(-50.0) == "[-70,-30]"

    def test_extreme_right(self):
        assert _yaw_to_bin(60.0) == "[30,70]"

    def test_beyond_range(self):
        assert _yaw_to_bin(-90.0) == "[-70,-30]"
        assert _yaw_to_bin(90.0) == "[30,70]"


class TestExpressionBin:
    def test_catalog_smile(self):
        r = FakeCollectionRecord(frame_idx=0, catalog_primary="warm_smile")
        assert _expression_bin(r) == "smile"

    def test_catalog_surprise(self):
        r = FakeCollectionRecord(frame_idx=0, catalog_primary="Surprise_Look")
        assert _expression_bin(r) == "surprise"

    def test_catalog_neutral(self):
        r = FakeCollectionRecord(frame_idx=0, catalog_primary="calm_gaze")
        assert _expression_bin(r) == "neutral"

    def test_fallback_smile_intensity(self):
        r = FakeCollectionRecord(frame_idx=0, smile_intensity=0.8)
        assert _expression_bin(r) == "smile"

    def test_fallback_surprise(self):
        r = FakeCollectionRecord(frame_idx=0, em_surprise=0.7)
        assert _expression_bin(r) == "surprise"

    def test_fallback_neutral(self):
        r = FakeCollectionRecord(frame_idx=0)
        assert _expression_bin(r) == "neutral"


class TestIngestCollection:
    def test_basic_roundtrip(self, tmp_path: Path, patch_bank_base):
        """Ingest → save → load → verify nodes exist."""
        emb = _make_embedding(42)
        records = [
            FakeCollectionRecord(frame_idx=10, e_id=emb, head_yaw=0.0, smile_intensity=0.8),
            FakeCollectionRecord(frame_idx=20, e_id=emb, head_yaw=-15.0, smile_intensity=0.1),
        ]
        person = FakePersonCollection(
            person_id=0,
            grid={
                "frontal|smile": [FakeSelectedFrame(frame_idx=10, cell_key="frontal|smile")],
                "left15|neutral": [FakeSelectedFrame(frame_idx=20, cell_key="left15|neutral")],
            },
        )
        coll = FakeCollectionResult(persons={0: person})
        frame_paths = {10: "/tmp/vid_frontal_smile_10.jpg", 20: "/tmp/vid_left15_neutral_20.jpg"}

        ir = ingest_collection(coll, records, frame_paths, member_id="test_user")

        assert 0 in ir.banks
        bank = ir.banks[0]
        assert len(bank.nodes) > 0

        # Verify persistence via sharded path
        shard = _shard("test_user")
        bank_path = tmp_path / "personmemory" / shard / "test_user" / "memory_bank.json"
        assert bank_path.exists()

        loaded = load_bank(bank_path)
        assert loaded.person_id == 0
        assert len(loaded.nodes) == len(bank.nodes)

    def test_yaw_bins_in_bank(self, tmp_path: Path, patch_bank_base):
        """Verify yaw bin metadata is correctly stored."""
        emb = _make_embedding(42)
        records = [
            FakeCollectionRecord(frame_idx=10, e_id=emb, head_yaw=0.0),
        ]
        person = FakePersonCollection(
            person_id=0,
            grid={"frontal|neutral": [FakeSelectedFrame(frame_idx=10)]},
        )
        coll = FakeCollectionResult(persons={0: person})

        ir = ingest_collection(coll, records, {10: "/tmp/f.jpg"}, member_id="yaw_test")

        node = ir.banks[0].nodes[0]
        assert "[-5,5]" in node.meta_hist.yaw_bins

    def test_rep_images_populated(self, tmp_path: Path, patch_bank_base):
        """Verify representative images contain frame paths."""
        emb = _make_embedding(42)
        records = [
            FakeCollectionRecord(frame_idx=10, e_id=emb),
        ]
        person = FakePersonCollection(
            person_id=0,
            grid={"frontal|neutral": [
                FakeSelectedFrame(frame_idx=10, cell_key="frontal|neutral"),
            ]},
        )
        coll = FakeCollectionResult(persons={0: person})
        frame_paths = {10: "/bank/frames/vid_frontal_neutral_10.jpg"}

        ir = ingest_collection(coll, records, frame_paths, member_id="rep_test")

        node = ir.banks[0].nodes[0]
        assert len(node.rep_images) == 1
        assert node.rep_images[0] == "/bank/frames/vid_frontal_neutral_10.jpg"

    def test_skip_frame_without_embedding(self, tmp_path: Path, patch_bank_base):
        """Frames without e_id should be skipped."""
        records = [
            FakeCollectionRecord(frame_idx=10, e_id=None),
        ]
        person = FakePersonCollection(
            person_id=0,
            grid={"frontal|neutral": [FakeSelectedFrame(frame_idx=10)]},
        )
        coll = FakeCollectionResult(persons={0: person})

        ir = ingest_collection(coll, records, {10: "/tmp/f.jpg"}, member_id="skip_test")

        assert ir.banks == {}

    def test_empty_collection(self, tmp_path: Path):
        """Empty collection → empty dict."""
        coll = FakeCollectionResult(persons={})
        ir = ingest_collection(coll, [], {}, member_id="empty")
        assert ir.banks == {}
        assert ir.total_persons == 0

    def test_none_collection(self, tmp_path: Path):
        """None collection → empty IngestResult."""
        ir = ingest_collection(None, [], {}, member_id="none")
        assert ir.banks == {}
        assert ir.total_persons == 0

    def test_only_main_person(self, tmp_path: Path, patch_bank_base):
        """Only person_id=0 (main) is ingested; other persons are ignored."""
        emb0 = _make_embedding(42)
        emb1 = _make_embedding(99)
        records = [
            FakeCollectionRecord(frame_idx=10, e_id=emb0),
            FakeCollectionRecord(frame_idx=20, e_id=emb1),
        ]
        p0 = FakePersonCollection(
            person_id=0,
            grid={"frontal|neutral": [FakeSelectedFrame(frame_idx=10)]},
        )
        p1 = FakePersonCollection(
            person_id=1,
            grid={"frontal|neutral": [FakeSelectedFrame(frame_idx=20)]},
        )
        coll = FakeCollectionResult(persons={0: p0, 1: p1})

        ir = ingest_collection(coll, records, {10: "/tmp/f.jpg"}, member_id="main_only")

        assert 0 in ir.banks
        assert 1 not in ir.banks
        assert ir.total_persons == 1

    def test_missing_record_for_frame(self, tmp_path: Path, patch_bank_base):
        """Frame index not in records → skip."""
        emb = _make_embedding(42)
        records = [
            FakeCollectionRecord(frame_idx=999, e_id=emb),  # different frame_idx
        ]
        person = FakePersonCollection(
            person_id=0,
            grid={"frontal|neutral": [FakeSelectedFrame(frame_idx=10)]},  # no matching record
        )
        coll = FakeCollectionResult(persons={0: person})

        ir = ingest_collection(coll, records, {}, member_id="missing")

        assert ir.banks == {}

    def test_stats_content(self, tmp_path: Path, patch_bank_base):
        """Verify IngestStats fields including member_id."""
        emb = _make_embedding(42)
        records = [
            FakeCollectionRecord(frame_idx=10, e_id=emb, head_yaw=0.0),
            FakeCollectionRecord(frame_idx=20, e_id=None),  # will be skipped
        ]
        person = FakePersonCollection(
            person_id=0,
            grid={
                "frontal|neutral": [FakeSelectedFrame(frame_idx=10)],
                "left|neutral": [FakeSelectedFrame(frame_idx=20)],
            },
        )
        coll = FakeCollectionResult(persons={0: person})

        ir = ingest_collection(coll, records, {10: "/tmp/f.jpg"}, member_id="stats_user")

        assert len(ir.stats) == 1
        s = ir.stats[0]
        assert s.person_id == 0
        assert s.member_id == "stats_user"
        assert s.frames_total == 2
        assert s.frames_ingested == 1
        assert s.frames_skipped == 1
        assert "[-5,5]" in s.yaw_bins
        assert "memory_bank.json" in s.bank_path

    def test_summary_message(self, tmp_path: Path, patch_bank_base):
        """Verify human-readable summary."""
        emb = _make_embedding(42)
        records = [FakeCollectionRecord(frame_idx=10, e_id=emb)]
        person = FakePersonCollection(
            person_id=0,
            grid={"frontal|neutral": [FakeSelectedFrame(frame_idx=10)]},
        )
        coll = FakeCollectionResult(persons={0: person})

        ir = ingest_collection(coll, records, {10: "/tmp/f.jpg"}, member_id="summary_user")

        summary = ir.summary()
        assert "person_0" in summary
        assert "1/1 frames" in summary
        assert "Memory bank saved" in summary

    def test_empty_summary(self):
        """Empty IngestResult summary."""
        ir = IngestResult()
        assert ir.summary() == "No persons ingested"

    def test_cumulative_update(self, tmp_path: Path, patch_bank_base):
        """Two ingests with same member_id → bank accumulates nodes."""
        emb1 = _make_embedding(42)
        emb2 = _make_embedding(99)  # different embedding → new node

        # First ingest
        records1 = [FakeCollectionRecord(frame_idx=10, e_id=emb1, head_yaw=0.0)]
        person1 = FakePersonCollection(
            person_id=0,
            grid={"frontal|neutral": [FakeSelectedFrame(frame_idx=10)]},
        )
        coll1 = FakeCollectionResult(persons={0: person1})
        ir1 = ingest_collection(coll1, records1, {10: "/tmp/f10.jpg"}, member_id="cumul_user")
        nodes_after_first = len(ir1.banks[0].nodes)

        # Second ingest (same member_id, different embedding)
        records2 = [FakeCollectionRecord(frame_idx=50, e_id=emb2, head_yaw=20.0)]
        person2 = FakePersonCollection(
            person_id=0,
            grid={"right|neutral": [FakeSelectedFrame(frame_idx=50, cell_key="right|neutral")]},
        )
        coll2 = FakeCollectionResult(persons={0: person2})
        ir2 = ingest_collection(coll2, records2, {50: "/tmp/f50.jpg"}, member_id="cumul_user")

        nodes_after_second = len(ir2.banks[0].nodes)
        assert nodes_after_second >= nodes_after_first + 1

        # Verify persisted bank has all nodes
        shard = _shard("cumul_user")
        bank_path = tmp_path / "personmemory" / shard / "cumul_user" / "memory_bank.json"
        loaded = load_bank(bank_path)
        assert len(loaded.nodes) == nodes_after_second

    def test_cumulative_hit_count(self, tmp_path: Path, patch_bank_base):
        """Same embedding twice → hit_count increases on existing node."""
        emb = _make_embedding(42)

        # First ingest
        records1 = [FakeCollectionRecord(frame_idx=10, e_id=emb)]
        person1 = FakePersonCollection(
            person_id=0,
            grid={"frontal|neutral": [FakeSelectedFrame(frame_idx=10)]},
        )
        coll1 = FakeCollectionResult(persons={0: person1})
        ir1 = ingest_collection(coll1, records1, {10: "/tmp/f10.jpg"}, member_id="hit_user")
        hit1 = ir1.banks[0].nodes[0].meta_hist.hit_count

        # Second ingest (same embedding → same node updated)
        records2 = [FakeCollectionRecord(frame_idx=20, e_id=emb)]
        person2 = FakePersonCollection(
            person_id=0,
            grid={"frontal|neutral": [FakeSelectedFrame(frame_idx=20)]},
        )
        coll2 = FakeCollectionResult(persons={0: person2})
        ir2 = ingest_collection(coll2, records2, {20: "/tmp/f20.jpg"}, member_id="hit_user")
        hit2 = ir2.banks[0].nodes[0].meta_hist.hit_count

        assert hit2 > hit1


class TestShardPath:
    def test_shard_deterministic(self):
        """Same member_id always produces same shard."""
        assert _shard("park_001") == _shard("park_001")

    def test_shard_length(self):
        """Shard is exactly 2 hex chars."""
        s = _shard("test")
        assert len(s) == 2
        assert all(c in "0123456789abcdef" for c in s)

    def test_shard_matches_sha1(self):
        """Verify shard matches SHA1 prefix."""
        mid = "park_001"
        expected = hashlib.sha1(mid.encode()).hexdigest()[:2]
        assert _shard(mid) == expected

    def test_get_bank_path_contains_shard(self):
        """get_bank_path includes shard directory."""
        path = get_bank_path("kim_042")
        shard = _shard("kim_042")
        assert shard in path.parts
        assert "kim_042" in path.parts
        assert path.name == "memory_bank.json"


class TestListMemberIds:
    def test_empty_base(self, tmp_path, monkeypatch):
        """No base dir → empty list."""
        monkeypatch.setattr("personmemory.paths.get_bank_base_dir", lambda: tmp_path / "nonexistent")
        assert list_member_ids() == []

    def test_lists_members(self, tmp_path, monkeypatch):
        """Create several member banks and verify listing."""
        base = tmp_path / "personmemory"
        monkeypatch.setattr("personmemory.paths.get_bank_base_dir", lambda: base)

        # Create fake banks
        for mid in ["alpha", "beta", "gamma"]:
            shard = _shard(mid)
            d = base / shard / mid
            d.mkdir(parents=True)
            (d / "memory_bank.json").write_text("{}")

        ids = list_member_ids()
        assert sorted(ids) == ["alpha", "beta", "gamma"]

    def test_ignores_dirs_without_bank(self, tmp_path, monkeypatch):
        """Directories without memory_bank.json are not listed."""
        base = tmp_path / "personmemory"
        monkeypatch.setattr("personmemory.paths.get_bank_base_dir", lambda: base)

        shard = _shard("has_bank")
        d1 = base / shard / "has_bank"
        d1.mkdir(parents=True)
        (d1 / "memory_bank.json").write_text("{}")

        shard2 = _shard("no_bank")
        d2 = base / shard2 / "no_bank"
        d2.mkdir(parents=True)
        # No memory_bank.json here

        ids = list_member_ids()
        assert "has_bank" in ids
        assert "no_bank" not in ids


class TestSaveSelectedFrames:
    def test_empty_result(self):
        """Empty collection → empty dict."""
        result = save_selected_frames("/tmp/vid.mp4", None, [], "member1")
        assert result == {}

    def test_no_person_zero(self):
        """No person_id=0 → empty dict."""
        p1 = FakePersonCollection(person_id=1, grid={"frontal|neutral": [FakeSelectedFrame(frame_idx=10)]})
        coll = FakeCollectionResult(persons={1: p1})
        result = save_selected_frames("/tmp/vid.mp4", coll, [], "member1")
        assert result == {}

    def test_missing_cv2_returns_empty(self, monkeypatch):
        """If cv2/visualbase not importable → graceful empty dict."""
        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "cv2":
                raise ImportError("no cv2")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)

        person = FakePersonCollection(
            person_id=0,
            grid={"frontal|neutral": [FakeSelectedFrame(frame_idx=10)]},
        )
        coll = FakeCollectionResult(persons={0: person})
        result = save_selected_frames("/tmp/vid.mp4", coll, [], "member1")
        assert result == {}

    def test_ingest_with_missing_frame_path(self, tmp_path: Path, patch_bank_base):
        """Frame without a saved path gets image_path='' in bank."""
        emb = _make_embedding(42)
        records = [FakeCollectionRecord(frame_idx=10, e_id=emb)]
        person = FakePersonCollection(
            person_id=0,
            grid={"frontal|neutral": [FakeSelectedFrame(frame_idx=10)]},
        )
        coll = FakeCollectionResult(persons={0: person})

        # No frame_paths for frame 10
        ir = ingest_collection(coll, records, {}, member_id="no_path_test")

        assert 0 in ir.banks
        node = ir.banks[0].nodes[0]
        assert len(node.rep_images) == 1
        assert node.rep_images[0] == ""


class TestLookupFrames:
    def _write_manifest(self, tmp_path, member_id, entries):
        """Helper: write frames.json manifest for a member."""
        from personmemory.paths import get_bank_dir
        frames_dir = get_bank_dir(member_id) / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        import json
        with open(frames_dir / "frames.json", "w") as f:
            json.dump(entries, f)
        return frames_dir

    def test_no_manifest(self, tmp_path, patch_bank_base):
        """No frames.json → empty list."""
        assert lookup_frames("nonexistent") == []

    def test_filter_by_pose(self, tmp_path, patch_bank_base):
        """Filter by pose_name."""
        entries = [
            {"file": "a.jpg", "pose_name": "frontal", "category": "smile", "cell_score": 0.8},
            {"file": "b.jpg", "pose_name": "left30", "category": "smile", "cell_score": 0.7},
            {"file": "c.jpg", "pose_name": "left30", "category": "neutral", "cell_score": 0.6},
        ]
        self._write_manifest(tmp_path, "lookup_test", entries)

        results = lookup_frames("lookup_test", pose="left30")
        assert len(results) == 2
        assert all(r["pose_name"] == "left30" for r in results)
        # Sorted by cell_score desc
        assert results[0]["cell_score"] >= results[1]["cell_score"]

    def test_filter_by_category(self, tmp_path, patch_bank_base):
        """Filter by category."""
        entries = [
            {"file": "a.jpg", "pose_name": "frontal", "category": "smile", "cell_score": 0.9},
            {"file": "b.jpg", "pose_name": "left30", "category": "neutral", "cell_score": 0.5},
        ]
        self._write_manifest(tmp_path, "cat_test", entries)

        results = lookup_frames("cat_test", category="smile")
        assert len(results) == 1
        assert results[0]["category"] == "smile"
        assert "path" in results[0]

    def test_top_k(self, tmp_path, patch_bank_base):
        """top_k limits results."""
        entries = [
            {"file": f"{i}.jpg", "pose_name": "frontal", "category": "smile", "cell_score": i * 0.1}
            for i in range(5)
        ]
        self._write_manifest(tmp_path, "topk_test", entries)

        results = lookup_frames("topk_test", top_k=2)
        assert len(results) == 2
        assert results[0]["cell_score"] > results[1]["cell_score"]

    def test_combined_filter(self, tmp_path, patch_bank_base):
        """Filter by both pose and category."""
        entries = [
            {"file": "a.jpg", "pose_name": "left30", "category": "smile", "cell_score": 0.9},
            {"file": "b.jpg", "pose_name": "left30", "category": "neutral", "cell_score": 0.5},
            {"file": "c.jpg", "pose_name": "frontal", "category": "smile", "cell_score": 0.8},
        ]
        self._write_manifest(tmp_path, "combo_test", entries)

        results = lookup_frames("combo_test", pose="left30", category="smile")
        assert len(results) == 1
        assert results[0]["file"] == "a.jpg"
