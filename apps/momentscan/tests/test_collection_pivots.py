"""Tests for collection/pivots.py."""

import pytest
from dataclasses import dataclass
from pathlib import Path

from momentscan.algorithm.collection.pivots import (
    DEFAULT_PIVOTS,
    DEFAULT_POSES,
    ExpressionPivot,
    PoseTarget,
    classify_expression,
    classify_pose,
    load_collection_pivots,
    load_pivots,
    load_poses,
    _evaluate_rule,
)


# ── Rule evaluation ──


class TestEvaluateRule:
    def test_gte(self):
        assert _evaluate_rule(1.0, ">= 1.0") is True
        assert _evaluate_rule(0.9, ">= 1.0") is False

    def test_lte(self):
        assert _evaluate_rule(1.0, "<= 1.0") is True
        assert _evaluate_rule(1.1, "<= 1.0") is False

    def test_gt(self):
        assert _evaluate_rule(1.1, "> 1.0") is True
        assert _evaluate_rule(1.0, "> 1.0") is False

    def test_lt(self):
        assert _evaluate_rule(0.9, "< 1.0") is True
        assert _evaluate_rule(1.0, "< 1.0") is False

    def test_eq(self):
        assert _evaluate_rule(1.0, "== 1.0") is True
        assert _evaluate_rule(1.1, "== 1.0") is False

    def test_invalid(self):
        assert _evaluate_rule(1.0, "invalid") is False


# ── Pose classification ──


class TestClassifyPose:
    def test_defaults_count(self):
        assert len(DEFAULT_POSES) == 4

    def test_frontal(self):
        assert classify_pose(0, 0, DEFAULT_POSES) == "frontal"
        assert classify_pose(5, 0, DEFAULT_POSES) == "frontal"
        assert classify_pose(-5, 0, DEFAULT_POSES) == "frontal"

    def test_three_quarter(self):
        assert classify_pose(30, 0, DEFAULT_POSES) == "three-quarter"
        assert classify_pose(-30, 0, DEFAULT_POSES) == "three-quarter"

    def test_side_profile(self):
        assert classify_pose(55, 0, DEFAULT_POSES) == "side-profile"
        assert classify_pose(-55, 0, DEFAULT_POSES) == "side-profile"

    def test_looking_up(self):
        assert classify_pose(10, 20, DEFAULT_POSES) == "looking-up"

    def test_symmetric_yaw(self):
        """Left and right yaw should map to same pose."""
        for yaw in [30, -30]:
            result = classify_pose(yaw, 0, DEFAULT_POSES)
            assert result == "three-quarter"

    def test_extreme_out_of_range(self):
        """Very extreme pose may exceed all r_accept."""
        result = classify_pose(90, 45, DEFAULT_POSES)
        assert result is None

    def test_empty_poses(self):
        assert classify_pose(0, 0, []) is None

    def test_custom_pose(self):
        custom = [PoseTarget("test", yaw=45, pitch=0, r_accept=10)]
        assert classify_pose(45, 0, custom) == "test"
        assert classify_pose(10, 0, custom) is None


# ── Expression classification ──


@dataclass
class MockRecord:
    """Minimal record for expression classification tests."""
    au12_lip_corner: float = 0.0
    au25_lips_part: float = 0.0
    au6_cheek_raiser: float = 0.0
    au26_jaw_drop: float = 0.0
    smile_intensity: float = 0.0
    clip_axes: dict = None

    def __post_init__(self):
        if self.clip_axes is None:
            self.clip_axes = {}


class TestClassifyExpression:
    def test_neutral(self):
        r = MockRecord()
        assert classify_expression(r, DEFAULT_PIVOTS) == "neutral"

    def test_smile(self):
        r = MockRecord(au12_lip_corner=1.5)
        assert classify_expression(r, DEFAULT_PIVOTS) == "smile"

    def test_excited(self):
        r = MockRecord(au12_lip_corner=2.0, au25_lips_part=2.0)
        assert classify_expression(r, DEFAULT_PIVOTS) == "excited"

    def test_surprised(self):
        r = MockRecord(au12_lip_corner=0.5, au25_lips_part=2.0)
        assert classify_expression(r, DEFAULT_PIVOTS) == "surprised"

    def test_priority_excited_over_smile(self):
        """Excited (priority=1) beats smile (priority=2) when both match."""
        r = MockRecord(au12_lip_corner=2.0, au25_lips_part=2.0)
        assert classify_expression(r, DEFAULT_PIVOTS) == "excited"

    def test_smile_fallback_via_smile_intensity(self):
        """smile_intensity >= 0.4 triggers smile even without AU values."""
        r = MockRecord(smile_intensity=0.5)
        assert classify_expression(r, DEFAULT_PIVOTS) == "smile"

    def test_excited_fallback_via_smile_intensity(self):
        """smile_intensity >= 0.7 triggers excited via fallback."""
        r = MockRecord(smile_intensity=0.8)
        assert classify_expression(r, DEFAULT_PIVOTS) == "excited"

    def test_au_smile_beats_intensity_fallback(self):
        """AU-based smile (priority=2) beats intensity fallback (priority=10)."""
        r = MockRecord(au12_lip_corner=1.5, smile_intensity=0.5)
        assert classify_expression(r, DEFAULT_PIVOTS) == "smile"

    def test_low_smile_intensity_stays_neutral(self):
        """smile_intensity below 0.4 → neutral."""
        r = MockRecord(smile_intensity=0.3)
        assert classify_expression(r, DEFAULT_PIVOTS) == "neutral"

    def test_custom_pivots(self):
        custom = [
            ExpressionPivot("warm", {"warm_smile": ">= 0.5"}, priority=1),
            ExpressionPivot("default", {}, priority=99),
        ]
        r = MockRecord(clip_axes={"warm_smile": 0.7})
        assert classify_expression(r, custom) == "warm"

    def test_fallback_when_no_match(self):
        custom = [
            ExpressionPivot("rare", {"au12_lip_corner": ">= 10.0"}, priority=1),
            ExpressionPivot("default", {}, priority=99),
        ]
        r = MockRecord()
        assert classify_expression(r, custom) == "default"


# ── YAML loading ──


class TestLoadPoses:
    def test_nonexistent_dir(self, tmp_path):
        poses = load_poses(tmp_path / "nonexistent")
        assert len(poses) == len(DEFAULT_POSES)

    def test_empty_dir(self, tmp_path):
        poses_dir = tmp_path / "poses"
        poses_dir.mkdir()
        poses = load_poses(poses_dir)
        assert len(poses) == len(DEFAULT_POSES)

    def test_load_yaml(self, tmp_path):
        poses_dir = tmp_path / "poses"
        poses_dir.mkdir()
        (poses_dir / "frontal.yaml").write_text(
            "name: frontal\nyaw: 0\npitch: 0\nr_accept: 15\n"
        )
        (poses_dir / "side.yaml").write_text(
            "name: side\nyaw: 60\npitch: 0\nr_accept: 20\n"
        )
        poses = load_poses(poses_dir)
        assert len(poses) == 2
        names = {p.name for p in poses}
        assert "frontal" in names
        assert "side" in names


class TestLoadPivots:
    def test_nonexistent_dir(self, tmp_path):
        pivots = load_pivots(tmp_path / "nonexistent")
        assert len(pivots) == len(DEFAULT_PIVOTS)

    def test_load_subdirectory_format(self, tmp_path):
        pivots_dir = tmp_path / "pivots"
        warm_dir = pivots_dir / "warm_smile"
        warm_dir.mkdir(parents=True)
        (warm_dir / "pivot.yaml").write_text(
            "name: warm_smile\nrules:\n  au12_lip_corner: '>= 1.0'\npriority: 1\n"
        )
        pivots = load_pivots(pivots_dir)
        assert len(pivots) == 1
        assert pivots[0].name == "warm_smile"
        assert pivots[0].rules == {"au12_lip_corner": ">= 1.0"}
        assert pivots[0].priority == 1

    def test_load_flat_format(self, tmp_path):
        pivots_dir = tmp_path / "pivots"
        pivots_dir.mkdir()
        (pivots_dir / "neutral.yaml").write_text(
            "name: neutral\nrules: {}\npriority: 99\n"
        )
        pivots = load_pivots(pivots_dir)
        assert len(pivots) == 1
        assert pivots[0].name == "neutral"


class TestLoadCollectionPivots:
    def test_none_dir(self):
        poses, pivots = load_collection_pivots(None)
        assert len(poses) == len(DEFAULT_POSES)
        assert len(pivots) == len(DEFAULT_PIVOTS)

    def test_with_dir(self, tmp_path):
        coll_dir = tmp_path / "portrait-v1"
        poses_dir = coll_dir / "poses"
        pivots_dir = coll_dir / "pivots"
        poses_dir.mkdir(parents=True)
        pivots_dir.mkdir(parents=True)

        (poses_dir / "frontal.yaml").write_text(
            "name: frontal\nyaw: 0\npitch: 0\nr_accept: 15\n"
        )
        warm_dir = pivots_dir / "warm"
        warm_dir.mkdir()
        (warm_dir / "pivot.yaml").write_text(
            "name: warm\nrules:\n  au12_lip_corner: '>= 1.0'\npriority: 1\n"
        )

        poses, pivots = load_collection_pivots(coll_dir)
        assert len(poses) == 1
        assert len(pivots) == 1
