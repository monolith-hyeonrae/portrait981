"""Tests for Phase 3: Identity Builder."""

import json
import pytest
import numpy as np
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from pathlib import Path

from momentscan.algorithm.identity.types import (
    BucketLabel,
    IdentityConfig,
    IdentityFrame,
    IdentityRecord,
    IdentityResult,
    PersonIdentity,
)
from momentscan.algorithm.identity.buckets import (
    classify_yaw,
    classify_pitch,
    classify_expression,
    classify_frame,
)
from momentscan.algorithm.identity.builder import IdentityBuilder
from momentscan.algorithm.identity.extract import extract_identity_record
from momentscan.algorithm.identity.export import export_identity_metadata


# ── Helpers ──

def _make_embedding(dim: int = 512, seed: int = 0) -> np.ndarray:
    """L2-normalized random embedding."""
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    return v / np.linalg.norm(v)


def _make_record(
    frame_idx: int = 0,
    timestamp_ms: float = 0.0,
    e_id_seed: Optional[int] = None,
    e_head_seed: Optional[int] = None,
    head_yaw: float = 0.0,
    head_pitch: float = 0.0,
    smile_intensity: float = 0.0,
    mouth_open_ratio: float = 0.0,
    eye_open_ratio: float = 0.5,
    face_confidence: float = 0.9,
    face_area_ratio: float = 0.05,
    blur_score: float = 100.0,
    brightness: float = 128.0,
    person_id: int = 0,
) -> IdentityRecord:
    """테스트용 IdentityRecord 생성."""
    return IdentityRecord(
        frame_idx=frame_idx,
        timestamp_ms=timestamp_ms,
        e_id=_make_embedding(512, e_id_seed) if e_id_seed is not None else None,
        head_yaw=head_yaw,
        head_pitch=head_pitch,
        smile_intensity=smile_intensity,
        mouth_open_ratio=mouth_open_ratio,
        eye_open_ratio=eye_open_ratio,
        face_confidence=face_confidence,
        face_area_ratio=face_area_ratio,
        blur_score=blur_score,
        brightness=brightness,
        person_id=person_id,
    )


# ── Bucket classification tests ──

class TestBucketClassification:
    def test_yaw_frontal(self):
        assert classify_yaw(0.0) == "[-10,10]"
        assert classify_yaw(5.0) == "[-10,10]"
        assert classify_yaw(-5.0) == "[-10,10]"

    def test_yaw_edges(self):
        # Edges fall into the bin where value <= upper bound
        assert classify_yaw(-10.0) == "[-30,-10]"  # -10 <= -10 upper edge
        assert classify_yaw(-9.9) == "[-10,10]"
        assert classify_yaw(10.0) == "[-10,10]"    # 10 <= 10 upper edge
        assert classify_yaw(10.1) == "[10,30]"
        assert classify_yaw(-60.0) == "[-90,-60]"
        assert classify_yaw(-59.9) == "[-60,-30]"
        assert classify_yaw(60.0) == "[30,60]"
        assert classify_yaw(60.1) == "[60,90]"
        assert classify_yaw(-90.0) == "[-90,-60]"
        assert classify_yaw(90.0) == "[60,90]"

    def test_yaw_all_bins(self):
        # 7 bins total
        bins_seen = set()
        for yaw in [-75, -45, -20, 0, 20, 45, 75]:
            bins_seen.add(classify_yaw(yaw))
        assert len(bins_seen) == 7

    def test_yaw_clamp(self):
        assert classify_yaw(-120.0) == "[-90,-60]"
        assert classify_yaw(120.0) == "[60,90]"

    def test_pitch_neutral(self):
        assert classify_pitch(0.0) == "neutral"

    def test_pitch_edges(self):
        assert classify_pitch(-30.0) == "down"
        assert classify_pitch(-10.0) == "down"       # -10 <= -10 upper edge
        assert classify_pitch(-9.9) == "neutral"
        assert classify_pitch(10.0) == "neutral"     # 10 <= 10 upper edge
        assert classify_pitch(10.1) == "up"

    def test_pitch_all_bins(self):
        bins_seen = set()
        for pitch in [-20, 0, 20]:
            bins_seen.add(classify_pitch(pitch))
        assert len(bins_seen) == 3

    def test_expression_neutral(self):
        assert classify_expression(0.0, 0.0, 0.5) == "neutral"

    def test_expression_smile(self):
        assert classify_expression(0.6, 0.0, 0.5) == "smile"

    def test_expression_mouth_open(self):
        assert classify_expression(0.0, 0.6, 0.5) == "mouth_open"

    def test_expression_eyes_closed(self):
        assert classify_expression(0.0, 0.0, 0.1) == "eyes_closed"

    def test_expression_priority_eyes_over_smile(self):
        # eyes_closed has highest priority
        assert classify_expression(0.6, 0.6, 0.1) == "eyes_closed"

    def test_expression_priority_smile_over_mouth(self):
        assert classify_expression(0.6, 0.6, 0.5) == "smile"

    def test_classify_frame_composite(self):
        label = classify_frame(0.0, 0.0, 0.0, 0.0, 0.5)
        assert label.yaw_bin == "[-10,10]"
        assert label.pitch_bin == "neutral"
        assert label.expression_bin == "neutral"

    def test_bucket_label_key(self):
        label = BucketLabel(yaw_bin="[-10,10]", pitch_bin="neutral", expression_bin="smile")
        assert label.key == "[-10,10]|neutral|smile"


# ── Extract tests ──

@dataclass
class MockFace:
    confidence: float = 0.9
    area_ratio: float = 0.05
    bbox: tuple = (0.1, 0.1, 0.3, 0.3)
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0
    expression: float = 0.0
    signals: Dict[str, float] = field(default_factory=dict)
    embedding: Optional[np.ndarray] = None


@dataclass
class MockOutput:
    faces: List[Any] = field(default_factory=list)


@dataclass
class MockShotQualityOutput:
    """Mimics ShotQualityOutput (crop boxes only — no embeddings)."""
    head_crop_box: Optional[tuple] = None
    image_size: Optional[tuple] = None
    head_blur: float = 0.0


@dataclass
class MockObs:
    source: str = ""
    data: Any = None
    signals: Dict[str, float] = field(default_factory=dict)


@dataclass
class MockFlowData:
    observations: List[Any] = field(default_factory=list)


@dataclass
class MockFrame:
    frame_id: int = 0
    t_src_ns: int = 0


class TestExtractIdentityRecord:
    def test_no_results(self):
        frame = MockFrame()
        assert extract_identity_record(frame, []) is None

    def test_no_face_detect(self):
        frame = MockFrame()
        obs = MockObs(source="body.pose")
        flow = MockFlowData(observations=[obs])
        assert extract_identity_record(frame, [flow]) is None

    def test_no_embedding(self):
        face = MockFace(embedding=None)
        output = MockOutput(faces=[face])
        obs = MockObs(source="face.detect", data=output)
        flow = MockFlowData(observations=[obs])
        frame = MockFrame()
        assert extract_identity_record(frame, [flow]) is None

    def test_basic_extraction(self):
        emb = _make_embedding(512, seed=1)
        face = MockFace(embedding=emb, yaw=10.0, pitch=-5.0)
        output = MockOutput(faces=[face])
        obs = MockObs(source="face.detect", data=output)
        flow = MockFlowData(observations=[obs])
        frame = MockFrame(frame_id=42, t_src_ns=1_000_000_000)

        record = extract_identity_record(frame, [flow])
        assert record is not None
        assert record.frame_idx == 42
        assert record.timestamp_ms == 1000.0
        assert record.head_yaw == 10.0
        assert record.head_pitch == -5.0
        # e_id should be L2-normalized
        assert abs(np.linalg.norm(record.e_id) - 1.0) < 1e-5

    def test_shot_quality_crop_extraction(self):
        """shot.quality provides crop box coordinates for identity crops."""
        emb_id = _make_embedding(512, seed=1)
        face = MockFace(embedding=emb_id)
        face_output = MockOutput(faces=[face])
        face_obs = MockObs(source="face.detect", data=face_output)

        shot_data = MockShotQualityOutput(
            head_crop_box=(10, 20, 100, 120),
            image_size=(640, 480),
        )
        shot_obs = MockObs(source="shot.quality", data=shot_data)

        flow = MockFlowData(observations=[face_obs, shot_obs])
        frame = MockFrame()

        record = extract_identity_record(frame, [flow])
        assert record is not None
        assert record.head_crop_box == (10, 20, 100, 120)
        assert record.image_size == (640, 480)

    def test_expression_extraction(self):
        emb_id = _make_embedding(512, seed=1)
        face_detect = MockFace(embedding=emb_id)
        detect_output = MockOutput(faces=[face_detect])
        detect_obs = MockObs(source="face.detect", data=detect_output)

        expr_face = MockFace(
            expression=0.7,
            signals={"em_happy": 0.8, "em_neutral": 0.1},
        )
        expr_output = MockOutput(faces=[expr_face])
        expr_obs = MockObs(source="face.expression", data=expr_output)

        flow = MockFlowData(observations=[detect_obs, expr_obs])
        frame = MockFrame()

        record = extract_identity_record(frame, [flow])
        assert record is not None
        assert record.smile_intensity == 0.8
        assert record.mouth_open_ratio == 0.7
        assert abs(record.eye_open_ratio - 0.9) < 1e-5  # 1 - 0.1

    def test_picks_largest_face(self):
        emb1 = _make_embedding(512, seed=1)
        emb2 = _make_embedding(512, seed=2)
        face1 = MockFace(embedding=emb1, area_ratio=0.03)
        face2 = MockFace(embedding=emb2, area_ratio=0.08)
        output = MockOutput(faces=[face1, face2])
        obs = MockObs(source="face.detect", data=output)
        flow = MockFlowData(observations=[obs])
        frame = MockFrame()

        record = extract_identity_record(frame, [flow])
        assert record is not None
        assert record.face_area_ratio == 0.08


# ── Medoid tests ──

class TestMedoid:
    def test_cluster_with_outlier(self):
        """4 similar + 1 outlier → medoid should be in cluster."""
        builder = IdentityBuilder()

        # Cluster: embeddings near each other
        base = _make_embedding(512, seed=100)
        cluster_records = []
        for i in range(4):
            r = _make_record(frame_idx=i, e_id_seed=100)
            # Add small perturbation
            r.e_id = base + np.random.default_rng(i).standard_normal(512).astype(np.float32) * 0.01
            r.e_id = r.e_id / np.linalg.norm(r.e_id)
            cluster_records.append(r)

        # Outlier: very different
        outlier = _make_record(frame_idx=99, e_id_seed=999)
        all_records = cluster_records + [outlier]

        prototype, proto_idx = builder._compute_medoid(all_records)
        assert proto_idx != 99  # Should not pick outlier

    def test_subsampling(self):
        """Large set → subsample via medoid_max_candidates."""
        cfg = IdentityConfig(medoid_max_candidates=10)
        builder = IdentityBuilder(config=cfg)

        records = [_make_record(frame_idx=i, e_id_seed=i) for i in range(50)]
        prototype, proto_idx = builder._compute_medoid(records)
        assert prototype.shape == (512,)
        assert abs(np.linalg.norm(prototype) - 1.0) < 1e-5


# ── Builder tests ──

class TestIdentityBuilder:
    def test_empty_records(self):
        builder = IdentityBuilder()
        result = builder.build([])
        assert result.frame_count == 0
        assert len(result.persons) == 0

    def test_too_few_strict_gate_frames(self):
        """모든 프레임이 strict gate 미달 → PersonIdentity 없음."""
        builder = IdentityBuilder()
        records = [
            _make_record(
                frame_idx=i,
                e_id_seed=i,
                face_confidence=0.3,  # Below gate_face_confidence=0.7
            )
            for i in range(10)
        ]
        result = builder.build(records)
        assert len(result.persons) == 0

    def test_basic_identity_build(self):
        """100개 합성 레코드 → anchor/coverage/challenge 생성."""
        # Use same base embedding for all (same person)
        base_emb = _make_embedding(512, seed=42)

        records = []
        for i in range(100):
            r = _make_record(
                frame_idx=i,
                timestamp_ms=i * 100.0,
                e_id_seed=42,

                face_confidence=0.9,
                blur_score=100.0,
                head_yaw=(i % 20 - 10) * 3.0,  # -30 to 30
                head_pitch=(i % 10 - 5) * 3.0,  # -15 to 15
                smile_intensity=0.5 if i % 5 == 0 else 0.1,
                eye_open_ratio=0.5,
            )
            # Make embeddings similar to base (same person)
            r.e_id = base_emb + np.random.default_rng(i).standard_normal(512).astype(np.float32) * 0.05
            r.e_id = r.e_id / np.linalg.norm(r.e_id)
            records.append(r)

        builder = IdentityBuilder()
        result = builder.build(records)

        assert result.frame_count == 100
        assert 0 in result.persons

        person = result.persons[0]
        assert len(person.anchor_frames) > 0
        assert len(person.anchor_frames) <= 5  # anchor_count default
        assert len(person.coverage_frames) > 0
        assert person.prototype_frame_idx >= 0

        # All anchors should be frontal
        for f in person.anchor_frames:
            assert f.set_type == "anchor"

        # Coverage should have diverse buckets
        coverage_buckets = {f.bucket.key for f in person.coverage_frames}
        assert len(coverage_buckets) > 1

    def test_challenge_stable_threshold(self):
        """challenge_min_stable 미만이면 challenge에 포함 안 됨."""
        base_emb = _make_embedding(512, seed=42)

        records = []
        # 10 good records (same person)
        for i in range(10):
            r = _make_record(
                frame_idx=i, e_id_seed=42,
                face_confidence=0.9, blur_score=100.0,
            )
            r.e_id = base_emb + np.random.default_rng(i).standard_normal(512).astype(np.float32) * 0.02
            r.e_id = r.e_id / np.linalg.norm(r.e_id)
            records.append(r)

        # 5 unstable records (very different embedding)
        for i in range(10, 15):
            r = _make_record(
                frame_idx=i, e_id_seed=i * 100, e_head_seed=i,
                face_confidence=0.9, blur_score=100.0,
            )
            records.append(r)

        builder = IdentityBuilder()
        result = builder.build(records)

        if 0 in result.persons:
            person = result.persons[0]
            for f in person.challenge_frames:
                assert f.stable_score >= builder.config.challenge_min_stable

    def test_anchor_frontal_only(self):
        """앵커는 anchor_max_yaw 이내만."""
        base_emb = _make_embedding(512, seed=42)
        cfg = IdentityConfig(anchor_max_yaw=10.0)

        records = []
        for i in range(50):
            yaw = (i - 25) * 2.0  # -50 to 48
            r = _make_record(
                frame_idx=i, e_id_seed=42, face_confidence=0.9,
                blur_score=100.0, head_yaw=yaw,
            )
            r.e_id = base_emb + np.random.default_rng(i).standard_normal(512).astype(np.float32) * 0.02
            r.e_id = r.e_id / np.linalg.norm(r.e_id)
            records.append(r)

        builder = IdentityBuilder(config=cfg)
        result = builder.build(records)

        if 0 in result.persons:
            for f in result.persons[0].anchor_frames:
                # Find original record
                orig = [r for r in records if r.frame_idx == f.frame_idx][0]
                assert abs(orig.head_yaw) <= cfg.anchor_max_yaw

    def test_coverage_diverse_buckets(self):
        """coverage는 다양한 bucket에서 선택."""
        base_emb = _make_embedding(512, seed=42)
        cfg = IdentityConfig(coverage_max_per_bucket=1)

        records = []
        yaws = [-30, -15, 0, 15, 30]
        for i, yaw in enumerate(yaws):
            for j in range(5):
                idx = i * 5 + j
                r = _make_record(
                    frame_idx=idx, e_id_seed=42, e_head_seed=idx,
                    face_confidence=0.9, blur_score=100.0,
                    head_yaw=float(yaw),
                )
                r.e_id = base_emb + np.random.default_rng(idx).standard_normal(512).astype(np.float32) * 0.02
                r.e_id = r.e_id / np.linalg.norm(r.e_id)
                records.append(r)

        builder = IdentityBuilder(config=cfg)
        result = builder.build(records)

        if 0 in result.persons:
            person = result.persons[0]
            yaw_bins = {f.bucket.yaw_bin for f in person.coverage_frames}
            # Should have at least 2 different yaw bins
            assert len(yaw_bins) >= 2


# ── Anchor dedup tests ──

class TestAnchorDedup:
    def test_consecutive_anchors_filtered(self):
        """연속 프레임은 anchor에서도 시간 간격으로 건너뛴다."""
        base_emb = _make_embedding(512, seed=42)
        cfg = IdentityConfig(
            anchor_count=5,
            anchor_min_interval_ms=2000.0,
            anchor_max_similarity=0.99,
        )

        # 20 frontal frames, 100ms apart → all same bucket
        records = []
        for i in range(20):
            r = _make_record(
                frame_idx=i, timestamp_ms=i * 100.0,
                e_id_seed=42, face_confidence=0.9, blur_score=100.0,
                head_yaw=0.0,
            )
            r.e_id = base_emb + np.random.default_rng(i).standard_normal(512).astype(np.float32) * 0.02
            r.e_id = r.e_id / np.linalg.norm(r.e_id)
            records.append(r)

        builder = IdentityBuilder(config=cfg)
        result = builder.build(records)

        if 0 in result.persons:
            person = result.persons[0]
            # 20 frames span 1900ms < 2000ms gap, so at most 1 anchor
            assert len(person.anchor_frames) <= 1

    def test_spaced_anchors_selected(self):
        """충분히 떨어진 프레임은 anchor 복수 선택."""
        base_emb = _make_embedding(512, seed=42)
        cfg = IdentityConfig(
            anchor_count=5,
            anchor_min_interval_ms=1000.0,
            anchor_max_similarity=0.99,
        )

        # 20 frontal frames, 2초 간격
        records = []
        for i in range(20):
            r = _make_record(
                frame_idx=i, timestamp_ms=i * 2000.0,
                e_id_seed=42, face_confidence=0.9, blur_score=100.0,
                head_yaw=0.0,
            )
            r.e_id = base_emb + np.random.default_rng(i).standard_normal(512).astype(np.float32) * 0.02
            r.e_id = r.e_id / np.linalg.norm(r.e_id)
            records.append(r)

        builder = IdentityBuilder(config=cfg)
        result = builder.build(records)

        if 0 in result.persons:
            person = result.persons[0]
            assert len(person.anchor_frames) == 5

            # 선택된 anchor 간 시간 간격 확인
            timestamps = sorted(f.timestamp_ms for f in person.anchor_frames)
            for i in range(1, len(timestamps)):
                assert timestamps[i] - timestamps[i - 1] >= cfg.anchor_min_interval_ms


# ── Coverage dedup tests ──

class TestCoverageDedup:
    def test_temporal_gap_filters_consecutive_frames(self):
        """같은 버킷의 연속 프레임은 시간 간격 제한으로 건너뛴다."""
        base_emb = _make_embedding(512, seed=42)
        cfg = IdentityConfig(
            coverage_max_per_bucket=2,
            coverage_min_interval_ms=2000.0,
        )

        # 10 frames, 100ms apart, same yaw/pitch → same bucket
        records = []
        for i in range(10):
            r = _make_record(
                frame_idx=i, timestamp_ms=i * 100.0,
                e_id_seed=42, face_confidence=0.9, blur_score=100.0,
                head_yaw=0.0, head_pitch=0.0,
            )
            r.e_id = base_emb + np.random.default_rng(i).standard_normal(512).astype(np.float32) * 0.02
            r.e_id = r.e_id / np.linalg.norm(r.e_id)
            records.append(r)

        builder = IdentityBuilder(config=cfg)
        result = builder.build(records)

        if 0 in result.persons:
            person = result.persons[0]
            # All 10 frames span only 900ms < 2000ms gap,
            # so coverage should get at most 1 from this bucket
            assert len(person.coverage_frames) <= 1

    def test_temporal_gap_allows_distant_frames(self):
        """시간 간격이 충분히 떨어진 프레임은 같은 버킷에서 복수 선택."""
        base_emb = _make_embedding(512, seed=42)
        cfg = IdentityConfig(
            anchor_count=2,  # anchor를 적게 잡아 coverage 후보 확보
            coverage_max_per_bucket=3,
            coverage_min_interval_ms=1000.0,
            coverage_max_similarity=0.99,  # 유사도 체크 사실상 비활성화
        )

        # 15 프레임, 2초 간격 — anchor 2개 뽑고도 13개 남음
        records = []
        for i in range(15):
            r = _make_record(
                frame_idx=i, timestamp_ms=i * 2000.0,  # 2초 간격
                e_id_seed=42, face_confidence=0.9, blur_score=100.0,
                head_yaw=0.0, head_pitch=0.0,
            )
            r.e_id = base_emb + np.random.default_rng(i).standard_normal(512).astype(np.float32) * 0.02
            r.e_id = r.e_id / np.linalg.norm(r.e_id)
            records.append(r)

        builder = IdentityBuilder(config=cfg)
        result = builder.build(records)

        if 0 in result.persons:
            person = result.persons[0]
            # 2초 간격 > min_interval_ms 1초 → 최대 3개 선택 가능
            assert len(person.coverage_frames) >= 2

    def test_visual_similarity_filters_duplicates(self):
        """시각적 유사도 필터 제거됨 → coverage는 max_per_bucket까지 선택된다."""
        base_emb = _make_embedding(512, seed=42)
        cfg = IdentityConfig(
            coverage_max_per_bucket=3,
            coverage_min_interval_ms=0.0,  # 시간 제한 비활성화
            coverage_max_similarity=0.8,
        )

        records = []
        for i in range(6):
            r = _make_record(
                frame_idx=i, timestamp_ms=i * 5000.0,  # 충분한 간격
                e_id_seed=42, face_confidence=0.9, blur_score=100.0,
                head_yaw=0.0, head_pitch=0.0,
            )
            r.e_id = base_emb + np.random.default_rng(i).standard_normal(512).astype(np.float32) * 0.02
            r.e_id = r.e_id / np.linalg.norm(r.e_id)
            records.append(r)

        builder = IdentityBuilder(config=cfg)
        result = builder.build(records)

        if 0 in result.persons:
            person = result.persons[0]
            # 시각적 유사도 필터 없음 → coverage는 max_per_bucket(3)까지 선택
            assert len(person.coverage_frames) <= 3


# ── Query (맥락 하이라이트) tests ──

class TestPersonIdentityQuery:
    def _make_person(self) -> PersonIdentity:
        """테스트용 PersonIdentity 생성."""
        anchors = [
            IdentityFrame(
                frame_idx=0, timestamp_ms=0.0, set_type="anchor",
                bucket=BucketLabel("[-10,10]", "neutral", "neutral"),
                quality_score=0.9, stable_score=0.8,
            ),
            IdentityFrame(
                frame_idx=1, timestamp_ms=100.0, set_type="anchor",
                bucket=BucketLabel("[-10,10]", "neutral", "smile"),
                quality_score=0.85, stable_score=0.75,
            ),
        ]
        coverage = [
            IdentityFrame(
                frame_idx=10, timestamp_ms=1000.0, set_type="coverage",
                bucket=BucketLabel("[10,30]", "neutral", "neutral"),
                quality_score=0.7, stable_score=0.6,
            ),
            IdentityFrame(
                frame_idx=11, timestamp_ms=1100.0, set_type="coverage",
                bucket=BucketLabel("[-60,-30]", "down", "smile"),
                quality_score=0.65, stable_score=0.55,
            ),
        ]
        challenge = [
            IdentityFrame(
                frame_idx=20, timestamp_ms=2000.0, set_type="challenge",
                bucket=BucketLabel("[60,90]", "up", "mouth_open"),
                quality_score=0.5, stable_score=0.45,
            ),
        ]
        return PersonIdentity(
            person_id=0,
            prototype_frame_idx=0,
            anchor_frames=anchors,
            coverage_frames=coverage,
            challenge_frames=challenge,
        )

    def test_query_smile(self):
        person = self._make_person()
        results = person.query(expression_bin="smile")
        assert len(results) == 2
        assert all(f.bucket.expression_bin == "smile" for f in results)

    def test_query_top_k_limit(self):
        person = self._make_person()
        results = person.query(expression_bin="smile", top_k=1)
        assert len(results) == 1
        assert results[0].quality_score == 0.85  # Highest quality smile

    def test_query_combined(self):
        person = self._make_person()
        results = person.query(yaw_bin="[-10,10]", expression_bin="neutral")
        assert len(results) == 1
        assert results[0].frame_idx == 0

    def test_query_no_match(self):
        person = self._make_person()
        results = person.query(expression_bin="eyes_closed")
        assert len(results) == 0

    def test_query_all(self):
        person = self._make_person()
        results = person.query(top_k=100)
        assert len(results) == 5  # all frames


# ── Export tests ──

class TestExportIdentityMetadata:
    def test_export_creates_meta_json(self, tmp_path):
        person = PersonIdentity(
            person_id=0,
            prototype_frame_idx=5,
            anchor_frames=[
                IdentityFrame(
                    frame_idx=5, timestamp_ms=500.0, set_type="anchor",
                    bucket=BucketLabel("[-10,10]", "neutral", "neutral"),
                    quality_score=0.9, stable_score=0.8,
                ),
            ],
            coverage_frames=[],
            challenge_frames=[],
        )
        result = IdentityResult(
            persons={0: person},
            frame_count=100,
            config=IdentityConfig(),
        )

        export_identity_metadata(result, tmp_path)

        meta_path = tmp_path / "identity" / "person_0" / "meta.json"
        assert meta_path.exists()

        with open(meta_path) as f:
            meta = json.load(f)

        assert meta["person_id"] == 0
        assert meta["prototype_frame_idx"] == 5
        assert meta["anchor_count"] == 1
        assert len(meta["anchors"]) == 1
        assert meta["anchors"][0]["frame_idx"] == 5
        assert meta["anchors"][0]["bucket"] == "[-10,10]|neutral|neutral"

    def test_export_empty_result(self, tmp_path):
        result = IdentityResult(frame_count=0, config=IdentityConfig())
        export_identity_metadata(result, tmp_path)
        assert not (tmp_path / "identity").exists()


# ── IdentityResult dataclass tests ──

class TestIdentityResult:
    def test_default(self):
        r = IdentityResult()
        assert r.frame_count == 0
        assert len(r.persons) == 0
        assert r.config is None

    def test_with_config(self):
        cfg = IdentityConfig(tau_id=0.5, anchor_count=3)
        r = IdentityResult(config=cfg)
        assert r.config.tau_id == 0.5
        assert r.config.anchor_count == 3


# ── IdentityConfig tests ──

class TestIdentityConfig:
    def test_defaults(self):
        cfg = IdentityConfig()
        assert cfg.tau_id == 0.35
        assert cfg.anchor_count == 5
        assert cfg.anchor_max_yaw == 15.0
        assert cfg.gate_face_confidence == 0.7
        assert cfg.gate_blur_min == 50.0
        assert cfg.loose_face_confidence == 0.5
        assert cfg.coverage_max_per_bucket == 2
        assert cfg.challenge_count == 8
        assert cfg.challenge_min_stable == 0.4
        assert cfg.medoid_max_candidates == 200

    def test_custom(self):
        cfg = IdentityConfig(tau_id=0.5, anchor_count=3)
        assert cfg.tau_id == 0.5
        assert cfg.anchor_count == 3


# ── Bank bridge tests ──

class TestBankBridge:
    """IdentityResult → MemoryBank registration tests."""

    def _build_result_and_records(self, n_frames=50):
        """테스트용 IdentityResult + records 생성."""
        base_emb = _make_embedding(512, seed=42)

        records = []
        for i in range(n_frames):
            r = _make_record(
                frame_idx=i,
                timestamp_ms=i * 2000.0,
                e_id_seed=42,

                face_confidence=0.9,
                face_area_ratio=0.1,   # higher for quality > 0.5
                blur_score=200.0,      # higher for quality > 0.5
                head_yaw=(i % 20 - 10) * 3.0,
                head_pitch=(i % 10 - 5) * 3.0,
                smile_intensity=0.5 if i % 5 == 0 else 0.1,
                eye_open_ratio=0.5,
            )
            r.e_id = base_emb + np.random.default_rng(i).standard_normal(512).astype(np.float32) * 0.05
            r.e_id = r.e_id / np.linalg.norm(r.e_id)
            records.append(r)

        builder = IdentityBuilder()
        result = builder.build(records)
        return result, records

    def test_basic_registration(self, tmp_path):
        """선택된 프레임이 MemoryBank에 등록된다."""
        from momentscan.algorithm.identity.bank_bridge import register_to_bank

        result, records = self._build_result_and_records()
        assert 0 in result.persons

        reg = register_to_bank(result, records, tmp_path)

        assert reg.persons_registered == 1
        assert reg.frames_registered > 0
        assert 0 in reg.nodes_created
        assert reg.nodes_created[0] > 0

        # Bank file should exist
        bank_path = tmp_path / "identity" / "person_0" / "memory_bank.json"
        assert bank_path.exists()

        # Verify bank contents
        from momentbank import load_bank
        bank = load_bank(bank_path)
        assert bank.person_id == 0
        assert len(bank.nodes) > 0

    def test_bank_accumulation(self, tmp_path):
        """두 번 호출하면 기존 bank를 로드하여 누적한다."""
        from momentscan.algorithm.identity.bank_bridge import register_to_bank
        from momentbank import load_bank

        result, records = self._build_result_and_records()

        # First registration
        reg1 = register_to_bank(result, records, tmp_path)
        bank1 = load_bank(tmp_path / "identity" / "person_0" / "memory_bank.json")
        nodes_after_1 = len(bank1.nodes)
        hit_sum_1 = sum(n.meta_hist.hit_count for n in bank1.nodes)

        # Second registration (same data, simulates re-processing)
        reg2 = register_to_bank(result, records, tmp_path)
        bank2 = load_bank(tmp_path / "identity" / "person_0" / "memory_bank.json")
        hit_sum_2 = sum(n.meta_hist.hit_count for n in bank2.nodes)

        # Hit counts should increase (frames merged into existing nodes or new ones created)
        assert hit_sum_2 > hit_sum_1

    def test_empty_result(self, tmp_path):
        """빈 결과 → 아무것도 등록하지 않는다."""
        from momentscan.algorithm.identity.bank_bridge import register_to_bank

        result = IdentityResult(frame_count=0, config=IdentityConfig())
        records = []

        reg = register_to_bank(result, records, tmp_path)
        assert reg.persons_registered == 0
        assert reg.frames_registered == 0

    def test_separate_bank_dir(self, tmp_path):
        """bank_dir를 별도로 지정하면 해당 경로에 저장."""
        from momentscan.algorithm.identity.bank_bridge import register_to_bank

        result, records = self._build_result_and_records()
        output_dir = tmp_path / "output"
        bank_dir = tmp_path / "bank"

        reg = register_to_bank(result, records, output_dir, bank_dir=bank_dir)

        # Bank should be in bank_dir, not output_dir
        assert (bank_dir / "identity" / "person_0" / "memory_bank.json").exists()
        assert not (output_dir / "identity" / "person_0" / "memory_bank.json").exists()

    def test_bank_has_bucket_metadata(self, tmp_path):
        """등록된 bank node의 meta_hist에 bucket 정보가 있다."""
        from momentscan.algorithm.identity.bank_bridge import register_to_bank
        from momentbank import load_bank

        result, records = self._build_result_and_records()
        register_to_bank(result, records, tmp_path)

        bank = load_bank(tmp_path / "identity" / "person_0" / "memory_bank.json")

        # At least one node should have yaw/pitch/expression bins
        has_meta = False
        for node in bank.nodes:
            if node.meta_hist.yaw_bins or node.meta_hist.pitch_bins:
                has_meta = True
                break
        assert has_meta


# ── Pivot Assignment tests ──

class TestPivotAssignment:
    """Portrait pivot system tests (Phase A3)."""

    def test_imports(self):
        from momentscan.algorithm.identity.pivots import (
            PosePivot, ExpressionPivot, PivotAssignment,
            POSE_PIVOTS, EXPRESSION_PIVOTS,
            assign_pivot, assign_pivot_fallback, pivot_to_bucket,
        )
        assert len(POSE_PIVOTS) == 5
        assert len(EXPRESSION_PIVOTS) == 4

    def test_assign_pivot_frontal_neutral(self):
        from momentscan.algorithm.identity.pivots import assign_pivot
        result = assign_pivot(0, 0, {"AU12": 0.0, "AU25": 0.0, "AU26": 0.0})
        assert result is not None
        assert result.pose.name == "frontal"
        assert result.expression.name == "neutral"
        assert result.pose_distance < 1.0

    def test_assign_pivot_three_quarter(self):
        from momentscan.algorithm.identity.pivots import assign_pivot
        result = assign_pivot(28, 0, {"AU12": 0.5, "AU25": 0.3, "AU26": 0.1})
        assert result is not None
        assert result.pose.name == "three-quarter"
        assert result.expression.name == "neutral"

    def test_assign_pivot_smile(self):
        from momentscan.algorithm.identity.pivots import assign_pivot
        result = assign_pivot(0, 0, {"AU12": 2.0, "AU25": 0.5, "AU26": 0.3})
        assert result is not None
        assert result.expression.name == "smile"

    def test_assign_pivot_excited(self):
        from momentscan.algorithm.identity.pivots import assign_pivot
        result = assign_pivot(0, 0, {"AU12": 2.0, "AU25": 2.0, "AU26": 0.3})
        assert result is not None
        assert result.expression.name == "excited"

    def test_assign_pivot_surprised(self):
        from momentscan.algorithm.identity.pivots import assign_pivot
        result = assign_pivot(0, 0, {"AU12": 0.3, "AU25": 2.0, "AU26": 1.8})
        assert result is not None
        assert result.expression.name == "surprised"

    def test_assign_pivot_rejected(self):
        """r_accept 초과 시 None."""
        from momentscan.algorithm.identity.pivots import assign_pivot
        # Very extreme pose (90, 45) - far from all pivots
        result = assign_pivot(90, 45, {"AU12": 0.0, "AU25": 0.0, "AU26": 0.0})
        assert result is None

    def test_assign_pivot_symmetric_yaw(self):
        """좌우 대칭 - yaw=-30과 yaw=30은 같은 pivot."""
        from momentscan.algorithm.identity.pivots import assign_pivot
        r_left = assign_pivot(-30, 0, {"AU12": 0.0, "AU25": 0.0, "AU26": 0.0})
        r_right = assign_pivot(30, 0, {"AU12": 0.0, "AU25": 0.0, "AU26": 0.0})
        assert r_left is not None
        assert r_right is not None
        assert r_left.pose.name == r_right.pose.name
        assert abs(r_left.pose_distance - r_right.pose_distance) < 0.01

    def test_assign_pivot_fallback_smile(self):
        from momentscan.algorithm.identity.pivots import assign_pivot_fallback
        result = assign_pivot_fallback(0, 0, smile_intensity=0.6, mouth_open_ratio=0.1)
        assert result is not None
        assert result.expression.name == "smile"

    def test_assign_pivot_fallback_excited(self):
        from momentscan.algorithm.identity.pivots import assign_pivot_fallback
        result = assign_pivot_fallback(0, 0, smile_intensity=0.5, mouth_open_ratio=0.7)
        assert result is not None
        assert result.expression.name == "excited"

    def test_assign_pivot_fallback_surprised(self):
        from momentscan.algorithm.identity.pivots import assign_pivot_fallback
        result = assign_pivot_fallback(0, 0, smile_intensity=0.1, mouth_open_ratio=0.7)
        assert result is not None
        assert result.expression.name == "surprised"

    def test_pivot_to_bucket(self):
        from momentscan.algorithm.identity.pivots import assign_pivot, pivot_to_bucket
        assignment = assign_pivot(0, 0, {"AU12": 2.0, "AU25": 0.3, "AU26": 0.1})
        bucket = pivot_to_bucket(assignment)
        assert bucket.yaw_bin == "frontal"
        assert bucket.expression_bin == "smile"

    def test_pivot_name_format(self):
        from momentscan.algorithm.identity.pivots import assign_pivot
        result = assign_pivot(0, 0, {"AU12": 2.0, "AU25": 2.0, "AU26": 0.0})
        assert result.pivot_name == "frontal|excited"

    def test_builder_with_au_records(self):
        """Builder uses AU-based pivot assignment when au_intensities available."""
        records = []
        for i in range(10):
            r = _make_record(
                frame_idx=i,
                timestamp_ms=i * 2000.0,
                e_id_seed=1,  # Same person
                head_yaw=0.0,
                head_pitch=0.0,
            )
            r.au_intensities = {"AU12": 2.0, "AU25": 0.5, "AU26": 0.3}
            records.append(r)

        result = IdentityBuilder().build(records)
        assert 0 in result.persons
        person = result.persons[0]

        # All frames should have smile expression from AU
        for f in person.anchor_frames:
            assert f.pivot_name is not None
            assert "smile" in f.pivot_name
            assert f.pivot_distance >= 0

    def test_builder_fallback_without_au(self):
        """Builder uses fallback pivot when no au_intensities."""
        records = []
        for i in range(10):
            r = _make_record(
                frame_idx=i,
                timestamp_ms=i * 2000.0,
                e_id_seed=1,
                head_yaw=0.0,
                head_pitch=0.0,
                smile_intensity=0.6,
                mouth_open_ratio=0.1,
            )
            records.append(r)

        result = IdentityBuilder().build(records)
        assert 0 in result.persons
        person = result.persons[0]

        for f in person.anchor_frames:
            assert f.pivot_name is not None
            assert "smile" in f.pivot_name

    def test_pivot_coverage_in_person_identity(self):
        """PersonIdentity.pivot_coverage tracks pivot distribution."""
        records = []
        for i in range(10):
            r = _make_record(
                frame_idx=i,
                timestamp_ms=i * 2000.0,
                e_id_seed=1,
                head_yaw=0.0,
            )
            r.au_intensities = {"AU12": 0.0, "AU25": 0.0, "AU26": 0.0}
            records.append(r)

        result = IdentityBuilder().build(records)
        person = result.persons[0]
        assert isinstance(person.pivot_coverage, dict)
        # All frontal|neutral
        if person.pivot_coverage:
            assert "frontal|neutral" in person.pivot_coverage

    def test_identity_frame_pivot_fields(self):
        """IdentityFrame has pivot_name and pivot_distance fields."""
        f = IdentityFrame(
            frame_idx=0,
            timestamp_ms=0.0,
            set_type="anchor",
            bucket=BucketLabel("frontal", "frontal", "smile"),
            pivot_name="frontal|smile",
            pivot_distance=2.5,
        )
        assert f.pivot_name == "frontal|smile"
        assert f.pivot_distance == 2.5

    def test_identity_record_au_intensities_field(self):
        """IdentityRecord has au_intensities field."""
        r = IdentityRecord(frame_idx=0, timestamp_ms=0.0)
        assert r.au_intensities is None
        r.au_intensities = {"AU12": 2.0}
        assert r.au_intensities["AU12"] == 2.0
