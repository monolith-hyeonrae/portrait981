"""Tests for collection/extract.py."""

import numpy as np
import pytest
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from momentscan.algorithm.collection.extract import (
    extract_collection_record,
    reset_extract_state,
)


# ── Mock types (same pattern as test_identity_builder.py) ──


def _make_embedding(dim: int = 512, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    return v / np.linalg.norm(v)


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
class MockObs:
    source: str = ""
    data: Any = None
    signals: Dict[str, float] = field(default_factory=dict)
    metadata: Optional[Dict] = None


@dataclass
class MockFlowData:
    observations: List[Any] = field(default_factory=list)


@dataclass
class MockFrame:
    frame_id: int = 0
    t_src_ns: int = 0


@dataclass
class MockFaceQualityOutput:
    face_blur: float = 0.0
    face_crop_box: Optional[tuple] = None
    image_size: Optional[tuple] = None


@dataclass
class MockHeadPoseEstimate:
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0


@dataclass
class MockHeadPoseOutput:
    estimates: List[Any] = field(default_factory=list)


@dataclass
class MockFaceAUOutput:
    au_intensities: List[Dict[str, float]] = field(default_factory=list)


@dataclass
class MockPortraitScoreOutput:
    head_aesthetic: float = 0.0


@dataclass
class MockGateResult:
    role: str = "main"
    suitability: float = 0.0


@dataclass
class MockGateOutput:
    main_gate_passed: bool = True
    main_fail_reasons: tuple = ()
    results: List[Any] = field(default_factory=list)


class TestExtractCollectionRecord:
    def setup_method(self):
        reset_extract_state()

    def test_no_results(self):
        frame = MockFrame()
        assert extract_collection_record(frame, []) is None

    def test_basic_face_detection(self):
        emb = _make_embedding(512, seed=1)
        face = MockFace(embedding=emb, yaw=10.0, pitch=-5.0, confidence=0.85)
        output = MockOutput(faces=[face])
        obs = MockObs(source="face.detect", data=output)
        flow = MockFlowData(observations=[obs])
        frame = MockFrame(frame_id=42, t_src_ns=1_000_000_000)

        record = extract_collection_record(frame, [flow])
        assert record is not None
        assert record.frame_idx == 42
        assert record.timestamp_ms == 1000.0
        assert record.face_detected is True
        assert record.face_confidence == 0.85
        assert record.head_yaw == 10.0
        assert record.head_pitch == -5.0
        assert record.e_id is not None
        assert abs(np.linalg.norm(record.e_id) - 1.0) < 1e-5

    def test_no_face_still_returns_record(self):
        """Unlike identity extract, collection extract returns record even without face."""
        obs = MockObs(source="frame.quality", signals={"blur_score": 50.0})
        flow = MockFlowData(observations=[obs])
        frame = MockFrame(frame_id=1, t_src_ns=100_000_000)

        record = extract_collection_record(frame, [flow])
        assert record is not None
        assert record.face_detected is False
        assert record.e_id is None
        assert record.blur_score == 50.0

    def test_expression_extraction(self):
        emb = _make_embedding(512, seed=1)
        face_detect = MockFace(embedding=emb)
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

        record = extract_collection_record(frame, [flow])
        assert record is not None
        assert record.smile_intensity == 0.8
        assert record.mouth_open_ratio == 0.7
        assert abs(record.eye_open_ratio - 0.9) < 1e-5

    def test_au_extraction(self):
        emb = _make_embedding(512, seed=1)
        face = MockFace(embedding=emb)
        detect_obs = MockObs(source="face.detect", data=MockOutput(faces=[face]))

        au_data = MockFaceAUOutput(
            au_intensities=[{"AU6": 1.5, "AU12": 2.0, "AU25": 0.5, "AU26": 0.3}]
        )
        au_obs = MockObs(source="face.au", data=au_data)

        flow = MockFlowData(observations=[detect_obs, au_obs])
        frame = MockFrame()

        record = extract_collection_record(frame, [flow])
        assert record is not None
        assert record.au6_cheek_raiser == 1.5
        assert record.au12_lip_corner == 2.0
        assert record.au25_lips_part == 0.5
        assert record.au26_jaw_drop == 0.3
        assert record.au_intensities is not None
        # AU12 max strategy: min(2.0/3.0, 1.0) ≈ 0.667 > 0 → should boost smile
        assert record.smile_intensity >= 0.66

    def test_head_pose_override(self):
        emb = _make_embedding(512, seed=1)
        face = MockFace(embedding=emb, yaw=5.0, pitch=3.0)
        detect_obs = MockObs(source="face.detect", data=MockOutput(faces=[face]))

        pose_est = MockHeadPoseEstimate(yaw=12.5, pitch=-8.0, roll=2.0)
        pose_data = MockHeadPoseOutput(estimates=[pose_est])
        pose_obs = MockObs(source="head.pose", data=pose_data)

        flow = MockFlowData(observations=[detect_obs, pose_obs])
        frame = MockFrame()

        record = extract_collection_record(frame, [flow])
        assert record is not None
        # head.pose should override geometric estimates
        assert record.head_yaw == 12.5
        assert record.head_pitch == -8.0
        assert record.head_roll == 2.0

    def test_face_quality_crop(self):
        emb = _make_embedding(512, seed=1)
        face = MockFace(embedding=emb)
        detect_obs = MockObs(source="face.detect", data=MockOutput(faces=[face]))

        fq_data = MockFaceQualityOutput(
            face_blur=250.0,
            face_crop_box=(10, 20, 100, 120),
            image_size=(640, 480),
        )
        fq_obs = MockObs(source="face.quality", data=fq_data)

        flow = MockFlowData(observations=[detect_obs, fq_obs])
        frame = MockFrame()

        record = extract_collection_record(frame, [flow])
        assert record is not None
        assert record.face_blur == 250.0
        assert record.face_crop_box == (10, 20, 100, 120)
        assert record.image_size == (640, 480)

    def test_gate_extraction(self):
        emb = _make_embedding(512, seed=1)
        face = MockFace(embedding=emb)
        detect_obs = MockObs(source="face.detect", data=MockOutput(faces=[face]))

        gate_data = MockGateOutput(
            main_gate_passed=False,
            results=[MockGateResult(role="passenger", suitability=0.8)],
        )
        gate_obs = MockObs(source="face.gate", data=gate_data)

        flow = MockFlowData(observations=[detect_obs, gate_obs])
        frame = MockFrame()

        record = extract_collection_record(frame, [flow])
        assert record is not None
        assert record.gate_passed is False
        assert record.passenger_suitability == 0.8

    def test_person_id_from_classify(self):
        emb = _make_embedding(512, seed=1)
        face = MockFace(embedding=emb)
        detect_obs = MockObs(source="face.detect", data=MockOutput(faces=[face]))

        classify_obs = MockObs(
            source="face.classify",
            signals={"main_confidence": 0.95},
        )

        flow = MockFlowData(observations=[detect_obs, classify_obs])
        frame = MockFrame()

        record = extract_collection_record(frame, [flow])
        assert record is not None
        assert record.person_id == 0

    def test_picks_largest_face(self):
        emb1 = _make_embedding(512, seed=1)
        emb2 = _make_embedding(512, seed=2)
        face1 = MockFace(embedding=emb1, area_ratio=0.03)
        face2 = MockFace(embedding=emb2, area_ratio=0.08)
        output = MockOutput(faces=[face1, face2])
        obs = MockObs(source="face.detect", data=output)
        flow = MockFlowData(observations=[obs])
        frame = MockFrame()

        record = extract_collection_record(frame, [flow])
        assert record is not None
        assert record.face_area_ratio == 0.08
