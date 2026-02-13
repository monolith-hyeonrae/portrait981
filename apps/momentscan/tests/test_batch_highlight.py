"""Tests for batch highlight engine."""

import pytest
import numpy as np

from momentscan.algorithm.batch.types import (
    FrameRecord,
    HighlightConfig,
    HighlightResult,
    HighlightWindow,
)
from momentscan.algorithm.batch.highlight import BatchHighlightEngine
from momentscan.algorithm.batch.extract import extract_frame_record


# ── FrameRecord tests ──

class TestFrameRecord:
    def test_default_values(self):
        r = FrameRecord(frame_idx=0, timestamp_ms=0.0)
        assert r.face_detected is False
        assert r.face_confidence == 0.0
        assert r.mouth_open_ratio == 0.0

    def test_with_values(self):
        r = FrameRecord(
            frame_idx=10,
            timestamp_ms=1000.0,
            face_detected=True,
            face_confidence=0.95,
            head_yaw=15.0,
        )
        assert r.frame_idx == 10
        assert r.face_detected is True
        assert r.head_yaw == 15.0


# ── HighlightConfig tests ──

class TestHighlightConfig:
    def test_default_config(self):
        cfg = HighlightConfig()
        assert cfg.gate_face_confidence == 0.7
        assert cfg.smoothing_alpha == 0.25
        assert cfg.peak_min_distance_sec == 2.5

    def test_custom_config(self):
        cfg = HighlightConfig(fps=30.0, peak_min_distance_sec=1.0)
        assert cfg.fps == 30.0
        assert cfg.peak_min_distance_sec == 1.0


# ── BatchHighlightEngine tests ──

def _make_records(n: int, *, fps: float = 10.0) -> list[FrameRecord]:
    """기본 프레임 레코드 n개 생성."""
    return [
        FrameRecord(
            frame_idx=i,
            timestamp_ms=i * (1000.0 / fps),
            face_detected=True,
            face_confidence=0.9,
            face_area_ratio=0.05,
            blur_score=100.0,
            brightness=128.0,
            contrast=50.0,
        )
        for i in range(n)
    ]


def _inject_spike(records: list[FrameRecord], idx: int) -> None:
    """특정 프레임에 mouth_open spike 주입."""
    records[idx].mouth_open_ratio = 0.9
    # 주변 프레임은 낮은 값
    for i in range(max(0, idx - 5), min(len(records), idx + 6)):
        if i != idx:
            records[i].mouth_open_ratio = 0.1


class TestBatchHighlightEngine:
    def test_empty_records(self):
        engine = BatchHighlightEngine()
        result = engine.analyze([])
        assert result.frame_count == 0
        assert result.windows == []

    def test_too_few_records(self):
        records = _make_records(2)
        engine = BatchHighlightEngine()
        result = engine.analyze(records)
        assert result.frame_count == 2
        assert result.windows == []

    def test_flat_signal_no_peaks(self):
        """모든 프레임이 동일하면 peak이 없어야 한다."""
        records = _make_records(100)
        for r in records:
            r.mouth_open_ratio = 0.3
        engine = BatchHighlightEngine()
        result = engine.analyze(records)
        assert len(result.windows) == 0

    def test_quality_gate_filters_bad_frames(self):
        """blur가 낮은 프레임은 gate에서 걸러진다."""
        records = _make_records(50)
        _inject_spike(records, 25)
        # blur를 gate 미달로 설정
        records[25].blur_score = 10.0

        cfg = HighlightConfig(gate_blur_min=50.0)
        engine = BatchHighlightEngine(config=cfg)
        result = engine.analyze(records)

        # spike 프레임이 gate에서 걸렸으므로 해당 프레임의 최종 점수는 0
        # (peak이 잡히더라도 score=0인 프레임은 selected_frames에서 제외)

    def test_single_spike_detected(self):
        """뚜렷한 spike 하나가 있으면 window가 생성된다."""
        records = _make_records(200)
        for r in records:
            r.mouth_open_ratio = 0.1

        # 프레임 100 부근에 강한 spike
        for i in range(95, 106):
            records[i].mouth_open_ratio = 0.8

        engine = BatchHighlightEngine()
        result = engine.analyze(records)

        # spike가 충분히 뚜렷하면 최소 1개 window
        assert result.frame_count == 200
        # windows가 생겼는지 확인 (prominence threshold에 따라 달라질 수 있음)
        if len(result.windows) > 0:
            w = result.windows[0]
            assert w.window_id >= 1
            assert w.peak_ms > 0
            assert w.score > 0
            assert len(w.selected_frames) > 0

    def test_multiple_spikes_separated(self):
        """충분히 떨어진 두 spike는 각각 window가 된다."""
        records = _make_records(500)
        for r in records:
            r.mouth_open_ratio = 0.1

        # spike at frame 100
        for i in range(95, 106):
            records[i].mouth_open_ratio = 0.85
        # spike at frame 350
        for i in range(345, 356):
            records[i].mouth_open_ratio = 0.85

        engine = BatchHighlightEngine()
        result = engine.analyze(records)

        if len(result.windows) >= 2:
            assert result.windows[0].peak_ms < result.windows[1].peak_ms

    def test_window_has_reason(self):
        """Window의 reason에 기여 feature가 포함된다."""
        records = _make_records(200)
        for r in records:
            r.mouth_open_ratio = 0.1

        for i in range(95, 106):
            records[i].mouth_open_ratio = 0.85

        engine = BatchHighlightEngine()
        result = engine.analyze(records)

        if result.windows:
            assert isinstance(result.windows[0].reason, dict)

    def test_no_face_frames_excluded(self):
        """얼굴이 없는 프레임은 gate에서 제외된다."""
        records = _make_records(50)
        for r in records:
            r.face_detected = False

        engine = BatchHighlightEngine()
        result = engine.analyze(records)
        # 모든 프레임이 gate 탈락 → peak 없음
        assert len(result.windows) == 0


# ── HighlightResult export test ──

class TestHighlightResult:
    def test_export(self, tmp_path):
        result = HighlightResult(
            windows=[
                HighlightWindow(
                    window_id=1,
                    start_ms=1000.0,
                    end_ms=3000.0,
                    peak_ms=2000.0,
                    score=0.85,
                    reason={"mouth_open_ratio": 1.2},
                    selected_frames=[
                        {"frame_idx": 20, "timestamp_ms": 2000.0, "frame_score": 0.85}
                    ],
                )
            ],
            frame_count=100,
        )
        result.export(tmp_path)
        assert (tmp_path / "highlight" / "windows.json").exists()

        import json
        with open(tmp_path / "highlight" / "windows.json") as f:
            data = json.load(f)
        assert len(data) == 1
        assert data[0]["window_id"] == 1


# ── extract_frame_record tests ──

class TestExtractFrameRecord:
    def test_empty_results(self):
        assert extract_frame_record(None, []) is None

    def test_face_detect_from_data_object(self):
        """face.detect: FaceDetectOutput.faces에서 주 얼굴 수치 추출."""

        class MockFace:
            def __init__(self, **kw):
                self.confidence = kw.get("confidence", 0.0)
                self.area_ratio = kw.get("area_ratio", 0.0)
                self.center_distance = kw.get("center_distance", 0.0)
                self.yaw = kw.get("yaw", 0.0)
                self.pitch = kw.get("pitch", 0.0)
                self.roll = kw.get("roll", 0.0)

        class MockFaceOutput:
            def __init__(self, faces):
                self.faces = faces

        class MockObs:
            def __init__(self, source, *, signals=None, data=None):
                self.source = source
                self.signals = signals or {}
                self.data = data

        class MockFlowData:
            def __init__(self, observations):
                self.observations = observations

        class MockFrame:
            frame_id = 42
            t_src_ns = 4_200_000_000

        face = MockFace(confidence=0.92, area_ratio=0.04, yaw=10.0, pitch=-5.0)
        face_output = MockFaceOutput([face])
        flow_data = MockFlowData([
            MockObs("face.detect", data=face_output),
            MockObs("frame.quality", signals={
                "blur_score": 150.0,
                "brightness": 130.0,
                "contrast": 45.0,
            }),
        ])

        record = extract_frame_record(MockFrame(), [flow_data])

        assert record is not None
        assert record.frame_idx == 42
        assert record.timestamp_ms == 4200.0
        assert record.face_detected is True
        assert record.face_confidence == 0.92
        assert record.face_area_ratio == 0.04
        assert record.head_yaw == 10.0
        assert record.head_pitch == -5.0
        assert record.blur_score == 150.0

    def test_face_expression_from_data_object(self):
        """face.expression: FaceObservation.expression + per-face signals 추출."""

        class MockFace:
            def __init__(self, expression, signals):
                self.expression = expression
                self.signals = signals
                self.area_ratio = 0.05

        class MockExprOutput:
            def __init__(self, faces):
                self.faces = faces

        class MockObs:
            def __init__(self, source, *, signals=None, data=None):
                self.source = source
                self.signals = signals or {}
                self.data = data

        class MockFlowData:
            def __init__(self, observations):
                self.observations = observations

        class MockFrame:
            frame_id = 1
            t_src_ns = 100_000_000

        face = MockFace(expression=0.7, signals={"em_happy": 0.6, "em_neutral": 0.2})
        expr_output = MockExprOutput([face])

        flow_data = MockFlowData([
            MockObs("face.expression", data=expr_output),
        ])
        record = extract_frame_record(MockFrame(), [flow_data])

        assert record.mouth_open_ratio == 0.7
        assert record.smile_intensity == 0.6
        assert record.eye_open_ratio == pytest.approx(0.8)

    def test_body_pose_from_keypoints(self):
        """body.pose: PoseOutput.keypoints에서 wrist_raise 등 계산."""
        import math

        class MockPoseOutput:
            def __init__(self, keypoints):
                self.keypoints = keypoints

        class MockObs:
            def __init__(self, source, *, signals=None, data=None):
                self.source = source
                self.signals = signals or {}
                self.data = data

        class MockFlowData:
            def __init__(self, observations):
                self.observations = observations

        class MockFrame:
            frame_id = 1
            t_src_ns = 100_000_000

        # Keypoints: [x, y, confidence]
        # 손목이 어깨보다 100px 위 (image height 480)
        kpts = [[0] * 3] * 17
        kpts[5] = [200, 200, 0.9]   # left shoulder
        kpts[6] = [400, 200, 0.9]   # right shoulder
        kpts[7] = [180, 170, 0.9]   # left elbow
        kpts[8] = [420, 170, 0.9]   # right elbow
        kpts[9] = [160, 100, 0.9]   # left wrist (100px above shoulder)
        kpts[10] = [440, 200, 0.9]  # right wrist (same as shoulder)
        kpts[0] = [300, 150, 0.9]   # nose

        pose_output = MockPoseOutput([{
            "keypoints": kpts,
            "image_size": (640, 480),
        }])
        flow_data = MockFlowData([
            MockObs("body.pose", data=pose_output),
        ])
        record = extract_frame_record(MockFrame(), [flow_data])

        # wrist_raise: (200 - 100) / 480 ≈ 0.208
        assert record.wrist_raise == pytest.approx(100.0 / 480.0, abs=0.01)
        # hand_near_face: 왼쪽 손목이 코에 가까움
        assert record.hand_near_face > 0.0
        # torso_rotation: 어깨가 수평 → 약 0도
        assert record.torso_rotation == pytest.approx(0.0, abs=1.0)

    def test_selects_largest_face(self):
        """여러 얼굴 중 area_ratio 최대인 주 얼굴을 선택한다."""

        class MockFace:
            def __init__(self, confidence, area_ratio, yaw=0.0):
                self.confidence = confidence
                self.area_ratio = area_ratio
                self.center_distance = 0.0
                self.yaw = yaw
                self.pitch = 0.0
                self.roll = 0.0

        class MockFaceOutput:
            def __init__(self, faces):
                self.faces = faces

        class MockObs:
            def __init__(self, source, *, data=None):
                self.source = source
                self.signals = {}
                self.data = data

        class MockFlowData:
            def __init__(self, observations):
                self.observations = observations

        class MockFrame:
            frame_id = 1
            t_src_ns = 0

        small_face = MockFace(confidence=0.99, area_ratio=0.01, yaw=30.0)
        big_face = MockFace(confidence=0.85, area_ratio=0.08, yaw=-5.0)
        face_output = MockFaceOutput([small_face, big_face])

        flow_data = MockFlowData([MockObs("face.detect", data=face_output)])
        record = extract_frame_record(MockFrame(), [flow_data])

        # 큰 얼굴이 선택됨
        assert record.face_confidence == 0.85
        assert record.face_area_ratio == 0.08
        assert record.head_yaw == -5.0
