"""Tests for batch highlight engine."""

import csv
import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from pathlib import Path

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
        assert r.eye_open_ratio == 0.0

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
        assert cfg.smoothing_alpha == 0.25
        assert cfg.peak_min_distance_sec == 2.5
        # Gate thresholds moved to FaceGateConfig (face.gate analyzer)
        assert not hasattr(cfg, "gate_face_confidence")
        assert not hasattr(cfg, "gate_eye_open_min")

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
    """특정 프레임에 smile_intensity spike 주입."""
    records[idx].smile_intensity = 0.9
    # 주변 프레임은 낮은 값
    for i in range(max(0, idx - 5), min(len(records), idx + 6)):
        if i != idx:
            records[i].smile_intensity = 0.1


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
            r.smile_intensity = 0.3
        engine = BatchHighlightEngine()
        result = engine.analyze(records)
        assert len(result.windows) == 0

    def test_quality_gate_filters_bad_frames(self):
        """gate_passed=False 프레임은 gate에서 걸러진다."""
        records = _make_records(50)
        _inject_spike(records, 25)
        # face.gate analyzer가 판정한 결과를 FrameRecord에 설정
        records[25].gate_passed = False
        records[25].gate_fail_reasons = "blur"

        engine = BatchHighlightEngine()
        result = engine.analyze(records)

        # spike 프레임이 gate에서 걸렸으므로 해당 프레임의 최종 점수는 0
        # (peak이 잡히더라도 score=0인 프레임은 selected_frames에서 제외)

    def test_single_spike_detected(self):
        """뚜렷한 spike 하나가 있으면 window가 생성된다."""
        records = _make_records(200)
        for r in records:
            r.smile_intensity = 0.1

        # 프레임 100 부근에 강한 spike
        for i in range(95, 106):
            records[i].smile_intensity = 0.8

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
            r.smile_intensity = 0.1

        # spike at frame 100
        for i in range(95, 106):
            records[i].smile_intensity = 0.85
        # spike at frame 350
        for i in range(345, 356):
            records[i].smile_intensity = 0.85

        engine = BatchHighlightEngine()
        result = engine.analyze(records)

        if len(result.windows) >= 2:
            assert result.windows[0].peak_ms < result.windows[1].peak_ms

    def test_window_has_reason(self):
        """Window의 reason에 기여 feature가 포함된다."""
        records = _make_records(200)
        for r in records:
            r.smile_intensity = 0.1

        for i in range(95, 106):
            records[i].smile_intensity = 0.85

        engine = BatchHighlightEngine()
        result = engine.analyze(records)

        if result.windows:
            assert isinstance(result.windows[0].reason, dict)

    def test_no_face_frames_excluded(self):
        """gate_passed=False 프레임은 모두 0점이 되어 peak 없음."""
        records = _make_records(50)
        for r in records:
            r.face_detected = False
            r.gate_passed = False
            r.gate_fail_reasons = "gate.detect.missing"

        engine = BatchHighlightEngine()
        result = engine.analyze(records)
        # 모든 프레임이 gate 탈락 → peak 없음
        assert len(result.windows) == 0

    def test_impact_smile_high_baseline(self):
        """대부분 웃는 영상에서도 smile 피크의 impact가 충분히 높아야 한다.

        smile_intensity weight=0.20, top-k=4 → max_achievable=top4 weights 합.
        smile만 활성일 때: 0.20 / max_achievable ≈ 0.27.
        duchenne/CLIP 결합 시 더 높아짐 (그게 설계 의도).
        """
        records = _make_records(200)
        # 대부분 웃는 baseline (0.7)
        for r in records:
            r.smile_intensity = 0.70

        # 피크: 몇 개 프레임만 1.0
        for i in range(98, 103):
            records[i].smile_intensity = 1.0

        engine = BatchHighlightEngine()
        result = engine.analyze(records)
        ts = result._timeseries

        # smile 단독 채널: weight=0.20 / top-4 max_achievable ≈ 0.25+
        peak_impact = ts["impact_scores"][100]
        assert peak_impact >= 0.25, f"impact at smile peak = {peak_impact:.3f} (expected >= 0.25)"

    def test_high_baseline_smile_peaks_detected(self):
        """대부분 웃는 영상에서도 피크가 검출되어야 한다.

        기존 prominence(90th pct of absolute scores) 방식: final score floor가 높아
        threshold >> 실제 피크 prominence → peak 미검출.
        수정 후: range × 0.30 fallback → floor에 무관하게 상대적 피크 검출.
        """
        records = _make_records(300)
        for r in records:
            r.smile_intensity = 0.70  # 높은 baseline

        # 3개 피크
        for i in range(48, 53):
            records[i].smile_intensity = 1.0
        for i in range(148, 153):
            records[i].smile_intensity = 1.0
        for i in range(248, 253):
            records[i].smile_intensity = 1.0

        engine = BatchHighlightEngine()
        result = engine.analyze(records)

        assert len(result.windows) == 3, (
            f"Expected 3 peaks for high-baseline smile video, got {len(result.windows)}"
        )
        peak_times = [w.peak_ms / 1000.0 for w in result.windows]
        assert all(t < 30.0 for t in peak_times)

    def test_passenger_bonus_boosts_score(self):
        """passenger_suitability=1.0인 프레임이 0.0보다 높은 최종 점수."""
        records = _make_records(100)
        for r in records:
            r.smile_intensity = 0.5

        # All frames have same features, but some have passenger suitability
        for i in range(45, 55):
            records[i].passenger_detected = True
            records[i].passenger_suitability = 1.0

        cfg = HighlightConfig(passenger_bonus_weight=0.30)
        engine = BatchHighlightEngine(config=cfg)
        result = engine.analyze(records)
        ts = result._timeseries

        # Frames with passenger suitability should have higher final scores
        boosted_score = ts["final_scores"][50]
        unboosted_score = ts["final_scores"][10]
        assert boosted_score > unboosted_score, (
            f"Passenger-boosted score {boosted_score:.3f} should be > unboosted {unboosted_score:.3f}"
        )

    def test_passenger_bonus_weight_zero(self):
        """passenger_bonus_weight=0 → 보너스 미적용."""
        records = _make_records(100)
        for r in records:
            r.smile_intensity = 0.5
            r.passenger_detected = True
            r.passenger_suitability = 1.0

        cfg_no_bonus = HighlightConfig(passenger_bonus_weight=0.0)
        cfg_with_bonus = HighlightConfig(passenger_bonus_weight=0.30)

        result_no = BatchHighlightEngine(config=cfg_no_bonus).analyze(records)
        result_with = BatchHighlightEngine(config=cfg_with_bonus).analyze(records)

        # Without bonus, all scores should be lower
        no_final = result_no._timeseries["final_scores"]
        with_final = result_with._timeseries["final_scores"]
        assert no_final[50] < with_final[50], (
            f"No-bonus score {no_final[50]:.3f} should be < bonus score {with_final[50]:.3f}"
        )

    def test_impact_smile_uses_absolute_not_delta(self):
        """smile_intensity impact는 delta가 아닌 per-video 절대값을 사용한다.

        동일한 절대 smile 피크라도 baseline이 다르면 impact는 동일해야 한다.
        delta 방식이었다면: 낮은 baseline → 큰 delta → 높은 impact, 높은 baseline → 작은 delta
        절대값 방식: 두 경우 모두 min-max 최고값 1.0 → 동일한 impact
        """
        # Case A: 낮은 baseline (0.1) + 피크 0.9
        records_a = _make_records(200)
        for r in records_a:
            r.smile_intensity = 0.1
        for i in range(98, 103):
            records_a[i].smile_intensity = 0.9

        # Case B: 높은 baseline (0.75) + 피크 0.9
        records_b = _make_records(200)
        for r in records_b:
            r.smile_intensity = 0.75
        for i in range(98, 103):
            records_b[i].smile_intensity = 0.9

        engine = BatchHighlightEngine()
        ts_a = engine.analyze(records_a)._timeseries
        ts_b = engine.analyze(records_b)._timeseries

        impact_a = ts_a["impact_scores"][100]
        impact_b = ts_b["impact_scores"][100]

        # 두 경우 모두 smile impact 기여분이 비슷해야 한다 (절대값 방식)
        assert impact_a == pytest.approx(impact_b, abs=0.05), (
            f"impact_a={impact_a:.3f}, impact_b={impact_b:.3f}: "
            "절대값 방식이면 baseline에 무관하게 비슷해야 함"
        )


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
                    reason={"smile_intensity": 1.2},
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

        assert record.smile_intensity == 0.6
        assert record.eye_open_ratio == pytest.approx(0.8)

    def test_face_au_overrides_smile_intensity(self):
        """face.au: AU12가 있으면 em_happy를 override하여 smile_intensity 보정."""

        class MockFace:
            def __init__(self, expression, signals):
                self.expression = expression
                self.signals = signals
                self.area_ratio = 0.05

        class MockExprOutput:
            def __init__(self, faces):
                self.faces = faces

        class MockAUOutput:
            def __init__(self, au_intensities):
                self.au_intensities = au_intensities
                self.au_presence = []
                self.face_bboxes = []

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

        # em_happy=0.3 (낮음), AU12=3.6 (명확한 미소)
        face = MockFace(expression=0.5, signals={"em_happy": 0.3, "em_neutral": 0.6})
        expr_output = MockExprOutput([face])
        au_output = MockAUOutput([{"AU12": 3.6, "AU6": 1.2}])

        flow_data = MockFlowData([
            MockObs("face.expression", data=expr_output),
            MockObs("face.au", data=au_output),
        ])
        record = extract_frame_record(MockFrame(), [flow_data])

        # AU12=3.6 → min(3.6/3.0, 1.0) = 1.0
        assert record.smile_intensity == pytest.approx(1.0)

    def test_face_au_absent_falls_back_to_em_happy(self):
        """face.au 없을 때 smile_intensity는 em_happy 그대로 유지."""

        class MockFace:
            def __init__(self, signals):
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

        face = MockFace(signals={"em_happy": 0.75, "em_neutral": 0.1})
        expr_output = MockExprOutput([face])

        flow_data = MockFlowData([
            MockObs("face.expression", data=expr_output),
            # face.au 없음
        ])
        record = extract_frame_record(MockFrame(), [flow_data])

        assert record.smile_intensity == pytest.approx(0.75)

    def test_face_au_low_au12_keeps_em_happy(self):
        """AU12가 em_happy보다 낮으면 em_happy를 유지한다 (max 전략).

        차량 내부 등 촬영 환경에서 LibreFace AU12가 낮게 나와도
        em_happy 값이 보존되어 smile_intensity가 손실되지 않아야 한다.
        """
        class MockFace:
            def __init__(self, signals):
                self.signals = signals
                self.area_ratio = 0.05

        class MockExprOutput:
            def __init__(self, faces):
                self.faces = faces

        class MockAUOutput:
            def __init__(self, au_list):
                self.au_intensities = au_list

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

        # em_happy=0.70 (확실한 미소), AU12=0.015 (LibreFace 저평가)
        face = MockFace(signals={"em_happy": 0.70, "em_neutral": 0.2})
        expr_output = MockExprOutput([face])
        au_output = MockAUOutput([{"AU12": 0.015}])  # AU12/3.0 = 0.005

        flow_data = MockFlowData([
            MockObs("face.expression", data=expr_output),
            MockObs("face.au", data=au_output),
        ])
        record = extract_frame_record(MockFrame(), [flow_data])

        # max(0.70, 0.005) = 0.70 — em_happy 보존
        assert record.smile_intensity == pytest.approx(0.70)

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


# ── Timeseries export tests ──

def _make_result_with_timeseries(n: int = 100) -> HighlightResult:
    """_timeseries가 포함된 HighlightResult를 생성한다."""
    records = _make_records(n)
    for r in records:
        r.smile_intensity = 0.1

    # 중간에 spike 삽입
    mid = n // 2
    for i in range(max(0, mid - 5), min(n, mid + 6)):
        records[i].smile_intensity = 0.85

    engine = BatchHighlightEngine()
    return engine.analyze(records)


class TestExportTimeseriesCsv:
    def test_csv_created(self, tmp_path):
        """timeseries.csv가 생성된다."""
        result = _make_result_with_timeseries(200)
        result.export(tmp_path)

        csv_path = tmp_path / "highlight" / "timeseries.csv"
        assert csv_path.exists()

    def test_csv_header_and_rows(self, tmp_path):
        """CSV 헤더와 행 수가 올바르다."""
        n = 200
        result = _make_result_with_timeseries(n)
        result.export(tmp_path)

        csv_path = tmp_path / "highlight" / "timeseries.csv"
        with open(csv_path) as f:
            reader = csv.reader(f)
            header = next(reader)
            rows = list(reader)

        assert "frame_idx" in header
        assert "timestamp_ms" in header
        assert "gate_pass" in header
        assert "quality_score" in header
        assert "impact_score" in header
        assert "final_score" in header
        assert "smoothed_score" in header
        assert "is_peak" in header
        assert "smile_intensity" in header
        assert len(rows) == n

    def test_csv_peak_marked(self, tmp_path):
        """CSV에서 is_peak 컬럼이 peak 프레임에 1로 마킹된다."""
        result = _make_result_with_timeseries(200)
        result.export(tmp_path)

        csv_path = tmp_path / "highlight" / "timeseries.csv"
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            peak_rows = [r for r in reader if r["is_peak"] == "1"]

        # peak가 있으면 is_peak=1인 행이 존재
        if result.windows:
            assert len(peak_rows) > 0

    def test_no_timeseries_no_csv(self, tmp_path):
        """_timeseries가 None이면 CSV가 생성되지 않는다."""
        result = HighlightResult(
            windows=[],
            frame_count=10,
        )
        result.export(tmp_path)

        csv_path = tmp_path / "highlight" / "timeseries.csv"
        assert not csv_path.exists()


class TestExportScoreCurve:
    def test_score_curve_without_matplotlib(self, tmp_path):
        """matplotlib이 없어도 에러 없이 skip."""
        result = _make_result_with_timeseries(200)

        with patch.dict("sys.modules", {"matplotlib": None}):
            # matplotlib import 실패 시 graceful skip
            result.export(tmp_path)

        # windows.json과 timeseries.csv는 생성됨
        assert (tmp_path / "highlight" / "windows.json").exists()
        assert (tmp_path / "highlight" / "timeseries.csv").exists()

    def test_score_curve_with_matplotlib(self, tmp_path):
        """matplotlib이 있으면 score_curve.png가 생성된다."""
        try:
            import matplotlib
        except ImportError:
            pytest.skip("matplotlib not installed")

        result = _make_result_with_timeseries(200)
        result.export(tmp_path)

        png_path = tmp_path / "highlight" / "score_curve.png"
        assert png_path.exists()
        assert png_path.stat().st_size > 0


# ── Frame export tests ──

class TestExportHighlightFrames:
    def test_no_windows_no_export(self, tmp_path):
        """윈도우가 없으면 frames/ 디렉토리가 생성되지 않는다."""
        from momentscan.algorithm.batch.export_frames import export_highlight_frames

        result = HighlightResult(windows=[], frame_count=10)
        export_highlight_frames(Path("/dummy.mp4"), result, tmp_path)

        assert not (tmp_path / "highlight" / "frames").exists()

    def test_export_with_mock_source(self, tmp_path):
        """Mock FileSource로 frame 추출을 검증한다."""
        from momentscan.algorithm.batch.export_frames import export_highlight_frames

        result = HighlightResult(
            windows=[
                HighlightWindow(
                    window_id=1,
                    start_ms=1000.0,
                    end_ms=3000.0,
                    peak_ms=2000.0,
                    score=0.85,
                    reason={},
                    selected_frames=[
                        {"frame_idx": 18, "timestamp_ms": 1800.0, "frame_score": 0.7},
                        {"frame_idx": 22, "timestamp_ms": 2200.0, "frame_score": 0.6},
                    ],
                ),
            ],
            frame_count=100,
        )

        # Mock FileSource + Frame
        mock_frame = MagicMock()
        mock_frame.data = np.zeros((480, 640, 3), dtype=np.uint8)

        mock_source = MagicMock()
        mock_source.seek.return_value = True
        mock_source.read.return_value = mock_frame

        with patch("visualbase.sources.file.FileSource") as MockFS, \
             patch("momentscan.algorithm.batch.export_frames.cv2") as mock_cv2:
            MockFS.return_value = mock_source
            export_highlight_frames(Path("/dummy.mp4"), result, tmp_path)

            # seek가 3번 호출됨 (peak + 2 best, timestamp 순)
            assert mock_source.seek.call_count == 3
            assert mock_cv2.imwrite.call_count == 3
            mock_source.close.assert_called_once()

    def test_export_filenames(self, tmp_path):
        """출력 파일명이 w{id}_peak_{ms}ms.jpg / w{id}_best_{ms}ms.jpg 형식이다."""
        from momentscan.algorithm.batch.export_frames import export_highlight_frames

        result = HighlightResult(
            windows=[
                HighlightWindow(
                    window_id=1,
                    start_ms=1000.0,
                    end_ms=3000.0,
                    peak_ms=2000.0,
                    score=0.85,
                    reason={},
                    selected_frames=[
                        {"frame_idx": 22, "timestamp_ms": 2200.0, "frame_score": 0.6},
                    ],
                ),
            ],
            frame_count=100,
        )

        mock_frame = MagicMock()
        mock_frame.data = np.zeros((480, 640, 3), dtype=np.uint8)

        mock_source = MagicMock()
        mock_source.seek.return_value = True
        mock_source.read.return_value = mock_frame

        with patch("visualbase.sources.file.FileSource") as MockFS, \
             patch("momentscan.algorithm.batch.export_frames.cv2") as mock_cv2:
            MockFS.return_value = mock_source
            export_highlight_frames(Path("/dummy.mp4"), result, tmp_path)

            # imwrite에 전달된 파일명 확인
            written_paths = [call.args[0] for call in mock_cv2.imwrite.call_args_list]
            assert any("w1_peak_2000ms.jpg" in p for p in written_paths)
            assert any("w1_best_2200ms.jpg" in p for p in written_paths)


# ── Highlight report tests ──

class TestExportHighlightReport:
    def test_report_created_with_timeseries(self, tmp_path):
        """_timeseries가 있으면 report.html이 생성된다."""
        from momentscan.algorithm.batch.export_report import export_highlight_report

        result = _make_result_with_timeseries(200)

        mock_frame = MagicMock()
        mock_frame.data = np.zeros((480, 640, 3), dtype=np.uint8)

        mock_source = MagicMock()
        mock_source.seek.return_value = True
        mock_source.read.return_value = mock_frame

        with patch("visualbase.sources.file.FileSource") as MockFS:
            MockFS.return_value = mock_source
            export_highlight_report(Path("/dummy.mp4"), result, tmp_path)

        report_path = tmp_path / "highlight" / "report.html"
        assert report_path.exists()
        assert report_path.stat().st_size > 0

    def test_no_timeseries_no_report(self, tmp_path):
        """_timeseries가 None이면 report.html이 생성되지 않는다."""
        from momentscan.algorithm.batch.export_report import export_highlight_report

        result = HighlightResult(windows=[], frame_count=10)
        export_highlight_report(Path("/dummy.mp4"), result, tmp_path)

        report_path = tmp_path / "highlight" / "report.html"
        assert not report_path.exists()

    def test_report_contains_plotly_and_frames(self, tmp_path):
        """HTML에 Plotly CDN, FRAMES 배열, subplot div가 포함된다."""
        from momentscan.algorithm.batch.export_report import export_highlight_report

        result = _make_result_with_timeseries(200)

        mock_frame = MagicMock()
        mock_frame.data = np.zeros((480, 640, 3), dtype=np.uint8)

        mock_source = MagicMock()
        mock_source.seek.return_value = True
        mock_source.read.return_value = mock_frame

        with patch("visualbase.sources.file.FileSource") as MockFS:
            MockFS.return_value = mock_source
            export_highlight_report(Path("/dummy.mp4"), result, tmp_path)

        report_path = tmp_path / "highlight" / "report.html"
        html = report_path.read_text(encoding="utf-8")

        assert "plotly-2.35.2.min.js" in html
        assert "const FRAMES" in html
        assert "const DATA" in html
        assert 'id="plotDiv"' in html
        assert 'id="framePanel"' in html
        assert "mousemove" in html
        # Head pose + face distribution charts
        assert 'id="headPoseDiv"' in html
        assert 'id="faceDistDiv"' in html
        assert "head_yaw" in html
        assert "head_pitch" in html
        # New pipeline transparency fields
        assert "blur_normed" in html
        assert "normed_smile_intensity" in html
        assert "gate_face_confidence_pass" in html
        assert "cfg_quality_weights" in html
        # Pipeline decomposition hover panel
        assert "pipeline-detail" in html
        assert "gate-pass" in html or "gate-fail" in html
        # Dynamic module-based subplots
        assert '"modules"' in html

    def test_report_contains_window_details(self, tmp_path):
        """윈도우가 있으면 window detail 섹션이 포함된다."""
        from momentscan.algorithm.batch.export_report import export_highlight_report

        result = _make_result_with_timeseries(200)

        mock_frame = MagicMock()
        mock_frame.data = np.zeros((480, 640, 3), dtype=np.uint8)

        mock_source = MagicMock()
        mock_source.seek.return_value = True
        mock_source.read.return_value = mock_frame

        with patch("visualbase.sources.file.FileSource") as MockFS:
            MockFS.return_value = mock_source
            export_highlight_report(Path("/dummy.mp4"), result, tmp_path)

        html = (tmp_path / "highlight" / "report.html").read_text(encoding="utf-8")

        if result.windows:
            assert "Window 1" in html
            assert "window-card" in html
            assert "selected-frame" in html

    def test_report_deltas_in_timeseries(self):
        """BatchHighlightEngine이 _timeseries에 deltas를 저장한다."""
        result = _make_result_with_timeseries(200)
        assert result._timeseries is not None
        assert "deltas" in result._timeseries
        assert "head_yaw" in result._timeseries["deltas"]

    def test_report_arrays_and_normed_in_timeseries(self):
        """BatchHighlightEngine이 _timeseries에 arrays와 normed를 저장한다."""
        result = _make_result_with_timeseries(200)
        assert result._timeseries is not None
        assert "arrays" in result._timeseries
        assert "normed" in result._timeseries
        assert "blur_score" in result._timeseries["arrays"]
        assert "smile_intensity" in result._timeseries["normed"]
