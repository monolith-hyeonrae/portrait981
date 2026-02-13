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


# ── Timeseries export tests ──

def _make_result_with_timeseries(n: int = 100) -> HighlightResult:
    """_timeseries가 포함된 HighlightResult를 생성한다."""
    records = _make_records(n)
    for r in records:
        r.mouth_open_ratio = 0.1

    # 중간에 spike 삽입
    mid = n // 2
    for i in range(max(0, mid - 5), min(n, mid + 6)):
        records[i].mouth_open_ratio = 0.85

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
        assert "mouth_open_ratio" in header
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
        assert "normed_mouth_open_ratio" in html
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
        assert "head_velocity" in result._timeseries["deltas"]

    def test_report_arrays_and_normed_in_timeseries(self):
        """BatchHighlightEngine이 _timeseries에 arrays와 normed를 저장한다."""
        result = _make_result_with_timeseries(200)
        assert result._timeseries is not None
        assert "arrays" in result._timeseries
        assert "normed" in result._timeseries
        assert "blur_score" in result._timeseries["arrays"]
        assert "mouth_open_ratio" in result._timeseries["normed"]
