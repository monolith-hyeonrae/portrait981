"""momentscan-report 독립 테스트.

momentscan-report 패키지의 3개 public API를 검증:
- export_highlight_report: 타임라인 HTML
- export_collection_report: 컬렉션 갤러리 HTML
- export_report: 통합 탭 HTML
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from momentscan.algorithm.batch.highlight import BatchHighlightEngine
from momentscan.algorithm.batch.types import FrameRecord, HighlightResult
from momentscan.algorithm.collection.types import (
    CollectionRecord,
    CollectionResult,
    PersonCollection,
    SelectedFrame,
)

from momentscan_report import (
    export_collection_report,
    export_highlight_report,
    export_report,
)


# ── Helpers ──


def _make_records(n: int = 100, *, fps: float = 10.0) -> list[FrameRecord]:
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


def _make_result_with_timeseries(n: int = 200) -> HighlightResult:
    records = _make_records(n)
    for r in records:
        r.smile_intensity = 0.1
    mid = n // 2
    for i in range(max(0, mid - 5), min(n, mid + 6)):
        records[i].smile_intensity = 0.85
    engine = BatchHighlightEngine()
    return engine.analyze(records)


def _mock_file_source():
    """FileSource mock: seek → True, read → 480×640 black frame."""
    mock_frame = MagicMock()
    mock_frame.data = np.zeros((480, 640, 3), dtype=np.uint8)
    mock_source = MagicMock()
    mock_source.seek.return_value = True
    mock_source.read.return_value = mock_frame
    return mock_source


def _make_collection_result() -> CollectionResult:
    sf1 = SelectedFrame(
        frame_idx=10,
        timestamp_ms=1000.0,
        pose_name="frontal",
        pivot_name="warm_smile",
        cell_key="frontal|warm_smile",
        quality_score=0.8,
        cell_score=0.75,
    )
    sf2 = SelectedFrame(
        frame_idx=50,
        timestamp_ms=5000.0,
        pose_name="left30",
        pivot_name="neutral",
        cell_key="left30|neutral",
        quality_score=0.7,
        cell_score=0.65,
    )
    person = PersonCollection(
        person_id=1,
        prototype_frame_idx=10,
        grid={
            "frontal|warm_smile": [sf1],
            "left30|neutral": [sf2],
        },
        pose_coverage={"frontal": 1, "left30": 1},
        category_coverage={"warm_smile": 1, "neutral": 1},
    )
    return CollectionResult(
        persons={1: person},
        frame_count=100,
    )


def _make_collection_records(n: int = 50) -> list[CollectionRecord]:
    return [
        CollectionRecord(
            frame_idx=i,
            timestamp_ms=i * 100.0,
            face_detected=True,
            face_confidence=0.9,
            head_yaw=float(i % 30 - 15),
            head_pitch=0.0,
            smile_intensity=0.3 if i % 3 == 0 else 0.1,
            gate_passed=True,
        )
        for i in range(n)
    ]


# ── export_highlight_report ──


class TestExportHighlightReport:
    def test_report_created(self, tmp_path):
        result = _make_result_with_timeseries(200)
        with patch("visualbase.sources.file.FileSource") as MockFS:
            MockFS.return_value = _mock_file_source()
            export_highlight_report(Path("/dummy.mp4"), result, tmp_path)

        report = tmp_path / "highlight" / "report.html"
        assert report.exists()
        assert report.stat().st_size > 0

    def test_no_timeseries_skips(self, tmp_path):
        result = HighlightResult(windows=[], frame_count=10)
        export_highlight_report(Path("/dummy.mp4"), result, tmp_path)

        assert not (tmp_path / "highlight" / "report.html").exists()

    def test_html_contains_plotly_and_data(self, tmp_path):
        result = _make_result_with_timeseries(200)
        with patch("visualbase.sources.file.FileSource") as MockFS:
            MockFS.return_value = _mock_file_source()
            export_highlight_report(Path("/dummy.mp4"), result, tmp_path)

        html = (tmp_path / "highlight" / "report.html").read_text(encoding="utf-8")
        assert "plotly" in html.lower()
        assert "const FRAMES" in html
        assert "const DATA" in html
        assert 'id="plotDiv"' in html
        assert 'id="framePanel"' in html


# ── export_collection_report ──


class TestExportCollectionReport:
    def test_report_created(self, tmp_path):
        result = _make_collection_result()
        records = _make_collection_records()
        with patch("visualbase.sources.file.FileSource") as MockFS:
            MockFS.return_value = _mock_file_source()
            export_collection_report(
                Path("/dummy.mp4"), result, records, tmp_path,
            )

        report = tmp_path / "collection_report.html"
        assert report.exists()
        assert report.stat().st_size > 0

    def test_empty_persons_skips(self, tmp_path):
        result = CollectionResult(persons={}, frame_count=0)
        records = []
        export_collection_report(
            Path("/dummy.mp4"), result, records, tmp_path,
        )
        assert not (tmp_path / "collection_report.html").exists()

    def test_html_contains_person_data(self, tmp_path):
        result = _make_collection_result()
        records = _make_collection_records()
        with patch("visualbase.sources.file.FileSource") as MockFS:
            MockFS.return_value = _mock_file_source()
            export_collection_report(
                Path("/dummy.mp4"), result, records, tmp_path,
            )

        html = (tmp_path / "collection_report.html").read_text(encoding="utf-8")
        assert "Person" in html or "person" in html
        assert "frontal" in html or "pose" in html.lower()


# ── export_report (unified) ──


class TestExportReport:
    def test_highlight_only(self, tmp_path):
        result = _make_result_with_timeseries(200)
        with patch("visualbase.sources.file.FileSource") as MockFS:
            MockFS.return_value = _mock_file_source()
            export_report(
                Path("/dummy.mp4"),
                highlight_result=result,
                output_dir=tmp_path,
            )

        report = tmp_path / "report.html"
        assert report.exists()
        assert report.stat().st_size > 0

    def test_collection_only(self, tmp_path):
        col_result = _make_collection_result()
        records = _make_collection_records()
        with patch("visualbase.sources.file.FileSource") as MockFS:
            MockFS.return_value = _mock_file_source()
            export_report(
                Path("/dummy.mp4"),
                collection_result=col_result,
                collection_records=records,
                output_dir=tmp_path,
            )

        report = tmp_path / "report.html"
        assert report.exists()

    def test_both_results(self, tmp_path):
        hl_result = _make_result_with_timeseries(200)
        col_result = _make_collection_result()
        records = _make_collection_records()
        with patch("visualbase.sources.file.FileSource") as MockFS:
            MockFS.return_value = _mock_file_source()
            export_report(
                Path("/dummy.mp4"),
                highlight_result=hl_result,
                collection_result=col_result,
                collection_records=records,
                output_dir=tmp_path,
            )

        report = tmp_path / "report.html"
        assert report.exists()
        html = report.read_text(encoding="utf-8")
        # Unified report should have tab UI
        assert "tab" in html.lower() or "Timeline" in html

    def test_no_data_skips(self, tmp_path):
        export_report(
            Path("/dummy.mp4"),
            output_dir=tmp_path,
        )
        assert not (tmp_path / "report.html").exists()


# ── Import test ──


class TestPackageImports:
    def test_public_api_importable(self):
        import momentscan_report

        assert hasattr(momentscan_report, "export_report")
        assert hasattr(momentscan_report, "export_highlight_report")
        assert hasattr(momentscan_report, "export_collection_report")

    def test_submodule_imports(self):
        from momentscan_report.highlight import export_highlight_report as f1
        from momentscan_report.collection import export_collection_report as f2
        from momentscan_report.unified import export_report as f3

        assert callable(f1)
        assert callable(f2)
        assert callable(f3)
