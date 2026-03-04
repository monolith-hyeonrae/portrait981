"""Unified tabbed HTML report — Timeline + Collection.

두 리포트의 데이터 빌더 함수를 재사용하여 단일 report.html에
Timeline 탭과 Collection 탭을 합친다.

- 탭 하나만 있으면 탭 바 숨기고 해당 콘텐츠만 표시
- 둘 다 있으면 탭 전환 UI 표시
- Collection 탭의 Plotly 차트는 deferred rendering (display:none 문제 회피)
"""

from __future__ import annotations

import base64
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2

logger = logging.getLogger(__name__)

THUMB_WIDTH = 320
THUMB_WIDTH_COLLECTION = 220
JPEG_QUALITY = 70


# ── Public API ──


def export_report(
    video_path: Path,
    *,
    highlight_result=None,
    collection_result=None,
    collection_records: Optional[List] = None,
    output_dir: Path,
) -> None:
    """통합 탭 리포트를 생성한다.

    Args:
        video_path: 원본 비디오 경로.
        highlight_result: BatchHighlightEngine 분석 결과 (HighlightResult).
        collection_result: CollectionEngine.collect() 결과 (CollectionResult).
        collection_records: 전체 CollectionRecord 리스트.
        output_dir: 출력 루트 디렉토리. report.html로 저장.
    """
    has_timeline = (
        highlight_result is not None
        and getattr(highlight_result, "_timeseries", None) is not None
    )
    has_collection = (
        collection_result is not None
        and collection_result.persons
    )

    if not has_timeline and not has_collection:
        logger.info("No data for report — skipping")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Build data for each tab ──
    timeline_data = None
    timeline_frames_b64: Dict[int, str] = {}
    timeline_html_parts: Dict[str, str] = {}

    collection_data = None
    collection_frames_b64: Dict[int, str] = {}
    collection_html_parts: Dict[str, str] = {}

    # Timeline data
    if has_timeline:
        from momentscan.algorithm.batch.export_report import (
            _build_chart_data,
            _build_summary_html as _build_timeline_summary_html,
            _build_window_detail_html,
            _build_config_table_html as _build_timeline_config_table_html,
            _build_field_reference_html,
            _select_thumbnail_indices,
        )

        ts = highlight_result._timeseries
        records = ts["records"]

        timeline_data = _build_chart_data(highlight_result)

        thumb_indices = _select_thumbnail_indices(
            n_frames=len(records),
            peaks=ts["peaks"].tolist(),
            windows=highlight_result.windows,
            records=records,
        )

        timeline_html_parts["summary"] = _build_timeline_summary_html(
            video_path.name, highlight_result,
        )
        timeline_html_parts["windows"] = _build_window_detail_html(
            highlight_result, {},  # frames filled later
        )
        timeline_html_parts["config"] = _build_timeline_config_table_html(
            highlight_result,
        )
        timeline_html_parts["field_ref"] = _build_field_reference_html()
    else:
        thumb_indices = []

    # Collection data
    collection_thumb_indices: List[int] = []
    if has_collection:
        from momentscan.algorithm.collection.export_report import (
            _build_persons_data,
            _build_summary_html as _build_collection_summary_html,
            _build_gallery_html,
            _build_config_table_html as _build_collection_config_table_html,
            _collect_all_selected,
        )

        all_selected = _collect_all_selected(collection_result)
        collection_thumb_indices = sorted({f.frame_idx for f in all_selected})

        persons_data = _build_persons_data(collection_result, collection_records)
        collection_data = {"persons": persons_data}

        collection_html_parts["summary"] = _build_collection_summary_html(
            video_path.name, collection_result, collection_records,
        )
        collection_html_parts["config"] = _build_collection_config_table_html(
            collection_result,
        )

    # ── Single-pass thumbnail extraction ──
    timeline_frames_b64, collection_frames_b64 = _extract_thumbnails_unified(
        video_path,
        highlight_result=highlight_result if has_timeline else None,
        timeline_indices=thumb_indices,
        collection_records=collection_records if has_collection else None,
        collection_frame_indices=collection_thumb_indices,
    )

    # Re-build window detail with actual thumbnails
    if has_timeline:
        from momentscan.algorithm.batch.export_report import (
            _build_window_detail_html,
        )
        timeline_html_parts["windows"] = _build_window_detail_html(
            highlight_result, timeline_frames_b64,
        )

    # Build collection gallery HTML (needs thumbnails)
    if has_collection:
        from momentscan.algorithm.collection.export_report import (
            _build_gallery_html,
        )
        collection_galleries: Dict[int, str] = {}
        for pid, person in collection_result.persons.items():
            collection_galleries[pid] = _build_gallery_html(
                person, collection_frames_b64,
            )
        collection_html_parts["galleries"] = collection_galleries

    # ── Assemble HTML ──
    html = _build_unified_html(
        video_name=video_path.name,
        has_timeline=has_timeline,
        has_collection=has_collection,
        timeline_data=timeline_data,
        timeline_frames_b64=timeline_frames_b64,
        timeline_html_parts=timeline_html_parts,
        collection_data=collection_data,
        collection_result=collection_result,
        collection_html_parts=collection_html_parts,
    )

    report_path = output_dir / "report.html"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html)

    size_mb = report_path.stat().st_size / (1024 * 1024)
    logger.info("Exported unified report: %s (%.1f MB)", report_path, size_mb)


# ── Unified thumbnail extraction ──


def _extract_thumbnails_unified(
    video_path: Path,
    *,
    highlight_result=None,
    timeline_indices: List[int],
    collection_records: Optional[List] = None,
    collection_frame_indices: List[int],
) -> tuple:
    """단일 비디오 패스로 Timeline + Collection 썸네일을 모두 추출.

    Returns:
        (timeline_b64, collection_b64) — 각각 {index: base64} dict.
    """
    from visualbase.sources.file import FileSource

    # Timeline: point_index → timestamp_ms
    timeline_targets: Dict[int, float] = {}
    if highlight_result is not None and highlight_result._timeseries:
        ts_records = highlight_result._timeseries["records"]
        for pidx in timeline_indices:
            if 0 <= pidx < len(ts_records):
                timeline_targets[pidx] = ts_records[pidx].timestamp_ms

    # Collection: frame_idx → timestamp_ms
    collection_targets: Dict[int, float] = {}
    if collection_records:
        fidx_to_ts = {r.frame_idx: r.timestamp_ms for r in collection_records}
        for fidx in collection_frame_indices:
            if fidx in fidx_to_ts:
                collection_targets[fidx] = fidx_to_ts[fidx]

    # Merge all targets: (key, timestamp_ms, source, thumb_width)
    all_jobs = []
    for pidx, ts_ms in timeline_targets.items():
        all_jobs.append((pidx, ts_ms, "timeline", THUMB_WIDTH))
    for fidx, ts_ms in collection_targets.items():
        all_jobs.append((fidx, ts_ms, "collection", THUMB_WIDTH_COLLECTION))

    if not all_jobs:
        return {}, {}

    # Sort by timestamp for forward-seek optimization
    all_jobs.sort(key=lambda x: x[1])

    timeline_b64: Dict[int, str] = {}
    collection_b64: Dict[int, str] = {}

    source = FileSource(str(video_path))
    source.open()

    try:
        for key, ts_ms, src, tw in all_jobs:
            t_ns = int(ts_ms * 1_000_000)
            if not source.seek(t_ns):
                continue

            frame = source.read()
            if frame is None:
                continue

            img = frame.data
            h, w = img.shape[:2]
            new_w = tw
            new_h = int(h * new_w / w)
            thumb = cv2.resize(img, (new_w, new_h))

            _, buf = cv2.imencode(
                ".jpg", thumb, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY],
            )
            b64 = base64.b64encode(buf).decode("ascii")

            if src == "timeline":
                timeline_b64[key] = b64
            else:
                collection_b64[key] = b64
    finally:
        source.close()

    logger.info(
        "Extracted thumbnails: %d timeline + %d collection",
        len(timeline_b64), len(collection_b64),
    )
    return timeline_b64, collection_b64


# ── HTML assembly ──


_TAB_CSS = """
  .tab-bar {
    display: flex; gap: 0; margin: 32px 0 0;
    border-bottom: 2px solid #e8e8e8;
  }
  .tab-btn {
    padding: 10px 24px;
    font-size: 0.82em; font-weight: 600; letter-spacing: 0.04em;
    text-transform: uppercase;
    color: #999; background: none; border: none;
    cursor: pointer; position: relative;
    transition: color 0.15s;
  }
  .tab-btn:hover { color: #666; }
  .tab-btn.active {
    color: #111;
  }
  .tab-btn.active::after {
    content: '';
    position: absolute; bottom: -2px; left: 0; right: 0;
    height: 2px; background: #111;
  }
  .tab-content { display: none; padding-top: 8px; }
  .tab-content.active { display: block; }
"""

# Timeline-specific CSS (from batch/export_report.py)
_TIMELINE_CSS = """
  .main-layout {
    display: flex; gap: 32px; align-items: flex-start;
  }
  .chart-column { flex: 1; min-width: 0; }
  .frame-panel {
    position: sticky; top: 32px;
    width: 340px; flex-shrink: 0;
    background: #fff; border: 1px solid #e8e8e8;
    border-radius: 6px; padding: 12px;
  }
  .frame-panel img {
    width: 100%; border-radius: 3px; background: #f0f0f0;
  }
  .frame-panel .meta {
    margin-top: 10px; font-size: 0.82em; line-height: 1.7;
    font-variant-numeric: tabular-nums;
  }
  .frame-panel .meta span { color: #999; }
  .frame-panel .meta strong { color: #111; font-weight: 600; }
  .frame-panel .pipeline-detail {
    font-family: 'SF Mono', 'Fira Code', 'Consolas', monospace;
    font-size: 0.75em; line-height: 1.6;
    white-space: pre; margin-top: 8px; color: #444;
  }
  .gate-pass { color: #2e7d32; }
  .gate-warn { color: #e65100; }
  .gate-fail { color: #c62828; font-weight: 600; }
  .section-label { color: #999; font-weight: 600; }
  .window-card {
    background: #fff; border: 1px solid #e8e8e8;
    border-radius: 6px; margin: 12px 0; overflow: hidden;
  }
  .window-header {
    padding: 12px 16px; border-bottom: 1px solid #f0f0f0;
    font-weight: 600; font-size: 0.9em;
    display: flex; justify-content: space-between; color: #333;
  }
  .window-score { color: #111; font-variant-numeric: tabular-nums; }
  .window-body { padding: 14px 16px; }
  .window-meta { color: #999; font-size: 0.82em; margin-bottom: 10px; }
  .reason-chart { margin: 10px 0; }
  .reason-row {
    display: flex; align-items: center; gap: 8px;
    margin: 4px 0; font-size: 0.82em;
  }
  .reason-name {
    width: 140px; color: #666; text-align: right;
    font-variant-numeric: tabular-nums;
  }
  .reason-bar {
    height: 6px; background: #111; border-radius: 3px; min-width: 2px;
  }
  .reason-val {
    color: #999; font-size: 0.8em; width: 50px;
    font-variant-numeric: tabular-nums;
  }
  .field-ref-table td { font-size: 0.85em; }
  .field-ref-table code {
    background: #f5f5f5; padding: 1px 5px; border-radius: 3px;
    font-size: 0.9em; color: #333;
  }
  .rationale-cell { color: #555; font-size: 0.82em; max-width: 300px; }
  .role-badge {
    display: inline-block; padding: 1px 7px; border-radius: 3px;
    font-size: 0.72em; font-weight: 600; letter-spacing: 0.03em;
    text-transform: uppercase;
  }
  .role-gate { background: #e8f5e9; color: #2e7d32; }
  .role-quality { background: #e3f2fd; color: #1565c0; }
  .role-impact { background: #fbe9e7; color: #d84315; }
  .role-bonus { background: #f3e5f5; color: #6a1b9a; }
  .role-info { background: #f5f5f5; color: #999; }
  .selected-frames {
    display: flex; gap: 10px; margin-top: 12px;
    overflow-x: auto; padding-bottom: 4px;
  }
  .selected-frame {
    flex-shrink: 0; width: 140px;
    background: #fafafa; border: 1px solid #e8e8e8;
    border-radius: 4px; overflow: hidden;
  }
  .selected-frame img { width: 100%; display: block; }
  .thumb-placeholder { width: 100%; height: 80px; background: #f0f0f0; }
  .sf-meta {
    padding: 6px 8px; font-size: 0.78em; line-height: 1.5; color: #666;
    font-variant-numeric: tabular-nums;
  }
  .sf-meta strong { color: #333; }
  #plotDiv { width: 100%; }
"""

# Collection-specific CSS (from collection/export_report.py)
_COLLECTION_CSS = """
  .person-summary {
    background: #fff; border: 1px solid #e8e8e8;
    border-radius: 6px; padding: 16px 20px; margin: 12px 0;
  }
  .person-title {
    font-weight: 700; font-size: 1em; color: #111; margin-bottom: 8px;
  }
  .person-stats { display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 8px; }
  .badge {
    display: inline-block; padding: 2px 10px; border-radius: 12px;
    font-size: 0.78em; font-weight: 600;
  }
  .badge.grid { background: #e8f5e9; color: #2e7d32; }
  .badge.cells { background: #e3f2fd; color: #1565c0; }
  .badge.mode { background: #f5f5f5; color: #666; }
  .person-coverage {
    color: #888; font-size: 0.82em;
    font-variant-numeric: tabular-nums;
  }
  h4.gallery-title {
    font-size: 0.85em; font-weight: 600; margin: 24px 0 12px;
    padding: 6px 12px; border-radius: 4px; display: inline-block;
  }
  h4.grid { background: #e8f5e9; color: #2e7d32; }
  .gallery-grid {
    display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 16px;
  }
  .frame-card {
    width: 180px; background: #fff; border: 1px solid #e8e8e8;
    border-radius: 6px; overflow: hidden;
    transition: box-shadow 0.15s;
  }
  .frame-card:hover { box-shadow: 0 2px 12px rgba(0,0,0,0.08); }
  .frame-card.grid { border-left: 3px solid #2e7d32; }
  .frame-card img { width: 100%; display: block; }
  .frame-meta {
    padding: 6px 8px; font-size: 0.82em;
    display: flex; justify-content: space-between; color: #333;
    border-bottom: 1px solid #f0f0f0;
  }
  .frame-meta strong { color: #111; }
  .frame-meta span { color: #999; }
  .frame-detail {
    padding: 6px 8px; font-size: 0.72em; line-height: 1.5;
    color: #666; font-variant-numeric: tabular-nums;
  }
  .bucket-label {
    font-family: 'SF Mono', 'Fira Code', 'Consolas', monospace;
    font-size: 0.9em; color: #444; margin-bottom: 2px;
  }
"""

# Shared CSS (both reports use these)
_SHARED_CSS = """
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Helvetica Neue', sans-serif;
    background: #fafafa; color: #1a1a1a;
    padding: 48px 40px; max-width: 1400px; margin: 0 auto;
    font-size: 14px; line-height: 1.6;
    -webkit-font-smoothing: antialiased;
  }
  h1 {
    font-size: 1.6em; font-weight: 700; letter-spacing: -0.03em;
    color: #111; margin-bottom: 2px;
  }
  h2 {
    font-size: 0.75em; font-weight: 600; letter-spacing: 0.08em;
    text-transform: uppercase; color: #999;
    margin: 48px 0 16px; padding-bottom: 8px;
    border-bottom: 1px solid #e8e8e8;
  }
  .subtitle {
    color: #999; font-size: 0.9em; margin-bottom: 8px; font-weight: 400;
  }
  .summary-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
    gap: 1px; margin: 16px 0;
    background: #e8e8e8; border: 1px solid #e8e8e8;
    border-radius: 6px; overflow: hidden;
  }
  .stat-card { background: #fff; padding: 14px 16px; }
  .stat-card .label {
    color: #999; font-size: 0.7em; font-weight: 500;
    text-transform: uppercase; letter-spacing: 0.05em;
  }
  .stat-card .value {
    font-size: 1.5em; font-weight: 700; margin-top: 2px;
    letter-spacing: -0.02em; color: #111;
  }
  .config-table {
    width: 100%; border-collapse: collapse; margin: 12px 0; font-size: 0.82em;
  }
  .config-table th {
    text-align: left; padding: 8px 12px;
    border-bottom: 2px solid #e0e0e0; color: #999;
    font-weight: 600; font-size: 0.9em;
    text-transform: uppercase; letter-spacing: 0.04em;
  }
  .config-table td {
    padding: 6px 12px; border-bottom: 1px solid #f0f0f0;
    font-variant-numeric: tabular-nums;
  }
  .config-table tr:hover { background: #fafafa; }
  .chart-row { display: flex; gap: 24px; margin: 16px 0; }
  .chart-row > div { flex: 1; min-width: 0; }
  .muted { color: #bbb; font-style: italic; }
  .footer {
    margin-top: 56px; text-align: center;
    color: #ccc; font-size: 0.75em; letter-spacing: 0.02em;
  }
  @media (max-width: 900px) {
    body { padding: 24px 16px; }
    .main-layout { flex-direction: column; }
    .frame-panel { position: static; width: 100%; }
    .chart-row { flex-direction: column; }
    .gallery-grid { justify-content: center; }
  }
"""

# Tab switching JS
_TAB_JS = r"""
document.querySelectorAll('.tab-btn').forEach(function(btn) {
  btn.addEventListener('click', function() {
    var target = this.dataset.tab;
    document.querySelectorAll('.tab-content').forEach(function(el) {
      el.style.display = 'none';
      el.classList.remove('active');
    });
    document.querySelectorAll('.tab-btn').forEach(function(el) {
      el.classList.remove('active');
    });
    document.getElementById('tab-' + target).style.display = 'block';
    document.getElementById('tab-' + target).classList.add('active');
    this.classList.add('active');
    if (target === 'collection' && !window._collectionRendered) {
      window._collectionRendered = true;
      if (typeof renderCollectionCharts === 'function') renderCollectionCharts();
    }
  });
});
"""


def _build_unified_html(
    video_name: str,
    *,
    has_timeline: bool,
    has_collection: bool,
    timeline_data: Optional[Dict[str, Any]],
    timeline_frames_b64: Dict[int, str],
    timeline_html_parts: Dict[str, str],
    collection_data: Optional[Dict[str, Any]],
    collection_result,
    collection_html_parts: Dict[str, str],
) -> str:
    """완전한 통합 HTML 문서를 조립한다."""
    safe_name = video_name.replace("&", "&amp;").replace("<", "&lt;")
    show_tabs = has_timeline and has_collection

    # CSS
    css = _SHARED_CSS
    if has_timeline:
        css += _TIMELINE_CSS
    if has_collection:
        css += _COLLECTION_CSS
    if show_tabs:
        css += _TAB_CSS

    # ── Timeline tab content ──
    timeline_tab_html = ""
    if has_timeline:
        timeline_tab_html = (
            '  <h2>Summary</h2>\n'
            + timeline_html_parts["summary"] + '\n'
            '  <h2>Timeseries Analysis</h2>\n'
            '  <div class="main-layout">\n'
            '    <div class="chart-column"><div id="plotDiv"></div></div>\n'
            '    <div class="frame-panel" id="framePanel">\n'
            '      <img id="frameImg" src="data:image/gif;base64,'
            'R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7" alt="hover frame">\n'
            '      <div class="meta" id="frameMeta">'
            '<div>Hover over the chart to see frame</div></div>\n'
            '      <div class="pipeline-detail" id="pipelineDetail"></div>\n'
            '    </div>\n'
            '  </div>\n'
            '  <h2>Head Pose &amp; Face Position</h2>\n'
            '  <div class="chart-row">\n'
            '    <div id="headPoseDiv"></div>\n'
            '    <div id="faceDistDiv"></div>\n'
            '  </div>\n'
            '  <h2>Highlight Windows</h2>\n'
            + timeline_html_parts["windows"] + '\n'
            '  <h2>Field Reference</h2>\n'
            + timeline_html_parts["field_ref"] + '\n'
            '  <h2>Configuration</h2>\n'
            + timeline_html_parts["config"] + '\n'
        )

    # ── Collection tab content ──
    collection_tab_html = ""
    if has_collection:
        persons_html = ""
        galleries = collection_html_parts.get("galleries", {})
        for pid, person in collection_result.persons.items():
            gallery_html = galleries.get(pid, "")
            # Check if PCA data is available
            has_pca = "pca" in (collection_data or {}).get("persons", {}).get(str(pid), collection_data.get("persons", {}).get(pid, {}))
            pca_section = ""
            if has_pca:
                pca_section = (
                    f'  <h2>Person {pid} &mdash; Signal Space</h2>\n'
                    f'  <div id="pca_{pid}"></div>\n'
                )
            persons_html += (
                f'<div class="person-section">\n'
                f'  <h2>Person {pid} &mdash; Pose Distribution</h2>\n'
                f'  <div class="chart-row">\n'
                f'    <div id="scatter_{pid}"></div>\n'
                f'    <div id="heatmap_{pid}"></div>\n'
                f'  </div>\n'
                f'  <h2>Person {pid} &mdash; Category Distribution</h2>\n'
                f'  <div id="category_{pid}"></div>\n'
                + pca_section +
                f'  <h2>Person {pid} &mdash; Selected Frames</h2>\n'
                f'  {gallery_html}\n'
                f'</div>\n'
            )

        collection_tab_html = (
            '  <h2>Summary</h2>\n'
            + collection_html_parts["summary"] + '\n'
            + persons_html
            + '  <h2>Configuration</h2>\n'
            + collection_html_parts["config"] + '\n'
        )

    # ── Body HTML ──
    body_html = f'  <h1>MomentScan Report</h1>\n  <p class="subtitle">{safe_name}</p>\n'

    if show_tabs:
        body_html += (
            '  <div class="tab-bar">\n'
            '    <button class="tab-btn active" data-tab="timeline">Timeline</button>\n'
            '    <button class="tab-btn" data-tab="collection">Collection</button>\n'
            '  </div>\n'
            '  <div id="tab-timeline" class="tab-content active">\n'
            + timeline_tab_html
            + '  </div>\n'
            '  <div id="tab-collection" class="tab-content">\n'
            + collection_tab_html
            + '  </div>\n'
        )
    elif has_timeline:
        body_html += timeline_tab_html
    elif has_collection:
        body_html += collection_tab_html

    body_html += '  <div class="footer">Generated by MomentScan</div>\n'

    # ── JavaScript ──
    # Import JS from existing modules (as raw strings)
    from momentscan.algorithm.batch.export_report import _JS_MAIN as _TIMELINE_JS
    from momentscan.algorithm.collection.export_report import _JS_MAIN as _COLLECTION_JS

    script_parts = []

    # Data injection using namespaced REPORT object
    report_obj: Dict[str, Any] = {}
    if has_timeline:
        report_obj["timeline"] = {
            "data": "__TIMELINE_DATA__",
            "frames": "__TIMELINE_FRAMES__",
        }
    if has_collection:
        report_obj["collection"] = {
            "data": "__COLLECTION_DATA__",
        }

    # We build the script manually to inject large JSON without double-serializing
    script_parts.append("<script>")

    # Inject data
    if has_timeline:
        timeline_data_json = json.dumps(timeline_data)
        timeline_frames_json = json.dumps(timeline_frames_b64)
        script_parts.append(
            f"var REPORT_TIMELINE_DATA = {timeline_data_json};\n"
            f"var REPORT_TIMELINE_FRAMES = {timeline_frames_json};\n"
        )

    if has_collection:
        collection_data_json = json.dumps(collection_data)
        script_parts.append(
            f"var REPORT_COLLECTION_DATA = {collection_data_json};\n"
        )

    # Timeline JS (IIFE with DATA/FRAMES aliases)
    if has_timeline:
        script_parts.append(
            "(function() {\n"
            "var DATA = REPORT_TIMELINE_DATA;\n"
            "var FRAMES = REPORT_TIMELINE_FRAMES;\n"
            + _TIMELINE_JS
            + "\n})();\n"
        )

    # Collection JS — deferred rendering for Plotly display:none issue
    if has_collection:
        if show_tabs:
            # Deferred: render when tab is first activated
            script_parts.append(
                "function renderCollectionCharts() {\n"
                "var DATA = REPORT_COLLECTION_DATA;\n"
                + _COLLECTION_JS
                + "\n}\n"
            )
        else:
            # No tabs — render immediately
            script_parts.append(
                "(function() {\n"
                "var DATA = REPORT_COLLECTION_DATA;\n"
                + _COLLECTION_JS
                + "\n})();\n"
            )

    # Tab switching JS
    if show_tabs:
        script_parts.append(_TAB_JS)

    script_parts.append("</script>")

    return (
        '<!DOCTYPE html>\n'
        '<html lang="ko">\n<head>\n'
        '<meta charset="UTF-8">\n'
        '<meta name="viewport" content="width=device-width,initial-scale=1">\n'
        '<title>MomentScan Report \u2014 ' + safe_name + '</title>\n'
        '<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>\n'
        '<style>' + css + '</style>\n'
        '<link rel="preconnect" href="https://fonts.googleapis.com">\n'
        '<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">\n'
        '</head>\n<body>\n'
        + body_html
        + "\n".join(script_parts)
        + '\n</body>\n</html>'
    )
