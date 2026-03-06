"""Interactive collection analysis HTML report.

Plotly 기반 인터랙티브 차트 + 프레임 썸네일 갤러리.
단일 .html 파일에 프레임 썸네일을 base64 임베드하여 자체 완결.

시각화:
- Summary: person별 grid 셀 수, pose/category coverage
- Yaw x Pitch scatter: 전체 레코드 + 선택된 프레임 (category별 컬러)
- Pose × Category coverage heatmap
- Category distribution: bar chart
- 10D Signal Space (PCA 2D): 시그널 벡터 차원 축소 scatter
- Frame thumbnail gallery: cell_key 그룹별
- Config table: CollectionConfig 파라미터
"""

from __future__ import annotations

import base64
import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from momentscan.algorithm.collection.types import (
    CollectionConfig,
    CollectionRecord,
    CollectionResult,
    PersonCollection,
    SelectedFrame,
)

logger = logging.getLogger(__name__)

THUMB_WIDTH = 220
JPEG_QUALITY = 70

# Category colors for consistent coloring
_CATEGORY_COLORS = ['#2e7d32', '#1565c0', '#e65100', '#6a1b9a', '#c62828',
                     '#00695c', '#4527a0', '#bf360c', '#1b5e20', '#0d47a1']


# ── Public API ──


def export_collection_report(
    video_path: Path,
    result: CollectionResult,
    records: List[CollectionRecord],
    output_dir: Path,
    *,
    highlights: Optional[List] = None,
) -> None:
    """인터랙티브 collection 분석 HTML 리포트를 생성한다.

    Args:
        video_path: 원본 비디오 경로.
        result: CollectionEngine.collect() 결과.
        records: 전체 CollectionRecord (산점도용).
        output_dir: 출력 루트 디렉토리. collection_report.html로 저장.
        highlights: Optional highlight windows from BatchHighlightEngine.
    """
    if not result.persons:
        logger.info("No persons — skipping collection HTML report")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect all selected frames for thumbnail extraction
    all_selected = _collect_all_selected(result)
    frames_b64 = _extract_thumbnails(video_path, records, all_selected)

    # Build per-person chart data
    persons_data = _build_persons_data(result, records)

    # Build HTML
    html = _build_html(
        video_name=video_path.name,
        result=result,
        records=records,
        persons_data=persons_data,
        frames_b64=frames_b64,
    )

    report_path = output_dir / "collection_report.html"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html)

    size_mb = report_path.stat().st_size / (1024 * 1024)
    logger.info("Exported collection report: %s (%.1f MB)", report_path, size_mb)


# ── Thumbnail extraction ──


def _collect_all_selected(result: CollectionResult) -> List[SelectedFrame]:
    """Collect all selected frames across persons."""
    frames = []
    for person in result.persons.values():
        frames.extend(person.all_frames())
    return frames


def _extract_thumbnails(
    video_path: Path,
    records: List[CollectionRecord],
    selected_frames: List[SelectedFrame],
) -> Dict[int, str]:
    """선택된 프레임의 썸네일을 추출하여 base64 인코딩.

    Returns:
        {frame_idx: base64_jpeg_string} 매핑.
    """
    from visualbase.sources.file import FileSource

    fidx_to_ts = {r.frame_idx: r.timestamp_ms for r in records}

    target_indices = sorted({f.frame_idx for f in selected_frames})
    targets = [(idx, fidx_to_ts.get(idx, 0.0)) for idx in target_indices]
    targets.sort(key=lambda x: x[1])

    frames_b64: Dict[int, str] = {}
    source = FileSource(str(video_path))
    source.open()

    try:
        for idx, ts_ms in targets:
            t_ns = int(ts_ms * 1_000_000)
            if not source.seek(t_ns):
                continue

            frame = source.read()
            if frame is None:
                continue

            img = frame.data
            h, w = img.shape[:2]
            new_w = THUMB_WIDTH
            new_h = int(h * new_w / w)
            thumb = cv2.resize(img, (new_w, new_h))

            _, buf = cv2.imencode(
                ".jpg", thumb, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
            )
            frames_b64[idx] = base64.b64encode(buf).decode("ascii")
    finally:
        source.close()

    logger.info(
        "Extracted %d/%d frame thumbnails for collection report",
        len(frames_b64), len(target_indices),
    )
    return frames_b64


# ── PCA computation ──


def _compute_pca_2d(
    records: List[CollectionRecord],
    selected: List[SelectedFrame],
) -> Optional[Dict[str, Any]]:
    """10D 시그널 벡터 → PCA 2D 프로젝션 데이터 (Plotly용).

    numpy.linalg.svd 기반 — scikit-learn 불필요.

    Returns:
        PCA projection data dict, or None if insufficient data.
    """
    try:
        from momentscan.algorithm.batch.catalog_scoring import (
            SIGNAL_FIELDS,
            extract_signal_vector,
        )
        from momentscan.algorithm.batch.types import FrameRecord
    except ImportError:
        return None

    if len(records) < 3:
        return None

    # Build signal vectors for all records
    def _record_to_vec(r: CollectionRecord) -> np.ndarray:
        vec = np.zeros(len(SIGNAL_FIELDS), dtype=np.float64)
        for i, field in enumerate(SIGNAL_FIELDS):
            vec[i] = float(getattr(r, field, 0.0))
        return vec

    vectors = np.array([_record_to_vec(r) for r in records])

    # Center + SVD
    mean = vectors.mean(axis=0)
    centered = vectors - mean
    try:
        U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
        return None

    pc2 = Vt[:2]  # (2, D)
    projected = centered @ pc2.T  # (N, 2)

    # Explained variance ratio
    total_var = (S ** 2).sum()
    explained = []
    for i in range(min(2, len(S))):
        explained.append(float((S[i] ** 2) / total_var) if total_var > 0 else 0.0)
    while len(explained) < 2:
        explained.append(0.0)

    # Map frame_idx → record index
    fidx_to_ridx = {r.frame_idx: i for i, r in enumerate(records)}

    # Selected frames by category
    selected_by_cat: Dict[str, Dict[str, list]] = {}
    for f in selected:
        cat = f.pivot_name or "other"
        if cat not in selected_by_cat:
            selected_by_cat[cat] = {"x": [], "y": [], "frame_idx": [], "cell_score": []}
        ridx = fidx_to_ridx.get(f.frame_idx)
        if ridx is not None:
            selected_by_cat[cat]["x"].append(float(projected[ridx, 0]))
            selected_by_cat[cat]["y"].append(float(projected[ridx, 1]))
            selected_by_cat[cat]["frame_idx"].append(f.frame_idx)
            selected_by_cat[cat]["cell_score"].append(round(f.cell_score, 4))

    # Category centroids (mean of selected frames per category)
    centroids = {}
    for cat, data in selected_by_cat.items():
        if data["x"]:
            centroids[cat] = {
                "x": float(np.mean(data["x"])),
                "y": float(np.mean(data["y"])),
            }

    return {
        "all_x": [float(v) for v in projected[:, 0]],
        "all_y": [float(v) for v in projected[:, 1]],
        "selected": selected_by_cat,
        "centroids": centroids,
        "explained_variance": explained,
    }


# ── Data builders ──


def _build_persons_data(
    result: CollectionResult,
    records: List[CollectionRecord],
) -> Dict[int, Dict[str, Any]]:
    """Person별 차트 데이터를 구성한다."""
    data = {}
    for pid, person in result.persons.items():
        person_records = [r for r in records if r.person_id == pid]
        data[pid] = _build_person_data(person, person_records)
    return data


def _build_person_data(
    person: PersonCollection,
    records: List[CollectionRecord],
) -> Dict[str, Any]:
    """한 person의 차트 데이터."""
    record_map = {r.frame_idx: r for r in records}

    all_yaw = [r.head_yaw for r in records]
    all_pitch = [r.head_pitch for r in records]

    # Selected frames by category (for scatter coloring)
    all_frames = person.all_frames()
    categories = sorted({f.pivot_name or "other" for f in all_frames})

    scatter_selected: Dict[str, Dict[str, list]] = {}
    for cat in categories:
        cat_frames = [f for f in all_frames if (f.pivot_name or "other") == cat]
        scatter_selected[cat] = {
            "yaw": [
                round(record_map[f.frame_idx].head_yaw, 1)
                if f.frame_idx in record_map else 0.0
                for f in cat_frames
            ],
            "pitch": [
                round(record_map[f.frame_idx].head_pitch, 1)
                if f.frame_idx in record_map else 0.0
                for f in cat_frames
            ],
            "frame_idx": [f.frame_idx for f in cat_frames],
            "quality": [round(f.quality_score, 3) for f in cat_frames],
            "stable": [round(f.stable_score, 3) for f in cat_frames],
            "pose": [f.pose_name for f in cat_frames],
            "pivot": [f.pivot_name for f in cat_frames],
            "cell_score": [round(f.cell_score, 3) for f in cat_frames],
        }

    heatmap = _build_heatmap(person)
    cat_dist = _build_category_dist(person)

    # PCA data
    pca = _compute_pca_2d(records, all_frames)

    result_data = {
        "all_yaw": all_yaw,
        "all_pitch": all_pitch,
        "selected": scatter_selected,
        "heatmap": heatmap,
        "category_dist": cat_dist,
    }
    if pca is not None:
        result_data["pca"] = pca

    return result_data


def _build_heatmap(person: PersonCollection) -> Dict[str, Any]:
    """Pose × Category heatmap data (built from grid)."""
    all_frames = person.all_frames()

    # Extract unique poses and categories from grid
    poses = sorted({f.pose_name or "other" for f in all_frames})
    categories = sorted({f.pivot_name or "other" for f in all_frames})

    if not poses:
        poses = ["other"]
    if not categories:
        categories = ["other"]

    grid = [[0] * len(poses) for _ in range(len(categories))]

    for f in all_frames:
        pose = f.pose_name or "other"
        cat = f.pivot_name or "other"
        xi = poses.index(pose) if pose in poses else len(poses) - 1
        yi = categories.index(cat) if cat in categories else len(categories) - 1
        grid[yi][xi] += 1

    tooltip = [
        [
            str(grid[yi][xi]) if grid[yi][xi] > 0 else "empty"
            for xi in range(len(poses))
        ]
        for yi in range(len(categories))
    ]

    return {
        "z": grid,
        "x": poses,
        "y": categories,
        "tooltip": tooltip,
    }


def _build_category_dist(person: PersonCollection) -> Dict[str, int]:
    """Category distribution (aggregated from grid)."""
    dist: Dict[str, int] = {}
    for f in person.all_frames():
        key = f.pivot_name or "other"
        dist[key] = dist.get(key, 0) + 1
    return dist


# ── HTML builders ──


def _build_summary_html(
    video_name: str,
    result: CollectionResult,
    records: List[CollectionRecord],
) -> str:
    """Summary section."""
    n_records = len(records)
    n_persons = len(result.persons)

    person_cards = ""
    for pid, person in result.persons.items():
        n_total = len(person.all_frames())
        n_cells = len(person.grid)
        n_poses = len(person.pose_coverage)
        n_cats = len(person.category_coverage)
        mode = "catalog" if person.catalog_mode else "fallback"

        person_cards += f"""
        <div class="person-summary">
          <div class="person-title">Person {pid}</div>
          <div class="person-stats">
            <span class="badge grid">{n_total} frames</span>
            <span class="badge cells">{n_cells} cells</span>
            <span class="badge mode">{mode}</span>
          </div>
          <div class="person-coverage">
            poses: {n_poses} &middot;
            categories: {n_cats}
          </div>
        </div>"""

    return f"""
    <div class="summary-grid">
      <div class="stat-card">
        <div class="label">Video</div>
        <div class="value" style="font-size:1em;word-break:break-all">{video_name}</div>
      </div>
      <div class="stat-card">
        <div class="label">Records</div>
        <div class="value">{n_records}</div>
      </div>
      <div class="stat-card">
        <div class="label">Persons</div>
        <div class="value">{n_persons}</div>
      </div>
    </div>
    {person_cards}"""


def _build_gallery_html(
    person: PersonCollection,
    frames_b64: Dict[int, str],
) -> str:
    """Selected frames thumbnail gallery grouped by cell_key."""
    # Group frames by cell_key
    cells: Dict[str, List[SelectedFrame]] = {}
    for key, frames in person.grid.items():
        cells[key] = frames

    if not cells:
        return '<p style="color:#999">No frames selected</p>'

    html = ""
    for cell_key in sorted(cells.keys()):
        frames = cells[cell_key]
        html += f'<h4 class="gallery-title grid">{cell_key} ({len(frames)})</h4>\n'
        html += '<div class="gallery-grid">\n'

        for f in frames:
            b64 = frames_b64.get(f.frame_idx)
            img_tag = (
                f'<img src="data:image/jpeg;base64,{b64}" alt="frame">'
                if b64
                else '<div class="thumb-placeholder"></div>'
            )
            pose_str = f.pose_name or "—"
            pivot_str = f.pivot_name or "—"

            html += f"""
            <div class="frame-card grid">
              {img_tag}
              <div class="frame-meta">
                <strong>#{f.frame_idx}</strong>
                <span>{f.timestamp_ms / 1000.0:.2f}s</span>
              </div>
              <div class="frame-detail">
                <div class="bucket-label">{pose_str} | {pivot_str}</div>
                <div>quality: {f.quality_score:.3f}</div>
                <div>cell: {f.cell_score:.3f}</div>
                <div>pose_fit: {f.pose_fit:.3f}</div>
                <div>catalog_sim: {f.catalog_sim:.3f}</div>
                <div>stable: {f.stable_score:.3f}</div>
              </div>
            </div>"""

        html += "</div>\n"

    return html


def _build_config_table_html(result: CollectionResult) -> str:
    """CollectionConfig parameter table."""
    cfg = result.config or CollectionConfig()
    cfg_dict = asdict(cfg) if hasattr(cfg, "__dataclass_fields__") else {}
    if not cfg_dict:
        return ""

    rows = ""
    for key, val in cfg_dict.items():
        rows += f"<tr><td>{key}</td><td>{val}</td></tr>"

    return f"""
    <table class="config-table">
      <thead><tr><th>Parameter</th><th>Value</th></tr></thead>
      <tbody>{rows}</tbody>
    </table>"""


# ── CSS ──

_CSS = """
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
  h4.gallery-title {
    font-size: 0.85em; font-weight: 600; margin: 24px 0 12px;
    padding: 6px 12px; border-radius: 4px; display: inline-block;
  }
  h4.grid { background: #e8f5e9; color: #2e7d32; }
  .subtitle {
    color: #999; font-size: 0.9em; margin-bottom: 32px; font-weight: 400;
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
  .chart-row { display: flex; gap: 24px; margin: 16px 0; }
  .chart-row > div { flex: 1; min-width: 0; }
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
  .thumb-placeholder { width: 100%; height: 100px; background: #f0f0f0; }
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
  .footer {
    margin-top: 56px; text-align: center;
    color: #ccc; font-size: 0.75em; letter-spacing: 0.02em;
  }
  @media (max-width: 900px) {
    body { padding: 24px 16px; }
    .chart-row { flex-direction: column; }
    .gallery-grid { justify-content: center; }
  }
"""

# ── JS (Plotly charts) ──

_JS_MAIN = r"""
(function() {
  var axFont = {family:'Inter, sans-serif', size:10, color:'#999'};
  var gridColor = '#eee';

  var COLORS = ['#2e7d32','#1565c0','#e65100','#6a1b9a','#c62828',
                '#00695c','#4527a0','#bf360c','#1b5e20','#0d47a1'];

  Object.keys(DATA.persons).forEach(function(pid) {
    var p = DATA.persons[pid];

    // ── Scatter: Yaw x Pitch ──
    var scatterTraces = [];

    scatterTraces.push({
      x: p.all_yaw, y: p.all_pitch, name: 'all records',
      type:'scatter', mode:'markers',
      marker:{size:3, opacity:0.25, color:'#bbb'},
      hovertemplate:'yaw: %{x:.1f}\u00b0<br>pitch: %{y:.1f}\u00b0<extra>all</extra>'
    });

    var cats = Object.keys(p.selected);
    cats.forEach(function(cat, ci) {
      var sel = p.selected[cat];
      if (!sel || !sel.frame_idx.length) return;
      var htext = sel.frame_idx.map(function(fi, i) {
        return '#' + fi + '<br>y=' + sel.yaw[i].toFixed(1) + '\u00b0 p=' + sel.pitch[i].toFixed(1) + '\u00b0' +
               '<br>' + sel.pose[i] + ' | ' + sel.pivot[i] +
               '<br>q=' + sel.quality[i] + ' cell=' + (sel.cell_score ? sel.cell_score[i] : '');
      });
      scatterTraces.push({
        x: sel.yaw, y: sel.pitch, name: cat,
        text: htext,
        type:'scatter', mode:'markers',
        marker:{size:8, color:COLORS[ci % COLORS.length], opacity:0.85,
                line:{width:1, color:'#fff'}},
        hovertemplate:'%{text}<extra>' + cat + '</extra>'
      });
    });

    Plotly.newPlot('scatter_' + pid, scatterTraces, {
      height:350, margin:{l:50,r:30,t:24,b:40},
      paper_bgcolor:'rgba(0,0,0,0)', plot_bgcolor:'#fff',
      font:{family:'Inter, sans-serif', color:'#666', size:11},
      showlegend:true,
      legend:{orientation:'h', y:1.15, x:0.5, xanchor:'center', font:{size:10}},
      xaxis:{title:{text:'Yaw (\u00b0)', font:axFont}, gridcolor:gridColor,
             zeroline:true, zerolinecolor:'#ddd', tickfont:axFont,
             range:[-80,80]},
      yaxis:{title:{text:'Pitch (\u00b0)', font:axFont}, gridcolor:gridColor,
             zeroline:true, zerolinecolor:'#ddd', tickfont:axFont,
             range:[-40,40]}
    }, {responsive:true, displayModeBar:false});

    // ── Heatmap: Pose × Category Coverage ──
    var hm = p.heatmap;
    Plotly.newPlot('heatmap_' + pid, [{
      z: hm.z, x: hm.x, y: hm.y,
      type:'heatmap',
      colorscale:[[0,'#f5f5f5'],[0.5,'#90caf9'],[1,'#1565c0']],
      showscale:true,
      text: hm.tooltip,
      hovertemplate:'pose: %{x}<br>category: %{y}<br>count: %{z}<br>%{text}<extra></extra>',
      colorbar:{thickness:12, len:0.8, tickfont:{size:10}}
    }], {
      height:300, margin:{l:80,r:30,t:24,b:60},
      paper_bgcolor:'rgba(0,0,0,0)', plot_bgcolor:'#fff',
      font:{family:'Inter, sans-serif', color:'#666', size:11},
      xaxis:{title:{text:'Pose', font:axFont}, tickfont:axFont, tickangle:-30},
      yaxis:{title:{text:'Category', font:axFont}, tickfont:axFont}
    }, {responsive:true, displayModeBar:false});

    // ── Bar: Category Distribution ──
    var dist = p.category_dist;
    var catLabels = Object.keys(dist);
    var catValues = catLabels.map(function(c) { return dist[c]; });
    var catColors = catLabels.map(function(c, i) { return COLORS[i % COLORS.length]; });

    Plotly.newPlot('category_' + pid, [{
      x: catLabels, y: catValues, name: 'frames',
      type:'bar', marker:{color:catColors}
    }], {
      height:300, margin:{l:50,r:30,t:24,b:50},
      paper_bgcolor:'rgba(0,0,0,0)', plot_bgcolor:'#fff',
      font:{family:'Inter, sans-serif', color:'#666', size:11},
      xaxis:{title:{text:'Category', font:axFont}, tickfont:axFont},
      yaxis:{title:{text:'Count', font:axFont}, gridcolor:gridColor, tickfont:axFont}
    }, {responsive:true, displayModeBar:false});

    // ── PCA: 10D Signal Space (2D projection) ──
    if (p.pca) {
      var pca = p.pca;
      var pcaTraces = [];

      // Background: all records (grey)
      pcaTraces.push({
        x: pca.all_x, y: pca.all_y,
        type:'scatter', mode:'markers',
        marker:{size:3, opacity:0.15, color:'#bbb'},
        name:'all frames'
      });

      // Selected frames by category
      var pcaCats = Object.keys(pca.selected);
      pcaCats.forEach(function(cat, ci) {
        var sel = pca.selected[cat];
        var htext = sel.frame_idx.map(function(idx, j) {
          return 'frame ' + idx + '<br>score: ' + sel.cell_score[j].toFixed(3);
        });
        pcaTraces.push({
          x: sel.x, y: sel.y,
          text: htext,
          type:'scatter', mode:'markers',
          marker:{size:10, color:COLORS[ci % COLORS.length], opacity:0.85},
          name: cat,
          hovertemplate:'%{text}<extra>' + cat + '</extra>'
        });
      });

      // Centroids (star markers)
      var centroidCats = Object.keys(pca.centroids);
      centroidCats.forEach(function(cat, ci) {
        var c = pca.centroids[cat];
        pcaTraces.push({
          x:[c.x], y:[c.y],
          type:'scatter', mode:'markers',
          marker:{size:16, symbol:'star', color:COLORS[ci % COLORS.length]},
          name: cat + ' centroid',
          showlegend:false
        });
      });

      var pcTitle = '10D Signal Space (PCA — ' +
        (pca.explained_variance[0]*100).toFixed(0) + '% + ' +
        (pca.explained_variance[1]*100).toFixed(0) + '% variance)';

      Plotly.newPlot('pca_' + pid, pcaTraces, {
        height:400, margin:{l:50,r:30,t:40,b:40},
        paper_bgcolor:'rgba(0,0,0,0)', plot_bgcolor:'#fff',
        title:{text:pcTitle, font:{size:12, color:'#666'}},
        font:{family:'Inter, sans-serif', color:'#666', size:11},
        showlegend:true,
        legend:{orientation:'h', y:1.15, x:0.5, xanchor:'center', font:{size:10}},
        xaxis:{title:{text:'PC1', font:axFont}, gridcolor:gridColor, tickfont:axFont},
        yaxis:{title:{text:'PC2', font:axFont}, gridcolor:gridColor, tickfont:axFont}
      }, {responsive:true, displayModeBar:false});
    }
  });

})();
"""


def _build_html(
    video_name: str,
    result: CollectionResult,
    records: List[CollectionRecord],
    persons_data: Dict[int, Dict[str, Any]],
    frames_b64: Dict[int, str],
) -> str:
    """완전한 HTML 문서를 조립한다."""
    summary_html = _build_summary_html(video_name, result, records)
    config_html = _build_config_table_html(result)

    persons_html = ""
    for pid, person in result.persons.items():
        gallery_html = _build_gallery_html(person, frames_b64)
        has_pca = "pca" in persons_data.get(pid, {})

        pca_div = f'<div id="pca_{pid}"></div>' if has_pca else ""
        pca_section = ""
        if has_pca:
            pca_section = f"""
          <h2>Person {pid} &mdash; Signal Space</h2>
          {pca_div}"""

        persons_html += f"""
        <div class="person-section">
          <h2>Person {pid} &mdash; Pose Distribution</h2>
          <div class="chart-row">
            <div id="scatter_{pid}"></div>
            <div id="heatmap_{pid}"></div>
          </div>

          <h2>Person {pid} &mdash; Category Distribution</h2>
          <div id="category_{pid}"></div>
          {pca_section}

          <h2>Person {pid} &mdash; Selected Frames</h2>
          {gallery_html}
        </div>"""

    data_json = json.dumps({"persons": persons_data})
    safe_name = video_name.replace("&", "&amp;").replace("<", "&lt;")

    return (
        '<!DOCTYPE html>\n'
        '<html lang="ko">\n<head>\n'
        '<meta charset="UTF-8">\n'
        '<meta name="viewport" content="width=device-width,initial-scale=1">\n'
        '<title>Collection Report \u2014 ' + safe_name + '</title>\n'
        '<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>\n'
        '<style>' + _CSS + '</style>\n'
        '<link rel="preconnect" href="https://fonts.googleapis.com">\n'
        '<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">\n'
        '</head>\n<body>\n'
        '  <h1>Collection Analysis Report</h1>\n'
        '  <p class="subtitle">' + safe_name + '</p>\n'
        '  <h2>Summary</h2>\n' + summary_html + '\n'
        + persons_html + '\n'
        + '  <h2>Configuration</h2>\n' + config_html + '\n'
        '  <div class="footer">Generated by MomentScan Collection Report</div>\n'
        '<script>\nconst DATA = ' + data_json + ';\n'
        + _JS_MAIN
        + '\n</script>\n</body>\n</html>'
    )
