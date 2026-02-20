"""Interactive identity analysis HTML report.

Plotly 기반 인터랙티브 차트 + 프레임 썸네일 갤러리.
단일 .html 파일에 프레임 썸네일을 base64 임베드하여 자체 완결.

시각화:
- Summary: person별 anchor/coverage/challenge 수, bucket coverage
- Yaw x Pitch scatter: 전체 레코드 + 선택된 프레임 (set_type별 컬러)
- Bucket coverage heatmap: yaw(7) x pitch(5) 그리드
- Expression distribution: bar chart (set_type별 stacked)
- Frame thumbnail gallery: anchor/coverage/challenge 그룹별
- Config table: IdentityConfig 파라미터
"""

from __future__ import annotations

import base64
import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

import cv2

from momentscan.algorithm.identity.types import (
    IdentityConfig,
    IdentityFrame,
    IdentityRecord,
    IdentityResult,
    PersonIdentity,
)

logger = logging.getLogger(__name__)

THUMB_WIDTH = 220

# Pivot-based heatmap labels
_POSE_PIVOT_LABELS = [
    "frontal", "three-quarter", "side-profile", "looking-up", "three-quarter-up", "fallback",
]
_EXPR_PIVOT_LABELS = [
    "neutral", "smile", "excited", "surprised", "mouth_open", "eyes_closed",
]
JPEG_QUALITY = 70


# ── Public API ──


def export_identity_report(
    video_path: Path,
    result: IdentityResult,
    records: List[IdentityRecord],
    output_dir: Path,
) -> None:
    """인터랙티브 identity 분석 HTML 리포트를 생성한다.

    Args:
        video_path: 원본 비디오 경로.
        result: IdentityBuilder.build() 결과 (선택된 프레임).
        records: 전체 IdentityRecord (산점도용 — 선택 안 된 프레임도 포함).
        output_dir: 출력 루트 디렉토리. identity/report.html로 저장.
    """
    if not result.persons:
        logger.info("No persons — skipping identity HTML report")
        return

    identity_dir = output_dir / "identity"
    identity_dir.mkdir(parents=True, exist_ok=True)

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
        output_dir=output_dir,
    )

    report_path = identity_dir / "report.html"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html)

    size_mb = report_path.stat().st_size / (1024 * 1024)
    logger.info("Exported identity report: %s (%.1f MB)", report_path, size_mb)


# ── Thumbnail extraction ──


def _collect_all_selected(result: IdentityResult) -> List[IdentityFrame]:
    """Collect all selected frames across persons."""
    frames = []
    for person in result.persons.values():
        frames.extend(person.anchor_frames)
        frames.extend(person.coverage_frames)
        frames.extend(person.challenge_frames)
    return frames


def _extract_thumbnails(
    video_path: Path,
    records: List[IdentityRecord],
    selected_frames: List[IdentityFrame],
) -> Dict[int, str]:
    """선택된 프레임의 썸네일을 추출하여 base64 인코딩.

    Returns:
        {frame_idx: base64_jpeg_string} 매핑.
    """
    from visualbase.sources.file import FileSource

    # Build frame_idx → timestamp_ms mapping from records
    fidx_to_ts = {r.frame_idx: r.timestamp_ms for r in records}

    # Collect frame indices
    target_indices = sorted({f.frame_idx for f in selected_frames})

    # Timestamp-sorted for sequential seek optimization
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
        "Extracted %d/%d frame thumbnails for identity report",
        len(frames_b64), len(target_indices),
    )
    return frames_b64


# ── Data builders ──


def _build_persons_data(
    result: IdentityResult,
    records: List[IdentityRecord],
) -> Dict[int, Dict[str, Any]]:
    """Person별 차트 데이터를 구성한다."""
    data = {}
    for pid, person in result.persons.items():
        # Collect all records for this person
        person_records = [r for r in records if r.person_id == pid]
        data[pid] = _build_person_data(person, person_records)
    return data


def _build_person_data(
    person: PersonIdentity,
    records: List[IdentityRecord],
) -> Dict[str, Any]:
    """한 person의 차트 데이터."""
    # Build frame_idx → record lookup for numeric yaw/pitch
    record_map = {r.frame_idx: r for r in records}

    # All records: scatter plot (unselected)
    all_yaw = [r.head_yaw for r in records]
    all_pitch = [r.head_pitch for r in records]

    # Selected frames by set_type
    selected_by_type: Dict[str, List[IdentityFrame]] = {
        "anchor": person.anchor_frames,
        "coverage": person.coverage_frames,
        "challenge": person.challenge_frames,
    }

    scatter_selected = {}
    for stype, frames in selected_by_type.items():
        scatter_selected[stype] = {
            # Numeric degrees from IdentityRecord (accurate for scatter plot)
            "yaw": [round(record_map[f.frame_idx].head_yaw, 1) if f.frame_idx in record_map else 0.0 for f in frames],
            "pitch": [round(record_map[f.frame_idx].head_pitch, 1) if f.frame_idx in record_map else 0.0 for f in frames],
            "frame_idx": [f.frame_idx for f in frames],
            "quality": [round(f.quality_score, 3) for f in frames],
            "stable": [round(f.stable_score, 3) for f in frames],
        }

    # Heatmap: yaw(7) x pitch(5) = count of selected frames
    heatmap = _build_heatmap(person)

    # Expression distribution by set_type
    expression_dist = _build_expression_dist(person)

    return {
        "all_yaw": all_yaw,
        "all_pitch": all_pitch,
        "selected": scatter_selected,
        "heatmap": heatmap,
        "expression_dist": expression_dist,
    }


def _build_heatmap(person: PersonIdentity) -> Dict[str, Any]:
    """Pose pivot × Expression heatmap data.

    X axis: pose pivot names (frontal → side-profile → … → fallback)
    Y axis: expression labels (neutral → smile → excited → …)

    Pivot frames: pose label from f.bucket.yaw_bin (set to pivot name by pivot_to_bucket())
    Fallback frames: pose label = "fallback" (f.pivot_name is None)
    """
    x_labels = _POSE_PIVOT_LABELS
    y_labels = _EXPR_PIVOT_LABELS
    grid = [[0] * len(x_labels) for _ in range(len(y_labels))]

    all_frames = (
        person.anchor_frames + person.coverage_frames + person.challenge_frames
    )
    for f in all_frames:
        # Pivot frames have pose name in yaw_bin; fallback frames have old-style interval
        pose_label = f.bucket.yaw_bin if f.pivot_name is not None else "fallback"
        expr_label = f.bucket.expression_bin

        xi = x_labels.index(pose_label) if pose_label in x_labels else -1
        yi = y_labels.index(expr_label) if expr_label in y_labels else -1
        if xi >= 0 and yi >= 0:
            grid[yi][xi] += 1

    tooltip = [
        [str(grid[yi][xi]) if grid[yi][xi] > 0 else "empty" for xi in range(len(x_labels))]
        for yi in range(len(y_labels))
    ]

    return {
        "z": grid,
        "x": x_labels,
        "y": y_labels,
        "tooltip": tooltip,
    }


def _build_expression_dist(person: PersonIdentity) -> Dict[str, Dict[str, int]]:
    """Expression distribution by set_type."""
    expressions = ["neutral", "smile", "mouth_open", "eyes_closed", "excited", "surprised"]
    dist: Dict[str, Dict[str, int]] = {}

    for stype, frames in [
        ("anchor", person.anchor_frames),
        ("coverage", person.coverage_frames),
        ("challenge", person.challenge_frames),
    ]:
        counts = {e: 0 for e in expressions}
        for f in frames:
            if f.bucket.expression_bin in counts:
                counts[f.bucket.expression_bin] += 1
        dist[stype] = counts

    return dist


# ── HTML builders ──


def _build_summary_html(
    video_name: str,
    result: IdentityResult,
    records: List[IdentityRecord],
) -> str:
    """Summary section."""
    n_records = len(records)
    n_persons = len(result.persons)

    person_cards = ""
    for pid, person in result.persons.items():
        n_a = len(person.anchor_frames)
        n_c = len(person.coverage_frames)
        n_ch = len(person.challenge_frames)
        n_total = n_a + n_c + n_ch
        yaw_bins = len(person.yaw_coverage)
        pitch_bins = len(person.pitch_coverage)
        expr_bins = len(person.expression_coverage)

        person_cards += f"""
        <div class="person-summary">
          <div class="person-title">Person {pid}</div>
          <div class="person-stats">
            <span class="badge anchor">{n_a} anchors</span>
            <span class="badge coverage">{n_c} coverage</span>
            <span class="badge challenge">{n_ch} challenge</span>
            <span class="badge total">{n_total} total</span>
          </div>
          <div class="person-coverage">
            pose pivots: {yaw_bins}/{len(_POSE_PIVOT_LABELS)} &middot;
            expression: {expr_bins}/4
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
    person: PersonIdentity,
    frames_b64: Dict[int, str],
) -> str:
    """Selected frames thumbnail gallery."""
    sections = [
        ("Anchors", "anchor", person.anchor_frames),
        ("Coverage", "coverage", person.coverage_frames),
        ("Challenge", "challenge", person.challenge_frames),
    ]

    html = ""
    for title, css_class, frames in sections:
        if not frames:
            html += f'<h4 class="gallery-title {css_class}">{title} (0)</h4>\n'
            continue

        html += f'<h4 class="gallery-title {css_class}">{title} ({len(frames)})</h4>\n'
        html += '<div class="gallery-grid">\n'

        for f in frames:
            b64 = frames_b64.get(f.frame_idx)
            img_tag = (
                f'<img src="data:image/jpeg;base64,{b64}" alt="frame">'
                if b64
                else '<div class="thumb-placeholder"></div>'
            )
            bucket_str = f"{f.bucket.yaw_bin} | {f.bucket.pitch_bin} | {f.bucket.expression_bin}"
            novelty_line = (
                f'<div>novelty: {f.novelty_score:.3f}</div>'
                if f.set_type == "challenge" else ""
            )
            pivot_line = (
                f'<div class="bucket-label">pivot: {f.pivot_name} (d={f.pivot_distance:.1f})</div>'
                if f.pivot_name else ""
            )

            html += f"""
            <div class="frame-card {css_class}">
              {img_tag}
              <div class="frame-meta">
                <strong>#{f.frame_idx}</strong>
                <span>{f.timestamp_ms / 1000.0:.2f}s</span>
              </div>
              <div class="frame-detail">
                <div class="bucket-label">{bucket_str}</div>
                {pivot_line}
                <div>quality: {f.quality_score:.3f}</div>
                <div>stable: {f.stable_score:.3f}</div>
                {novelty_line}
              </div>
            </div>"""

        html += "</div>\n"

    return html


def _build_bank_status_html(
    output_dir: Path,
    result: IdentityResult,
    frames_b64: Dict[int, str],
) -> str:
    """Bank Status section: node table + rep image thumbnails.

    Looks for memory_bank.json in output_dir/identity/person_{pid}/.
    If no bank files exist, returns empty string.
    """
    sections = ""

    for pid in result.persons:
        bank_path = output_dir / "identity" / f"person_{pid}" / "memory_bank.json"
        if not bank_path.exists():
            continue

        try:
            with open(bank_path, "r") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue

        nodes = data.get("nodes", [])
        k_max = data.get("_config", {}).get("k_max", "?")

        # Node table rows
        rows = ""
        for node in nodes:
            nid = node.get("node_id", "?")
            meta = node.get("meta_hist", {})
            hits = meta.get("hit_count", 0)
            q_best = meta.get("quality_best", 0.0)
            q_mean = meta.get("quality_mean", 0.0)
            rep_images = node.get("rep_images", [])

            yaw_str = ", ".join(
                f"{k}: {v}" for k, v in meta.get("yaw_bins", {}).items()
            )
            pitch_str = ", ".join(
                f"{k}: {v}" for k, v in meta.get("pitch_bins", {}).items()
            )
            expr_str = ", ".join(
                f"{k}: {v}" for k, v in meta.get("expression_bins", {}).items()
            )

            # Rep image thumbnails (inline from crop files)
            thumbs_html = ""
            for img_str in rep_images:
                img_p = Path(img_str)
                if img_p.exists():
                    import cv2 as _cv2
                    img = _cv2.imread(str(img_p))
                    if img is not None:
                        h, w = img.shape[:2]
                        new_w = 80
                        new_h = int(h * new_w / w) if w > 0 else 80
                        thumb = _cv2.resize(img, (new_w, new_h))
                        _, buf = _cv2.imencode(
                            ".jpg", thumb,
                            [_cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY],
                        )
                        b64 = base64.b64encode(buf).decode("ascii")
                        thumbs_html += (
                            f'<img src="data:image/jpeg;base64,{b64}" '
                            f'style="width:60px;height:auto;border-radius:3px;margin:2px" '
                            f'title="{img_p.name}">'
                        )
                    else:
                        thumbs_html += f'<span style="color:#ccc">{img_p.name}</span> '
                else:
                    thumbs_html += f'<span style="color:#ccc">{Path(img_str).name}</span> '

            bucket_detail = ""
            if yaw_str:
                bucket_detail += f"<div>yaw: {yaw_str}</div>"
            if pitch_str:
                bucket_detail += f"<div>pitch: {pitch_str}</div>"
            if expr_str:
                bucket_detail += f"<div>expression: {expr_str}</div>"

            rows += f"""
            <tr>
              <td>{nid}</td>
              <td>{hits}</td>
              <td>{q_best:.2f}</td>
              <td>{q_mean:.2f}</td>
              <td style="font-size:0.78em">{bucket_detail}</td>
              <td>{thumbs_html}</td>
            </tr>"""

        sections += f"""
        <div class="person-summary" style="margin-top:12px">
          <div class="person-title">Person {pid} &mdash; Bank</div>
          <div class="person-coverage">{len(nodes)} nodes &middot; k_max={k_max}</div>
          <table class="config-table" style="margin-top:8px">
            <thead>
              <tr>
                <th>Node</th><th>Hits</th><th>Q Best</th><th>Q Mean</th>
                <th>Buckets</th><th>Rep Images</th>
              </tr>
            </thead>
            <tbody>{rows}</tbody>
          </table>
        </div>"""

    if not sections:
        return ""

    return sections


def _build_config_table_html(result: IdentityResult) -> str:
    """IdentityConfig parameter table."""
    cfg = result.config or IdentityConfig()
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
  h4.anchor { background: #e8f5e9; color: #2e7d32; }
  h4.coverage { background: #e3f2fd; color: #1565c0; }
  h4.challenge { background: #fff3e0; color: #e65100; }
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
  .badge.anchor { background: #e8f5e9; color: #2e7d32; }
  .badge.coverage { background: #e3f2fd; color: #1565c0; }
  .badge.challenge { background: #fff3e0; color: #e65100; }
  .badge.total { background: #f5f5f5; color: #666; }
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
  .frame-card.anchor { border-left: 3px solid #2e7d32; }
  .frame-card.coverage { border-left: 3px solid #1565c0; }
  .frame-card.challenge { border-left: 3px solid #e65100; }
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
# Uses DATA global from inline JSON. Plain string to avoid brace escaping.

_JS_MAIN = r"""
(function() {
  var axFont = {family:'Inter, sans-serif', size:10, color:'#999'};
  var gridColor = '#eee';

  var SET_COLORS = {anchor:'#2e7d32', coverage:'#1565c0', challenge:'#e65100'};

  Object.keys(DATA.persons).forEach(function(pid) {
    var p = DATA.persons[pid];

    // ── Scatter: Yaw x Pitch ──
    var scatterTraces = [];

    // All records (unselected, gray)
    scatterTraces.push({
      x: p.all_yaw, y: p.all_pitch, name: 'all records',
      type:'scatter', mode:'markers',
      marker:{size:3, opacity:0.25, color:'#bbb'},
      hovertemplate:'yaw: %{x:.1f}\u00b0<br>pitch: %{y:.1f}\u00b0<extra>all</extra>'
    });

    // Selected frames by set_type
    ['anchor','coverage','challenge'].forEach(function(stype) {
      var sel = p.selected[stype];
      if (!sel || !sel.frame_idx.length) return;
      var htext = sel.frame_idx.map(function(fi, i) {
        return '#' + fi + '<br>y=' + sel.yaw[i].toFixed(1) + '\u00b0 p=' + sel.pitch[i].toFixed(1) + '\u00b0<br>q=' + sel.quality[i] + ' s=' + sel.stable[i];
      });
      scatterTraces.push({
        x: sel.yaw, y: sel.pitch, name: stype,
        text: htext,
        type:'scatter', mode:'markers',
        marker:{size:8, color:SET_COLORS[stype], opacity:0.85,
                line:{width:1, color:'#fff'}},
        hovertemplate:'%{text}<extra>' + stype + '</extra>'
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
             range:[-50,50]},
      yaxis:{title:{text:'Pitch (\u00b0)', font:axFont}, gridcolor:gridColor,
             zeroline:true, zerolinecolor:'#ddd', tickfont:axFont,
             range:[-35,35]}
    }, {responsive:true, displayModeBar:false});

    // ── Heatmap: Bucket Coverage ──
    var hm = p.heatmap;
    Plotly.newPlot('heatmap_' + pid, [{
      z: hm.z, x: hm.x, y: hm.y,
      type:'heatmap',
      colorscale:[[0,'#f5f5f5'],[0.5,'#90caf9'],[1,'#1565c0']],
      showscale:true,
      text: hm.tooltip,
      hovertemplate:'yaw: %{x}<br>pitch: %{y}<br>count: %{z}<br>%{text}<extra></extra>',
      colorbar:{thickness:12, len:0.8, tickfont:{size:10}}
    }], {
      height:300, margin:{l:80,r:30,t:24,b:50},
      paper_bgcolor:'rgba(0,0,0,0)', plot_bgcolor:'#fff',
      font:{family:'Inter, sans-serif', color:'#666', size:11},
      xaxis:{title:{text:'Yaw Bin', font:axFont}, tickfont:axFont,
             tickangle:-30},
      yaxis:{title:{text:'Pitch Bin', font:axFont}, tickfont:axFont}
    }, {responsive:true, displayModeBar:false});

    // ── Bar: Expression Distribution ──
    var expr = p.expression_dist;
    var exprLabels = ['neutral','smile','mouth_open','eyes_closed','excited','surprised'];
    var barTraces = [];
    ['anchor','coverage','challenge'].forEach(function(stype) {
      var vals = exprLabels.map(function(e) { return expr[stype][e] || 0; });
      barTraces.push({
        x: exprLabels, y: vals, name: stype,
        type:'bar', marker:{color:SET_COLORS[stype]}
      });
    });

    Plotly.newPlot('expression_' + pid, barTraces, {
      height:300, margin:{l:50,r:30,t:24,b:50},
      barmode:'stack',
      paper_bgcolor:'rgba(0,0,0,0)', plot_bgcolor:'#fff',
      font:{family:'Inter, sans-serif', color:'#666', size:11},
      showlegend:true,
      legend:{orientation:'h', y:1.15, x:0.5, xanchor:'center', font:{size:10}},
      xaxis:{title:{text:'Expression', font:axFont}, tickfont:axFont},
      yaxis:{title:{text:'Count', font:axFont}, gridcolor:gridColor, tickfont:axFont}
    }, {responsive:true, displayModeBar:false});
  });


})();
"""


def _build_html(
    video_name: str,
    result: IdentityResult,
    records: List[IdentityRecord],
    persons_data: Dict[int, Dict[str, Any]],
    frames_b64: Dict[int, str],
    output_dir: Path | None = None,
) -> str:
    """완전한 HTML 문서를 조립한다."""
    summary_html = _build_summary_html(video_name, result, records)
    config_html = _build_config_table_html(result)
    bank_html = ""
    if output_dir is not None:
        bank_html = _build_bank_status_html(output_dir, result, frames_b64)

    # Per-person chart divs + gallery
    persons_html = ""
    for pid, person in result.persons.items():
        gallery_html = _build_gallery_html(person, frames_b64)

        persons_html += f"""
        <div class="person-section">
          <h2>Person {pid} &mdash; Face Pose Distribution</h2>
          <div class="chart-row">
            <div id="scatter_{pid}"></div>
            <div id="heatmap_{pid}"></div>
          </div>

          <h2>Person {pid} &mdash; Expression Distribution</h2>
          <div id="expression_{pid}"></div>

          <h2>Person {pid} &mdash; Selected Frames</h2>
          {gallery_html}
        </div>"""

    # Serialize data
    data_json = json.dumps({"persons": persons_data})
    safe_name = video_name.replace("&", "&amp;").replace("<", "&lt;")

    return (
        '<!DOCTYPE html>\n'
        '<html lang="ko">\n<head>\n'
        '<meta charset="UTF-8">\n'
        '<meta name="viewport" content="width=device-width,initial-scale=1">\n'
        '<title>Identity Report \u2014 ' + safe_name + '</title>\n'
        '<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>\n'
        '<style>' + _CSS + '</style>\n'
        '<link rel="preconnect" href="https://fonts.googleapis.com">\n'
        '<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">\n'
        '</head>\n<body>\n'
        '  <h1>Identity Analysis Report</h1>\n'
        '  <p class="subtitle">' + safe_name + '</p>\n'
        '  <h2>Summary</h2>\n' + summary_html + '\n'
        + persons_html + '\n'
        + ('  <h2>Bank Status</h2>\n' + bank_html + '\n' if bank_html else '')
        + '  <h2>Configuration</h2>\n' + config_html + '\n'
        '  <div class="footer">Generated by MomentScan Identity Report</div>\n'
        '<script>\nconst DATA = ' + data_json + ';\n'
        + _JS_MAIN
        + '\n</script>\n</body>\n</html>'
    )
