"""Interactive highlight analysis HTML report.

Plotly 기반 인터랙티브 시계열 차트 + hover 프레임 동기화 리포트 생성.
단일 .html 파일에 프레임 썸네일을 base64 임베드하여 자체 완결.

서브플롯은 2개 고정(Score Pipeline, Score Decomposition) +
활성 분석 모듈별 동적 생성. 모듈이 추가/제거되면 자동 반영.
"""

from __future__ import annotations

import base64
import json
import logging
from collections import OrderedDict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np

from momentscan.algorithm.batch.field_mapping import PIPELINE_FIELD_MAPPINGS
from momentscan.algorithm.batch.types import (
    HighlightConfig,
    HighlightResult,
    FrameRecord,
)

logger = logging.getLogger(__name__)

THUMB_WIDTH = 320
JPEG_QUALITY = 70

# ── Module display metadata ──

_MODULE_PALETTES: Dict[str, List[str]] = {
    "face.detect": ["#2e7d32", "#e65100", "#1565c0", "#6a1b9a", "#c62828", "#00838f"],
    "face.expression": ["#d84315", "#1565c0", "#2e7d32"],
    "body.pose": ["#00838f", "#d84315", "#6a1b9a", "#2e7d32"],
    "hand.gesture": ["#6a1b9a", "#e65100", "#00838f", "#c62828"],
    "vision.embed": ["#2e7d32", "#00838f", "#d84315"],
    "frame.quality": ["#1565c0", "#e65100", "#2e7d32"],
    "face.classify": ["#f9a825"],
    "frame.scoring": ["#757575"],
}

_MODULE_LABELS: Dict[str, str] = {
    "face.detect": "Face Detect",
    "face.expression": "Expression",
    "body.pose": "Body Pose",
    "hand.gesture": "Hand Gesture",
    "vision.embed": "Vision Embed",
    "frame.quality": "Frame Quality",
    "face.classify": "Face Classifier",
    "frame.scoring": "Frame Scoring",
}

_DEFAULT_PALETTE = ["#546e7a", "#607d8b", "#78909c", "#90a4ae", "#b0bec5"]

# Boolean fields: excluded from module subplot traces (redundant with gate_mask)
_SKIP_CHART_FIELDS = {"face_detected"}


# ── Public API ──


def export_highlight_report(
    video_path: Path,
    result: HighlightResult,
    output_dir: Path,
) -> None:
    """인터랙티브 하이라이트 분석 HTML 리포트를 생성한다.

    Args:
        video_path: 원본 비디오 경로.
        result: BatchHighlightEngine 분석 결과 (_timeseries 포함).
        output_dir: 출력 루트 디렉토리. highlight/report.html로 저장.
    """
    if result._timeseries is None:
        logger.info("No timeseries data — skipping HTML report")
        return

    highlight_dir = output_dir / "highlight"
    highlight_dir.mkdir(parents=True, exist_ok=True)

    ts = result._timeseries
    records: List[FrameRecord] = ts["records"]
    if not records:
        return

    # 프레임 썸네일 추출 (샘플링)
    thumb_indices = _select_thumbnail_indices(
        n_frames=len(records),
        peaks=ts["peaks"].tolist(),
        windows=result.windows,
        records=records,
    )
    frames_b64 = _extract_thumbnails(video_path, records, thumb_indices)

    # 차트 데이터 준비
    chart_data = _build_chart_data(result)

    # HTML 조립
    html = _build_html(
        video_name=video_path.name,
        result=result,
        chart_data=chart_data,
        frames_b64=frames_b64,
    )

    report_path = highlight_dir / "report.html"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html)

    size_mb = report_path.stat().st_size / (1024 * 1024)
    logger.info("Exported highlight report: %s (%.1f MB)", report_path, size_mb)


# ── Thumbnail extraction ──


MAX_THUMBNAILS = 200


def _select_thumbnail_indices(
    n_frames: int,
    peaks: List[int],
    windows: list,
    records: List[FrameRecord],
    max_thumbs: int = MAX_THUMBNAILS,
) -> List[int]:
    """추출할 프레임 인덱스를 선별한다.

    중요 프레임(peak, window selected) 우선 포함 후 균등 샘플링으로 보충.

    Returns:
        정렬된 point_index 리스트.
    """
    important: set[int] = set()

    # Peak frames
    for p in peaks:
        if 0 <= p < n_frames:
            important.add(int(p))

    # Window selected frames
    fidx_to_pidx = {r.frame_idx: i for i, r in enumerate(records)}
    for w in windows:
        for sf in getattr(w, "selected_frames", []):
            pidx = fidx_to_pidx.get(sf.get("frame_idx"))
            if pidx is not None:
                important.add(pidx)

    # Budget for uniform sampling
    remaining = max(0, max_thumbs - len(important))
    if remaining > 0 and n_frames > remaining:
        step = n_frames / remaining
        for i in range(remaining):
            idx = int(i * step)
            important.add(idx)
    elif remaining > 0:
        # n_frames가 budget 이하면 전부 포함
        important.update(range(n_frames))

    return sorted(important)


def _extract_thumbnails(
    video_path: Path,
    records: List[FrameRecord],
    indices: List[int] | None = None,
) -> Dict[int, str]:
    """비디오에서 지정된 프레임의 썸네일을 추출하여 base64 인코딩.

    Args:
        video_path: 원본 비디오 경로.
        records: 전체 FrameRecord 리스트.
        indices: 추출할 point_index 리스트. None이면 전체 추출.

    Returns:
        {point_index: base64_jpeg_string} 매핑.
    """
    from visualbase.sources.file import FileSource

    if indices is None:
        indices = list(range(len(records)))

    # 시간순 정렬하여 순차 seek 최적화 (forward seek이 backward보다 빠름)
    timestamps = [(idx, records[idx].timestamp_ms) for idx in indices]
    timestamps.sort(key=lambda x: x[1])

    frames_b64: Dict[int, str] = {}
    source = FileSource(str(video_path))
    source.open()

    try:
        for idx, ts_ms in timestamps:
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
        "Extracted %d/%d frame thumbnails for report",
        len(frames_b64), len(records),
    )
    return frames_b64


# ── Chart data builder ──


def _minmax(arr: np.ndarray) -> np.ndarray:
    """0~1 min-max 정규화 (helper)."""
    mn, mx = arr.min(), arr.max()
    if mx - mn < 1e-8:
        return np.zeros_like(arr)
    return (arr - mn) / (mx - mn)


def _build_chart_data(result: HighlightResult) -> Dict[str, Any]:
    """Plotly 서브플롯에 필요한 데이터를 JSON-serializable dict로 구성.

    modules 메타데이터를 포함하여 JS에서 동적으로 서브플롯을 생성할 수 있게 한다.
    """
    ts = result._timeseries
    records: List[FrameRecord] = ts["records"]
    cfg = result.config or HighlightConfig()
    n = len(records)

    time_sec = [r.timestamp_ms / 1000.0 for r in records]

    data: Dict[str, Any] = {
        "time_sec": time_sec,
        "final_scores": ts["final_scores"].tolist(),
        "smoothed": ts["smoothed"].tolist(),
        "peaks": ts["peaks"].tolist(),
        "gate_mask": ts["gate_mask"].astype(int).tolist(),
        "quality_scores": ts["quality_scores"].tolist(),
        "impact_scores": ts["impact_scores"].tolist(),
    }

    # Windows
    data["windows"] = [
        {
            "start_sec": w.start_ms / 1000.0,
            "end_sec": w.end_ms / 1000.0,
            "peak_sec": w.peak_ms / 1000.0,
            "window_id": w.window_id,
            "score": w.score,
            "reason": w.reason,
        }
        for w in result.windows
    ]

    # ── Module detection & per-field arrays ──
    module_groups: OrderedDict[str, list] = OrderedDict()
    for fm in PIPELINE_FIELD_MAPPINGS:
        module_groups.setdefault(fm.source, []).append(fm)

    default_rec = FrameRecord(frame_idx=0, timestamp_ms=0.0)
    modules_meta: List[Dict[str, Any]] = []

    for mod_name, mappings in module_groups.items():
        is_active = False
        for fm in mappings:
            vals = [getattr(r, fm.record_field) for r in records]
            default_val = getattr(default_rec, fm.record_field)
            if any(v != default_val for v in vals):
                is_active = True
            # Always add to data (hover panel needs all fields)
            if fm.record_field not in data:
                if isinstance(default_val, bool):
                    data[fm.record_field] = [int(v) for v in vals]
                else:
                    data[fm.record_field] = [float(v) for v in vals]

        if not is_active:
            continue

        palette = _MODULE_PALETTES.get(mod_name, _DEFAULT_PALETTE)
        label = _MODULE_LABELS.get(mod_name, mod_name)

        fields_meta = []
        ci = 0
        for fm in mappings:
            if fm.record_field in _SKIP_CHART_FIELDS:
                continue
            fields_meta.append({
                "key": fm.record_field,
                "label": fm.record_field,
                "color": palette[ci % len(palette)],
                "role": fm.scoring_role,
                "description": fm.description,
                "rationale": fm.rationale,
            })
            ci += 1

        modules_meta.append({
            "name": mod_name,
            "label": label,
            "fields": fields_meta,
        })

    data["modules"] = modules_meta

    # ── Gate thresholds (for reference lines on module subplots) ──
    data["gate_thresholds"] = {
        "face_confidence": [cfg.gate_face_confidence],
        "face_area_ratio": [cfg.gate_face_area_ratio],
        "eye_open_ratio": [cfg.gate_eye_open_min],
        "blur_score": [cfg.gate_blur_min],
        "brightness": [cfg.gate_exposure_min, cfg.gate_exposure_max],
    }

    # ── Quality decomposition arrays (for hover panel) ──
    arrays = ts.get("arrays", {})
    blur_arr = arrays.get("blur_score", np.array([r.blur_score for r in records]))
    face_area_arr = arrays.get(
        "face_area_ratio", np.array([r.face_area_ratio for r in records])
    )

    data["blur_normed"] = _minmax(blur_arr).tolist()
    data["face_size_normed"] = _minmax(face_area_arr).tolist()
    data["frontalness"] = [
        float(np.clip(1.0 - abs(r.head_yaw) / cfg.frontalness_max_yaw, 0.0, 1.0))
        for r in records
    ]
    data["face_identity"] = [float(r.face_identity) for r in records]

    # ── Normalized impact deltas (for hover panel) ──
    normed = ts.get("normed", {})
    for out_key, normed_key in (
        ("normed_face_change", "face_change"),
        ("normed_body_change", "body_change"),
        ("normed_smile_intensity", "smile_intensity"),
        ("normed_head_yaw", "head_yaw"),
        ("normed_mouth_open_ratio", "mouth_open_ratio"),
        ("normed_head_velocity", "head_velocity"),
        ("normed_wrist_raise", "wrist_raise"),
        ("normed_torso_rotation", "torso_rotation"),
        ("normed_face_area_ratio", "face_area_ratio"),
        ("normed_brightness", "brightness"),
    ):
        arr = normed.get(normed_key, np.zeros(n))
        data[out_key] = np.maximum(arr, 0.0).tolist()

    # head_velocity (derived, for hover)
    deltas = ts.get("deltas", {})
    hv = deltas.get("head_velocity", np.zeros(n))
    data["head_velocity"] = hv.tolist() if isinstance(hv, np.ndarray) else hv

    # ── Per-gate-condition boolean arrays (for hover panel) ──
    data["gate_face_detected"] = [int(r.face_detected) for r in records]
    data["gate_face_confidence_pass"] = [
        int(r.face_confidence >= cfg.gate_face_confidence) for r in records
    ]
    data["gate_face_area_pass"] = [
        int(r.face_area_ratio >= cfg.gate_face_area_ratio) for r in records
    ]
    data["gate_blur_pass"] = [
        int(r.blur_score <= 0 or r.blur_score >= cfg.gate_blur_min) for r in records
    ]
    data["gate_brightness_pass"] = [
        int(
            r.brightness <= 0
            or (cfg.gate_exposure_min <= r.brightness <= cfg.gate_exposure_max)
        )
        for r in records
    ]
    data["gate_eye_open_pass"] = [
        int(r.eye_open_ratio <= 0 or r.eye_open_ratio >= cfg.gate_eye_open_min)
        for r in records
    ]

    # ── Config weights (for hover panel) ──
    data["cfg_quality_weights"] = {
        "blur": cfg.quality_blur_weight,
        "face_size": cfg.quality_face_size_weight,
        "face_identity": cfg.quality_face_identity_weight,
        "frontalness": cfg.quality_frontalness_weight,
    }
    data["cfg_impact_weights"] = {
        "face_change": cfg.impact_face_change_weight,
        "body_change": cfg.impact_body_change_weight,
        "smile_intensity": cfg.impact_smile_intensity_weight,
        "head_yaw": cfg.impact_head_yaw_delta_weight,
        "head_velocity": cfg.impact_head_velocity_weight,
        "torso_rotation": cfg.impact_torso_rotation_weight,
    }
    data["cfg_impact_top_k"] = cfg.impact_top_k

    # ── Threshold values (for hover panel) ──
    data["thresholds"] = {
        "gate_face_confidence": cfg.gate_face_confidence,
        "gate_face_area_ratio": cfg.gate_face_area_ratio,
        "gate_eye_open_min": cfg.gate_eye_open_min,
        "gate_blur_min": cfg.gate_blur_min,
        "gate_exposure_min": cfg.gate_exposure_min,
        "gate_exposure_max": cfg.gate_exposure_max,
    }

    return data


# ── Summary / Window detail / Config table (unchanged) ──


def _build_summary_html(
    video_name: str,
    result: HighlightResult,
) -> str:
    """비디오 요약 정보 HTML."""
    ts = result._timeseries
    records = ts["records"]
    cfg = result.config or HighlightConfig()
    n = len(records)
    duration_sec = records[-1].timestamp_ms / 1000.0 if records else 0
    n_windows = len(result.windows)
    n_peaks = len(ts["peaks"])
    gate_pass = int(ts["gate_mask"].sum())
    gate_pct = 100.0 * gate_pass / n if n > 0 else 0

    return f"""
    <div class="summary-grid">
      <div class="stat-card">
        <div class="label">Video</div>
        <div class="value" style="font-size:1em;word-break:break-all">{video_name}</div>
      </div>
      <div class="stat-card">
        <div class="label">Frames</div>
        <div class="value">{n}</div>
      </div>
      <div class="stat-card">
        <div class="label">Duration</div>
        <div class="value">{duration_sec:.1f}s</div>
      </div>
      <div class="stat-card">
        <div class="label">FPS</div>
        <div class="value">{cfg.fps:.0f}</div>
      </div>
      <div class="stat-card">
        <div class="label">Windows</div>
        <div class="value">{n_windows}</div>
      </div>
      <div class="stat-card">
        <div class="label">Peaks</div>
        <div class="value">{n_peaks}</div>
      </div>
      <div class="stat-card">
        <div class="label">Gate Pass</div>
        <div class="value">{gate_pct:.0f}%</div>
      </div>
    </div>"""


def _build_window_detail_html(
    result: HighlightResult,
    frames_b64: Dict[int, str],
) -> str:
    """Window별 상세 정보 HTML (선택 프레임 썸네일 포함)."""
    if not result.windows:
        return '<p class="muted">No highlight windows detected.</p>'

    ts = result._timeseries
    records: List[FrameRecord] = ts["records"]
    # frame_idx → point index 매핑
    fidx_to_pidx = {r.frame_idx: i for i, r in enumerate(records)}

    html_parts = []
    for w in result.windows:
        reason_items = ""
        for feat, val in sorted(w.reason.items(), key=lambda x: -x[1]):
            bar_w = min(100, int(val * 30))
            reason_items += (
                f'<div class="reason-row">'
                f'<span class="reason-name">{feat}</span>'
                f'<div class="reason-bar" style="width:{bar_w}%"></div>'
                f'<span class="reason-val">{val:.2f}</span>'
                f'</div>'
            )

        # Selected frame thumbnails
        frames_html = '<div class="selected-frames">'
        for sf in w.selected_frames:
            pidx = fidx_to_pidx.get(sf["frame_idx"])
            b64 = frames_b64.get(pidx) if pidx is not None else None
            img_tag = (
                f'<img src="data:image/jpeg;base64,{b64}" alt="frame">'
                if b64 else '<div class="thumb-placeholder"></div>'
            )
            frames_html += (
                f'<div class="selected-frame">'
                f'{img_tag}'
                f'<div class="sf-meta">'
                f'<strong>#{sf["frame_idx"]}</strong> '
                f'{sf["timestamp_ms"] / 1000.0:.2f}s<br>'
                f'score {sf["frame_score"]:.3f}'
                f'</div>'
                f'</div>'
            )
        frames_html += '</div>'

        html_parts.append(f"""
        <div class="window-card">
          <div class="window-header">
            Window {w.window_id}
            <span class="window-score">score: {w.score:.3f}</span>
          </div>
          <div class="window-body">
            <div class="window-meta">
              {w.start_ms / 1000.0:.2f}s — {w.end_ms / 1000.0:.2f}s
              (peak: {w.peak_ms / 1000.0:.2f}s)
            </div>
            <div class="reason-chart">{reason_items}</div>
            {frames_html}
          </div>
        </div>""")

    return "\n".join(html_parts)


def _build_config_table_html(result: HighlightResult) -> str:
    """HighlightConfig 파라미터 테이블 HTML."""
    cfg = result.config or HighlightConfig()
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


def _build_field_reference_html() -> str:
    """분석 필드별 scoring role + rationale 레퍼런스 테이블 HTML."""
    role_badges = {
        "gate": ('<span class="role-badge role-gate">gate</span>', 0),
        "quality": ('<span class="role-badge role-quality">quality</span>', 1),
        "impact": ('<span class="role-badge role-impact">impact</span>', 2),
        "info": ('<span class="role-badge role-info">info</span>', 3),
    }

    rows = ""
    for fm in PIPELINE_FIELD_MAPPINGS:
        badge_html, _ = role_badges.get(fm.scoring_role or "info", role_badges["info"])
        rationale = fm.rationale or "\u2014"
        rows += (
            f"<tr>"
            f"<td><code>{fm.record_field}</code></td>"
            f"<td>{fm.source}</td>"
            f"<td>{badge_html}</td>"
            f"<td>{fm.description}</td>"
            f"<td class='rationale-cell'>{rationale}</td>"
            f"</tr>"
        )

    return f"""
    <table class="config-table field-ref-table">
      <thead><tr>
        <th>Field</th><th>Source</th><th>Role</th>
        <th>Description</th><th>Rationale</th>
      </tr></thead>
      <tbody>{rows}</tbody>
    </table>"""


# ── HTML assembly ──
# CSS and JS are plain strings (no f-string) to avoid brace-escaping hell.

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
  .subtitle {
    color: #999; font-size: 0.9em; margin-bottom: 32px; font-weight: 400;
  }
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
  .gate-fail { color: #c62828; font-weight: 600; }
  .section-label { color: #999; font-weight: 600; }
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
  .chart-row { display: flex; gap: 24px; margin: 16px 0; }
  .chart-row > div { flex: 1; min-width: 0; }
  .muted { color: #bbb; font-style: italic; }
  .footer {
    margin-top: 56px; text-align: center;
    color: #ccc; font-size: 0.75em; letter-spacing: 0.02em;
  }
  #plotDiv { width: 100%; }
  @media (max-width: 900px) {
    body { padding: 24px 16px; }
    .main-layout { flex-direction: column; }
    .frame-panel { position: static; width: 100%; }
  }
"""

_JS_MAIN = r"""
(function() {
  var t = DATA.time_sec;
  var nPoints = t.length;
  if (nPoints === 0) return;

  var N_FIXED = 2;
  var nMods = DATA.modules.length;
  var nRows = N_FIXED + nMods;

  function yRef(row) { return row === 0 ? 'y' : 'y' + (row + 1); }
  function hexAlpha(hex, a) {
    var r = parseInt(hex.slice(1,3),16);
    var g = parseInt(hex.slice(3,5),16);
    var b = parseInt(hex.slice(5,7),16);
    return 'rgba('+r+','+g+','+b+','+a+')';
  }
  function fmtN(v, d) { return v == null ? '\u2014' : v.toFixed(d); }

  // Pre-compute sorted frame indices for nearest-thumbnail lookup
  var frameKeys = Object.keys(FRAMES).map(Number).sort(function(a,b){return a-b;});
  function nearestFrame(pi) {
    if (FRAMES[String(pi)] !== undefined) return pi;
    var lo = 0, hi = frameKeys.length - 1, best = -1;
    while (lo <= hi) {
      var mid = (lo + hi) >> 1;
      if (frameKeys[mid] <= pi) { best = mid; lo = mid + 1; }
      else { hi = mid - 1; }
    }
    var cA = best >= 0 ? frameKeys[best] : null;
    var cB = best + 1 < frameKeys.length ? frameKeys[best + 1] : null;
    if (cA == null) return cB;
    if (cB == null) return cA;
    return (pi - cA <= cB - pi) ? cA : cB;
  }

  // ── Compute subplot domains ──
  var gap = 0.03;
  var rowH = (1.0 - gap * (nRows - 1)) / nRows;
  function rowDomain(r) {
    // row 0 = top, row nRows-1 = bottom
    var top = 1.0 - r * (rowH + gap);
    return [top - rowH, top];
  }

  // Each row gets its own x-axis + y-axis pair (separate hover zones)
  function xRef(row) { return row === 0 ? 'x' : 'x' + (row + 1); }
  function xaKey(row) { return row === 0 ? 'xaxis' : 'xaxis' + (row + 1); }
  function yaKey(row) { return row === 0 ? 'yaxis' : 'yaxis' + (row + 1); }

  // ── Window shading shapes ──
  var shapes = [];
  DATA.windows.forEach(function(w) {
    for (var row = 0; row < nRows; row++) {
      shapes.push({
        type:'rect', xref: xRef(row), yref: yRef(row),
        x0: w.start_sec, x1: w.end_sec, y0: 0, y1: 1,
        fillcolor:'rgba(0,0,0,0.03)', line:{width:0}, layer:'below'
      });
    }
  });

  // ── Fixed traces ──
  var traces = [];

  // Subplot 0 (x/y): Score Pipeline
  traces.push({x:t, y:DATA.final_scores, name:'final_score', type:'scatter', mode:'lines',
    line:{color:'rgba(0,0,0,0.12)', width:1}, xaxis:'x', yaxis:'y', hoverinfo:'x+y'});
  traces.push({x:t, y:DATA.smoothed, name:'smoothed', type:'scatter', mode:'lines',
    line:{color:'#111', width:2}, xaxis:'x', yaxis:'y', hoverinfo:'x+y'});
  var pX = DATA.peaks.map(function(i){return t[i]});
  var pY = DATA.peaks.map(function(i){return DATA.smoothed[i]});
  traces.push({x:pX, y:pY, name:'peak', type:'scatter', mode:'markers',
    marker:{color:'#e53935', size:7, symbol:'circle'}, xaxis:'x', yaxis:'y', hoverinfo:'x+y'});

  // Subplot 1 (x2/y2): Score Decomposition
  traces.push({x:t, y:DATA.gate_mask, name:'gate_mask', type:'scatter', mode:'lines',
    line:{color:'#2e7d32', width:1, shape:'hv'}, fill:'tozeroy',
    fillcolor:'rgba(46,125,50,0.06)', xaxis: xRef(1), yaxis:'y2', hoverinfo:'x+y'});
  traces.push({x:t, y:DATA.quality_scores, name:'quality', type:'scatter', mode:'lines',
    line:{color:'#1565c0', width:1.5}, xaxis: xRef(1), yaxis:'y2', hoverinfo:'x+y'});
  traces.push({x:t, y:DATA.impact_scores, name:'impact', type:'scatter', mode:'lines',
    line:{color:'#d84315', width:1.5}, xaxis: xRef(1), yaxis:'y2', hoverinfo:'x+y'});

  // ── Dynamic module subplots ──
  DATA.modules.forEach(function(mod, mi) {
    var row = N_FIXED + mi;
    var xa = xRef(row);
    var ya = yRef(row);
    mod.fields.forEach(function(f) {
      if (!DATA[f.key]) return;
      var roleBadge = f.role ? ' [' + f.role + ']' : '';
      var htempl = '<b>' + f.label + '</b>' + roleBadge +
        '<br>%{y:.3f}' +
        '<extra></extra>';
      traces.push({
        x: t, y: DATA[f.key], name: f.label,
        type:'scatter', mode:'lines',
        line:{color: f.color, width: 1.2},
        xaxis: xa, yaxis: ya,
        legendgroup: mod.name,
        hovertemplate: htempl
      });
    });
  });

  // ── Gate threshold reference lines ──
  var threshShapes = [];
  DATA.modules.forEach(function(mod, mi) {
    var row = N_FIXED + mi;
    var xa = xRef(row);
    var ya = yRef(row);
    mod.fields.forEach(function(f) {
      var th = DATA.gate_thresholds[f.key];
      if (!th) return;
      th.forEach(function(v) {
        threshShapes.push({
          type:'line', xref: xa, yref: ya,
          x0: t[0], x1: t[t.length-1], y0: v, y1: v,
          line:{color: hexAlpha(f.color, 0.35), width:1, dash:'dash'}
        });
      });
    });
  });

  // ── Layout (separate x-axis per row for independent hover zones) ──
  var gridColor = '#eee';
  var axFont = {family:'Inter, sans-serif', size:10, color:'#999'};

  var layout = {
    height: Math.max(500, 200 + nRows * 160),
    margin: {l:50, r:20, t:30, b:40},
    paper_bgcolor:'rgba(0,0,0,0)', plot_bgcolor:'#fff',
    font: {family:'Inter, sans-serif', color:'#666', size:11},
    showlegend: true,
    legend: {orientation:'h', y:1.02, x:0.5, xanchor:'center', font:{size:10, color:'#999'}},
    hovermode: 'x',
    shapes: shapes.concat(threshShapes)
  };

  // Axis labels
  var yLabels = ['Score', 'Decomposition'];
  DATA.modules.forEach(function(mod) { yLabels.push(mod.label); });

  for (var r = 0; r < nRows; r++) {
    var dom = rowDomain(r);
    // X-axis: anchored to its row's y-axis, linked range via matches
    layout[xaKey(r)] = {
      anchor: yRef(r),
      gridcolor: gridColor, zeroline: false, tickfont: axFont,
      matches: r === 0 ? undefined : 'x',
      showticklabels: r === nRows - 1
    };
    if (r === nRows - 1) {
      layout[xaKey(r)].title = {text:'Time (s)', font: axFont};
    }
    // Y-axis: domain slice, anchored to its row's x-axis
    layout[yaKey(r)] = {
      domain: dom,
      anchor: xRef(r),
      title:{text: yLabels[r], font: axFont},
      gridcolor: gridColor, zeroline:false, tickfont: axFont
    };
    if (r === 1) layout[yaKey(r)].range = [-0.05, 1.1];
  }

  Plotly.newPlot('plotDiv', traces, layout, {
    responsive: true, displayModeBar: true,
    modeBarButtonsToRemove: ['lasso2d', 'select2d']
  });

  // ── Hover → frame panel + pipeline detail ──
  // Uses mousemove on the plot div instead of plotly_hover,
  // because Plotly's hover system is unreliable across stacked subplots.
  var plotDiv = document.getElementById('plotDiv');
  var frameImg = document.getElementById('frameImg');
  var frameMeta = document.getElementById('frameMeta');
  var pipelineDetail = document.getElementById('pipelineDetail');

  function gIcon(ok) {
    return ok
      ? '<span class="gate-pass">\u2713</span>'
      : '<span class="gate-fail">\u2717</span>';
  }

  // Binary search for nearest time index from data x value
  function timeToIndex(xVal) {
    var lo = 0, hi = nPoints - 1;
    while (lo < hi) {
      var mid = (lo + hi) >> 1;
      if (t[mid] < xVal) lo = mid + 1; else hi = mid;
    }
    if (lo > 0 && Math.abs(t[lo-1] - xVal) < Math.abs(t[lo] - xVal)) lo--;
    return lo;
  }

  // Convert mouse pixel x → data x using Plotly's internal axis layout.
  // Uses the primary xaxis (all subplot x-axes share the same range via matches).
  function pixelToTime(mouseClientX) {
    var fl = plotDiv._fullLayout;
    if (!fl) return null;
    // Find any xaxis object (try xaxis first, then xaxis2, etc.)
    var xa = fl.xaxis;
    if (!xa) {
      for (var k in fl) {
        if (k.indexOf('xaxis') === 0 && fl[k] && fl[k]._length) { xa = fl[k]; break; }
      }
    }
    if (!xa || !xa._length) return null;
    var bb = plotDiv.getBoundingClientRect();
    var px = mouseClientX - bb.left - fl.margin.l;
    if (px < 0 || px > xa._length) return null;
    var range = xa.range;
    return range[0] + (px / xa._length) * (range[1] - range[0]);
  }

  var lastPi = -1;
  function updatePanel(pi) {
    if (pi === lastPi) return;
    lastPi = pi;

    // Thumbnail (nearest available)
    var ni = nearestFrame(pi);
    if (ni != null) {
      var b64 = FRAMES[String(ni)];
      if (b64) frameImg.src = 'data:image/jpeg;base64,' + b64;
    }

    // Basic meta
    var tsec = DATA.time_sec[pi];
    frameMeta.innerHTML =
      '<div><span>Frame</span> <strong>#' + pi + '</strong> &nbsp; ' +
      '<strong>' + fmtN(tsec, 2) + 's</strong></div>';

    // ── Pipeline decomposition ──
    var D = DATA;
    var th = D.thresholds;
    var qw = D.cfg_quality_weights;
    var iw = D.cfg_impact_weights;

    // Gate
    var gp = D.gate_mask[pi];
    var h = '<span class="section-label">\u2500\u2500 Gate: ' +
      (gp ? '<span class="gate-pass">PASS</span>' : '<span class="gate-fail">FAIL</span>') +
      ' \u2500\u2500</span>\n';
    h += '  ' + gIcon(D.gate_face_detected[pi])       + ' face_detected\n';
    h += '  ' + gIcon(D.gate_face_confidence_pass[pi]) + ' face_conf    ' + fmtN(D.face_confidence[pi],3) + ' > ' + fmtN(th.gate_face_confidence,3) + '\n';
    h += '  ' + gIcon(D.gate_face_area_pass[pi])       + ' face_area    ' + fmtN(D.face_area_ratio[pi],3) + ' > ' + fmtN(th.gate_face_area_ratio,3) + '\n';
    h += '  ' + gIcon(D.gate_blur_pass[pi])            + ' blur         ' + fmtN(D.blur_score[pi],1)      + ' > ' + fmtN(th.gate_blur_min,1) + '\n';
    h += '  ' + gIcon(D.gate_brightness_pass[pi])      + ' brightness   ' + fmtN(D.brightness[pi],1)      + ' \u2208 [' + th.gate_exposure_min + ', ' + th.gate_exposure_max + ']\n';
    h += '  ' + gIcon(D.gate_eye_open_pass[pi])       + ' eye_open     ' + fmtN(D.eye_open_ratio[pi],3)   + ' > ' + fmtN(th.gate_eye_open_min,3) + '\n';

    // Quality
    var qs = D.quality_scores[pi];
    var blV = D.blur_normed[pi], fsV = D.face_size_normed[pi];
    var fidV = D.face_identity[pi], frV = D.frontalness[pi];
    h += '\n<span class="section-label">\u2500\u2500 Quality: ' + fmtN(qs,3) + ' \u2500\u2500</span>\n';
    h += '  blur_norm     ' + fmtN(qw.blur,2)       + ' \u00d7 ' + fmtN(blV,3) + ' = ' + fmtN(qw.blur*blV,3) + '\n';
    h += '  face_size     ' + fmtN(qw.face_size,2)  + ' \u00d7 ' + fmtN(fsV,3) + ' = ' + fmtN(qw.face_size*fsV,3) + '\n';
    if (fidV > 0) {
      h += '  face_identity ' + fmtN(qw.face_identity,2) + ' \u00d7 ' + fmtN(fidV,3) + ' = ' + fmtN(qw.face_identity*fidV,3) + '\n';
    } else {
      h += '  frontalness   ' + fmtN(qw.frontalness,2)+ ' \u00d7 ' + fmtN(frV,3) + ' = ' + fmtN(qw.frontalness*frV,3) + ' <span style="color:#999">(fallback)</span>\n';
    }

    // Impact
    var is_ = D.impact_scores[pi];
    var topK = D.cfg_impact_top_k || 3;
    var impF = [
      ['face_change ', 'face_change', D.normed_face_change[pi]],
      ['body_change ', 'body_change', D.normed_body_change[pi]],
      ['smile       ', 'smile_intensity',  D.normed_smile_intensity[pi]],
      ['yaw_\u0394       ', 'head_yaw',         D.normed_head_yaw[pi]],
      ['head_vel    ', 'head_velocity',    D.normed_head_velocity[pi]],
      ['torso       ', 'torso_rotation',   D.normed_torso_rotation[pi]]
    ];
    // Compute weighted values and sort for top-K display
    var impWV = impF.map(function(f) {
      var w = iw[f[1]] || 0;
      return {label: f[0], key: f[1], raw: f[2], w: w, wv: w * f[2]};
    }).sort(function(a,b) { return b.wv - a.wv; });
    h += '\n<span class="section-label">\u2500\u2500 Impact: ' + fmtN(is_,3) + ' (top-' + topK + ') \u2500\u2500</span>\n';
    for (var j = 0; j < impWV.length; j++) {
      var e = impWV[j];
      var mark = j < topK ? '\u25cf' : ' ';
      h += '  ' + mark + ' ' + e.label + ' ' + fmtN(e.w,2) + ' \u00d7 ' + fmtN(e.raw,3) + ' = ' + fmtN(e.wv,3) + '\n';
    }

    // Final
    var fs = D.final_scores[pi];
    var sm = D.smoothed[pi];
    h += '\n<span class="section-label">\u2500\u2500 Final \u2500\u2500</span>\n';
    h += '  ' + fmtN(qs,3) + ' \u00d7 ' + fmtN(is_,3) + ' = ' + fmtN(fs,3) + '\n';
    h += '  smoothed: ' + fmtN(sm,3) + '\n';

    pipelineDetail.innerHTML = h;
  }

  // mousemove: pixel → time → frame index (works on ALL subplot rows)
  plotDiv.addEventListener('mousemove', function(evt) {
    var xVal = pixelToTime(evt.clientX);
    if (xVal == null) return;
    var pi = timeToIndex(xVal);
    if (pi >= 0 && pi < nPoints) updatePanel(pi);
  });

  // ── Auxiliary: Head Pose timeline ──
  var hpTraces = [
    {x:t, y:DATA.head_yaw,   name:'yaw',   type:'scatter', mode:'lines', line:{color:'#d84315', width:1.5}},
    {x:t, y:DATA.head_pitch,  name:'pitch', type:'scatter', mode:'lines', line:{color:'#1565c0', width:1.5}},
    {x:t, y:DATA.head_roll,   name:'roll',  type:'scatter', mode:'lines', line:{color:'#2e7d32', width:1.5}}
  ];
  Plotly.newPlot('headPoseDiv', hpTraces, {
    height:260, margin:{l:50,r:20,t:24,b:36},
    paper_bgcolor:'rgba(0,0,0,0)', plot_bgcolor:'#fff',
    font:{family:'Inter, sans-serif', color:'#666', size:11},
    showlegend:true,
    legend:{orientation:'h', y:1.12, x:0.5, xanchor:'center', font:{size:10, color:'#999'}},
    hovermode:'x unified',
    xaxis:{title:{text:'Time (s)', font:axFont}, gridcolor:gridColor, zeroline:false, tickfont:axFont},
    yaxis:{title:{text:'Degrees', font:axFont}, gridcolor:gridColor, zeroline:true, zerolinecolor:'#ddd', tickfont:axFont}
  }, {responsive:true, displayModeBar:false});

  // ── Auxiliary: Face Position Distribution (yaw vs pitch scatter) ──
  Plotly.newPlot('faceDistDiv', [{
    x:DATA.head_yaw, y:DATA.head_pitch, name:'',
    type:'scatter', mode:'markers',
    marker:{size:4, opacity:0.4, color:DATA.time_sec,
      colorscale:[[0,'#ddd'],[1,'#111']],
      colorbar:{title:{text:'time(s)', font:{size:10}}, thickness:10, len:0.6}},
    hovertemplate:'yaw: %{x:.1f}\u00b0<br>pitch: %{y:.1f}\u00b0<extra></extra>'
  }], {
    height:260, margin:{l:50,r:60,t:24,b:36},
    paper_bgcolor:'rgba(0,0,0,0)', plot_bgcolor:'#fff',
    font:{family:'Inter, sans-serif', color:'#666', size:11},
    showlegend:false,
    xaxis:{title:{text:'Yaw (\u00b0)', font:axFont}, gridcolor:gridColor, zeroline:true, zerolinecolor:'#ddd', tickfont:axFont},
    yaxis:{title:{text:'Pitch (\u00b0)', font:axFont}, gridcolor:gridColor, zeroline:true, zerolinecolor:'#ddd', tickfont:axFont, scaleanchor:'x'}
  }, {responsive:true, displayModeBar:false});
})();
"""


def _build_html(
    video_name: str,
    result: HighlightResult,
    chart_data: Dict[str, Any],
    frames_b64: Dict[int, str],
) -> str:
    """완전한 HTML 문서를 조립한다.

    CSS와 JS는 module-level 상수로 분리하여 brace-escaping 문제를 회피.
    """
    summary_html = _build_summary_html(video_name, result)
    window_detail_html = _build_window_detail_html(result, frames_b64)
    config_table_html = _build_config_table_html(result)
    field_ref_html = _build_field_reference_html()

    chart_data_json = json.dumps(chart_data)
    frames_json = json.dumps(frames_b64)

    safe_name = video_name.replace("&", "&amp;").replace("<", "&lt;")

    return (
        '<!DOCTYPE html>\n'
        '<html lang="ko">\n<head>\n'
        '<meta charset="UTF-8">\n'
        '<meta name="viewport" content="width=device-width,initial-scale=1">\n'
        '<title>Highlight Report \u2014 ' + safe_name + '</title>\n'
        '<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>\n'
        '<style>' + _CSS + '</style>\n'
        '<link rel="preconnect" href="https://fonts.googleapis.com">\n'
        '<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">\n'
        '</head>\n<body>\n'
        '  <h1>Highlight Analysis Report</h1>\n'
        '  <p class="subtitle">' + safe_name + '</p>\n'
        '  <h2>Summary</h2>\n' + summary_html + '\n'
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
        '  <h2>Highlight Windows</h2>\n' + window_detail_html + '\n'
        '  <h2>Field Reference</h2>\n' + field_ref_html + '\n'
        '  <h2>Configuration</h2>\n' + config_table_html + '\n'
        '  <div class="footer">Generated by MomentScan Highlight Report</div>\n'
        '<script>\nconst DATA = ' + chart_data_json + ';\n'
        'const FRAMES = ' + frames_json + ';\n'
        + _JS_MAIN
        + '\n</script>\n</body>\n</html>'
    )
