"""MomentscanV2 debug visualization — per-frame judgment overlay.

Video frame + right-side panel showing judgment basis:
  - Gate: pass/fail + reasons
  - Expression: probability distribution bar chart
  - Pose: probability distribution bar chart
  - Key signals: AU, quality, segmentation values

Usage:
    from momentscan.v2_debug import DebugV2

    app = DebugV2(expression_model="models/bind_v11.pkl")
    results = app.run("video.mp4", fps=2)
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import cv2
import numpy as np

from momentscan.v2 import MomentscanV2, FrameResult, MOMENTSCAN_MODULES

logger = logging.getLogger("momentscan.v2_debug")

# Colors (BGR)
_GREEN = (0, 200, 0)
_RED = (0, 0, 200)
_YELLOW = (0, 220, 220)
_CYAN = (220, 200, 0)
_WHITE = (255, 255, 255)
_GRAY = (140, 140, 140)
_DARK_GRAY = (80, 80, 80)
_BG = (30, 30, 30)
_PANEL_BG = (25, 25, 25)

PANEL_W = 280


def _compute_gate_severity(signals: dict) -> float:
    """Compute continuous gate severity (0=perfect, 1=severely failing).

    Each check contributes proportionally to how far the signal exceeds
    the threshold. Max severity is clamped to 1.0.
    """
    import math
    severity = 0.0

    # Exposure: how far outside [50, 200]
    expo = signals.get("face_exposure", 100.0)
    if expo > 0:
        if expo < 50:
            severity += (50 - expo) / 50  # 0 at 50, 1.0 at 0
        elif expo > 200:
            severity += (expo - 200) / 55  # 0 at 200, 1.0 at 255

    # Contrast: below 0.10
    contrast = signals.get("face_contrast", 0.5)
    if 0 < contrast < 0.10:
        severity += (0.10 - contrast) / 0.10

    # Clipping
    severity += min(1.0, signals.get("clipped_ratio", 0) / 0.15)
    severity += min(1.0, signals.get("crushed_ratio", 0) / 0.15)

    # Blur: below 5
    blur = signals.get("face_blur", 50.0)
    if 0 < blur < 5:
        severity += (5 - blur) / 5

    # Confidence: below 0.7
    conf = signals.get("face_confidence", 0.85)
    if 0 < conf < 0.7:
        severity += (0.7 - conf) / 0.7

    # Pose: yaw/pitch/roll/combined
    yaw = abs(signals.get("head_yaw_dev", 0))
    pitch = abs(signals.get("head_pitch", 0))
    roll = abs(signals.get("head_roll", 0))
    if yaw > 55:
        severity += (yaw - 55) / 35  # 0 at 55, 1.0 at 90
    if pitch > 35:
        severity += (pitch - 35) / 55
    if roll > 35:
        severity += (roll - 35) / 55
    combined = math.sqrt(yaw**2 + pitch**2 + roll**2)
    if combined > 55:
        severity += (combined - 55) / 45

    # Signal validity
    seg = signals.get("seg_face", -1)
    if seg >= 0 and seg < 0.01:
        severity += 0.5
    au_keys = [k for k in signals if k.startswith("au")]
    if au_keys:
        au_sum = sum(signals[k] for k in au_keys)
        if au_sum < 0.05:
            severity += 0.3

    return min(1.0, severity)


def _put(img, text, x, y, color=_WHITE, scale=0.4, thickness=1):
    cv2.putText(img, str(text), (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thickness, cv2.LINE_AA)


def _bar(img, x, y, w, val, color=_WHITE, h=10):
    """Draw a horizontal bar (0~1)."""
    val = max(0.0, min(1.0, val))
    cv2.rectangle(img, (x, y), (x + w, y + h), (50, 50, 50), -1)
    fill = int(w * val)
    if fill > 0:
        cv2.rectangle(img, (x, y), (x + fill, y + h), color, -1)


def _section_title(img, x, y, text):
    """Draw a section title with underline."""
    _put(img, text, x, y, _CYAN, 0.42, 1)
    cv2.line(img, (x, y + 4), (x + PANEL_W - 20, y + 4), _DARK_GRAY, 1)
    return y + 20


def _draw_prob_bars(img, x, y, scores: dict, highlight: str = ""):
    """Draw probability distribution as horizontal bars."""
    if not scores:
        _put(img, "(no model)", x, y, _GRAY)
        return y + 16
    for name, prob in sorted(scores.items(), key=lambda kv: -kv[1]):
        color = _GREEN if name == highlight else _GRAY
        _put(img, f"{name:>10s}", x, y, color, 0.35)
        _bar(img, x + 80, y - 8, 120, prob, color, 8)
        _put(img, f"{prob:.0%}", x + 205, y, color, 0.32)
        y += 16
    return y


def _draw_panel(panel: np.ndarray, result: FrameResult, fps: float, coverage_grid: dict = None, sh_9: np.ndarray = None):
    """Draw the right-side judgment panel."""
    if coverage_grid is None:
        coverage_grid = {}
    x, y = 10, 20

    # --- Header ---
    shoot_text = "SHOOT" if result.is_shoot else ("GATE FAIL" if not result.gate_passed else "CUT(xgb)")
    shoot_color = _GREEN if result.is_shoot else (_RED if not result.gate_passed else _YELLOW)
    _put(panel, shoot_text, x, y, shoot_color, 0.7, 2)
    _put(panel, f"#{result.frame_idx}", x + 150, y, _GRAY, 0.4)
    y += 10
    if fps > 0:
        _put(panel, f"{fps:.1f} fps", x + 150, y + 12, _GRAY, 0.35)
    y += 25

    if not result.face_detected:
        _put(panel, "No face detected", x, y, _GRAY, 0.45)
        return

    j = result.judgment

    # --- Gate ---
    y = _section_title(panel, x, y, "Gate")
    if j.gate_passed:
        _put(panel, "PASSED", x, y, _GREEN, 0.4)
        y += 18
    else:
        for reason in j.gate_reasons:
            _put(panel, f"x {reason}", x, y, _RED, 0.35)
            y += 14
        y += 4

    # --- Gate signals ---
    sigs = result.signals
    gate_items = [
        ("confidence", sigs.get("face_confidence", 0), 1.0),
        ("blur", sigs.get("face_blur", 0), 500.0),
        ("exposure", sigs.get("face_exposure", 0), 255.0),
        ("contrast", sigs.get("face_contrast", 0), 1.0),
        ("clipped", sigs.get("clipped_ratio", 0), 0.3),
        ("crushed", sigs.get("crushed_ratio", 0), 0.3),
    ]
    for label, val, max_val in gate_items:
        norm = val / max_val if max_val > 0 else 0
        _put(panel, f"{label:>10s}", x, y, _GRAY, 0.3)
        _bar(panel, x + 75, y - 7, 100, norm, _WHITE, 7)
        _put(panel, f"{val:.2f}", x + 180, y, _GRAY, 0.3)
        y += 14
    y += 8

    # --- Expression ---
    y = _section_title(panel, x, y, "Expression")
    y = _draw_prob_bars(panel, x, y, j.expression_scores, j.expression)
    y += 6

    # --- Pose ---
    y = _section_title(panel, x, y, "Pose")
    y = _draw_prob_bars(panel, x, y, j.pose_scores, j.pose)
    y += 6

    # --- AU Face Map ---
    y = _section_title(panel, x, y, "Action Units")
    _draw_au_face_map(panel, x, y, sigs)
    y += 110

    # --- Lighting Sphere ---
    y = _section_title(panel, x, y, "Lighting")
    _draw_lighting_sphere(panel, x, y, sigs, sh_9=sh_9)
    y += 70

    # --- Head Pose (compact) ---
    yaw = sigs.get("head_yaw_dev", 0)
    pitch = sigs.get("head_pitch", 0)
    roll = sigs.get("head_roll", 0)
    _put(panel, f"Head  Y:{yaw:.0f}  P:{pitch:.0f}  R:{roll:.0f}", x, y, _GRAY, 0.32)
    y += 18

    # --- Expression × Pose Coverage Grid ---
    y = _section_title(panel, x, y, "Coverage")
    _draw_coverage_grid(panel, x, y, coverage_grid)
    y += 80  # grid height


def _au_intensity_color(val: float):
    """AU intensity → color: dark(0) → green(0.3) → yellow(0.6) → red(1.0)."""
    val = max(0.0, min(1.0, val))
    if val < 0.1:
        return (40, 40, 40)  # inactive
    if val < 0.3:
        t = val / 0.3
        return (0, int(150 * t), 0)
    if val < 0.6:
        t = (val - 0.3) / 0.3
        return (0, 150 + int(55 * t), int(200 * t))
    t = (val - 0.6) / 0.4
    return (0, int(205 * (1 - t * 0.5)), 200 + int(55 * t))


def _draw_au_face_map(img, ox, oy, sigs: dict):
    """Draw schematic face with AU activation hotspots.

    Each AU is positioned at its anatomical location on a natural face outline.
    Circle size + color indicates activation intensity.
    """
    cx = ox + 65
    cy = oy + 50
    c = _DARK_GRAY  # wireframe color

    # --- Face contour (jawline + forehead) ---
    face_pts = np.array([
        [cx, cy - 52],          # forehead top
        [cx + 15, cy - 50],
        [cx + 30, cy - 44],
        [cx + 40, cy - 34],     # temple
        [cx + 44, cy - 20],
        [cx + 44, cy - 5],      # cheekbone
        [cx + 42, cy + 10],
        [cx + 36, cy + 25],     # jaw
        [cx + 26, cy + 38],
        [cx + 14, cy + 46],     # chin approach
        [cx, cy + 50],          # chin
        [cx - 14, cy + 46],
        [cx - 26, cy + 38],
        [cx - 36, cy + 25],
        [cx - 42, cy + 10],
        [cx - 44, cy - 5],
        [cx - 44, cy - 20],
        [cx - 40, cy - 34],
        [cx - 30, cy - 44],
        [cx - 15, cy - 50],
        [cx, cy - 52],
    ], dtype=np.int32)
    cv2.polylines(img, [face_pts], False, c, 1, cv2.LINE_AA)

    # --- Eyebrows ---
    # Left eyebrow
    lbrow = np.array([
        [cx - 30, cy - 26], [cx - 24, cy - 30], [cx - 16, cy - 31],
        [cx - 9, cy - 29],
    ], dtype=np.int32)
    cv2.polylines(img, [lbrow], False, c, 1, cv2.LINE_AA)
    # Right eyebrow
    rbrow = np.array([
        [cx + 9, cy - 29], [cx + 16, cy - 31], [cx + 24, cy - 30],
        [cx + 30, cy - 26],
    ], dtype=np.int32)
    cv2.polylines(img, [rbrow], False, c, 1, cv2.LINE_AA)

    # --- Eyes (almond shape) ---
    def _draw_eye(ecx, ecy):
        eye = np.array([
            [ecx - 11, ecy], [ecx - 7, ecy - 5], [ecx - 2, ecy - 6],
            [ecx + 3, ecy - 5], [ecx + 10, ecy],
            [ecx + 3, ecy + 4], [ecx - 2, ecy + 4],
            [ecx - 7, ecy + 3], [ecx - 11, ecy],
        ], dtype=np.int32)
        cv2.polylines(img, [eye], False, c, 1, cv2.LINE_AA)
        # Iris
        cv2.circle(img, (ecx, ecy), 3, c, 1, cv2.LINE_AA)

    _draw_eye(cx - 19, cy - 14)
    _draw_eye(cx + 19, cy - 14)

    # --- Nose ---
    # Bridge
    nose_bridge = np.array([
        [cx - 3, cy - 10], [cx - 2, cy - 2], [cx - 4, cy + 5],
    ], dtype=np.int32)
    cv2.polylines(img, [nose_bridge], False, c, 1, cv2.LINE_AA)
    nose_bridge_r = np.array([
        [cx + 3, cy - 10], [cx + 2, cy - 2], [cx + 4, cy + 5],
    ], dtype=np.int32)
    cv2.polylines(img, [nose_bridge_r], False, c, 1, cv2.LINE_AA)
    # Nostrils
    nose_bot = np.array([
        [cx - 10, cy + 8], [cx - 6, cy + 10], [cx, cy + 8],
        [cx + 6, cy + 10], [cx + 10, cy + 8],
    ], dtype=np.int32)
    cv2.polylines(img, [nose_bot], False, c, 1, cv2.LINE_AA)

    # --- Mouth (upper + lower lip) ---
    upper_lip = np.array([
        [cx - 18, cy + 22], [cx - 10, cy + 19], [cx - 4, cy + 20],
        [cx, cy + 18],  # cupid's bow
        [cx + 4, cy + 20], [cx + 10, cy + 19], [cx + 18, cy + 22],
    ], dtype=np.int32)
    cv2.polylines(img, [upper_lip], False, c, 1, cv2.LINE_AA)
    lower_lip = np.array([
        [cx - 18, cy + 22], [cx - 10, cy + 27], [cx - 4, cy + 29],
        [cx, cy + 30],
        [cx + 4, cy + 29], [cx + 10, cy + 27], [cx + 18, cy + 22],
    ], dtype=np.int32)
    cv2.polylines(img, [lower_lip], False, c, 1, cv2.LINE_AA)

    # AU positions (relative to face center) + labels
    # Format: (au_signal_key, dx, dy, label)
    # AU positions matched to new face wireframe anatomy
    au_positions = [
        ("au1_inner_brow",    -10, -30, "1"),   # inner brow raiser
        ("au2_outer_brow",    -28, -27, "2"),   # outer brow raiser (left)
        ("au2_outer_brow",    +28, -27, "2"),   # outer brow raiser (right)
        ("au4_brow_lowerer",    0, -24, "4"),   # brow lowerer (center)
        ("au5_upper_lid",     -19, -18, "5"),   # upper lid raiser (left)
        ("au5_upper_lid",     +19, -18, "5"),   # upper lid raiser (right)
        ("au6_cheek_raiser",  -32,  -2, "6"),   # cheek raiser (left)
        ("au6_cheek_raiser",  +32,  -2, "6"),   # cheek raiser (right)
        ("au9_nose_wrinkler",   0,   5, "9"),   # nose wrinkler
        ("au12_lip_corner",   -18,  22, "12"),  # lip corner puller (left)
        ("au12_lip_corner",   +18,  22, "12"),  # lip corner puller (right)
        ("au15_lip_depressor",-20,  27, "15"),  # lip corner depressor (left)
        ("au15_lip_depressor",+20,  27, "15"),  # lip corner depressor (right)
        ("au17_chin_raiser",    0,  38, "17"),  # chin raiser
        ("au20_lip_stretcher",-14,  24, "20"),  # lip stretcher (left)
        ("au20_lip_stretcher",+14,  24, "20"),  # lip stretcher (right)
        ("au25_lips_part",      0,  24, "25"),  # lips part
        ("au26_jaw_drop",       0,  46, "26"),  # jaw drop
    ]

    # Draw AU hotspots
    drawn_labels = set()
    for key, dx, dy, label in au_positions:
        val = sigs.get(key, 0.0)
        px = cx + dx
        py = cy + dy

        color = _au_intensity_color(val)
        radius = 3 + int(val * 5)  # 3~8px

        if val >= 0.1:
            # Filled circle with glow
            cv2.circle(img, (px, py), radius + 2, color, -1, cv2.LINE_AA)
            cv2.circle(img, (px, py), radius, (min(255, color[0] + 50), min(255, color[1] + 50), min(255, color[2] + 50)), -1, cv2.LINE_AA)
        else:
            cv2.circle(img, (px, py), 2, (50, 50, 50), -1)

        # Label (avoid duplicates for symmetric AUs)
        label_key = f"{label}_{dx > 0}"
        if label_key not in drawn_labels and val >= 0.1:
            _put(img, label, px + radius + 2, py + 3, _GRAY, 0.22)
            drawn_labels.add(label_key)

    # Legend: top active AUs with values (right side)
    lx = ox + 140
    ly = oy + 5
    _put(img, "Active:", lx, ly, _CYAN, 0.3)
    ly += 14
    au_vals = [(k, sigs.get(k, 0)) for k in sorted(sigs.keys()) if k.startswith("au")]
    au_vals.sort(key=lambda x: -x[1])
    for k, v in au_vals[:5]:
        if v < 0.1:
            break
        short = k.split("_", 1)[1] if "_" in k else k
        color = _au_intensity_color(v)
        _put(img, f"{short}: {v:.2f}", lx, ly, color, 0.28)
        ly += 12


def _draw_lighting_sphere(img, ox, oy, sigs: dict, sh_9: np.ndarray = None):
    """구면 조명 히트맵 — SfSNet SH 9계수로 정확하게 렌더링.

    sh_9가 있으면 9계수 전체 사용, 없으면 signal의 4값으로 근사.

    ⚠️ 좌표계: SfSNet은 얼굴 기준(거울 반전).
      구면 렌더링 시 nx=-screen_x, ny=-screen_y로 변환하여
      이미지 좌표계와 일치시킴.
    """
    import math

    cx = ox + 35
    cy = oy + 30
    radius = 25

    sh_x = sigs.get("sh_dir_x", 0)
    sh_y = sigs.get("sh_dir_y", 0)
    sh_str = sigs.get("sh_dir_strength", 0)

    c1, c2, c3, c4, c5 = 0.886, 1.023, 0.248, 0.858, 0.429

    grid_size = 20  # 해상도 향상 (12 → 20)
    for i in range(grid_size):
        for j in range(grid_size):
            screen_x = (j / (grid_size - 1)) * 2 - 1
            screen_y = (i / (grid_size - 1)) * 2 - 1
            r2 = screen_x**2 + screen_y**2
            if r2 > 1:
                continue

            if sh_9 is not None:
                # DPR SH 9계수 렌더링 — DPR demo와 동일한 좌표계
                # demo: x=linspace(-1,1), z=linspace(1,-1), y=-sqrt(1-x²-z²)
                # screen_x: 화면 우측 양수 = DPR X 양수
                # screen_y: 화면 아래 양수 = DPR Z 음수
                norm_X = screen_x
                norm_Z = -screen_y
                norm_Y = -math.sqrt(max(0, 1 - norm_X**2 - norm_Z**2))

                # DPR SH_basis (utils_SH.py와 동일, att 포함)
                att0, att1, att2 = math.pi, 2*math.pi/3, math.pi/4
                sh_b = [
                    0.5/math.sqrt(math.pi) * att0,                                    # 0: 1
                    math.sqrt(3)/2/math.sqrt(math.pi) * norm_Y * att1,                # 1: Y
                    math.sqrt(3)/2/math.sqrt(math.pi) * norm_Z * att1,                # 2: Z
                    math.sqrt(3)/2/math.sqrt(math.pi) * norm_X * att1,                # 3: X
                    math.sqrt(15)/2/math.sqrt(math.pi) * norm_Y*norm_X * att2,        # 4: YX
                    math.sqrt(15)/2/math.sqrt(math.pi) * norm_Y*norm_Z * att2,        # 5: YZ
                    math.sqrt(5)/4/math.sqrt(math.pi) * (3*norm_Z**2-1) * att2,       # 6: 3Z²-1
                    math.sqrt(15)/2/math.sqrt(math.pi) * norm_X*norm_Z * att2,        # 7: XZ
                    math.sqrt(15)/4/math.sqrt(math.pi) * (norm_X**2-norm_Y**2) * att2, # 8: X²-Y²
                ]
                intensity = sum(sh_9[i] * sh_b[i] for i in range(9))
            else:
                # signal 4값으로 근사 (이미 이미지 좌표계)
                sh_amb = sigs.get("sh_ambient", 0.5)
                intensity = sh_amb + sh_x * screen_x + sh_y * (-screen_y)

            intensity = max(0, min(1, intensity))

            if intensity < 0.3:
                t = intensity / 0.3
                b, g, r = int(80 * t), int(30 * t), 0
            elif intensity < 0.6:
                t = (intensity - 0.3) / 0.3
                b, g, r = int(80 * (1 - t)), int(30 + 170 * t), int(200 * t)
            else:
                t = (intensity - 0.6) / 0.4
                b, g, r = 0, int(200 * (1 - t * 0.5)), int(200 + 55 * t)

            px = cx + int(screen_x * radius)
            py = cy + int(screen_y * radius)
            cv2.circle(img, (px, py), 2, (b, g, r), -1)

    cv2.circle(img, (cx, cy), radius + 1, _DARK_GRAY, 1, cv2.LINE_AA)
    cv2.line(img, (cx - 4, cy), (cx + 4, cy), (60, 60, 60), 1)
    cv2.line(img, (cx, cy - 4), (cx, cy + 4), (60, 60, 60), 1)

    # Dominant direction arrow — 밝은 쪽을 가리킴, 길이 = strength
    # DPR: sh_dir_x > 0 = 이미지 우측 밝음, sh_dir_y > 0 = 이미지 상단 밝음
    # 화살표 길이를 strength에 비례 (약한 빛 = 짧은 화살표)
    if sh_str > 0.05:  # 최소 threshold (약한 빛은 화살표 안 그림)
        arrow_len = min(sh_str / 0.5, 1.0) * radius * 0.8  # strength 비례
        norm = max(abs(sh_x), abs(sh_y), 0.001)
        ax = int(sh_x / norm * arrow_len)
        ay = int(-sh_y / norm * arrow_len)  # screen_y 반전
        cv2.arrowedLine(img, (cx, cy), (cx + ax, cy + ay),
                       (0, 255, 255), 2, tipLength=0.3)

    # Labels
    lx = ox + 70
    ly = oy + 5
    pattern = "dramatic" if sh_str > 0.3 else ("natural" if sh_str > 0.15 else "flat")
    hardness = sigs.get("light_hardness", 0)
    rembrandt = sigs.get("rembrandt_score", 0)

    color = (0, 200, 255) if pattern == "dramatic" else ((200, 180, 0) if pattern == "natural" else _GRAY)
    _put(img, pattern, lx, ly, color, 0.35, 1)
    ly += 14
    _put(img, f"str: {sh_str:.2f}", lx, ly, _GRAY, 0.28)
    ly += 12
    _put(img, f"hard: {hardness:.2f}", lx, ly, _GRAY, 0.28)
    ly += 12
    if rembrandt > 0:
        _put(img, "REMBRANDT", lx, ly, (0, 200, 255), 0.28, 1)


def _draw_coverage_grid(img, x, y, grid: dict):
    """Draw expression × pose heatmap grid.

    grid: {(expression, pose): best_confidence}
    """
    expr_cats = ["cheese", "chill", "edge", "goofy", "hype"]
    pose_cats = ["front", "angle", "side"]

    cell_w, cell_h = 55, 16
    label_w = 50

    # Column headers (pose)
    for j, pose in enumerate(pose_cats):
        px = x + label_w + j * cell_w + 5
        _put(img, pose, px, y, _GRAY, 0.28)
    y += 12

    # Rows (expression)
    for i, expr in enumerate(expr_cats):
        ry = y + i * cell_h
        color = _get_expr_color(expr)
        _put(img, expr, x, ry + 11, color, 0.3)

        for j, pose in enumerate(pose_cats):
            cx = x + label_w + j * cell_w
            conf = grid.get((expr, pose), 0.0)

            # Cell background: intensity mapped to brightness
            if conf > 0:
                brightness = int(40 + 160 * conf)
                # Tint with expression color
                bc = tuple(int(c * conf * 0.6) for c in color)
                cv2.rectangle(img, (cx, ry), (cx + cell_w - 2, ry + cell_h - 2), bc, -1)
                _put(img, f"{conf:.0%}", cx + 8, ry + 11, _WHITE, 0.28)
            else:
                cv2.rectangle(img, (cx, ry), (cx + cell_w - 2, ry + cell_h - 2), (35, 35, 35), -1)
                _put(img, "-", cx + 22, ry + 11, _DARK_GRAY, 0.28)


def _draw_face_bbox(image: np.ndarray, faces, gate_passed: bool, video_wh: tuple[int, int] = None):
    """Draw face bboxes and head pose axes on video frame."""
    if video_wh:
        w, h = video_wh
    else:
        h, w = image.shape[:2]
    for face in faces:
        fx, fy, fw, fh = face.bbox  # normalized
        x1 = int(fx * w)
        y1 = int(fy * h)
        x2 = int((fx + fw) * w)
        y2 = int((fy + fh) * h)

        # bbox color: green=gate pass, red=gate fail
        color = _GREEN if gate_passed else _RED
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        # Face ID + confidence
        _put(image, f"id:{face.face_id} {face.confidence:.0%}", x1, y1 - 5, color, 0.35)

        # Head pose cube (얼굴 바라보는 방향 시각화)
        cube_x = x2 + 5  # bbox 우측에 배치
        cube_y = y1
        cube_size = max(20, (y2 - y1) // 3)
        _draw_pose_cube(image, cube_x, cube_y, face.yaw, face.pitch, face.roll, cube_size)


def _draw_lighting_probes(image, face, observations, sigs, vw, vh):
    """영상 프레임 위에 조명 probe 위치와 밝기를 직접 표시.

    seg map은 face crop 좌표계(512×512)이므로
    face bbox 내부 상대 좌표로 매핑하여 표시.
    """
    fx, fy, fw, fh = face.bbox
    x1, y1 = int(fx * vw), int(fy * vh)
    x2, y2 = int((fx + fw) * vw), int((fy + fh) * vh)
    bbox_w, bbox_h = x2 - x1, y2 - y1
    mid_x = (x1 + x2) // 2
    mid_y = (y1 + y2) // 2

    if bbox_w < 10 or bbox_h < 10:
        return

    # 4분면 경계선 (점선)
    for i in range(x1, x2, 6):
        cv2.line(image, (i, mid_y), (min(i + 3, x2), mid_y), (40, 40, 40), 1)
    for i in range(y1, y2, 6):
        cv2.line(image, (mid_x, i), (mid_x, min(i + 3, y2)), (40, 40, 40), 1)

    # Get segmentation map from observations
    seg = None
    for obs in observations:
        if getattr(obs, "source", "") == "face.parse":
            data = getattr(obs, "data", None)
            if data and hasattr(data, "results") and data.results:
                seg = data.results[0].class_map
                break

    if seg is None:
        return

    seg_h, seg_w = seg.shape[:2]

    # Get crop_box from face.parse output (expanded region, not face bbox)
    crop_box = None
    for obs in observations:
        if getattr(obs, "source", "") == "face.parse":
            data = getattr(obs, "data", None)
            if data and hasattr(data, "results") and data.results:
                crop_box = data.results[0].crop_box  # (x, y, w, h) pixel coords
                break

    if crop_box is None or crop_box[2] < 1 or crop_box[3] < 1:
        return

    cb_x, cb_y, cb_w, cb_h = crop_box

    # Helper: seg 좌표 → video 프레임 좌표 (seg는 crop_box 기준)
    def seg_to_frame(sx, sy):
        rx = cb_x + int(sx / seg_w * cb_w)
        ry = cb_y + int(sy / seg_h * cb_h)
        return rx, ry

    # Nose region (class 10) — cyan outline
    nose_mask = (seg == 10)
    abs_yaw = abs(sigs.get("head_yaw_dev", 0))

    # Skin 영역 윤곽선 표시 (BiSeNet class 1)
    skin_seg = (seg == 1).astype(np.uint8)
    if skin_seg.sum() > 10:
        skin_resized = cv2.resize(skin_seg, (cb_w, cb_h), interpolation=cv2.INTER_NEAREST)
        contours_skin, _ = cv2.findContours(skin_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours_skin:
            cnt_frame = cnt.copy().astype(np.float32)
            cnt_frame[:, :, 0] += cb_x
            cnt_frame[:, :, 1] += cb_y
            cv2.drawContours(image, [cnt_frame.astype(np.int32)], -1, (80, 50, 0), 1, cv2.LINE_AA)

    if nose_mask.sum() > 10:
        # Nose contour (윤곽선)
        contours, _ = cv2.findContours(
            nose_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        for cnt in contours:
            cnt_frame = cnt.copy().astype(np.float32)
            cnt_frame[:, :, 0] = cb_x + cnt[:, :, 0] / seg_w * cb_w
            cnt_frame[:, :, 1] = cb_y + cnt[:, :, 1] / seg_h * cb_h
            cv2.drawContours(image, [cnt_frame.astype(np.int32)], -1, (255, 255, 0), 1, cv2.LINE_AA)

        # Nose center + position ratio 표시
        nose_coords = np.where(nose_mask)
        seg_nose_cx = int(nose_coords[1].mean())
        seg_nose_cy = int(nose_coords[0].mean())
        nose_cx, nose_cy = seg_to_frame(seg_nose_cx, seg_nose_cy)
        cv2.circle(image, (nose_cx, nose_cy), 3, (0, 255, 255), -1)

        # bbox 내 위치 비율 표시
        np_x = sigs.get("nose_position_x", 0.5)
        np_y = sigs.get("nose_position_y", 0.5)
        _put(image, f"nx:{np_x:.2f} ny:{np_y:.2f}", nose_cx - 25, nose_cy - 10, (0, 255, 255), 0.25)


def _draw_pose_cube(image, ox, oy, yaw, pitch, roll, size=30):
    """3D 큐브로 head pose 시각화 — 얼굴이 바라보는 방향."""
    import math

    yaw_r = math.radians(yaw)
    pitch_r = math.radians(-pitch)  # 6DRepNet pitch → 이미지 좌표계 반전
    roll_r = math.radians(-roll)    # 6DRepNet roll → 이미지 좌표계 반전

    # Rotation matrices
    Ry = np.array([[math.cos(yaw_r), 0, math.sin(yaw_r)],
                   [0, 1, 0],
                   [-math.sin(yaw_r), 0, math.cos(yaw_r)]])
    Rx = np.array([[1, 0, 0],
                   [0, math.cos(pitch_r), -math.sin(pitch_r)],
                   [0, math.sin(pitch_r), math.cos(pitch_r)]])
    Rz = np.array([[math.cos(roll_r), -math.sin(roll_r), 0],
                   [math.sin(roll_r), math.cos(roll_r), 0],
                   [0, 0, 1]])
    R = Rz @ Ry @ Rx

    # Cube vertices (centered at origin)
    s = size * 0.5
    verts = np.array([
        [-s, -s, -s], [+s, -s, -s], [+s, +s, -s], [-s, +s, -s],  # back face
        [-s, -s, +s], [+s, -s, +s], [+s, +s, +s], [-s, +s, +s],  # front face
    ], dtype=float)

    # Rotate
    rotated = (R @ verts.T).T

    # Project to 2D (simple orthographic)
    cx = ox + size
    cy = oy + size
    pts = [(int(cx + v[0]), int(cy - v[1])) for v in rotated]

    # Edges: back face (0-3), front face (4-7), connecting (0-4, 1-5, ...)
    back_edges = [(0,1), (1,2), (2,3), (3,0)]
    front_edges = [(4,5), (5,6), (6,7), (7,4)]
    side_edges = [(0,4), (1,5), (2,6), (3,7)]

    # Z-sort faces for proper occlusion
    back_z = np.mean([rotated[i][2] for i in [0,1,2,3]])
    front_z = np.mean([rotated[i][2] for i in [4,5,6,7]])

    # Colors: front face = 밝은색 (얼굴이 보는 쪽), back = 어두운색
    front_color = (0, 200, 200)  # cyan (얼굴 방향)
    back_color = (60, 60, 60)
    side_color = (100, 100, 100)

    # Draw back-to-front
    if front_z < back_z:
        # Front face is behind → draw front first
        draw_order = [(front_edges, front_color), (side_edges, side_color), (back_edges, back_color)]
    else:
        draw_order = [(back_edges, back_color), (side_edges, side_color), (front_edges, front_color)]

    for edges, color in draw_order:
        for i, j in edges:
            cv2.line(image, pts[i], pts[j], color, 1, cv2.LINE_AA)

    # Front face fill (반투명 — 얼굴이 보는 면 강조)
    front_pts_arr = np.array([pts[4], pts[5], pts[6], pts[7]])
    if front_z >= back_z:
        overlay = image.copy()
        cv2.fillPoly(overlay, [front_pts_arr], front_color)
        cv2.addWeighted(overlay, 0.2, image, 0.8, 0, image)
        for i, j in front_edges:
            cv2.line(image, pts[i], pts[j], front_color, 2, cv2.LINE_AA)

    # 코 표시 (front face 중앙에 점)
    nose_3d = R @ np.array([0, 0, s * 1.2])
    nose_pt = (int(cx + nose_3d[0]), int(cy - nose_3d[1]))
    cv2.circle(image, nose_pt, 2, (0, 255, 255), -1)

    # Angle label
    _put(image, f"Y:{yaw:.0f} P:{pitch:.0f} R:{roll:.0f}",
         ox, oy + size * 2 + 12, _GRAY, 0.25)


TIMELINE_H = 100
AU_HEATMAP_H = 60

_AU_NAMES = [
    "au1", "au2", "au4", "au5", "au6", "au9",
    "au12", "au15", "au17", "au20", "au25", "au26",
]
_AU_SHORT = ["1", "2", "4", "5", "6", "9", "12", "15", "17", "20", "25", "26"]


def _draw_au_heatmap(canvas: np.ndarray, history: list, video_w: int, y_offset: int):
    """Draw AU activation heatmap: x=time, y=AU channel, color=intensity.

    Each row is one AU, columns are frames, color goes from black(0) to yellow(1).
    """
    total_w = video_w + PANEL_W
    n_au = len(_AU_NAMES)
    n = len(history)

    # Background
    cv2.rectangle(canvas, (0, y_offset), (total_w, y_offset + AU_HEATMAP_H), _PANEL_BG, -1)
    cv2.line(canvas, (0, y_offset), (total_w, y_offset), _DARK_GRAY, 1)

    if n == 0:
        return

    # AU label area
    label_w = 25
    plot_w = total_w - label_w
    row_h = max(1, (AU_HEATMAP_H - 4) // n_au)

    # Row labels
    for i, short in enumerate(_AU_SHORT):
        ry = y_offset + 2 + i * row_h
        _put(canvas, short, 2, ry + row_h - 1, _DARK_GRAY, 0.22)

    # Heatmap cells
    col_w = max(1, plot_w / max(n, 1))
    for frame_i, entry in enumerate(history):
        au_vals = entry.get("au_values", {})
        cx = label_w + int(frame_i * col_w)
        cx2 = label_w + int((frame_i + 1) * col_w)
        if cx2 <= cx:
            cx2 = cx + 1

        for au_i, au_key in enumerate(_AU_NAMES):
            full_key = [k for k in au_vals if k.startswith(au_key)]
            val = au_vals.get(full_key[0], 0.0) if full_key else 0.0
            val = max(0.0, min(1.0, val))

            ry = y_offset + 2 + au_i * row_h
            # Color: black → green → yellow → red
            if val < 0.3:
                r, g, b = 0, int(val / 0.3 * 100), 0
            elif val < 0.6:
                t = (val - 0.3) / 0.3
                r, g, b = int(t * 200), 100 + int(t * 100), 0
            else:
                t = (val - 0.6) / 0.4
                r, g, b = 200 + int(t * 55), int(200 * (1 - t * 0.5)), 0

            cv2.rectangle(canvas, (cx, ry), (cx2, ry + row_h - 1), (b, g, r), -1)

    # Gate-fail overlay on AU heatmap
    for frame_i, entry in enumerate(history):
        if not entry.get("face_detected", False) or not entry.get("gate_passed", False):
            cx = label_w + int(frame_i * col_w)
            cx2 = label_w + int((frame_i + 1) * col_w)
            cx2 = max(cx2, cx + 1)
            y1 = y_offset + 2
            y2 = y_offset + AU_HEATMAP_H - 2
            if cx2 > cx and y2 > y1:
                overlay = canvas[y1:y2, cx:cx2].copy()
                red_fill = np.full_like(overlay, (15, 15, 50))
                cv2.addWeighted(overlay, 0.3, red_fill, 0.7, 0, canvas[y1:y2, cx:cx2])

    # Current position marker
    if n > 0:
        cx = label_w + int((n - 1) * col_w)
        cv2.line(canvas, (cx, y_offset + 2), (cx, y_offset + AU_HEATMAP_H - 2), _WHITE, 1)

    # Title
    _put(canvas, "AU", total_w - 25, y_offset + 12, _CYAN, 0.28)


# Per-category colors (BGR) — distinct, readable on dark background
_EXPR_COLORS = {
    "cheese": (0, 200, 255),    # orange
    "chill":  (200, 180, 0),    # cyan-ish
    "edge":   (180, 0, 180),    # magenta
    "goofy":  (0, 220, 150),    # green-yellow
    "hype":   (0, 150, 255),    # orange-red
    "cut":    (60, 60, 200),    # dim red
    "pass":   (100, 100, 100),  # gray
}


def _get_expr_color(name: str):
    return _EXPR_COLORS.get(name, _GRAY)


def _draw_timeline(canvas: np.ndarray, history: list, video_w: int):
    """Draw timeline: x=frame, y=per-category confidence curves.

    Each expression category gets its own colored line.
    Gate-fail frames marked as red zone at bottom.
    """
    ch, cw = canvas.shape[:2]
    ty = ch - TIMELINE_H
    total_w = video_w + PANEL_W
    plot_h = TIMELINE_H - 22

    # Background
    cv2.rectangle(canvas, (0, ty), (total_w, ch), _PANEL_BG, -1)
    cv2.line(canvas, (0, ty), (total_w, ty), _DARK_GRAY, 1)

    n = len(history)
    if n == 0:
        return

    # Y-axis labels
    _put(canvas, "1.0", 2, ty + 12, _DARK_GRAY, 0.25)
    _put(canvas, "0.5", 2, ty + plot_h // 2 + 5, _DARK_GRAY, 0.25)
    mid_y = ty + plot_h // 2
    cv2.line(canvas, (25, mid_y), (total_w, mid_y), (35, 35, 35), 1)

    px_start = 25
    plot_w = total_w - px_start

    # Collect all category names from history
    all_categories = set()
    for entry in history:
        all_categories.update(entry.get("expression_scores", {}).keys())
    categories = sorted(all_categories)

    # Draw per-category curves (gate fail = dimmed, gate pass = full color)
    for cat in categories:
        color_full = _get_expr_color(cat)
        color_dim = tuple(max(0, c // 3) for c in color_full)
        prev_pt = None
        prev_gated = False
        for i, entry in enumerate(history):
            x = px_start + int(i * plot_w / max(n, 1))
            scores = entry.get("expression_scores", {})
            val = scores.get(cat, 0.0)
            y = ty + plot_h - int(val * plot_h)

            if not entry.get("face_detected", False):
                prev_pt = None
                continue

            gate_passed = entry.get("gate_passed", True)
            color = color_full if gate_passed else color_dim
            pt = (x, y)
            if prev_pt is not None:
                cv2.line(canvas, prev_pt, pt, color, 1, cv2.LINE_AA)
            prev_pt = pt

    # Gate severity: inverted red square wave (severity → height from bottom)
    prev_sev_pt = None
    for i, entry in enumerate(history):
        x = px_start + int(i * plot_w / max(n, 1))
        x2 = px_start + int((i + 1) * plot_w / max(n, 1))
        x2 = max(x2, x + 1)
        sev = entry.get("gate_severity", 0.0)
        gate_passed = entry.get("gate_passed", True)
        face = entry.get("face_detected", False)

        if not face:
            prev_sev_pt = None
            continue

        # Draw severity as filled area from bottom
        sev_h = int(sev * plot_h)
        if sev_h > 0:
            fill_top = ty + plot_h - sev_h
            fill_bot = ty + plot_h
            # Semi-transparent red fill
            if fill_bot > fill_top and x2 > x:
                roi = canvas[fill_top:fill_bot, x:x2]
                if roi.size > 0:
                    red_fill = np.full_like(roi, (30, 30, 160))
                    cv2.addWeighted(roi, 0.4, red_fill, 0.6, 0, canvas[fill_top:fill_bot, x:x2])

        # Severity outline (red square wave)
        sev_y = ty + plot_h - sev_h
        sev_pt = (x, sev_y)
        if prev_sev_pt is not None:
            # Horizontal then vertical (square wave shape)
            cv2.line(canvas, prev_sev_pt, (x, prev_sev_pt[1]), _RED, 1)
            cv2.line(canvas, (x, prev_sev_pt[1]), sev_pt, _RED, 1)
        prev_sev_pt = (x2, sev_y)

    # SHOOT markers: green triangle at top of winner line
    for i, entry in enumerate(history):
        if entry.get("is_shoot", False):
            x = px_start + int(i * plot_w / max(n, 1))
            scores = entry.get("expression_scores", {})
            if scores:
                top_val = max(scores.values())
                y = ty + plot_h - int(top_val * plot_h)
                pts = np.array([[x, y - 7], [x - 3, y - 2], [x + 3, y - 2]], dtype=np.int32)
                cv2.fillPoly(canvas, [pts], _GREEN)

    # Current position marker
    if n > 0:
        cx = px_start + int((n - 1) * plot_w / max(n, 1))
        cv2.line(canvas, (cx, ty + 2), (cx, ty + plot_h), _WHITE, 1)

    # Legend at bottom: category colors
    ly = ch - 5
    lx = px_start
    for cat in categories:
        color = _get_expr_color(cat)
        _put(canvas, cat, lx, ly, color, 0.28)
        lx += len(cat) * 7 + 12
    _put(canvas, "gate_sev", lx + 5, ly, _RED, 0.28)


def _draw_overlay(image: np.ndarray, result: FrameResult, fps: float = 0.0,
                  observations: list = None, history: list = None,
                  coverage_grid: dict = None, sh_9: np.ndarray = None) -> np.ndarray:
    """Compose video frame + right-side panel + bottom timeline + AU heatmap."""
    h, w = image.shape[:2]

    # AU heatmap height
    au_heatmap_h = 60

    # Create canvas: video + panel + AU heatmap + timeline
    total_h = h + au_heatmap_h + TIMELINE_H
    canvas = np.zeros((total_h, w + PANEL_W, 3), dtype=np.uint8)
    canvas[:h, :w] = image
    canvas[:h, w:] = _PANEL_BG

    # Draw face bboxes + head pose on video area
    if observations:
        faces = _extract_faces(observations)
        if faces:
            _draw_face_bbox(canvas, faces, result.gate_passed, video_wh=(w, h))
            # Lighting probes on face
            _draw_lighting_probes(canvas, faces[0], observations, result.signals, w, h)

    # Draw panel
    panel_roi = canvas[:h, w:]
    _draw_panel(panel_roi, result, fps, coverage_grid=coverage_grid or {}, sh_9=sh_9)

    # Draw AU heatmap (between video and timeline)
    if history is not None:
        _draw_au_heatmap(canvas, history, w, h)

    # Draw timeline (below AU heatmap)
    if history is not None:
        _draw_timeline(canvas, history, w)

    # Video overlay: shoot indicator on top-left
    if result.face_detected:
        shoot_color = _GREEN if result.is_shoot else (_RED if not result.gate_passed else _YELLOW)
        cv2.circle(canvas, (20, 20), 8, shoot_color, -1)

    return canvas


def _extract_faces(observations):
    """Extract FaceObservation list from raw observations."""
    for obs in observations:
        if getattr(obs, "source", "") == "face.detect":
            data = getattr(obs, "data", None)
            if data and hasattr(data, "faces"):
                return data.faces
    return []


class DebugV2(MomentscanV2):
    """MomentscanV2 with debug visualization.

    Inherits MomentscanV2 and adds cv2 window overlay in on_frame().
    """

    def __init__(
        self,
        expression_model=None,
        pose_model=None,
        show_window: bool = True,
        output_path: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(expression_model=expression_model, pose_model=pose_model, **kwargs)
        self._show_window = show_window
        self._output_path = output_path
        self._writer = None
        self._last_time = 0.0
        self._fps = 0.0
        self._sfsnet = None  # SfSNet for debug SH sphere
        self._timeline_history: list[dict] = []
        self._coverage_grid: dict[tuple[str, str], float] = {}

    def setup(self):
        super().setup()
        self._last_time = time.monotonic()
        self._fps = 0.0
        self._timeline_history = []
        self._coverage_grid = {}
        logger.info("DebugV2: window=%s, output=%s", self._show_window, self._output_path)
        # DPR for debug SH sphere (9계수 전체 렌더링)
        try:
            from momentscan.face_lighting.dpr import DPRLighting
            self._dpr_debug = DPRLighting()
            self._dpr_debug.initialize()
            logger.info("DebugV2: DPR loaded for SH sphere")
        except Exception as e:
            self._dpr_debug = None
            logger.debug("DebugV2: DPR unavailable (%s)", e)

    def on_frame(self, frame, terminal_results):
        # Collect observations before super() consumes them
        observations = []
        for flow_data in terminal_results:
            observations.extend(getattr(flow_data, "observations", []))

        # Run v2 judgment (accumulates FrameResult)
        super().on_frame(frame, terminal_results)

        # FPS tracking
        now = time.monotonic()
        dt = now - self._last_time
        self._fps = 1.0 / dt if dt > 0 else 0.0
        self._last_time = now

        # Get the result we just appended
        result = self._results[-1]
        image = getattr(frame, "data", None)
        if image is None:
            return True

        # Accumulate timeline history
        j = result.judgment
        # Gate fail일 때도 expression scores를 계산 (debug용, 참고 표시)
        if j.expression_scores:
            scores = dict(j.expression_scores)
        elif result.face_detected and self.judge.expression_strategy is not None:
            scores = self.judge.expression_strategy.predict(result.signals) or {}
        else:
            scores = {}
        au_vals = {k: v for k, v in result.signals.items() if k.startswith("au")}
        gate_severity = _compute_gate_severity(result.signals) if result.face_detected else 0.0
        self._timeline_history.append({
            "gate_passed": result.gate_passed,
            "is_shoot": result.is_shoot,
            "face_detected": result.face_detected,
            "expression_scores": dict(scores),
            "au_values": au_vals,
            "gate_severity": gate_severity,
        })

        # Update coverage grid (best confidence per expression×pose bucket)
        if result.is_shoot:
            key = (result.expression, result.pose)
            if result.expression_conf > self._coverage_grid.get(key, 0.0):
                self._coverage_grid[key] = result.expression_conf

        # Extract SH 9 coefficients for sphere rendering (debug only)
        # DPR은 portrait 크기 입력 → crop_box (확장된 얼굴 영역) 사용
        sh_9 = None
        if self._dpr_debug is not None and result.face_detected and image is not None:
            try:
                # crop_box 추출
                crop_box = None
                for obs in observations:
                    if getattr(obs, "source", "") == "face.parse":
                        data = getattr(obs, "data", None)
                        if data and hasattr(data, "results") and data.results:
                            crop_box = data.results[0].crop_box
                            break

                if crop_box is not None:
                    cb_x, cb_y, cb_w, cb_h = crop_box
                    ih, iw = image.shape[:2]
                    cb_x1 = max(0, cb_x)
                    cb_y1 = max(0, cb_y)
                    cb_x2 = min(iw, cb_x + cb_w)
                    cb_y2 = min(ih, cb_y + cb_h)
                    portrait_crop = image[cb_y1:cb_y2, cb_x1:cb_x2]
                    sh_9 = self._dpr_debug.estimate(portrait_crop)
                else:
                    sh_9 = self._dpr_debug.estimate(image)
            except Exception:
                pass

        # Draw overlay
        debug_image = _draw_overlay(
            image, result, self._fps, observations,
            self._timeline_history, self._coverage_grid, sh_9=sh_9,
        )

        # Initialize writer on first frame
        if self._output_path and self._writer is None:
            h, w = debug_image.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self._writer = cv2.VideoWriter(self._output_path, fourcc, 10, (w, h))

        if self._writer:
            self._writer.write(debug_image)

        if self._show_window:
            cv2.imshow("MomentscanV2 Debug", debug_image)
            # Wait at least 30ms so frames are visible; space=pause, q=quit
            key = cv2.waitKey(30) & 0xFF
            if key == ord("q"):
                return False
            elif key == ord(" "):
                cv2.waitKey(0)  # pause until any key

        return True

    def teardown(self):
        super().teardown()
        if self._writer:
            self._writer.release()
            self._writer = None
            logger.info("Debug video saved: %s", self._output_path)
        if self._show_window:
            cv2.destroyAllWindows()
