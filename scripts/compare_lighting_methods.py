"""세 가지 조명 분석 방법 비교.

A. BiSeNet probe — 코/볼/이마 영역별 밝기 분석
B. Landmarks SH — 3D face normals + Spherical Harmonics fitting
C. SfSNet — 경량 lighting estimator (TODO: 모델 준비 후)

Usage:
    uv run python scripts/compare_lighting_methods.py ~/Videos/reaction_test/251227002408802.mp4
"""

from __future__ import annotations
import sys
import logging
import os
import cv2
import numpy as np

logging.disable(logging.CRITICAL)
os.chdir("/home/hyeonrae/repo/monolith/portrait981")


# ══════════════════════════════════════════════════════════════
# Method A: BiSeNet Probe — 코/볼/이마 영역별 밝기 분석
# ══════════════════════════════════════════════════════════════

# BiSeNet class indices:
# 1: skin, 2-3: l/r eyebrow, 4-5: l/r eye, 6: glasses
# 7-8: l/r ear, 9: earring, 10: nose
# 11: mouth(inner), 12: upper lip, 13: lower lip
# 14: neck, 15: necklace, 16: cloth, 17: hair, 18: hat

def method_a_bisenet_probe(gray, seg_map, bbox_crop_offset):
    """BiSeNet 영역별 조명 probe.

    코를 중심으로 주변부(좌볼, 우볼, 이마, 턱)의 밝기를 측정.
    코는 얼굴에서 가장 돌출되어 조명 방향에 가장 민감.
    """
    if seg_map is None:
        return None

    h, w = gray.shape[:2]
    result = {}

    # 영역별 밝기 추출
    regions = {
        "nose": [10],
        "left_eye": [4],
        "right_eye": [5],
        "left_brow": [2],
        "right_brow": [3],
        "skin": [1],       # 전체 피부 (볼 포함)
        "upper_lip": [12],
        "lower_lip": [13],
    }

    for name, classes in regions.items():
        mask = np.isin(seg_map, classes)
        if mask.sum() < 5:
            result[name] = {"mean": 0, "std": 0, "count": 0}
            continue
        pixels = gray[mask]
        result[name] = {
            "mean": float(pixels.mean()),
            "std": float(pixels.std()),
            "count": int(mask.sum()),
        }

    # 코 좌우 probe: 코 영역의 좌/우 절반 밝기 비교
    nose_mask = np.isin(seg_map, [10])
    if nose_mask.sum() > 10:
        nose_coords = np.where(nose_mask)
        nose_center_x = int(nose_coords[1].mean())

        nose_left = gray[nose_mask & (np.arange(w)[None, :] < nose_center_x)]
        nose_right = gray[nose_mask & (np.arange(w)[None, :] >= nose_center_x)]

        result["nose_left_mean"] = float(nose_left.mean()) if len(nose_left) > 0 else 0
        result["nose_right_mean"] = float(nose_right.mean()) if len(nose_right) > 0 else 0
        result["nose_lr_ratio"] = (
            max(result["nose_left_mean"], 0.1) / max(result["nose_right_mean"], 0.1)
        )

        # 코 그림자 방향: 코 바로 옆 피부의 밝기
        # 코 왼쪽 5px 폭의 피부 vs 코 오른쪽 5px 폭의 피부
        skin_mask = np.isin(seg_map, [1])
        probe_width = 8
        left_probe = skin_mask.copy()
        left_probe[:, nose_center_x:] = False
        left_probe[:, :max(0, nose_center_x - probe_width)] = False
        right_probe = skin_mask.copy()
        right_probe[:, :nose_center_x] = False
        right_probe[:, min(w, nose_center_x + probe_width):] = False

        left_skin = gray[left_probe] if left_probe.sum() > 0 else np.array([0])
        right_skin = gray[right_probe] if right_probe.sum() > 0 else np.array([0])

        result["nose_shadow_left"] = float(left_skin.mean())
        result["nose_shadow_right"] = float(right_skin.mean())
        result["nose_shadow_direction"] = "left" if left_skin.mean() < right_skin.mean() else "right"

    # 이마 vs 턱 (상하 광원)
    skin_mask = np.isin(seg_map, [1])
    if skin_mask.sum() > 20:
        skin_coords = np.where(skin_mask)
        skin_center_y = int(skin_coords[0].mean())
        upper_skin = gray[skin_mask & (np.arange(h)[:, None] < skin_center_y)]
        lower_skin = gray[skin_mask & (np.arange(h)[:, None] >= skin_center_y)]
        result["forehead_brightness"] = float(upper_skin.mean()) if len(upper_skin) > 0 else 0
        result["chin_brightness"] = float(lower_skin.mean()) if len(lower_skin) > 0 else 0

    return result


# ══════════════════════════════════════════════════════════════
# Method B: Landmarks SH — Spherical Harmonics fitting
# ══════════════════════════════════════════════════════════════

# BFM mean face 3D coordinates for approximate normals
# Using a simplified face model with key landmark positions
_MEAN_FACE_3D = np.array([
    # 5-point landmarks approximate 3D (x, y, z) normalized
    [-0.3, -0.3, 0.5],   # left eye
    [0.3, -0.3, 0.5],    # right eye
    [0.0, 0.0, 0.7],     # nose tip
    [-0.2, 0.3, 0.4],    # left mouth
    [0.2, 0.3, 0.4],     # right mouth
], dtype=np.float64)


def _estimate_normals_from_pose(yaw_deg, pitch_deg, roll_deg, n_points=32):
    """Head pose로부터 얼굴 표면의 approximate normals를 생성.

    구면(hemisphere) 위의 점들을 head pose로 회전하여
    얼굴 표면의 normal 방향을 근사.
    """
    yaw = np.radians(yaw_deg)
    pitch = np.radians(pitch_deg)
    roll = np.radians(roll_deg)

    # Rotation matrices
    Ry = np.array([[np.cos(yaw), 0, np.sin(yaw)],
                   [0, 1, 0],
                   [-np.sin(yaw), 0, np.cos(yaw)]])
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(pitch), -np.sin(pitch)],
                   [0, np.sin(pitch), np.cos(pitch)]])
    Rz = np.array([[np.cos(roll), -np.sin(roll), 0],
                   [np.sin(roll), np.cos(roll), 0],
                   [0, 0, 1]])

    R = Rz @ Ry @ Rx

    # Generate hemisphere normals (facing camera = +z direction)
    # Then rotate by head pose
    normals = []
    side = int(np.sqrt(n_points))
    for i in range(side):
        for j in range(side):
            # Map to [-1, 1] range
            u = (i / (side - 1)) * 2 - 1
            v = (j / (side - 1)) * 2 - 1
            r2 = u * u + v * v
            if r2 > 1:
                continue
            # Hemisphere normal
            nz = np.sqrt(1 - r2)
            n = np.array([u * 0.5, v * 0.5, nz])
            n = n / np.linalg.norm(n)
            # Rotate by head pose
            n_rotated = R @ n
            normals.append(n_rotated)

    return np.array(normals)


def _sh_basis_order2(normals):
    """Order-2 Spherical Harmonics basis (9 coefficients).

    Args:
        normals: (N, 3) array of unit normals

    Returns:
        (N, 9) SH basis matrix
    """
    x, y, z = normals[:, 0], normals[:, 1], normals[:, 2]
    N = len(normals)
    basis = np.zeros((N, 9))

    # Order 0
    basis[:, 0] = 1.0                           # Y_00

    # Order 1
    basis[:, 1] = y                              # Y_1,-1
    basis[:, 2] = z                              # Y_1,0
    basis[:, 3] = x                              # Y_1,1

    # Order 2
    basis[:, 4] = x * y                          # Y_2,-2
    basis[:, 5] = y * z                          # Y_2,-1
    basis[:, 6] = 3 * z * z - 1                  # Y_2,0
    basis[:, 7] = x * z                          # Y_2,1
    basis[:, 8] = x * x - y * y                  # Y_2,2

    return basis


def method_b_landmarks_sh(gray, face_bbox, yaw, pitch, roll):
    """Landmarks + head pose → SH lighting estimation.

    1. Head pose로 face normals 근사
    2. Face crop 내 pixel intensity 샘플링
    3. SH basis와 pixel intensity로 least-squares fitting
    → 9 SH coefficients
    """
    h, w = gray.shape[:2]
    bx, by, bw, bh = face_bbox

    # Pixel coords
    if all(0 <= v <= 1.1 for v in face_bbox):
        px1, py1 = int(bx * w), int(by * h)
        px2, py2 = int((bx + bw) * w), int((by + bh) * h)
    else:
        px1, py1 = int(bx), int(by)
        px2, py2 = int(bx + bw), int(by + bh)

    px1, py1 = max(0, px1), max(0, py1)
    px2, py2 = min(w, px2), min(h, py2)

    if px2 <= px1 or py2 <= py1:
        return None

    face_gray = gray[py1:py2, px1:px2].astype(np.float64)
    fh, fw = face_gray.shape

    # Generate normals from head pose
    normals = _estimate_normals_from_pose(yaw, pitch, roll, n_points=64)
    n_normals = len(normals)

    # Sample pixel intensities at corresponding positions
    # Map normal grid positions to face crop pixels
    side = int(np.sqrt(n_normals + 10))
    intensities = []
    valid_normals = []
    idx = 0
    for i in range(side):
        for j in range(side):
            if idx >= n_normals:
                break
            py = int(i / (side - 1) * (fh - 1))
            px = int(j / (side - 1) * (fw - 1))
            if 0 <= py < fh and 0 <= px < fw:
                intensities.append(face_gray[py, px] / 255.0)
                valid_normals.append(normals[idx])
            idx += 1

    if len(intensities) < 9:
        return None

    intensities = np.array(intensities)
    valid_normals = np.array(valid_normals)

    # SH fitting: intensity = SH_basis @ SH_coefficients
    basis = _sh_basis_order2(valid_normals)

    # Least squares: min ||basis @ coeffs - intensities||²
    coeffs, residuals, rank, sv = np.linalg.lstsq(basis, intensities, rcond=None)

    # Interpret SH coefficients
    result = {
        "sh_coefficients": coeffs.tolist(),
        "sh_ambient": float(coeffs[0]),           # overall brightness
        "sh_y_direction": float(coeffs[1]),        # up-down
        "sh_z_direction": float(coeffs[2]),        # front-back
        "sh_x_direction": float(coeffs[3]),        # left-right
        "sh_dominant_direction": _sh_direction(coeffs),
        "sh_directional_strength": float(np.sqrt(coeffs[1]**2 + coeffs[2]**2 + coeffs[3]**2)),
    }

    return result


def _sh_direction(coeffs):
    """SH 계수에서 dominant light direction 추출."""
    x, y, z = coeffs[3], coeffs[1], coeffs[2]
    mag = np.sqrt(x**2 + y**2 + z**2)
    if mag < 0.01:
        return "ambient"

    # Dominant axis
    ax = abs(x)
    ay = abs(y)
    az = abs(z)

    if ax > ay and ax > az:
        return "right" if x > 0 else "left"
    elif ay > ax and ay > az:
        return "bottom" if y > 0 else "top"
    else:
        return "front" if z > 0 else "back"


# ══════════════════════════════════════════════════════════════
# Main: 비교 실행
# ══════════════════════════════════════════════════════════════

def main():
    video_path = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser(
        "~/Videos/reaction_test/251227002408802.mp4")

    from momentscan.app import Momentscan

    app = Momentscan(expression_model="models/bind_v12.pkl", pose_model="models/pose_v10.pkl")
    results = app.run(video_path, fps=2)

    shoot = sorted([r for r in results if r.is_shoot],
                   key=lambda r: r.signals.get("lighting_ratio", 0), reverse=True)[:10]

    print(f"Video: {os.path.basename(video_path)}")
    print(f"Top 10 SHOOT by lighting_ratio:\n")

    for r in shoot:
        s = r.signals
        print(f"{'='*70}")
        print(f"Frame #{r.frame_idx}  {r.expression} {r.pose}  conf={r.expression_conf:.0%}")
        print(f"  Current: ratio={s.get('lighting_ratio',1):.2f} std={s.get('face_brightness_std',0):.1f} "
              f"dir_x={s.get('light_direction_x',0):.0f} dir_y={s.get('light_direction_y',0):.0f} "
              f"rembrandt={s.get('rembrandt_score',0):.0f} hardness={s.get('light_hardness',0):.2f}")

        img = r.image
        if img is None:
            print("  (no image)")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)

        # --- Method A: BiSeNet probe ---
        # Get segmentation from observations (stored during analysis)
        # We need to re-run face.parse for this frame, or use stored data
        # For now, use a simple approach: run face.parse on the frame
        try:
            from vpx.face_parse import FaceParseAnalyzer
            from vpx.face_detect import FaceDetectionAnalyzer
            from visualbase.core.frame import Frame

            # Create frame
            h, w = img.shape[:2]
            frame = Frame(data=img, frame_id=r.frame_idx, t_src_ns=0, width=w, height=h)

            # Run face.detect
            if not hasattr(main, '_det'):
                main._det = FaceDetectionAnalyzer()
                main._det.initialize()
                main._parse = FaceParseAnalyzer()
                main._parse.initialize()

            det_obs = main._det.process(frame)
            parse_obs = main._parse.process(frame, deps={"face.detect": det_obs})

            if parse_obs and parse_obs.data and parse_obs.data.results:
                seg = parse_obs.data.results[0].class_map
                # Crop seg to face bbox
                faces = det_obs.data.faces
                face = max(faces, key=lambda f: f.area_ratio)
                bbox = face.bbox
                bx, by, bw, bh = bbox
                if all(0 <= v <= 1.1 for v in bbox):
                    fpx1, fpy1 = int(bx * w), int(by * h)
                    fpx2, fpy2 = int((bx + bw) * w), int((by + bh) * h)
                else:
                    fpx1, fpy1 = int(bx), int(by)
                    fpx2, fpy2 = int(bx + bw), int(by + bh)

                fpx1, fpy1 = max(0, fpx1), max(0, fpy1)
                fpx2, fpy2 = min(w, fpx2), min(h, fpy2)

                # Resize seg to frame size if needed
                if seg.shape[:2] != (h, w):
                    seg = cv2.resize(seg, (w, h), interpolation=cv2.INTER_NEAREST)

                seg_crop = seg[fpy1:fpy2, fpx1:fpx2]
                gray_crop = gray[fpy1:fpy2, fpx1:fpx2]

                probe = method_a_bisenet_probe(gray_crop, seg_crop, (fpx1, fpy1))
                if probe:
                    nose_lr = probe.get("nose_lr_ratio", 1)
                    shadow_dir = probe.get("nose_shadow_direction", "?")
                    forehead = probe.get("forehead_brightness", 0)
                    chin = probe.get("chin_brightness", 0)
                    nose_shadow_l = probe.get("nose_shadow_left", 0)
                    nose_shadow_r = probe.get("nose_shadow_right", 0)
                    print(f"  [A] BiSeNet probe: nose_lr={nose_lr:.2f} shadow={shadow_dir} "
                          f"nose_shadow(L={nose_shadow_l:.0f} R={nose_shadow_r:.0f}) "
                          f"forehead={forehead:.0f} chin={chin:.0f}")

                # Method B: Landmarks SH
                yaw = s.get("head_yaw_dev", 0)
                pitch = s.get("head_pitch", 0)
                roll = s.get("head_roll", 0)
                sh = method_b_landmarks_sh(gray, bbox, yaw, pitch, roll)
                if sh:
                    print(f"  [B] SH fitting: ambient={sh['sh_ambient']:.3f} "
                          f"x={sh['sh_x_direction']:.3f} y={sh['sh_y_direction']:.3f} "
                          f"z={sh['sh_z_direction']:.3f} "
                          f"dir={sh['sh_dominant_direction']} "
                          f"strength={sh['sh_directional_strength']:.3f}")
                    print(f"      SH[0-8]: {[f'{c:.3f}' for c in sh['sh_coefficients']]}")

        except Exception as e:
            print(f"  Error: {e}")

    print(f"\n[C] SfSNet: TODO — 모델 다운로드 후 적용 예정")


if __name__ == "__main__":
    main()
