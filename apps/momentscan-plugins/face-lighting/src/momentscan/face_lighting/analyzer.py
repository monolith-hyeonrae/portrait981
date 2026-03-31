"""FaceLightingAnalyzer — 얼굴 조명 패턴 분석.

세 가지 분석 방법:
  A. BiSeNet probe — 코/볼/이마 영역별 상대 밝기 분석
  B. Landmarks SH — 3D normals + Spherical Harmonics fitting
  C. DPR — Deep Portrait Relighting (Phase 2)

모든 밝기 표현은 상대 contrast 기반 (Michelson contrast):
  contrast = (bright - dark) / (bright + dark + ε)
  → 피부톤, 날씨, 카메라 노출에 불변

출력 signals:
  # 기본 (4분면 기반)
  lighting_ratio, face_brightness_std, highlight_ratio, shadow_ratio
  light_direction_x, light_direction_y, rembrandt_score, light_hardness
  # BiSeNet probe (코 중심)
  nose_contrast_lr      — 코 좌우 Michelson contrast (-1~1, 양수=좌측이 밝음)
  nose_shadow_contrast  — 코 옆 피부 좌우 contrast
  forehead_chin_contrast — 이마-턱 contrast (양수=이마가 밝음)
  # SH (Spherical Harmonics)
  sh_ambient, sh_dir_x, sh_dir_y, sh_dir_strength
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import cv2
import numpy as np

from vpx.sdk import Module, Observation

logger = logging.getLogger("momentscan.face_lighting")

_EPS = 1.0  # epsilon for Michelson contrast denominator


def _michelson(a: float, b: float) -> float:
    """Michelson contrast: (a - b) / (a + b + ε). Range: [-1, 1]."""
    return (a - b) / (a + b + _EPS)


@dataclass
class LightingOutput:
    """조명 분석 결과."""
    # 4분면 기반
    lighting_pattern: str = "flat"
    lighting_ratio: float = 1.0
    shadow_side: str = "center"
    face_brightness_mean: float = 0.0
    face_brightness_std: float = 0.0
    highlight_ratio: float = 0.0
    shadow_ratio: float = 0.0
    left_brightness: float = 0.0
    right_brightness: float = 0.0
    # 9분면 분석
    light_direction: str = "flat"
    light_direction_x: float = 0.0       # Michelson 좌우 (양수=우측 밝음)
    light_direction_y: float = 0.0       # Michelson 상하 (양수=상단 밝음)
    rembrandt_triangle: bool = False
    light_hardness: float = 0.0
    # DPR SH
    sh_dir_x: float = 0.0
    sh_dir_y: float = 0.0
    sh_dir_strength: float = 0.0
    sh_coefficients: list = field(default_factory=lambda: [0.0]*9)  # raw 9계수


class FaceLightingAnalyzer(Module):
    """Portrait 조명 패턴 분석기.

    세 가지 방법으로 조명을 분석하고 상대 contrast 기반으로 표현.
    """

    name = "face.lighting"
    depends = ("face.detect", "face.parse")

    def initialize(self):
        self._dpr = None
        try:
            from momentscan.face_lighting.dpr import DPRLighting
            self._dpr = DPRLighting()
            self._dpr.initialize()
            logger.info("FaceLightingAnalyzer: DPR loaded")
        except Exception as e:
            logger.warning("FaceLightingAnalyzer: DPR unavailable (%s), using pixel methods only", e)
        logger.info("FaceLightingAnalyzer initialized")

    def process(self, frame, deps=None, **kwargs) -> Observation:
        signals = {}
        data = LightingOutput()
        _obs = lambda: Observation(
            source=self.name, signals=signals, data=data,
            frame_id=getattr(frame, "frame_id", 0),
            t_ns=getattr(frame, "t_src_ns", 0),
        )

        if deps is None:
            return _obs()

        det_obs = deps.get("face.detect")
        if det_obs is None or det_obs.data is None:
            return _obs()

        faces = getattr(det_obs.data, "faces", [])
        if not faces:
            return _obs()

        face = max(faces, key=lambda f: getattr(f, "area_ratio", 0.0))
        bbox = getattr(face, "bbox", None)
        if bbox is None or len(bbox) != 4:
            return _obs()

        image = getattr(frame, "data", None)
        if image is None:
            return _obs()

        h, w = image.shape[:2]
        bx, by, bw, bh = bbox
        if all(0 <= v <= 1.1 for v in bbox):
            px1, py1 = int(bx * w), int(by * h)
            px2, py2 = int((bx + bw) * w), int((by + bh) * h)
        else:
            px1, py1 = int(bx), int(by)
            px2, py2 = int(bx + bw), int(by + bh)

        px1, py1 = max(0, px1), max(0, py1)
        px2, py2 = min(w, px2), min(h, py2)
        if px2 <= px1 or py2 <= py1:
            return _obs()

        face_crop = image[py1:py2, px1:px2]
        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY).astype(np.float32)
        face_mask = self._get_face_mask(deps, gray.shape, px1, py1, px2, py2, w, h)

        if face_mask is not None and face_mask.shape == gray.shape:
            valid_pixels = gray[face_mask > 0]
        else:
            valid_pixels = gray.flatten()

        if len(valid_pixels) < 10:
            return _obs()

        # seg_map 먼저 추출
        seg_map = self._get_seg_map(deps, gray.shape, px1, py1, px2, py2, w, h)

        # ── 기본 좌우/상하 분석 (skin only) ──
        self._compute_basic_lighting(gray, face_mask, valid_pixels, seg_map, data)

        # ── 9분면 분석 + Rembrandt (skin pixel만 사용) ──
        self._compute_sectors(gray, face_mask, seg_map, data)

        # ── Method C: DPR SH (preferred) or Method B: geometric SH (fallback) ──
        #
        # ── Method C: DPR SH (이미지 좌표계, 반전 불필요) ──
        # DPR은 portrait 크기 입력을 기대 → crop_box (확장된 portrait 영역) 사용
        # face bbox만 쓰면 배경이 없어서 좋지만 너무 타이트 → DPR 성능 저하
        # crop_box는 BiSeNet이 사용하는 확장 영역 (expand=1.3, 1:1 비율)
        if self._dpr is not None:
            try:
                _, crop_box = self._get_parse_result(deps)
                if crop_box is not None:
                    cb_x, cb_y, cb_w, cb_h = crop_box
                    cb_x1 = max(0, cb_x)
                    cb_y1 = max(0, cb_y)
                    cb_x2 = min(w, cb_x + cb_w)
                    cb_y2 = min(h, cb_y + cb_h)
                    portrait_crop = image[cb_y1:cb_y2, cb_x1:cb_x2]
                else:
                    portrait_crop = face_crop

                desc = self._dpr.estimate_lighting_descriptor(portrait_crop)
                data.sh_dir_x = desc["sh_dir_x"]
                data.sh_dir_y = desc["sh_dir_y"]
                data.sh_dir_strength = desc["sh_dir_strength"]
                data.sh_coefficients = desc.get("sh_coefficients", [0.0]*9)
            except Exception:
                pass  # 4분면 분석 결과만 사용

        # ── Build signals (모두 상대값) ──
        signals = {
            # 기본
            "lighting_ratio": data.lighting_ratio,
            "face_brightness_std": data.face_brightness_std,
            "highlight_ratio": data.highlight_ratio,
            "shadow_ratio": data.shadow_ratio,
            # 9분면 방향 (skin only, Michelson contrast)
            "light_direction_x": data.light_direction_x,
            "light_direction_y": data.light_direction_y,
            "rembrandt_score": 1.0 if data.rembrandt_triangle else 0.0,
            "light_hardness": data.light_hardness,
            # DPR SH summary
            "sh_dir_x": data.sh_dir_x,
            "sh_dir_y": data.sh_dir_y,
            "sh_dir_strength": data.sh_dir_strength,
            # DPR SH 9계수 raw
            "sh_0": data.sh_coefficients[0] if len(data.sh_coefficients) > 0 else 0.0,
            "sh_1": data.sh_coefficients[1] if len(data.sh_coefficients) > 1 else 0.0,
            "sh_2": data.sh_coefficients[2] if len(data.sh_coefficients) > 2 else 0.0,
            "sh_3": data.sh_coefficients[3] if len(data.sh_coefficients) > 3 else 0.0,
            "sh_4": data.sh_coefficients[4] if len(data.sh_coefficients) > 4 else 0.0,
            "sh_5": data.sh_coefficients[5] if len(data.sh_coefficients) > 5 else 0.0,
            "sh_6": data.sh_coefficients[6] if len(data.sh_coefficients) > 6 else 0.0,
            "sh_7": data.sh_coefficients[7] if len(data.sh_coefficients) > 7 else 0.0,
            "sh_8": data.sh_coefficients[8] if len(data.sh_coefficients) > 8 else 0.0,
        }

        return _obs()

    # ── Method A: 기본 좌우/상하 (상대 contrast) ──

    def _compute_basic_lighting(self, gray, mask, valid_pixels, seg, data):
        """기본 좌우 분석 — skin pixel 우선, face mask fallback."""
        crop_h, crop_w = gray.shape[:2]
        mid_x = crop_w // 2

        data.face_brightness_mean = float(valid_pixels.mean())
        data.face_brightness_std = float(valid_pixels.std())

        # skin mask 우선 (머리카락/배경 제외)
        if seg is not None and seg.shape == gray.shape:
            skin = (seg == 1).astype(np.float32)
        elif mask is not None and mask.shape == gray.shape:
            skin = mask
        else:
            skin = None

        if skin is not None:
            left_px = gray[:, :mid_x][skin[:, :mid_x] > 0]
            right_px = gray[:, mid_x:][skin[:, mid_x:] > 0]
        else:
            left_px = gray[:, :mid_x].flatten()
            right_px = gray[:, mid_x:].flatten()

        left_mean = float(left_px.mean()) if len(left_px) > 0 else 0.0
        right_mean = float(right_px.mean()) if len(right_px) > 0 else 0.0

        data.left_brightness = left_mean
        data.right_brightness = right_mean
        bright_side = max(left_mean, right_mean)
        dark_side = max(min(left_mean, right_mean), 1.0)
        data.lighting_ratio = bright_side / dark_side

        if abs(left_mean - right_mean) < 5:
            data.shadow_side = "center"
        elif left_mean < right_mean:
            data.shadow_side = "left"
        else:
            data.shadow_side = "right"

        hl_threshold = data.face_brightness_mean + 2.0 * data.face_brightness_std
        sh_threshold = max(data.face_brightness_mean - 1.5 * data.face_brightness_std, 10)
        data.highlight_ratio = float((valid_pixels > hl_threshold).sum() / len(valid_pixels))
        data.shadow_ratio = float((valid_pixels < sh_threshold).sum() / len(valid_pixels))

        if data.lighting_ratio > 1.4 and data.face_brightness_std > 35:
            data.lighting_pattern = "dramatic"
        elif data.lighting_ratio > 1.15 and data.face_brightness_std > 20:
            data.lighting_pattern = "natural"
        elif data.highlight_ratio > 0.15:
            data.lighting_pattern = "harsh"
        elif data.shadow_ratio > 0.40:
            data.lighting_pattern = "backlit"
        else:
            data.lighting_pattern = "flat"

    # ── 9분면 분석 (skin only) ──

    def _compute_sectors(self, gray, mask, seg, data):
        """9분면 밝기 분석 — 원형 9섹터 (40° 간격) + 중심.

        skin pixel만 사용하여 머리카락/배경 오염 방지.
        4분면보다 구면 분석에 가까운 방향 해상도.

        섹터 배치 (이미지 좌표계):
            sector 0: 우측 (0°)       sector 4: 하단 (180°)
            sector 1: 우상 (40°)      sector 5: 좌하 (220°)
            sector 2: 상단 (80°)      sector 6: 좌측 (260°)
            sector 3: 좌상 (120°)     sector 7: 좌하근처 (300°)
            sector 8: 우하 (340°)     center: 중앙 영역
        """
        import math
        crop_h, crop_w = gray.shape[:2]
        mid_x, mid_y = crop_w // 2, crop_h // 2

        # skin mask
        if seg is not None and seg.shape == gray.shape:
            skin = (seg == 1).astype(np.float32)
        elif mask is not None and mask.shape == gray.shape:
            skin = mask
        else:
            skin = np.ones_like(gray)

        # 9 섹터 (40° 간격) + 중심
        n_sectors = 9
        sector_angle = 2 * math.pi / n_sectors
        radius = min(mid_x, mid_y) * 0.85
        center_r = radius * 0.3  # 중심 영역 반경

        sector_means = np.zeros(n_sectors)
        sector_counts = np.zeros(n_sectors)
        center_vals = []

        for py in range(crop_h):
            for px in range(crop_w):
                if skin[py, px] < 0.5:
                    continue
                dx = px - mid_x
                dy = py - mid_y
                dist = math.sqrt(dx*dx + dy*dy)

                if dist < center_r:
                    center_vals.append(gray[py, px])
                elif dist < radius:
                    # 각도 계산 (이미지 좌표: 우측=0°, 반시계 방향)
                    angle = math.atan2(-dy, dx)  # -dy: 이미지 y 반전
                    if angle < 0:
                        angle += 2 * math.pi
                    sector_idx = int(angle / sector_angle) % n_sectors
                    sector_means[sector_idx] += gray[py, px]
                    sector_counts[sector_idx] += 1

        # 평균 계산
        for i in range(n_sectors):
            if sector_counts[i] > 0:
                sector_means[i] /= sector_counts[i]

        center_mean = float(np.mean(center_vals)) if center_vals else 0.0
        global_mean = float(np.mean(sector_means[sector_counts > 0])) if (sector_counts > 0).any() else center_mean

        # ── Signal 계산 ──

        # light_direction_x: 우측 섹터(0,1,8) vs 좌측 섹터(4,5,6) contrast
        right_secs = [i for i in [0, 1, 8] if sector_counts[i] > 0]
        left_secs = [i for i in [4, 5, 6] if sector_counts[i] > 0]
        if right_secs and left_secs:
            r_mean = np.mean([sector_means[i] for i in right_secs])
            l_mean = np.mean([sector_means[i] for i in left_secs])
            data.light_direction_x = _michelson(float(r_mean), float(l_mean))

        # light_direction_y: 상단 섹터(2,3,1) vs 하단 섹터(5,6,7) contrast
        top_secs = [i for i in [1, 2, 3] if sector_counts[i] > 0]
        bot_secs = [i for i in [5, 6, 7] if sector_counts[i] > 0]
        if top_secs and bot_secs:
            t_mean = np.mean([sector_means[i] for i in top_secs])
            b_mean = np.mean([sector_means[i] for i in bot_secs])
            data.light_direction_y = _michelson(float(t_mean), float(b_mean))

        # Rembrandt: 한쪽이 밝고 반대 하단에 밝은 점
        data.rembrandt_triangle = False
        if data.lighting_ratio > 1.3:
            bright_idx = int(np.argmax(sector_means)) if (sector_counts > 0).any() else -1
            if bright_idx >= 0:
                opposite_idx = (bright_idx + n_sectors // 2) % n_sectors
                if sector_counts[opposite_idx] > 0 and global_mean > 0:
                    opp_ratio = sector_means[opposite_idx] / (global_mean + _EPS)
                    if 0.6 < opp_ratio < 0.9:
                        data.rembrandt_triangle = True

        # Light hardness: 인접 섹터 간 최대 차이
        max_grad = 0
        for i in range(n_sectors):
            j = (i + 1) % n_sectors
            if sector_counts[i] > 0 and sector_counts[j] > 0:
                grad = abs(sector_means[i] - sector_means[j]) / (global_mean + _EPS)
                max_grad = max(max_grad, grad)
        data.light_hardness = min(float(max_grad), 1.0)

    # ── Segmentation helpers ──

    def _get_parse_result(self, deps):
        """face.parse 결과에서 seg map + crop_box 추출."""
        parse_obs = deps.get("face.parse")
        if parse_obs is None:
            return None, None
        data = getattr(parse_obs, "data", None)
        if data is None:
            return None, None
        results = getattr(data, "results", [])
        if not results:
            return None, None
        seg = results[0].class_map
        crop_box = results[0].crop_box  # (x, y, w, h) pixel coords of expanded crop
        if seg is None:
            return None, None
        return seg, crop_box

    def _get_face_mask(self, deps, gray_shape, px1, py1, px2, py2, w, h):
        """BiSeNet → 얼굴 영역 binary mask (face bbox 크기로 매핑)."""
        seg, crop_box = self._get_parse_result(deps)
        if seg is None or crop_box is None:
            return None

        face_classes = [1, 2, 3, 4, 5, 10, 11, 12, 13]
        full_mask = np.isin(seg, face_classes).astype(np.float32)

        # seg map → face bbox 영역 추출 (crop_box 좌표계 경유)
        face_mask = self._seg_to_bbox(full_mask, crop_box, px1, py1, px2, py2)
        if face_mask is None:
            return None
        if face_mask.shape != gray_shape:
            face_mask = cv2.resize(face_mask, (gray_shape[1], gray_shape[0]),
                                   interpolation=cv2.INTER_NEAREST)
        return face_mask

    def _get_seg_map(self, deps, gray_shape, px1, py1, px2, py2, w, h):
        """BiSeNet → class index map (face bbox 크기로 매핑)."""
        seg, crop_box = self._get_parse_result(deps)
        if seg is None or crop_box is None:
            return None

        seg_crop = self._seg_to_bbox(seg, crop_box, px1, py1, px2, py2)
        if seg_crop is None:
            return None
        if seg_crop.shape != gray_shape:
            seg_crop = cv2.resize(seg_crop, (gray_shape[1], gray_shape[0]),
                                  interpolation=cv2.INTER_NEAREST)
        return seg_crop

    def _seg_to_bbox(self, seg_map, crop_box, px1, py1, px2, py2):
        """seg map (crop_box 좌표계) → face bbox 영역 추출.

        seg map은 crop_box (확장된 portrait 영역) 전체를 커버.
        face bbox는 crop_box 안의 일부분.
        face bbox에 해당하는 seg map 영역을 잘라낸다.
        """
        cb_x, cb_y, cb_w, cb_h = crop_box
        seg_h, seg_w = seg_map.shape[:2]

        if cb_w < 1 or cb_h < 1:
            return None

        # face bbox 좌표를 seg map 좌표로 변환
        # face pixel (px1, py1) → crop_box 내 상대 위치 → seg 좌표
        sx1 = int((px1 - cb_x) / cb_w * seg_w)
        sy1 = int((py1 - cb_y) / cb_h * seg_h)
        sx2 = int((px2 - cb_x) / cb_w * seg_w)
        sy2 = int((py2 - cb_y) / cb_h * seg_h)

        # Clamp
        sx1, sy1 = max(0, sx1), max(0, sy1)
        sx2, sy2 = min(seg_w, sx2), min(seg_h, sy2)

        if sx2 <= sx1 or sy2 <= sy1:
            return None

        return seg_map[sy1:sy2, sx1:sx2]

    def cleanup(self):
        pass
