"""Batch highlight detection engine.

highlight_rules.md 설계를 따르는 배치 분석 엔진:
1. Numeric feature delta 계산
2. Per-video 정규화 (MAD z-score)
3. QualityGate (hard filter)
4. Scoring: quality_score × impact_score
5. Temporal smoothing + peak detection
6. Window 생성 + best frame 선택
"""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np

from momentscan.algorithm.batch.field_mapping import (
    PIPELINE_DELTA_SPECS,
    PIPELINE_DERIVED_FIELDS,
)
from momentscan.algorithm.batch.types import (
    FrameRecord,
    HighlightConfig,
    HighlightResult,
    HighlightWindow,
)

logger = logging.getLogger(__name__)


class BatchHighlightEngine:
    """Per-video 정규화 + peak detection 기반 하이라이트 분석."""

    def __init__(self, config: Optional[HighlightConfig] = None):
        self.config = config or HighlightConfig()

    def analyze(self, records: List[FrameRecord]) -> HighlightResult:
        """전체 비디오의 프레임 레코드를 분석하여 하이라이트 구간을 찾는다.

        Args:
            records: on_frame()에서 축적한 FrameRecord 리스트.

        Returns:
            HighlightResult with detected windows.
        """
        if len(records) < 3:
            logger.info("Too few frames (%d) — skipping batch analysis", len(records))
            return HighlightResult(frame_count=len(records), config=self.config)

        cfg = self.config
        n = len(records)
        logger.info("[1/7] Building feature arrays (%d frames, %d features)", n, 10)

        # 1. 수치 배열 구성
        arrays = self._build_arrays(records)

        # 2. Temporal delta 계산
        logger.info("[2/7] Computing temporal deltas (EMA baseline)")
        deltas = self._compute_deltas(arrays)

        # 3. Per-video 정규화 (MAD z-score → [0,1] rescale)
        logger.info("[3/7] Normalizing per-video (MAD z-score + rescale)")
        normed = self._normalize(deltas)
        normed = self._rescale_normed(normed)

        # 4. Quality gate (hard filter)
        gate_mask = self._apply_quality_gate(records)
        gate_pass = int(gate_mask.sum())
        logger.info("[4/7] Quality gate: %d/%d frames passed (%.0f%%)",
                     gate_pass, n, 100.0 * gate_pass / n)

        # 5. Quality score (연속 품질)
        quality_scores = self._compute_quality_scores(arrays, normed)

        # 6. Impact score
        impact_scores = self._compute_impact_scores(normed)

        # 7. Final score = quality × impact (gate 통과 프레임만)
        final_scores = np.where(gate_mask, quality_scores * impact_scores, 0.0)
        logger.info("[5/7] Scoring complete (quality × impact)")

        # 8. Temporal smoothing (EMA)
        smoothed = self._smooth_ema(final_scores, alpha=cfg.smoothing_alpha)
        logger.info("[6/7] Temporal smoothing (EMA α=%.2f)", cfg.smoothing_alpha)

        # 9. Peak detection
        peaks = self._detect_peaks(smoothed)
        logger.info("[7/7] Peak detection: %d peaks found", len(peaks))

        # 10. Window 생성 + best frame 선택
        windows = self._generate_windows(peaks, records, final_scores, normed)

        result = HighlightResult(
            windows=windows,
            frame_count=n,
            config=self.config,
        )

        # 검증 산출물용 중간 데이터 저장
        result._timeseries = {
            "records": records,
            "gate_mask": gate_mask,
            "quality_scores": quality_scores,
            "impact_scores": impact_scores,
            "final_scores": final_scores,
            "smoothed": smoothed,
            "peaks": peaks,
            "deltas": deltas,
            "arrays": arrays,
            "normed": normed,
        }

        return result

    # ── Step 1: 배열 구성 ──

    def _build_arrays(self, records: List[FrameRecord]) -> dict[str, np.ndarray]:
        """FrameRecord 리스트를 feature별 numpy 배열로 변환.

        delta 대상 필드 + derived 필드의 소스 + scoring에 필요한 추가 필드를
        PIPELINE_DELTA_SPECS / PIPELINE_DERIVED_FIELDS에서 자동 수집한다.
        """
        # delta 대상 필드
        fields: set[str] = {spec.record_field for spec in PIPELINE_DELTA_SPECS}
        # derived 필드의 소스
        for derived in PIPELINE_DERIVED_FIELDS:
            fields.update(derived.source_fields)
        # scoring에 필요하지만 delta/derived에 없는 추가 필드
        fields.update(("blur_score", "eye_open_ratio", "face_confidence",
                        "face_recog_quality", "embed_delta_face"))

        return {
            field: np.array([getattr(r, field) for r in records])
            for field in sorted(fields)
        }

    # ── Step 2: Temporal delta ──

    def _compute_deltas(self, arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """EMA 기준선 대비 변화량(delta)을 계산.

        delta(t) = |feature(t) - EMA(feature, alpha)|
        highlight_rules.md §4.D
        """
        cfg = self.config
        deltas = {}
        for spec in PIPELINE_DELTA_SPECS:
            arr = arrays[spec.record_field]
            ema = self._compute_ema(arr, cfg.delta_alpha)
            deltas[spec.record_field] = np.abs(arr - ema)

        # embed_delta_face is already a delta → pass-through (no "delta of delta")
        if "embed_delta_face" in arrays:
            deltas["embed_delta_face"] = arrays["embed_delta_face"]

        # Derived fields
        for derived in PIPELINE_DERIVED_FIELDS:
            if derived.name == "head_velocity":
                yaw = arrays["head_yaw"]
                pitch = arrays["head_pitch"]
                dt = 1.0 / cfg.fps
                dyaw = np.abs(np.diff(yaw, prepend=yaw[0])) / dt
                dpitch = np.abs(np.diff(pitch, prepend=pitch[0])) / dt
                deltas["head_velocity"] = np.sqrt(dyaw**2 + dpitch**2)

        return deltas

    # ── Step 3: Per-video 정규화 ──

    def _normalize(self, deltas: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """MAD 기반 z-score 정규화.

        z = (x - median) / MAD
        highlight_rules.md §5
        """
        normed = {}
        for key, arr in deltas.items():
            median = np.median(arr)
            mad = np.median(np.abs(arr - median))
            if mad > 1e-8:
                normed[key] = (arr - median) / mad
            else:
                normed[key] = np.zeros_like(arr)
        return normed

    def _rescale_normed(self, normed: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """MAD z-score를 [0, 1] 범위로 재정규화.

        MAD z-score는 feature별 레인지가 크게 달라 가중합에서 불균형을 초래.
        (예: smile z_max=16 vs head_yaw z_max=5 → weight 무시하고 smile 지배)

        Percentile capping으로 양수 z-score를 [0, 1]로 통일하여
        weight가 실질적 기여도를 정확히 제어하도록 한다.
        """
        pct = self.config.normed_cap_percentile
        rescaled = {}
        for key, arr in normed.items():
            pos = np.maximum(arr, 0.0)
            cap = np.percentile(pos, pct)
            if cap > 1e-8:
                rescaled[key] = np.minimum(pos / cap, 1.0)
            else:
                rescaled[key] = np.zeros_like(arr)
        return rescaled

    # ── Step 4: Quality gate ──

    def _apply_quality_gate(self, records: List[FrameRecord]) -> np.ndarray:
        """Hard filter: gate를 통과하지 못하면 score=0.

        highlight_rules.md §6 Step 1
        """
        cfg = self.config
        mask = np.ones(len(records), dtype=bool)

        for i, r in enumerate(records):
            if not r.face_detected:
                mask[i] = False
            elif r.face_confidence < cfg.gate_face_confidence:
                mask[i] = False
            elif r.face_area_ratio < cfg.gate_face_area_ratio:
                mask[i] = False
            elif r.blur_score > 0 and r.blur_score < cfg.gate_blur_min:
                # blur_score=0은 미측정 → 통과
                mask[i] = False
            elif r.brightness > 0 and not (cfg.gate_exposure_min <= r.brightness <= cfg.gate_exposure_max):
                # brightness=0은 미측정 → 통과
                mask[i] = False
            elif r.eye_open_ratio > 0 and r.eye_open_ratio < cfg.gate_eye_open_min:
                # eye_open_ratio=0은 미측정 → 통과
                mask[i] = False

        return mask

    # ── Step 5: Quality score ──

    def _compute_quality_scores(
        self,
        arrays: dict[str, np.ndarray],
        normed: dict[str, np.ndarray],
    ) -> np.ndarray:
        """연속 품질 점수.

        ArcFace face_recog_quality 사용 시:
          quality = blur * w_blur + face_size * w_size + face_recog * w_recog
        ArcFace 없는 프레임은 frontalness fallback.

        highlight_rules.md §6 Step 2
        """
        cfg = self.config

        # Per-video 정규화된 blur (높을수록 좋음 → 0~1 clamp)
        blur = arrays["blur_score"]
        blur_normed = self._minmax_normalize(blur)

        # Face size (face_area_ratio, 높을수록 좋음)
        face_size_normed = self._minmax_normalize(arrays["face_area_ratio"])

        # ArcFace quality (already 0~1 range)
        face_recog = arrays["face_recog_quality"]

        # Frontalness fallback (1 - |yaw|/max_yaw, 정면에 가까울수록 1)
        frontalness = np.clip(1.0 - np.abs(arrays["head_yaw"]) / cfg.frontalness_max_yaw, 0.0, 1.0)

        # Per-frame: use face_recog where available, frontalness as fallback
        has_recog = face_recog > 0
        face_quality = np.where(has_recog, face_recog, frontalness)
        recog_weight = np.where(has_recog, cfg.quality_face_recog_weight, cfg.quality_frontalness_weight)

        # Rebalance: blur + face_size weights must adjust with recog weight
        total_other = cfg.quality_blur_weight + cfg.quality_face_size_weight
        total = total_other + recog_weight
        quality = (
            (cfg.quality_blur_weight / total) * blur_normed
            + (cfg.quality_face_size_weight / total) * face_size_normed
            + (recog_weight / total) * face_quality
        )
        return np.clip(quality, 0.0, 1.0)

    # ── Step 6: Impact score ──

    def _compute_impact_scores(self, normed: dict[str, np.ndarray]) -> np.ndarray:
        """감정/동작 변화 점수.

        highlight_rules.md §6 Step 3
        """
        cfg = self.config

        # ReLU: 정규화된 delta 중 양수만 사용 (평균 이상 변화)
        def relu(x: np.ndarray) -> np.ndarray:
            return np.maximum(x, 0.0)

        impact = (
            cfg.impact_smile_intensity_weight * relu(normed["smile_intensity"])
            + cfg.impact_head_yaw_delta_weight * relu(normed["head_yaw"])
            + cfg.impact_mouth_open_weight * relu(normed["mouth_open_ratio"])
            + cfg.impact_head_velocity_weight * relu(normed["head_velocity"])
            + cfg.impact_wrist_raise_weight * relu(normed["wrist_raise"])
            + cfg.impact_torso_rotation_weight * relu(normed["torso_rotation"])
            + cfg.impact_face_size_change_weight * relu(normed["face_area_ratio"])
            + cfg.impact_exposure_change_weight * relu(normed["brightness"])
        )
        # DINOv2 embed delta (if available in normed)
        if "embed_delta_face" in normed:
            impact = impact + cfg.impact_embed_face_weight * relu(normed["embed_delta_face"])
        return impact

    # ── Step 8: Smoothing ──

    def _smooth_ema(self, scores: np.ndarray, alpha: float) -> np.ndarray:
        """EMA smoothing for spike noise removal.

        highlight_rules.md §7
        """
        return self._compute_ema(scores, alpha)

    # ── Step 9: Peak detection ──

    def _detect_peaks(self, smoothed: np.ndarray) -> np.ndarray:
        """scipy.signal.find_peaks 기반 peak detection.

        highlight_rules.md §7
        """
        from scipy.signal import find_peaks

        cfg = self.config
        min_distance = int(cfg.peak_min_distance_sec * cfg.fps)
        if min_distance < 1:
            min_distance = 1

        # Prominence: per-video 상대적 (상위 percentile)
        positive_scores = smoothed[smoothed > 0]
        if len(positive_scores) == 0:
            return np.array([], dtype=int)

        prominence = np.percentile(positive_scores, cfg.peak_prominence_percentile)
        if prominence <= 0:
            prominence = np.max(smoothed) * 0.1

        peaks, _ = find_peaks(
            smoothed,
            distance=min_distance,
            prominence=prominence,
        )
        return peaks

    # ── Step 10: Window 생성 ──

    def _generate_windows(
        self,
        peaks: np.ndarray,
        records: List[FrameRecord],
        final_scores: np.ndarray,
        normed: dict[str, np.ndarray],
    ) -> List[HighlightWindow]:
        """Peak 기준 ±window_half_sec 구간을 생성하고 best frame을 선택.

        highlight_rules.md §7-§8
        """
        cfg = self.config
        half_frames = int(cfg.window_half_sec * cfg.fps)
        n = len(records)
        windows = []

        for wid, peak_idx in enumerate(peaks):
            start_idx = max(0, peak_idx - half_frames)
            end_idx = min(n - 1, peak_idx + half_frames)

            peak_rec = records[peak_idx]
            start_rec = records[start_idx]
            end_rec = records[end_idx]

            # Reason: 정규화된 delta 중 상위 기여 feature
            reason = self._build_reason(normed, peak_idx)

            # Best frame selection within window
            window_scores = final_scores[start_idx:end_idx + 1]
            window_indices = np.argsort(window_scores)[::-1][:cfg.best_frame_count]

            selected = []
            for local_idx in window_indices:
                abs_idx = start_idx + local_idx
                if final_scores[abs_idx] <= 0:
                    continue
                r = records[abs_idx]
                selected.append({
                    "frame_idx": r.frame_idx,
                    "timestamp_ms": r.timestamp_ms,
                    "frame_score": float(final_scores[abs_idx]),
                })

            windows.append(HighlightWindow(
                window_id=wid + 1,
                start_ms=start_rec.timestamp_ms,
                end_ms=end_rec.timestamp_ms,
                peak_ms=peak_rec.timestamp_ms,
                score=float(final_scores[peak_idx]),
                reason=reason,
                selected_frames=selected,
            ))

        return windows

    # ── Helpers ──

    def _build_reason(self, normed: dict[str, np.ndarray], idx: int) -> dict[str, float]:
        """Peak 지점에서 각 feature의 기여도를 반환."""
        reason = {}
        for key, arr in normed.items():
            val = float(arr[idx])
            if val > 0.5:  # 평균 이상 기여하는 feature만
                reason[key] = round(val, 2)
        return reason

    @staticmethod
    def _compute_ema(arr: np.ndarray, alpha: float) -> np.ndarray:
        """Exponential Moving Average."""
        ema = np.zeros_like(arr)
        ema[0] = arr[0]
        for i in range(1, len(arr)):
            ema[i] = alpha * arr[i] + (1 - alpha) * ema[i - 1]
        return ema

    @staticmethod
    def _minmax_normalize(arr: np.ndarray) -> np.ndarray:
        """0~1 min-max 정규화."""
        mn, mx = arr.min(), arr.max()
        if mx - mn < 1e-8:
            return np.zeros_like(arr)
        return (arr - mn) / (mx - mn)
