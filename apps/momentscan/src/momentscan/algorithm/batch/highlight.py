"""Batch highlight detection engine.

highlight_rules.md 설계를 따르는 배치 분석 엔진:
1. Numeric feature delta 계산
2. Per-video 정규화 (MAD z-score)
3. QualityGate (hard filter)
4. Scoring: quality_blend × quality + impact_blend × impact (가산)
5. Temporal smoothing (gate-pass only EMA) + peak detection
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
        impact_scores = self._compute_impact_scores(arrays, normed)

        # 7. Final score = quality_blend × quality + impact_blend × impact (gate 통과만)
        wq = cfg.final_quality_blend
        wi = cfg.final_impact_blend
        final_scores = np.where(
            gate_mask,
            wq * quality_scores + wi * impact_scores,
            0.0,
        )
        logger.info("[5/7] Scoring complete (%.2f×quality + %.2f×impact)", wq, wi)

        # 8. Temporal smoothing (gate-pass only EMA)
        smoothed = self._smooth_ema_gated(final_scores, gate_mask, alpha=cfg.smoothing_alpha)
        logger.info("[6/7] Temporal smoothing (gate-pass only EMA α=%.2f)", cfg.smoothing_alpha)

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
        fields.update((
            "blur_score", "eye_open_ratio", "face_confidence", "face_identity",
            "head_blur",
            # CLIP 4축 (portrait_best = max of 4)
            "clip_disney_smile", "clip_charisma", "clip_wild_roar", "clip_playful_cute",
            # composites (info only)
            "duchenne_smile", "wild_intensity", "chill_score",
        ))

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
        """FrameRecord.gate_passed를 읽어 gate mask 생성.

        Gate 판정은 face.gate analyzer가 DAG에서 수행 (main face 기준).
        Gate 미실행(gate_passed=True default) 시 모든 프레임 통과.
        """
        return np.array([r.gate_passed for r in records], dtype=bool)

    # ── Step 5: Quality score ──

    def _compute_quality_scores(
        self,
        arrays: dict[str, np.ndarray],
        normed: dict[str, np.ndarray],
    ) -> np.ndarray:
        """연속 품질 점수.

        face.quality 필드 우선, 없으면 frame-level fallback.
        ArcFace face_identity 없으면 frontalness fallback.

        highlight_rules.md §6 Step 2
        """
        cfg = self.config

        # Sharpness: head_blur(face.quality) 우선, fallback → frame blur_score
        head_blur = arrays["head_blur"]
        has_head_blur = head_blur > 0
        blur_source = np.where(has_head_blur, head_blur, arrays["blur_score"])
        blur_normed = self._minmax_normalize(blur_source)

        # Face size
        face_size_normed = self._minmax_normalize(arrays["face_area_ratio"])

        # ArcFace face_identity (already 0~1 range)
        face_id = arrays["face_identity"]
        frontalness = np.clip(1.0 - np.abs(arrays["head_yaw"]) / cfg.frontalness_max_yaw, 0.0, 1.0)
        has_identity = face_id > 0
        face_quality = np.where(has_identity, face_id, frontalness)
        recog_weight = np.where(has_identity, cfg.quality_face_identity_weight, cfg.quality_frontalness_weight)

        # Build weighted sum dynamically
        w_blur = cfg.quality_head_blur_weight
        w_size = cfg.quality_face_size_weight
        total = w_blur + w_size + recog_weight

        quality = (
            w_blur * blur_normed
            + w_size * face_size_normed
            + recog_weight * face_quality
        ) / (total + 1e-8)

        return np.clip(quality, 0.0, 1.0)

    # ── Step 6: Impact score ──

    def _compute_impact_scores(
        self,
        arrays: dict[str, np.ndarray],
        normed: dict[str, np.ndarray],
    ) -> np.ndarray:
        """감정/동작 변화 점수 (Top-K weighted mean).

        highlight_rules.md §6 Step 3

        각 시그널의 가중 기여도를 계산한 뒤, 프레임별로 상위 K개만
        선택하여 평균. 노이즈 누적을 방지하고 두드러진 변화에 집중.

        smile_intensity는 예외:
        - head_yaw/velocity 등 모션 시그널은 '변화량' delta가 의미 있음
        - smile은 '절대값이 높은 프레임' = 가장 웃는 순간이 좋은 사진
        - delta 방식은 베이스라인(평상시 smile)이 높을 때 피크 delta가 작아져
          smile이 가득한 영상에서 impact가 0.3 이하로 낮아지는 문제 발생
        - per-video min-max 정규화 절대값을 사용해 '이 영상에서 가장 웃는 순간' 포착
        """
        cfg = self.config

        def relu(x: np.ndarray) -> np.ndarray:
            return np.maximum(x, 0.0)

        # smile: delta가 아닌 per-video min-max 절대값 사용
        smile_abs = self._minmax_normalize(arrays["smile_intensity"])

        # Portrait: CLIP 4축 중 프레임별 최상위 1개만 채택
        portrait_best = np.maximum.reduce([
            self._minmax_normalize(arrays["clip_disney_smile"]),
            self._minmax_normalize(arrays["clip_charisma"]),
            self._minmax_normalize(arrays["clip_wild_roar"]),
            self._minmax_normalize(arrays["clip_playful_cute"]),
        ])

        # (signal_name, weight, values)
        channels: list[tuple[str, float, np.ndarray]] = [
            ("smile_intensity", cfg.impact_smile_intensity_weight, smile_abs),
            ("head_yaw", cfg.impact_head_yaw_delta_weight, relu(normed["head_yaw"])),
            ("portrait_best", cfg.impact_portrait_weight, portrait_best),
        ]

        n = len(normed["smile_intensity"])
        k = min(cfg.impact_top_k, len(channels))

        # Stack weighted contributions: shape (n_channels, n_frames)
        weights_arr = np.array([w for _, w, _ in channels])   # (C,)
        weighted = np.array([w * v for _, w, v in channels])  # (C, N)

        # 고정 denominator: K개 최대 가중치의 합 (e.g., k=3 → 0.30+0.20+0.15=0.65)
        # 기여가 없는 채널의 weight이 분모에 섞이는 것을 방지.
        # 의미: "top-K 채널 모두 v=1.0일 때 최대 impact=1.0"
        max_achievable = float(np.sort(weights_arr)[::-1][:k].sum())

        # Per-frame top-K: select K channels by weighted contribution
        top_k_indices = np.argsort(-weighted, axis=0)[:k]  # (K, N)
        top_k_values = np.take_along_axis(weighted, top_k_indices, axis=0)  # (K, N)

        # sum(w_i × v_i for top-K) / max_achievable → [0, 1] range
        impact = top_k_values.sum(axis=0) / (max_achievable + 1e-8)

        return impact

    # ── Step 8: Smoothing ──

    def _smooth_ema_gated(
        self, scores: np.ndarray, gate_mask: np.ndarray, alpha: float,
    ) -> np.ndarray:
        """Gate-pass only EMA smoothing.

        gate_mask=0 프레임은 EMA에 0을 주입하지 않고 이전 smoothed 값을 유지.
        이렇게 하면 gate-fail 구간이 피크를 과도하게 감쇄하는 문제가 해소된다.
        """
        n = len(scores)
        smoothed = np.zeros(n)
        smoothed[0] = scores[0]
        for i in range(1, n):
            if gate_mask[i]:
                smoothed[i] = alpha * scores[i] + (1 - alpha) * smoothed[i - 1]
            else:
                smoothed[i] = smoothed[i - 1]
        return smoothed

    # ── Step 9: Peak detection ──

    def _detect_peaks(self, smoothed: np.ndarray) -> np.ndarray:
        """scipy.signal.find_peaks 기반 peak detection.

        highlight_rules.md §7

        prominence 계산 방식:
        - 절대값 기준 (기존): np.percentile(positive, pct) — 점수 floor가 높을 때
          threshold도 높아져 피크를 놓침 (모두 웃는 영상에서 문제)
        - 상대값 기준 (추가): score_range × relative_factor — floor에 무관하게
          영상 내 변화폭의 일정 비율을 threshold로 사용
        두 값의 min을 사용: 어느 쪽이든 더 관대한 기준으로 피크 허용.
        """
        from scipy.signal import find_peaks

        cfg = self.config
        min_distance = int(cfg.peak_min_distance_sec * cfg.fps)
        if min_distance < 1:
            min_distance = 1

        positive_scores = smoothed[smoothed > 0]
        if len(positive_scores) == 0:
            return np.array([], dtype=int)

        # 절대값 기준: 기존 percentile 방식
        abs_prominence = np.percentile(positive_scores, cfg.peak_prominence_percentile)
        if abs_prominence <= 0:
            abs_prominence = np.max(smoothed) * 0.1

        # 상대값 기준: 영상 내 score 변화폭의 30%
        # score floor가 높을 때 절대값 threshold가 너무 커지는 문제 보완
        score_range = float(np.max(smoothed)) - float(np.percentile(positive_scores, 10))
        range_prominence = score_range * 0.30

        # 두 기준 중 관대한(작은) 값 선택
        prominence = min(abs_prominence, range_prominence) if range_prominence > 1e-6 else abs_prominence

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
