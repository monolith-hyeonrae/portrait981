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

        # 7. Final score = quality_blend × quality + impact_blend × impact (전 프레임)
        wq = cfg.final_quality_blend
        wi = cfg.final_impact_blend
        final_scores = wq * quality_scores + wi * impact_scores
        logger.info("[5/7] Scoring complete (%.2f×quality + %.2f×impact)", wq, wi)

        # 7.5. Passenger bonus (additive)
        wp = cfg.passenger_bonus_weight
        if wp > 0:
            passenger_bonus = self._compute_passenger_bonus(records)
            boosted_count = int((passenger_bonus > 0).sum())
            if boosted_count > 0:
                final_scores = final_scores + wp * passenger_bonus
                logger.info("[5.5/7] Passenger bonus applied (weight=%.2f, %.0f%% frames boosted)",
                            wp, 100.0 * boosted_count / n)

        # 8. Temporal smoothing (standard EMA)
        smoothed = self._smooth_ema(final_scores, alpha=cfg.smoothing_alpha)
        logger.info("[6/7] Temporal smoothing (EMA α=%.2f)", cfg.smoothing_alpha)

        # 9. Peak detection (gate-passed frames only)
        peaks = self._detect_peaks(smoothed, gate_mask)
        logger.info("[7/7] Peak detection: %d peaks found", len(peaks))

        # 10. Window 생성 + best frame 선택 (gate 필터 적용)
        windows = self._generate_windows(peaks, records, final_scores, normed, gate_mask)

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
        CLIP axes와 composites는 dict에서 동적으로 추출.
        """
        # delta 대상 필드 (composite_* and clip_axis_* are built dynamically below)
        fields: set[str] = set()
        for spec in PIPELINE_DELTA_SPECS:
            if not spec.record_field.startswith(("composite_", "clip_axis_")):
                fields.add(spec.record_field)
        # derived 필드의 소스
        for derived in PIPELINE_DERIVED_FIELDS:
            fields.update(derived.source_fields)
        # scoring에 필요하지만 delta/derived에 없는 추가 필드
        fields.update((
            "blur_score", "eye_open_ratio", "face_confidence", "face_identity",
            "face_blur",
            # catalog (portrait_best fallback)
            "catalog_best",
        ))

        arrays = {
            f: np.array([getattr(r, f) for r in records])
            for f in sorted(fields)
        }

        # Dynamic CLIP axes → "clip_axis_{name}" keys
        all_clip_names: set[str] = set()
        for r in records:
            all_clip_names.update(r.clip_axes.keys())
        for name in sorted(all_clip_names):
            arrays[f"clip_axis_{name}"] = np.array(
                [r.clip_axes.get(name, 0.0) for r in records]
            )

        # Dynamic composites → "composite_{name}" keys
        all_composite_names: set[str] = set()
        for r in records:
            all_composite_names.update(r.composites.keys())
        for name in sorted(all_composite_names):
            arrays[f"composite_{name}"] = np.array(
                [r.composites.get(name, 0.0) for r in records]
            )

        # catalog_scores → per-category arrays "catalog_cat_{name}"
        all_cat_names: set[str] = set()
        for r in records:
            all_cat_names.update(r.catalog_scores.keys())
        for name in sorted(all_cat_names):
            arrays[f"catalog_cat_{name}"] = np.array(
                [r.catalog_scores.get(name, 0.0) for r in records]
            )

        # bind_scores → per-category arrays "bind_cat_{name}"
        all_bind_names: set[str] = set()
        for r in records:
            all_bind_names.update(r.bind_scores.keys())
        for name in sorted(all_bind_names):
            arrays[f"bind_cat_{name}"] = np.array(
                [r.bind_scores.get(name, 0.0) for r in records]
            )

        # bind_best (scalar)
        if all_bind_names:
            arrays["bind_best"] = np.array(
                [r.bind_best for r in records]
            )

        return arrays

    # ── Step 2: Temporal delta ──

    def _compute_deltas(self, arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """EMA 기준선 대비 변화량(delta)을 계산.

        delta(t) = |feature(t) - EMA(feature, alpha)|
        highlight_rules.md §4.D
        """
        cfg = self.config
        deltas = {}
        for spec in PIPELINE_DELTA_SPECS:
            arr = arrays.get(spec.record_field)
            if arr is None:
                continue
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

        # Sharpness: face_blur(face.quality) 우선, fallback → frame blur_score
        face_blur = arrays["face_blur"]
        has_face_blur = face_blur > 0
        blur_source = np.where(has_face_blur, face_blur, arrays["blur_score"])
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
        w_blur = cfg.quality_face_blur_weight
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
        """감정/동작 변화 점수.

        우선순위:
        1. bind 모드 (TreeStrategy): bind_cat_* 확률 채널 사용
        2. 카탈로그 모드: catalog_cat_* 유사도 채널 사용
        3. 폴백: 기존 3채널 (smile, yaw_delta, portrait_best)

        bind + catalog 동시 사용 시: bind 50% + catalog 50% 블렌드.
        """
        bind_cat_keys = [k for k in sorted(arrays) if k.startswith("bind_cat_")]
        catalog_cat_keys = [k for k in sorted(arrays) if k.startswith("catalog_cat_")]

        if bind_cat_keys and catalog_cat_keys:
            # Both available: blend 50/50
            bind_impact = self._compute_catalog_impact(arrays, bind_cat_keys)
            catalog_impact = self._compute_catalog_impact(arrays, catalog_cat_keys)
            return 0.5 * bind_impact + 0.5 * catalog_impact
        if bind_cat_keys:
            return self._compute_catalog_impact(arrays, bind_cat_keys)
        if catalog_cat_keys:
            return self._compute_catalog_impact(arrays, catalog_cat_keys)
        return self._compute_legacy_impact(arrays, normed)

    def _compute_catalog_impact(
        self,
        arrays: dict[str, np.ndarray],
        catalog_cat_keys: list[str],
    ) -> np.ndarray:
        """카탈로그 모드 Impact: 카테고리별 유사도를 채널로 사용.

        각 카테고리의 시그널 프로파일이 smile/pose/AU/CLIP을 모두 인코딩하므로
        카테고리별 유사도 = 일반화된 Impact 시그널.
        균등 weight, per-video min-max 정규화, top-K weighted sum.
        """
        cfg = self.config
        n_cats = len(catalog_cat_keys)
        w = 1.0 / n_cats  # 균등 weight
        k = min(n_cats, cfg.impact_top_k)

        # Per-category: min-max normalize each category's similarity timeline
        normed_cats = [self._minmax_normalize(arrays[key]) for key in catalog_cat_keys]

        # Stack: (C, N)
        weighted = np.array([w * v for v in normed_cats])

        # max_achievable: top-K 채널 모두 v=1.0일 때
        max_achievable = k * w

        # Per-frame top-K selection
        top_k_indices = np.argsort(-weighted, axis=0)[:k]  # (K, N)
        top_k_values = np.take_along_axis(weighted, top_k_indices, axis=0)  # (K, N)

        impact = top_k_values.sum(axis=0) / (max_achievable + 1e-8)
        return impact

    def _compute_legacy_impact(
        self,
        arrays: dict[str, np.ndarray],
        normed: dict[str, np.ndarray],
    ) -> np.ndarray:
        """폴백: 기존 3채널 Impact (smile, yaw_delta, portrait_best).

        smile_intensity는 예외:
        - head_yaw/velocity 등 모션 시그널은 '변화량' delta가 의미 있음
        - smile은 '절대값이 높은 프레임' = 가장 웃는 순간이 좋은 사진
        - per-video min-max 정규화 절대값을 사용해 '이 영상에서 가장 웃는 순간' 포착
        """
        cfg = self.config

        def relu(x: np.ndarray) -> np.ndarray:
            return np.maximum(x, 0.0)

        # smile: delta가 아닌 per-video min-max 절대값 사용
        smile_abs = self._minmax_normalize(arrays["smile_intensity"])

        # Portrait: catalog_best 우선, fallback → CLIP 동적 축 max
        catalog_best = arrays.get("catalog_best", np.zeros(len(smile_abs)))
        has_catalog = catalog_best.max() > 0
        if has_catalog:
            portrait_best = self._minmax_normalize(catalog_best)
        else:
            clip_axis_keys = [k for k in sorted(arrays) if k.startswith("clip_axis_")]
            if clip_axis_keys:
                portrait_best = np.maximum.reduce([
                    self._minmax_normalize(arrays[k]) for k in clip_axis_keys
                ])
            else:
                portrait_best = np.zeros(len(smile_abs))

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

        # 고정 denominator: K개 최대 가중치의 합
        max_achievable = float(np.sort(weights_arr)[::-1][:k].sum())

        # Per-frame top-K: select K channels by weighted contribution
        top_k_indices = np.argsort(-weighted, axis=0)[:k]  # (K, N)
        top_k_values = np.take_along_axis(weighted, top_k_indices, axis=0)  # (K, N)

        impact = top_k_values.sum(axis=0) / (max_achievable + 1e-8)
        return impact

    # ── Step 8: Smoothing ──

    def _smooth_ema(
        self, scores: np.ndarray, alpha: float,
    ) -> np.ndarray:
        """Standard EMA smoothing.

        Gate와 무관하게 전 프레임에 동일한 EMA를 적용.
        Gate 필터링은 peak detection과 best frame 선택에서 수행.
        """
        n = len(scores)
        smoothed = np.zeros(n)
        smoothed[0] = scores[0]
        for i in range(1, n):
            smoothed[i] = alpha * scores[i] + (1 - alpha) * smoothed[i - 1]
        return smoothed

    # ── Step 9: Peak detection ──

    def _detect_peaks(self, smoothed: np.ndarray, gate_mask: np.ndarray | None = None) -> np.ndarray:
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

        # Gate 필터: gate-passed 프레임만 peak으로 인정
        if gate_mask is not None and len(peaks) > 0:
            peaks = peaks[gate_mask[peaks]]

        return peaks

    # ── Step 10: Window 생성 ──

    def _generate_windows(
        self,
        peaks: np.ndarray,
        records: List[FrameRecord],
        final_scores: np.ndarray,
        normed: dict[str, np.ndarray],
        gate_mask: np.ndarray | None = None,
    ) -> List[HighlightWindow]:
        """Peak 기준 ±window_half_sec 구간을 생성하고 best frame을 선택.

        gate_mask가 있으면 gate-passed 프레임만 best frame 후보로 사용.

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

            # Best frame selection within window (gate-passed only)
            window_scores = final_scores[start_idx:end_idx + 1].copy()
            if gate_mask is not None:
                window_gate = gate_mask[start_idx:end_idx + 1]
                window_scores[~window_gate] = -1.0  # gate-failed → 후보 제외
            window_indices = np.argsort(window_scores)[::-1][:cfg.best_frame_count]

            selected = []
            for local_idx in window_indices:
                abs_idx = start_idx + local_idx
                if final_scores[abs_idx] <= 0:
                    continue
                if gate_mask is not None and not gate_mask[abs_idx]:
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

    # ── Passenger bonus ──

    @staticmethod
    def _compute_passenger_bonus(records: List[FrameRecord]) -> np.ndarray:
        """동승자 suitability를 보너스 배열로 변환."""
        return np.array([r.passenger_suitability for r in records], dtype=np.float64)

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
