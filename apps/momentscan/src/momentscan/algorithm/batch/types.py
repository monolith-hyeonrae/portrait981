"""Batch analysis data types.

highlight_rules.md 설계를 따르는 데이터 구조 정의.
"""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class FrameRecord:
    """프레임별 수치 feature 레코드.

    on_frame()에서 추출한 원본 수치를 저장.
    정규화는 BatchHighlightEngine에서 전체 비디오 기준으로 수행.
    """

    frame_idx: int
    timestamp_ms: float

    # Face features (from vpx-face-detect)
    face_detected: bool = False
    face_confidence: float = 0.0
    face_area_ratio: float = 0.0
    face_center_distance: float = 0.0
    head_yaw: float = 0.0
    head_pitch: float = 0.0
    head_roll: float = 0.0

    # Expression features (from vpx-face-expression)
    eye_open_ratio: float = 0.0
    smile_intensity: float = 0.0

    # Quality features (from frame.quality)
    blur_score: float = 0.0
    brightness: float = 0.0
    contrast: float = 0.0

    # Face classifier (from face.classify)
    main_face_confidence: float = 0.0

    # ArcFace identity (from face.detect)
    face_identity: float = 0.0   # ArcFace anchor similarity (quality)

    # Face quality (from face.quality — head crop region metrics)
    head_blur: float = 0.0           # head_crop Laplacian variance (face sharpness)
    head_exposure: float = 0.0       # head_crop mean brightness
    head_aesthetic: float = 0.0      # CLIP portrait quality score [0,1]
    head_contrast: float = 0.0      # CV = std/mean (skin-tone invariant)
    clipped_ratio: float = 0.0      # overexposed (>250) pixel ratio
    crushed_ratio: float = 0.0      # underexposed (<5) pixel ratio
    mask_method: str = ""            # "parsing" | "landmark" | "center_patch"
    parsing_coverage: float = 0.0    # face.parse segmentation coverage (0 = 미측정)
    # Semantic segmentation ratios (from class_map, info-only)
    seg_face: float = 0.0            # face region (skin+brow+eye+ear+nose+mouth+lip)
    seg_eye: float = 0.0             # eye (l_eye+r_eye) pixel ratio
    seg_mouth: float = 0.0           # mouth (mouth+u_lip+l_lip) pixel ratio
    seg_hair: float = 0.0            # hair pixel ratio
    eye_pixel_ratio: float = 0.0     # = seg_eye (eye_open 교차검증용)

    # CLIP axis scores (from portrait.score → metadata._clip_axes)
    clip_disney_smile: float = 0.0
    clip_charisma: float = 0.0
    clip_wild_roar: float = 0.0
    clip_playful_cute: float = 0.0

    # AU features (from face.au)
    au6_cheek_raiser: float = 0.0   # AU6: 눈가 주름 (Duchenne marker)
    au12_lip_corner: float = 0.0    # AU12: 입꼬리 올림 (smile)
    au25_lips_part: float = 0.0     # AU25: 입술 벌림
    au26_jaw_drop: float = 0.0      # AU26: 턱 벌림

    # Expression neutral (from face.expression)
    em_neutral: float = 0.0         # HSEmotion neutral probability

    # Cross-analyzer composites (derived in extract.py)
    duchenne_smile: float = 0.0     # disney_smile × (AU6 + AU12) — genuine warm smile
    wild_intensity: float = 0.0     # wild_roar × max(AU25, AU26) — 실제로 입 벌림 확인
    chill_score: float = 0.0        # neutral_high × all_axes_low — 무표정 탑승

    # Face baseline (from face.baseline)
    baseline_n: int = 0                  # observation count (n < 2 = not converged)
    baseline_area_mean: float = 0.0      # main face area_ratio mean
    baseline_area_std: float = 0.0       # main face area_ratio std

    # Gate result (from face.gate analyzer — main face)
    gate_passed: bool = True        # default True: gate 미실행 시 통과로 간주
    gate_fail_reasons: str = ""     # comma-separated fail condition names

    # Passenger gate (from face.gate analyzer)
    passenger_detected: bool = False
    passenger_gate_passed: bool = True
    passenger_gate_fail_reasons: str = ""
    passenger_face_area_ratio: float = 0.0
    passenger_head_blur: float = 0.0
    passenger_head_exposure: float = 0.0


@dataclass
class HighlightWindow:
    """하이라이트 구간 하나.

    peak detection으로 찾은 구간 + 그 안의 best frame.
    """

    window_id: int
    start_ms: float
    end_ms: float
    peak_ms: float
    score: float
    reason: Dict[str, float] = field(default_factory=dict)
    selected_frames: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class HighlightConfig:
    """배치 하이라이트 분석 설정.

    highlight_rules.md §6 Scoring: Gate → 가산 (quality + impact).
    Gate thresholds는 FaceGateConfig (face.gate analyzer)로 이동.
    """

    # Quality score weights (§6 Step 2)
    quality_head_blur_weight: float = 0.30       # head_crop sharpness (portrait-specific)
    quality_face_size_weight: float = 0.20
    quality_face_identity_weight: float = 0.30   # ArcFace anchor similarity
    quality_frontalness_weight: float = 0.25     # fallback (ArcFace 없을 때만 사용)

    # Final score = quality_blend × quality + impact_blend × impact (가산)
    final_quality_blend: float = 0.35
    final_impact_blend: float = 0.65

    # Impact score weights (§6 Step 3) — Top-K weighted mean
    impact_top_k: int = 3  # 상위 K개 시그널만 사용 (현재 3채널: smile, yaw, portrait)
    impact_smile_intensity_weight: float = 0.25       # 미소 강도 (절대값)
    impact_head_yaw_delta_weight: float = 0.15        # 머리 회전 변화량
    impact_portrait_weight: float = 0.25              # portrait.score 최상위 1개 (duchenne/wild/charisma/chill 중 max)

    # Delta computation
    delta_alpha: float = 0.1           # EMA baseline alpha (for temporal deltas)
    frontalness_max_yaw: float = 45.0  # frontalness 정규화 기준각

    # Normalization
    normed_cap_percentile: float = 98.0  # MAD z-score → [0,1] rescaling cap

    # Temporal smoothing (§7)
    smoothing_alpha: float = 0.25  # EMA alpha

    # Peak detection (§7)
    peak_min_distance_sec: float = 2.5
    peak_prominence_percentile: float = 90.0

    # Window generation (§7)
    window_half_sec: float = 1.0

    # Best frame selection (§8)
    best_frame_count: int = 3

    # FPS (set from pipeline)
    fps: float = 10.0


@dataclass
class HighlightResult:
    """배치 하이라이트 분석 결과."""

    windows: List[HighlightWindow] = field(default_factory=list)
    frame_count: int = 0
    config: Optional[HighlightConfig] = None
    _timeseries: Optional[Dict[str, Any]] = field(default=None, repr=False)

    def export(self, output_dir: Path) -> None:
        """결과를 파일로 출력한다.

        Args:
            output_dir: 출력 디렉토리. highlight/ 하위에 저장.
        """
        highlight_dir = output_dir / "highlight"
        highlight_dir.mkdir(parents=True, exist_ok=True)

        # windows.json
        windows_data = [asdict(w) for w in self.windows]
        with open(highlight_dir / "windows.json", "w") as f:
            json.dump(windows_data, f, indent=2, ensure_ascii=False)

        # timeseries.csv + score_curve.png
        if self._timeseries is not None:
            self._export_timeseries_csv(highlight_dir)
            self._export_score_curve(highlight_dir)

    def _export_timeseries_csv(self, highlight_dir: Path) -> None:
        """프레임별 점수 데이터를 CSV로 출력한다."""
        ts = self._timeseries
        records: List[FrameRecord] = ts["records"]
        gate_mask: np.ndarray = ts["gate_mask"]
        quality_scores: np.ndarray = ts["quality_scores"]
        impact_scores: np.ndarray = ts["impact_scores"]
        final_scores: np.ndarray = ts["final_scores"]
        smoothed: np.ndarray = ts["smoothed"]
        peaks: np.ndarray = ts["peaks"]

        peak_set = set(int(p) for p in peaks)

        csv_path = highlight_dir / "timeseries.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            header = [
                "frame_idx", "timestamp_ms",
                "gate_pass", "quality_score", "impact_score",
                "final_score", "smoothed_score", "is_peak",
                # raw features
                "face_detected", "face_confidence", "face_area_ratio",
                "head_yaw", "head_pitch",
                "eye_open_ratio", "smile_intensity",
                "blur_score", "brightness",
                # identity
                "face_identity",
                # shot quality
                "head_blur", "head_exposure", "head_aesthetic",
                "head_contrast", "clipped_ratio", "crushed_ratio",
                "mask_method", "parsing_coverage",
                # semantic segmentation ratios
                "seg_face", "seg_eye", "seg_mouth", "seg_hair",
                "eye_pixel_ratio",
                # CLIP axes
                "clip_disney_smile", "clip_charisma",
                "clip_wild_roar", "clip_playful_cute",
                # AU features
                "au6_cheek_raiser", "au12_lip_corner",
                "au25_lips_part", "au26_jaw_drop",
                # expression neutral
                "em_neutral",
                # composites
                "duchenne_smile", "wild_intensity", "chill_score",
                # baseline
                "baseline_n", "baseline_area_mean", "baseline_area_std",
                # gate result
                "gate_fail_reasons",
            ]
            writer.writerow(header)

            for i, r in enumerate(records):
                writer.writerow([
                    r.frame_idx,
                    f"{r.timestamp_ms:.1f}",
                    int(gate_mask[i]),
                    f"{quality_scores[i]:.4f}",
                    f"{impact_scores[i]:.4f}",
                    f"{final_scores[i]:.4f}",
                    f"{smoothed[i]:.4f}",
                    int(i in peak_set),
                    int(r.face_detected),
                    f"{r.face_confidence:.3f}",
                    f"{r.face_area_ratio:.4f}",
                    f"{r.head_yaw:.1f}",
                    f"{r.head_pitch:.1f}",
                    f"{r.eye_open_ratio:.3f}",
                    f"{r.smile_intensity:.3f}",
                    f"{r.blur_score:.1f}",
                    f"{r.brightness:.1f}",
                    f"{r.face_identity:.4f}",
                    f"{r.head_blur:.1f}",
                    f"{r.head_exposure:.1f}",
                    f"{r.head_aesthetic:.4f}",
                    f"{r.head_contrast:.4f}",
                    f"{r.clipped_ratio:.4f}",
                    f"{r.crushed_ratio:.4f}",
                    r.mask_method,
                    f"{r.parsing_coverage:.4f}",
                    # semantic segmentation ratios
                    f"{r.seg_face:.4f}",
                    f"{r.seg_eye:.4f}",
                    f"{r.seg_mouth:.4f}",
                    f"{r.seg_hair:.4f}",
                    f"{r.eye_pixel_ratio:.4f}",
                    # CLIP axes
                    f"{r.clip_disney_smile:.4f}",
                    f"{r.clip_charisma:.4f}",
                    f"{r.clip_wild_roar:.4f}",
                    f"{r.clip_playful_cute:.4f}",
                    # AU features
                    f"{r.au6_cheek_raiser:.3f}",
                    f"{r.au12_lip_corner:.3f}",
                    f"{r.au25_lips_part:.3f}",
                    f"{r.au26_jaw_drop:.3f}",
                    # expression neutral
                    f"{r.em_neutral:.3f}",
                    # composites
                    f"{r.duchenne_smile:.4f}",
                    f"{r.wild_intensity:.4f}",
                    f"{r.chill_score:.4f}",
                    # baseline
                    r.baseline_n,
                    f"{r.baseline_area_mean:.4f}",
                    f"{r.baseline_area_std:.4f}",
                    # gate result
                    r.gate_fail_reasons,
                ])

        logger.info("Exported timeseries CSV: %s (%d rows)", csv_path, len(records))

    def _export_score_curve(self, highlight_dir: Path) -> None:
        """시간축 점수 그래프 + peak 마커를 PNG로 출력한다.

        matplotlib이 없으면 skip.
        """
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            logger.info("matplotlib not installed — skipping score_curve.png")
            return

        ts = self._timeseries
        records: List[FrameRecord] = ts["records"]
        final_scores: np.ndarray = ts["final_scores"]
        smoothed: np.ndarray = ts["smoothed"]
        peaks: np.ndarray = ts["peaks"]

        time_sec = np.array([r.timestamp_ms / 1000.0 for r in records])

        fig, ax = plt.subplots(figsize=(14, 4))
        ax.plot(time_sec, final_scores, alpha=0.3, linewidth=0.5, label="final_score")
        ax.plot(time_sec, smoothed, linewidth=1.0, label="smoothed")

        if len(peaks) > 0:
            ax.scatter(
                time_sec[peaks], smoothed[peaks],
                color="red", zorder=5, s=40, label="peak",
            )

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Score")
        ax.set_title("Highlight Score Curve")
        ax.legend(loc="upper right", fontsize=8)
        ax.set_xlim(time_sec[0], time_sec[-1])
        fig.tight_layout()

        png_path = highlight_dir / "score_curve.png"
        fig.savefig(png_path, dpi=120)
        plt.close(fig)
        logger.info("Exported score curve: %s", png_path)
