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
    mouth_open_ratio: float = 0.0
    eye_open_ratio: float = 0.0
    smile_intensity: float = 0.0

    # Pose features (from vpx-body-pose)
    wrist_raise: float = 0.0
    elbow_angle_change: float = 0.0
    torso_rotation: float = 0.0
    hand_near_face: float = 0.0

    # Quality features (from frame.quality)
    blur_score: float = 0.0
    brightness: float = 0.0
    contrast: float = 0.0

    # Face classifier (from face.classify)
    main_face_confidence: float = 0.0

    # Frame scoring (from frame.scoring)
    frame_score: float = 0.0


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

    highlight_rules.md §6 Scoring: Gate × Impact 곱 구조.
    """

    # Quality gate thresholds (§6 Step 1)
    gate_face_confidence: float = 0.7
    gate_face_area_ratio: float = 0.01
    gate_blur_min: float = 50.0  # Laplacian variance, TBD from data
    gate_exposure_min: float = 40.0
    gate_exposure_max: float = 220.0

    # Quality score weights (§6 Step 2)
    quality_blur_weight: float = 0.4
    quality_face_size_weight: float = 0.3
    quality_frontalness_weight: float = 0.3

    # Impact score weights (§6 Step 3)
    impact_mouth_open_weight: float = 0.30
    impact_head_velocity_weight: float = 0.20
    impact_wrist_raise_weight: float = 0.15
    impact_torso_rotation_weight: float = 0.15
    impact_face_size_change_weight: float = 0.10
    impact_exposure_change_weight: float = 0.10

    # Delta computation
    delta_alpha: float = 0.1           # EMA baseline alpha (for temporal deltas)
    frontalness_max_yaw: float = 45.0  # frontalness 정규화 기준각

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
                "mouth_open_ratio", "eye_open_ratio", "smile_intensity",
                "wrist_raise", "torso_rotation",
                "blur_score", "brightness",
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
                    f"{r.mouth_open_ratio:.3f}",
                    f"{r.eye_open_ratio:.3f}",
                    f"{r.smile_intensity:.3f}",
                    f"{r.wrist_raise:.3f}",
                    f"{r.torso_rotation:.3f}",
                    f"{r.blur_score:.1f}",
                    f"{r.brightness:.1f}",
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
