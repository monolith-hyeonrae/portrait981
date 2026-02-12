"""Batch analysis data types.

highlight_rules.md 설계를 따르는 데이터 구조 정의.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional


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
