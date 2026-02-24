"""Output types for face gate analyzer (per-face independent gate)."""

from dataclasses import dataclass, field
from typing import List


@dataclass(frozen=True)
class FaceGateConfig:
    """Gate condition thresholds.

    Main face uses standard thresholds. Passenger uses relaxed thresholds.
    Zero values (unmeasured) pass by default.
    """

    # Main face thresholds
    face_confidence_min: float = 0.7
    face_area_ratio_min: float = 0.02
    head_blur_min: float = 30.0       # face crop Laplacian variance
    frame_blur_min: float = 50.0      # frame-level fallback
    exposure_min: float = 40.0
    exposure_max: float = 220.0
    head_yaw_max: float = 70.0        # main only
    head_pitch_max: float = 50.0      # main only

    # Passenger relaxed thresholds
    passenger_area_ratio_min: float = 0.01
    passenger_blur_min: float = 20.0

    # Local contrast exposure thresholds (from face.quality mask-based metrics)
    contrast_min: float = 0.05        # CV < 0.05 = flat/washed out
    clipped_max: float = 0.3          # >30% overexposed pixels
    crushed_max: float = 0.3          # >30% underexposed pixels


@dataclass
class FaceGateResult:
    """Per-face gate judgment."""

    face_id: int = 0
    role: str = ""                    # from face.classify
    gate_passed: bool = False
    fail_reasons: tuple[str, ...] = ()
    face_bbox: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)  # normalized (x,y,w,h)
    # Measured values (for debug)
    face_area_ratio: float = 0.0
    head_blur: float = 0.0
    exposure: float = 0.0
    head_yaw: float = 0.0
    head_pitch: float = 0.0
    head_contrast: float = 0.0
    clipped_ratio: float = 0.0
    crushed_ratio: float = 0.0


@dataclass
class FaceGateOutput:
    """Per-frame gate output with per-face results."""

    results: List[FaceGateResult] = field(default_factory=list)
    main_gate_passed: bool = True     # frame-level: main face gate
    main_fail_reasons: tuple[str, ...] = ()


__all__ = ["FaceGateConfig", "FaceGateResult", "FaceGateOutput"]
