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
    face_blur_min: float = 5.0        # face crop Laplacian variance (224×280 crop 기준)
    frame_blur_min: float = 50.0      # frame-level fallback
    exposure_min: float = 40.0
    exposure_max: float = 220.0

    # Passenger suitability thresholds (soft scoring, no hard gate)
    passenger_confidence_min: float = 0.5

    # Local contrast exposure thresholds (from face.quality mask-based metrics)
    contrast_min: float = 0.05        # CV < 0.05 = flat/washed out
    clipped_max: float = 0.3          # >30% overexposed pixels
    crushed_max: float = 0.3          # >30% underexposed pixels

    # Parsing coverage threshold (from face.quality BiSeNet mask, face bbox 기준)
    parsing_coverage_min: float = 0.50  # BiSeNet coverage < 50% = 노출 불량 proxy

    # Seg-based gate thresholds (require parsing_coverage >= parsing_coverage_min)
    seg_mouth_min: float = 0.01         # mouth seg < 1% when parsing OK → mask occlusion
    seg_face_min: float = 0.10          # face seg < 10% when parsing OK → exposure collapse


@dataclass
class FaceGateResult:
    """Per-face gate judgment."""

    face_id: int = 0
    role: str = ""                    # from face.classify
    gate_passed: bool = False
    fail_reasons: tuple[str, ...] = ()
    suitability: float = 0.0         # passenger soft score (0.0–1.0)
    confidence: float = 0.0          # face detection confidence (debug)
    face_bbox: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)  # normalized (x,y,w,h)
    # Measured values (for debug)
    face_area_ratio: float = 0.0
    face_blur: float = 0.0
    exposure: float = 0.0
    head_yaw: float = 0.0
    head_pitch: float = 0.0
    face_contrast: float = 0.0
    clipped_ratio: float = 0.0
    crushed_ratio: float = 0.0
    parsing_coverage: float = 0.0
    seg_mouth: float = 0.0
    seg_face: float = 0.0


@dataclass
class FaceGateOutput:
    """Per-frame gate output with per-face results."""

    results: List[FaceGateResult] = field(default_factory=list)
    main_gate_passed: bool = True     # frame-level: main face gate
    main_fail_reasons: tuple[str, ...] = ()


__all__ = ["FaceGateConfig", "FaceGateResult", "FaceGateOutput"]
