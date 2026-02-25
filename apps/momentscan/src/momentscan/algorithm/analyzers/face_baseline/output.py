"""Output types for face baseline analyzer."""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass(frozen=True)
class FaceBaselineProfile:
    """Per-face identity baseline (Welford online stats)."""

    face_id: int
    role: str                     # "main" | "passenger"
    n: int                        # observation count
    area_ratio_mean: float
    area_ratio_std: float
    center_x_mean: float
    center_x_std: float
    center_y_mean: float
    center_y_std: float


@dataclass
class FaceBaselineOutput:
    """Per-frame baseline output."""

    profiles: List[FaceBaselineProfile] = field(default_factory=list)
    main_profile: Optional[FaceBaselineProfile] = None
    passenger_profile: Optional[FaceBaselineProfile] = None


__all__ = ["FaceBaselineProfile", "FaceBaselineOutput"]
