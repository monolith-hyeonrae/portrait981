"""Output types for frame scoring."""

from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class ScoringConfig:
    """Configuration for frame scoring weights and thresholds.

    Attributes:
        weight_technical: Weight for technical quality score (default: 0.45).
        weight_action: Weight for action/aesthetics score (default: 0.35).
        weight_identity: Weight for identity safety score (default: 0.20).
        enable_hard_filters: Whether to apply hard filters (default: True).
        min_blur_score: Minimum blur score to pass filter (default: 30).
        min_face_confidence: Minimum face confidence (default: 0.5).
        max_head_yaw: Maximum head yaw angle for frontal preference (default: 45).
        max_head_pitch: Maximum head pitch angle (default: 30).
    """

    # Component weights (should sum to 1.0)
    weight_technical: float = 0.45
    weight_action: float = 0.35
    weight_identity: float = 0.20

    # Hard filter settings
    enable_hard_filters: bool = True
    min_blur_score: float = 30.0
    min_face_confidence: float = 0.5
    max_head_yaw: float = 45.0
    max_head_pitch: float = 30.0

    # Technical quality thresholds
    optimal_brightness_min: float = 80.0
    optimal_brightness_max: float = 180.0
    min_contrast: float = 30.0

    # Action/aesthetics thresholds
    frontal_yaw_bonus: float = 25.0  # Yaw within this gets bonus
    expression_boost_threshold: float = 0.3  # Expression intensity for boost


@dataclass
class ScoreBreakdown:
    """Detailed breakdown of score components.

    Attributes:
        technical: Technical quality sub-scores.
        action: Action/aesthetics sub-scores.
        identity: Identity safety sub-scores.
    """

    technical: Dict[str, float] = field(default_factory=dict)
    action: Dict[str, float] = field(default_factory=dict)
    identity: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Dict[str, float]]:
        """Convert to dictionary."""
        return {
            "technical": self.technical,
            "action": self.action,
            "identity": self.identity,
        }


@dataclass
class ScoreResult:
    """Result of frame scoring.

    Attributes:
        total_score: Final combined score [0, 1].
        technical_score: Technical quality score [0, 1].
        action_score: Action/aesthetics score [0, 1].
        identity_score: Identity safety score [0, 1].
        is_filtered: Whether frame was filtered out.
        filter_reason: Reason for filtering (if filtered).
        breakdown: Detailed score breakdown.
    """

    total_score: float
    technical_score: float
    action_score: float
    identity_score: float
    is_filtered: bool = False
    filter_reason: Optional[str] = None
    breakdown: ScoreBreakdown = field(default_factory=ScoreBreakdown)

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "total_score": self.total_score,
            "technical_score": self.technical_score,
            "action_score": self.action_score,
            "identity_score": self.identity_score,
            "is_filtered": self.is_filtered,
            "filter_reason": self.filter_reason,
            "breakdown": self.breakdown.to_dict(),
        }
