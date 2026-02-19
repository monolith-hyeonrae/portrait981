"""Portrait pivot system for identity frame classification.

Replaces the fixed 84-bin grid (7 yaw × 3 pitch × 4 expression) with
portrait-driven pivots: 5 pose pivots × 4 expression pivots = 20 bins.

Pose pivots use symmetric yaw (|yaw|) to treat left/right equally.
Expression pivots use AU intensities (AU12=smile, AU25/AU26=mouth open)
for reliable excited/surprised discrimination.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from momentscan.algorithm.identity.types import BucketLabel


@dataclass(frozen=True)
class PosePivot:
    """Named pose target defined by (yaw, pitch) center.

    yaw is absolute (symmetric left/right).
    """

    name: str
    yaw: float
    pitch: float


@dataclass(frozen=True)
class ExpressionPivot:
    """Named expression category."""

    name: str


@dataclass
class PivotAssignment:
    """Result of assign_pivot(): matched pose + expression + distance."""

    pose: PosePivot
    expression: ExpressionPivot
    pose_distance: float

    @property
    def pivot_name(self) -> str:
        return f"{self.pose.name}|{self.expression.name}"


# ── Pivot Definitions ──

POSE_PIVOTS: List[PosePivot] = [
    PosePivot("frontal", yaw=0, pitch=0),
    PosePivot("three-quarter", yaw=30, pitch=0),
    PosePivot("side-profile", yaw=60, pitch=0),
    PosePivot("looking-up", yaw=15, pitch=20),
    PosePivot("three-quarter-up", yaw=30, pitch=15),
]

EXPRESSION_PIVOTS: List[ExpressionPivot] = [
    ExpressionPivot("neutral"),
    ExpressionPivot("smile"),
    ExpressionPivot("excited"),
    ExpressionPivot("surprised"),
]

# Convenience references
NEUTRAL = EXPRESSION_PIVOTS[0]
SMILE = EXPRESSION_PIVOTS[1]
EXCITED = EXPRESSION_PIVOTS[2]
SURPRISED = EXPRESSION_PIVOTS[3]

DEFAULT_R_ACCEPT = 15.0

# AU thresholds (DISFA 0-5 scale)
AU12_SMILE_THRESHOLD = 1.0
AU_MOUTH_OPEN_THRESHOLD = 1.5

# HSEmotion fallback thresholds
FALLBACK_SMILE_THRESHOLD = 0.4
FALLBACK_MOUTH_OPEN_THRESHOLD = 0.5
FALLBACK_EXCITED_SMILE_THRESHOLD = 0.3


# ── Expression Classification ──


def classify_expression_from_au(au_intensities: Dict[str, float]) -> ExpressionPivot:
    """Classify expression from AU intensities.

    AU12 >= 1.0 → smile component
    max(AU25, AU26) >= 1.5 → mouth open component
    Both → excited; smile only → smile; mouth only → surprised; neither → neutral.
    """
    au12 = au_intensities.get("AU12", 0.0)
    au25 = au_intensities.get("AU25", 0.0)
    au26 = au_intensities.get("AU26", 0.0)

    smile = au12 >= AU12_SMILE_THRESHOLD
    mouth_open = max(au25, au26) >= AU_MOUTH_OPEN_THRESHOLD

    if smile and mouth_open:
        return EXCITED
    if not smile and mouth_open:
        return SURPRISED
    if smile:
        return SMILE
    return NEUTRAL


def classify_expression_fallback(
    smile_intensity: float,
    mouth_open_ratio: float,
) -> ExpressionPivot:
    """Classify expression from HSEmotion signals (fallback when no AU data).

    Less reliable for excited/surprised discrimination.
    """
    if mouth_open_ratio >= FALLBACK_MOUTH_OPEN_THRESHOLD and smile_intensity >= FALLBACK_EXCITED_SMILE_THRESHOLD:
        return EXCITED
    if mouth_open_ratio >= FALLBACK_MOUTH_OPEN_THRESHOLD:
        return SURPRISED
    if smile_intensity >= FALLBACK_SMILE_THRESHOLD:
        return SMILE
    return NEUTRAL


# ── Pose Distance ──


def _pose_distance(yaw: float, pitch: float, pivot: PosePivot) -> float:
    """Euclidean distance in pose space. Uses |yaw| for left-right symmetry."""
    d_yaw = abs(yaw) - pivot.yaw
    d_pitch = pitch - pivot.pitch
    return math.sqrt(d_yaw ** 2 + d_pitch ** 2)


def _find_nearest_pose(
    yaw: float, pitch: float, *, r_accept: float = DEFAULT_R_ACCEPT
) -> Optional[Tuple[PosePivot, float]]:
    """Find the nearest pose pivot within r_accept radius.

    Returns:
        (PosePivot, distance) or None if all pivots exceed r_accept.
    """
    best_pivot = None
    best_dist = float("inf")

    for pivot in POSE_PIVOTS:
        d = _pose_distance(yaw, pitch, pivot)
        if d < best_dist:
            best_dist = d
            best_pivot = pivot

    if best_dist > r_accept:
        return None
    return best_pivot, best_dist


# ── Main API ──


def assign_pivot(
    yaw: float,
    pitch: float,
    au_intensities: Optional[Dict[str, float]] = None,
    *,
    r_accept: float = DEFAULT_R_ACCEPT,
) -> Optional[PivotAssignment]:
    """Assign a portrait pivot to a frame based on pose + expression.

    Args:
        yaw: Head yaw in degrees.
        pitch: Head pitch in degrees.
        au_intensities: Dict of AU name -> intensity (from face.au).
            If None, expression defaults to neutral.
        r_accept: Maximum pose distance to accept assignment.

    Returns:
        PivotAssignment or None if no pose pivot is within r_accept.
    """
    result = _find_nearest_pose(yaw, pitch, r_accept=r_accept)
    if result is None:
        return None

    pose_pivot, pose_dist = result

    if au_intensities:
        expr_pivot = classify_expression_from_au(au_intensities)
    else:
        expr_pivot = NEUTRAL

    return PivotAssignment(
        pose=pose_pivot,
        expression=expr_pivot,
        pose_distance=pose_dist,
    )


def assign_pivot_fallback(
    yaw: float,
    pitch: float,
    smile_intensity: float,
    mouth_open_ratio: float,
    *,
    r_accept: float = DEFAULT_R_ACCEPT,
) -> Optional[PivotAssignment]:
    """Assign pivot using HSEmotion signals (when AU data unavailable).

    Same pose logic, but expression uses fallback classifier.
    """
    result = _find_nearest_pose(yaw, pitch, r_accept=r_accept)
    if result is None:
        return None

    pose_pivot, pose_dist = result
    expr_pivot = classify_expression_fallback(smile_intensity, mouth_open_ratio)

    return PivotAssignment(
        pose=pose_pivot,
        expression=expr_pivot,
        pose_distance=pose_dist,
    )


def pivot_to_bucket(assignment: PivotAssignment) -> BucketLabel:
    """Convert PivotAssignment to BucketLabel for backward compatibility.

    Maps pivot names to bucket bins:
    - yaw_bin: pose pivot name (e.g., "frontal", "three-quarter")
    - pitch_bin: pose pivot name (same, since pivots encode both)
    - expression_bin: expression pivot name (e.g., "smile", "excited")
    """
    return BucketLabel(
        yaw_bin=assignment.pose.name,
        pitch_bin=assignment.pose.name,
        expression_bin=assignment.expression.name,
    )
