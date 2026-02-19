"""Bucket classification for identity frames.

yaw 7 × pitch 3 × expression 4 = 84 buckets.
"""

from __future__ import annotations

from momentscan.algorithm.identity.types import BucketLabel

# Yaw bins (7): ±90° — 완전 측면 포함
YAW_EDGES = [-90, -60, -30, -10, 10, 30, 60, 90]
YAW_LABELS = [
    "[-90,-60]", "[-60,-30]", "[-30,-10]", "[-10,10]",
    "[10,30]", "[30,60]", "[60,90]",
]

# Pitch bins (3): down / neutral / up
PITCH_EDGES = [-30, -10, 10, 30]
PITCH_LABELS = ["down", "neutral", "up"]


def classify_yaw(yaw: float) -> str:
    """Yaw angle → bin label."""
    yaw = max(-90.0, min(90.0, yaw))
    for i in range(len(YAW_LABELS)):
        if yaw <= YAW_EDGES[i + 1]:
            return YAW_LABELS[i]
    return YAW_LABELS[-1]


def classify_pitch(pitch: float) -> str:
    """Pitch angle → bin label."""
    pitch = max(-30.0, min(30.0, pitch))
    for i in range(len(PITCH_LABELS)):
        if pitch <= PITCH_EDGES[i + 1]:
            return PITCH_LABELS[i]
    return PITCH_LABELS[-1]


def classify_expression(
    smile_intensity: float,
    mouth_open_ratio: float,
    eye_open_ratio: float,
) -> str:
    """Expression signals → bin label.

    Priority: eyes_closed > smile > mouth_open > neutral
    """
    if eye_open_ratio < 0.15:
        return "eyes_closed"
    if smile_intensity > 0.4:
        return "smile"
    if mouth_open_ratio > 0.4:
        return "mouth_open"
    return "neutral"


def classify_frame(
    yaw: float,
    pitch: float,
    smile_intensity: float,
    mouth_open_ratio: float,
    eye_open_ratio: float,
) -> BucketLabel:
    """프레임의 포즈/표정 → 복합 버킷."""
    return BucketLabel(
        yaw_bin=classify_yaw(yaw),
        pitch_bin=classify_pitch(pitch),
        expression_bin=classify_expression(smile_intensity, mouth_open_ratio, eye_open_ratio),
    )
