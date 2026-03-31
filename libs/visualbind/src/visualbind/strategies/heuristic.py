"""Heuristic binding strategy: threshold-based quality gate + expression rules.

face.gate의 하드코딩 로직을 visualbind strategy로 흡수.
Day 1에 즉시 동작 가능 (학습 불필요).

Implements :class:`~visualbind.strategies.BindingStrategy`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class GateConfig:
    """Quality gate threshold configuration.

    모든 threshold는 raw signal 기준 (정규화 전).
    """

    # Exposure (face.quality mask-based)
    exposure_min: float = 50.0
    exposure_max: float = 200.0

    # Local contrast (CV = std/mean)
    contrast_min: float = 0.10

    # Pixel-level clipping
    clipped_max: float = 0.15
    crushed_max: float = 0.15

    # Blur (Laplacian variance)
    face_blur_min: float = 5.0

    # Detection confidence
    confidence_min: float = 0.7

    # Head pose limits (signal reliability boundary)
    yaw_max: float = 55.0       # |yaw| > 55° → AU/expression unreliable
    pitch_max: float = 35.0     # |pitch| > 35° → looking up/down too much (6DRepNet saturates)
    roll_max: float = 35.0      # |roll| > 35° → head tilted too much
    pose_combined_max: float = 55.0  # sqrt(yaw² + pitch² + roll²) — catches multi-axis degradation

    # Signal validity (catches pose estimator failures)
    seg_face_min: float = 0.01   # BiSeNet face segmentation ratio — 0 means face not visible
    au_sum_min: float = 0.05     # sum of all 12 AUs — near 0 means AU estimation failed


class HeuristicStrategy:
    """Threshold-based binding strategy (face.gate equivalent).

    Implements :class:`~visualbind.strategies.BindingStrategy`.

    학습 없이 하드코딩 threshold로 판단.
    visualpath → visualbind 진화의 시작점.
    데이터가 쌓이면 TreeStrategy로 대체.
    """

    def __init__(self, config: Optional[GateConfig] = None) -> None:
        self.config = config or GateConfig()
        self._signal_fields: list[str] = []

    def fit(self, vectors: dict[str, np.ndarray], **kwargs: object) -> None:
        """No-op: heuristic strategy doesn't learn from data."""
        pass

    def set_signal_fields(self, fields: tuple[str, ...] | list[str]) -> None:
        """Set signal field names for raw value lookup."""
        self._signal_fields = list(fields)

    def predict(self, frame_vec: np.ndarray) -> dict[str, float]:
        """Score frame based on quality thresholds.

        Returns:
            Dict with 'gate_passed' (1.0 or 0.0) and 'gate_reasons' keys.
        """
        result = {"gate_passed": 1.0}
        fails = self.check_gate(frame_vec)
        if fails:
            result["gate_passed"] = 0.0
        return result

    def check_gate(self, frame_vec: np.ndarray) -> list[str]:
        """Check quality gate conditions. Returns list of fail reasons."""
        cfg = self.config
        fails: list[str] = []

        raw = self._get_raw_signals(frame_vec)

        # Exposure absolute brightness
        face_exposure = raw.get("face_exposure", 0.0)
        if face_exposure > 0:
            if face_exposure < cfg.exposure_min:
                fails.append("gate.exposure.too_dark")
            elif face_exposure > cfg.exposure_max:
                fails.append("gate.exposure.too_bright")

        # Local contrast
        face_contrast = raw.get("face_contrast", 0.0)
        if face_contrast > 0 and face_contrast < cfg.contrast_min:
            fails.append("gate.exposure.low_contrast")

        # Pixel clipping
        clipped = raw.get("clipped_ratio", 0.0)
        crushed = raw.get("crushed_ratio", 0.0)
        if clipped > cfg.clipped_max:
            fails.append("gate.exposure.clipped")
        if crushed > cfg.crushed_max:
            fails.append("gate.exposure.crushed")

        # Blur
        face_blur = raw.get("face_blur", 0.0)
        if face_blur > 0 and face_blur < cfg.face_blur_min:
            fails.append("gate.blur")

        # Confidence
        confidence = raw.get("face_confidence", 0.0)
        if confidence > 0 and confidence < cfg.confidence_min:
            fails.append("gate.confidence")

        # Head pose validity
        yaw = raw.get("head_yaw_dev", 0.0)
        pitch = raw.get("head_pitch", 0.0)
        roll = raw.get("head_roll", 0.0)
        if abs(yaw) > cfg.yaw_max:
            fails.append("gate.pose.extreme_yaw")
        if abs(pitch) > cfg.pitch_max:
            fails.append("gate.pose.extreme_pitch")
        if abs(roll) > cfg.roll_max:
            fails.append("gate.pose.extreme_roll")

        return fails

    def check_gate_from_signals(self, signals: dict[str, float]) -> list[str]:
        """Check gate from raw signal dict (no normalization needed)."""
        cfg = self.config
        fails: list[str] = []

        face_exposure = signals.get("face_exposure", 0.0)
        if face_exposure > 0:
            if face_exposure < cfg.exposure_min:
                fails.append("gate.exposure.too_dark")
            elif face_exposure > cfg.exposure_max:
                fails.append("gate.exposure.too_bright")

        face_contrast = signals.get("face_contrast", 0.0)
        if face_contrast > 0 and face_contrast < cfg.contrast_min:
            fails.append("gate.exposure.low_contrast")

        if signals.get("clipped_ratio", 0.0) > cfg.clipped_max:
            fails.append("gate.exposure.clipped")
        if signals.get("crushed_ratio", 0.0) > cfg.crushed_max:
            fails.append("gate.exposure.crushed")

        face_blur = signals.get("face_blur", 0.0)
        if face_blur > 0 and face_blur < cfg.face_blur_min:
            fails.append("gate.blur")

        confidence = signals.get("face_confidence", 0.0)
        if confidence > 0 and confidence < cfg.confidence_min:
            fails.append("gate.confidence")

        # Signal validity — catches pose estimator failures on extreme profiles
        seg_face = signals.get("seg_face", -1.0)
        if seg_face >= 0 and seg_face < cfg.seg_face_min:
            fails.append("gate.signal.no_face_seg")

        au_keys = [k for k in signals if k.startswith("au")]
        if au_keys:
            au_sum = sum(signals[k] for k in au_keys)
            if au_sum < cfg.au_sum_min:
                fails.append("gate.signal.no_au_activity")

        # Head pose validity — extreme angles make AU/expression unreliable
        yaw = signals.get("head_yaw_dev", 0.0)
        pitch = signals.get("head_pitch", 0.0)
        roll = signals.get("head_roll", 0.0)
        if abs(yaw) > cfg.yaw_max:
            fails.append("gate.pose.extreme_yaw")
        if abs(pitch) > cfg.pitch_max:
            fails.append("gate.pose.extreme_pitch")
        if abs(roll) > cfg.roll_max:
            fails.append("gate.pose.extreme_roll")

        # Combined magnitude — catches multi-axis degradation
        # (individual axes below threshold but combination is unreliable)
        import math
        combined = math.sqrt(yaw**2 + pitch**2 + roll**2)
        if combined > cfg.pose_combined_max:
            if not any("pose" in f for f in fails):  # avoid duplicate if already caught
                fails.append("gate.pose.combined")

        return fails

    def _get_raw_signals(self, frame_vec: np.ndarray) -> dict[str, float]:
        """Denormalize signal vector back to raw values for threshold comparison."""
        from visualbind.signals import SIGNAL_RANGES

        raw = {}
        fields = self._signal_fields
        if not fields:
            from visualbind.signals import SIGNAL_FIELDS
            fields = list(SIGNAL_FIELDS)

        for i, f in enumerate(fields):
            if i >= len(frame_vec):
                break
            lo, hi = SIGNAL_RANGES.get(f, (0.0, 1.0))
            raw[f] = float(frame_vec[i]) * (hi - lo) + lo

        return raw
