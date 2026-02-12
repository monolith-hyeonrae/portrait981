"""Frame scorer for ranking frames by quality and aesthetics.

Combines multiple scoring components with configurable weights to produce
a final score for each frame. Supports hard filters for disqualification.

Scoring Components:
    - Technical (0.45): blur, exposure, sharpness, occlusion
    - Action (0.35): pose energy, expression, composition
    - Identity (0.20): face consistency, stability

Example:
    >>> scorer = FrameScorer()
    >>> result = scorer.score(face_obs=face_obs, quality_obs=quality_obs)
    >>> if not result.is_filtered:
    ...     print(f"Frame score: {result.total_score:.2f}")
"""

from typing import Dict, List, Optional, Tuple
import logging

from vpx.sdk import Observation

from facemoment.algorithm.analyzers.frame_scoring.output import (
    ScoringConfig,
    ScoreBreakdown,
    ScoreResult,
)
from facemoment.algorithm.analyzers.frame_scoring.types import FilterFunc

logger = logging.getLogger(__name__)


def _get_faces(obs):
    """Get faces from observation data."""
    if obs.data and hasattr(obs.data, 'faces'):
        return obs.data.faces
    return []


class FrameScorer:
    """Scores frames based on quality, aesthetics, and identity.

    Combines multiple scoring components with configurable weights.
    Supports hard filters for automatic disqualification.

    Args:
        config: Scoring configuration (default: ScoringConfig()).

    Example:
        >>> scorer = FrameScorer(ScoringConfig(weight_technical=0.5))
        >>> result = scorer.score(face_obs=face_obs, quality_obs=quality_obs)
        >>> print(f"Score: {result.total_score:.2f}")
    """

    def __init__(self, config: Optional[ScoringConfig] = None):
        self._config = config or ScoringConfig()
        self._hard_filters: List[Tuple[str, FilterFunc]] = self._build_hard_filters()

    def _build_hard_filters(self) -> List[Tuple[str, FilterFunc]]:
        """Build list of hard filter functions."""
        filters = []

        # No face detected
        def no_face_filter(face_obs, pose_obs, quality_obs):
            if face_obs is None:
                return True, "no_face_obs"
            face_count = face_obs.signals.get("face_count", 0)
            if face_count == 0:
                return True, "no_face_detected"
            return False, ""

        filters.append(("no_face", no_face_filter))

        # Severe blur
        def blur_filter(face_obs, pose_obs, quality_obs):
            if quality_obs is None:
                return False, ""  # Can't check, don't filter
            blur_score = quality_obs.signals.get("blur_score", 100)
            if blur_score < self._config.min_blur_score:
                return True, f"severe_blur ({blur_score:.1f} < {self._config.min_blur_score})"
            return False, ""

        filters.append(("blur", blur_filter))

        # Low face confidence
        def confidence_filter(face_obs, pose_obs, quality_obs):
            if face_obs is None or not _get_faces(face_obs):
                return False, ""  # Handled by no_face filter
            main_face = _get_faces(face_obs)[0]
            if main_face.confidence < self._config.min_face_confidence:
                return True, f"low_confidence ({main_face.confidence:.2f} < {self._config.min_face_confidence})"
            return False, ""

        filters.append(("confidence", confidence_filter))

        # Head cut-off (face not fully inside frame)
        def cutoff_filter(face_obs, pose_obs, quality_obs):
            if face_obs is None or not _get_faces(face_obs):
                return False, ""
            main_face = _get_faces(face_obs)[0]
            if not main_face.inside_frame:
                return True, "head_cutoff"
            return False, ""

        filters.append(("cutoff", cutoff_filter))

        return filters

    def score(
        self,
        face_obs: Optional[Observation] = None,
        pose_obs: Optional[Observation] = None,
        quality_obs: Optional[Observation] = None,
        classifier_obs: Optional[Observation] = None,
    ) -> ScoreResult:
        """Score a frame based on available observations."""
        breakdown = ScoreBreakdown()

        # Apply hard filters first
        if self._config.enable_hard_filters:
            for filter_name, filter_func in self._hard_filters:
                is_filtered, reason = filter_func(face_obs, pose_obs, quality_obs)
                if is_filtered:
                    return ScoreResult(
                        total_score=0.0,
                        technical_score=0.0,
                        action_score=0.0,
                        identity_score=0.0,
                        is_filtered=True,
                        filter_reason=reason,
                        breakdown=breakdown,
                    )

        # Calculate component scores
        technical_score, breakdown.technical = self._score_technical(
            face_obs, quality_obs
        )
        action_score, breakdown.action = self._score_action(
            face_obs, pose_obs, classifier_obs
        )
        identity_score, breakdown.identity = self._score_identity(
            face_obs, classifier_obs
        )

        # Weighted sum
        total_score = (
            self._config.weight_technical * technical_score
            + self._config.weight_action * action_score
            + self._config.weight_identity * identity_score
        )

        return ScoreResult(
            total_score=total_score,
            technical_score=technical_score,
            action_score=action_score,
            identity_score=identity_score,
            is_filtered=False,
            filter_reason=None,
            breakdown=breakdown,
        )

    def _score_technical(
        self,
        face_obs: Optional[Observation],
        quality_obs: Optional[Observation],
    ) -> Tuple[float, Dict[str, float]]:
        """Calculate technical quality score."""
        breakdown = {}
        scores = []

        if quality_obs is not None:
            blur = quality_obs.signals.get("blur_score", 0)
            blur_score = min(1.0, blur / 100.0)
            breakdown["blur"] = blur_score
            scores.append(blur_score)

            brightness = quality_obs.signals.get("brightness", 128)
            if brightness < self._config.optimal_brightness_min:
                brightness_score = brightness / self._config.optimal_brightness_min
            elif brightness > self._config.optimal_brightness_max:
                brightness_score = max(
                    0, 1 - (brightness - self._config.optimal_brightness_max) / 75
                )
            else:
                brightness_score = 1.0
            breakdown["brightness"] = brightness_score
            scores.append(brightness_score)

            contrast = quality_obs.signals.get("contrast", 0)
            contrast_score = min(1.0, contrast / 60.0)
            breakdown["contrast"] = contrast_score
            scores.append(contrast_score)

        if face_obs is not None and _get_faces(face_obs):
            main_face = _get_faces(face_obs)[0]
            conf_score = main_face.confidence
            breakdown["face_confidence"] = conf_score
            scores.append(conf_score)

        if not scores:
            return 0.5, breakdown

        return sum(scores) / len(scores), breakdown

    def _score_action(
        self,
        face_obs: Optional[Observation],
        pose_obs: Optional[Observation],
        classifier_obs: Optional[Observation],
    ) -> Tuple[float, Dict[str, float]]:
        """Calculate action/aesthetics score."""
        breakdown = {}
        scores = []

        if face_obs is not None and _get_faces(face_obs):
            main_face = _get_faces(face_obs)[0]

            yaw = abs(main_face.yaw) if main_face.yaw else 0
            pitch = abs(main_face.pitch) if main_face.pitch else 0

            if yaw <= self._config.frontal_yaw_bonus:
                yaw_score = 1.0
            elif yaw <= self._config.max_head_yaw:
                yaw_score = 1.0 - (yaw - self._config.frontal_yaw_bonus) / (
                    self._config.max_head_yaw - self._config.frontal_yaw_bonus
                )
            else:
                yaw_score = 0.3

            pitch_score = max(0.3, 1.0 - pitch / self._config.max_head_pitch)

            direction_score = (yaw_score + pitch_score) / 2
            breakdown["face_direction"] = direction_score
            scores.append(direction_score)

            expression = main_face.expression if main_face.expression else 0
            happy = main_face.signals.get("em_happy", 0)
            neutral = main_face.signals.get("em_neutral", 1)

            if expression > self._config.expression_boost_threshold:
                expr_score = 0.7 + 0.3 * min(1.0, expression)
            elif happy > 0.3:
                expr_score = 0.8 + 0.2 * happy
            else:
                expr_score = 0.5 + 0.3 * (1 - neutral)

            breakdown["expression"] = expr_score
            scores.append(expr_score)

            center_dist = main_face.center_distance
            composition_score = max(0.3, 1.0 - center_dist * 1.5)
            breakdown["composition"] = composition_score
            scores.append(composition_score)

        if pose_obs is not None:
            person_count = pose_obs.signals.get("person_count", 0)
            hands_raised = pose_obs.signals.get("hands_raised_count", 0)

            pose_score = 0.5
            if person_count > 0:
                pose_score = 0.6
            if hands_raised > 0:
                pose_score = 0.9

            breakdown["pose_energy"] = pose_score
            scores.append(pose_score)

        if not scores:
            return 0.5, breakdown

        return sum(scores) / len(scores), breakdown

    def _score_identity(
        self,
        face_obs: Optional[Observation],
        classifier_obs: Optional[Observation],
    ) -> Tuple[float, Dict[str, float]]:
        """Calculate identity safety score."""
        breakdown = {}
        scores = []

        if face_obs is not None and _get_faces(face_obs):
            main_face = _get_faces(face_obs)[0]

            stability_score = main_face.confidence
            breakdown["face_stability"] = stability_score
            scores.append(stability_score)

            inside_score = 1.0 if main_face.inside_frame else 0.5
            breakdown["inside_frame"] = inside_score
            scores.append(inside_score)

        if classifier_obs is not None and classifier_obs.data:
            data = classifier_obs.data
            if hasattr(data, "main_face") and data.main_face:
                main_conf = data.main_face.face.confidence
                track_len = data.main_face.track_length

                track_score = min(1.0, track_len / 10.0)
                breakdown["track_stability"] = track_score
                scores.append(track_score)

                breakdown["main_confidence"] = main_conf
                scores.append(main_conf)

        if not scores:
            return 0.5, breakdown

        return sum(scores) / len(scores), breakdown

    def add_filter(self, name: str, filter_func: FilterFunc) -> None:
        """Add a custom hard filter."""
        self._hard_filters.append((name, filter_func))

    def remove_filter(self, name: str) -> bool:
        """Remove a hard filter by name."""
        for i, (filter_name, _) in enumerate(self._hard_filters):
            if filter_name == name:
                self._hard_filters.pop(i)
                return True
        return False

    @property
    def config(self) -> ScoringConfig:
        """Get current scoring configuration."""
        return self._config

    @config.setter
    def config(self, value: ScoringConfig) -> None:
        """Set scoring configuration."""
        self._config = value
