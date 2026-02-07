"""Shared pipeline utility functions."""

from typing import List

from visualbase import Frame

from facemoment.moment_detector.extractors.base import Observation


def merge_observations(observations: List[Observation], frame: Frame) -> Observation:
    """Merge multiple observations into a single observation for fusion.

    Combines signals and metadata from all observations into one,
    using the first face-containing observation as the base for faces.
    Propagates main_face_id from face_classifier to merged signals.

    Args:
        observations: List of observations from different extractors.
        frame: The source frame.

    Returns:
        Merged observation with combined signals, faces, and metadata.
    """
    if not observations:
        return Observation(
            source="merged",
            frame_id=frame.frame_id,
            t_ns=frame.t_src_ns,
            signals={},
            faces=[],
            metadata={},
        )

    # Find the observation with face data
    base_obs = None
    for obs in observations:
        if hasattr(obs, 'faces') and obs.faces:
            base_obs = obs
            break

    if base_obs is None:
        base_obs = observations[0]

    # Merge signals from all observations
    merged_signals = {}
    merged_metadata = {}

    for obs in observations:
        merged_signals.update(obs.signals)
        merged_metadata[obs.source] = obs.metadata

        # Copy classifier main_face_id to signals for fusion
        if obs.source == "face_classifier" and obs.data is not None:
            if hasattr(obs.data, 'main_face') and obs.data.main_face is not None:
                merged_signals["main_face_id"] = obs.data.main_face.face.face_id

    return Observation(
        source="merged",
        frame_id=frame.frame_id,
        t_ns=frame.t_src_ns,
        signals=merged_signals,
        faces=getattr(base_obs, 'faces', []),
        metadata=merged_metadata,
    )
