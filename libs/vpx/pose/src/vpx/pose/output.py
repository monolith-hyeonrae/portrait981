"""Output type for pose analyzer."""

from dataclasses import dataclass, field
from typing import List, Dict, Any


@dataclass
class PoseOutput:
    """Output from PoseAnalyzer.

    Attributes:
        keypoints: List of keypoint dicts per person.
        person_count: Number of detected persons.
    """
    keypoints: List[Dict[str, Any]] = field(default_factory=list)
    person_count: int = 0


__all__ = ["PoseOutput"]
