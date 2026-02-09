"""Output type for gesture analyzer."""

from dataclasses import dataclass, field
from typing import List, Dict, Any


@dataclass
class GestureOutput:
    """Output from GestureAnalyzer.

    Attributes:
        gestures: List of detected gestures.
        hand_landmarks: Raw hand landmark data.
    """
    gestures: List[str] = field(default_factory=list)
    hand_landmarks: List[Dict[str, Any]] = field(default_factory=list)


__all__ = ["GestureOutput"]
