"""Backend protocol definitions for expression analysis."""

from dataclasses import dataclass, field
from typing import Protocol, List, Dict
import numpy as np

from vpx.face_detect.backends.base import DetectedFace


@dataclass
class FaceExpression:
    """Result from expression analysis backend.

    Attributes:
        action_units: Dictionary of Action Unit activations {AU_name: intensity}.
        emotions: Dictionary of emotion probabilities {emotion: probability}.
        expression_intensity: Overall expression intensity [0, 1].
        dominant_emotion: Most likely emotion label.
    """

    action_units: Dict[str, float] = field(default_factory=dict)
    emotions: Dict[str, float] = field(default_factory=dict)
    expression_intensity: float = 0.0
    dominant_emotion: str = "neutral"


class ExpressionBackend(Protocol):
    """Protocol for expression/AU analysis backends.

    Implementations analyze facial expressions from detected face regions.
    Examples: Py-Feat, FER, EmotionNet.
    """

    def initialize(self, device: str = "cuda:0") -> None:
        """Initialize the backend and load models."""
        ...

    def analyze(
        self, image: np.ndarray, faces: List[DetectedFace]
    ) -> List[FaceExpression]:
        """Analyze expressions for detected faces."""
        ...

    def cleanup(self) -> None:
        """Release resources and unload models."""
        ...


__all__ = ["FaceExpression", "ExpressionBackend"]
