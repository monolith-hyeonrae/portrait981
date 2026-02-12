"""FrameScoringAnalyzer — Module wrapper around FrameScorer.

Integrates frame scoring into the FlowGraph pipeline as a proper Module.
Consumes face.detect (required) and optional analyzers, produces ScoreResult.
"""

from typing import Optional

from vpx.sdk import Module, Observation

from facemoment.algorithm.analyzers.frame_scoring.output import (
    ScoringConfig,
    ScoreResult,
)
from facemoment.algorithm.analyzers.frame_scoring.scorer import FrameScorer


class FrameScoringAnalyzer(Module):
    """Frame scoring analyzer.

    Wraps FrameScorer as a Module for use in FlowGraph pipelines.

    depends: ["face.detect"]
    optional_depends: ["face.expression", "face.classify", "body.pose", "frame.quality"]
    """

    depends = ["face.detect"]
    optional_depends = ["face.expression", "face.classify", "body.pose", "frame.quality"]

    def __init__(self, config: Optional[ScoringConfig] = None):
        self._scorer = FrameScorer(config)

    @property
    def name(self) -> str:
        return "frame.scoring"

    def process(self, frame, deps=None) -> Optional[Observation]:
        """Score a frame using available dependency observations.

        Args:
            frame: Input frame with .frame_id, .t_src_ns.
            deps: Dict of observations from dependency modules.

        Returns:
            Observation with ScoreResult in .data and score signals.
        """
        if not deps:
            result = self._scorer.score()
        else:
            result = self._scorer.score(
                face_obs=deps.get("face.detect"),
                pose_obs=deps.get("body.pose"),
                quality_obs=deps.get("frame.quality"),
                classifier_obs=deps.get("face.classify"),
            )

        return Observation(
            source=self.name,
            frame_id=getattr(frame, 'frame_id', 0),
            t_ns=getattr(frame, 't_src_ns', 0),
            signals={
                "total_score": result.total_score,
                "technical_score": result.technical_score,
                "action_score": result.action_score,
                "identity_score": result.identity_score,
                "is_filtered": 1.0 if result.is_filtered else 0.0,
            },
            data=result,
        )
