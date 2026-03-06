"""Backward-compat shim — portrait.score has moved to momentscan."""

from momentscan.algorithm.analyzers.portrait_score.backends import *  # noqa: F401, F403
from momentscan.algorithm.analyzers.portrait_score.backends import (  # noqa: F401
    CLIPPortraitScorer,
    PromptBreakdown,
    AxisDefinition,
    AxisScore,
    CompositeDefinition,
)
