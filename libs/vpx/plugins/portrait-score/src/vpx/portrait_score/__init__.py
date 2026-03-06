"""Backward-compat shim — portrait.score has moved to momentscan."""

from momentscan.algorithm.analyzers.portrait_score import (  # noqa: F401
    PortraitScoreAnalyzer,
    PortraitScoreOutput,
)

__all__ = [
    "PortraitScoreAnalyzer",
    "PortraitScoreOutput",
]
