"""Batch analysis modules for momentscan Phase 1.

Per-video normalization + peak detection 기반 하이라이트 분석.
"""

from momentscan.algorithm.batch.highlight import BatchHighlightEngine
from momentscan.algorithm.batch.types import (
    FrameRecord,
    HighlightWindow,
    HighlightResult,
    HighlightConfig,
)

__all__ = [
    "BatchHighlightEngine",
    "FrameRecord",
    "HighlightWindow",
    "HighlightResult",
    "HighlightConfig",
]
