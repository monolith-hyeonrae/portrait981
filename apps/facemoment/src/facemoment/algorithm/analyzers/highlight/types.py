"""Type definitions for highlight fusion."""

from typing import Optional
from dataclasses import dataclass


@dataclass
class ExpressionState:
    """State for expression spike detection using EWMA."""

    ewma: float = 0.0
    ewma_var: float = 0.01  # Variance for z-score
    count: int = 0


@dataclass
class AdaptiveEmotionState:
    """Adaptive per-face happy tracking with baseline and recent values.

    Tracks happy emotion with:
    - baseline: Slow-moving average representing the person's "normal" happy level
    - recent: Fast-moving average representing current happy level
    - spike: recent - baseline, representing relative change from normal
    - spike_start_ns: When spike first exceeded threshold (for sustained detection)

    This allows detecting happy spikes relative to each person's baseline,
    not absolute thresholds.
    """

    # Baseline (slow EWMA, α≈0.02) - person's typical happy level
    baseline: float = 0.3

    # Recent (fast EWMA, α≈0.08) - current happy level (smoothed over ~1sec)
    recent: float = 0.3

    # Frame count for warmup
    count: int = 0

    # Spike sustain tracking
    spike_start_ns: Optional[int] = None  # When spike first exceeded threshold

    @property
    def spike(self) -> float:
        """Happy spike (recent - baseline)."""
        return self.recent - self.baseline
