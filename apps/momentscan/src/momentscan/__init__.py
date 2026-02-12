# Enable sub-package discovery across multiple installed packages
# (e.g. momentscan-face-detect, momentscan-expression, momentscan-pose, etc.)
import pkgutil
__path__ = pkgutil.extend_path(__path__, __name__)

"""momentscan - Moment highlight detection for video analysis.

Quick Start:
    >>> import momentscan as ms
    >>> result = ms.run("video.mp4")
    >>> print(f"Found {len(result.highlights)} highlights")

With options:
    >>> result = ms.run("video.mp4", fps=10)
    >>> result = ms.run("video.mp4", output_dir="./output")

Frame Scoring:
    >>> from momentscan.algorithm.analyzers.frame_scoring import FrameScorer
    >>> scorer = FrameScorer()
    >>> result = scorer.score(face_obs=face_obs, quality_obs=quality_obs)
    >>> print(f"Score: {result.total_score:.2f}")
"""

from momentscan.main import (
    # Configuration
    DEFAULT_FPS,
    # Result type
    Result,
    # App class
    MomentscanApp,
    # High-level API
    run,
)


__all__ = [
    # Configuration
    "DEFAULT_FPS",
    # High-level API
    "run",
    "MomentscanApp",
    "Result",
]
