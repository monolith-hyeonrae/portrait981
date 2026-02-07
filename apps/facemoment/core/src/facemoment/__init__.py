# Enable sub-package discovery across multiple installed packages
# (e.g. facemoment-face-detect, facemoment-expression, facemoment-pose, etc.)
import pkgutil
__path__ = pkgutil.extend_path(__path__, __name__)

"""facemoment - Face and moment detection for video analysis.

Quick Start:
    >>> import facemoment as fm
    >>> result = fm.run("video.mp4")
    >>> print(f"Found {len(result.triggers)} highlights")

With options:
    >>> result = fm.run("video.mp4", fps=10, cooldown=3.0)
    >>> result = fm.run("video.mp4", output_dir="./clips")

Frame Scoring:
    >>> from facemoment.moment_detector.scoring import FrameScorer
    >>> scorer = FrameScorer()
    >>> result = scorer.score(face_obs=face_obs, quality_obs=quality_obs)
    >>> print(f"Score: {result.total_score:.2f}")

Pipeline (recommended):
    >>> from facemoment.pipeline import FacemomentPipeline
    >>> pipeline = FacemomentPipeline(extractors=["face", "pose"])
    >>> triggers = pipeline.run(frames)
"""

import warnings as _warnings

from facemoment.main import (
    # Configuration
    DEFAULT_FPS,
    DEFAULT_COOLDOWN,
    # Result type
    Result,
    # High-level API
    run,
    build_modules,
    build_graph,
)

from facemoment.moment_detector import MomentDetector


def __getattr__(name):
    if name == "visualize":
        _warnings.warn(
            "facemoment.visualize is deprecated. Use DebugVisualizer directly: "
            "from facemoment.moment_detector.visualize import DebugVisualizer",
            DeprecationWarning,
            stacklevel=2,
        )
        from facemoment.tools.visualizer import visualize
        return visualize
    if name == "DetectorVisualizer":
        _warnings.warn(
            "facemoment.DetectorVisualizer is deprecated. Use DebugVisualizer directly: "
            "from facemoment.moment_detector.visualize import DebugVisualizer",
            DeprecationWarning,
            stacklevel=2,
        )
        from facemoment.tools.visualizer import DetectorVisualizer
        return DetectorVisualizer
    raise AttributeError(f"module 'facemoment' has no attribute {name!r}")


__all__ = [
    # Configuration
    "DEFAULT_FPS",
    "DEFAULT_COOLDOWN",
    # High-level API
    "run",
    "build_modules",
    "build_graph",
    "Result",
    # Core (deprecated)
    "MomentDetector",
    # Visualization (deprecated, lazy-loaded)
    "visualize",
    "DetectorVisualizer",
]
