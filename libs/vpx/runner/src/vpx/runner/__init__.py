"""vpx-runner - Lightweight analyzer runner for vpx modules.

Example:
    >>> from vpx.runner import LiteRunner
    >>> result = runner.run("video.mp4", max_frames=10)
    >>> print(result.frame_count)
"""

from vpx.runner.runner import LiteRunner, RunResult

__all__ = ["LiteRunner", "RunResult"]
