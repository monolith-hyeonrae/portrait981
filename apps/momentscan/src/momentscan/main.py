"""High-level API for momentscan.

    >>> import momentscan as ms
    >>> results = ms.run("video.mp4")
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Union

logger = logging.getLogger(__name__)

DEFAULT_FPS = 2
DEFAULT_BACKEND = "simple"


def run(
    video: Union[str, Path],
    *,
    expression_model: Optional[str] = None,
    pose_model: Optional[str] = None,
    fps: int = DEFAULT_FPS,
) -> list:
    """Analyze a video and return per-frame results.

    Returns:
        list[FrameResult]
    """
    from momentscan.v2 import MomentscanV2
    app = MomentscanV2(
        expression_model=expression_model,
        pose_model=pose_model,
    )
    return app.run(video, fps=fps)
