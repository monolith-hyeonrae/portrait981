"""High-level API for momentscan.

    >>> import momentscan as ms
    >>> results = ms.run("video.mp4")
    >>> result = ms.extract_signals(image_bgr)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Union

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_FPS = 2
DEFAULT_BACKEND = "simple"


def run(
    video: Union[str, Path],
    *,
    quality_model: Optional[str] = None,
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
        quality_model=quality_model,
        expression_model=expression_model,
        pose_model=pose_model,
    )
    return app.run(video, fps=fps)


def extract_signals(
    image: np.ndarray,
    *,
    quality_model: Optional[str] = None,
    expression_model: Optional[str] = None,
    pose_model: Optional[str] = None,
):
    """Extract 65D signals from a single image via FlowGraph pipeline.

    Same analyzers and bind_observations path as video analysis.

    Args:
        image: BGR image (np.ndarray).

    Returns:
        FrameResult with signals, judgment, face_detected, etc.
    """
    from momentscan.v2 import MomentscanV2
    app = MomentscanV2(
        quality_model=quality_model,
        expression_model=expression_model,
        pose_model=pose_model,
    )
    return app.run_single_image(image)
