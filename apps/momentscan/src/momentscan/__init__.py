"""momentscan — Portrait moment analysis.

    >>> import momentscan as ms
    >>> results = ms.run("video.mp4")
    >>> result = ms.extract_signals(image_bgr)
"""

from __future__ import annotations

# Enable sub-package discovery across multiple installed packages
import pkgutil
__path__ = pkgutil.extend_path(__path__, __name__)

from pathlib import Path
from typing import Optional, Union

import numpy as np

from momentscan.app import Momentscan

__all__ = ["Momentscan", "run", "extract_signals"]


def run(
    video: Union[str, Path],
    *,
    quality_model: Optional[str] = None,
    expression_model: Optional[str] = None,
    pose_model: Optional[str] = None,
    fps: int = 2,
) -> list:
    """Analyze a video and return per-frame results.

    Returns:
        list[FrameResult]
    """
    app = Momentscan(
        quality_model=quality_model,
        expression_model=expression_model,
        pose_model=pose_model,
    )
    try:
        return app.scan(video, fps=fps)
    finally:
        app.shutdown()


def extract_signals(
    image: np.ndarray,
    *,
    quality_model: Optional[str] = None,
    expression_model: Optional[str] = None,
    pose_model: Optional[str] = None,
):
    """Extract 65D signals from a single image via FlowGraph pipeline.

    Returns:
        FrameResult with signals, judgment, face_detected, etc.
    """
    return Momentscan(
        quality_model=quality_model,
        expression_model=expression_model,
        pose_model=pose_model,
    ).run_single_image(image)
