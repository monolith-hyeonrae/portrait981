# Enable sub-package discovery across multiple installed packages
import pkgutil
__path__ = pkgutil.extend_path(__path__, __name__)

"""momentscan - Moment analysis for video.

Quick Start (v2):
    >>> import momentscan as ms
    >>> results = ms.run("video.mp4", expression_model="models/bind_v4.pkl")

Legacy (v1):
    >>> results = ms.run("video.mp4", version="v1")
"""

from momentscan.main import (
    DEFAULT_FPS,
    DEFAULT_VERSION,
    Result,
    MomentscanApp,
    run,
)
from momentscan.v2 import MomentscanV2, FrameResult


__all__ = [
    "DEFAULT_FPS",
    "DEFAULT_VERSION",
    "run",
    # v2
    "MomentscanV2",
    "FrameResult",
    # v1 (legacy)
    "MomentscanApp",
    "Result",
]
