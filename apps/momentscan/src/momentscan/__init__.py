# Enable sub-package discovery across multiple installed packages
import pkgutil
__path__ = pkgutil.extend_path(__path__, __name__)

"""momentscan — Portrait moment analysis.

    >>> import momentscan as ms
    >>> results = ms.run("video.mp4")
    >>> result = ms.extract_signals(image_bgr)
"""

from momentscan.main import DEFAULT_FPS, DEFAULT_BACKEND, run, extract_signals

__all__ = ["DEFAULT_FPS", "DEFAULT_BACKEND", "run", "extract_signals"]
