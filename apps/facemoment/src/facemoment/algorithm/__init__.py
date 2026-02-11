import pkgutil
__path__ = pkgutil.extend_path(__path__, __name__)

from vpx.sdk import BaseAnalyzer, Observation
from visualpath.core.module import Module

# Backwards compatibility alias
BaseFusion = Module

__all__ = ["BaseAnalyzer", "Observation", "BaseFusion"]
