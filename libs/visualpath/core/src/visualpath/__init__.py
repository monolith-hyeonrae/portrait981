"""visualpath - Video analysis pipeline platform.

visualpath provides a plugin-based platform for building video analysis pipelines.

Quick Start:
    >>> import visualpath as vp
    >>>
    >>> # Run a video analysis pipeline (preferred)
    >>> result = vp.run("video.mp4", modules=[face_detector, smile_trigger])
    >>>
    >>> # Create a custom analyzer (decorator)
    >>> @vp.analyzer("brightness")
    >>> def check_brightness(frame):
    ...     return {"brightness": float(frame.data.mean())}
    >>>
    >>> # Create a custom fusion (decorator)
    >>> @vp.fusion(sources=["face"], cooldown=2.0)
    >>> def smile_detector(face):
    ...     if face.get("happy", 0) > 0.5:
    ...         return vp.trigger("smile", score=face["happy"])

For advanced usage, see:
- visualpath.core: Module, Observation (unified API)
- visualpath.flow: FlowGraph, DAG-based pipeline
- visualpath.process: Distributed processing (IPC, workers)
"""

# Enable sub-package discovery across multiple installed packages
# (e.g. visualpath-isolation, visualpath-pathway, visualpath-cli)
import pkgutil
__path__ = pkgutil.extend_path(__path__, __name__)

try:
    from visualpath._version import __version__
except ImportError:
    __version__ = "0.0.0.dev0"

# =============================================================================
# High-level API (recommended)
# =============================================================================
from visualpath.api import (
    # Configuration
    DEFAULT_FPS,
    DEFAULT_COOLDOWN,
    DEFAULT_PRE_SEC,
    DEFAULT_POST_SEC,
    # Decorators
    analyzer,
    fusion,
    trigger,
    # Registry (unified)
    get_module,
    list_modules,
    # Registry (aliases)
    get_analyzer,
    get_fusion,
    list_analyzers,
    list_fusions,
    # Types
    TriggerSpec,
)

# Pipeline runner (from runner.py)
from visualpath.runner import (
    run,
    process_video,  # alias for backward compatibility
    ProcessResult,
    get_backend,
    resolve_modules,
)

# App (convention layer)
from visualpath.app import App

# =============================================================================
# Core exports (for advanced use)
# =============================================================================
from visualpath.core.module import Module
from visualpath.core.observation import Observation, DummyAnalyzer
from visualpath.core.isolation import IsolationLevel, IsolationConfig
from visualpath.core.path import Path, PathConfig, PathOrchestrator
from visualpath.core.capabilities import Capability, ModuleCapabilities, PortSchema
from visualpath.core.compat import CompatibilityReport, check_compatibility
from visualpath.core.error_policy import ErrorPolicy

__all__ = [
    # Configuration
    "DEFAULT_FPS",
    "DEFAULT_COOLDOWN",
    "DEFAULT_PRE_SEC",
    "DEFAULT_POST_SEC",
    # High-level API
    "analyzer",
    "fusion",
    "trigger",
    "run",
    "process_video",
    "get_backend",
    "resolve_modules",
    "get_module",
    "list_modules",
    "get_analyzer",
    "get_fusion",
    "list_analyzers",
    "list_fusions",
    "ProcessResult",
    "TriggerSpec",
    # App
    "App",
    # Core (unified API)
    "Module",
    "Observation",
    "DummyAnalyzer",
    "IsolationLevel",
    "IsolationConfig",
    "Path",
    "PathConfig",
    "PathOrchestrator",
    # Capabilities
    "Capability",
    "ModuleCapabilities",
    "PortSchema",
    "CompatibilityReport",
    "check_compatibility",
    # Error policy
    "ErrorPolicy",
]
