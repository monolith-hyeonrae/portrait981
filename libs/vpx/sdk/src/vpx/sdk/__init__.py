"""vpx-sdk — Minimal SDK for building visualpath modules/plugins.

Owns the core analyzer types (Module, Observation, ProcessingStep)
and re-exports capability/boundary types from visualpath.

Example:
    >>> from vpx.sdk import Module, Observation, Capability, ModuleCapabilities
    >>> from vpx.sdk.testing import PluginTestHarness
"""

# Analyzer base types (owned by vpx-sdk)
from vpx.sdk.module import Module, BaseAnalyzer
from vpx.sdk.observation import Observation
from vpx.sdk.steps import ProcessingStep, processing_step, get_processing_steps

# Capability & boundary types (re-exported from visualpath)
from visualpath.core.capabilities import Capability, ModuleCapabilities, PortSchema
from visualpath.core.error_policy import ErrorPolicy
from visualpath.core.compat import check_compatibility, CompatibilityReport
from visualpath.core.graph import toposort_modules

# Model path utilities
from vpx.sdk.paths import get_home_dir, get_models_dir

# Crop utilities
from vpx.sdk.crop import CropRatio, face_crop, BBoxSmoother

# Declarative visualization marks
from vpx.sdk.marks import (
    DrawStyle,
    Mark,
    BBoxMark,
    KeypointsMark,
    BarMark,
    LabelMark,
)

__all__ = [
    # Analyzer base
    "Module",
    "BaseAnalyzer",
    "Observation",
    "ProcessingStep",
    "processing_step",
    "get_processing_steps",
    # Capabilities
    "Capability",
    "ModuleCapabilities",
    "PortSchema",
    # Error policy
    "ErrorPolicy",
    # Compatibility
    "check_compatibility",
    "CompatibilityReport",
    # Graph utilities
    "toposort_modules",
    # Model paths
    "get_home_dir",
    "get_models_dir",
    # Crop utilities
    "CropRatio",
    "face_crop",
    "BBoxSmoother",
    # Visualization marks
    "DrawStyle",
    "Mark",
    "BBoxMark",
    "KeypointsMark",
    "BarMark",
    "LabelMark",
]
