"""Core abstractions for visualpath.

This module provides the fundamental building blocks for video analysis pipelines:

Primary interface:
- Module: Unified base class for all processing components
- Observation: Analysis results and trigger decisions

Other:
- IsolationLevel: Enum for plugin isolation levels
- Path: A group of modules with shared configuration
- PathOrchestrator: Orchestrates multiple Paths
"""

from visualpath.core.module import Module, RuntimeInfo
from visualpath.core.observation import Observation, DummyAnalyzer
from visualpath.core.isolation import IsolationLevel, IsolationConfig
from visualpath.core.path import Path, PathConfig, PathOrchestrator
from visualpath.core.capabilities import Capability, ModuleCapabilities, PortSchema
from visualpath.core.compat import CompatibilityReport, check_compatibility
from visualpath.core.error_policy import ErrorPolicy
from visualpath.core.profile import ProfileName, ExecutionProfile, resolve_profile
from visualpath.core.graph import toposort_modules

__all__ = [
    # Primary interface
    "Module",
    "RuntimeInfo",
    # Data types
    "Observation",
    # Testing
    "DummyAnalyzer",
    # Isolation
    "IsolationLevel",
    "IsolationConfig",
    # Path
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
    # Profiles
    "ProfileName",
    "ExecutionProfile",
    "resolve_profile",
    # Graph utilities
    "toposort_modules",
]
