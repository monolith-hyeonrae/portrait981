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
]
