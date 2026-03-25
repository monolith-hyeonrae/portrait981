"""VisualBind — multi-observer signal binding framework.

Combines outputs from multiple vision analyzers (AU, emotion, pose, CLIP, etc.)
into unified per-frame scores via pluggable binding strategies.

    >>> from visualbind import CatalogStrategy, extract_signal_vector_from_dict
    >>> strategy = CatalogStrategy()
    >>> strategy.fit({"warm_smile": ref_vectors, "cool_gaze": ref_vectors2})
    >>> scores = strategy.predict(frame_vec)
"""

from visualbind.signals import (
    SIGNAL_FIELDS,
    SIGNAL_RANGES,
    normalize_signal,
    get_signal_fields,
    extract_signal_vector_from_dict,
)
from visualbind.profile import CategoryProfile, load_profiles, save_profiles
from visualbind.analyzer import compute_correlation_matrix, compute_neff
from visualbind.strategies import BindingStrategy
from visualbind.strategies.catalog import CatalogStrategy
from visualbind.strategies.tree import TreeStrategy
from visualbind.strategies.two_stage import TwoStageStrategy
from visualbind.strategies.heuristic import HeuristicStrategy, GateConfig
from visualbind.selector import select_frames, SelectionResult, SelectedFrame

__all__ = [
    # signals
    "SIGNAL_FIELDS",
    "SIGNAL_RANGES",
    "normalize_signal",
    "get_signal_fields",
    "extract_signal_vector_from_dict",
    # profile
    "CategoryProfile",
    "load_profiles",
    "save_profiles",
    # analyzer
    "compute_correlation_matrix",
    "compute_neff",
    # strategies
    "BindingStrategy",
    "CatalogStrategy",
    "TreeStrategy",
    "TwoStageStrategy",
    "HeuristicStrategy",
    "GateConfig",
    # selector
    "select_frames",
    "SelectionResult",
    "SelectedFrame",
]
