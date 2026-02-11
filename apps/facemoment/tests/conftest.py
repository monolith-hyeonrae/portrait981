"""Shared test fixtures and helpers for facemoment tests."""

import importlib.util
import sys
from pathlib import Path

# Load helpers module from the tests directory using importlib to avoid
# polluting sys.path (which causes test module name collisions in monorepo).
_helpers_path = Path(__file__).resolve().parent / "helpers.py"
_spec = importlib.util.spec_from_file_location("helpers", _helpers_path)
_helpers = importlib.util.module_from_spec(_spec)
sys.modules["helpers"] = _helpers
_spec.loader.exec_module(_helpers)

# Re-export helpers for backward compatibility
from helpers import create_test_video, create_mock_frame  # noqa: F401
