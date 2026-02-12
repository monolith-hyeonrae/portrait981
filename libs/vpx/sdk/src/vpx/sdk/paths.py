"""Model checkpoint path utilities.

Centralizes ML model storage to a single directory (default: ``{CWD}/models``).
Override with the ``VPX_MODELS_DIR`` environment variable.
"""

import os
from pathlib import Path


def get_models_dir() -> Path:
    """Return the models directory, creating it if it doesn't exist.

    Resolution order:
        1. ``VPX_MODELS_DIR`` environment variable (absolute or relative to CWD).
        2. ``{CWD}/models`` (default).

    Returns:
        Absolute path to the models directory.
    """
    models_dir = Path(os.environ.get("VPX_MODELS_DIR", "models"))
    if not models_dir.is_absolute():
        models_dir = Path.cwd() / models_dir
    models_dir.mkdir(parents=True, exist_ok=True)
    return models_dir
