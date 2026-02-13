"""Model checkpoint and home directory path utilities.

Centralizes ML model storage to ``~/.portrait981/models`` by default.
Override with ``VPX_MODELS_DIR`` or ``PORTRAIT981_HOME`` environment variables.
"""

import os
import warnings
from pathlib import Path


def get_home_dir() -> Path:
    """Return the portrait981 home directory, creating it if needed.

    Resolution order:
        1. ``PORTRAIT981_HOME`` environment variable.
        2. ``~/.portrait981`` (default).

    Returns:
        Absolute path to the home directory.
    """
    home = os.environ.get("PORTRAIT981_HOME")
    if home:
        home_dir = Path(home)
    else:
        home_dir = Path.home() / ".portrait981"
    home_dir.mkdir(parents=True, exist_ok=True)
    return home_dir


def get_models_dir() -> Path:
    """Return the models directory, creating it if it doesn't exist.

    Resolution order:
        1. ``VPX_MODELS_DIR`` environment variable (absolute or relative to CWD).
        2. ``{home}/models`` where *home* is from :func:`get_home_dir`.

    When neither env var is set, checks for a legacy ``{CWD}/models``
    directory and emits a :class:`UserWarning` if found but the new
    default location does not yet exist.

    Returns:
        Absolute path to the models directory.
    """
    env_val = os.environ.get("VPX_MODELS_DIR")
    if env_val:
        models_dir = Path(env_val)
        if not models_dir.is_absolute():
            models_dir = Path.cwd() / models_dir
    else:
        models_dir = get_home_dir() / "models"
        # Migration warning
        cwd_models = Path.cwd() / "models"
        if cwd_models.is_dir() and not models_dir.is_dir():
            warnings.warn(
                f"Found models in {cwd_models} but default moved to {models_dir}. "
                f"Set VPX_MODELS_DIR={cwd_models} or move models.",
                UserWarning,
                stacklevel=2,
            )
    models_dir.mkdir(parents=True, exist_ok=True)
    return models_dir
