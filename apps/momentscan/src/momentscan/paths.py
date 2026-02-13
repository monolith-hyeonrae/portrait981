"""Momentscan output path utilities.

Provides a default output directory under ``~/.portrait981/momentscan/output/``
keyed by video filename stem.
"""

import re
from pathlib import Path

from vpx.sdk.paths import get_home_dir


def get_output_dir(video_path: str | Path) -> Path:
    """Return the default output directory for a video file.

    Resolution: ``{home}/momentscan/output/{sanitized_stem}/``

    If the directory already exists, a numeric suffix is appended to
    avoid overwriting previous results::

        reaction/  →  reaction_2/  →  reaction_3/  → ...

    The directory is **not** created by this function; the caller is
    responsible for calling ``mkdir()`` when ready to write.

    Args:
        video_path: Path to the source video file.

    Returns:
        Absolute path to a non-existing output directory.
    """
    stem = Path(video_path).stem
    sanitized = _sanitize_stem(stem)
    parent = get_home_dir() / "momentscan" / "output"
    candidate = parent / sanitized
    if not candidate.exists():
        return candidate
    n = 2
    while True:
        candidate = parent / f"{sanitized}_{n}"
        if not candidate.exists():
            return candidate
        n += 1


def _sanitize_stem(stem: str) -> str:
    """Sanitize a video filename stem for use as a directory name.

    - Spaces → underscores
    - Remove characters not in ``[a-zA-Z0-9_.-]`` and non-ASCII word chars
    - Collapse consecutive underscores
    - Strip leading/trailing underscores and dots
    - Fall back to ``"untitled"`` if empty
    """
    # Replace spaces with underscores
    s = stem.replace(" ", "_")
    # Keep alphanumeric, underscore, hyphen, dot, and unicode word chars
    s = re.sub(r"[^\w.\-]", "", s)
    # Collapse consecutive underscores
    s = re.sub(r"_+", "_", s)
    # Strip leading/trailing underscores and dots
    s = s.strip("_.")
    return s if s else "untitled"
