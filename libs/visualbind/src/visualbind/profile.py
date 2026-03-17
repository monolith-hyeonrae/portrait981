"""CategoryProfile dataclass and I/O (load/save).

Profiles are stored as ``_profile.json`` inside each category directory::

    catalog_path/
      categories/
        warm_smile/_profile.json
        cool_gaze/_profile.json
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import yaml

from .signals import SIGNAL_FIELDS as _VB_SIGNAL_FIELDS, _NDIM

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CategoryProfile:
    """Per-category signal profile.

    Contains the mean signal centroid and Fisher ratio importance weights
    computed from reference images.
    """

    name: str
    mean_signals: np.ndarray       # (D,) normalized mean signals
    importance_weights: np.ndarray  # (D,) Fisher ratio weights (sum=1)
    n_refs: int                    # number of reference images


def load_profiles(catalog_path: Path) -> List[CategoryProfile]:
    """Load ``_profile.json`` files from a catalog directory.

    Args:
        catalog_path: catalog root directory.

    Returns:
        List of :class:`CategoryProfile`.

    Raises:
        FileNotFoundError: catalog_path or categories/ does not exist.
        ValueError: No valid profiles found or dimension mismatch.
    """
    categories_dir = catalog_path / "categories"
    if not categories_dir.is_dir():
        raise FileNotFoundError(
            f"Categories directory not found: {categories_dir}"
        )

    profiles: List[CategoryProfile] = []
    for cat_dir in sorted(categories_dir.iterdir()):
        if not cat_dir.is_dir() or cat_dir.name.startswith("_"):
            continue

        profile_path = cat_dir / "_profile.json"
        if not profile_path.exists():
            has_refs = any(
                f.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".avif")
                for f in cat_dir.iterdir()
            )
            if has_refs:
                raise FileNotFoundError(
                    f"Missing _profile.json in category '{cat_dir.name}' "
                    f"(has reference images). Build profiles first."
                )
            continue

        with open(profile_path) as f:
            data = json.load(f)

        mean_signals = np.array(data["mean_signals"], dtype=np.float64)
        importance_weights = np.array(data["importance_weights"], dtype=np.float64)

        # Dimension validation: use signal_fields from JSON metadata when present
        # (supports loading profiles with different dimensionality, e.g. 21D legacy).
        # When signal_fields is absent, accept any self-consistent dimension.
        stored_fields = data.get("signal_fields")
        if stored_fields:
            expected_ndim = len(stored_fields)
            if len(mean_signals) != expected_ndim:
                raise ValueError(
                    f"Profile '{cat_dir.name}' has {len(mean_signals)} signals "
                    f"(expected {expected_ndim}). Regenerate profiles."
                )
        if len(mean_signals) != len(importance_weights):
            raise ValueError(
                f"Profile '{cat_dir.name}' has {len(mean_signals)} signals "
                f"but {len(importance_weights)} weights — dimension mismatch."
            )

        profiles.append(CategoryProfile(
            name=data.get("name", cat_dir.name),
            mean_signals=mean_signals,
            importance_weights=importance_weights,
            n_refs=data.get("n_refs", 0),
        ))

    if not profiles:
        raise ValueError(
            f"No valid category profiles found in {categories_dir}."
        )

    logger.info("Loaded %d catalog profiles from %s", len(profiles), catalog_path)
    return profiles


def save_profiles(
    catalog_path: Path,
    profiles: List[CategoryProfile],
    signal_fields: tuple[str, ...] | list[str] | None = None,
) -> None:
    """Save ``_profile.json`` for each category.

    Args:
        catalog_path: catalog root directory.
        profiles: profiles to save.
        signal_fields: field names to store as metadata.  ``None`` uses the
            visualbind default :data:`SIGNAL_FIELDS`.  Callers with a different
            dimensionality (e.g. momentscan 21D) should pass their own field tuple.
    """
    fields_meta = list(signal_fields) if signal_fields is not None else list(_VB_SIGNAL_FIELDS)

    for profile in profiles:
        cat_dir = catalog_path / "categories" / profile.name
        cat_dir.mkdir(parents=True, exist_ok=True)

        data = {
            "name": profile.name,
            "mean_signals": profile.mean_signals.tolist(),
            "importance_weights": profile.importance_weights.tolist(),
            "n_refs": profile.n_refs,
            "signal_fields": fields_meta,
        }

        profile_path = cat_dir / "_profile.json"
        with open(profile_path, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    logger.info("Saved %d profiles to %s", len(profiles), catalog_path)


def load_clip_axes(catalog_path: Path) -> list:
    """Load CLIP axis definitions from catalog category.yaml files.

    Each category directory may contain a ``category.yaml`` with a ``clip_axis``
    section defining prompts, negative prompts, action, and threshold.

    Args:
        catalog_path: catalog root directory.

    Returns:
        List of axis definition dicts with keys:
        ``name``, ``prompts``, ``neg_prompts``, ``action``, ``threshold``.
        Empty list if no clip_axis sections found.
    """
    categories_dir = catalog_path / "categories"
    if not categories_dir.is_dir():
        return []

    axes: list[dict] = []
    for cat_dir in sorted(categories_dir.iterdir()):
        if not cat_dir.is_dir() or cat_dir.name.startswith("_"):
            continue

        yaml_path = cat_dir / "category.yaml"
        if not yaml_path.exists():
            continue

        with open(yaml_path) as f:
            data = yaml.safe_load(f) or {}

        clip_axis = data.get("clip_axis")
        if not clip_axis:
            continue

        prompts = clip_axis.get("prompts", [])
        neg_prompts = clip_axis.get("neg_prompts", [])
        action = clip_axis.get("action", "select")
        threshold = float(data.get("threshold", 0.5))

        if not prompts:
            continue

        axes.append({
            "name": data.get("name", cat_dir.name),
            "prompts": tuple(prompts),
            "neg_prompts": tuple(neg_prompts),
            "action": action,
            "threshold": threshold,
        })

    if axes:
        logger.info("Loaded %d CLIP axes from %s", len(axes), catalog_path)

    return axes
