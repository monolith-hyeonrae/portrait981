"""File-based pose + expression pivot system.

Loads pose targets and expression pivots from YAML files.
Falls back to code-defined defaults when no YAML directory is provided.

Directory layout:
    collections/portrait-v1/
      poses/
        frontal.yaml
        three-quarter.yaml
        ...
      pivots/
        warm_smile/
          pivot.yaml
        cool_gaze/
          pivot.yaml
        ...
"""

from __future__ import annotations

import logging
import math
import operator
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Lazy yaml import — only needed when loading files
_yaml = None


def _get_yaml():
    global _yaml
    if _yaml is None:
        import yaml
        _yaml = yaml
    return _yaml


# ── Data Types ──


@dataclass(frozen=True)
class PoseTarget:
    """Named pose target defined by (yaw, pitch) center + acceptance radius.

    yaw is absolute (symmetric left/right matching via |yaw|).
    """

    name: str
    yaw: float
    pitch: float
    r_accept: float = 15.0


@dataclass(frozen=True)
class ExpressionPivot:
    """Named expression pivot with signal-based rules.

    rules: Dict mapping signal_name → condition string (e.g., ">= 1.0").
    Empty rules = catch-all (typically "neutral").
    """

    name: str
    rules: Dict[str, str] = field(default_factory=dict)
    priority: int = 10


# ── Defaults (no YAML fallback) ──

DEFAULT_POSES: List[PoseTarget] = [
    PoseTarget("frontal", yaw=0, pitch=0, r_accept=15),
    PoseTarget("three-quarter", yaw=30, pitch=0, r_accept=15),
    PoseTarget("side-profile", yaw=55, pitch=0, r_accept=20),
    PoseTarget("looking-up", yaw=10, pitch=20, r_accept=15),
]

DEFAULT_PIVOTS: List[ExpressionPivot] = [
    ExpressionPivot("neutral", {}, priority=99),
    # AU-based (precise, requires face.au analyzer)
    ExpressionPivot(
        "excited",
        {"au12_lip_corner": ">= 1.0", "au25_lips_part": ">= 1.5"},
        priority=1,
    ),
    ExpressionPivot("smile", {"au12_lip_corner": ">= 1.0"}, priority=2),
    ExpressionPivot("surprised", {"au25_lips_part": ">= 1.5"}, priority=3),
    # smile_intensity fallback (works with face.expression alone)
    ExpressionPivot("excited", {"smile_intensity": ">= 0.7"}, priority=9),
    ExpressionPivot("smile", {"smile_intensity": ">= 0.4"}, priority=10),
]


# ── YAML Loading ──


def load_poses(poses_dir: Path) -> List[PoseTarget]:
    """Load PoseTarget list from poses/ directory.

    Each YAML file defines one pose target:
        name: three-quarter
        yaw: 30
        pitch: 0
        r_accept: 15

    Returns DEFAULT_POSES if directory doesn't exist or is empty.
    """
    if not poses_dir.is_dir():
        logger.debug("Poses dir not found: %s — using defaults", poses_dir)
        return list(DEFAULT_POSES)

    yaml = _get_yaml()
    poses = []
    for path in sorted(poses_dir.glob("*.yaml")):
        with open(path) as f:
            data = yaml.safe_load(f)
        if not data:
            continue
        poses.append(PoseTarget(
            name=str(data.get("name", path.stem)),
            yaw=float(data.get("yaw", 0)),
            pitch=float(data.get("pitch", 0)),
            r_accept=float(data.get("r_accept", 15.0)),
        ))

    if not poses:
        logger.debug("No pose YAMLs in %s — using defaults", poses_dir)
        return list(DEFAULT_POSES)

    logger.info("Loaded %d poses from %s", len(poses), poses_dir)
    return poses


def load_pivots(pivots_dir: Path) -> List[ExpressionPivot]:
    """Load ExpressionPivot list from pivots/ directory.

    Each subdirectory contains a pivot.yaml:
        name: warm_smile
        rules:
          au12_lip_corner: ">= 1.0"
          au6_cheek_raiser: ">= 0.5"
        priority: 1

    Returns DEFAULT_PIVOTS if directory doesn't exist or is empty.
    """
    if not pivots_dir.is_dir():
        logger.debug("Pivots dir not found: %s — using defaults", pivots_dir)
        return list(DEFAULT_PIVOTS)

    yaml = _get_yaml()
    pivots = []

    # Check subdirectories first (pivots/warm_smile/pivot.yaml)
    for subdir in sorted(pivots_dir.iterdir()):
        pivot_file = subdir / "pivot.yaml" if subdir.is_dir() else None
        if pivot_file is None or not pivot_file.exists():
            # Also check for flat YAML files (pivots/warm_smile.yaml)
            if subdir.is_file() and subdir.suffix == ".yaml":
                pivot_file = subdir
            else:
                continue

        with open(pivot_file) as f:
            data = yaml.safe_load(f)
        if not data:
            continue

        rules = data.get("rules", {})
        # Ensure all rule values are strings
        rules = {str(k): str(v) for k, v in rules.items()}

        pivots.append(ExpressionPivot(
            name=str(data.get("name", subdir.stem if subdir.is_dir() else subdir.stem)),
            rules=rules,
            priority=int(data.get("priority", 10)),
        ))

    if not pivots:
        logger.debug("No pivot YAMLs in %s — using defaults", pivots_dir)
        return list(DEFAULT_PIVOTS)

    logger.info("Loaded %d expression pivots from %s", len(pivots), pivots_dir)
    return pivots


def load_collection_pivots(
    collection_dir: Optional[Path],
) -> Tuple[List[PoseTarget], List[ExpressionPivot]]:
    """Load both poses and pivots from a collection directory.

    Args:
        collection_dir: Path like collections/portrait-v1/ containing
            poses/ and pivots/ subdirectories. None = use defaults.

    Returns:
        (poses, pivots) tuple.
    """
    if collection_dir is None:
        return list(DEFAULT_POSES), list(DEFAULT_PIVOTS)

    collection_dir = Path(collection_dir)
    poses = load_poses(collection_dir / "poses")
    pivots = load_pivots(collection_dir / "pivots")
    return poses, pivots


# ── Classification ──

# Rule parsing: ">= 1.0", "< 0.5", etc.
_RULE_PATTERN = re.compile(r"(>=|<=|>|<|==|!=)\s*(-?[\d.]+)")

_OPS = {
    ">=": operator.ge,
    "<=": operator.le,
    ">": operator.gt,
    "<": operator.lt,
    "==": operator.eq,
    "!=": operator.ne,
}


def _evaluate_rule(value: float, condition: str) -> bool:
    """Evaluate a rule condition like '>= 1.0' against a value."""
    m = _RULE_PATTERN.match(condition.strip())
    if not m:
        return False
    op_str, threshold_str = m.groups()
    return _OPS[op_str](value, float(threshold_str))


def classify_pose(
    yaw: float,
    pitch: float,
    poses: List[PoseTarget],
) -> Optional[str]:
    """Classify frame pose using nearest PoseTarget within r_accept.

    Uses |yaw| for left-right symmetric matching.

    Returns:
        Pose name, or None if no pose is within r_accept.
    """
    best_name = None
    best_dist = float("inf")

    for pose in poses:
        d_yaw = abs(yaw) - pose.yaw
        d_pitch = pitch - pose.pitch
        dist = math.sqrt(d_yaw ** 2 + d_pitch ** 2)
        if dist < best_dist:
            best_dist = dist
            best_name = pose.name

    if best_name is None:
        return None

    # Check r_accept of the best match
    best_pose = next(p for p in poses if p.name == best_name)
    if best_dist > best_pose.r_accept:
        return None

    return best_name


def classify_expression(
    record,  # CollectionRecord — duck typed for testability
    pivots: List[ExpressionPivot],
) -> str:
    """Classify frame expression using signal-based rules.

    Evaluates each pivot's rules against record attributes.
    Returns the matching pivot with lowest priority number (highest precedence).
    Falls back to first empty-rules pivot (typically "neutral").

    Args:
        record: Object with signal attributes (au12_lip_corner, etc.).
        pivots: ExpressionPivot list sorted by priority internally.

    Returns:
        Pivot name string.
    """
    sorted_pivots = sorted(pivots, key=lambda p: p.priority)

    for pivot in sorted_pivots:
        if not pivot.rules:
            continue  # Skip catch-all during rule matching
        if _match_rules(record, pivot.rules):
            return pivot.name

    # Fallback: first pivot with empty rules, or first overall
    for pivot in sorted_pivots:
        if not pivot.rules:
            return pivot.name

    return sorted_pivots[0].name if sorted_pivots else "unknown"


def _match_rules(record, rules: Dict[str, str]) -> bool:
    """Check if all rules match against record attributes.

    Checks direct attributes first, then falls back to clip_axes dict
    for dynamic CLIP axis values.
    """
    for signal_name, condition in rules.items():
        value = getattr(record, signal_name, None)
        if value is None:
            # Fallback to clip_axes dict for dynamic CLIP axis values
            clip_axes = getattr(record, "clip_axes", {})
            value = clip_axes.get(signal_name, 0.0)
        if not _evaluate_rule(float(value), condition):
            return False
    return True
