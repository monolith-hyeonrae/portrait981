"""Workflow template loading and injection."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

# Default templates directory: package-relative templates/
_PACKAGE_DIR = Path(__file__).parent
_DEFAULT_TEMPLATES_DIR = _PACKAGE_DIR / "templates"


def load_template(name: str, templates_dir: Optional[Path] = None) -> dict:
    """Load a workflow JSON template by name or file path.

    If *name* points to an existing ``.json`` file it is loaded directly.
    Otherwise it is treated as a template name and looked up in
    *templates_dir* (or the package ``templates/`` directory).

    Args:
        name: Template name (without .json) **or** path to a .json file.
        templates_dir: Directory to search. Defaults to package templates/.

    Returns:
        Parsed workflow dict.

    Raises:
        FileNotFoundError: If template file not found.
    """
    # Direct file path — load as-is
    direct = Path(name)
    if direct.suffix == ".json" and direct.exists():
        with open(direct, "r", encoding="utf-8") as f:
            return json.load(f)

    # Also try {name}.json in current directory
    cwd_path = Path(f"{name}.json")
    if cwd_path.exists():
        with open(cwd_path, "r", encoding="utf-8") as f:
            return json.load(f)

    # Template name lookup in templates_dir or package templates/
    search_dir = Path(templates_dir) if templates_dir else _DEFAULT_TEMPLATES_DIR
    path = search_dir / f"{name}.json"

    if not path.exists():
        searched = [str(cwd_path.resolve()), str(path)]
        raise FileNotFoundError(
            f"Workflow template '{name}' not found. Searched:\n"
            + "\n".join(f"  - {s}" for s in searched)
        )

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def inject_references(
    workflow: dict,
    ref_paths: List[str],
    node_ids: Optional[List[str]] = None,
) -> dict:
    """Inject reference image paths into LoadImage nodes.

    Target nodes are selected by:
    1. *node_ids* — explicit node IDs (highest priority).
    2. Nodes with ``_meta.role="reference"`` (default convention).

    If there are more ref_paths than target nodes, paths cycle.
    If there are more target nodes than ref_paths, extra nodes get
    cycled paths.

    Args:
        workflow: Workflow dict (will be deep-copied).
        ref_paths: List of image file paths.
        node_ids: Optional explicit node IDs to inject into.

    Returns:
        New workflow dict with injected references.
    """
    if not ref_paths:
        return workflow

    wf = copy.deepcopy(workflow)

    if node_ids:
        # Explicit node targeting
        targets = node_ids
    else:
        # Auto-detect by _meta.role="reference"
        targets = []
        for node_id, node in wf.items():
            if not isinstance(node, dict):
                continue
            if node.get("class_type") != "LoadImage":
                continue
            meta = node.get("_meta", {})
            if meta.get("role") == "reference":
                targets.append(node_id)

    for i, node_id in enumerate(targets):
        if node_id not in wf:
            continue
        path = ref_paths[i % len(ref_paths)]
        wf[node_id]["inputs"]["image"] = path

    return wf


def inject_prompt(workflow: dict, prompt: str) -> dict:
    """Inject text prompt into CLIPTextEncode positive node.

    Finds nodes with class_type="CLIPTextEncode" and _meta.role="positive",
    then replaces their text input.

    Args:
        workflow: Workflow dict (will be deep-copied).
        prompt: Text prompt to inject.

    Returns:
        New workflow dict with injected prompt.
    """
    if not prompt:
        return workflow

    wf = copy.deepcopy(workflow)

    for node_id, node in wf.items():
        if not isinstance(node, dict):
            continue
        if node.get("class_type") != "CLIPTextEncode":
            continue
        meta = node.get("_meta", {})
        if meta.get("role") == "positive":
            wf[node_id]["inputs"]["text"] = prompt

    return wf
