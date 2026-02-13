"""Persistence layer for MemoryBank.

JSON save/load with numpy ndarray ↔ list conversion.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from momentbank.types import MemoryNode, NodeMeta


def save_bank(bank: Any, path: str | Path) -> None:
    """Save MemoryBank to JSON.

    numpy arrays are converted to lists. Includes _version metadata.

    Args:
        bank: MemoryBank instance to save.
        path: Output JSON file path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "person_id": bank.person_id,
        "k": len(bank.nodes),
        "nodes": [_node_to_dict(node) for node in bank.nodes],
        "_config": {
            "k_max": bank.k_max,
            "alpha": bank.alpha,
            "tau_merge": bank.tau_merge,
            "tau_new": bank.tau_new,
            "tau_close": bank.tau_close,
            "q_update_min": bank.q_update_min,
            "q_new_min": bank.q_new_min,
            "temperature": bank.temperature,
            "top_p": bank.top_p,
            "anchor_min_weight": bank.anchor_min_weight,
        },
        "_next_id": bank._next_id,
        "_version": {
            "app": "momentbank",
            "app_version": "0.1.0",
            "embed_model": "arcface-r100",
            "gate_version": "v1",
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_bank(path: str | Path) -> Any:
    """Load MemoryBank from JSON.

    Lists are converted back to numpy arrays and L2 normalized.

    Args:
        path: Path to memory_bank.json.

    Returns:
        MemoryBank instance.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    from momentbank.bank import MemoryBank

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Memory bank file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    config = data.get("_config", {})
    bank = MemoryBank(
        person_id=data.get("person_id", 0),
        k_max=config.get("k_max", 10),
        alpha=config.get("alpha", 0.1),
        tau_merge=config.get("tau_merge", 0.5),
        tau_new=config.get("tau_new", 0.3),
        tau_close=config.get("tau_close", 0.8),
        q_update_min=config.get("q_update_min", 0.3),
        q_new_min=config.get("q_new_min", 0.5),
        temperature=config.get("temperature", 0.1),
        top_p=config.get("top_p", 3),
        anchor_min_weight=config.get("anchor_min_weight", 0.15),
    )

    bank._next_id = data.get("_next_id", 0)
    bank.nodes = [_dict_to_node(node_data) for node_data in data.get("nodes", [])]

    return bank


def _node_to_dict(node: MemoryNode) -> dict:
    """Convert MemoryNode to JSON-serializable dict."""
    return {
        "node_id": node.node_id,
        "vec_id": node.vec_id.tolist(),
        "rep_images": node.rep_images,
        "rep_qualities": node._rep_qualities,
        "meta_hist": {
            "yaw_bins": node.meta_hist.yaw_bins,
            "pitch_bins": node.meta_hist.pitch_bins,
            "expression_bins": node.meta_hist.expression_bins,
            "quality_best": node.meta_hist.quality_best,
            "quality_mean": node.meta_hist.quality_mean,
            "hit_count": node.meta_hist.hit_count,
            "last_updated_ms": node.meta_hist.last_updated_ms,
        },
    }


def _dict_to_node(data: dict) -> MemoryNode:
    """Convert dict from JSON to MemoryNode."""
    vec = np.array(data["vec_id"], dtype=np.float32)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm

    meta_data = data.get("meta_hist", {})
    meta = NodeMeta(
        yaw_bins=meta_data.get("yaw_bins", {}),
        pitch_bins=meta_data.get("pitch_bins", {}),
        expression_bins=meta_data.get("expression_bins", {}),
        quality_best=meta_data.get("quality_best", 0.0),
        quality_mean=meta_data.get("quality_mean", 0.0),
        hit_count=meta_data.get("hit_count", 0),
        last_updated_ms=meta_data.get("last_updated_ms", 0.0),
    )

    return MemoryNode(
        node_id=data["node_id"],
        vec_id=vec,
        rep_images=data.get("rep_images", []),
        meta_hist=meta,
        _rep_qualities=data.get("rep_qualities", []),
    )
