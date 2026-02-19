"""Bank command for momentscan CLI.

Shows MemoryBank contents stored in memory_bank.json files.
Supports direct .json path or output directory (globs person_*/memory_bank.json).
"""

import json
from pathlib import Path
from typing import Dict, List

from momentscan.cli.utils import BOLD, DIM, ITALIC, RESET


def run_bank(args):
    """Show MemoryBank contents."""
    path = Path(args.path)
    bank_files = _resolve_bank_files(path)

    if not bank_files:
        print(f"No memory_bank.json found at {path}")
        return

    for bank_path in bank_files:
        _print_bank(bank_path)


def _resolve_bank_files(path: Path) -> List[Path]:
    """Resolve path argument to list of memory_bank.json files.

    - .json file -> direct
    - directory -> glob identity/person_*/memory_bank.json
    """
    if path.is_file() and path.suffix == ".json":
        return [path]

    if path.is_dir():
        # Try identity/person_*/memory_bank.json
        found = sorted(path.glob("identity/person_*/memory_bank.json"))
        if found:
            return found
        # Try person_*/memory_bank.json (if already in identity/)
        found = sorted(path.glob("person_*/memory_bank.json"))
        if found:
            return found
        # Try direct memory_bank.json
        direct = path / "memory_bank.json"
        if direct.exists():
            return [direct]

    return []


def _print_bank(bank_path: Path) -> None:
    """Print a single memory_bank.json contents."""
    with open(bank_path, "r") as f:
        data = json.load(f)

    person_id = data.get("person_id", "?")
    nodes = data.get("nodes", [])
    k_max = data.get("_config", {}).get("k_max", "?")

    print()
    print(f"{BOLD}{'Bank':<10}{RESET}{bank_path}")
    print(f"          {DIM}person_id={person_id} · {len(nodes)} nodes · k_max={k_max}{RESET}")

    if not nodes:
        print(f"          {DIM}(empty){RESET}")
        return

    print()
    print(f"{BOLD}{'Nodes':<10}{RESET}")

    for node in nodes:
        node_id = node.get("node_id", "?")
        meta = node.get("meta_hist", {})
        hit_count = meta.get("hit_count", 0)
        q_best = meta.get("quality_best", 0.0)
        q_mean = meta.get("quality_mean", 0.0)
        rep_images = node.get("rep_images", [])

        print(f"  #{node_id}  {DIM}hits={hit_count} · q_best={q_best:.2f} · q_mean={q_mean:.2f} · {len(rep_images)} images{RESET}")

        # Yaw histogram
        yaw_bins = meta.get("yaw_bins", {})
        if yaw_bins:
            yaw_str = ", ".join(f"{k}={v}" for k, v in yaw_bins.items())
            print(f"      {DIM}yaw: {yaw_str}{RESET}")

        # Pitch histogram
        pitch_bins = meta.get("pitch_bins", {})
        if pitch_bins:
            pitch_str = ", ".join(f"{k}={v}" for k, v in pitch_bins.items())
            print(f"      {DIM}pitch: {pitch_str}{RESET}")

        # Expression histogram
        expr_bins = meta.get("expression_bins", {})
        if expr_bins:
            expr_str = ", ".join(f"{k}={v}" for k, v in expr_bins.items())
            print(f"      {DIM}expression: {expr_str}{RESET}")

        # Rep images
        if rep_images:
            for img_path in rep_images:
                img_name = Path(img_path).name
                print(f"      {DIM}{ITALIC}{img_name}{RESET}")

    print()
