"""Bank command for momentscan CLI.

Browse and inspect memory banks stored under ~/.portrait981/momentbank/.

Subcommands:
    momentscan bank list                  # all members with summary
    momentscan bank show <member_id>      # single member bank detail
    momentscan bank show <path>           # legacy path-based access
"""

import json
from pathlib import Path
from typing import Dict, List

from momentscan.cli.utils import BOLD, DIM, ITALIC, RESET


def run_bank(args):
    """Route to bank subcommand."""
    bank_cmd = getattr(args, "bank_command", None)

    if bank_cmd == "list":
        _run_bank_list(args)
    elif bank_cmd == "show":
        _run_bank_show(args)
    elif bank_cmd == "get":
        _run_bank_get(args)
    else:
        # No subcommand → show help hint
        print(f"Usage: momentscan bank {{list,show}}")
        print()
        print(f"  {BOLD}list{RESET}              List all members with bank summary")
        print(f"  {BOLD}show{RESET} <target>     Show bank details (member ID or file path)")


def _run_bank_list(args):
    """List all members with summary stats."""
    try:
        from momentbank.paths import get_bank_base_dir, _shard
        from momentbank.persistence import load_bank
    except ImportError:
        print("momentbank is not installed")
        return

    base = get_bank_base_dir()
    if not base.exists():
        print(f"{DIM}No banks found (base dir does not exist: {base}){RESET}")
        return

    # Scan all shards
    entries = []
    for shard_dir in sorted(base.iterdir()):
        if not shard_dir.is_dir() or len(shard_dir.name) != 2:
            continue
        for member_dir in sorted(shard_dir.iterdir()):
            bank_file = member_dir / "memory_bank.json"
            if not bank_file.exists():
                continue
            try:
                with open(bank_file, "r") as f:
                    data = json.load(f)
                nodes = data.get("nodes", [])
                total_hits = sum(
                    n.get("meta_hist", {}).get("hit_count", 0) for n in nodes
                )
                q_best = max(
                    (n.get("meta_hist", {}).get("quality_best", 0.0) for n in nodes),
                    default=0.0,
                )
                size_kb = bank_file.stat().st_size / 1024
                entries.append({
                    "member_id": member_dir.name,
                    "nodes": len(nodes),
                    "hits": total_hits,
                    "q_best": q_best,
                    "size_kb": size_kb,
                    "shard": shard_dir.name,
                })
            except (json.JSONDecodeError, KeyError):
                entries.append({
                    "member_id": member_dir.name,
                    "nodes": 0,
                    "hits": 0,
                    "q_best": 0.0,
                    "size_kb": 0.0,
                    "shard": shard_dir.name,
                })

    if not entries:
        print(f"{DIM}No banks found in {base}{RESET}")
        return

    # Sort
    sort_key = getattr(args, "sort", "id")
    if sort_key == "nodes":
        entries.sort(key=lambda e: e["nodes"], reverse=True)
    elif sort_key == "hits":
        entries.sort(key=lambda e: e["hits"], reverse=True)
    else:
        entries.sort(key=lambda e: e["member_id"])

    print()
    print(f"{BOLD}{'Banks':<10}{RESET}{DIM}{len(entries)} members · {base}{RESET}")
    print()

    # Header
    print(f"  {'member_id':<24} {'nodes':>5} {'hits':>6} {'q_best':>6} {'size':>7}")
    print(f"  {'─' * 24} {'─' * 5} {'─' * 6} {'─' * 6} {'─' * 7}")

    for e in entries:
        print(
            f"  {e['member_id']:<24} "
            f"{e['nodes']:>5} "
            f"{e['hits']:>6} "
            f"{e['q_best']:>6.2f} "
            f"{e['size_kb']:>6.1f}K"
        )

    print()
    total_nodes = sum(e["nodes"] for e in entries)
    total_hits = sum(e["hits"] for e in entries)
    print(f"  {DIM}Total: {len(entries)} members · {total_nodes} nodes · {total_hits} hits{RESET}")
    print()


def _run_bank_show(args):
    """Show bank details for a member ID or path."""
    target = args.target
    target_path = Path(target)

    # Case 1: target is a file path or directory (legacy support)
    if target_path.exists():
        bank_files = _resolve_bank_files(target_path)
        if bank_files:
            for bank_path in bank_files:
                _print_bank(bank_path)
            return

    # Case 2: target is a member_id → resolve via paths module
    try:
        from momentbank.paths import get_bank_path
    except ImportError:
        print("momentbank is not installed")
        return

    bank_path = get_bank_path(target)
    if bank_path.exists():
        _print_bank(bank_path)
    else:
        print(f"No bank found for member '{target}' (checked {bank_path})")


def _run_bank_get(args):
    """Get frames by pose/category and optionally open them."""
    try:
        from momentbank.ingest import lookup_frames
    except ImportError:
        print("momentbank is not installed")
        return

    member_id = args.member_id
    pose = getattr(args, "pose", None)
    category = getattr(args, "category", None)
    top_k = getattr(args, "top", 1)
    open_image = getattr(args, "open_image", False)

    results = lookup_frames(member_id, pose=pose, category=category, top_k=top_k)

    if not results:
        filters = []
        if pose:
            filters.append(f"pose={pose}")
        if category:
            filters.append(f"category={category}")
        filter_str = f" ({', '.join(filters)})" if filters else ""
        print(f"No frames found for '{member_id}'{filter_str}")
        return

    print()
    print(f"{BOLD}{'Get':<10}{RESET}{member_id}  {DIM}{len(results)} frame(s){RESET}")
    print()

    for r in results:
        pose_name = r.get("pose_name", "")
        cat = r.get("category", "")
        q = r.get("quality", 0)
        cs = r.get("cell_score", 0)
        bbox = r.get("face_bbox")
        bbox_str = f"  bbox={_fmt_bbox(bbox)}" if bbox else ""
        path = r.get("path", "")

        print(f"  {pose_name:<12}{cat:<16}{DIM}q={q:.2f}  score={cs:.2f}{bbox_str}{RESET}")
        if path:
            print(f"  {DIM}{ITALIC}file://{path}{RESET}")

    print()

    # Open first image
    if open_image and results:
        path = results[0].get("path", "")
        if path and Path(path).exists():
            import subprocess
            import sys
            if sys.platform == "darwin":
                subprocess.Popen(["open", path])
            else:
                subprocess.Popen(["xdg-open", path])
            print(f"{DIM}Opened {Path(path).name}{RESET}")
        elif path:
            print(f"{DIM}File not found: {path}{RESET}")


def _resolve_bank_files(path: Path) -> List[Path]:
    """Resolve path argument to list of memory_bank.json files.

    - .json file -> direct
    - directory -> glob various patterns
    """
    if path.is_file() and path.suffix == ".json":
        return [path]

    if path.is_dir():
        # Try momentbank/{shard}/{member}/memory_bank.json (new sharded layout)
        found = sorted(path.glob("momentbank/??/*/memory_bank.json"))
        if found:
            return found
        # Try identity/person_*/memory_bank.json (legacy)
        found = sorted(path.glob("identity/person_*/memory_bank.json"))
        if found:
            return found
        # Try person_*/memory_bank.json
        found = sorted(path.glob("person_*/memory_bank.json"))
        if found:
            return found
        # Try {shard}/{member}/memory_bank.json (if pointing at base dir)
        found = sorted(path.glob("??/*/memory_bank.json"))
        if found:
            return found
        # Try direct memory_bank.json
        direct = path / "memory_bank.json"
        if direct.exists():
            return [direct]

    return []


def _fmt_bbox(bbox) -> str:
    """Format bbox list as compact string."""
    if not bbox or len(bbox) < 4:
        return ""
    return f"[{bbox[0]:.0f},{bbox[1]:.0f},{bbox[2]:.0f},{bbox[3]:.0f}]"


def _print_bank(bank_path: Path) -> None:
    """Print a single memory_bank.json contents."""
    with open(bank_path, "r") as f:
        data = json.load(f)

    person_id = data.get("person_id", "?")
    nodes = data.get("nodes", [])
    k_max = data.get("_config", {}).get("k_max", "?")

    # Try to infer member_id from path: .../momentbank/{shard}/{member_id}/memory_bank.json
    member_id = None
    parts = bank_path.parts
    if len(parts) >= 2:
        candidate_shard = parts[-3] if len(parts) >= 3 else ""
        if len(candidate_shard) == 2:
            member_id = parts[-2]

    print()
    title = member_id or f"person_{person_id}"
    print(f"{BOLD}{'Bank':<10}{RESET}{title}")
    print(f"          {DIM}{bank_path}{RESET}")
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

        # Rep images (clickable file:// links)
        if rep_images:
            for img_path in rep_images:
                p = Path(img_path)
                abs_path = p.resolve() if p.exists() else p
                exists_mark = "" if p.exists() else " (missing)"
                print(f"      {DIM}{ITALIC}file://{abs_path}{exists_mark}{RESET}")

    # Show saved frames from manifest
    frames_dir = bank_path.parent / "frames"
    manifest_path = frames_dir / "frames.json"
    if manifest_path.exists():
        try:
            manifest = json.load(open(manifest_path))
        except (json.JSONDecodeError, OSError):
            manifest = []
        if manifest:
            # Group by pose
            poses: Dict[str, List] = {}
            for entry in manifest:
                pose = entry.get("pose_name", "unknown")
                poses.setdefault(pose, []).append(entry)

            total = len(manifest)
            print()
            print(f"{BOLD}{'Frames':<10}{RESET}{DIM}{total} saved · {len(poses)} poses · {frames_dir}{RESET}")
            for pose_name, entries in sorted(poses.items()):
                for e in sorted(entries, key=lambda x: -x.get("cell_score", 0)):
                    cat = e.get("category", "")
                    q = e.get("quality", 0)
                    cs = e.get("cell_score", 0)
                    bbox = e.get("face_bbox")
                    bbox_str = f" bbox={_fmt_bbox(bbox)}" if bbox else ""
                    fpath = frames_dir / e["file"]
                    exists = fpath.exists()
                    mark = "" if exists else " (missing)"
                    print(f"  {DIM}{pose_name:<12}{cat:<16}q={q:.2f}  score={cs:.2f}{bbox_str}{mark}{RESET}")
                    if exists:
                        print(f"      {DIM}{ITALIC}file://{fpath.resolve()}{RESET}")
    elif frames_dir.is_dir():
        frame_files = sorted(f for f in frames_dir.iterdir() if f.suffix == ".jpg")
        if frame_files:
            print()
            print(f"{BOLD}{'Frames':<10}{RESET}{DIM}{len(frame_files)} saved · {frames_dir} (no manifest){RESET}")
            for f in frame_files:
                size_kb = f.stat().st_size / 1024
                print(f"  {DIM}{f.name}  ({size_kb:.0f}K){RESET}")

    print()
