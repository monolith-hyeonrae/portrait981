"""Process command for momentscan CLI.

Uses the unified ``ms.run()`` execution path for both normal and
distributed modes. Distributed mode builds an IsolationConfig and
passes it to ``ms.run(isolation=...)``.
"""

import sys
import json
import time
from pathlib import Path
from datetime import datetime

from momentscan.cli.utils import (
    setup_observability,
    cleanup_observability,
    detect_distributed_mode,
    BOLD, DIM, ITALIC, RESET,
)


def run_process(args):
    """Run video processing and clip extraction.

    All modes use the unified execution path:
        ms.run() -> vp.run() -> Backend.execute()

    Distributed mode (--distributed, --venv-*) builds IsolationConfig
    and passes it to ms.run(isolation=...), which auto-switches to
    the worker backend.
    """
    import momentscan as ms
    from momentscan.main import MomentscanApp
    from visualpath.core.compat import build_distributed_config

    # Setup observability
    trace_level = getattr(args, 'trace', 'off')
    trace_output = getattr(args, 'trace_output', None)
    hub, file_sink = setup_observability(trace_level, trace_output)

    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
    else:
        from momentscan.paths import get_output_dir
        output_dir = get_output_dir(args.path)
    output_dir.mkdir(parents=True, exist_ok=True)

    backend = getattr(args, 'backend', 'simple')
    profile = getattr(args, 'profile', None)
    distributed = detect_distributed_mode(args)

    # Determine analyzer set
    analyzer_names = None  # ms.run() default

    # Build isolation config for distributed mode
    isolation_config = None
    if distributed:
        effective_names = analyzer_names or [
            "face.detect", "face.expression", "body.pose", "hand.gesture",
            "vision.embed", "frame.quality", "frame.scoring",
        ]
        app = MomentscanApp()
        resolved = app.configure_modules(effective_names)
        venv_paths = _collect_venv_paths(args, effective_names)
        isolation_config = build_distributed_config(
            resolved, venv_paths=venv_paths
        )

    batch_size = getattr(args, 'batch_size', 1)

    # Print header
    mode = "distributed" if distributed else "library"
    _  = "          "  # 10-char indent

    print()
    print(f"{BOLD}{'Process':<10}{RESET}{args.path}")
    batch_label = f" · batch: {batch_size}" if batch_size > 1 else ""
    print(f"{_}{DIM}{args.fps} fps · {ITALIC}{mode}{RESET}")
    print(f"{_}{DIM}backend: {backend}{batch_label} · output: {output_dir}{RESET}")
    print()

    start_time = time.time()

    # Process video via unified path (batch highlight)
    print(f"{DIM}Processing...{RESET}")
    try:
        result = ms.run(
            args.path,
            analyzers=analyzer_names,
            output_dir=str(output_dir),
            fps=args.fps,
            batch_size=batch_size,
            backend=backend,
            profile=profile,
            isolation=isolation_config,
        )
    except Exception as e:
        print(f"\nError during processing: {e}")
        cleanup_observability(hub, file_sink)
        sys.exit(1)

    print()

    elapsed = time.time() - start_time
    n_highlights = len(result.highlights)

    print(f"{BOLD}{'Summary':<10}{RESET}{DIM}{elapsed:.1f}s · {result.frame_count} frames · {n_highlights} highlights{RESET}")
    print(f"{_}{DIM}backend: {ITALIC}{result.actual_backend}{RESET}")
    _print_backend_stats(result.stats)
    print()

    # Print highlight windows
    if result.highlights:
        print(f"{BOLD}{'Highlights':<10}{RESET}")
        for w in result.highlights:
            reason_str = ", ".join(f"{k}={v:.1f}" for k, v in w.reason.items()) if w.reason else ""
            n_frames = len(w.selected_frames)
            print(f"  #{w.window_id}  {DIM}{w.start_ms/1000:.1f}s-{w.end_ms/1000:.1f}s · score={w.score:.2f} · {n_frames} frames · {ITALIC}{reason_str}{RESET}")
        print()

    # Print exported file locations
    _print_exports(output_dir)

    # Save processing report if requested
    if args.report:
        report = {
            "video_source": str(Path(args.path).resolve()),
            "processed_at": datetime.now().isoformat(),
            "mode": mode,
            "backend": result.actual_backend,
            "settings": {
                "fps": args.fps,
            },
            "results": {
                "frames_processed": result.frame_count,
                "highlights_found": n_highlights,
                "processing_time_sec": elapsed,
            },
            "highlights": [
                {
                    "window_id": w.window_id,
                    "start_ms": w.start_ms,
                    "end_ms": w.end_ms,
                    "peak_ms": w.peak_ms,
                    "score": w.score,
                    "reason": w.reason,
                    "selected_frames": w.selected_frames,
                }
                for w in result.highlights
            ],
        }

        report_path = Path(args.report)
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"{DIM}Report saved: {report_path}{RESET}")

    cleanup_observability(hub, file_sink)


def _print_exports(output_dir: Path):
    """Print exported file locations under output_dir/highlight/."""
    highlight_dir = output_dir / "highlight"
    if not highlight_dir.exists():
        return

    print(f"{BOLD}{'Exports':<10}{RESET}{highlight_dir}")
    _  = "          "

    report_path = None
    for child in sorted(highlight_dir.iterdir()):
        if child.is_dir():
            files = sorted(child.iterdir())
            if files:
                print(f"  {DIM}{child.name}/{RESET}  {DIM}({len(files)} files){RESET}")
                for f in files:
                    print(f"    {DIM}{f.name}{RESET}")
        else:
            size_kb = child.stat().st_size / 1024
            print(f"  {DIM}{child.name}{RESET}  {DIM}({size_kb:.1f} KB){RESET}")
            if child.name == "report.html":
                report_path = child.resolve()

    if report_path:
        print(f"\n  {BOLD}Report{RESET}  file://{report_path}")
    print()


def _collect_venv_paths(args, analyzer_names):
    """Collect venv paths from CLI args.

    Maps --venv-face, --venv-pose, --venv-gesture to analyzer names.

    Returns:
        Dict mapping analyzer name to venv path (only non-None entries).
    """
    venv_face = getattr(args, 'venv_face', None)
    venv_pose = getattr(args, 'venv_pose', None)
    venv_gesture = getattr(args, 'venv_gesture', None)

    venv_map = {
        'face.detect': venv_face,
        'face.expression': venv_face,
        'body.pose': venv_pose,
        'hand.gesture': venv_gesture,
    }

    return {name: path for name, path in venv_map.items()
            if path and name in analyzer_names}


def _print_backend_stats(stats):
    """Print backend execution statistics."""
    if not stats:
        return

    parts = []
    throughput = stats.get("throughput_fps", 0)
    if throughput > 0:
        parts.append(f"{throughput:.1f} fps")

    pipeline_sec = stats.get("pipeline_duration_sec", 0)
    if pipeline_sec > 0:
        parts.append(f"pipeline {pipeline_sec:.1f}s")

    if parts:
        print(f"          {DIM}{' · '.join(parts)}{RESET}")

    per_analyzer = stats.get("per_analyzer_time_ms", {})
    if per_analyzer:
        avg_ms = stats.get("avg_analysis_ms", 0)
        p95_ms = stats.get("p95_analysis_ms", 0)
        failed = stats.get("analyses_failed", 0)
        print()
        print(f"{BOLD}{'Timing':<10}{RESET}")
        for name, ema_ms in sorted(per_analyzer.items()):
            print(f"  {name:<20}{DIM}{ema_ms:6.1f}ms {ITALIC}avg{RESET}")
        if avg_ms > 0:
            print(f"  {'(total)':<20}{DIM}{avg_ms:6.1f}ms avg · {p95_ms:.1f}ms p95{RESET}")
        if failed > 0:
            print(f"  {DIM}{failed} failed{RESET}")
