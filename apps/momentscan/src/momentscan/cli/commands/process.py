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
    from visualbase import VisualBase, FileSource

    # Setup observability
    trace_level = getattr(args, 'trace', 'off')
    trace_output = getattr(args, 'trace_output', None)
    hub, file_sink = setup_observability(trace_level, trace_output)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    backend = getattr(args, 'backend', 'pathway')
    profile = getattr(args, 'profile', None)
    distributed = detect_distributed_mode(args)

    # Determine analyzer set
    analyzer_names = None  # ms.run() default

    # Build isolation config for distributed mode
    isolation_config = None
    if distributed:
        effective_names = analyzer_names or [
            "face.detect", "face.expression", "body.pose", "hand.gesture"
        ]
        app = MomentscanApp()
        resolved = app.configure_modules(effective_names)
        venv_paths = _collect_venv_paths(args, effective_names)
        isolation_config = build_distributed_config(
            resolved, venv_paths=venv_paths
        )

    # Print header
    mode = "distributed" if distributed else "library"
    _  = "          "  # 10-char indent

    print()
    print(f"{BOLD}{'Process':<10}{RESET}{args.path}")
    print(f"{_}{DIM}{args.fps} fps · cooldown {args.cooldown}s · {ITALIC}{mode}{RESET}")
    print(f"{_}{DIM}backend: {backend} · output: {output_dir}{RESET}")
    print()

    # Initialize visualbase for real-time clip extraction
    vb = VisualBase(clip_output_dir=output_dir)
    vb.connect(FileSource(args.path))

    # Track metadata for each clip
    clip_metadata = []
    clips = []
    triggers_fired = 0
    start_time = time.time()

    def on_trigger(trigger):
        nonlocal triggers_fired
        triggers_fired += 1
        event_time_sec = trigger.event_time_ns / 1e9 if trigger.event_time_ns else 0

        reason = trigger.metadata.get("reason", "highlight") if trigger.metadata else "highlight"
        score = trigger.metadata.get("score", 0.0) if trigger.metadata else 0.0

        meta = {
            "trigger_id": len(clip_metadata) + 1,
            "reason": reason,
            "score": score,
            "timestamp_sec": event_time_sec,
            "metadata": trigger.metadata,
        }
        clip_metadata.append(meta)
        print(f"\n  {BOLD}TRIGGER #{meta['trigger_id']}{RESET}  {reason}  {DIM}score={score:.2f} · t={event_time_sec:.2f}s{RESET}")

        # Extract clip via visualbase
        clip_result = vb.trigger(trigger)
        clips.append(clip_result)

    # Process video via unified path
    print(f"{DIM}Processing...{RESET}")
    try:
        result = ms.run(
            args.path,
            analyzers=analyzer_names,
            fps=args.fps,
            cooldown=args.cooldown,
            backend=backend,
            profile=profile,
            isolation=isolation_config,
            on_trigger=on_trigger,
        )
    except Exception as e:
        print(f"\nError during processing: {e}")
        vb.disconnect()
        cleanup_observability(hub, file_sink)
        sys.exit(1)

    vb.disconnect()
    print()

    elapsed = time.time() - start_time
    clips_extracted = len([c for c in clips if c.success])

    print(f"{BOLD}{'Summary':<10}{RESET}{DIM}{elapsed:.1f}s · {result.frame_count} frames · {triggers_fired} triggers · {clips_extracted} clips{RESET}")
    print(f"{_}{DIM}backend: {ITALIC}{result.actual_backend}{RESET}")
    _print_backend_stats(result.stats)
    print()

    # Save metadata for each clip
    if clips:
        print(f"{BOLD}{'Clips':<10}{RESET}")
    for i, clip in enumerate(clips):
        if clip.success and clip.output_path:
            clip_path = Path(clip.output_path)
            meta_path = clip_path.with_suffix(".json")

            trigger_meta = clip_metadata[i] if i < len(clip_metadata) else {}

            full_meta = {
                "clip_id": clip_path.stem,
                "video_source": str(Path(args.path).resolve()),
                "created_at": datetime.now().isoformat(),
                "mode": mode,
                "trigger": trigger_meta,
                "clip": {
                    "output_path": str(clip_path),
                    "duration_sec": clip.duration_sec,
                    "success": clip.success,
                },
            }

            with open(meta_path, "w") as f:
                json.dump(full_meta, f, indent=2, default=str)

            reason = trigger_meta.get('reason', 'unknown')
            print(f"  {clip_path.name}  {DIM}{clip.duration_sec:.2f}s · {ITALIC}{reason}{RESET}")
        else:
            print(f"  {DIM}FAILED: {clip.error}{RESET}")

    # Save processing report if requested
    if args.report:
        report = {
            "video_source": str(Path(args.path).resolve()),
            "processed_at": datetime.now().isoformat(),
            "mode": mode,
            "backend": result.actual_backend,
            "settings": {
                "fps": args.fps,
                "cooldown_sec": args.cooldown,
            },
            "results": {
                "frames_processed": result.frame_count,
                "triggers_fired": triggers_fired,
                "clips_extracted": clips_extracted,
                "processing_time_sec": elapsed,
            },
            "clips": [
                {
                    "clip_id": Path(c.output_path).stem if c.output_path else None,
                    "success": c.success,
                    "duration_sec": c.duration_sec,
                    "error": c.error,
                    "trigger": clip_metadata[i] if i < len(clip_metadata) else None,
                }
                for i, c in enumerate(clips)
            ],
        }

        report_path = Path(args.report)
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"{DIM}Report saved: {report_path}{RESET}")

    cleanup_observability(hub, file_sink)


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
