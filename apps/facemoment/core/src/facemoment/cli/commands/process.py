"""Process command for facemoment CLI."""

import sys
import json
import time
from pathlib import Path
from datetime import datetime

from facemoment.cli.utils import (
    setup_observability,
    cleanup_observability,
    detect_distributed_mode,
    detect_ml_mode,
    probe_extractors,
)


def run_process(args):
    """Run video processing and clip extraction."""
    # Check if distributed mode is requested
    distributed = detect_distributed_mode(args)
    backend = getattr(args, 'backend', 'pathway')

    if distributed:
        _run_distributed(
            args,
            getattr(args, 'config', None),
            getattr(args, 'venv_face', None),
            getattr(args, 'venv_pose', None),
            getattr(args, 'venv_gesture', None),
            backend,
        )
    else:
        _run_library(args, backend)


def _run_distributed(args, config_path, venv_face, venv_pose, venv_gesture, backend="pathway"):
    """Run processing in distributed mode using PipelineOrchestrator."""
    from facemoment.pipeline import (
        PipelineOrchestrator,
        PipelineConfig,
        create_default_config,
    )
    from facemoment.pipeline.pathway_pipeline import PATHWAY_AVAILABLE

    # Setup observability
    trace_level = getattr(args, 'trace', 'off')
    trace_output = getattr(args, 'trace_output', None)
    hub, file_sink = setup_observability(trace_level, trace_output)

    output_dir = Path(args.output_dir)

    # Determine effective backend
    effective_backend = backend if PATHWAY_AVAILABLE or backend == "simple" else "simple"

    print(f"Processing: {args.path}")
    print(f"FPS: {args.fps}")
    print(f"Output: {output_dir}")
    print(f"Mode: DISTRIBUTED")
    print(f"Backend: {effective_backend}" + (" (pathway unavailable, using simple)" if backend == "pathway" and not PATHWAY_AVAILABLE else ""))
    print("-" * 50)

    # Load or create config
    if config_path:
        print(f"Loading config from: {config_path}")
        config = PipelineConfig.from_yaml(config_path)
        # Override output dir and fps from CLI if provided
        config.clip_output_dir = str(output_dir)
        config.fps = args.fps
        config.fusion.cooldown_sec = args.cooldown
    else:
        config = create_default_config(
            venv_face=venv_face,
            venv_pose=venv_pose,
            venv_gesture=venv_gesture,
            clip_output_dir=str(output_dir),
            fps=args.fps,
            cooldown_sec=args.cooldown,
            backend=effective_backend,
        )

    # Print extractor configuration
    print("Extractors:")
    for ext_config in config.extractors:
        isolation = ext_config.effective_isolation.name
        venv = ext_config.venv_path or "(current)"
        print(f"  {ext_config.name}: {isolation} [{venv}]")

    print(f"Fusion: {config.fusion.name} (cooldown={config.fusion.cooldown_sec}s)")
    print("-" * 50)

    # Create orchestrator
    orchestrator = PipelineOrchestrator.from_config(config)

    # Track metadata for each clip
    clip_metadata = []
    start_time = time.time()

    def on_trigger(trigger, result):
        event_time_sec = trigger.event_time_ns / 1e9 if trigger.event_time_ns else 0
        meta = {
            "trigger_id": len(clip_metadata) + 1,
            "reason": result.trigger_reason,
            "score": result.trigger_score,
            "timestamp_sec": event_time_sec,
            "metadata": result.metadata,
        }
        clip_metadata.append(meta)
        print(f"\n  TRIGGER #{meta['trigger_id']}: {result.trigger_reason} (score={result.trigger_score:.2f}, t={event_time_sec:.2f}s)")

    orchestrator.set_on_trigger(on_trigger)

    # Progress tracking
    last_progress = [0]
    def on_frame(frame):
        if frame.frame_id % 100 == 0 and frame.frame_id > last_progress[0]:
            last_progress[0] = frame.frame_id
            print(f"\r  Processing frame {frame.frame_id}...", end="", flush=True)

    orchestrator.set_on_frame(on_frame)

    # Process video
    print("Processing video...")
    try:
        clips = orchestrator.run(args.path, fps=args.fps)
    except Exception as e:
        print(f"\nError during processing: {e}")
        cleanup_observability(hub, file_sink)
        sys.exit(1)

    print()

    # Get stats
    stats = orchestrator.get_stats()
    elapsed = stats.processing_time_sec

    print("-" * 50)
    print(f"Processing complete in {elapsed:.1f}s")
    print(f"  Frames processed: {stats.frames_processed}")
    print(f"  Triggers fired: {stats.triggers_fired}")
    print(f"  Clips extracted: {stats.clips_extracted}")
    if stats.avg_frame_time_ms > 0:
        print(f"  Avg frame time: {stats.avg_frame_time_ms:.1f}ms")
    print()

    # Print worker stats
    if stats.worker_stats:
        print("Worker statistics:")
        for name, ws in stats.worker_stats.items():
            if ws["frames"] > 0:
                avg_ms = ws["total_ms"] / ws["frames"]
                print(f"  {name}: {ws['frames']} frames, avg {avg_ms:.1f}ms, errors: {ws['errors']}")

    print()

    # Save metadata for each clip
    for i, clip in enumerate(clips):
        if clip.success and clip.output_path:
            clip_path = Path(clip.output_path)
            meta_path = clip_path.with_suffix(".json")

            trigger_meta = clip_metadata[i] if i < len(clip_metadata) else {}

            full_meta = {
                "clip_id": clip_path.stem,
                "video_source": str(Path(args.path).resolve()),
                "created_at": datetime.now().isoformat(),
                "mode": "distributed",
                "trigger": trigger_meta,
                "clip": {
                    "output_path": str(clip_path),
                    "duration_sec": clip.duration_sec,
                    "success": clip.success,
                },
            }

            with open(meta_path, "w") as f:
                json.dump(full_meta, f, indent=2, default=str)

            print(f"  [{i+1}] {clip_path.name} ({clip.duration_sec:.2f}s)")
            print(f"      Reason: {trigger_meta.get('reason', 'unknown')}")
        else:
            print(f"  [{i+1}] FAILED: {clip.error}")

    # Save processing report if requested
    if args.report:
        report = {
            "video_source": str(Path(args.path).resolve()),
            "processed_at": datetime.now().isoformat(),
            "mode": "distributed",
            "settings": {
                "fps": args.fps,
                "cooldown_sec": args.cooldown,
                "extractors": [
                    {
                        "name": ext.name,
                        "isolation": ext.effective_isolation.name,
                        "venv_path": ext.venv_path,
                    }
                    for ext in config.extractors
                ],
            },
            "results": {
                "frames_processed": stats.frames_processed,
                "triggers_fired": stats.triggers_fired,
                "clips_extracted": stats.clips_extracted,
                "processing_time_sec": elapsed,
                "avg_frame_time_ms": stats.avg_frame_time_ms,
            },
            "worker_stats": stats.worker_stats,
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
        print()
        print(f"Report saved: {report_path}")

    cleanup_observability(hub, file_sink)


def _run_library(args, backend="pathway"):
    """Run processing in library mode using FacemomentPipeline.

    Uses the same pipeline as distributed mode for consistent behavior:
    - FaceClassifier auto-injection when face extractor is used
    - HighlightFusion main_only mode
    """
    from visualbase import VisualBase, FileSource, ClipResult
    from facemoment.pipeline.pathway_pipeline import FacemomentPipeline, PATHWAY_AVAILABLE

    # Setup observability
    trace_level = getattr(args, 'trace', 'off')
    trace_output = getattr(args, 'trace_output', None)
    hub, file_sink = setup_observability(trace_level, trace_output)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    use_ml = args.use_ml
    ml_mode = detect_ml_mode(args)

    # Determine effective backend
    effective_backend = backend if PATHWAY_AVAILABLE or backend == "simple" else "simple"

    print(f"Processing: {args.path}")
    print(f"FPS: {args.fps}")
    print(f"Output: {output_dir}")
    print(f"Mode: LIBRARY")
    print(f"Backend: {effective_backend}" + (" (pathway unavailable, using simple)" if backend == "pathway" and not PATHWAY_AVAILABLE else ""))
    print(f"ML backends: {ml_mode}")
    print("-" * 50)

    # Determine which extractors to use
    probed = probe_extractors(use_ml=use_ml)
    extractor_names = probed["names"]
    face_available = probed["face"]
    pose_available = probed["pose"]
    gesture_available = probed["gesture"]

    for name in extractor_names:
        if name == "dummy":
            print("  DummyExtractor: enabled (fallback)")
        else:
            print(f"  {name.capitalize()}Extractor: enabled")

    # Print FaceClassifier info (auto-injected by FacemomentPipeline)
    if face_available:
        print("  FaceClassifierExtractor: enabled (auto-injected)")

    # Set up fusion config
    fusion_config = {
        "cooldown_sec": args.cooldown,
        "head_turn_velocity_threshold": args.head_turn_threshold,
        "main_only": True,  # Use main face for triggers
    }

    if face_available:
        print(f"  HighlightFusion: enabled (cooldown={args.cooldown}s)")
    else:
        print(f"  DummyFusion: enabled (threshold={args.threshold})")

    # Show available triggers
    print()
    print("Available triggers:")
    if face_available:
        print(f"  - expression_spike: enabled")
        print(f"  - head_turn: enabled")
        print(f"  - camera_gaze: enabled")
        print(f"  - passenger_interaction: enabled")
    if pose_available:
        print(f"  - hand_wave: enabled")
    if gesture_available:
        print(f"  - gesture_vsign: enabled")
        print(f"  - gesture_thumbsup: enabled")
    if not face_available and not pose_available:
        print(f"  - dummy triggers only")

    print("-" * 50)

    # Create pipeline
    pipeline = FacemomentPipeline(
        extractors=extractor_names,
        fusion_config=fusion_config,
        auto_inject_classifier=face_available,
    )

    # Initialize visualbase for clip extraction
    vb = VisualBase(clip_output_dir=output_dir)
    vb.connect(FileSource(args.path))

    # Track metadata for each clip
    clip_metadata = []
    clips = []
    frames_processed = 0
    triggers_fired = 0
    start_time = time.time()

    def on_trigger(trigger):
        nonlocal triggers_fired
        triggers_fired += 1
        event_time_sec = trigger.event_time_ns / 1e9 if trigger.event_time_ns else 0

        # Get reason and score from trigger metadata
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
        print(f"\n  TRIGGER #{meta['trigger_id']}: {reason} (score={score:.2f}, t={event_time_sec:.2f}s)")

        # Extract clip
        clip_result = vb.trigger(trigger)
        clips.append(clip_result)

    # Progress tracking
    last_progress = [0]

    def frame_generator():
        nonlocal frames_processed
        for frame in vb.get_stream(fps=args.fps):
            frames_processed += 1
            if frame.frame_id % 100 == 0 and frame.frame_id > last_progress[0]:
                last_progress[0] = frame.frame_id
                print(f"\r  Processing frame {frame.frame_id}...", end="", flush=True)
            yield frame

    # Process video
    print("Processing video...")
    try:
        pipeline.run(frame_generator(), on_trigger=on_trigger)
    except Exception as e:
        print(f"\nError during processing: {e}")
        vb.disconnect()
        cleanup_observability(hub, file_sink)
        sys.exit(1)

    vb.disconnect()
    print()

    elapsed = time.time() - start_time
    clips_extracted = len([c for c in clips if c.success])

    print("-" * 50)
    print(f"Processing complete in {elapsed:.1f}s")
    print(f"  Frames processed: {frames_processed}")
    print(f"  Triggers fired: {triggers_fired}")
    print(f"  Clips extracted: {clips_extracted}")
    print()

    # Save metadata for each clip
    for i, clip in enumerate(clips):
        if clip.success and clip.output_path:
            clip_path = Path(clip.output_path)
            meta_path = clip_path.with_suffix(".json")

            trigger_meta = clip_metadata[i] if i < len(clip_metadata) else {}

            full_meta = {
                "clip_id": clip_path.stem,
                "video_source": str(Path(args.path).resolve()),
                "created_at": datetime.now().isoformat(),
                "mode": "library",
                "trigger": trigger_meta,
                "clip": {
                    "output_path": str(clip_path),
                    "duration_sec": clip.duration_sec,
                    "success": clip.success,
                },
            }

            with open(meta_path, "w") as f:
                json.dump(full_meta, f, indent=2, default=str)

            print(f"  [{i+1}] {clip_path.name} ({clip.duration_sec:.2f}s)")
            print(f"      Reason: {trigger_meta.get('reason', 'unknown')}")
        else:
            print(f"  [{i+1}] FAILED: {clip.error}")

    # Save processing report if requested
    if args.report:
        report = {
            "video_source": str(Path(args.path).resolve()),
            "processed_at": datetime.now().isoformat(),
            "mode": "library",
            "backend": pipeline.actual_backend or effective_backend,
            "settings": {
                "fps": args.fps,
                "cooldown_sec": args.cooldown,
                "ml_mode": ml_mode,
                "extractors": extractor_names,
            },
            "results": {
                "frames_processed": frames_processed,
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
        print()
        print(f"Report saved: {report_path}")

    cleanup_observability(hub, file_sink)
