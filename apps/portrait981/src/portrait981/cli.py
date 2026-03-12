"""portrait981 CLI — p981 command."""

from __future__ import annotations

import argparse
import logging
import signal
import sys
from pathlib import Path
from typing import List, Optional, Sequence

from portrait981.pipeline import Portrait981Pipeline
from portrait981.progress import BatchProgress
from portrait981.types import JobSpec, JobStatus, PipelineConfig, StepEvent


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    """Add common arguments shared across subcommands."""
    parser.add_argument("--collection", default=None, help="Collection/catalog path")
    parser.add_argument("--comfy-url", default="http://127.0.0.1:8188", help="ComfyUI URL")
    parser.add_argument("--api-key", default=None, help="ComfyUI API key")
    parser.add_argument("--output", default=None, help="Output directory")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="p981", description="portrait981 pipeline CLI")
    sub = parser.add_subparsers(dest="command")

    # p981 run
    run_p = sub.add_parser("run", help="Run full E2E pipeline for a single video")
    run_p.add_argument("video", help="Path to video file")
    run_p.add_argument("--member-id", required=True, help="Member identifier")
    run_p.add_argument("--pose", default=None, help="Pose filter for lookup")
    run_p.add_argument("--category", default=None, help="Category filter for lookup")
    run_p.add_argument("--prompt", default="", help="Style prompt")
    run_p.add_argument("--workflow", default="default", help="Workflow template")
    run_p.add_argument("--top-k", type=int, default=3, help="Max reference images")
    run_p.add_argument("--scan-only", action="store_true", help="Skip generation")
    _add_common_args(run_p)

    # p981 batch
    batch_p = sub.add_parser("batch", help="Process multiple videos")
    batch_p.add_argument("directory", help="Directory containing videos")
    batch_p.add_argument(
        "--member-id-from",
        choices=["filename", "parent"],
        default="filename",
        help="Derive member_id from filename stem or parent dir",
    )
    batch_p.add_argument("--workers", type=int, default=1, help="Scan workers")
    batch_p.add_argument("--scan-only", action="store_true", help="Skip generation")
    _add_common_args(batch_p)

    # p981 scan
    scan_p = sub.add_parser("scan", help="Scan only (no generation)")
    scan_p.add_argument("video", help="Path to video file")
    scan_p.add_argument("--member-id", required=True, help="Member identifier")
    _add_common_args(scan_p)

    # p981 generate
    gen_p = sub.add_parser("generate", help="Generate only (from existing bank)")
    gen_p.add_argument("member_id", help="Member identifier")
    gen_p.add_argument("--pose", default=None, help="Pose filter")
    gen_p.add_argument("--category", default=None, help="Category filter")
    gen_p.add_argument("--prompt", default="", help="Style prompt")
    gen_p.add_argument("--workflow", default="default", help="Workflow template")
    gen_p.add_argument("--top-k", type=int, default=3, help="Max reference images")
    _add_common_args(gen_p)

    # p981 status
    status_p = sub.add_parser("status", help="Query bank status for a member")
    status_p.add_argument("member_id", help="Member identifier")
    status_p.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    return parser


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def _discover_videos(directory: str) -> List[Path]:
    """Find video files in directory."""
    d = Path(directory)
    if not d.is_dir():
        print(f"Error: {directory} is not a directory", file=sys.stderr)
        sys.exit(1)
    return sorted(p for p in d.iterdir() if p.suffix.lower() in VIDEO_EXTENSIONS)


def _derive_member_id(video_path: Path, mode: str) -> str:
    if mode == "parent":
        return video_path.parent.name
    return video_path.stem


# -- Step progress callback for CLI --

_STEP_ICONS = {
    "started": "...",
    "completed": "OK",
    "failed": "FAIL",
    "skipped": "SKIP",
}


_last_progress_len = [0]  # mutable for closure


def _console_progress(event: StepEvent) -> None:
    """Print step progress to stderr. For single-run mode."""
    if event.status == "progress":
        # Inline update: overwrite same line
        fps_info = ""
        if event.elapsed_sec > 0.5:
            fps_info = f"  {event.frame_id / event.elapsed_sec:.1f} fps"
        msg = f"\r  [>>] scan  frame {event.frame_id}  ({event.elapsed_sec:.0f}s){fps_info}"
        _last_progress_len[0] = len(msg)
        print(msg, end="", file=sys.stderr, flush=True)
        return

    # Clear progress line if any
    if _last_progress_len[0] > 0:
        print("\r" + " " * _last_progress_len[0] + "\r", end="", file=sys.stderr)
        _last_progress_len[0] = 0

    icon = _STEP_ICONS.get(event.status, "??")
    elapsed = f" ({event.elapsed_sec:.1f}s)" if event.elapsed_sec > 0 else ""
    detail = f" -- {event.detail}" if event.detail else ""
    print(f"  [{icon}] {event.step}{elapsed}{detail}", file=sys.stderr)


# -- Handlers --

def _handle_run(args: argparse.Namespace) -> None:
    config = PipelineConfig(
        comfy_urls=[args.comfy_url],
        api_key=args.api_key,
        default_collection_path=args.collection,
    )
    job = JobSpec(
        video_path=args.video,
        member_id=args.member_id,
        pose=args.pose,
        category=args.category,
        prompt=args.prompt,
        workflow=args.workflow,
        collection_path=args.collection,
        output_dir=args.output,
        top_k=args.top_k,
        scan_only=args.scan_only,
    )
    pipeline = Portrait981Pipeline(config, on_step=_console_progress)
    try:
        result = pipeline.run_one(job)
        _print_result(result)
    finally:
        pipeline.shutdown()


def _handle_batch(args: argparse.Namespace) -> None:
    videos = _discover_videos(args.directory)
    if not videos:
        print("No video files found.", file=sys.stderr)
        sys.exit(1)

    config = PipelineConfig(
        max_scan_workers=args.workers,
        comfy_urls=[args.comfy_url],
        api_key=args.api_key,
        default_collection_path=args.collection,
    )
    jobs = [
        JobSpec(
            video_path=str(v),
            member_id=_derive_member_id(v, args.member_id_from),
            collection_path=args.collection,
            output_dir=args.output,
            scan_only=args.scan_only,
        )
        for v in videos
    ]
    progress = BatchProgress(total=len(jobs))
    pipeline = Portrait981Pipeline(config, on_step=progress.on_step)

    # SIGINT → interrupt pipeline (stop remaining jobs, don't kill current scan)
    prev_handler = signal.getsignal(signal.SIGINT)

    def _sigint_handler(signum, frame):
        pipeline.interrupt()
        # Restore previous handler so a second Ctrl+C exits immediately
        signal.signal(signal.SIGINT, prev_handler)

    signal.signal(signal.SIGINT, _sigint_handler)
    try:
        with progress:
            results = pipeline.run_batch(jobs)

        # Summary
        done = sum(1 for r in results if r.status == JobStatus.DONE)
        partial = sum(1 for r in results if r.status == JobStatus.PARTIAL)
        failed = sum(1 for r in results if r.status == JobStatus.FAILED)
        print(f"\nBatch complete: {done} done, {partial} partial, {failed} failed",
              file=sys.stderr)

        for r in results:
            _print_result(r)
    finally:
        signal.signal(signal.SIGINT, prev_handler)
        pipeline.shutdown()


def _handle_scan(args: argparse.Namespace) -> None:
    config = PipelineConfig(
        comfy_urls=[args.comfy_url],
        api_key=args.api_key,
    )
    job = JobSpec(
        video_path=args.video,
        member_id=args.member_id,
        collection_path=args.collection,
        output_dir=args.output,
        scan_only=True,
    )
    pipeline = Portrait981Pipeline(config, on_step=_console_progress)
    try:
        result = pipeline.run_one(job)
        _print_result(result)
    finally:
        pipeline.shutdown()


def _handle_generate(args: argparse.Namespace) -> None:
    config = PipelineConfig(
        comfy_urls=[args.comfy_url],
        api_key=args.api_key,
    )
    job = JobSpec(
        member_id=args.member_id,
        pose=args.pose,
        category=args.category,
        prompt=args.prompt,
        workflow=args.workflow,
        top_k=args.top_k,
        generate_only=True,
    )
    pipeline = Portrait981Pipeline(config, on_step=_console_progress)
    try:
        result = pipeline.run_one(job)
        _print_result(result)
    finally:
        pipeline.shutdown()


def _handle_status(args: argparse.Namespace) -> None:
    from momentbank.ingest import lookup_frames

    frames = lookup_frames(args.member_id)
    if not frames:
        print(f"No frames found for member '{args.member_id}'")
        return

    # Summarize by pose x category
    coverage: dict[str, dict[str, int]] = {}
    for f in frames:
        pose = f.get("pose_name", "unknown")
        cat = f.get("category", "unknown")
        coverage.setdefault(pose, {})
        coverage[pose][cat] = coverage[pose].get(cat, 0) + 1

    print(f"Member: {args.member_id}  ({len(frames)} frames)")
    print(f"{'Pose':<15} {'Category':<20} {'Count':>5}")
    print("-" * 42)
    for pose in sorted(coverage):
        for cat in sorted(coverage[pose]):
            print(f"{pose:<15} {cat:<20} {coverage[pose][cat]:>5}")


def _print_result(result) -> None:
    """Print a job result summary."""
    status = result.status.value.upper()
    print(f"\n[{status}] member={result.job.member_id}")

    if result.scan_result:
        sr = result.scan_result
        highlights = getattr(sr, "highlights", [])
        print(f"  Scan: {getattr(sr, 'frame_count', '?')} frames, {len(highlights)} highlights")

    if result.ref_count:
        print(f"  Refs: {result.ref_count} reference images")

    if result.generation_result:
        gr = result.generation_result
        if getattr(gr, "success", False):
            print(f"  Generated: {getattr(gr, 'output_paths', [])}")
        else:
            print(f"  Generation failed: {getattr(gr, 'error', 'unknown')}")

    if result.error:
        print(f"  Error: {result.error}")

    t = result.timing
    parts = []
    if t.scan_sec > 0:
        parts.append(f"scan={t.scan_sec:.1f}s")
    if t.lookup_sec > 0:
        parts.append(f"lookup={t.lookup_sec:.1f}s")
    if t.generate_sec > 0:
        parts.append(f"gen={t.generate_sec:.1f}s")
    parts.append(f"total={t.total_sec:.1f}s")
    print(f"  Timing: {' '.join(parts)}")

    # Retry hint for partial failures
    if result.status == JobStatus.PARTIAL:
        mid = result.job.member_id
        print(f"\n  Scan data preserved. Retry generation with:")
        print(f"    p981 generate {mid}")


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if getattr(args, "verbose", False):
        logging.basicConfig(level=logging.DEBUG)

    handlers = {
        "run": _handle_run,
        "batch": _handle_batch,
        "scan": _handle_scan,
        "generate": _handle_generate,
        "status": _handle_status,
    }

    handler = handlers.get(args.command)
    if handler is None:
        parser.print_help()
        sys.exit(1)
    handler(args)


if __name__ == "__main__":
    main()
