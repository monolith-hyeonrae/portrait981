"""CLI for vpx-runner: ``vpx run`` and ``vpx list``."""

import argparse
import sys
import logging

logger = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vpx",
        description="Lightweight analyzer runner for vpx modules",
    )
    sub = parser.add_subparsers(dest="command")

    # vpx run
    run_p = sub.add_parser("run", help="Run analyzers on a source")
    run_p.add_argument(
        "analyzers",
        help="Comma-separated analyzer names (e.g. face.detect,face.expression)",
    )
    run_p.add_argument(
        "--input", "-i",
        required=True,
        help="Input source: file path or camera index (int)",
    )
    run_p.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Target FPS for analysis (skip frames to match)",
    )
    run_p.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Stop after N processed frames",
    )
    run_p.add_argument(
        "--viz",
        choices=["text", "live", "save"],
        default="text",
        help="Visualization mode (default: text)",
    )
    run_p.add_argument(
        "-o", "--output",
        default=None,
        help="Output path for --viz=save",
    )
    run_p.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output",
    )

    # vpx list
    list_p = sub.add_parser("list", help="List registered analyzers")
    list_p.add_argument(
        "--verbose",
        action="store_true",
        help="Show entry point details",
    )

    # vpx new
    new_p = sub.add_parser("new", help="Scaffold a new vpx module")
    new_p.add_argument("name", help="Module name in dot notation (e.g. face.landmark)")
    new_p.add_argument(
        "--internal",
        action="store_true",
        help="Create as app-internal module instead of vpx plugin",
    )
    new_p.add_argument(
        "--app",
        default="facemoment",
        help="Target app for --internal modules (default: facemoment)",
    )
    new_p.add_argument(
        "--depends",
        default="",
        help="Comma-separated dependency module names (e.g. face.detect)",
    )
    new_p.add_argument(
        "--no-backend",
        action="store_true",
        help="Skip backends/ directory generation",
    )
    new_p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print file list without creating anything",
    )

    return parser


def _cmd_list(args: argparse.Namespace) -> None:
    """Handle ``vpx list``."""
    from visualpath.plugin.discovery import discover_analyzers

    analyzers = discover_analyzers()
    if not analyzers:
        print("No analyzers registered.")
        return

    for name in sorted(analyzers.keys()):
        if args.verbose:
            ep = analyzers[name]
            print(f"  {name:20s}  {ep.value}")
        else:
            print(f"  {name}")


def _resolve_input(input_str: str):
    """Resolve --input to a source path or camera index."""
    try:
        return int(input_str)
    except ValueError:
        return input_str


def _cmd_run(args: argparse.Namespace) -> None:
    """Handle ``vpx run``."""
    from vpx.runner.runner import LiteRunner

    analyzer_names = [n.strip() for n in args.analyzers.split(",")]
    source = _resolve_input(args.input)

    if args.viz == "text":
        runner = LiteRunner(
            analyzer_names,
            on_observation=_text_observation_callback,
        )
        result = runner.run(
            source,
            fps=args.fps,
            max_frames=args.max_frames,
        )
        print(f"\nDone: {result.frame_count} frames, modules: {result.module_names}")

    elif args.viz in ("live", "save"):
        try:
            from vpx.viz import TextOverlay
        except ImportError:
            print(
                "vpx-viz is not installed. Install with:\n"
                "  pip install vpx-runner[viz]",
                file=sys.stderr,
            )
            sys.exit(1)

        if args.viz == "live":
            _run_live(analyzer_names, source, args)
        else:
            _run_save(analyzer_names, source, args)


def _text_observation_callback(name: str, obs) -> None:
    """Print observation summary to stdout."""
    signals = getattr(obs, "signals", {})
    sig_str = ", ".join(f"{k}={v}" for k, v in signals.items())
    frame_id = getattr(obs, "frame_id", "?")
    print(f"  [{name}] frame={frame_id} {sig_str}")


def _run_live(analyzer_names, source, args):
    """Run with live display visualization."""
    from vpx.viz import FrameDisplay
    from vpx.runner.runner import LiteRunner

    display = FrameDisplay(title="vpx")

    def on_frame(frame, observations):
        cont = display.update(frame, observations)
        if not cont:
            raise _StopIteration()

    runner = LiteRunner(analyzer_names, on_frame=on_frame)
    try:
        result = runner.run(source, fps=args.fps, max_frames=args.max_frames)
    except _StopIteration:
        pass
    finally:
        display.close()


def _run_save(analyzer_names, source, args):
    """Run with video save visualization."""
    from vpx.viz import VideoSaver
    from vpx.runner.runner import LiteRunner

    output_path = args.output
    if not output_path:
        print("--output / -o is required with --viz=save", file=sys.stderr)
        sys.exit(1)

    saver = None

    def on_frame(frame, observations):
        nonlocal saver
        if saver is None:
            # Lazy init on first frame
            import numpy as np
            data = frame if isinstance(frame, np.ndarray) else frame.data
            h, w = data.shape[:2]
            fps = getattr(source, "fps", 30.0) if not isinstance(source, (str, int)) else 30.0
            saver = VideoSaver(output_path, fps=fps, width=w, height=h)
        saver.update(frame, observations)

    runner = LiteRunner(analyzer_names, on_frame=on_frame)
    try:
        result = runner.run(source, fps=args.fps, max_frames=args.max_frames)
    finally:
        if saver:
            saver.close()
    print(f"Saved {result.frame_count} frames to {output_path}")


class _StopIteration(Exception):
    """Signal to stop the runner loop (e.g. ESC pressed)."""
    pass


def _cmd_new(args: argparse.Namespace) -> None:
    """Handle ``vpx new``."""
    from vpx.runner.scaffold import scaffold_plugin, scaffold_internal

    depends = [d.strip() for d in args.depends.split(",") if d.strip()]

    try:
        if args.internal:
            paths = scaffold_internal(
                args.name,
                depends=depends,
                dry_run=args.dry_run,
                app_name=args.app,
            )
        else:
            paths = scaffold_plugin(
                args.name,
                depends=depends,
                no_backend=args.no_backend,
                dry_run=args.dry_run,
            )
    except (ValueError, FileExistsError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    if args.dry_run:
        print("Files that would be created:")
    else:
        print("Created files:")
    for p in paths:
        print(f"  {p}")


def main():
    """Entry point for ``vpx`` CLI."""
    parser = _build_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    if getattr(args, "verbose", False):
        logging.basicConfig(level=logging.DEBUG)

    if args.command == "list":
        _cmd_list(args)
    elif args.command == "run":
        _cmd_run(args)
    elif args.command == "new":
        _cmd_new(args)


if __name__ == "__main__":
    main()
