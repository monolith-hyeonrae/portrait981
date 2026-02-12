"""Command-line interface for momentscan."""

import sys
import argparse
import logging


def _add_distributed_args(parser):
    """Add --distributed, --venv-* args to a parser."""
    parser.add_argument(
        "--distributed", action="store_true",
        help="Enable distributed processing with VenvWorker (requires zmq)"
    )
    parser.add_argument(
        "--venv-face", type=str, metavar="PATH",
        help="Path to venv for face analyzer (enables VENV isolation)"
    )
    parser.add_argument(
        "--venv-pose", type=str, metavar="PATH",
        help="Path to venv for pose analyzer (enables VENV isolation)"
    )
    parser.add_argument(
        "--venv-gesture", type=str, metavar="PATH",
        help="Path to venv for gesture analyzer (enables VENV isolation)"
    )


def _add_trace_args(parser):
    """Add --trace, --trace-output args to a parser."""
    parser.add_argument("--trace", choices=["off", "minimal", "normal", "verbose"], default="off")
    parser.add_argument("--trace-output", type=str, help="Output file for trace records (JSONL)")



def main():
    parser = argparse.ArgumentParser(
        description="MomentScan - Portrait highlight capture",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  momentscan debug video.mp4                  # All analyzers
  momentscan debug video.mp4 -e face          # Face only
  momentscan debug video.mp4 -e face,pose     # Face + Pose
  momentscan debug video.mp4 -e raw           # Raw video preview (no analysis)
  momentscan process video.mp4 -o ./clips     # Extract highlight clips
  momentscan process video.mp4 --distributed  # Distributed mode (venv workers)
""",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose logging"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # info command
    info_parser = subparsers.add_parser(
        "info",
        help="Show system info and available components",
        description="Display analyzers, backends, triggers, and pipeline structure.",
    )
    info_parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Show detailed info including device capabilities",
    )
    info_parser.add_argument(
        "--deps", action="store_true",
        help="Show analyzer dependency graph",
    )
    info_parser.add_argument(
        "--graph", nargs="?", const="ascii", choices=["ascii", "dot"],
        help="Show pipeline FlowGraph (ascii or dot format, default: ascii)",
    )
    info_parser.add_argument(
        "--steps", action="store_true",
        help="Show internal processing steps of each analyzer",
    )

    # debug command (unified)
    debug_parser = subparsers.add_parser(
        "debug",
        help="Debug analyzers with visualization",
        description="Run analyzers on video with real-time visualization.",
    )
    debug_parser.add_argument("path", help="Path to video file")
    debug_parser.add_argument("--fps", type=float, default=10.0, help="Analysis FPS (default: 10)")
    debug_parser.add_argument(
        "-e", "--analyzer",
        type=str,
        default="all",
        help="Analyzer(s) to run: face, pose, quality, gesture, all, raw (default: all). Use 'raw' for video-only preview without analysis. Comma-separated for multiple.",
    )
    debug_parser.add_argument("--no-window", action="store_true", help="Disable interactive window")
    debug_parser.add_argument("--output", "-o", type=str, help="Save debug video to file")
    debug_parser.add_argument("--device", type=str, default="cuda:0", help="Device for ML (default: cuda:0)")
    _add_trace_args(debug_parser)
    debug_parser.add_argument(
        "--profile", action="store_true",
        help="Show per-component timing information (detection, expression)"
    )
    _add_distributed_args(debug_parser)
    debug_parser.add_argument(
        "--roi", type=str, metavar="X1,Y1,X2,Y2",
        help="Face analysis ROI in normalized coords (0-1). Default: 0.1,0.1,0.9,0.9 (center 80%%)"
    )
    debug_parser.add_argument(
        "--backend", choices=["pathway", "simple"], default=None,
        help="Execution backend: 'pathway' (Pathway streaming) or 'simple' (sequential). "
             "Default: inline (same analyzers/fusion as pathway, smooth visualization)"
    )
    debug_parser.add_argument(
        "--batch-size", type=int, default=1, metavar="N",
        help="Batch size for GPU module processing (default: 1). "
             "When > 1, modules with BATCHING capability use process_batch() for optimization."
    )
    debug_parser.add_argument(
        "--report", type=str, metavar="PATH",
        help="Generate HTML debug report after session (e.g. --report report.html)"
    )
    debug_parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable verbose logging (show all third-party and internal messages)"
    )

    # process command
    proc_parser = subparsers.add_parser("process", help="Process video and extract highlight clips")
    proc_parser.add_argument("path", help="Path to video file")
    proc_parser.add_argument("--fps", type=int, default=10, help="Analysis FPS (default: 10)")
    proc_parser.add_argument("--output-dir", "-o", type=str, default="./clips", help="Output directory")
    proc_parser.add_argument("--report", type=str, help="Save processing report to JSON file")
    _add_trace_args(proc_parser)
    proc_parser.add_argument(
        "--backend", choices=["pathway", "simple"], default="simple",
        help="Execution backend: 'simple' (sequential, default) or 'pathway' (streaming)"
    )
    proc_parser.add_argument(
        "--profile", choices=["lite", "platform"], default=None,
        help="Execution profile: 'lite' (inline, no observability) or 'platform' (process isolation, observability)"
    )
    _add_distributed_args(proc_parser)
    proc_parser.add_argument(
        "--roi", type=str, metavar="X1,Y1,X2,Y2",
        help="Face analysis ROI in normalized coords (0-1). Default: 0.1,0.1,0.9,0.9 (center 80%%)"
    )
    proc_parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable verbose logging (show all third-party and internal messages)"
    )

    args = parser.parse_args()

    from momentscan.cli.utils import suppress_thirdparty_noise, configure_log_levels, StderrFilter

    if getattr(args, "verbose", False):
        logging.basicConfig(level=logging.DEBUG)
    else:
        suppress_thirdparty_noise()
        StderrFilter().install()
        logging.basicConfig(level=logging.INFO)
        configure_log_levels()

    from momentscan.cli import commands

    if args.command == "info":
        commands.run_info(args)

    elif args.command == "debug":
        commands.run_debug(args)

    elif args.command == "process":
        commands.run_process(args)

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
