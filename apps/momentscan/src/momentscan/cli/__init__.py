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
    info_parser.add_argument(
        "--scoring", action="store_true",
        help="Show detailed highlight scoring pipeline",
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
        "--output-dir", type=str, metavar="DIR",
        help="Export batch analysis results (timeseries.csv, score_curve.png, peak frames) to directory"
    )
    debug_parser.add_argument(
        "--report", type=str, metavar="PATH",
        help="Generate HTML debug report after session (e.g. --report report.html)"
    )
    debug_parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable verbose logging (show all third-party and internal messages)"
    )
    debug_parser.add_argument(
        "--caption", action="store_true",
        help="Enable CoCa text captioner (shows generated captions in overlay)"
    )
    debug_parser.add_argument(
        "--collection", type=str, metavar="PATH",
        help="Path to collection/catalog directory (e.g. data/catalogs/portrait-v1). "
             "Loads signal profiles and pose/pivot definitions. "
             "Without this flag, uses built-in poses × AU-rule classification."
    )
    debug_parser.add_argument(
        "--member-id", type=str, metavar="ID",
        help="Member ID for cumulative bank storage. "
             "Default: video file stem."
    )
    debug_parser.add_argument(
        "--bind-model", type=str, metavar="PATH",
        help="Path to trained visualbind TreeStrategy model directory. "
             "When provided, uses the model for additional frame scoring "
             "alongside catalog scoring."
    )

    # process command
    proc_parser = subparsers.add_parser("process", help="Process video and extract highlight clips")
    proc_parser.add_argument("path", help="Path to video file")
    proc_parser.add_argument("--fps", type=int, default=10, help="Analysis FPS (default: 10)")
    proc_parser.add_argument("--output-dir", "-o", type=str, default=None, help="Output directory (default: ~/.portrait981/momentscan/output/{video_stem}/)")
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
        "--batch-size", type=int, default=1, metavar="N",
        help="Batch size for GPU module processing (default: 1). "
             "When > 1, modules with BATCHING capability use process_batch() for optimization."
    )
    proc_parser.add_argument(
        "--roi", type=str, metavar="X1,Y1,X2,Y2",
        help="Face analysis ROI in normalized coords (0-1). Default: 0.1,0.1,0.9,0.9 (center 80%%)"
    )
    proc_parser.add_argument(
        "--collection", type=str, metavar="PATH",
        help="Path to collection/catalog directory (e.g. data/catalogs/portrait-v1). "
             "Loads signal profiles and pose/pivot definitions. "
             "Without this flag, uses built-in poses × AU-rule classification."
    )
    proc_parser.add_argument(
        "--member-id", type=str, metavar="ID",
        help="Member ID for cumulative bank storage. "
             "Default: video file stem."
    )
    proc_parser.add_argument(
        "--bind-model", type=str, metavar="PATH",
        help="Path to trained visualbind TreeStrategy model directory. "
             "When provided, uses the model for additional frame scoring "
             "alongside catalog scoring."
    )
    proc_parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable verbose logging (show all third-party and internal messages)"
    )

    # collect command (alias for process)
    collect_parser = subparsers.add_parser(
        "collect",
        help="Collect portrait frames from video (alias for process)",
        description="Analyze video and collect portrait frames with pose×expression diversity. "
                    "This is an alias for 'process' with the same options.",
    )
    collect_parser.add_argument("path", help="Path to video file")
    collect_parser.add_argument("--fps", type=int, default=10, help="Analysis FPS (default: 10)")
    collect_parser.add_argument("--output-dir", "-o", type=str, default=None, help="Output directory (default: ~/.portrait981/momentscan/output/{video_stem}/)")
    collect_parser.add_argument("--report", type=str, help="Save processing report to JSON file")
    _add_trace_args(collect_parser)
    collect_parser.add_argument(
        "--backend", choices=["pathway", "simple"], default="simple",
        help="Execution backend: 'simple' (sequential, default) or 'pathway' (streaming)"
    )
    collect_parser.add_argument(
        "--profile", choices=["lite", "platform"], default=None,
        help="Execution profile: 'lite' (inline, no observability) or 'platform' (process isolation, observability)"
    )
    _add_distributed_args(collect_parser)
    collect_parser.add_argument(
        "--batch-size", type=int, default=1, metavar="N",
        help="Batch size for GPU module processing (default: 1)."
    )
    collect_parser.add_argument(
        "--roi", type=str, metavar="X1,Y1,X2,Y2",
        help="Face analysis ROI in normalized coords (0-1). Default: 0.1,0.1,0.9,0.9 (center 80%%)"
    )
    collect_parser.add_argument(
        "--collection", type=str, metavar="PATH",
        help="Path to collection/catalog directory (e.g. data/catalogs/portrait-v1). "
             "Loads signal profiles and pose/pivot definitions. "
             "Without this flag, uses built-in poses × AU-rule classification."
    )
    collect_parser.add_argument(
        "--member-id", type=str, metavar="ID",
        help="Member ID for cumulative bank storage. "
             "Default: video file stem."
    )
    collect_parser.add_argument(
        "--bind-model", type=str, metavar="PATH",
        help="Path to trained visualbind TreeStrategy model directory. "
             "When provided, uses the model for additional frame scoring "
             "alongside catalog scoring."
    )
    collect_parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable verbose logging (show all third-party and internal messages)"
    )

    # bank command (with subcommands)
    bank_parser = subparsers.add_parser(
        "bank",
        help="Browse memory banks",
        description="Browse and inspect memory banks stored under ~/.portrait981/momentbank/.",
    )
    bank_sub = bank_parser.add_subparsers(dest="bank_command")

    # bank list
    bank_list_parser = bank_sub.add_parser(
        "list",
        help="List all members with bank summary",
    )
    bank_list_parser.add_argument(
        "--sort", choices=["id", "nodes", "hits"], default="id",
        help="Sort order (default: id)",
    )

    # bank show <member_id_or_path>
    bank_show_parser = bank_sub.add_parser(
        "show",
        help="Show memory bank details for a member or path",
    )
    bank_show_parser.add_argument(
        "target",
        help="Member ID or path to memory_bank.json / output directory",
    )

    # bank get <member_id>
    bank_get_parser = bank_sub.add_parser(
        "get",
        help="Get frames from bank by pose/category",
    )
    bank_get_parser.add_argument(
        "member_id",
        help="Member ID",
    )
    bank_get_parser.add_argument(
        "--pose", type=str, default=None,
        help="Filter by pose (e.g. left30, frontal)",
    )
    bank_get_parser.add_argument(
        "--category", type=str, default=None,
        help="Filter by category (e.g. warm_smile, neutral)",
    )
    bank_get_parser.add_argument(
        "--top", type=int, default=1,
        help="Number of results (default: 1)",
    )
    bank_get_parser.add_argument(
        "--open", action="store_true", default=False,
        dest="open_image",
        help="Open the image with default viewer",
    )

    # catalog-build command
    catalog_parser = subparsers.add_parser(
        "catalog-build",
        help="Build signal profiles from reference image catalog",
        description="Analyze reference images to generate per-category signal profiles "
                    "for multi-signal matching.",
    )
    catalog_parser.add_argument(
        "path",
        help="Path to catalog directory (e.g. data/catalogs/portrait-v1)",
    )
    catalog_parser.add_argument(
        "--no-cache", action="store_true",
        help="Disable cache — force re-analysis of all reference images",
    )
    catalog_parser.add_argument(
        "--report", type=str, metavar="PATH", nargs="?", const="catalog_report.html",
        help="Generate HTML separation report (default: catalog_report.html)",
    )
    catalog_parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    from momentscan.cli.utils import suppress_thirdparty_noise, configure_log_levels, StderrFilter

    if getattr(args, "verbose", False):
        logging.basicConfig(level=logging.DEBUG, format="%(name)s %(message)s", stream=sys.stdout)
    else:
        suppress_thirdparty_noise()
        StderrFilter().install()
        logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
        configure_log_levels()

    from momentscan.cli import commands

    if args.command == "info":
        commands.run_info(args)

    elif args.command == "debug":
        commands.run_debug(args)

    elif args.command in ("process", "collect"):
        commands.run_process(args)

    elif args.command == "bank":
        commands.run_bank(args)

    elif args.command == "catalog-build":
        commands.run_catalog_build(args)

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
