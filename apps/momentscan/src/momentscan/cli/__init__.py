"""Command-line interface for momentscan."""

import sys
import argparse
import logging


def main():
    parser = argparse.ArgumentParser(
        description="MomentScan — portrait moment analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  momentscan run video.mp4                    # Analyze video
  momentscan run video.mp4 --debug            # With visualization
  momentscan run video.mp4 --report out.html  # Generate HTML report
  momentscan info                             # System info
""",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose logging"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # --- run command ---
    run_parser = subparsers.add_parser(
        "run",
        help="Analyze video",
        description="Run momentscan analysis on a video file.",
    )
    run_parser.add_argument("path", help="Path to video file")
    run_parser.add_argument("--fps", type=int, default=2, help="Analysis FPS (default: 2)")
    run_parser.add_argument("--quality-model", type=str, default="models/quality_v1.pkl",
                           help="Quality (shoot/cut) model path")
    run_parser.add_argument("--bind-model", type=str, default="models/bind_v12.pkl",
                           help="Expression model path")
    run_parser.add_argument("--pose-model", type=str, default="models/pose_v10.pkl",
                           help="Pose model path")
    run_parser.add_argument("--top-k", type=int, default=10, help="Top K frames to select")
    run_parser.add_argument("--debug", action="store_true",
                           help="Enable debug visualization (cv2 window)")
    run_parser.add_argument("--no-window", action="store_true",
                           help="Disable window (use with --output)")
    run_parser.add_argument("--output", "-o", type=str,
                           help="Save debug video to file")
    run_parser.add_argument("--report", type=str,
                           help="Generate HTML report to file")
    run_parser.add_argument("--ingest", type=str, metavar="MEMBER_ID",
                           help="Ingest SHOOT frames into personmemory for this member")
    run_parser.add_argument("-v", "--verbose", action="store_true")

    # --- info command ---
    info_parser = subparsers.add_parser(
        "info",
        help="Show system info and available components",
    )
    info_parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Show detailed info",
    )
    info_parser.add_argument("--deps", action="store_true", help="Show dependency graph")
    info_parser.add_argument(
        "--graph", nargs="?", const="ascii", choices=["ascii", "dot"],
        help="Show pipeline FlowGraph",
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

    if args.command == "info":
        from momentscan.cli import commands
        commands.run_info(args)

    elif args.command == "run":
        _run_scan(args)

    else:
        parser.print_help()
        sys.exit(1)


def _run_scan(args):
    """Execute momentscan analysis — CLI thin wrapper."""
    from pathlib import Path as _P

    use_debug = getattr(args, "debug", False)
    output_path = getattr(args, "output", None)
    no_window = getattr(args, "no_window", False)

    if use_debug or output_path:
        from momentscan.app.debug import MomentscanDebug
        show_window = not no_window
        if not show_window and not output_path:
            print("Error: --output is required when using --no-window")
            sys.exit(1)
        app = MomentscanDebug(
            quality_model=args.quality_model,
            expression_model=args.bind_model,
            pose_model=args.pose_model,
            show_window=show_window,
            output_path=output_path,
        )
    else:
        from momentscan.app import Momentscan
        app = Momentscan(
            quality_model=args.quality_model,
            expression_model=args.bind_model,
            pose_model=args.pose_model,
        )

    results = app.scan(args.path, fps=args.fps)
    selected = app.select_frames(results, top_k=args.top_k)

    # 결과 출력
    total = len(results)
    shoot = sum(1 for r in results if r.is_shoot)
    gated = sum(1 for r in results if not r.gate_passed and r.face_detected)

    print(f"\n{'='*50}")
    print(f"Video: {_P(args.path).name}")
    print(f"Frames: {total} | SHOOT: {shoot} | Gate rejected: {gated}")
    print(f"{'='*50}")

    if selected:
        print(f"\nSelected {len(selected)} frames:")
        for r in selected:
            print(f"  #{r.frame_idx:4d}  {r.expression:8s} ({r.expression_conf:.0%})  {r.pose:6s} ({r.pose_conf:.0%})")
    else:
        print("\nNo frames selected.")

    # Ingest
    ingest_member = getattr(args, "ingest", None)
    if ingest_member:
        shoot_frames = [r for r in results if r.is_shoot]
        try:
            from personmemory import PersonMemory
            mem = PersonMemory(ingest_member)
            stats = mem.ingest(workflow_id=_P(args.path).stem, frames=shoot_frames)
            p = mem.profile()
            print(f"\nIngested into '{ingest_member}': {stats['new_nodes']} new, {stats['updated_nodes']} updated nodes")
            print(f"  Total: {p.n_nodes} nodes, {p.n_total_frames} frames, {p.n_visits} visits")
        except ImportError:
            print("\npersonmemory not installed — skipping ingest")

    # Report
    report_path = getattr(args, "report", None)
    if report_path:
        from momentscan.app.report import export_report
        summary = app.summary(results)
        export_report(results, selected, report_path,
                       video_name=_P(args.path).stem, summary=summary)
        print(f"\nReport: {report_path}")

    app.shutdown()


if __name__ == "__main__":
    main()
