"""Debug command for momentscan CLI.

Uses the unified ``ms.run(on_frame=handler)`` execution path so that
debug and process share the same pipeline. Visualization is provided
via :class:`DebugFrameHandler` as an ``on_frame`` callback.
"""

import os
import sys
from typing import Dict, List, Optional

from momentscan.cli.utils import (
    create_video_stream,
    create_video_writer,
    setup_observability,
    cleanup_observability,
    BOLD, DIM, ITALIC, RESET,
)


def run_debug(args):
    """Run debug session with selected analyzers.

    Uses visualpath building blocks (FlowGraph, PathNode) directly instead
    of ms.run() so we can initialize workers before printing the header,
    giving us per-analyzer PIDs for the status table.

    Supports:
    - Single analyzer: -e face.detect, -e body.pose, -e frame.quality, -e hand.gesture
    - Multiple analyzers: -e face.detect,body.pose or -e all (default)
    - Raw video preview: -e raw (no analysis, verify video input)
    - Distributed mode: --distributed (uses WorkerModule for process isolation)
    - Backend selection: --backend pathway (default) or simple
    """
    from momentscan.main import MomentscanApp
    from momentscan.cli.debug_handler import DebugFrameHandler
    from momentscan.cli.utils import detect_distributed_mode
    from visualpath.backends.simple.executor import GraphExecutor
    from visualpath.flow.graph import FlowGraph
    from visualpath.flow.nodes.source import SourceNode
    from visualpath.flow.nodes.path import PathNode
    from visualpath.core.compat import build_conflict_isolation, build_distributed_config

    # Parse analyzer selection
    analyzer_arg = getattr(args, 'analyzer', 'all')
    selected = _parse_analyzer_arg(analyzer_arg)

    show_window = not getattr(args, 'no_window', False)
    if not show_window and not args.output:
        print("Error: --output is required when using --no-window")
        sys.exit(1)

    # Raw video preview mode (no analysis)
    if 'raw' in selected:
        _run_raw_preview(args, show_window)
        return

    # Resolve selected analyzers to names
    analyzer_names = _resolve_selected_to_names(selected, args)

    # Determine backend and distributed isolation
    backend = getattr(args, 'backend', None) or 'simple'
    distributed = detect_distributed_mode(args)
    isolation_config = None

    # Resolve modules via MomentscanApp
    app = MomentscanApp(cooldown=2.0)
    resolved = app.configure_modules(analyzer_names)

    if distributed:
        venv_paths = _collect_venv_paths(args, analyzer_names)
        isolation_config = build_distributed_config(resolved, venv_paths=venv_paths)
        backend = 'worker'
    else:
        # Auto-detect resource group conflicts (same as vp.run())
        isolation_config = build_conflict_isolation(resolved)
        if isolation_config is not None:
            backend = 'worker'

    # ROI
    roi = _parse_roi(getattr(args, 'roi', None)) or (0.3, 0.1, 0.7, 0.6)

    # Observability
    trace_level = getattr(args, 'trace', 'off')
    trace_output = getattr(args, 'trace_output', None)
    hub, file_sink = setup_observability(trace_level, trace_output)

    # --- Build pipeline (same steps as vp.run()) ---
    graph = FlowGraph(entry_node="source")
    graph.add_node(SourceNode(name="source"))
    graph.add_node(PathNode(
        name="pipeline",
        modules=resolved,
        parallel=True,
        isolation=isolation_config,
    ))
    graph.add_edge("source", "pipeline")

    # For distributed, wrap isolated modules in WorkerModule
    if isolation_config is not None:
        from visualpath.backends.worker import WorkerBackend
        graph = WorkerBackend()._wrap_isolated_modules(graph)

    # Initialize graph (starts worker processes → PIDs available)
    batch_size = getattr(args, 'batch_size', 1)
    executor = GraphExecutor(graph, batch_size=batch_size)
    executor.initialize()

    # --- Print header (after init so we have PIDs) ---
    # Get video metadata
    try:
        _vb, _source, _ = create_video_stream(args.path, fps=args.fps)
        source_fps = _source.fps
        source_frames = _source.frame_count
        _vb.disconnect()
    except IOError:
        source_fps = 0
        source_frames = 0

    # --- Header ---
    _  = "          "  # 10-char indent for sub-lines

    analyzer_label = (f"{', '.join(selected)}" if len(selected) > 1
                      else selected[0])
    roi_pct = f"{int(roi[0]*100)}%-{int(roi[2]*100)}% x {int(roi[1]*100)}%-{int(roi[3]*100)}%"

    print()
    print(f"{BOLD}{'Debug':<10}{RESET}{args.path}")
    print(f"{_}{DIM}{source_frames} frames · {source_fps:.1f} fps · {ITALIC}{analyzer_label}{RESET}")
    batch_label = f" · batch: {batch_size}" if batch_size > 1 else ""
    print(f"{_}{DIM}backend: {backend} · window: {'on' if show_window else 'off'} · ROI: {roi_pct}{batch_label}{RESET}")
    print()

    # --- Analyzer table ---
    module_pids = _get_module_pids(graph)
    origins = _get_analyzer_origins()
    main_pid = os.getpid()

    print(f"{BOLD}{'Analyzers':<10}{RESET}{DIM}pid={main_pid}{RESET}")
    all_rows = []
    for name in analyzer_names:
        origin = origins.get(name, "core")
        pid = module_pids.get(name)
        if pid is not None and pid != main_pid:
            iso_label = "process"
            if isolation_config is not None:
                iso_label = isolation_config.get_level(name).name.lower()
            all_rows.append((name, origin, iso_label, str(pid)))
        else:
            all_rows.append((name, origin, "inline", str(main_pid)))
    # Auto-injected classifier
    if 'face.detect' in analyzer_names and 'face.classify' not in analyzer_names:
        origin = origins.get('face.classify', 'core')
        cls_pid = module_pids.get('face.classify', main_pid)
        all_rows.append(('face.classify', origin, f'{ITALIC}auto-injected{RESET}{DIM}', str(cls_pid)))
    # Fusion
    all_rows.append(('highlight', 'core', f'{ITALIC}HighlightFusion{RESET}{DIM}', 'main'))

    for name, origin, iso, pid_str in all_rows:
        print(f"  {name:<18}{DIM}{ITALIC}{origin:<6}{RESET}{DIM}{iso:<18}pid={pid_str}{RESET}")
    print()

    # --- Controls ---
    print(f"{BOLD}{'Controls':<10}{RESET}{DIM}[q] quit · [r] reset · [space] pause{RESET}")
    print(f"{BOLD}{'Layers':<10}{RESET}{DIM}[1] face · [2] pose · [3] ROI · [4] stats{RESET}")
    print(f"{_}{DIM}[5] timeline · [6] trigger · [7] fusion · [8] frame info{RESET}")
    print()

    handler = DebugFrameHandler(
        show_window=show_window,
        output_path=getattr(args, 'output', None),
        fps=int(args.fps),
        roi=roi,
        backend_label=backend.upper(),
    )

    # --- Frame loop ---
    from visualbase import Trigger
    from momentscan.main import Result

    triggers: list = []
    frame_count = 0

    def collect_trigger(data):
        for result in data.results:
            if result.should_trigger and result.trigger:
                triggers.append(result.trigger)

    graph.on_trigger(collect_trigger)

    try:
        vb, source, stream = create_video_stream(args.path, fps=int(args.fps))
        try:
            if batch_size <= 1:
                # Frame-by-frame (original path)
                for frame in stream:
                    frame_count += 1
                    terminal_results = executor.process(frame)
                    if not handler(frame, terminal_results):
                        break
            else:
                # Batch collection
                batch_frames = []
                stopped = False
                for frame in stream:
                    frame_count += 1
                    batch_frames.append(frame)

                    if len(batch_frames) >= batch_size:
                        batch_results = executor.process_batch(batch_frames)
                        for bf, br in zip(batch_frames, batch_results):
                            if not handler(bf, br):
                                stopped = True
                                break
                        batch_frames = []
                        if stopped:
                            break

                # Flush remaining
                if batch_frames and not stopped:
                    batch_results = executor.process_batch(batch_frames)
                    for bf, br in zip(batch_frames, batch_results):
                        if not handler(bf, br):
                            break
        finally:
            vb.disconnect()

        fps = int(args.fps)
        result = Result(
            triggers=triggers,
            frame_count=frame_count,
            duration_sec=frame_count / fps if frame_count > 0 else 0.0,
        )
        handler.print_summary(result)
    except KeyboardInterrupt:
        print(f"\n{DIM}Interrupted{RESET}")
    finally:
        executor.cleanup()
        handler.cleanup()
        cleanup_observability(hub, file_sink)


def _resolve_selected_to_names(selected: List[str], args) -> List[str]:
    """Convert parsed analyzer selection to ms.run() analyzer names.

    Args:
        selected: Result of _parse_analyzer_arg() (e.g. ['all'], ['face.detect', 'body.pose']).
        args: CLI args.

    Returns:
        List of analyzer names for ms.run().
    """
    if 'all' in selected:
        return ['face.detect', 'face.expression', 'body.pose', 'hand.gesture']

    names = []
    for s in selected:
        names.append(s)
        # Auto-add face.expression when face.detect is selected
        if s == 'face.detect' and 'face.expression' not in selected:
            names.append('face.expression')

    return names if names else ['face.detect', 'face.expression', 'body.pose', 'hand.gesture']


# ---------------------------------------------------------------------------
# Helper functions (kept from original)
# ---------------------------------------------------------------------------

def _get_module_pids(graph) -> Dict[str, int]:
    """Extract PIDs from initialized graph modules.

    Inspects PathNode modules for WorkerModule.runtime_info to get
    per-module PIDs. Inline modules return the main process PID.

    Returns:
        Dict mapping module name to its process PID.
    """
    import os
    from visualpath.flow.nodes.path import PathNode

    pids: Dict[str, int] = {}
    main_pid = os.getpid()

    for node in graph.nodes.values():
        if not isinstance(node, PathNode):
            continue
        for module in node.modules:
            ri = getattr(module, 'runtime_info', None)
            if ri is not None:
                pids[module.name] = ri.pid
            else:
                pids[module.name] = main_pid

    return pids


def _collect_venv_paths(args, analyzer_names: List[str]) -> Dict[str, str]:
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


def _get_analyzer_origins() -> Dict[str, str]:
    """Resolve analyzer package origins from entry points.

    Returns:
        Dict mapping analyzer name to origin label ("vpx" or "core").
    """
    try:
        from visualpath.plugin import discover_modules
        ep_map = discover_modules()
    except ImportError:
        return {}
    origins = {}
    for name, ep in ep_map.items():
        module_path = ep.value.split(":")[0]
        origins[name] = "vpx" if module_path.startswith("vpx.") else "core"
    return origins


def _parse_analyzer_arg(arg: str) -> List[str]:
    """Parse analyzer argument into list of analyzer names."""
    if arg == 'all':
        return ['all']
    if arg in ('raw', 'none'):
        return ['raw']

    parts = [p.strip().lower() for p in arg.split(',')]
    valid = {'face.detect', 'body.pose', 'frame.quality', 'hand.gesture', 'all', 'raw', 'none'}
    for p in parts:
        if p not in valid:
            print(f"Warning: Unknown analyzer '{p}', valid: {valid}")
    return [p for p in parts if p in valid]


def _run_raw_preview(args, show_window: bool):
    """Run raw video preview without any analysis."""
    import cv2

    try:
        vb, source, stream = create_video_stream(args.path, fps=args.fps)
    except IOError:
        print(f"Error: Cannot open {args.path}")
        sys.exit(1)

    duration_sec = source.frame_count / source.fps if source.fps > 0 else 0
    _  = "          "  # 10-char indent

    print()
    print(f"{BOLD}{'Raw':<10}{RESET}{args.path}")
    print(f"{_}{DIM}{source.frame_count} frames · {source.fps:.1f} fps · {source.width}x{source.height}{RESET}")
    print(f"{_}{DIM}preview: {args.fps} fps · duration: {duration_sec:.1f}s ({duration_sec/60:.1f}min){RESET}")
    print(f"{_}{ITALIC}{DIM}no analysis{RESET}")
    print()
    print(f"{BOLD}{'Controls':<10}{RESET}{DIM}[q] quit · [space] pause{RESET}")
    print()

    writer = None
    if args.output:
        writer = create_video_writer(args.output, args.fps, source.width, source.height)

    frame_count = 0
    try:
        for frame in stream:
            image = frame.data
            overlay = image.copy()
            info_text = f"Frame: {frame.frame_id} | Time: {frame.t_src_ns / 1e9:.2f}s | {source.width}x{source.height}"
            cv2.putText(overlay, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(
                overlay, "[RAW PREVIEW - No Analysis]", (10, source.height - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2,
            )

            if writer:
                writer.write(overlay)
            if show_window:
                cv2.imshow("Raw Preview", overlay)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord(" "):
                    cv2.waitKey(0)

            frame_count += 1
            if frame_count % 100 == 0:
                print(f"\rFrame {frame_count}/{source.frame_count}", end="", flush=True)

        print()
    finally:
        vb.disconnect()
        if writer:
            writer.release()
            print(f"Saved: {args.output}")
        if show_window:
            cv2.destroyAllWindows()

    print(f"Previewed {frame_count} frames")


def _parse_roi(roi_str: Optional[str]) -> Optional[tuple]:
    """Parse ROI string to tuple."""
    if not roi_str:
        return None
    try:
        parts = [float(x.strip()) for x in roi_str.split(',')]
        if len(parts) != 4:
            print(f"Warning: Invalid ROI format '{roi_str}', expected x1,y1,x2,y2")
            return None
        x1, y1, x2, y2 = parts
        if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
            print(f"Warning: Invalid ROI values '{roi_str}', must be 0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1")
            return None
        return (x1, y1, x2, y2)
    except ValueError:
        print(f"Warning: Cannot parse ROI '{roi_str}'")
        return None
