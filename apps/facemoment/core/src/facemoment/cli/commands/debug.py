"""Debug command for facemoment CLI.

Unified debug command with DebugSession that consolidates the three
debug modes (pathway, simple, distributed) into a single session class.
"""

import sys
from typing import List, Optional, Tuple, Dict, Any

from facemoment.cli.utils import (
    create_video_stream,
    setup_observability,
    cleanup_observability,
    detect_distributed_mode,
    create_video_writer,
    score_frame,
    BOLD, DIM, ITALIC, RESET,
)


def run_debug(args):
    """Run unified debug session with selected extractors.

    Supports:
    - Single extractor: -e face, -e pose, -e quality, -e gesture
    - Multiple extractors: -e face,pose or -e all (default)
    - Raw video preview: -e raw (no analysis, verify video input)
    - Dummy mode: --no-ml (replaces legacy 'visualize' command)
    - Distributed mode: --distributed (uses VenvWorker for process isolation)
    - Backend selection: --backend pathway (default) or simple
    """
    # Parse extractor selection
    extractor_arg = getattr(args, 'extractor', 'all')
    selected = _parse_extractor_arg(extractor_arg)

    show_window = not getattr(args, 'no_window', False)
    if not show_window and not args.output:
        print("Error: --output is required when using --no-window")
        sys.exit(1)

    # Raw video preview mode (no analysis)
    if 'raw' in selected:
        _run_raw_preview(args, show_window)
        return

    # Check if distributed mode is requested
    distributed = detect_distributed_mode(args)

    session = PathwayDebugSession(
        args, selected, show_window,
        venv_face=getattr(args, 'venv_face', None) if distributed else None,
        venv_pose=getattr(args, 'venv_pose', None) if distributed else None,
        venv_gesture=getattr(args, 'venv_gesture', None) if distributed else None,
    )
    session.run()


# ---------------------------------------------------------------------------
# DebugSession base
# ---------------------------------------------------------------------------

class DebugSession:
    """Base class for debug sessions.

    Consolidates the common setup/loop/teardown pattern shared by
    pathway, simple, and distributed debug modes.

    Subclasses override:
    - _setup_pipeline(): create extractors, fusion, monitor
    - _process_frame(): run extraction and fusion for one frame
    - _teardown_pipeline(): cleanup pipeline-specific resources
    - _print_summary(): print session-specific summary
    """

    def __init__(self, args, selected: List[str], show_window: bool):
        self.args = args
        self.selected = selected
        self.show_window = show_window
        self.profile_mode = getattr(args, 'profile', False)
        self.roi = _parse_roi(getattr(args, 'roi', None)) or (0.3, 0.1, 0.7, 0.6)
        self.backend_label = ""

        # Initialized in _setup()
        self.vb = None
        self.source = None
        self.stream = None
        self.hub = None
        self.file_sink = None
        self.visualizer = None
        self.writer = None
        self.writer_initialized = False
        self.frame_count = 0
        self.scorer = None  # FrameScorer for frame quality scoring

        # Report data (Phase 19c)
        self.report_data: Optional[Dict[str, Any]] = None

    def run(self):
        """Execute the full debug session."""
        self._setup()
        try:
            self._loop()
        finally:
            self._teardown()

    def _setup(self):
        """Common setup: observability, video, visualizer, scorer."""
        import cv2
        from facemoment.moment_detector.visualize import DebugVisualizer
        from facemoment.moment_detector.scoring import FrameScorer

        # Observability
        trace_level = getattr(self.args, 'trace', 'off')
        trace_output = getattr(self.args, 'trace_output', None)
        self.hub, self.file_sink = setup_observability(trace_level, trace_output)

        # Video
        try:
            self.vb, self.source, self.stream = create_video_stream(
                self.args.path, fps=self.args.fps
            )
        except IOError:
            print(f"Error: Cannot open {self.args.path}")
            sys.exit(1)

        # Pipeline-specific setup (must run before header to detect actual backend)
        self._setup_pipeline()

        # Header (after pipeline setup so backend_label is accurate)
        self._print_header()

        # ROI info
        roi_pct = f"{int(self.roi[0]*100)}%-{int(self.roi[2]*100)}% x {int(self.roi[1]*100)}%-{int(self.roi[3]*100)}%"
        print(f"  {DIM}ROI: {roi_pct}{RESET}")
        print()
        print(f"{BOLD}Controls{RESET}  {DIM}[q] quit  [r] reset  [space] pause{RESET}")
        print(f"{BOLD}Layers{RESET}    {DIM}[1] face  [2] pose  [3] ROI  [4] stats{RESET}")
        print(f"          {DIM}[5] timeline  [6] trigger  [7] fusion  [8] frame info{RESET}")
        print()

        # Profile backend info
        if self.profile_mode:
            self._print_backend_info()

        # Visualizer
        self.visualizer = DebugVisualizer()

        # Frame scorer
        self.scorer = FrameScorer()

    def _print_header(self):
        """Print session header."""
        print(f"{BOLD}Debug{RESET}  {self.args.path}")
        parts = [
            f"{self.source.frame_count} frames",
            f"{self.source.fps:.1f} fps",
            (f"{', '.join(self.selected)} extractors" if len(self.selected) > 1
             else f"{self.selected[0]} extractor"),
        ]
        print(f"  {DIM}{' · '.join(parts)}{RESET}")
        meta = [f"backend: {self.backend_label.lower()}"]
        meta.append(f"window: {'enabled' if self.show_window else 'disabled'}")
        if self.profile_mode:
            meta.append("profile: enabled")
        print(f"  {DIM}{' · '.join(meta)}{RESET}")

    def _print_backend_info(self):
        """Print backend info in profile mode. Override in subclasses."""
        pass

    def _setup_pipeline(self):
        """Override: create pipeline-specific resources."""
        raise NotImplementedError

    def _loop(self):
        """Common frame loop."""
        import cv2

        for frame in self.stream:
            # Pipeline-specific processing
            result = self._process_frame(frame)
            if result is None:
                continue

            debug_image, timing_info = result

            # Initialize writer on first frame (canvas includes panels)
            if self.args.output and not self.writer_initialized:
                dh, dw = debug_image.shape[:2]
                self.writer = create_video_writer(self.args.output, self.args.fps, dw, dh)
                self.writer_initialized = True

            if self.writer:
                self.writer.write(debug_image)

            if self.show_window:
                cv2.imshow("Debug", debug_image)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("r"):
                    self._on_reset()
                elif key == ord(" "):
                    cv2.waitKey(0)
                elif ord("1") <= key <= ord("8"):
                    self._toggle_layer(key - ord("0"))

            self.frame_count += 1

            # Progress output
            if self.profile_mode and timing_info:
                detect_ms = timing_info.get('detect_ms', 0)
                expr_ms = timing_info.get('expression_ms', 0)
                total_ms = timing_info.get('total_ms', 0)
                print(
                    f"\rFrame {frame.frame_id}: detect={detect_ms:.1f}ms, "
                    f"expression={expr_ms:.1f}ms, total={total_ms:.1f}ms    ",
                    end="", flush=True,
                )
            elif self.frame_count % 100 == 0:
                print(
                    f"\r{DIM}Frame {self.frame_count}/{self.source.frame_count}{RESET}",
                    end="", flush=True,
                )

        print()

    def _process_frame(self, frame) -> Optional[Tuple]:
        """Override: process one frame.

        Returns:
            (debug_image, timing_info) or None to skip frame.
        """
        raise NotImplementedError

    def _toggle_layer(self, layer_num: int):
        """Toggle a visualization layer by number (1-8)."""
        if self.visualizer is None:
            return
        from facemoment.moment_detector.visualize import DebugLayer
        try:
            layer = DebugLayer(layer_num)
        except ValueError:
            return
        new_state = self.visualizer.layers.toggle(layer)
        state_str = "ON" if new_state else "OFF"
        print(f"\n  Layer {layer_num} ({layer.name}): {state_str}")

    def _on_reset(self):
        """Called when user presses 'r'. Override to reset fusion/visualizer."""
        if self.visualizer:
            self.visualizer.reset()

    def _teardown(self):
        """Common teardown: cleanup video, writer, observability."""
        import cv2

        if self.vb:
            self.vb.disconnect()
        if self.writer:
            self.writer.release()
            print(f"Saved: {self.args.output}")
        if self.show_window:
            cv2.destroyAllWindows()

        self._teardown_pipeline()
        self._print_summary()

        # Generate HTML report if requested
        report_path = getattr(self.args, 'report', None)
        if report_path and self.report_data:
            try:
                from facemoment.moment_detector.visualize.report import generate_report
                generate_report(self.report_data, report_path)
                print(f"Report saved: {report_path}")
            except Exception as e:
                print(f"Warning: Failed to generate report: {e}")

        cleanup_observability(self.hub, self.file_sink)

        print(f"{DIM}Processed {self.frame_count} frames{RESET}")

    def _teardown_pipeline(self):
        """Override: cleanup pipeline-specific resources."""
        pass

    def _print_summary(self):
        """Override: print session-specific summary."""
        pass


# ---------------------------------------------------------------------------
# PathwayDebugSession
# ---------------------------------------------------------------------------

class PathwayDebugSession(DebugSession):
    """Debug session using FacemomentPipeline with PathwayMonitor.

    Default: inline processing (smooth frame-by-frame visualization).
    With --backend pathway: uses actual Pathway streaming engine via
    on_frame_result callback. Pathway batches frames which can cause
    stuttering in the visualization, but uses the real streaming pipeline.
    """

    def __init__(self, args, selected, show_window, *,
                 venv_face=None, venv_pose=None, venv_gesture=None):
        super().__init__(args, selected, show_window)
        self.backend_label = "PATHWAY"
        self.pipeline = None
        self.monitor = None
        self._use_pathway = False
        self._force_pathway = getattr(args, 'backend', None) == 'pathway'
        self._venv_paths = {}
        if venv_face:
            self._venv_paths["face"] = venv_face
        if venv_pose:
            self._venv_paths["pose"] = venv_pose
        if venv_gesture:
            self._venv_paths["gesture"] = venv_gesture

    def _setup_pipeline(self):
        from facemoment.pipeline.pathway_pipeline import FacemomentPipeline, PATHWAY_AVAILABLE
        from facemoment.observability.pathway_monitor import PathwayMonitor

        use_ml = self.args.use_ml
        backend = getattr(self.args, 'backend', None)

        # Build extractor list
        if use_ml is False:
            extractor_names = ['dummy']
        else:
            extractor_names = []
            if 'pose' in self.selected or 'all' in self.selected:
                extractor_names.append('pose')
            if 'gesture' in self.selected or 'all' in self.selected:
                extractor_names.append('gesture')
            if 'face' in self.selected or 'all' in self.selected:
                extractor_names.append('face')
            if 'quality' in self.selected or 'all' in self.selected:
                extractor_names.append('quality')
            if not extractor_names:
                extractor_names = ['dummy']

        distributed = detect_distributed_mode(self.args)

        self.pipeline = FacemomentPipeline(
            extractors=extractor_names,
            fusion_config={"cooldown_sec": 2.0, "main_only": True},
            auto_inject_classifier=True,
            venv_paths=self._venv_paths,
            distributed=distributed,
        )
        self.pipeline.initialize()

        # Detect actual backend
        # Default: inline (smooth frame-by-frame visualization)
        # --backend pathway: force Pathway streaming engine
        if use_ml is False:
            self._use_pathway = False
            self.backend_label = "INLINE (no-ml)"
        elif backend == "simple":
            self._use_pathway = False
            self.backend_label = "INLINE (simple)"
        elif self._force_pathway:
            if not PATHWAY_AVAILABLE:
                self._use_pathway = False
                self.backend_label = "INLINE (pathway unavailable)"
                print("WARNING: Pathway not available — falling back to inline")
            elif self.pipeline.workers:
                self._use_pathway = False
                self.backend_label = "INLINE (subprocess workers active)"
                print("WARNING: ProcessWorkers active — falling back to inline")
            else:
                self._use_pathway = True
                self.backend_label = "PATHWAY"
        else:
            # Default: inline for smooth debug visualization
            self._use_pathway = False
            self.backend_label = "PATHWAY (inline)"

        # Print extractor status
        initialized_names = {ext.name for ext in self.pipeline.extractors}
        worker_names = set(self.pipeline.workers.keys())
        origins = _get_extractor_origins()
        print(f"{BOLD}Extractors{RESET}")
        for name in extractor_names:
            origin = origins.get(name, "")
            if name in worker_names:
                worker = self.pipeline.workers[name]
                info = worker.worker_info
                level = info.isolation_level.lower()
                pid_str = f"  pid={info.pid}" if info.pid else ""
                venv_str = f"  venv={info.venv_path}" if level == "venv" and info.venv_path else ""
                print(f"  {name:<18}{origin:<6}{DIM}enabled  {level}{pid_str}{venv_str}{RESET}")
            elif name in initialized_names:
                print(f"  {name:<18}{origin:<6}{DIM}enabled  inline{RESET}")
            else:
                print(f"  {name:<18}{origin:<6}failed to load")

        if 'face' in extractor_names:
            origin = origins.get('face_classifier', 'core')
            if 'face_classifier' in initialized_names:
                print(f"  {'face_classifier':<18}{origin:<6}{DIM}enabled  auto-injected{RESET}")
            else:
                print(f"  {'face_classifier':<18}{origin:<6}failed to load")

        if not self.pipeline.extractors and not self.pipeline.workers:
            print("Error: No extractors available")
            self.vb.disconnect()
            cleanup_observability(self.hub, self.file_sink)
            sys.exit(1)

        if self.pipeline.fusion:
            print(f"  {'fusion':<18}{'core':<6}{DIM}HighlightFusion  main_only{RESET}")

        self.monitor = PathwayMonitor(hub=self.hub, target_fps=self.args.fps)

    def _print_backend_info(self):
        if self.pipeline:
            print("\nBackends:")
            for ext in self.pipeline.extractors:
                if hasattr(ext, 'get_backend_info'):
                    info = ext.get_backend_info()
                    for component, backend_name in info.items():
                        print(f"  {component.capitalize():12}: {backend_name}")
            print("-" * 50)

    def _loop(self):
        """Route to Pathway or inline (base) loop."""
        if self._use_pathway:
            self._loop_pathway()
        else:
            # Inline mode: reuse base class _loop() which calls _process_frame()
            super()._loop()

    def _loop_pathway(self):
        """Single-phase loop: visualize each frame as PathwayBackend processes it."""
        import cv2
        from visualpath.backends.pathway import PathwayBackend

        fusion = self.pipeline.fusion
        session = self  # capture for closure
        stop_requested = False

        def on_frame_result(frame, obs_list, fusion_result):
            nonlocal stop_requested
            if stop_requested:
                return

            obs_dict = {obs.source: obs for obs in obs_list}
            classifier_obs = obs_dict.get("face_classifier")
            face_obs = obs_dict.get("face") or obs_dict.get("dummy")
            is_gate_open = fusion.is_gate_open if fusion else False
            in_cooldown = fusion.in_cooldown if fusion else False

            # Frame scoring
            obs_dict["face"] = face_obs  # ensure face/dummy is keyed as "face"
            score_result = score_frame(session.scorer, obs_dict)

            debug_image = session.visualizer.create_debug_view(
                frame,
                face_obs=face_obs,
                pose_obs=obs_dict.get("pose"),
                gesture_obs=obs_dict.get("gesture"),
                quality_obs=obs_dict.get("quality"),
                classifier_obs=classifier_obs,
                fusion_result=fusion_result,
                is_gate_open=is_gate_open,
                in_cooldown=in_cooldown,
                roi=session.roi,
                backend_label=session.backend_label,
                score_result=score_result,
            )

            # Writer
            if session.args.output and not session.writer_initialized:
                dh, dw = debug_image.shape[:2]
                session.writer = create_video_writer(session.args.output, session.args.fps, dw, dh)
                session.writer_initialized = True

            if session.writer:
                session.writer.write(debug_image)

            if session.show_window:
                cv2.imshow("Debug", debug_image)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    stop_requested = True
                elif key == ord(" "):
                    cv2.waitKey(0)
                elif ord("1") <= key <= ord("8"):
                    session._toggle_layer(key - ord("0"))

            session.frame_count += 1

        backend = PathwayBackend(window_ns=self.pipeline._window_ns)
        backend.run(
            frames=self.stream,
            extractors=self.pipeline.extractors,
            fusion=fusion,
            on_frame_result=on_frame_result,
        )

    def _process_frame(self, frame):
        return self._process_frame_inline(frame)

    def _process_frame_inline(self, frame):
        """Process one frame inline (no PathwayBackend)."""
        from facemoment.pipeline.frame_processor import process_frame

        result = process_frame(
            frame, self.pipeline.extractors,
            classifier=self.pipeline._classifier,
            workers=self.pipeline.workers,
            fusion=self.pipeline.fusion,
            monitor=self.monitor,
        )

        score_result = score_frame(self.scorer, result.observations)

        debug_image = self.visualizer.create_debug_view(
            frame,
            face_obs=result.observations.get("face") or result.observations.get("dummy"),
            pose_obs=result.observations.get("pose"),
            gesture_obs=result.observations.get("gesture"),
            quality_obs=result.observations.get("quality"),
            classifier_obs=result.classifier_obs,
            fusion_result=result.fusion_result,
            is_gate_open=result.is_gate_open,
            in_cooldown=result.in_cooldown,
            timing=result.timing_info if self.profile_mode else None,
            roi=self.roi,
            monitor_stats=self.monitor.get_frame_stats(),
            backend_label=self.backend_label,
            score_result=score_result,
        )

        return debug_image, result.timing_info

    def _on_reset(self):
        if self.pipeline and self.pipeline.fusion:
            self.pipeline.fusion.reset()
        super()._on_reset()

    def _teardown_pipeline(self):
        if self.pipeline:
            self.pipeline.cleanup()

    def _print_summary(self):
        if self.monitor:
            summary = self.monitor.get_summary()
            _print_pathway_summary(summary)
            self.report_data = _build_report_data(
                summary, self.visualizer, self.backend_label,
            )


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _get_extractor_origins() -> dict:
    """Resolve extractor package origins from entry points.

    Returns:
        Dict mapping extractor name to origin label ("vpx" or "core").
    """
    try:
        from visualpath.plugin import discover_extractors
        ep_map = discover_extractors()
    except ImportError:
        return {}
    origins = {}
    for name, ep in ep_map.items():
        module_path = ep.value.split(":")[0]
        origins[name] = "vpx" if module_path.startswith("vpx.") else "core"
    return origins


def _print_pathway_summary(summary: dict) -> None:
    """Print Pathway pipeline session summary to console."""
    if not summary or summary.get("total_frames", 0) == 0:
        return

    print()
    print(f"{BOLD}Summary{RESET}")
    eff_fps = summary["effective_fps"]
    target = summary.get("target_fps", 0)
    ratio_str = f" / {target:.0f}" if target > 0 else ""
    parts = [
        f"{summary['wall_time_sec']:.1f}s",
        f"{summary['total_frames']} frames",
        f"{eff_fps:.1f}{ratio_str} fps",
        f"{summary['total_triggers']} triggers",
    ]
    print(f"  {DIM}{' · '.join(parts)}{RESET}")

    ext_stats = summary.get("extractor_stats", {})
    fusion_avg = summary.get("fusion_avg_ms", 0)

    if ext_stats:
        # Find bottleneck
        slowest = max(ext_stats, key=lambda k: ext_stats[k].get("avg_ms", 0))
        total_avg = sum(s.get("avg_ms", 0) for s in ext_stats.values()) + fusion_avg

        # Table header
        print()
        print(f"  {DIM}{'Extractor':<18} {'Avg':>7} {'P95':>7} {'Max':>7} {'Errors':>7}{RESET}")

        # Sort by avg_ms descending (bottleneck first)
        sorted_stats = sorted(
            ext_stats.items(),
            key=lambda x: x[1].get("avg_ms", 0),
            reverse=True,
        )
        for name, stats in sorted_stats:
            avg = stats.get("avg_ms", 0)
            p95 = stats.get("p95_ms", 0)
            mx = stats.get("max_ms", 0)
            errs = int(stats.get("errors", 0))
            suffix = ""
            if name == slowest and total_avg > 0:
                pct = avg / total_avg * 100
                suffix = f"  {DIM}{ITALIC}\u2190 bottleneck {pct:.0f}%{RESET}"
            print(f"  {name:<18} {avg:>7.1f} {p95:>7.1f} {mx:>7.1f} {errs:>7}{suffix}")

        if fusion_avg > 0:
            print(f"  {'fusion':<18} {fusion_avg:>7.1f}")

    print()
    gate_pct = summary.get("gate_open_pct", 0)
    print(f"  {DIM}gate: open {gate_pct:.0f}% of frames{RESET}")


def _build_report_data(
    summary: Optional[dict],
    visualizer,
    backend_label: str,
) -> Optional[Dict[str, Any]]:
    """Build report data dict from session results."""
    if not summary:
        return None
    return {
        "backend": backend_label,
        "summary": summary,
        "trigger_thumbs": (
            visualizer._stats_panel._trigger_thumbs if visualizer else []
        ),
    }


def _parse_extractor_arg(arg: str) -> List[str]:
    """Parse extractor argument into list of extractor names."""
    if arg == 'all':
        return ['all']
    if arg in ('raw', 'none'):
        return ['raw']

    parts = [p.strip().lower() for p in arg.split(',')]
    valid = {'face', 'pose', 'quality', 'gesture', 'all', 'raw', 'none'}
    for p in parts:
        if p not in valid:
            print(f"Warning: Unknown extractor '{p}', valid: {valid}")
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

    print("=" * 60)
    print("Raw Video Preview (No Analysis)")
    print("=" * 60)
    print(f"  File: {args.path}")
    print(f"  Resolution: {source.width} x {source.height}")
    print(f"  Native FPS: {source.fps:.2f}")
    print(f"  Preview FPS: {args.fps}")
    print(f"  Total frames: {source.frame_count}")
    print(f"  Duration: {duration_sec:.1f}s ({duration_sec/60:.1f}min)")
    print("-" * 60)
    print("Controls: [q] quit, [space] pause")
    print("-" * 60)

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


