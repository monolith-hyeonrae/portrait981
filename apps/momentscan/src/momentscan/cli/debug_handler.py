"""Debug frame handler for on_frame callback.

DebugFrameHandler implements the ``on_frame(frame, terminal_results)`` callback
protocol. It extracts observations from FlowData, renders a debug visualization,
and handles keyboard input for interactive debugging.

This replaces the old DebugSession/PathwayDebugSession hierarchy by reusing
the same ``ms.run()`` execution path as ``process``.
"""

import time
from typing import Dict, List, Optional, Tuple

import cv2

from momentscan.cli.utils import (
    create_video_writer,
    score_frame,
    BOLD, DIM, ITALIC, RESET,
)


class DebugFrameHandler:
    """Per-frame callback for debug visualization.

    Extracts observations from FlowData terminal results and renders them
    using DebugVisualizer. Handles cv2 window display and keyboard controls.
    Also tracks wall-clock FPS and per-observation timing to populate the
    stats panel (``monitor_stats``).

    Usage::

        handler = DebugFrameHandler(show_window=True, target_fps=10)
        result = ms.run("video.mp4", on_frame=handler)
        handler.print_summary(result)
        handler.cleanup()
    """

    def __init__(
        self,
        *,
        show_window: bool = True,
        output_path: Optional[str] = None,
        fps: int = 10,
        roi: Optional[Tuple[float, float, float, float]] = None,
        backend_label: str = "",
    ):
        from momentscan.visualize import DebugVisualizer
        from momentscan.visualize.embed_tracker import EmbedTracker
        from momentscan.algorithm.analyzers.frame_scoring import FrameScorer

        self.show_window = show_window
        self.output_path = output_path
        self.fps = fps
        self.roi = roi or (0.3, 0.1, 0.7, 0.6)
        self.backend_label = backend_label

        self.visualizer = DebugVisualizer()
        self.scorer = FrameScorer()
        self.embed_tracker = EmbedTracker()

        self._writer = None
        self._writer_initialized = False
        self.frame_count = 0

        # Timing tracking
        self._target_fps = fps
        self._frame_start: Optional[float] = None
        self._last_frame_ms: float = 0.0
        self._fps_window: List[float] = []  # recent frame timestamps
        self._fps_window_size = 30

    def __call__(self, frame, terminal_results: list) -> bool:
        """Process one frame for debug visualization.

        Args:
            frame: The current Frame object.
            terminal_results: List of FlowData from terminal nodes.

        Returns:
            True to continue, False to stop.
        """
        t0 = time.perf_counter()

        fd = terminal_results[0] if terminal_results else None
        if fd is None:
            return True

        # Build observation dict from FlowData
        observations: Dict[str, object] = {}
        for obs in fd.observations:
            observations[obs.source] = obs

        # Extract classifier
        classifier_obs = observations.get("face.classify")

        # Frame scoring
        score_result = score_frame(self.scorer, observations)

        # Build monitor_stats from observation timing data
        monitor_stats = self._build_monitor_stats(observations)

        # Embedding tracking (ArcFace quality + portrait score)
        embed_stats = self.embed_tracker.update(
            face_obs=observations.get("face.detect"),
            portrait_score_obs=observations.get("portrait.score"),
        )

        # Render debug view
        face_obs = observations.get("face.detect")
        debug_image = self.visualizer.create_debug_view(
            frame,
            face_obs=face_obs,
            quality_obs=observations.get("frame.quality"),
            classifier_obs=classifier_obs,
            fusion_result=None,
            is_gate_open=False,
            in_cooldown=False,
            roi=self.roi,
            monitor_stats=monitor_stats,
            backend_label=self.backend_label,
            score_result=score_result,
            expression_obs=observations.get("face.expression"),
            portrait_score_obs=observations.get("portrait.score"),
            embed_stats=embed_stats,
            face_au_obs=observations.get("face.au"),
            head_pose_obs=observations.get("head.pose"),
        )

        # Video writer
        if self.output_path and not self._writer_initialized:
            dh, dw = debug_image.shape[:2]
            self._writer = create_video_writer(self.output_path, self.fps, dw, dh)
            self._writer_initialized = True

        if self._writer:
            self._writer.write(debug_image)

        # Display
        if self.show_window:
            cv2.imshow("Debug", debug_image)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                return False
            elif key == ord(" "):
                cv2.waitKey(0)  # Pause until any key
            elif key == ord("r"):
                self.visualizer.reset()
            elif ord("1") <= key <= ord("8"):
                self._toggle_layer(key - ord("0"))

        self.frame_count += 1

        # Update timing
        t1 = time.perf_counter()
        self._last_frame_ms = (t1 - t0) * 1000
        self._fps_window.append(t1)
        if len(self._fps_window) > self._fps_window_size:
            self._fps_window.pop(0)

        # Progress output
        if self.frame_count % 100 == 0:
            fps = self._effective_fps()
            print(
                f"\r{DIM}Frame {self.frame_count} · {fps:.1f} fps{RESET}",
                end="", flush=True,
            )

        return True

    def _effective_fps(self) -> float:
        """Calculate effective FPS from recent frame timestamps."""
        if len(self._fps_window) < 2:
            return 0.0
        elapsed = self._fps_window[-1] - self._fps_window[0]
        if elapsed <= 0:
            return 0.0
        return (len(self._fps_window) - 1) / elapsed

    def _build_monitor_stats(self, observations: Dict[str, object]) -> Dict:
        """Build monitor_stats dict from observation timing and wall-clock data.

        Extracts per-analyzer timing from ``obs.timing`` dicts and combines
        with wall-clock FPS tracking to match the format expected by
        ``StatsPanel._draw_monitor_stats()``.
        """
        # Per-analyzer timing and metrics from observations
        analyzer_timings: Dict[str, float] = {}
        analyzer_metrics: Dict[str, Dict] = {}
        for name, obs in observations.items():
            timing = getattr(obs, "timing", None)
            if timing and isinstance(timing, dict):
                # Use total if available, otherwise sum all steps
                total = timing.get("total_ms")
                if total is not None:
                    analyzer_timings[name] = total
                else:
                    analyzer_timings[name] = sum(timing.values())
            # Extract module-internal metrics
            md = getattr(obs, "metadata", None)
            if md and isinstance(md, dict):
                m = md.get("_metrics")
                if m:
                    analyzer_metrics[name] = m

        # Bottleneck
        slowest = ""
        bottleneck_pct = 0.0
        if analyzer_timings:
            slowest = max(analyzer_timings, key=analyzer_timings.get)
            if self._last_frame_ms > 0:
                bottleneck_pct = analyzer_timings[slowest] / self._last_frame_ms * 100

        # FPS
        effective_fps = self._effective_fps()
        fps_ratio = effective_fps / self._target_fps if self._target_fps > 0 else 0.0

        result = {
            "frame_id": self.frame_count,
            "total_frame_ms": self._last_frame_ms,
            "analyzer_timings_ms": analyzer_timings,
            "fusion_ms": 0,
            "fusion_decision": "",
            "slowest_analyzer": slowest,
            "bottleneck_pct": bottleneck_pct,
            "main_face": "",
            "effective_fps": effective_fps,
            "target_fps": self._target_fps,
            "fps_ratio": fps_ratio,
        }
        if analyzer_metrics:
            result["analyzer_metrics"] = analyzer_metrics
        return result

    def _toggle_layer(self, layer_num: int) -> None:
        """Toggle a visualization layer."""
        from momentscan.visualize import DebugLayer

        try:
            layer = DebugLayer(layer_num)
        except ValueError:
            return
        new_state = self.visualizer.layers.toggle(layer)
        state_str = "ON" if new_state else "OFF"
        print(f"\n  {DIM}Layer {layer_num} ({ITALIC}{layer.name}{RESET}{DIM}): {state_str}{RESET}")

    def print_summary(self, result) -> None:
        """Print session summary."""
        print()
        highlights = getattr(result, "highlights", [])
        parts = [
            f"{result.frame_count} frames",
            f"{len(highlights)} highlights",
        ]
        if result.duration_sec > 0:
            fps = result.frame_count / result.duration_sec
            parts.append(f"{ITALIC}{fps:.1f} fps{RESET}{DIM}")
        print(f"{BOLD}{'Summary':<10}{RESET}{DIM}{' · '.join(parts)}{RESET}")
        print()

    def cleanup(self) -> None:
        """Release resources."""
        if self._writer:
            self._writer.release()
            print(f"{DIM}Saved: {ITALIC}{self.output_path}{RESET}")
        if self.show_window:
            cv2.destroyAllWindows()
