"""Convention-over-configuration pipeline runner.

Provides Level 2 API: subclass App, override hooks, call app.run(video).

Example:
    >>> import visualpath as vp
    >>>
    >>> class MyApp(vp.App):
    ...     modules = ["face.detect", "face.expression"]
    ...     backend = "pathway"
    ...
    ...     def configure_modules(self, modules):
    ...         modules = super().configure_modules(modules)
    ...         modules.append(MyFusion())
    ...         return modules
    ...
    ...     def after_run(self, result):
    ...         save_clips(result.triggers)
    ...         return result
    ...
    >>> app = MyApp()
    >>> result = app.run("video.mp4")
"""

import logging
from typing import Any, Callable, Optional, Sequence, Union

from visualpath.core.module import Module

logger = logging.getLogger(__name__)


class App:
    """Convention-over-configuration pipeline runner.

    Level 2 API: subclass -> override hooks -> app.run(video).

    Class-level attributes (override in subclass):
        modules: Default module list (names or Module instances).
        fps: Default frames per second.
        backend: Default execution backend.
    """

    # --- Class-level defaults (override in subclass) ---
    modules: Sequence[Union[str, Module]] = ()
    fps: int = 10  # DEFAULT_FPS
    backend: str = "simple"

    def run(
        self,
        video,
        *,
        modules=None,
        fps=None,
        backend=None,
        isolation=None,
        profile=None,
        on_trigger=None,
        on_frame=None,
    ):
        """Orchestrate the full lifecycle.

        Args:
            video: Path to video file.
            modules: Override module list. None = use class default.
            fps: Override FPS. None = use class default.
            backend: Override backend. None = use class default.
            isolation: Optional IsolationConfig.
            profile: Execution profile name (e.g. "lite", "platform").
            on_trigger: Callback when a trigger fires.
            on_frame: Per-frame callback (frame, results) -> bool.

        Returns:
            ProcessResult (or custom type from after_run).
        """
        from visualpath.runner import (
            ProcessResult,
            get_backend,
            resolve_modules as _resolve_modules,
            _open_video_source,
        )
        from visualpath.api import DEFAULT_FPS
        from visualpath.flow.graph import FlowGraph
        from visualpath.flow.nodes.source import SourceNode
        from visualpath.flow.nodes.path import PathNode

        # Store video path for hooks
        self.video = video

        self.setup()
        try:
            # Merge: explicit args > instance defaults > class defaults
            eff_modules = modules if modules is not None else list(self.modules)
            eff_fps = fps if fps is not None else self.fps
            eff_backend = backend if backend is not None else self.backend

            # 1. Configure modules (Hook)
            resolved = self.configure_modules(eff_modules)

            # 2. Resolve isolation (profile > explicit > auto-detect)
            eff_isolation = isolation
            if profile is not None:
                from visualpath.core.profile import ExecutionProfile, resolve_profile
                exec_profile = ExecutionProfile.from_name(profile)
                eff_isolation = resolve_profile(exec_profile, resolved)
                eff_backend = exec_profile.backend
            elif eff_isolation is not None:
                if eff_backend in ("simple", "pathway"):
                    eff_backend = "worker"
            else:
                from visualpath.core.compat import build_conflict_isolation
                eff_isolation = build_conflict_isolation(resolved)

            # Pathway: skip PROCESS isolation (UDF parallelism)
            if eff_backend == "pathway" and eff_isolation is not None:
                logger.info("Skipping PROCESS isolation for Pathway (UDF parallelism)")
                eff_isolation = None

            # 3. Build graph (Hook)
            graph = self.configure_graph(resolved, isolation=eff_isolation)

            # 4. Register trigger notification
            has_hook = type(self).on_trigger is not App.on_trigger
            if has_hook or on_trigger:
                def _trigger_adapter(data):
                    for obs in data.results:
                        if obs.should_trigger and obs.trigger:
                            if has_hook:
                                self.on_trigger(obs.trigger)
                            if on_trigger:
                                on_trigger(obs.trigger)
                graph.on_trigger(_trigger_adapter)

            # 5. Merge on_frame (Hook + explicit)
            has_frame_hook = type(self).on_frame is not App.on_frame
            frame_cb = None
            if has_frame_hook or on_frame:
                def _frame_handler(frame, terminal_results):
                    cont = True
                    if has_frame_hook:
                        r = self.on_frame(frame, terminal_results)
                        if r is False:
                            cont = False
                    if cont and on_frame:
                        r = on_frame(frame, terminal_results)
                        if r is False:
                            cont = False
                    return cont
                frame_cb = _frame_handler

            # 6. Execute
            engine = get_backend(eff_backend)
            frames, cleanup_fn = _open_video_source(video, eff_fps)
            try:
                pipeline_result = engine.execute(frames, graph, on_frame=frame_cb)
            finally:
                if cleanup_fn:
                    try:
                        cleanup_fn()
                    except Exception:
                        pass

            result = ProcessResult(
                triggers=pipeline_result.triggers,
                frame_count=pipeline_result.frame_count,
                duration_sec=(
                    pipeline_result.frame_count / eff_fps
                    if pipeline_result.frame_count > 0
                    else 0.0
                ),
                actual_backend=engine.name,
                stats=pipeline_result.stats,
            )

            # 7. After run (Hook)
            return self.after_run(result)
        finally:
            self.teardown()

    # --- Lifecycle Hooks ---

    def configure_modules(self, modules):
        """Hook: resolve module list. Default: string -> Module lookup."""
        from visualpath.runner import resolve_modules
        return resolve_modules(modules)

    def configure_graph(self, modules, *, isolation=None):
        """Hook: build FlowGraph. Default: Source -> PathNode (linear)."""
        from visualpath.flow.graph import FlowGraph
        from visualpath.flow.nodes.source import SourceNode
        from visualpath.flow.nodes.path import PathNode

        graph = FlowGraph(entry_node="source")
        graph.add_node(SourceNode(name="source"))
        graph.add_node(PathNode(
            name="pipeline",
            modules=modules,
            parallel=True,
            isolation=isolation,
        ))
        graph.add_edge("source", "pipeline")
        return graph

    def on_frame(self, frame, results):
        """Hook: per-frame callback. Return False to stop."""
        return True

    def on_trigger(self, trigger):
        """Hook: trigger notification."""
        pass

    def setup(self):
        """Hook: called after self.video is set, before execution. Initialize resources."""
        pass

    def teardown(self):
        """Hook: called in finally block, always runs. Release resources."""
        pass

    def after_run(self, result):
        """Hook: post-processing. Default: return ProcessResult as-is."""
        return result
