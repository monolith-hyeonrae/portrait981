"""Single-frame processing logic for facemoment.

Extracts the core deps-accumulation + worker + fusion pattern from
``PathwayDebugSession._process_frame_inline()`` into a reusable function.

The deps pattern matches ``SimpleInterpreter._interpret_modules()``::

    deps = {}
    for ext in analyzers:
        ext_deps = {n: deps[n] for n in ext.depends if n in deps}
        obs = ext.process(frame, ext_deps)
        deps[ext.name] = obs

Plus facemoment-specific additions:
- Composite "face" analyzer satisfies "face_detect" dependency
- Subprocess workers (ProcessWorker)
- Monitor hooks (begin/end_frame, begin/end_analyzer, etc.)
- Fusion update with classifier observation
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _run_workers_parallel(frame, workers, deps, observations, monitor):
    """Run subprocess workers in parallel and collect results.

    Workers are independent vpx analyzers (no inter-dependencies).
    Each worker.process() blocks on ZMQ I/O, so threading provides
    true parallelism here.  Results populate *deps* and *observations*
    so that subsequent inline analyzers can use them.

    PathwayMonitor is NOT thread-safe, so all monitor calls happen
    in the main thread after futures complete.  Pre-computed
    ``elapsed_ms`` from WorkerResult is passed to ``end_analyzer``
    to preserve accurate per-worker timing.
    """
    def _call(item):
        name, worker = item
        try:
            return name, worker.process(frame, deps=deps)
        except Exception as exc:
            logger.warning("Worker '%s' raised: %s", name, exc)
            return name, None

    with ThreadPoolExecutor(max_workers=len(workers)) as pool:
        futures = {pool.submit(_call, item): item[0] for item in workers.items()}
        results = {}
        for future in as_completed(futures):
            name = futures[future]
            results[name] = future.result()[1]

    # Record results in main thread (monitor is not thread-safe)
    for name in workers:
        result = results.get(name)
        if result is None:
            if monitor is not None:
                monitor.end_analyzer(name, None, elapsed_ms=0.0)
            continue

        if result.error:
            logger.warning("Worker '%s' error: %s", name, result.error)

        if result.observation:
            observations[name] = result.observation
            deps[name] = result.observation

        if monitor is not None:
            monitor.end_analyzer(
                name, result.observation, elapsed_ms=result.timing_ms,
            )


@dataclass
class FrameResult:
    """Result of processing a single frame."""

    observations: Dict[str, Any] = field(default_factory=dict)
    classifier_obs: Optional[Any] = None
    fusion_result: Optional[Any] = None
    is_gate_open: bool = False
    in_cooldown: bool = False
    timing_info: Optional[Dict[str, float]] = None


def process_frame(
    frame,
    analyzers,
    *,
    classifier=None,
    workers=None,
    fusion=None,
    merge_fn=None,
    monitor=None,
) -> FrameResult:
    """Process a single frame through analyzers, workers, and fusion.

    Args:
        frame: Input frame (must have ``frame_id``, ``t_src_ns``).
        analyzers: Ordered list of analyzer instances.
        classifier: The classifier analyzer instance (to identify its obs).
        workers: Dict of ``{name: worker}`` subprocess workers. Optional.
        fusion: Fusion instance with ``update(merged_obs, classifier_obs=)``. Optional.
        merge_fn: Callable ``(obs_list, frame) -> merged_obs`` for fusion input.
            Defaults to ``pipeline.utils.merge_observations``.
        monitor: PathwayMonitor instance for timing hooks. Optional.

    Returns:
        FrameResult with observations, fusion result, and timing info.
    """
    if workers is None:
        workers = {}

    if monitor is not None:
        monitor.begin_frame(frame)

    observations: Dict[str, Any] = {}
    deps: Dict[str, Any] = {}
    classifier_obs = None

    # --- Phase 1: Subprocess workers (before inline, in parallel) ---
    # Workers run first so their results are available as deps for
    # inline analyzers (e.g. face_classifier depends on face).
    # Independent workers run in parallel via ThreadPoolExecutor.
    if workers:
        _run_workers_parallel(frame, workers, deps, observations, monitor)

    # --- Phase 2: Inline analyzers (deps accumulated) ---
    for ext in analyzers:
        try:
            analyzer_deps = None
            if ext.depends:
                analyzer_deps = {
                    n: deps[n] for n in ext.depends if n in deps
                }
                # Composite "face" analyzer satisfies "face_detect" dependency
                if (
                    "face_detect" in ext.depends
                    and "face_detect" not in analyzer_deps
                    and "face" in deps
                ):
                    analyzer_deps["face"] = deps["face"]

            if monitor is not None:
                monitor.begin_analyzer(ext.name)
            try:
                obs = ext.process(frame, analyzer_deps)
            except TypeError:
                obs = ext.process(frame)

            sub_timings = getattr(obs, "timing", None) if obs else None
            if monitor is not None:
                monitor.end_analyzer(ext.name, obs, sub_timings=sub_timings)

            if obs:
                observations[ext.name] = obs
                deps[ext.name] = obs
                if ext is classifier:
                    classifier_obs = obs
        except Exception:
            if monitor is not None:
                monitor.end_analyzer(ext.name, None)

    if classifier_obs and monitor is not None:
        monitor.record_classifier(classifier_obs)

    # --- Fusion ---
    fusion_result = None
    is_gate_open = False
    in_cooldown = False

    if fusion:
        if merge_fn is None:
            from facemoment.pipeline.utils import merge_observations
            merge_fn = merge_observations

        obs_list = list(observations.values())
        merged_obs = merge_fn(obs_list, frame)

        # Extract main_face_id for monitor
        main_face_id = None
        main_face_source = "none"
        if classifier_obs and hasattr(classifier_obs, "data") and classifier_obs.data:
            data = classifier_obs.data
            if hasattr(data, "main_face") and data.main_face:
                main_face_id = data.main_face.face.face_id
                main_face_source = "classifier_obs"
        elif hasattr(merged_obs, "signals") and "main_face_id" in merged_obs.signals:
            main_face_id = merged_obs.signals["main_face_id"]
            main_face_source = "merged_signals"

        if monitor is not None:
            monitor.record_merge(obs_list, merged_obs, main_face_id, main_face_source)
            monitor.begin_fusion()

        fusion_result = fusion.update(merged_obs, classifier_obs=classifier_obs)

        if monitor is not None:
            monitor.end_fusion(fusion_result)

        is_gate_open = fusion.is_gate_open
        in_cooldown = fusion.in_cooldown

    if monitor is not None:
        monitor.end_frame(gate_open=is_gate_open)

    # --- Timing info (for profile mode) ---
    timing_info = None
    face_obs = observations.get("face")
    if face_obs and getattr(face_obs, "timing", None):
        timing_info = face_obs.timing

    return FrameResult(
        observations=observations,
        classifier_obs=classifier_obs,
        fusion_result=fusion_result,
        is_gate_open=is_gate_open,
        in_cooldown=in_cooldown,
        timing_info=timing_info,
    )
