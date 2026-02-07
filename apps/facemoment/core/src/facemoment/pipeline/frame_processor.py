"""Single-frame processing logic for facemoment.

Extracts the core deps-accumulation + worker + fusion pattern from
``PathwayDebugSession._process_frame_inline()`` into a reusable function.

The deps pattern matches ``SimpleInterpreter._interpret_modules()``::

    deps = {}
    for ext in extractors:
        ext_deps = {n: deps[n] for n in ext.depends if n in deps}
        obs = ext.process(frame, ext_deps)
        deps[ext.name] = obs

Plus facemoment-specific additions:
- Composite "face" extractor satisfies "face_detect" dependency
- Subprocess workers (ProcessWorker)
- Monitor hooks (begin/end_frame, begin/end_extractor, etc.)
- Fusion update with classifier observation
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


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
    extractors,
    *,
    classifier=None,
    workers=None,
    fusion=None,
    merge_fn=None,
    monitor=None,
) -> FrameResult:
    """Process a single frame through extractors, workers, and fusion.

    Args:
        frame: Input frame (must have ``frame_id``, ``t_src_ns``).
        extractors: Ordered list of extractor instances.
        classifier: The classifier extractor instance (to identify its obs).
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

    # --- Inline extractors (deps accumulated) ---
    for ext in extractors:
        try:
            extractor_deps = None
            if ext.depends:
                extractor_deps = {
                    n: deps[n] for n in ext.depends if n in deps
                }
                # Composite "face" extractor satisfies "face_detect" dependency
                if (
                    "face_detect" in ext.depends
                    and "face_detect" not in extractor_deps
                    and "face" in deps
                ):
                    extractor_deps["face"] = deps["face"]

            if monitor is not None:
                monitor.begin_extractor(ext.name)
            try:
                obs = ext.process(frame, extractor_deps)
            except TypeError:
                obs = ext.process(frame)

            sub_timings = getattr(obs, "timing", None) if obs else None
            if monitor is not None:
                monitor.end_extractor(ext.name, obs, sub_timings=sub_timings)

            if obs:
                observations[ext.name] = obs
                deps[ext.name] = obs
                if ext is classifier:
                    classifier_obs = obs
        except Exception:
            if monitor is not None:
                monitor.end_extractor(ext.name, None)

    # --- Subprocess workers ---
    for name, worker in workers.items():
        try:
            if monitor is not None:
                monitor.begin_extractor(name)
            result = worker.process(frame, deps=deps)
            if result.observation:
                observations[name] = result.observation
                deps[name] = result.observation
            if monitor is not None:
                monitor.end_extractor(name, result.observation)
        except Exception:
            if monitor is not None:
                monitor.end_extractor(name, None)

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
