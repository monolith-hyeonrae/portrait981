"""Info command for momentscan CLI.

Shows available analyzers, backends, and pipeline structure.
Dynamic introspection via discover_modules(), Module.capabilities,
get_processing_steps(), and HighlightConfig.
"""

from momentscan.algorithm.batch.field_mapping import (
    PIPELINE_FIELD_MAPPINGS,
    PIPELINE_DELTA_SPECS,
    PIPELINE_DERIVED_FIELDS,
)
from momentscan.cli.utils import BOLD, DIM, ITALIC, RESET


def run_info(args):
    """Show system information and available components."""
    if getattr(args, 'deps', False):
        _print_dependency_graph()
    elif getattr(args, 'graph', None) is not None:
        _print_flow_graph(args.graph)
    elif getattr(args, 'steps', False):
        _print_processing_steps()
    elif getattr(args, 'scoring', False):
        _print_scoring_detail()
    else:
        print(f"{BOLD}MomentScan - System Information{RESET}")
        print("=" * 60)
        _print_version_info()
        _print_modules_section(args.verbose)
        _print_scoring_section()
        if args.verbose:
            _print_device_info()


# ── Common helper ──

def _discover_all_modules():
    """Discover modules via entry points, load classes.

    Returns list of (name, entry_point, cls_or_error).
    Test modules (mock.*) are excluded.
    """
    from visualpath.plugin.discovery import discover_modules

    entries = discover_modules()
    result = []
    for name, ep in sorted(entries.items()):
        if name.startswith("mock."):
            continue
        try:
            cls = ep.load()
            result.append((name, ep, cls))
        except Exception as e:
            result.append((name, ep, e))
    return result


# ── Kept: already dynamic ──

def _print_version_info():
    """Print version information."""
    try:
        from importlib.metadata import version
        print(f"  momentscan: {version('momentscan')}")
    except Exception:
        print("  momentscan: (version not available)")

    try:
        import visualbase
        ver = getattr(visualbase, '__version__', 'installed')
        print(f"  visualbase: {ver}")
    except ImportError:
        print("  visualbase: NOT INSTALLED")


def _print_device_info():
    """Print device information."""
    print(f"\n{BOLD}[Device]{RESET}")
    # CUDA
    try:
        import torch
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                name = torch.cuda.get_device_name(i)
                mem = torch.cuda.get_device_properties(i).total_memory / 1e9
                print(f"  cuda:{i} - {name} ({mem:.1f} GB)")
        else:
            print("  CUDA: Not available")
    except ImportError:
        print("  PyTorch: Not installed")

    # ONNX Runtime
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        print(f"  ONNX Runtime: {', '.join(providers)}")
    except ImportError:
        print("  ONNX Runtime: Not installed")


# ── New: dynamic modules section ──

def _print_modules_section(verbose):
    """Print discovered modules with capabilities."""
    from visualpath.core.capabilities import Capability

    modules = _discover_all_modules()
    print(f"\n{BOLD}[Modules]{RESET}  {len(modules)} registered")

    for name, ep, cls_or_err in modules:
        if isinstance(cls_or_err, Exception):
            print(f"  {name:<18s}UNAVAILABLE ({cls_or_err})")
            continue

        cls = cls_or_err

        # origin: vpx.* -> "vpx", else "core"
        origin = "vpx" if ep.value.startswith("vpx.") else "core"

        # depends
        depends = getattr(cls, "depends", [])
        deps_str = ", ".join(depends) if depends else "-"

        # capabilities (safe: __init__ should not load models)
        caps_parts = []
        try:
            instance = cls()
            cap = instance.capabilities
            for member in Capability:
                if member != Capability.NONE and member in cap.flags:
                    caps_parts.append(member.name)
            if cap.resource_groups:
                caps_parts.extend(sorted(cap.resource_groups))
        except Exception:
            pass

        caps_str = "  ".join(caps_parts) if caps_parts else "-"

        print(f"  {name:<18s}{origin:<7s}depends: {deps_str:<20s}{caps_str}")


# ── New: scoring summary (basic view) ──

def _print_scoring_section():
    """Print highlight scoring summary from HighlightConfig defaults."""
    from momentscan.algorithm.batch.types import HighlightConfig

    cfg = HighlightConfig()

    print(f"\n{BOLD}[Highlight Scoring]{RESET}")
    print(f"  Quality gate    face_conf >= {cfg.gate_face_confidence:.2f}  "
          f"face_area >= {cfg.gate_face_area_ratio:.2f}  "
          f"blur >= {cfg.gate_blur_min:.0f}  "
          f"bright \u2208 [{cfg.gate_exposure_min:.0f}, {cfg.gate_exposure_max:.0f}]")
    print(f"  Quality score   blur: {cfg.quality_blur_weight:.2f}  "
          f"face_size: {cfg.quality_face_size_weight:.2f}  "
          f"face_recog: {cfg.quality_face_recog_weight:.2f}  "
          f"{DIM}(fallback frontalness: {cfg.quality_frontalness_weight:.2f}){RESET}")
    print(f"  Impact score    embed_face: {cfg.impact_embed_face_weight:.2f}  "
          f"smile: {cfg.impact_smile_intensity_weight:.2f}  "
          f"yaw: {cfg.impact_head_yaw_delta_weight:.2f}  "
          f"mouth: {cfg.impact_mouth_open_weight:.2f}  "
          f"head_vel: {cfg.impact_head_velocity_weight:.2f}")
    print(f"                  wrist: {cfg.impact_wrist_raise_weight:.2f}  "
          f"torso: {cfg.impact_torso_rotation_weight:.2f}  "
          f"face_\u0394: {cfg.impact_face_size_change_weight:.2f}  "
          f"bright_\u0394: {cfg.impact_exposure_change_weight:.2f}")
    print(f"  Temporal        EMA smooth \u03b1={cfg.smoothing_alpha:.2f}  "
          f"\u2192  peaks: dist\u2265{cfg.peak_min_distance_sec:.1f}s, "
          f"prominence\u2265p{cfg.peak_prominence_percentile:.0f}")
    print(f"  Window          peak \u00b1 {cfg.window_half_sec:.1f}s  "
          f"top {cfg.best_frame_count} frames")


# ── New: --scoring detail ──

def _print_scoring_detail():
    """Print detailed highlight scoring pipeline."""
    from collections import defaultdict
    from momentscan.algorithm.batch.types import HighlightConfig

    cfg = HighlightConfig()
    fps = cfg.fps

    # Feature Sources — from registry
    sources: dict[str, list[str]] = defaultdict(list)
    for fm in PIPELINE_FIELD_MAPPINGS:
        sources[fm.source].append(fm.record_field)

    total_fields = sum(len(v) for v in sources.values())
    print(f"{BOLD}[Feature Sources]{RESET}  {total_fields} fields from {len(sources)} analyzers")
    for source, fields in sources.items():
        print(f"  {source:<16s}\u2192 {', '.join(fields)}")

    # Temporal Delta — from registry
    delta_fields = [spec.record_field for spec in PIPELINE_DELTA_SPECS]
    print(f"\n{BOLD}[Temporal Delta]{RESET}  EMA \u03b1={cfg.delta_alpha:.2f}")
    print(f"  {', '.join(delta_fields)}")
    for derived in PIPELINE_DERIVED_FIELDS:
        if derived.name == "head_velocity":
            print(f"  + {derived.name} = {derived.description}")

    # Normalization
    print(f"\n{BOLD}[Normalization]{RESET}  MAD z-score per video")
    print(f"  z = (x - median) / MAD")

    # Quality Gate
    print(f"\n{BOLD}[Quality Gate]{RESET}  fail \u2192 score=0")
    print(f"  face_detected     == True")
    print(f"  face_confidence   >= {cfg.gate_face_confidence:.2f}")
    print(f"  face_area_ratio   >= {cfg.gate_face_area_ratio:.2f}")
    print(f"  blur_score        >= {cfg.gate_blur_min:.1f}    "
          f"{DIM}(0=unmeasured \u2192 pass){RESET}")
    print(f"  brightness        \u2208 [{cfg.gate_exposure_min:.0f}, {cfg.gate_exposure_max:.0f}] "
          f"{DIM}(0=unmeasured \u2192 pass){RESET}")

    # Quality Score
    print(f"\n{BOLD}[Quality Score]{RESET}  = \u03a3(weight \u00d7 feature)")
    print(f"  {cfg.quality_blur_weight:.2f}  blur_norm         {DIM}(min-max){RESET}")
    print(f"  {cfg.quality_face_size_weight:.2f}  face_size_norm    {DIM}(min-max){RESET}")
    print(f"  {cfg.quality_face_recog_weight:.2f}  face_recog_quality  "
          f"{DIM}(ArcFace anchor cosine sim){RESET}")
    print(f"  {cfg.quality_frontalness_weight:.2f}  frontalness       "
          f"{DIM}(fallback: 1 - |yaw|/{cfg.frontalness_max_yaw:.0f}, clamped){RESET}")

    # Impact Score
    print(f"\n{BOLD}[Impact Score]{RESET}  = \u03a3(weight \u00d7 ReLU(z-score delta))")
    print(f"  {cfg.impact_embed_face_weight:.2f}  embed_delta_face  {DIM}(DINOv2 temporal){RESET}")
    print(f"  {cfg.impact_smile_intensity_weight:.2f}  smile_intensity")
    print(f"  {cfg.impact_head_yaw_delta_weight:.2f}  head_yaw")
    print(f"  {cfg.impact_mouth_open_weight:.2f}  mouth_open_ratio")
    print(f"  {cfg.impact_head_velocity_weight:.2f}  head_velocity")
    print(f"  {cfg.impact_wrist_raise_weight:.2f}  wrist_raise")
    print(f"  {cfg.impact_torso_rotation_weight:.2f}  torso_rotation")
    print(f"  {cfg.impact_face_size_change_weight:.2f}  face_area_ratio {DIM}(\u0394){RESET}")
    print(f"  {cfg.impact_exposure_change_weight:.2f}  brightness {DIM}(\u0394){RESET}")

    # Final
    dist_frames = int(cfg.peak_min_distance_sec * fps)
    half_frames = int(cfg.window_half_sec * fps)
    print(f"\n{BOLD}[Final]{RESET}  quality_score \u00d7 impact_score  {DIM}(gated){RESET}")
    print(f"  \u2192 EMA smooth \u03b1={cfg.smoothing_alpha:.2f}")
    print(f"  \u2192 find_peaks: dist\u2265{dist_frames}f "
          f"({cfg.peak_min_distance_sec:.1f}s@{fps:.0f}fps), "
          f"prominence\u2265p{cfg.peak_prominence_percentile:.0f}")
    print(f"  \u2192 window: peak \u00b1 {half_frames}f ({cfg.window_half_sec:.1f}s), "
          f"top {cfg.best_frame_count} frames")


# ── Rewritten: --deps ──

def _print_dependency_graph():
    """Print analyzer dependency graph from discovered modules."""
    print("MomentScan - Analyzer Dependency Graph")
    print("=" * 60)

    modules = _discover_all_modules()
    loaded = [(name, cls) for name, ep, cls in modules if not isinstance(cls, Exception)]

    print(f"\n{BOLD}[Dependency Tree]{RESET}")
    print("-" * 60)

    roots = [(name, cls) for name, cls in loaded if not getattr(cls, "depends", [])]
    dependents = [(name, cls) for name, cls in loaded if getattr(cls, "depends", [])]

    for name, cls in roots:
        print(f"  {name}")
        children = [(n, c) for n, c in dependents if name in getattr(c, "depends", [])]
        for i, (child_name, child_cls) in enumerate(children):
            is_last = i == len(children) - 1
            prefix = "\u2514\u2500\u2500" if is_last else "\u251c\u2500\u2500"
            print(f"  {prefix} {child_name}")
        print()

    # Execution order
    print(f"{BOLD}[Execution Order]{RESET}")
    print("-" * 60)
    order = []
    added = set()
    for name, _cls in roots:
        order.append(name)
        added.add(name)
    for name, _cls in dependents:
        if name not in added:
            order.append(name)
            added.add(name)
    for i, name in enumerate(order, 1):
        cls = dict(loaded).get(name)
        dep_list = getattr(cls, "depends", []) if cls else []
        dep_str = f"  (depends: {', '.join(dep_list)})" if dep_list else ""
        print(f"  {i}. {name}{dep_str}")
    print()


# ── Rewritten: --graph ──

def _build_momentscan_flow_graph():
    """Build FlowGraph dynamically from discovered modules."""
    from visualpath.flow import FlowGraph
    from visualpath.flow.nodes.source import SourceNode
    from visualpath.flow.node import FlowNode
    from visualpath.flow.specs import NodeSpec

    class AnalyzerNode(FlowNode):
        """Stub node for visualization only."""
        def __init__(self, node_name):
            self._name = node_name

        @property
        def name(self):
            return self._name

        @property
        def spec(self) -> NodeSpec:
            return NodeSpec()

    modules = _discover_all_modules()

    # Build name -> depends mapping for available modules
    available = {}
    for name, ep, cls_or_err in modules:
        if isinstance(cls_or_err, Exception):
            continue
        available[name] = getattr(cls_or_err, "depends", [])

    # Filter out modules whose dependencies aren't available
    valid = {}
    for name, deps in available.items():
        if all(d in available for d in deps):
            valid[name] = deps

    graph = FlowGraph(entry_node="source")
    graph.add_node(SourceNode(name="source"))

    for name in sorted(valid):
        graph.add_node(AnalyzerNode(name))

    for name, deps in valid.items():
        if not deps:
            graph.add_edge("source", name)
        else:
            for dep in deps:
                if dep in valid:
                    graph.add_edge(dep, name)

    return graph, sorted(valid.keys())


def _print_flow_graph(fmt="ascii"):
    """Print pipeline FlowGraph visualization."""
    try:
        from visualpath.flow import FlowGraph  # noqa: F401
    except ImportError:
        print("Error: visualpath.flow not available")
        print("  Install visualpath >= 0.2.0 for FlowGraph support")
        return

    graph, analyzer_names = _build_momentscan_flow_graph()

    print("MomentScan - Pipeline FlowGraph")
    print("=" * 60)
    print(f"  Analyzers: {', '.join(analyzer_names)}")
    print()

    if fmt == "dot":
        print(graph.to_dot("MomentScan Pipeline"))
    else:
        print(graph.print_ascii())


# ── Rewritten: --steps ──

def _print_processing_steps():
    """Print processing steps from discovered modules."""
    print("MomentScan - Processing Pipeline Steps")
    print("=" * 70)
    print()

    modules = _discover_all_modules()

    for name, ep, cls_or_err in modules:
        if isinstance(cls_or_err, Exception):
            print(f"[{name}]  UNAVAILABLE ({cls_or_err})")
            print()
            continue

        cls = cls_or_err
        class_name = cls.__name__

        # Get processing steps
        steps = []
        if hasattr(cls, '_STEPS'):
            steps = cls._STEPS
        else:
            try:
                from vpx.sdk.steps import get_processing_steps
                steps = get_processing_steps(cls)
            except Exception:
                pass

        print(f"[{class_name}] (name: {name})")
        print("-" * 70)

        if not steps:
            print("  (no steps defined)")
        else:
            for i, step in enumerate(steps, 1):
                opt_marker = " (optional)" if step.optional else ""
                backend_str = f" [{step.backend}]" if step.backend else ""
                deps_str = f" (depends: {step.depends_on})" if step.depends_on else ""
                print(f"  {i}. {step.name}{opt_marker}")
                print(f"     {step.description}{backend_str}")
                print(f"     {step.input_type} \u2192 {step.output_type}{deps_str}")
                print()

        print()
