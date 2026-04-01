"""Info command for momentscan CLI.

Shows available analyzers, backends, and pipeline structure.
Dynamic introspection via discover_modules(), Module.capabilities,
and get_processing_steps().
"""

from momentscan.cli.utils import BOLD, DIM, ITALIC, RESET


def run_info(args):
    """Show system information and available components."""
    if getattr(args, 'deps', False):
        _print_dependency_graph()
    elif getattr(args, 'graph', None) is not None:
        _print_flow_graph(args.graph)
    elif getattr(args, 'steps', False):
        _print_processing_steps()
    else:
        print(f"{BOLD}MomentScan - System Information{RESET}")
        print("=" * 60)
        _print_version_info()
        _print_modules_section(args.verbose)
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
