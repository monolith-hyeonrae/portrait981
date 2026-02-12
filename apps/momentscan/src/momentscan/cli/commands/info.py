"""Info command for momentscan CLI.

Shows available analyzers, backends, and pipeline structure.
"""

import sys


def run_info(args):
    """Show system information and available components."""
    # Handle --deps flag
    if getattr(args, 'deps', False):
        _print_dependency_graph()
        return

    # Handle --graph flag
    graph_format = getattr(args, 'graph', None)
    if graph_format is not None:
        _print_flow_graph(graph_format)
        return

    # Handle --steps flag
    if getattr(args, 'steps', False):
        _print_processing_steps()
        return

    print("MomentScan - System Information")
    print("=" * 60)

    # Version info
    _print_version_info()

    # Analyzer availability
    print("\n[Analyzers]")
    print("-" * 60)
    _check_face_analyzer(verbose=args.verbose)
    _check_pose_analyzer(verbose=args.verbose)
    _check_gesture_analyzer(verbose=args.verbose)
    _check_quality_analyzer()

    # Fusion info
    print("\n[Fusion]")
    print("-" * 60)
    _print_fusion_info()

    # Trigger types
    print("\n[Trigger Types]")
    print("-" * 60)
    _print_trigger_types()

    # Pipeline structure
    print("\n[Pipeline Structure]")
    print("-" * 60)
    _print_pipeline_structure()

    # Device info
    if args.verbose:
        print("\n[Device]")
        print("-" * 60)
        _print_device_info()


def _print_version_info():
    """Print version information."""
    try:
        from importlib.metadata import version
        print(f"  momentscan: {version('momentscan')}")
    except Exception:
        print("  momentscan: (version not available)")

    try:
        import visualbase
        version = getattr(visualbase, '__version__', 'installed')
        print(f"  visualbase: {version}")
    except ImportError:
        print("  visualbase: NOT INSTALLED")


def _check_face_analyzer(verbose: bool = False):
    """Check FaceAnalyzer availability."""
    status = {"detection": None, "expression": None}

    # Detection backend (InsightFace)
    try:
        import insightface
        from vpx.face_detect.backends.insightface import InsightFaceSCRFD
        status["detection"] = f"InsightFace SCRFD (v{insightface.__version__})"
    except ImportError as e:
        status["detection"] = f"NOT AVAILABLE (insightface not installed)"
    except Exception as e:
        status["detection"] = f"ERROR: {e}"

    # Expression backend (HSEmotion or PyFeat)
    try:
        from vpx.face_expression.backends.hsemotion import HSEmotionBackend
        import hsemotion_onnx
        status["expression"] = "HSEmotion (fast)"
    except ImportError:
        try:
            from vpx.face_expression.backends.pyfeat import PyFeatBackend
            status["expression"] = "PyFeat (accurate, slow)"
        except ImportError:
            status["expression"] = "NOT AVAILABLE (install hsemotion-onnx or py-feat)"

    available = status["detection"] and "NOT" not in status["detection"]
    icon = "+" if available else "-"
    print(f"  [{icon}] FaceAnalyzer")
    print(f"        Detection:  {status['detection']}")
    print(f"        Expression: {status['expression']}")


def _check_pose_analyzer(verbose: bool = False):
    """Check PoseAnalyzer availability."""
    try:
        import ultralytics
        from vpx.body_pose.backends.yolo_pose import YOLOPoseBackend
        print(f"  [+] PoseAnalyzer")
        print(f"        Backend: YOLO-Pose (ultralytics v{ultralytics.__version__})")
    except ImportError:
        print(f"  [-] PoseAnalyzer")
        print(f"        Backend: NOT AVAILABLE (ultralytics not installed)")


def _check_gesture_analyzer(verbose: bool = False):
    """Check GestureAnalyzer availability."""
    try:
        import mediapipe
        print(f"  [+] GestureAnalyzer")
        print(f"        Backend: MediaPipe Hands (v{mediapipe.__version__})")
    except ImportError:
        print(f"  [-] GestureAnalyzer")
        print(f"        Backend: NOT AVAILABLE (mediapipe not installed)")


def _check_quality_analyzer():
    """Check QualityAnalyzer (always available)."""
    print(f"  [+] QualityAnalyzer")
    print(f"        Backend: OpenCV (blur/brightness/contrast)")


def _print_fusion_info():
    """Print fusion module information."""
    print("  HighlightFusion")
    print("    - Gate: quality + face conditions (hysteresis)")
    print("    - EWMA smoothing for stable signals")
    print("    - Configurable cooldown between triggers")


def _print_trigger_types():
    """Print available trigger types."""
    triggers = [
        ("expression_spike", "FaceAnalyzer", "Sudden expression change"),
        ("head_turn", "FaceAnalyzer", "Fast head rotation"),
        ("hand_wave", "PoseAnalyzer", "Hand waving motion"),
        ("camera_gaze", "FaceAnalyzer", "Looking at camera"),
        ("passenger_interaction", "FaceAnalyzer", "Passengers facing each other"),
        ("gesture_vsign", "GestureAnalyzer", "V-sign gesture"),
        ("gesture_thumbsup", "GestureAnalyzer", "Thumbs up gesture"),
    ]

    for trigger, source, desc in triggers:
        print(f"  {trigger:24s} [{source:16s}] {desc}")


def _print_pipeline_structure():
    """Print pipeline structure diagram."""
    print("""
  Video Source (visualbase)
       │
       ▼
  ┌─────────────────────────────────────────┐
  │              Analyzers                  │
  │  ┌─────────┐ ┌─────────┐ ┌───────────┐  │
  │  │  Face   │ │  Pose   │ │  Quality  │  │
  │  │(detect+ │ │ (YOLO)  │ │(blur/bright)│ │
  │  │ expr)  │ │         │ │           │  │
  │  └────┬────┘ └────┬────┘ └─────┬─────┘  │
  │       │           │            │        │
  │       └───────────┴────────────┘        │
  │                   │                     │
  │                   ▼                     │
  │  ┌─────────────────────────────────┐    │
  │  │       HighlightFusion           │    │
  │  │  Gate Check → Signal Analysis   │    │
  │  │  → Trigger Decision             │    │
  │  └─────────────┬───────────────────┘    │
  └────────────────┼────────────────────────┘
                   │
                   ▼
             Trigger Event
                   │
                   ▼
            Clip Extraction
""")


def _print_device_info():
    """Print device information."""
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


def _print_dependency_graph():
    """Print analyzer dependency graph."""
    print("MomentScan - Analyzer Dependency Graph")
    print("=" * 60)

    # Define analyzers with their dependencies
    analyzers = [
        ("face.detect", [], "Face detection (bbox, head pose)"),
        ("face.classify", ["face.detect"], "Face role classification (main/passenger/transient)"),
        ("face.expression", ["face.detect"], "Expression analysis (emotions)"),
        ("body.pose", [], "Pose estimation (keypoints)"),
        ("hand.gesture", [], "Gesture detection (hand signs)"),
        ("frame.quality", [], "Quality metrics (blur, brightness)"),
    ]

    # Build dependency graph
    print("\n[Dependency Tree]")
    print("-" * 60)

    # Group by root (no dependencies)
    roots = [(name, deps, desc) for name, deps, desc in analyzers if not deps]
    dependents = [(name, deps, desc) for name, deps, desc in analyzers if deps]

    for name, deps, desc in roots:
        print(f"  {name}")
        print(f"  │   {desc}")
        # Find children
        children = [(n, d, ds) for n, d, ds in dependents if name in d]
        for i, (child_name, child_deps, child_desc) in enumerate(children):
            is_last = i == len(children) - 1
            prefix = "└──" if is_last else "├──"
            print(f"  {prefix} {child_name}")
            print(f"  {'    ' if is_last else '│   '}   {child_desc}")
        print()

    # Execution order recommendation
    print("[Recommended Execution Order]")
    print("-" * 60)
    print("  For face analysis pipeline:")
    print("    1. face.detect     → 2. face_classifier → 3. face.expression → 4. fusion")
    print()
    print("  For full analysis pipeline:")
    print("    1. quality         (gate check)")
    print("    2. face_detect     (detection)")
    print("    3. face_classifier (depends on face_detect)")
    print("    4. face.expression (depends on face.detect)")
    print("    5. body.pose       (independent)")
    print("    6. hand.gesture    (independent)")
    print("    7. fusion          (combines all)")
    print()

    # Visualize as ASCII graph
    print("[ASCII Graph]")
    print("-" * 60)
    print("""
  ┌─────────────┐
  │   quality   │ (no deps)
  └──────┬──────┘
         │
  ┌──────┴──────┐
  │ face_detect │ (no deps)
  └──────┬──────┘
         │
    ┌────┴────┐
    │         │
    ▼         ▼
  ┌──────────────┐  ┌─────────────┐
  │  classifier  │  │  face.expr  │  depends: [face.detect]
  │ (main/pass)  │  │ (emotions)  │
  └──────┬───────┘  └──────┬──────┘
         │                 │
         └────────┬────────┘
                  │
  ┌───────────────┼───────────────┐
  │               │               │
  ▼               ▼               ▼
┌──────┐     ┌─────────┐     ┌─────────┐
│ pose │     │ gesture │     │ fusion  │
└──────┘     └─────────┘     └─────────┘
""")


def _build_momentscan_flow_graph():
    """Build a FlowGraph representing the momentscan pipeline.

    Creates a graph with individual analyzer nodes showing dependency
    relationships, rather than a single pipeline node.
    """
    from visualpath.flow import FlowGraph

    # Define all momentscan analyzers with their dependencies
    analyzer_defs = [
        ("frame.quality", [], "QualityAnalyzer"),
        ("face.detect", [], "FaceDetectionAnalyzer"),
        ("face.classify", ["face.detect"], "FaceClassifierAnalyzer"),
        ("face.expression", ["face.detect"], "ExpressionAnalyzer"),
        ("body.pose", [], "PoseAnalyzer"),
        ("hand.gesture", [], "GestureAnalyzer"),
    ]

    # Check which analyzers are actually available
    availability = {}
    availability["frame.quality"] = True  # always available
    try:
        from vpx.face_detect import FaceDetectionAnalyzer  # noqa: F401
        availability["face.detect"] = True
    except ImportError:
        availability["face.detect"] = False
    try:
        from momentscan.algorithm.analyzers.face_classifier import FaceClassifierAnalyzer  # noqa: F401
        availability["face.classify"] = True
    except ImportError:
        availability["face.classify"] = False
    try:
        from vpx.face_expression import ExpressionAnalyzer  # noqa: F401
        availability["face.expression"] = True
    except ImportError:
        availability["face.expression"] = False
    try:
        from vpx.body_pose import PoseAnalyzer  # noqa: F401
        availability["body.pose"] = True
    except ImportError:
        availability["body.pose"] = False
    try:
        from vpx.hand_gesture import GestureAnalyzer  # noqa: F401
        availability["hand.gesture"] = True
    except ImportError:
        availability["hand.gesture"] = False

    # Build graph manually to show dependency structure
    from visualpath.flow.nodes.source import SourceNode
    from visualpath.flow.node import FlowNode, FlowData

    # Lightweight stub node for graph visualization
    class AnalyzerNode(FlowNode):
        """Stub node for visualization only."""
        def __init__(self, node_name):
            self._name = node_name
        @property
        def name(self):
            return self._name
        def process(self, data: FlowData) -> list:
            return [data]

    class FusionNode(FlowNode):
        """Stub node for visualization only."""
        def __init__(self, node_name="fusion"):
            self._name = node_name
        @property
        def name(self):
            return self._name
        def process(self, data: FlowData) -> list:
            return [data]

    graph = FlowGraph(entry_node="source")
    graph.add_node(SourceNode(name="source"))

    available_names = []
    for name, deps, _cls_name in analyzer_defs:
        if not availability.get(name, False):
            continue
        # Skip if dependencies are not available
        if deps and not all(availability.get(d, False) for d in deps):
            continue
        graph.add_node(AnalyzerNode(name))
        available_names.append(name)

    # Add fusion node
    graph.add_node(FusionNode("fusion"))

    # Add edges: source → root analyzers, deps → dependents, all → fusion
    for name, deps, _cls_name in analyzer_defs:
        if name not in available_names:
            continue
        if not deps:
            graph.add_edge("source", name)
        else:
            for dep in deps:
                if dep in available_names:
                    graph.add_edge(dep, name)

    # All leaf analyzers → fusion
    for name in available_names:
        outgoing = graph.get_outgoing_edges(name)
        # Only connect to fusion if no other analyzer depends on this one
        if not outgoing:
            graph.add_edge(name, "fusion")

    return graph, available_names


def _print_flow_graph(fmt: str = "ascii"):
    """Print pipeline FlowGraph visualization.

    Args:
        fmt: Output format - "ascii" for terminal, "dot" for Graphviz DOT.
    """
    try:
        from visualpath.flow import FlowGraph
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


def _print_processing_steps():
    """Print internal processing steps of each analyzer."""
    print("MomentScan - Processing Pipeline Steps")
    print("=" * 70)
    print()
    print("Complete data transformation pipeline from video source to trigger output.")
    print("Each step shows: name, description [backend], input → output (dependencies)")
    print()

    # Source preprocessing (input pipeline)
    print("[SourceProcessor] (Input Pipeline)")
    print("-" * 70)
    try:
        from momentscan.algorithm.source import SourceProcessor
        for i, step in enumerate(SourceProcessor._STEPS, 1):
            opt_marker = " (optional)" if step.optional else ""
            backend_str = f" [{step.backend}]" if step.backend else ""
            deps_str = f" (depends: {step.depends_on})" if step.depends_on else ""
            print(f"  {i}. {step.name}{opt_marker}")
            print(f"     {step.description}{backend_str}")
            print(f"     {step.input_type} → {step.output_type}{deps_str}")
            print()
    except (ImportError, AttributeError):
        print("  (definition not available)")
    print()

    # Backend preprocessing (internal to ML backends)
    print("[BackendPreprocessor] (ML Backend Internal - for reference)")
    print("-" * 70)
    try:
        from momentscan.algorithm.source import BackendPreprocessor
        for i, step in enumerate(BackendPreprocessor._STEPS, 1):
            opt_marker = " (optional)" if step.optional else ""
            backend_str = f" [{step.backend}]" if step.backend else ""
            deps_str = f" (depends: {step.depends_on})" if step.depends_on else ""
            print(f"  {i}. {step.name}{opt_marker}")
            print(f"     {step.description}{backend_str}")
            print(f"     {step.input_type} → {step.output_type}{deps_str}")
            print()
    except (ImportError, AttributeError):
        print("  (definition not available)")
    print()

    analyzers_info = []

    def _get_steps(cls):
        """Get processing steps from class (supports both _STEPS and decorators)."""
        # Try class-level _STEPS first
        if hasattr(cls, '_STEPS'):
            return cls._STEPS
        # Try get_processing_steps (decorator-based)
        try:
            from vpx.sdk import get_processing_steps
            return get_processing_steps(cls)
        except Exception:
            return []

    # Split Face analyzers
    try:
        from vpx.face_detect import FaceDetectionAnalyzer
        analyzers_info.append(("FaceDetectionAnalyzer", "face.detect", _get_steps(FaceDetectionAnalyzer)))
    except (ImportError, AttributeError) as e:
        analyzers_info.append(("FaceDetectionAnalyzer", "face.detect", f"NOT AVAILABLE: {e}"))

    try:
        from vpx.face_expression import ExpressionAnalyzer
        analyzers_info.append(("ExpressionAnalyzer", "face.expression", _get_steps(ExpressionAnalyzer)))
    except (ImportError, AttributeError) as e:
        analyzers_info.append(("ExpressionAnalyzer", "face.expression", f"NOT AVAILABLE: {e}"))

    try:
        from momentscan.algorithm.analyzers.face_classifier import FaceClassifierAnalyzer
        analyzers_info.append(("FaceClassifierAnalyzer", "face.classify", _get_steps(FaceClassifierAnalyzer)))
    except (ImportError, AttributeError) as e:
        analyzers_info.append(("FaceClassifierAnalyzer", "face.classify", f"NOT AVAILABLE: {e}"))

    # PoseAnalyzer
    try:
        from vpx.body_pose import PoseAnalyzer
        analyzers_info.append(("PoseAnalyzer", "body.pose", _get_steps(PoseAnalyzer)))
    except (ImportError, AttributeError) as e:
        analyzers_info.append(("PoseAnalyzer", "body.pose", f"NOT AVAILABLE: {e}"))

    # GestureAnalyzer
    try:
        from vpx.hand_gesture import GestureAnalyzer
        analyzers_info.append(("GestureAnalyzer", "hand.gesture", _get_steps(GestureAnalyzer)))
    except (ImportError, AttributeError) as e:
        analyzers_info.append(("GestureAnalyzer", "hand.gesture", f"NOT AVAILABLE: {e}"))

    # QualityAnalyzer (decorator-based)
    try:
        from momentscan.algorithm.analyzers.quality import QualityAnalyzer
        analyzers_info.append(("QualityAnalyzer", "frame.quality", _get_steps(QualityAnalyzer)))
    except (ImportError, AttributeError) as e:
        analyzers_info.append(("QualityAnalyzer", "frame.quality", f"NOT AVAILABLE: {e}"))

    # Print each analyzer's steps with DAG info
    for class_name, name, steps in analyzers_info:
        print(f"[{class_name}] (name: {name})")
        print("-" * 70)

        if isinstance(steps, str):
            print(f"  {steps}")
        else:
            for i, step in enumerate(steps, 1):
                opt_marker = " (optional)" if step.optional else ""
                backend_str = f" [{step.backend}]" if step.backend else ""
                deps_str = f" (depends: {step.depends_on})" if step.depends_on else ""
                print(f"  {i}. {step.name}{opt_marker}")
                print(f"     {step.description}{backend_str}")
                print(f"     {step.input_type} → {step.output_type}{deps_str}")
                print()

        print()

    # Print ASCII visualization
    print("=" * 70)
    print("[Complete Pipeline Visualization]")
    print("-" * 70)
    print("""
  ┌─────────────────────────────────────────────────────────────────────┐
  │  SOURCE PREPROCESSING                                               │
  │  ┌──────────────┐   ┌───────────────┐   ┌────────┐   ┌───────────┐  │
  │  │ video_decode │ → │ frame_sampling│ → │ resize │ → │frame_create│  │
  │  │ [OpenCV]     │   │ [skip frames] │   │(option)│   │ [Frame]   │  │
  │  └──────────────┘   └───────────────┘   └────────┘   └─────┬─────┘  │
  └────────────────────────────────────────────────────────────┼────────┘
                                                               │
  Frame (BGR, original resolution)                             │
       ┌───────────────────────────────────────────────────────┘
       │
       ├────────────────────────────────┬──────────────────────────────────┐
       │                                │                                  │
       ▼                                ▼                                  ▼
  ┌─────────────────────┐    ┌──────────────────────┐    ┌──────────────────┐
  │  FaceDetectionExt   │    │    QualityAnalyzer  │    │   PoseAnalyzer  │
  │  ┌───────────────┐  │    │ ┌──────────────────┐ │    │ ┌──────────────┐ │
  │  │ detect        │  │    │ │ grayscale_convert│ │    │ │ pose_estim   │ │
  │  │ (→640x640)    │  │    │ │ blur_analysis    │ │    │ │ upper_body   │ │
  │  │ tracking      │  │    │ │ brightness       │ │    │ │ hands_raised │ │
  │  │ roi_filter    │  │    │ │ contrast         │ │    │ │ wave_detect  │ │
  │  └───────┬───────┘  │    │ │ quality_gate     │ │    │ │ aggregation  │ │
  └──────────┼──────────┘    │ └──────────────────┘ │    │ └──────────────┘ │
             │               └──────────────────────┘    └──────────────────┘
    ┌────────┴────────┐
    │                 │
    ▼                 ▼
  ┌────────────────┐  ┌───────────────────┐    ┌───────────────────────┐
  │ExpressionExt   │  │FaceClassifierExt  │    │   GestureAnalyzer    │
  │(depends:       │  │(depends:          │    │ ┌───────────────────┐ │
  │ face_detect)   │  │ face_detect)      │    │ │ hand_detection    │ │
  │ ┌────────────┐ │  │ ┌───────────────┐ │    │ │ finger_state      │ │
  │ │ expression │ │  │ │ track_update  │ │    │ │ gesture_classify  │ │
  │ │ aggregation│ │  │ │ noise_filter  │ │    │ │ aggregation       │ │
  │ └────────────┘ │  │ │ stability_chk │ │    │ └───────────────────┘ │
  └────────────────┘  │ │ role_classify │ │    └───────────────────────┘
                      │ │ role_assign   │ │
                      │ └───────────────┘ │
                      └───────────────────┘
             │                 │                           │
             └─────────────────┴───────────────────────────┘
                                        │
                                        ▼
                               ┌─────────────────┐
                               │HighlightFusion  │
                               │  (Trigger)      │
                               └─────────────────┘
                                        │
                                        ▼
                               ┌─────────────────┐
                               │  Clip Output    │
                               │  [visualbase]   │
                               └─────────────────┘
""")
