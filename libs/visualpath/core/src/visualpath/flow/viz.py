"""Visualization utilities for FlowGraph.

Provides DOT (Graphviz) and ASCII rendering of flow graphs.
"""

from typing import Dict, Set, TYPE_CHECKING

if TYPE_CHECKING:
    from visualpath.flow.graph import FlowGraph


def to_dot(graph: "FlowGraph", title: str = "FlowGraph") -> str:
    """Generate a Graphviz DOT representation of a flow graph.

    Args:
        graph: The flow graph to render.
        title: Title for the graph.

    Returns:
        DOT language string that can be rendered with Graphviz.
    """
    from visualpath.flow.nodes.source import SourceNode
    from visualpath.flow.nodes.path import PathNode
    from visualpath.flow.nodes.filter import FilterNode
    from visualpath.flow.nodes.sampler import SamplerNode
    from visualpath.flow.nodes.branch import BranchNode, FanOutNode
    from visualpath.flow.nodes.join import JoinNode, CollectorNode

    # Node type -> (shape, color)
    _style = {
        SourceNode: ("circle", "#4CAF50"),
        PathNode: ("box", "#2196F3"),
        FilterNode: ("diamond", "#FF9800"),
        SamplerNode: ("diamond", "#FF9800"),
        BranchNode: ("triangle", "#9C27B0"),
        FanOutNode: ("triangle", "#9C27B0"),
        JoinNode: ("invtriangle", "#E91E63"),
        CollectorNode: ("invtriangle", "#E91E63"),
    }
    default_style = ("box", "#607D8B")

    nodes = graph.nodes
    edges = graph.edges

    lines = [
        f'digraph "{title}" {{',
        '  rankdir=TB;',
        '  node [fontname="sans-serif", fontsize=10, style=filled, fillcolor="#EEEEEE"];',
        '  edge [fontname="sans-serif", fontsize=9];',
    ]

    for name, node in nodes.items():
        shape, color = default_style
        for cls, (s, c) in _style.items():
            if isinstance(node, cls):
                shape, color = s, c
                break

        label = name
        type_name = type(node).__name__
        if type_name != name:
            label = f"{name}\\n({type_name})"

        lines.append(
            f'  "{name}" [label="{label}", shape={shape}, fillcolor="{color}", fontcolor="white"];'
        )

    for edge in edges:
        attrs = ""
        if edge.path_filter:
            attrs = f' [label="{edge.path_filter}"]'
        lines.append(f'  "{edge.source}" -> "{edge.target}"{attrs};')

    lines.append("}")
    return "\n".join(lines)


def print_ascii(graph: "FlowGraph") -> str:
    """Generate a simple ASCII representation of a flow graph.

    Args:
        graph: The flow graph to render.

    Returns:
        ASCII string showing nodes and edges.
    """
    nodes = graph.nodes
    visited: Set[str] = set()
    lines: list = []

    def _render(name: str, indent: str = "") -> None:
        if name in visited:
            lines.append(f"{indent}-> [{name}] (ref)")
            return
        visited.add(name)

        type_name = type(nodes[name]).__name__
        lines.append(f"{indent}[{name}] ({type_name})")

        outgoing = graph.get_outgoing_edges(name)
        if len(outgoing) == 1:
            edge = outgoing[0]
            label = f" ({edge.path_filter})" if edge.path_filter else ""
            lines.append(f"{indent}  |{label}")
            lines.append(f"{indent}  v")
            _render(edge.target, indent)
        elif len(outgoing) > 1:
            for j, edge in enumerate(outgoing):
                label = f" {edge.path_filter}" if edge.path_filter else ""
                is_last = j == len(outgoing) - 1
                connector = "\u2514" if is_last else "\u251c"
                branch_indent = indent + ("    " if is_last else "\u2502   ")
                lines.append(f"{indent}  {connector}\u2500\u2500{label}")
                _render(edge.target, branch_indent)

    entry = graph.entry_node or (graph.topological_order()[0] if nodes else None)
    if entry:
        _render(entry)
    return "\n".join(lines)
