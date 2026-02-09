"""Plugins command for visualpath CLI."""

from visualpath.plugin.discovery import discover_analyzers, discover_fusions


def cmd_plugins_list() -> int:
    """List all available plugins.

    Returns:
        Exit code (0 for success).
    """
    print("Available Analyzers:")
    print("-" * 40)

    analyzers = discover_analyzers()
    if analyzers:
        for name, ep in sorted(analyzers.items()):
            # Get module path from entry point
            module_path = f"{ep.value}" if hasattr(ep, "value") else str(ep)
            print(f"  {name:<20} {module_path}")
    else:
        print("  (none found)")

    print()
    print("Available Fusions:")
    print("-" * 40)

    fusions = discover_fusions()
    if fusions:
        for name, ep in sorted(fusions.items()):
            module_path = f"{ep.value}" if hasattr(ep, "value") else str(ep)
            print(f"  {name:<20} {module_path}")
    else:
        print("  (none found)")

    print()
    print("To register plugins, add entry points in pyproject.toml:")
    print('  [project.entry-points."visualpath.analyzers"]')
    print('  my_analyzer = "mypackage.analyzers:MyAnalyzer"')

    return 0
