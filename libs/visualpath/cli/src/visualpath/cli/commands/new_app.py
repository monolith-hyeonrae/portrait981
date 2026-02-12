"""Scaffold a new visualpath-based analysis app.

Usage from CLI:
    visualpath new-app reportrait
    visualpath new-app reportrait --description "AI portrait transformation"
    visualpath new-app reportrait --dry-run
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import List, Optional


# ---------------------------------------------------------------------------
# Repo root detection
# ---------------------------------------------------------------------------

def _find_repo_root() -> Path:
    """Walk up from CWD looking for the workspace pyproject.toml."""
    cur = Path.cwd()
    for parent in (cur, *cur.parents):
        marker = parent / "pyproject.toml"
        if marker.exists() and (parent / "apps").is_dir():
            return parent
    raise RuntimeError(
        "Cannot locate the repository root. "
        "Run this command from inside the repository."
    )


# ---------------------------------------------------------------------------
# Name derivation
# ---------------------------------------------------------------------------

def derive_app_names(app_name: str) -> dict:
    """Derive conventional names from an app name.

    >>> derive_app_names("reportrait")
    {'app_name': 'reportrait', 'class_name': 'ReportraitApp',
     'python_package': 'reportrait', 'dir_name': 'reportrait'}

    >>> derive_app_names("traffic-monitor")
    {'app_name': 'traffic-monitor', 'class_name': 'TrafficMonitorApp',
     'python_package': 'traffic_monitor', 'dir_name': 'traffic-monitor'}
    """
    if "." in app_name:
        raise ValueError(
            f"App names must not contain dots. "
            f"Use 'vpx new' for analyzer modules, got '{app_name}'"
        )

    # CamelCase: traffic-monitor → TrafficMonitorApp
    class_name = "".join(part.capitalize() for part in app_name.split("-")) + "App"
    # Python package: traffic-monitor → traffic_monitor
    python_package = app_name.replace("-", "_")

    return {
        "app_name": app_name,
        "class_name": class_name,
        "python_package": python_package,
        "dir_name": app_name,
    }


# ---------------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------------

def _tpl_app_pyproject(names: dict, description: str) -> str:
    desc = description or f"{names['class_name']} analysis application"
    return f'''\
[project]
name = "{names["app_name"]}"
version = "0.1.0"
description = "{desc}"
requires-python = ">=3.10"
dependencies = [
    "visualbase",
    "visualpath[isolation]",
    "vpx-sdk",
    "numpy>=1.24.0",
    "opencv-python>=4.8.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
]

[project.scripts]
{names["app_name"]} = "{names["python_package"]}.cli:main"

[project.entry-points."visualpath.modules"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/{names["python_package"]}"]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["tests"]

[tool.uv.sources]
visualbase = {{ workspace = true }}
visualpath = {{ workspace = true }}
visualpath-isolation = {{ workspace = true }}
vpx-sdk = {{ workspace = true }}
'''


def _tpl_app_init(names: dict) -> str:
    return f'''\
import pkgutil
__path__ = pkgutil.extend_path(__path__, __name__)

from {names["python_package"]}.main import (
    Result,
    {names["class_name"]},
    run,
    DEFAULT_FPS,
    DEFAULT_BACKEND,
)

__all__ = [
    "Result",
    "{names["class_name"]}",
    "run",
    "DEFAULT_FPS",
    "DEFAULT_BACKEND",
]
'''


def _tpl_app_main(names: dict) -> str:
    return f'''\
"""High-level API for {names["app_name"]}.

    >>> import {names["python_package"]}
    >>> result = {names["python_package"]}.run("video.mp4")
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import visualpath as vp

logger = logging.getLogger(__name__)

DEFAULT_FPS = 10
DEFAULT_BACKEND = "simple"


@dataclass
class Result:
    """Result from run()."""

    triggers: list = field(default_factory=list)
    frame_count: int = 0
    duration_sec: float = 0.0
    actual_backend: str = ""
    stats: Dict[str, Any] = field(default_factory=dict)


class {names["class_name"]}(vp.App):
    """{names["class_name"]} analysis application."""

    fps = DEFAULT_FPS
    backend = DEFAULT_BACKEND

    def configure_modules(self, modules):
        """Configure analysis modules.

        Override to customize module selection and ordering.
        """
        names = list(modules) if modules else []
        return super().configure_modules(names)

    def on_trigger(self, trigger):
        """Handle a trigger event."""
        pass

    def after_run(self, result):
        """Wrap the framework result into an app-specific Result."""
        return Result(
            triggers=result.triggers,
            frame_count=result.frame_count,
            duration_sec=result.duration_sec,
            actual_backend=result.actual_backend,
            stats=result.stats,
        )


def run(
    video: Union[str, Path],
    *,
    modules: Optional[Sequence[str]] = None,
    fps: int = DEFAULT_FPS,
    backend: str = DEFAULT_BACKEND,
    on_trigger: Optional[Callable] = None,
    on_frame: Optional[Callable] = None,
) -> Result:
    """Process a video and return results.

    Args:
        video: Path to video file.
        modules: Analyzer names. None = default set.
        fps: Frames per second to process.
        backend: Execution backend ("simple", "pathway", or "worker").
        on_trigger: Callback when a trigger fires.
        on_frame: Per-frame callback.

    Returns:
        Result with triggers, frame_count, duration_sec, etc.
    """
    app = {names["class_name"]}()
    return app.run(
        video,
        modules=list(modules) if modules else None,
        fps=fps,
        backend=backend,
        on_trigger=on_trigger,
        on_frame=on_frame,
    )
'''


def _tpl_algorithm_init() -> str:
    return """\
import pkgutil
__path__ = pkgutil.extend_path(__path__, __name__)
"""


def _tpl_analyzers_init() -> str:
    return """\
import pkgutil
__path__ = pkgutil.extend_path(__path__, __name__)

from vpx.sdk import Module, Observation

__all__ = ["Module", "Observation"]
"""


def _tpl_app_test(names: dict) -> str:
    return f'''\
"""Basic tests for {names["app_name"]}."""

from unittest.mock import patch, MagicMock

from {names["python_package"]}.main import Result, {names["class_name"]}, DEFAULT_FPS, DEFAULT_BACKEND


class Test{names["class_name"]}:
    def test_instantiation(self):
        app = {names["class_name"]}()
        assert app.fps == DEFAULT_FPS
        assert app.backend == DEFAULT_BACKEND

    def test_result_defaults(self):
        r = Result()
        assert r.frame_count == 0
        assert r.duration_sec == 0.0
        assert r.triggers == []
        assert r.actual_backend == ""
        assert r.stats == {{}}

    def test_after_run(self):
        app = {names["class_name"]}()
        mock_result = MagicMock()
        mock_result.triggers = ["t1"]
        mock_result.frame_count = 42
        mock_result.duration_sec = 1.5
        mock_result.actual_backend = "simple"
        mock_result.stats = {{"fps": 10}}
        result = app.after_run(mock_result)
        assert isinstance(result, Result)
        assert result.frame_count == 42
'''


# ---------------------------------------------------------------------------
# Workspace registration
# ---------------------------------------------------------------------------

def _register_app_workspace_member(root: Path, member_path: str) -> None:
    """Add an app member to [tool.uv.workspace] members in root pyproject.toml.

    Inserts the new member right after existing ``apps/*`` entries
    to keep the list grouped logically.
    """
    pyproject = root / "pyproject.toml"
    text = pyproject.read_text()

    # Already registered?
    if f'"{member_path}"' in text:
        return

    # Find the last apps/* entry and insert after it
    pattern = r'(^    "apps/[^"]+",\n)(?!    "apps/)'
    match = None
    for match in re.finditer(pattern, text, re.MULTILINE):
        pass  # find last match

    if match:
        insert_pos = match.end()
        text = text[:insert_pos] + f'    "{member_path}",\n' + text[insert_pos:]
    else:
        # Fallback: insert before the closing ] of members list
        text = text.replace(
            "\n]\n",
            f'\n    "{member_path}",\n]\n',
            1,
        )

    pyproject.write_text(text)


# ---------------------------------------------------------------------------
# Scaffold function
# ---------------------------------------------------------------------------

def scaffold_app(
    app_name: str,
    description: str = "",
    dry_run: bool = False,
    repo_root: Optional[Path] = None,
) -> List[Path]:
    """Scaffold a new analysis app based on the vp.App pattern.

    Returns list of created (or would-be-created) file paths.
    """
    names = derive_app_names(app_name)
    root = repo_root or _find_repo_root()

    app_dir = root / "apps" / names["dir_name"]
    src_dir = app_dir / "src" / names["python_package"]
    algorithm_dir = src_dir / "algorithm"
    analyzers_dir = algorithm_dir / "analyzers"
    test_dir = app_dir / "tests"

    files: List[tuple[Path, str]] = [
        (app_dir / "pyproject.toml", _tpl_app_pyproject(names, description)),
        (src_dir / "__init__.py", _tpl_app_init(names)),
        (src_dir / "main.py", _tpl_app_main(names)),
        (algorithm_dir / "__init__.py", _tpl_algorithm_init()),
        (analyzers_dir / "__init__.py", _tpl_analyzers_init()),
        (test_dir / f"test_{names['python_package']}.py", _tpl_app_test(names)),
    ]

    paths = [f[0] for f in files]

    if dry_run:
        return paths

    if app_dir.exists():
        raise FileExistsError(
            f"Directory already exists: {app_dir}"
        )

    for path, content in files:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)

    # Register in workspace root pyproject.toml
    member_path = f"apps/{names['dir_name']}"
    _register_app_workspace_member(root, member_path)

    return paths


# ---------------------------------------------------------------------------
# CLI command handler
# ---------------------------------------------------------------------------

def cmd_new_app(
    name: str,
    description: str = "",
    dry_run: bool = False,
) -> int:
    """Handle ``visualpath new-app``."""
    try:
        paths = scaffold_app(name, description=description, dry_run=dry_run)
    except (ValueError, FileExistsError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    if dry_run:
        print("Files that would be created:")
    else:
        print("Created files:")
    for p in paths:
        print(f"  {p}")

    return 0
