"""Scaffold a new analysis module (vpx plugin or namespace plugin).

Usage from CLI:
    vpx new face.landmark                              # vpx plugin (libs/vpx/plugins/)
    vpx new face.landmark --depends face.detect        # with dependency
    vpx new face.landmark --namespace momentscan        # momentscan plugin (at CWD)
    vpx new face.landmark --namespace momentscan --output ./plugins  # at specified path
"""

from __future__ import annotations

import re
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
        if marker.exists() and (parent / "libs" / "vpx").is_dir():
            return parent
    raise RuntimeError(
        "Cannot locate the portrait981 repo root. "
        "Run this command from inside the repository."
    )


# ---------------------------------------------------------------------------
# Name derivation
# ---------------------------------------------------------------------------

def derive_names(module_name: str, namespace: str = "vpx") -> dict:
    """Derive all conventional names from a dot-notation module name.

    >>> derive_names("face.landmark")
    {'module_name': 'face.landmark', 'domain': 'face', 'action': 'landmark',
     'class_name': 'FaceLandmarkAnalyzer', 'output_class': 'FaceLandmarkOutput',
     'namespace': 'vpx', 'package_name': 'vpx-face-landmark',
     'python_module': 'face_landmark', 'dir_name': 'face-landmark'}

    >>> derive_names("face.landmark", namespace="momentscan")
    {'module_name': 'face.landmark', ...,
     'namespace': 'momentscan', 'package_name': 'momentscan-face-landmark', ...}
    """
    if "." not in module_name:
        raise ValueError(
            f"Module name must use dot notation (e.g. 'face.landmark'), "
            f"got '{module_name}'"
        )

    parts = module_name.split(".", 1)
    domain, action = parts[0], parts[1]

    # CamelCase: face.landmark → FaceLandmark
    camel = domain.capitalize() + action.capitalize()

    return {
        "module_name": module_name,
        "domain": domain,
        "action": action,
        "class_name": f"{camel}Analyzer",
        "output_class": f"{camel}Output",
        "namespace": namespace,
        "package_name": f"{namespace}-{domain}-{action}",
        "python_module": f"{domain}_{action}",
        "dir_name": f"{domain}-{action}",
    }


# ---------------------------------------------------------------------------
# Templates (vpx plugin)
# ---------------------------------------------------------------------------

def _tpl_pyproject(names: dict, depends: List[str]) -> str:
    dep_lines = '    "vpx-sdk",\n'
    for dep in depends:
        dep_names = derive_names(dep)
        dep_lines += f'    "{dep_names["package_name"]}",\n'

    source_lines = 'vpx-sdk = { workspace = true }\n'
    for dep in depends:
        dep_names = derive_names(dep)
        source_lines += f'{dep_names["package_name"]} = {{ workspace = true }}\n'

    return f'''\
[project]
name = "{names["package_name"]}"
version = "0.1.0"
description = "{names["class_name"].replace("Analyzer", "")} analyzer"
requires-python = ">=3.10"
dependencies = [
{dep_lines}]

[project.entry-points."visualpath.modules"]
"{names["module_name"]}" = "vpx.{names["python_module"]}:{names["class_name"]}"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/vpx"]

[tool.uv.sources]
{source_lines}'''


def _tpl_namespace_init() -> str:
    return """\
import pkgutil
__path__ = pkgutil.extend_path(__path__, __name__)
"""


def _tpl_init(names: dict, has_backend: bool) -> str:
    ns = names["namespace"]
    lines = [
        f'from {ns}.{names["python_module"]}.analyzer import {names["class_name"]}',
        f'from {ns}.{names["python_module"]}.output import {names["output_class"]}',
    ]
    all_names = [names["class_name"], names["output_class"]]

    if has_backend:
        backend_class = names["class_name"].replace("Analyzer", "Backend")
        lines.append(
            f'from {ns}.{names["python_module"]}.backends.base import {backend_class}'
        )
        all_names.append(backend_class)

    all_str = ", ".join(f'"{n}"' for n in all_names)
    imports = "\n".join(lines)

    return f'''\
{imports}

__all__ = [{all_str}]
'''


def _tpl_analyzer(names: dict, depends: List[str]) -> str:
    ns = names["namespace"]
    depends_attr = ""
    if depends:
        deps_str = ", ".join(f'"{d}"' for d in depends)
        depends_attr = f"\n    depends = [{deps_str}]"

    deps_docstring = ""
    if depends:
        deps_docstring = f"\n            deps: Dependency observations ({', '.join(depends)})."

    return f'''\
"""Analyzer module for {names["module_name"]}."""

from typing import Optional

from vpx.sdk import Module, Observation, ModuleCapabilities
from {ns}.{names["python_module"]}.output import {names["output_class"]}


class {names["class_name"]}(Module):
    """{names["class_name"].replace("Analyzer", "")} analyzer.

    Analyzes frames and produces {names["output_class"]}.
    """{depends_attr}

    @property
    def name(self) -> str:
        return "{names["module_name"]}"

    @property
    def capabilities(self) -> ModuleCapabilities:
        return ModuleCapabilities()

    def initialize(self) -> None:
        """Load models and resources."""
        pass

    def reset(self) -> None:
        """Reset per-run state (called between videos in warm mode)."""
        pass

    def release(self) -> None:
        """Release models and GPU resources (called at shutdown)."""
        pass

    def process(self, frame, deps=None) -> Optional[Observation]:
        """Process a single frame.

        Args:
            frame: Input frame with .data (BGR ndarray), .frame_id, .t_src_ns.{deps_docstring}

        Returns:
            Observation with {names["output_class"]} in .data.
        """
        output = {names["output_class"]}()

        return Observation(
            source=self.name,
            frame_id=frame.frame_id,
            t_ns=frame.t_src_ns,
            signals={{}},
            data=output,
        )
'''


def _tpl_output(names: dict) -> str:
    return f'''\
"""Output type for {names["module_name"]} analyzer."""

from dataclasses import dataclass


@dataclass
class {names["output_class"]}:
    """Output from {names["class_name"]}.

    Add domain-specific fields here.
    """

    pass
'''


def _tpl_backend_init() -> str:
    return ""


def _tpl_backend_base(names: dict) -> str:
    backend_class = names["class_name"].replace("Analyzer", "Backend")
    return f'''\
"""Backend protocol for {names["module_name"]}."""

from typing import Protocol
import numpy as np


class {backend_class}(Protocol):
    """Protocol for {names["module_name"]} backends.

    Implementations should be swappable without changing analyzer logic.
    """

    def initialize(self, device: str = "cpu") -> None:
        """Initialize the backend and load models."""
        ...

    def predict(self, image: np.ndarray):
        """Run inference on an image."""
        ...

    def cleanup(self) -> None:
        """Release resources and unload models."""
        ...
'''


def _tpl_test(names: dict) -> str:
    ns = names["namespace"]
    return f'''\
"""Basic tests for {names["module_name"]} analyzer."""

from vpx.sdk.testing import FakeFrame, PluginTestHarness, assert_valid_observation
from {ns}.{names["python_module"]} import {names["class_name"]}


class Test{names["class_name"]}:
    def setup_method(self):
        self.analyzer = {names["class_name"]}()
        self.analyzer.initialize()

    def teardown_method(self):
        self.analyzer.release()

    def test_harness(self):
        harness = PluginTestHarness()
        report = harness.check_module(self.analyzer)
        assert report.valid, report.errors

    def test_process_returns_observation(self):
        frame = FakeFrame.create()
        obs = self.analyzer.process(frame)
        assert_valid_observation(obs, module=self.analyzer)

    def test_name(self):
        assert self.analyzer.name == "{names["module_name"]}"
'''


# ---------------------------------------------------------------------------
# Templates (namespace plugin — e.g. momentscan)
# ---------------------------------------------------------------------------

def _tpl_ns_pyproject(names: dict, depends: List[str]) -> str:
    """pyproject.toml for a namespace plugin (non-vpx)."""
    ns = names["namespace"]
    dep_lines = '    "vpx-sdk",\n'
    for dep in depends:
        dep_names = derive_names(dep)
        dep_lines += f'    "{dep_names["package_name"]}",\n'

    source_lines = 'vpx-sdk = { workspace = true }\n'
    for dep in depends:
        dep_names = derive_names(dep)
        source_lines += f'{dep_names["package_name"]} = {{ workspace = true }}\n'

    return f'''\
[project]
name = "{names["package_name"]}"
version = "0.1.0"
description = "{names["class_name"].replace("Analyzer", "")} analyzer"
requires-python = ">=3.10"
dependencies = [
{dep_lines}]

[project.entry-points."visualpath.modules"]
"{names["module_name"]}" = "{ns}.{names["python_module"]}:{names["class_name"]}"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/{ns}"]

[tool.uv.sources]
{source_lines}
[tool.pytest.ini_options]
testpaths = ["tests"]
'''


# ---------------------------------------------------------------------------
# Workspace registration
# ---------------------------------------------------------------------------

def _register_workspace_member(root: Path, member_path: str) -> None:
    """Add a member to [tool.uv.workspace] members in root pyproject.toml.

    Inserts the new member right after existing ``libs/vpx/plugins/*`` entries
    to keep the list grouped logically.
    """
    pyproject = root / "pyproject.toml"
    text = pyproject.read_text()

    # Already registered?
    if f'"{member_path}"' in text:
        return

    # Find the last libs/vpx/plugins/* entry and insert after it
    pattern = r'(^    "libs/vpx/plugins/[^"]+",\n)(?!    "libs/vpx/plugins/)'
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


def _compute_member_path(root: Path, plugin_dir: Path) -> Optional[str]:
    """Compute workspace member path relative to repo root.

    Returns None if plugin_dir is outside the repo.
    """
    try:
        return str(plugin_dir.resolve().relative_to(root.resolve()))
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Scaffold functions
# ---------------------------------------------------------------------------

def scaffold_plugin(
    module_name: str,
    depends: Optional[List[str]] = None,
    no_backend: bool = False,
    dry_run: bool = False,
    repo_root: Optional[Path] = None,
) -> List[Path]:
    """Scaffold a vpx plugin package at libs/vpx/plugins/.

    Returns list of created (or would-be-created) file paths.
    """
    depends = depends or []
    names = derive_names(module_name, namespace="vpx")
    root = repo_root or _find_repo_root()

    plugin_dir = root / "libs" / "vpx" / "plugins" / names["dir_name"]
    src_dir = plugin_dir / "src" / "vpx" / names["python_module"]
    ns_dir = plugin_dir / "src" / "vpx"
    test_dir = plugin_dir / "tests"

    files: List[tuple[Path, str]] = [
        (plugin_dir / "pyproject.toml", _tpl_pyproject(names, depends)),
        (ns_dir / "__init__.py", _tpl_namespace_init()),
        (src_dir / "__init__.py", _tpl_init(names, has_backend=not no_backend)),
        (src_dir / "analyzer.py", _tpl_analyzer(names, depends)),
        (src_dir / "output.py", _tpl_output(names)),
    ]

    if not no_backend:
        backend_dir = src_dir / "backends"
        files.append((backend_dir / "__init__.py", _tpl_backend_init()))
        files.append((backend_dir / "base.py", _tpl_backend_base(names)))

    files.append((test_dir / f"test_{names['python_module']}.py", _tpl_test(names)))

    paths = [f[0] for f in files]

    if dry_run:
        return paths

    if plugin_dir.exists():
        raise FileExistsError(
            f"Directory already exists: {plugin_dir}"
        )

    for path, content in files:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)

    # Register in workspace root pyproject.toml
    member_path = f"libs/vpx/plugins/{names['dir_name']}"
    _register_workspace_member(root, member_path)

    return paths


def scaffold_namespace_plugin(
    module_name: str,
    namespace: str,
    depends: Optional[List[str]] = None,
    no_backend: bool = False,
    dry_run: bool = False,
    output_dir: Optional[Path] = None,
    repo_root: Optional[Path] = None,
) -> List[Path]:
    """Scaffold an independent plugin package with a custom namespace.

    Creates a full package (pyproject.toml, tests, namespace init) at the
    specified output directory or CWD.  The namespace determines the Python
    import path (e.g. ``momentscan.face_landmark``).

    Args:
        module_name: Module name in dot notation (e.g. 'face.landmark').
        namespace: Python namespace (e.g. 'momentscan').
        depends: Dependency module names.
        no_backend: Skip backends/ directory generation.
        dry_run: Print file list without creating anything.
        output_dir: Base directory for the plugin package. Defaults to CWD.
        repo_root: Repository root override (for workspace registration).

    Returns list of created (or would-be-created) file paths.
    """
    depends = depends or []
    names = derive_names(module_name, namespace=namespace)

    base_dir = output_dir or Path.cwd()
    plugin_dir = base_dir / names["dir_name"]
    ns_dir = plugin_dir / "src" / namespace
    src_dir = ns_dir / names["python_module"]
    test_dir = plugin_dir / "tests"

    files: List[tuple[Path, str]] = [
        (plugin_dir / "pyproject.toml", _tpl_ns_pyproject(names, depends)),
        (ns_dir / "__init__.py", _tpl_namespace_init()),
        (src_dir / "__init__.py", _tpl_init(names, has_backend=not no_backend)),
        (src_dir / "analyzer.py", _tpl_analyzer(names, depends)),
        (src_dir / "output.py", _tpl_output(names)),
    ]

    if not no_backend:
        backend_dir = src_dir / "backends"
        files.append((backend_dir / "__init__.py", _tpl_backend_init()))
        files.append((backend_dir / "base.py", _tpl_backend_base(names)))

    files.append((test_dir / f"test_{names['python_module']}.py", _tpl_test(names)))

    paths = [f[0] for f in files]

    if dry_run:
        return paths

    if plugin_dir.exists():
        raise FileExistsError(
            f"Directory already exists: {plugin_dir}"
        )

    for path, content in files:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)

    # Register in workspace if inside the repo
    try:
        root = repo_root or _find_repo_root()
        member_path = _compute_member_path(root, plugin_dir)
        if member_path:
            _register_workspace_member(root, member_path)
    except RuntimeError:
        pass  # Not inside a workspace — skip registration

    return paths
