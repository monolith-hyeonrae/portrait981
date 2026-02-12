"""Tests for visualpath new-app scaffolding."""

import pytest
from pathlib import Path

from visualpath.cli.commands.new_app import (
    derive_app_names,
    scaffold_app,
    _register_app_workspace_member,
)

# Minimal workspace pyproject.toml with apps section
_WORKSPACE_TOML_WITH_APPS = """\
[tool.uv.workspace]
members = [
    "libs/vpx/plugins/face-detect",
    "libs/vpx/plugins/hand-gesture",
    "libs/vpx/sdk",
    "apps/facemoment",
]
"""


@pytest.fixture
def repo_with_apps(tmp_path):
    """Create a minimal fake repo root with workspace pyproject.toml including apps."""
    (tmp_path / "libs" / "vpx").mkdir(parents=True)
    (tmp_path / "apps").mkdir()
    (tmp_path / "pyproject.toml").write_text(_WORKSPACE_TOML_WITH_APPS)
    return tmp_path


class TestDeriveAppNames:
    def test_basic(self):
        names = derive_app_names("reportrait")
        assert names["app_name"] == "reportrait"
        assert names["class_name"] == "ReportraitApp"
        assert names["python_package"] == "reportrait"
        assert names["dir_name"] == "reportrait"

    def test_hyphenated(self):
        names = derive_app_names("traffic-monitor")
        assert names["class_name"] == "TrafficMonitorApp"
        assert names["python_package"] == "traffic_monitor"
        assert names["dir_name"] == "traffic-monitor"

    def test_dot_raises(self):
        with pytest.raises(ValueError, match="must not contain dots"):
            derive_app_names("face.detect")

    def test_underscore(self):
        names = derive_app_names("my_app")
        assert names["class_name"] == "My_appApp"
        assert names["python_package"] == "my_app"
        assert names["dir_name"] == "my_app"


class TestScaffoldAppDryRun:
    def test_dry_run_returns_paths(self, repo_with_apps):
        paths = scaffold_app("reportrait", dry_run=True, repo_root=repo_with_apps)
        names = [p.name for p in paths]

        assert "pyproject.toml" in names
        assert "__init__.py" in names
        assert "main.py" in names
        assert "test_reportrait.py" in names

    def test_dry_run_does_not_create_files(self, repo_with_apps):
        scaffold_app("reportrait", dry_run=True, repo_root=repo_with_apps)
        app_dir = repo_with_apps / "apps" / "reportrait"
        assert not app_dir.exists()


class TestScaffoldAppCreatesFiles:
    def test_creates_all_files(self, repo_with_apps):
        paths = scaffold_app("reportrait", repo_root=repo_with_apps)
        for p in paths:
            assert p.exists(), f"Missing: {p}"

    def test_pyproject_content(self, repo_with_apps):
        scaffold_app(
            "reportrait",
            description="AI portrait transformation",
            repo_root=repo_with_apps,
        )
        content = (repo_with_apps / "apps" / "reportrait" / "pyproject.toml").read_text()

        assert 'name = "reportrait"' in content
        assert "AI portrait transformation" in content
        assert '"visualbase"' in content
        assert '"visualpath[isolation]"' in content
        assert '"vpx-sdk"' in content
        assert 'reportrait = "reportrait.cli:main"' in content
        assert 'packages = ["src/reportrait"]' in content

    def test_main_has_app_class(self, repo_with_apps):
        scaffold_app("reportrait", repo_root=repo_with_apps)
        content = (
            repo_with_apps / "apps" / "reportrait" / "src" / "reportrait" / "main.py"
        ).read_text()

        assert "class ReportraitApp(vp.App):" in content
        assert "def configure_modules" in content
        assert "def after_run" in content
        assert "def run(" in content
        assert "DEFAULT_FPS" in content
        assert "DEFAULT_BACKEND" in content

    def test_init_reexports(self, repo_with_apps):
        scaffold_app("reportrait", repo_root=repo_with_apps)
        content = (
            repo_with_apps / "apps" / "reportrait" / "src" / "reportrait" / "__init__.py"
        ).read_text()

        assert "pkgutil.extend_path" in content
        assert "from reportrait.main import" in content
        assert "Result" in content
        assert "ReportraitApp" in content
        assert "run" in content

    def test_test_content(self, repo_with_apps):
        scaffold_app("reportrait", repo_root=repo_with_apps)
        content = (
            repo_with_apps / "apps" / "reportrait" / "tests" / "test_reportrait.py"
        ).read_text()

        assert "class TestReportraitApp:" in content
        assert "def test_instantiation" in content
        assert "def test_result_defaults" in content


class TestAppWorkspaceRegistration:
    def test_registers_after_apps(self, repo_with_apps):
        scaffold_app("reportrait", repo_root=repo_with_apps)
        content = (repo_with_apps / "pyproject.toml").read_text()

        assert '"apps/reportrait"' in content
        # Should be after existing apps/facemoment
        lines = content.splitlines()
        app_lines = [l.strip().strip('",') for l in lines if '"apps/' in l]
        assert app_lines.index("apps/reportrait") > app_lines.index("apps/facemoment")

    def test_idempotent_registration(self, repo_with_apps):
        scaffold_app("reportrait", repo_root=repo_with_apps)
        # Manually call registration again
        _register_app_workspace_member(repo_with_apps, "apps/reportrait")
        content = (repo_with_apps / "pyproject.toml").read_text()
        assert content.count('"apps/reportrait"') == 1


class TestAlgorithmSkeleton:
    """Tests for algorithm/ skeleton in scaffolded apps."""

    def test_algorithm_dirs_created(self, repo_with_apps):
        scaffold_app("reportrait", repo_root=repo_with_apps)
        base = repo_with_apps / "apps" / "reportrait" / "src" / "reportrait"
        assert (base / "algorithm" / "__init__.py").exists()
        assert (base / "algorithm" / "analyzers" / "__init__.py").exists()

    def test_algorithm_init_content(self, repo_with_apps):
        scaffold_app("reportrait", repo_root=repo_with_apps)
        content = (
            repo_with_apps / "apps" / "reportrait" / "src" / "reportrait"
            / "algorithm" / "__init__.py"
        ).read_text()

        assert "pkgutil.extend_path" in content

    def test_analyzers_init_content(self, repo_with_apps):
        scaffold_app("reportrait", repo_root=repo_with_apps)
        content = (
            repo_with_apps / "apps" / "reportrait" / "src" / "reportrait"
            / "algorithm" / "analyzers" / "__init__.py"
        ).read_text()

        assert "pkgutil.extend_path" in content
        assert "from vpx.sdk import Module, Observation" in content
        assert '"Module"' in content
        assert '"Observation"' in content

    def test_dry_run_includes_algorithm_files(self, repo_with_apps):
        paths = scaffold_app("reportrait", dry_run=True, repo_root=repo_with_apps)
        names = [p.name for p in paths]

        # algorithm/__init__.py and analyzers/__init__.py both show as __init__.py
        init_count = names.count("__init__.py")
        assert init_count >= 3  # top-level + algorithm + analyzers


class TestRefuseExistingAppDir:
    def test_app_refuses_existing(self, repo_with_apps):
        scaffold_app("reportrait", repo_root=repo_with_apps)
        with pytest.raises(FileExistsError, match="already exists"):
            scaffold_app("reportrait", repo_root=repo_with_apps)


class TestGeneratedAppRuns:
    def test_app_instantiation(self, repo_with_apps):
        """Generated app code can be exec'd and App instantiated."""
        scaffold_app("reportrait", repo_root=repo_with_apps)

        import sys
        import types

        main_path = (
            repo_with_apps / "apps" / "reportrait" / "src" / "reportrait" / "main.py"
        )

        # Create fake vp module with App stub
        fake_vp = types.ModuleType("visualpath")

        class _FakeApp:
            fps = 10
            backend = "simple"

        fake_vp.App = _FakeApp
        old_vp = sys.modules.get("visualpath")
        sys.modules["visualpath"] = fake_vp

        try:
            ns = {}
            exec(compile(main_path.read_text(), str(main_path), "exec"), ns)
            AppCls = ns["ReportraitApp"]
            app = AppCls()
            assert app.fps == 10
            assert app.backend == "simple"

            Result = ns["Result"]
            r = Result()
            assert r.frame_count == 0
        finally:
            if old_vp is not None:
                sys.modules["visualpath"] = old_vp
            else:
                sys.modules.pop("visualpath", None)


class TestCLINewAppParser:
    def test_parse_basic(self):
        from visualpath.cli import create_parser

        parser = create_parser()
        args = parser.parse_args(["new-app", "reportrait"])
        assert args.command == "new-app"
        assert args.name == "reportrait"
        assert args.description == ""
        assert args.dry_run is False

    def test_parse_with_flags(self):
        from visualpath.cli import create_parser

        parser = create_parser()
        args = parser.parse_args([
            "new-app", "reportrait",
            "--description", "AI portrait transformation",
            "--dry-run",
        ])
        assert args.description == "AI portrait transformation"
        assert args.dry_run is True
