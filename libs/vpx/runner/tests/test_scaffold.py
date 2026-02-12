"""Tests for vpx new scaffolding."""

import pytest
from pathlib import Path

from vpx.runner.scaffold import derive_names, scaffold_plugin, scaffold_internal

# Minimal workspace pyproject.toml for tests
_WORKSPACE_TOML = """\
[tool.uv.workspace]
members = [
    "libs/vpx/plugins/face-detect",
    "libs/vpx/plugins/hand-gesture",
    "libs/vpx/sdk",
]
"""


@pytest.fixture
def repo(tmp_path):
    """Create a minimal fake repo root with workspace pyproject.toml."""
    (tmp_path / "libs" / "vpx").mkdir(parents=True)
    (tmp_path / "pyproject.toml").write_text(_WORKSPACE_TOML)
    return tmp_path


class TestDeriveNames:
    def test_basic(self):
        names = derive_names("face.landmark")
        assert names["module_name"] == "face.landmark"
        assert names["domain"] == "face"
        assert names["action"] == "landmark"
        assert names["class_name"] == "FaceLandmarkAnalyzer"
        assert names["output_class"] == "FaceLandmarkOutput"
        assert names["package_name"] == "vpx-face-landmark"
        assert names["python_module"] == "face_landmark"
        assert names["dir_name"] == "face-landmark"

    def test_different_domain(self):
        names = derive_names("scene.transition")
        assert names["class_name"] == "SceneTransitionAnalyzer"
        assert names["output_class"] == "SceneTransitionOutput"
        assert names["package_name"] == "vpx-scene-transition"
        assert names["python_module"] == "scene_transition"

    def test_no_dot_raises(self):
        with pytest.raises(ValueError, match="dot notation"):
            derive_names("facelandmark")


class TestScaffoldPluginDryRun:
    def test_dry_run_returns_paths(self, repo):
        paths = scaffold_plugin("face.landmark", dry_run=True, repo_root=repo)
        names = [p.name for p in paths]

        assert "pyproject.toml" in names
        assert "analyzer.py" in names
        assert "output.py" in names
        assert "base.py" in names  # backend
        assert "test_face_landmark.py" in names

    def test_dry_run_no_backend(self, repo):
        paths = scaffold_plugin(
            "face.landmark", no_backend=True, dry_run=True, repo_root=repo
        )
        names = [p.name for p in paths]

        assert "base.py" not in names
        assert "analyzer.py" in names

    def test_dry_run_does_not_create_files(self, repo):
        scaffold_plugin("face.landmark", dry_run=True, repo_root=repo)
        plugin_dir = repo / "libs" / "vpx" / "plugins" / "face-landmark"
        assert not plugin_dir.exists()

    def test_dry_run_does_not_modify_workspace(self, repo):
        original = (repo / "pyproject.toml").read_text()
        scaffold_plugin("face.landmark", dry_run=True, repo_root=repo)
        assert (repo / "pyproject.toml").read_text() == original


class TestScaffoldInternalDryRun:
    def test_dry_run_returns_paths(self, repo):
        paths = scaffold_internal(
            "scene.transition", dry_run=True, repo_root=repo
        )
        names = [p.name for p in paths]

        assert "__init__.py" in names
        assert "analyzer.py" in names
        assert "output.py" in names
        # internal modules don't have backends or tests
        assert "base.py" not in names
        assert len(paths) == 3

    def test_dry_run_does_not_create_files(self, repo):
        scaffold_internal("scene.transition", dry_run=True, repo_root=repo)
        pkg_dir = (
            repo / "apps" / "momentscan" / "src"
            / "momentscan" / "algorithm" / "analyzers" / "scene_transition"
        )
        assert not pkg_dir.exists()

    def test_dry_run_custom_app(self, repo):
        paths = scaffold_internal(
            "scene.transition", dry_run=True, repo_root=repo, app_name="reportrait"
        )
        # Paths should point to the custom app
        path_strs = [str(p) for p in paths]
        assert any("apps/reportrait" in s for s in path_strs)
        assert not any("apps/momentscan" in s for s in path_strs)


class TestScaffoldPluginCreatesFiles:
    def test_creates_all_files(self, repo):
        paths = scaffold_plugin("face.landmark", repo_root=repo)

        for p in paths:
            assert p.exists(), f"Missing: {p}"

    def test_file_structure(self, repo):
        scaffold_plugin("face.landmark", repo_root=repo)

        base = repo / "libs" / "vpx" / "plugins" / "face-landmark"
        assert (base / "pyproject.toml").exists()
        assert (base / "src" / "vpx" / "__init__.py").exists()
        assert (base / "src" / "vpx" / "face_landmark" / "__init__.py").exists()
        assert (base / "src" / "vpx" / "face_landmark" / "analyzer.py").exists()
        assert (base / "src" / "vpx" / "face_landmark" / "output.py").exists()
        assert (base / "src" / "vpx" / "face_landmark" / "backends" / "__init__.py").exists()
        assert (base / "src" / "vpx" / "face_landmark" / "backends" / "base.py").exists()
        assert (base / "tests" / "test_face_landmark.py").exists()

    def test_no_backend_skips_backends_dir(self, repo):
        scaffold_plugin("face.landmark", no_backend=True, repo_root=repo)

        base = repo / "libs" / "vpx" / "plugins" / "face-landmark"
        assert not (base / "src" / "vpx" / "face_landmark" / "backends").exists()

    def test_pyproject_content(self, repo):
        scaffold_plugin(
            "face.landmark", depends=["face.detect"], repo_root=repo
        )

        content = (
            repo / "libs" / "vpx" / "plugins" / "face-landmark" / "pyproject.toml"
        ).read_text()

        assert 'name = "vpx-face-landmark"' in content
        assert '"face.landmark" = "vpx.face_landmark:FaceLandmarkAnalyzer"' in content
        assert '"vpx-face-detect"' in content

    def test_pyproject_dep_sources(self, repo):
        scaffold_plugin(
            "face.landmark", depends=["face.detect"], repo_root=repo
        )

        content = (
            repo / "libs" / "vpx" / "plugins" / "face-landmark" / "pyproject.toml"
        ).read_text()

        assert "vpx-face-detect = { workspace = true }" in content

    def test_namespace_init(self, repo):
        scaffold_plugin("face.landmark", repo_root=repo)

        content = (
            repo / "libs" / "vpx" / "plugins" / "face-landmark"
            / "src" / "vpx" / "__init__.py"
        ).read_text()

        assert "pkgutil.extend_path" in content

    def test_analyzer_content(self, repo):
        scaffold_plugin(
            "face.landmark", depends=["face.detect"], repo_root=repo
        )

        content = (
            repo / "libs" / "vpx" / "plugins" / "face-landmark"
            / "src" / "vpx" / "face_landmark" / "analyzer.py"
        ).read_text()

        assert "class FaceLandmarkAnalyzer(Module):" in content
        assert 'depends = ["face.detect"]' in content
        assert 'return "face.landmark"' in content


class TestWorkspaceRegistration:
    def test_registers_in_workspace_members(self, repo):
        scaffold_plugin("face.landmark", repo_root=repo)

        content = (repo / "pyproject.toml").read_text()
        assert '"libs/vpx/plugins/face-landmark"' in content

    def test_inserted_after_existing_plugins(self, repo):
        scaffold_plugin("face.landmark", repo_root=repo)

        content = (repo / "pyproject.toml").read_text()
        lines = content.splitlines()
        member_lines = [l.strip().strip('",') for l in lines if "libs/vpx/plugins/" in l]
        # New entry should come after existing plugin entries
        assert "libs/vpx/plugins/face-landmark" in member_lines
        assert member_lines.index("libs/vpx/plugins/face-landmark") > member_lines.index("libs/vpx/plugins/hand-gesture")

    def test_idempotent_registration(self, repo):
        """Manually create the plugin dir then scaffold again — shouldn't duplicate."""
        # Write member manually first
        content = (repo / "pyproject.toml").read_text()
        content = content.replace(
            '"libs/vpx/plugins/hand-gesture",',
            '"libs/vpx/plugins/hand-gesture",\n    "libs/vpx/plugins/face-landmark",',
        )
        (repo / "pyproject.toml").write_text(content)

        # Now scaffold — should skip registration since member already listed
        # (need to avoid FileExistsError by using a different name)
        scaffold_plugin("body.track", repo_root=repo)
        final = (repo / "pyproject.toml").read_text()
        assert final.count('"libs/vpx/plugins/body-track"') == 1


class TestScaffoldInternalCreatesFiles:
    def test_creates_all_files(self, repo):
        paths = scaffold_internal("scene.transition", repo_root=repo)

        for p in paths:
            assert p.exists(), f"Missing: {p}"

    def test_internal_init_content(self, repo):
        scaffold_internal("scene.transition", repo_root=repo)

        content = (
            repo / "apps" / "momentscan" / "src"
            / "momentscan" / "algorithm" / "analyzers" / "scene_transition"
            / "__init__.py"
        ).read_text()

        assert "SceneTransitionAnalyzer" in content
        assert "SceneTransitionOutput" in content
        assert "momentscan.algorithm.analyzers.scene_transition" in content

    def test_internal_analyzer_uses_app_import_path(self, repo):
        """Internal module analyzer.py should use app import path, not vpx.*."""
        scaffold_internal("scene.transition", repo_root=repo)

        content = (
            repo / "apps" / "momentscan" / "src"
            / "momentscan" / "algorithm" / "analyzers" / "scene_transition"
            / "analyzer.py"
        ).read_text()

        # Should import from app path, NOT vpx path
        assert "momentscan.algorithm.analyzers.scene_transition.output" in content
        assert "vpx.scene_transition" not in content

    def test_internal_does_not_modify_workspace(self, repo):
        """Internal modules are sub-packages, not workspace members."""
        original = (repo / "pyproject.toml").read_text()
        scaffold_internal("scene.transition", repo_root=repo)
        assert (repo / "pyproject.toml").read_text() == original


class TestScaffoldInternalCustomApp:
    """Tests for --app flag with scaffold_internal."""

    def test_custom_app_creates_files(self, repo):
        paths = scaffold_internal(
            "scene.transition", repo_root=repo, app_name="reportrait"
        )
        for p in paths:
            assert p.exists(), f"Missing: {p}"

    def test_custom_app_file_structure(self, repo):
        scaffold_internal(
            "scene.transition", repo_root=repo, app_name="reportrait"
        )
        base = (
            repo / "apps" / "reportrait" / "src"
            / "reportrait" / "algorithm" / "analyzers" / "scene_transition"
        )
        assert (base / "__init__.py").exists()
        assert (base / "analyzer.py").exists()
        assert (base / "output.py").exists()

    def test_custom_app_init_content(self, repo):
        scaffold_internal(
            "scene.transition", repo_root=repo, app_name="reportrait"
        )
        content = (
            repo / "apps" / "reportrait" / "src"
            / "reportrait" / "algorithm" / "analyzers" / "scene_transition"
            / "__init__.py"
        ).read_text()

        assert "reportrait.algorithm.analyzers.scene_transition" in content
        assert "momentscan" not in content

    def test_custom_app_analyzer_import_path(self, repo):
        """Analyzer should import from app path, not vpx.*."""
        scaffold_internal(
            "scene.transition", repo_root=repo, app_name="reportrait"
        )
        content = (
            repo / "apps" / "reportrait" / "src"
            / "reportrait" / "algorithm" / "analyzers" / "scene_transition"
            / "analyzer.py"
        ).read_text()

        assert "reportrait.algorithm.analyzers.scene_transition.output" in content
        assert "vpx.scene_transition" not in content
        assert "momentscan" not in content

    def test_hyphenated_app_name(self, repo):
        scaffold_internal(
            "scene.transition", repo_root=repo, app_name="traffic-monitor"
        )
        content = (
            repo / "apps" / "traffic-monitor" / "src"
            / "traffic_monitor" / "algorithm" / "analyzers" / "scene_transition"
            / "__init__.py"
        ).read_text()

        assert "traffic_monitor.algorithm.analyzers.scene_transition" in content


class TestRefuseExistingDirectory:
    def test_plugin_refuses_existing(self, repo):
        # First call succeeds
        scaffold_plugin("face.landmark", repo_root=repo)

        # Second call raises
        with pytest.raises(FileExistsError, match="already exists"):
            scaffold_plugin("face.landmark", repo_root=repo)

    def test_internal_refuses_existing(self, repo):
        scaffold_internal("scene.transition", repo_root=repo)

        with pytest.raises(FileExistsError, match="already exists"):
            scaffold_internal("scene.transition", repo_root=repo)


class TestGeneratedModulePassesHarness:
    """Verify that the scaffolded analyzer can be imported and passes PluginTestHarness."""

    def test_plugin_harness(self, repo):
        """Generated plugin module passes PluginTestHarness checks."""
        scaffold_plugin("test.scaffold", repo_root=repo)

        # Read the generated analyzer source and exec it
        analyzer_path = (
            repo / "libs" / "vpx" / "plugins" / "test-scaffold"
            / "src" / "vpx" / "test_scaffold" / "analyzer.py"
        )
        output_path = (
            repo / "libs" / "vpx" / "plugins" / "test-scaffold"
            / "src" / "vpx" / "test_scaffold" / "output.py"
        )

        # Execute output.py first to define the output class
        output_ns = {}
        exec(compile(output_path.read_text(), str(output_path), "exec"), output_ns)
        TestScaffoldOutput = output_ns["TestScaffoldOutput"]

        # Patch the import in analyzer.py
        import sys
        import types

        # Create a fake vpx.test_scaffold.output module
        fake_mod = types.ModuleType("vpx.test_scaffold.output")
        fake_mod.TestScaffoldOutput = TestScaffoldOutput
        sys.modules["vpx.test_scaffold.output"] = fake_mod
        # Ensure parent exists
        if "vpx.test_scaffold" not in sys.modules:
            parent = types.ModuleType("vpx.test_scaffold")
            parent.output = fake_mod
            sys.modules["vpx.test_scaffold"] = parent

        try:
            analyzer_ns = {"__name__": "vpx.test_scaffold.analyzer"}
            exec(
                compile(analyzer_path.read_text(), str(analyzer_path), "exec"),
                analyzer_ns,
            )
            AnalyzerCls = analyzer_ns["TestScaffoldAnalyzer"]

            analyzer = AnalyzerCls()
            analyzer.initialize()

            from vpx.sdk.testing import PluginTestHarness, FakeFrame, assert_valid_observation

            harness = PluginTestHarness()
            report = harness.check_module(analyzer)
            assert report.valid, report.errors

            frame = FakeFrame.create()
            obs = analyzer.process(frame)
            assert_valid_observation(obs, module=analyzer)

            analyzer.cleanup()
        finally:
            sys.modules.pop("vpx.test_scaffold.output", None)
            sys.modules.pop("vpx.test_scaffold", None)


class TestCLINewParser:
    """Test that the 'new' subcommand parses correctly."""

    def test_parse_basic(self):
        from vpx.runner.cli import _build_parser

        parser = _build_parser()
        args = parser.parse_args(["new", "face.landmark"])
        assert args.command == "new"
        assert args.name == "face.landmark"
        assert args.internal is False
        assert args.depends == ""
        assert args.no_backend is False
        assert args.dry_run is False

    def test_parse_internal(self):
        from vpx.runner.cli import _build_parser

        parser = _build_parser()
        args = parser.parse_args(["new", "scene.transition", "--internal"])
        assert args.internal is True

    def test_parse_depends(self):
        from vpx.runner.cli import _build_parser

        parser = _build_parser()
        args = parser.parse_args(
            ["new", "face.landmark", "--depends", "face.detect"]
        )
        assert args.depends == "face.detect"

    def test_parse_no_backend(self):
        from vpx.runner.cli import _build_parser

        parser = _build_parser()
        args = parser.parse_args(["new", "face.landmark", "--no-backend"])
        assert args.no_backend is True

    def test_parse_dry_run(self):
        from vpx.runner.cli import _build_parser

        parser = _build_parser()
        args = parser.parse_args(["new", "face.landmark", "--dry-run"])
        assert args.dry_run is True

    def test_parse_app_flag(self):
        from vpx.runner.cli import _build_parser

        parser = _build_parser()
        args = parser.parse_args([
            "new", "scene.transition",
            "--internal", "--app", "reportrait",
        ])
        assert args.internal is True
        assert args.app == "reportrait"

    def test_parse_app_default(self):
        from vpx.runner.cli import _build_parser

        parser = _build_parser()
        args = parser.parse_args(["new", "scene.transition", "--internal"])
        assert args.app == "momentscan"

    def test_parse_all_flags(self):
        from vpx.runner.cli import _build_parser

        parser = _build_parser()
        args = parser.parse_args([
            "new", "face.landmark",
            "--internal", "--depends", "face.detect,body.pose",
            "--app", "reportrait",
            "--dry-run",
        ])
        assert args.internal is True
        assert args.depends == "face.detect,body.pose"
        assert args.app == "reportrait"
        assert args.dry_run is True
