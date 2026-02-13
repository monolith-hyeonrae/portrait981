"""Tests for vpx.sdk.paths — home and models directory resolution."""

import warnings
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Ensure path env vars are unset unless explicitly set by a test."""
    monkeypatch.delenv("PORTRAIT981_HOME", raising=False)
    monkeypatch.delenv("VPX_MODELS_DIR", raising=False)


# ── get_home_dir ─────────────────────────────────────────────────────

class TestGetHomeDir:
    def test_default_under_user_home(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        from vpx.sdk.paths import get_home_dir

        result = get_home_dir()
        assert result == tmp_path / ".portrait981"
        assert result.is_dir()

    def test_portrait981_home_override(self, tmp_path, monkeypatch):
        custom = tmp_path / "custom_home"
        monkeypatch.setenv("PORTRAIT981_HOME", str(custom))
        from vpx.sdk.paths import get_home_dir

        result = get_home_dir()
        assert result == custom
        assert result.is_dir()

    def test_creates_directory(self, tmp_path, monkeypatch):
        target = tmp_path / "deep" / "nested" / "home"
        monkeypatch.setenv("PORTRAIT981_HOME", str(target))
        from vpx.sdk.paths import get_home_dir

        assert not target.exists()
        result = get_home_dir()
        assert result.is_dir()


# ── get_models_dir ───────────────────────────────────────────────────

class TestGetModelsDir:
    def test_default_under_home(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PORTRAIT981_HOME", str(tmp_path / "home"))
        from vpx.sdk.paths import get_models_dir

        result = get_models_dir()
        assert result == tmp_path / "home" / "models"
        assert result.is_dir()

    def test_vpx_models_dir_absolute(self, tmp_path, monkeypatch):
        custom = tmp_path / "my_models"
        monkeypatch.setenv("VPX_MODELS_DIR", str(custom))
        from vpx.sdk.paths import get_models_dir

        result = get_models_dir()
        assert result == custom
        assert result.is_dir()

    def test_vpx_models_dir_relative(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("VPX_MODELS_DIR", "rel_models")
        from vpx.sdk.paths import get_models_dir

        result = get_models_dir()
        assert result == tmp_path / "rel_models"
        assert result.is_dir()

    def test_migration_warning(self, tmp_path, monkeypatch):
        """Warn when CWD/models exists but home/models does not."""
        monkeypatch.chdir(tmp_path)
        (tmp_path / "models").mkdir()
        home = tmp_path / "p981_home"
        monkeypatch.setenv("PORTRAIT981_HOME", str(home))
        from vpx.sdk.paths import get_models_dir

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            get_models_dir()

        user_warnings = [x for x in w if issubclass(x.category, UserWarning)]
        assert len(user_warnings) == 1
        assert "default moved to" in str(user_warnings[0].message)

    def test_no_warning_when_home_models_exists(self, tmp_path, monkeypatch):
        """No warning when home/models already exists."""
        monkeypatch.chdir(tmp_path)
        (tmp_path / "models").mkdir()
        home = tmp_path / "p981_home"
        (home / "models").mkdir(parents=True)
        monkeypatch.setenv("PORTRAIT981_HOME", str(home))
        from vpx.sdk.paths import get_models_dir

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            get_models_dir()

        user_warnings = [x for x in w if issubclass(x.category, UserWarning)]
        assert len(user_warnings) == 0

    def test_no_warning_when_no_cwd_models(self, tmp_path, monkeypatch):
        """No warning when CWD/models doesn't exist."""
        monkeypatch.chdir(tmp_path)
        home = tmp_path / "p981_home"
        monkeypatch.setenv("PORTRAIT981_HOME", str(home))
        from vpx.sdk.paths import get_models_dir

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            get_models_dir()

        user_warnings = [x for x in w if issubclass(x.category, UserWarning)]
        assert len(user_warnings) == 0
