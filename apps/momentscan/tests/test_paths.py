"""Tests for momentscan.paths — output directory resolution."""

from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("PORTRAIT981_HOME", raising=False)


# ── _sanitize_stem ───────────────────────────────────────────────────

class TestSanitizeStem:
    def test_simple(self):
        from momentscan.paths import _sanitize_stem
        assert _sanitize_stem("my_video") == "my_video"

    def test_spaces_to_underscores(self):
        from momentscan.paths import _sanitize_stem
        assert _sanitize_stem("my video file") == "my_video_file"

    def test_special_characters_removed(self):
        from momentscan.paths import _sanitize_stem
        result = _sanitize_stem("video@#$%test")
        assert result == "videotest"

    def test_consecutive_underscores_collapsed(self):
        from momentscan.paths import _sanitize_stem
        assert _sanitize_stem("a___b") == "a_b"

    def test_leading_trailing_stripped(self):
        from momentscan.paths import _sanitize_stem
        assert _sanitize_stem("_video_") == "video"
        assert _sanitize_stem(".video.") == "video"
        assert _sanitize_stem("_.video._") == "video"

    def test_empty_becomes_untitled(self):
        from momentscan.paths import _sanitize_stem
        assert _sanitize_stem("") == "untitled"
        assert _sanitize_stem("@#$%") == "untitled"

    def test_korean_preserved(self):
        from momentscan.paths import _sanitize_stem
        result = _sanitize_stem("테스트영상")
        assert "테스트영상" in result

    def test_mixed_korean_english(self):
        from momentscan.paths import _sanitize_stem
        result = _sanitize_stem("test 영상 file")
        assert result == "test_영상_file"

    def test_dots_and_hyphens_preserved(self):
        from momentscan.paths import _sanitize_stem
        result = _sanitize_stem("video-2024.01")
        assert result == "video-2024.01"


# ── get_output_dir ───────────────────────────────────────────────────

class TestGetOutputDir:
    def test_basic(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PORTRAIT981_HOME", str(tmp_path / "home"))
        from momentscan.paths import get_output_dir

        result = get_output_dir("/videos/reaction.mp4")
        assert result == tmp_path / "home" / "momentscan" / "output" / "reaction"

    def test_does_not_create_directory(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PORTRAIT981_HOME", str(tmp_path / "home"))
        from momentscan.paths import get_output_dir

        result = get_output_dir("/videos/test.mp4")
        # get_output_dir should NOT create the output dir itself
        assert not result.exists()

    def test_special_chars_in_name(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PORTRAIT981_HOME", str(tmp_path / "home"))
        from momentscan.paths import get_output_dir

        result = get_output_dir("/path/to/my video (2024).mp4")
        assert result.name == "my_video_2024"

    def test_korean_filename(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PORTRAIT981_HOME", str(tmp_path / "home"))
        from momentscan.paths import get_output_dir

        result = get_output_dir("/path/반응영상.mp4")
        assert "반응영상" in result.name

    def test_path_object_input(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PORTRAIT981_HOME", str(tmp_path / "home"))
        from momentscan.paths import get_output_dir

        result = get_output_dir(Path("/videos/clip.mp4"))
        assert result == tmp_path / "home" / "momentscan" / "output" / "clip"

    def test_existing_dir_gets_suffix(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PORTRAIT981_HOME", str(tmp_path / "home"))
        from momentscan.paths import get_output_dir

        # 첫 번째 호출 → base 이름
        first = get_output_dir("/videos/reaction.mp4")
        assert first.name == "reaction"

        # 디렉토리 생성 후 두 번째 호출 → _2
        first.mkdir(parents=True)
        second = get_output_dir("/videos/reaction.mp4")
        assert second.name == "reaction_2"

        # _2도 생성 후 세 번째 호출 → _3
        second.mkdir(parents=True)
        third = get_output_dir("/videos/reaction.mp4")
        assert third.name == "reaction_3"
