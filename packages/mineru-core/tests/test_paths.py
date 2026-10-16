"""Tests for mineru_core.paths."""

from __future__ import annotations

from pathlib import Path

from mineru_core.paths import PROJECT_ROOT, _find_project_root


class TestProjectRoot:
    def test_project_root_exists(self):
        assert PROJECT_ROOT is not None
        assert isinstance(PROJECT_ROOT, Path)

    def test_project_root_is_absolute(self):
        assert PROJECT_ROOT.is_absolute()

    def test_project_root_has_packages(self):
        """The root should contain the packages/ directory."""
        assert (PROJECT_ROOT / "packages").is_dir()

    def test_project_root_has_pyproject(self):
        """The root should contain pyproject.toml with [tool.uv] workspace."""
        pyproject = PROJECT_ROOT / "pyproject.toml"
        assert pyproject.exists()
        text = pyproject.read_text(encoding="utf-8")
        assert "[tool.uv]" in text
        assert "workspace" in text

    def test_find_project_root_detects_marker(self):
        """_find_project_root should return a directory with the marker."""
        root = _find_project_root()
        assert (root / "pyproject.toml").exists()
