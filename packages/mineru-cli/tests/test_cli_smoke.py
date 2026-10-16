"""CLI smoke tests — verify all subcommands can display --help without crash."""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

CLI_MODULE = "mineru_cli"
PYTHON = sys.executable

SUBCOMMANDS = [
    "health",
    "config",
    "demo",
    "track1",
    "track2",
    "track3",
    "trace",
    "run",
]


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [PYTHON, "-m", CLI_MODULE, *args],
        capture_output=True,
        text=True,
        timeout=15,
    )


class TestCliHelp:
    def test_no_args_shows_logo(self):
        """Invoking without args should show the logo, not crash."""
        proc = _run()
        assert proc.returncode == 0
        assert "Scientific Document Evaluation Toolkit" in proc.stdout

    def test_help_flag(self):
        proc = _run("--help")
        assert proc.returncode == 0
        assert "Usage:" in proc.stdout

    @pytest.mark.parametrize("cmd", SUBCOMMANDS)
    def test_subcommand_help(self, cmd: str):
        proc = _run(cmd, "--help")
        assert proc.returncode == 0, f"{cmd} --help failed:\n{proc.stderr}"
        assert "Usage:" in proc.stdout

    def test_logo_command(self):
        proc = _run("logo")
        assert proc.returncode == 0
        assert "Scientific Document Evaluation Toolkit" in proc.stdout

    def test_doctor_command_json_output(self):
        """doctor --json should produce valid JSON (may show degraded status)."""
        proc = _run("doctor", "--json")
        assert proc.returncode == 0, f"doctor stderr:\n{proc.stderr}"
        data = json.loads(proc.stdout)
        assert isinstance(data, dict)
