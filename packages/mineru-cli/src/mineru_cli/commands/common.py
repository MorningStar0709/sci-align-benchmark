"""Shared helpers for the MinerU CLI — migrated from controlsci.cli.common."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from mineru_core.paths import PROJECT_ROOT


def configure_utf8_stdio() -> None:
    """强制 UTF-8 stdio，保证 CLI JSON 在管道/重定向下不因 locale 而变。"""
    os.environ.setdefault("PYTHONUTF8", "1")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


configure_utf8_stdio()

console = Console()
err_console = Console(stderr=True)

SECRET_KEYS = ("token", "key", "secret", "password", "code")


def mask_secrets(data: Any) -> Any:
    if isinstance(data, dict):
        masked = {}
        for key, value in data.items():
            if any(part in str(key).lower() for part in SECRET_KEYS):
                masked[key] = "***" if value else ""
            else:
                masked[key] = mask_secrets(value)
        return masked
    if isinstance(data, list):
        return [mask_secrets(item) for item in data]
    return data


def print_json(data: Any, output: Path | None = None) -> None:
    text = json.dumps(data, ensure_ascii=False, indent=2) + "\n"
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
        return
    sys.stdout.write(text)


def fail(message: str, code: int = 1) -> None:
    err_console.print(f"[red]错误：[/red]{message}")
    raise typer.Exit(code)


def read_json(path: Path) -> Any:
    if not path.exists():
        fail(f"文件不存在：{path}")
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass


def run_command(args: list[str], *, cwd: Path | None = None, check: bool = True) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(args, cwd=str(cwd or PROJECT_ROOT), check=check)
    except FileNotFoundError:
        fail(f"命令不可用：{args[0]}")
    except subprocess.CalledProcessError as exc:
        fail(f"命令执行失败：{' '.join(args)}，退出码 {exc.returncode}", exc.returncode)
    raise AssertionError("unreachable")


def current_python() -> str:
    return sys.executable


def env_with_profile(profile: str | None = None) -> dict[str, str]:
    env = os.environ.copy()
    if profile:
        env["CONTROLMIND_PROFILE"] = profile
        env["CONTROLSCI_API_PROFILE"] = profile
    return env


def render_kv_panel(title: str, items: dict[str, Any]) -> None:
    table = Table(show_header=False, box=None, padding=(0, 1))
    table.add_column("Key", style="cyan")
    table.add_column("Value", style="white")
    for key, value in items.items():
        table.add_row(str(key), str(value))
    console.print(Panel(table, title=title, border_style="cyan"))


def status_label(ok: bool) -> str:
    return "[green]OK[/green]" if ok else "[red]FAIL[/red]"


def call_quiet(func, *args, quiet: bool = False, **kwargs):
    if not quiet:
        return func(*args, **kwargs)
    with redirect_stdout(sys.stderr):
        return func(*args, **kwargs)
