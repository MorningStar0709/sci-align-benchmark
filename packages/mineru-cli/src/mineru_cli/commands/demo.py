"""Demo service commands — migrated from controlsci.cli.demo."""

from __future__ import annotations

import re
import socket
import subprocess
import sys
from pathlib import Path
from typing import Annotated

import typer

from mineru_core.settings import get_settings
from mineru_cli.commands.common import PROJECT_ROOT, console, fail, print_json


app = typer.Typer(help="启动和检查本地工作台服务。")
API_LOG = PROJECT_ROOT / "controlsci-api-17001.log"
FRONTEND_LOG = PROJECT_ROOT / "cloud-demo-next.log"


def _port_open(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=0.5):
            return True
    except OSError:
        return False


@app.command("status")
def status(
    json_output: Annotated[bool, typer.Option("--json", help="输出 JSON。")] = False,
    output: Annotated[Path | None, typer.Option("--output", "-o", help="将 JSON 以 UTF-8 写入文件。")] = None,
) -> None:
    settings = get_settings()
    payload = {
        "api": {"url": settings.api_base_url, "open": _port_open(settings.api_host, settings.api_port)},
        "frontend": {"url": "http://127.0.0.1:3000", "open": _port_open("127.0.0.1", 3000)},
        "python": {"path": _resolve_python(settings.python_path), "configured": bool(settings.python_path)},
    }
    if json_output or output:
        print_json(payload, output=output)
        return
    console.print(f"API      {payload['api']['url']}      {'open' if payload['api']['open'] else 'closed'}")
    console.print(f"Frontend {payload['frontend']['url']} {'open' if payload['frontend']['open'] else 'closed'}")


@app.command("start")
def start(
    api_only: Annotated[bool, typer.Option("--api-only", help="只启动 FastAPI。")] = False,
    frontend_only: Annotated[bool, typer.Option("--frontend-only", help="只启动前端。")] = False,
    python_path: Annotated[str, typer.Option("--python", help="启动 FastAPI 使用的 Python。默认优先 CONTROLMIND_PYTHON / config / myenv。")] = "",
) -> None:
    settings = get_settings()
    if api_only and frontend_only:
        fail("--api-only 和 --frontend-only 不能同时使用")

    backend_python = _resolve_python(python_path or settings.python_path, strict=bool(python_path or settings.python_path))
    if not frontend_only and not backend_python:
        fail(
            "未找到可用于启动 FastAPI 的 Python。请设置 "
            "mineru config set python.path <path-to-python>"
        )

    if not frontend_only and not _port_open(settings.api_host, settings.api_port):
        subprocess.Popen(
            [backend_python, "-m", "mineru_api", "--host", settings.api_host, "--port", str(settings.api_port)],
            cwd=str(PROJECT_ROOT),
            stdout=API_LOG.open("a", encoding="utf-8"),
            stderr=subprocess.STDOUT,
        )
        console.print(f"[green]已启动 API[/green] {settings.api_base_url}")
        console.print(f"[dim]Python: {backend_python}[/dim]")
    elif not frontend_only:
        console.print(f"[yellow]API 已在运行[/yellow] {settings.api_base_url}")

    if not api_only and not _port_open("127.0.0.1", 3000):
        subprocess.Popen(
            ["npm", "run", "dev"],
            cwd=str(PROJECT_ROOT / "starboard"),
            stdout=FRONTEND_LOG.open("a", encoding="utf-8"),
            stderr=subprocess.STDOUT,
        )
        console.print("[green]已启动前端[/green] http://127.0.0.1:3000")
    elif not api_only:
        console.print("[yellow]前端已在运行[/yellow] http://127.0.0.1:3000")


@app.command("restart")
def restart(
    api_only: Annotated[bool, typer.Option("--api-only", help="只重启 FastAPI。")] = False,
    frontend_only: Annotated[bool, typer.Option("--frontend-only", help="只重启前端。")] = False,
    python_path: Annotated[str, typer.Option("--python", help="启动 FastAPI 使用的 Python。")] = "",
) -> None:
    settings = get_settings()
    if api_only and frontend_only:
        fail("--api-only 和 --frontend-only 不能同时使用")
    if not frontend_only:
        _stop_port(settings.api_host, settings.api_port)
    if not api_only:
        _stop_port("127.0.0.1", 3000)
    start(api_only=api_only, frontend_only=frontend_only, python_path=python_path)


def _resolve_python(configured: str = "", *, strict: bool = False) -> str:
    if configured:
        configured_path = Path(configured)
        if configured_path.exists():
            return str(configured_path)
        if strict:
            fail(f"指定的 Python 不存在：{configured}")
    candidates = [sys.executable]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return str(Path(candidate))
    return ""


_HOST_RE = re.compile(r"^(?:\d{1,3}\.){3}\d{1,3}$|^localhost$|^[a-zA-Z0-9][a-zA-Z0-9.\-]{0,253}[a-zA-Z0-9]$")


def _validate_host_port(host: str, port: int) -> None:
    """Reject host/port values that could be injected into shell commands."""
    if not _HOST_RE.match(host):
        fail(f"不合法的 host: {host!r}")
    if not (1 <= port <= 65535):
        fail(f"不合法的 port: {port}")


def _stop_port(host: str, port: int) -> None:
    _validate_host_port(host, port)
    if not _port_open(host, port):
        console.print(f"[dim]{host}:{port} 未运行[/dim]")
        return
    import psutil

    def _is_target(conn) -> bool:
        if conn.status != psutil.CONN_LISTEN or conn.lport != port or not conn.pid:
            return False
        laddr = conn.laddr.ip if conn.laddr else ""
        # 监听在通配地址上的进程同时服务 127.0.0.1，不能因 host 字面不等而漏杀
        return laddr == host or laddr in {"0.0.0.0", "::"}

    try:
        procs = [psutil.Process(c.pid) for c in psutil.net_connections(kind="inet") if _is_target(c)]
    except psutil.AccessDenied:
        fail(f"无权查询 {host}:{port} 的监听进程所属用户；请手动停止服务后再 start。")
        return
    except psutil.NoSuchProcess:
        procs = []

    if not procs:
        console.print(f"[yellow]未找到 {host}:{port} 的监听进程[/yellow]")
        return
    for proc in procs:
        try:
            proc.terminate()
        except psutil.NoSuchProcess:
            continue

    _, alive = psutil.wait_procs(procs, timeout=15)
    for proc in alive:
        try:
            proc.kill()
        except psutil.NoSuchProcess:
            continue
    _, still = psutil.wait_procs(alive, timeout=5)
    if still:
        fail(f"{host}:{port} 的进程未能停止（pid={[p.pid for p in still]}），端口仍被占用")
    console.print(
        f"[green]已停止[/green] {host}:{port} pid={','.join(str(p.pid) for p in procs)}"
    )
