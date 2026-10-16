"""Health and doctor commands — migrated from controlsci.cli.health."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.table import Table

from mineru_cli.commands.common import console, print_json, status_label


app = typer.Typer(help="检查运行服务、索引和验收产物。", invoke_without_command=True)


@app.callback(invoke_without_command=True)
def health_default(
    ctx: typer.Context,
    json_output: Annotated[bool, typer.Option("--json", help="输出 JSON。")] = False,
    output: Annotated[Path | None, typer.Option("--output", "-o", help="将 JSON 以 UTF-8 写入文件。")] = None,
) -> None:
    if ctx.invoked_subcommand is None:
        run_health(json_output=json_output, output=output)
        raise typer.Exit()


@app.command("run")
def run_health(
    json_output: Annotated[bool, typer.Option("--json", help="输出 JSON。")] = False,
    output: Annotated[Path | None, typer.Option("--output", "-o", help="将 JSON 以 UTF-8 写入文件。")] = None,
) -> None:
    from mineru_api.controllers.health import demo_health

    payload = demo_health()
    if json_output or output:
        print_json(payload, output=output)
        return
    table = Table(title="MinerU Health", show_lines=False)
    table.add_column("项目", style="cyan")
    table.add_column("状态")
    table.add_column("说明", style="white")
    for name, comp in payload.get("components", {}).items():
        table.add_row(name, status_label(bool(comp.get("available"))), comp.get("detail") or comp.get("path") or comp.get("url") or "")
    console.print(table)
    if payload.get("issues"):
        console.print("[yellow]提醒：[/yellow]" + "；".join(payload["issues"]))
    console.print(f"整体状态：{payload.get('status')}")


@app.command("doctor")
def doctor(
    json_output: Annotated[bool, typer.Option("--json", help="输出 JSON。")] = False,
    output: Annotated[Path | None, typer.Option("--output", "-o", help="将 JSON 以 UTF-8 写入文件。")] = None,
) -> None:
    from mineru_core.settings import get_settings
    from mineru_api.controllers.health import demo_health
    from mineru_api.controllers.medical_rag import health as rag_health
    from mineru_cli.commands.demo import _port_open, _resolve_python

    demo = demo_health()
    rag = rag_health()
    settings = get_settings()
    python_path = _resolve_python(settings.python_path)
    payload = {
        "python": {"path": python_path, "available": bool(python_path)},
        "frontend": {"url": "http://127.0.0.1:3000", "available": _port_open("127.0.0.1", 3000)},
        "api": {"url": settings.api_base_url, "available": _port_open(settings.api_host, settings.api_port)},
        "demo_health": demo,
        "medical_rag_health": rag,
    }
    if json_output or output:
        print_json(payload, output=output)
        return
    table = Table(title="MinerU Doctor")
    table.add_column("项目", style="cyan")
    table.add_column("状态")
    table.add_column("说明")
    table.add_row("Python", status_label(bool(python_path)), python_path or "未找到 myenv/config Python")
    table.add_row("Frontend", status_label(payload["frontend"]["available"]), payload["frontend"]["url"])
    table.add_row("FastAPI", status_label(payload["api"]["available"]), payload["api"]["url"])
    for name, comp in demo.get("components", {}).items():
        table.add_row(name, status_label(bool(comp.get("available"))), comp.get("detail") or comp.get("path") or comp.get("url") or "")
    console.print(table)
    for issue in demo.get("issues", []) + rag.get("issues", []):
        console.print(f"[yellow]- {issue}[/yellow]")
