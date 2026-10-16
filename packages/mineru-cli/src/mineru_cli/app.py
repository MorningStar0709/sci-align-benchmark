"""MinerU CLI entrypoint — migrated from controlsci.cli.app."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from mineru_cli.commands import config, demo, health, run, track1, track2, track3, trace
from mineru_cli.commands.common import console


app = typer.Typer(
    name="mineru",
    help="MinerU 科学文档评测与医学 RAG 工具链。",
    invoke_without_command=True,
    no_args_is_help=False,
)

LOGO = r"""
   ______            __             ____  ___           __
  / ____/___  ____  / /__________  / /  |/  /___  ____/ /
 / /   / __ \/ __ \/ __/ ___/ __ \/ / /|_/ / __ \/ __  /
/ /___/ /_/ / / / / /_/ /  / /_/ / / /  / / /_/ / /_/ /
\____/\____/_/ /_/\__/_/   \____/_/_/  /_/\____/\__,_/
"""


def print_logo() -> None:
    """Render the interactive CLI brand header."""
    console.print(LOGO, style="bold cyan")
    console.print("[bold]Scientific Document Evaluation Toolkit[/bold]")
    console.print("[dim]Local-first · Reproducible · Medical RAG[/dim]")
    console.print()
    console.print("Run [cyan]mineru --help[/cyan] to see commands.")


@app.callback()
def cli(ctx: typer.Context) -> None:
    """MinerU command line interface."""
    if ctx.invoked_subcommand is None:
        print_logo()
        raise typer.Exit()


app.add_typer(health.app, name="health")
app.add_typer(config.app, name="config")
app.add_typer(demo.app, name="demo")
app.add_typer(track1.app, name="track1")
app.add_typer(track2.app, name="track2")
app.add_typer(track3.app, name="track3")
app.add_typer(trace.app, name="trace")
app.add_typer(run.app, name="run")


@app.command("doctor")
def doctor(
    json_output: Annotated[bool, typer.Option("--json", help="输出 JSON。")] = False,
    output: Annotated[Path | None, typer.Option("--output", "-o", help="将 JSON 以 UTF-8 写入文件。")] = None,
) -> None:
    """运行完整环境诊断。"""
    health.doctor(json_output=json_output, output=output)


@app.command("logo")
def logo() -> None:
    """显示 MinerU 文字 Logo。"""
    print_logo()


def main() -> None:
    app()
