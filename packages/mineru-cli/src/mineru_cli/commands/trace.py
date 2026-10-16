"""Track trace and report artifact commands — migrated from controlsci.cli.trace."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from mineru_cli.commands.common import PROJECT_ROOT, console, current_python, print_json, run_command


app = typer.Typer(help="DATA-TRACE、验收包和报告状态。")


@app.command("bundle")
def bundle(
    rebuild: Annotated[bool, typer.Option("--rebuild", help="重新构建 data_trace_bundle。")] = False,
    json_output: Annotated[bool, typer.Option("--json", help="输出 JSON。")] = False,
    output: Annotated[Path | None, typer.Option("--output", "-o", help="将 JSON 以 UTF-8 写入文件。")] = None,
) -> None:
    bundle_dir = PROJECT_ROOT / "docs" / "submissions" / "data_trace_bundle"
    manifest = bundle_dir / "manifest.json"
    if rebuild:
        run_command([current_python(), "scripts/build_data_trace_bundle.py"], cwd=PROJECT_ROOT)
    payload = {"bundle_dir": str(bundle_dir), "manifest": str(manifest), "available": manifest.exists()}
    if json_output or output:
        print_json(payload, output=output)
        return
    console.print(f"DATA-TRACE bundle: {'可用' if manifest.exists() else '缺失'}")
    console.print(str(bundle_dir))
