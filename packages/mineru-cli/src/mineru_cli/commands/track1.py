"""Track1 Sci-Align commands — migrated from controlsci.cli.track1."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from mineru_cli.commands.common import console, print_json


app = typer.Typer(help="赛道一解析、出题、评分与验收。")


@app.command("validate")
def validate(
    sample: Annotated[int, typer.Option("--sample", help="验收样例数量。")] = 5,
    json_output: Annotated[bool, typer.Option("--json", help="输出 JSON。")] = False,
    output: Annotated[Path | None, typer.Option("--output", "-o", help="将 JSON 以 UTF-8 写入文件。")] = None,
) -> None:
    from mineru_api.schemas.demo import Track1ValidateChainRequest
    from mineru_api.services.acceptance import track1_validate_chain
    from mineru_core.runtime import RuntimeConfig

    payload = track1_validate_chain(
        Track1ValidateChainRequest(sample=sample, runtime_config=RuntimeConfig(profile="demo_replay"))
    )
    if json_output or output:
        print_json(payload, output=output)
        return
    console.print(f"Track1 验收状态：{payload.get('status')}")
    question_items = payload.get("questions", {}).get("questions", [])
    for item in question_items[:sample]:
        console.print(f"- {item.get('question') or item.get('prompt') or item.get('id')}")
    score = payload.get("score", {})
    if score:
        console.print(f"[dim]overall_score={score.get('overall_score')} · source={score.get('source')}[/dim]")
