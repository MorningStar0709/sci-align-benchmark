"""End-to-end acceptance commands — migrated from controlsci.cli.run."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.table import Table

from mineru_cli.commands.common import console, print_json


app = typer.Typer(help="运行 MinerU 统一验收协议。")


@app.command("acceptance")
def acceptance(
    json_output: Annotated[bool, typer.Option("--json", help="输出 JSON。")] = False,
    output: Annotated[Path | None, typer.Option("--output", "-o", help="将 JSON 以 UTF-8 写入文件。")] = None,
) -> None:
    from mineru_api.controllers.health import demo_health
    from mineru_api.schemas.demo import Track1ValidateChainRequest, Track2ActionRequest
    from mineru_api.services.acceptance import track1_validate_chain, track2_artifact_check
    from mineru_core.runtime import RuntimeConfig
    from mineru_api.controllers.medical_rag import health as rag_health
    from mineru_cli.commands.track2 import ARTIFACTS

    checks = []

    demo = demo_health()
    checks.append({"name": "health", "status": demo.get("status"), "ok": demo.get("status") in {"ready", "degraded"}})

    track1 = track1_validate_chain(Track1ValidateChainRequest(sample=4, runtime_config=RuntimeConfig(profile="demo_replay")))
    checks.append({"name": "track1.validate", "status": track1.get("status"), "ok": track1.get("status") == "ok"})

    track2_results = [
        track2_artifact_check(Track2ActionRequest(query=f"核验 {artifact}", artifact_kind=artifact, runtime_config=RuntimeConfig()))
        for artifact in ARTIFACTS
    ]
    track2_ok = all(item.get("status") == "ok" for item in track2_results)
    checks.append({"name": "track2.validate_all", "status": "ok" if track2_ok else "degraded", "ok": track2_ok})

    rag = rag_health()
    checks.append({"name": "track3.index", "status": rag.get("status"), "ok": rag.get("status") == "ok"})

    payload = {
        "status": "ok" if all(check["ok"] for check in checks) else "degraded",
        "checks": checks,
        "details": {
            "health": demo,
            "track1": {"chain_id": track1.get("chain_id"), "overall_score": track1.get("score", {}).get("overall_score")},
            "track2": [{"artifact_kind": item.get("artifact_kind"), "status": item.get("status")} for item in track2_results],
            "track3_index": rag,
        },
    }
    if json_output or output:
        print_json(payload, output=output)
        return

    table = Table(title="MinerU Acceptance")
    table.add_column("检查项", style="cyan")
    table.add_column("状态")
    for check in checks:
        table.add_row(check["name"], "[green]OK[/green]" if check["ok"] else f"[yellow]{check['status']}[/yellow]")
    console.print(table)
    console.print(f"整体状态：{payload['status']}")
