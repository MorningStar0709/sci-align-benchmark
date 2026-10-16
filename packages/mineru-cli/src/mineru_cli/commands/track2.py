"""Track2 Agent workflow commands — migrated from controlsci.cli.track2."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from mineru_cli.commands.common import console, print_json


app = typer.Typer(help="Agent 工作流规划、验收与来源产物核验。")


ARTIFACTS = [
    "flywheel",
    "eval40",
    "check_index",
    "evidence_bundle",
    "visual_audit",
    "router_robustness",
    "failure_recovery",
    "source_selection_ablation",
    "resource_pareto",
    "hard_document_stress",
    "sciverse_source_routing",
]


@app.command("plan")
def plan(
    query: Annotated[str, typer.Argument(help="自然语言任务目标。")],
    json_output: Annotated[bool, typer.Option("--json", help="输出 JSON。")] = False,
    output: Annotated[Path | None, typer.Option("--output", "-o", help="将 JSON 以 UTF-8 写入文件。")] = None,
) -> None:
    from mineru_api.schemas.demo import AgentPlanRequest
    from mineru_api.services.acceptance import track2_validate_chain
    from mineru_core.runtime import RuntimeConfig

    payload = track2_validate_chain(AgentPlanRequest(query=query, runtime_config=RuntimeConfig()))
    if json_output or output:
        print_json(payload, output=output)
        return
    console.print(payload.get("summary", ""))


@app.command("validate")
def validate(
    artifact: Annotated[str, typer.Option("--artifact", "-a", help="产物类型，或 all。")] = "all",
    json_output: Annotated[bool, typer.Option("--json", help="输出 JSON。")] = False,
    output_path: Annotated[Path | None, typer.Option("--output", "-o", help="将 JSON 以 UTF-8 写入文件。")] = None,
) -> None:
    from mineru_api.schemas.demo import Track2ActionRequest
    from mineru_api.services.acceptance import track2_artifact_check
    from mineru_core.runtime import RuntimeConfig

    allowed = ["all", *ARTIFACTS]
    if artifact not in allowed:
        allowed_text = ", ".join(allowed)
        raise typer.BadParameter(f"未知 artifact：{artifact}。可选值：{allowed_text}")

    selected = ARTIFACTS if artifact == "all" else [artifact]
    results = []
    for item in selected:
        payload = track2_artifact_check(
            Track2ActionRequest(query=f"核验 {item}", artifact_kind=item, runtime_config=RuntimeConfig())
        )
        results.append(payload)
    output = {"status": "ok" if all(r.get("status") == "ok" for r in results) else "degraded", "results": results}
    if json_output or output_path:
        print_json(output, output=output_path)
        return
    for result in results:
        console.print(f"[bold cyan]{result['artifact_kind']}[/bold cyan] {result['status']}")
        console.print(result.get("validation_summary", ""))
