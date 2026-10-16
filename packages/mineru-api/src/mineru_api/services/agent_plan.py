"""Agent plan helpers — agent execution plan generation.

Extracted from ``controlsci.api.demo_endpoints``.
"""

from __future__ import annotations

import sys

from mineru_core.paths import PROJECT_ROOT
from mineru_core.runtime import RuntimeConfig


def track2_acceptance_steps() -> list[dict]:
    """Return the Track 2 acceptance evaluation plan steps."""
    return [
        {
            "step": 1,
            "intent_id": "intent_router",
            "intent_name": "意图识别",
            "resource": "script",
            "tools": "agent_capabilities.json, agent_cli.py --dry-run",
            "status": "validated_artifact",
        },
        {
            "step": 2,
            "intent_id": "dag_build",
            "intent_name": "DAG 生成",
            "resource": "script",
            "tools": "intent dependencies, ordered execution plan",
            "status": "validated_artifact",
        },
        {
            "step": 3,
            "intent_id": "resource_schedule",
            "intent_name": "资源调度",
            "resource": "local_gpu",
            "tools": "resource_scheduler.py, local/API fallback policy",
            "status": "validated_artifact",
        },
        {
            "step": 4,
            "intent_id": "execution_summary",
            "intent_name": "执行摘要",
            "resource": "script",
            "tools": "dry-run summary, reproducible command",
            "status": "validated_artifact",
        },
        {
            "step": 5,
            "intent_id": "artifact_trace",
            "intent_name": "来源与产物核验",
            "resource": "script",
            "tools": "track2_agent_report.md, agent logs, agent source files",
            "status": "validated_artifact",
        },
    ]


def replay_plan_response(
    query_text: str,
    runtime: RuntimeConfig,
    steps: list[dict],
    raw_stderr: str,
) -> dict:
    """Build a dry-run plan response from a list of steps."""
    detected = [
        {"intent_id": s["intent_id"], "description": f"{s['intent_name']}：{s['tools']}"}
        for s in steps
    ]
    summary_lines = "\n".join(
        f"    [OK] Step {s['step']}: {s['intent_id']} → {s['tools']}"
        for s in steps
    )

    from mineru_core.settings import get_settings

    settings = get_settings()
    command = [
        sys.executable,
        str(PROJECT_ROOT / "benchmark" / "agent" / "agent_cli.py"),
        "--dry-run",
        "--local",
        query_text,
    ]

    summary = f"""Task: {query_text}
Status: replay-ready
Steps: {len(steps)}
Intents: {[s['intent_id'] for s in steps]}
{summary_lines}
Agent CLI: {' '.join(command)}
Stderr: {raw_stderr[:500] if raw_stderr else '(none)'}"""

    return {
        "status": "ok",
        "query": query_text,
        "runtime": runtime.model_dump(),
        "steps": steps,
        "summary": summary,
        "detected_intents": detected,
        "raw_stderr": raw_stderr,
    }


__all__ = [
    "track2_acceptance_steps",
    "replay_plan_response",
]
