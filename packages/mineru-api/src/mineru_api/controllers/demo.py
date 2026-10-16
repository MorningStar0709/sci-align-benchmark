"""Demo agent endpoints — plan, parse, upload.

Migrated from ``controlsci.api.demo_endpoints``.
"""

from __future__ import annotations

import subprocess
import sys

from fastapi import APIRouter, HTTPException

from mineru_core.paths import PROJECT_ROOT
from mineru_core.runtime import RuntimeConfig, normalize_runtime
from mineru_core.settings import get_settings

from mineru_api.schemas.demo import AgentPlanRequest, Track1ParseUrlRequest

router = APIRouter(prefix="/api/demo", tags=["demo"])

ROOT = PROJECT_ROOT


@router.post("/agent/plan")
def demo_agent_plan(req: AgentPlanRequest):
    """Generate an agent execution plan for a natural language query."""
    query = req.query
    planner = req.planner
    local_model = req.local_model
    api_provider = req.api_provider
    privacy_policy = req.privacy_policy
    runtime_config_raw = req.runtime_config

    runtime = normalize_runtime(runtime_config_raw or RuntimeConfig())
    if privacy_policy:
        runtime.privacy_policy = privacy_policy
    if local_model:
        runtime.local_model = local_model
    if api_provider:
        runtime.api_provider = api_provider

    if not query:
        raise HTTPException(status_code=400, detail={"error": "query 字段必填"})

    if planner == "deepseek" or runtime.api_provider is not None:
        live_mode = runtime.profile not in ("demo_replay",)
        command = [
            sys.executable,
            str(ROOT / "benchmark" / "agent" / "agent_cli.py"),
            "--dry-run",
            query,
        ]
        if runtime.local_model:
            command.extend(["--local", runtime.local_model])
        if runtime.api_provider:
            command.extend(["--api", runtime.api_provider])

        try:
            completed = subprocess.run(command, capture_output=True, text=True, timeout=60, encoding="utf-8", errors="replace")
            raw_stderr = completed.stderr.strip()
            raw_stdout = completed.stdout.strip()
        except subprocess.TimeoutExpired:
            raise HTTPException(status_code=504, detail={"error": "Agent CLI 超时(60s)"})
        except Exception as exc:
            raise HTTPException(status_code=502, detail={"error": str(exc)[:200]})

        stdout_lines = raw_stdout.splitlines()
        summary = "\n".join(stdout_lines[:50])

        return {
            "status": "ok",
            "query": query,
            "runtime": runtime.model_dump(),
            "command": " ".join(command),
            "summary": summary,
            "raw_stdout": raw_stdout,
            "raw_stderr": raw_stderr,
        }

    steps = [
        {"step": 1, "intent_id": "dataset_stratified_sample", "intent_name": "四维分层抽样", "tools": "core.json stratified/equidistant sampler"},
        {"step": 2, "intent_id": "model_evaluate", "intent_name": "模型答题与评分", "tools": "mineru_benchmark.eval.evaluate, judge.py"},
        {"step": 3, "intent_id": "dimension_report", "intent_name": "四维结果汇总", "tools": "leaderboard_complete.json, dimension_scores"},
    ]
    command = [sys.executable, str(ROOT / "benchmark" / "agent" / "agent_cli.py"), "--dry-run", "--local", query]
    summary_lines = "\n".join(f"    [OK] Step {s['step']}: {s['intent_id']} \u2192 {s['tools']}" for s in steps)
    summary = f"Task: {query}\nStatus: replay-ready\nSteps: {len(steps)}\n{summary_lines}"

    return {
        "status": "ok",
        "query": query,
        "runtime": runtime.model_dump(),
        "command": " ".join(command),
        "summary": summary,
        "steps": steps,
        "detected_intents": [
            {"intent_id": s["intent_id"], "description": f"{s['tools']}"}
            for s in steps
        ],
    }


@router.post("/track1/parse_url")
def track1_parse_url(req: Track1ParseUrlRequest):
    """Parse a document from a URL using MinerU official API."""
    settings = get_settings()
    if not settings.mineru_official_token:
        raise HTTPException(status_code=503, detail={"error": "MINERU_API_TOKEN 未配置"})

    import re
    if not re.match(r"^https?://", req.url or "", flags=re.IGNORECASE):
        raise HTTPException(status_code=400, detail={"error": "需要可访问的 http(s) 文件 URL"})

    try:
        import httpx
        response = httpx.post(
            f"{settings.mineru_official_base_url}/api/v4/extract/task",
            headers={"Authorization": f"Bearer {settings.mineru_official_token}", "Content-Type": "application/json"},
            json={"url": req.url, "model_version": req.model_version},
            timeout=20,
            trust_env=False,
        )
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail={"error": "MinerU 官方 API 请求超时"})
    except Exception as exc:
        raise HTTPException(status_code=502, detail={"error": str(exc)[:200]})

    try:
        data = response.json()
    except Exception:
        data = {"raw": response.text[:500]}
    if response.status_code != 200 or data.get("code") not in (0, None):
        raise HTTPException(status_code=response.status_code, detail={"error": data.get("message") or data})
    return data
