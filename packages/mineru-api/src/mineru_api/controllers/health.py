"""Health and quota endpoints.

Migrated from ``controlsci.api.demo_endpoints`` (routes /api/demo/health, /api/demo/quota).
"""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter

from mineru_core.settings import get_settings

from mineru_api.services.demo_utils import (
    docker_ok,
    file_ok,
    local_llm_tags,
    public_medical_evidence_ok,
    read_json,
)
from mineru_api.services.mineru_client import MinerUClient

router = APIRouter(tags=["health"])


@router.get("/api/demo/health")
def demo_health():
    settings = get_settings()
    mineru_api_available = MinerUClient(settings.mineru_base_url).is_available()
    llm_ok, llm_models = local_llm_tags()
    evidence_bundle = file_ok("docs", "submissions", "data_trace_bundle")

    components = {
        "mineru_local_api": {"available": mineru_api_available},
        "local_llm": {"available": llm_ok, "models": llm_models if llm_ok else []},
        "docker": {"available": docker_ok()},
        "public_medical_evidence_bundle": {"available": evidence_bundle, "checks": read_json("docs/submissions/data_trace_bundle/09_medical_rag/medical_rag_eval.json").get("_checks") if evidence_bundle else None},
        "public_medical_evidence_detail": {"available": public_medical_evidence_ok()},
        "mineru_official_api": {
            "available": bool(settings.mineru_official_token),
            "base_url": settings.mineru_official_base_url,
            "configured": bool(settings.mineru_official_token),
        },
        "api_providers": {"available": True},
    }
    issues = []
    if not mineru_api_available:
        issues.append("MinerU 本地 API 无响应，请确认 mineru_api 已启动")
    if not llm_ok:
        issues.append("llama.cpp 服务无响应，请确认已启动")
    if not evidence_bundle:
        issues.append("公开医疗证据包不存在，请运行构建脚本")
    if not settings.mineru_official_token:
        issues.append("MINERU_API_TOKEN 未配置，官方 API 不可用")

    return {
        "status": "ok" if not issues else "degraded",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "profile": settings.cloud_demo_profile,
        "components": components,
        "issues": issues,
    }


@router.get("/api/demo/quota")
def demo_quota():
    from mineru_api.services.quota import quota_status
    return quota_status()


@router.get("/api/health")
def root_health():
    from mineru_api.schemas.medical import HealthResponse
    from mineru_api.controllers.medical_rag import health as rag_health
    payload = rag_health()
    return HealthResponse(**payload)
