"""Cloud demo helpers — URL validation, access code, boundary checks.

Migrated from ``cloud_demo_standalone.app``.
"""

from __future__ import annotations

import ipaddress
import os
import socket
from pathlib import Path
from urllib.parse import unquote, urlparse

from fastapi import HTTPException
from mineru_core.settings import PROVIDERS


OFFICIAL_URL_EXTENSIONS = {
    ".pdf", ".doc", ".docx", ".ppt", ".pptx",
    ".png", ".jpg", ".jpeg", ".jp2", ".webp", ".gif", ".bmp",
    ".html",
}


def format_extensions(extensions: set[str]) -> str:
    return "/".join(ext.lstrip(".").upper() for ext in sorted(extensions))


def is_supported_official_url(raw_url: str) -> bool:
    parsed = urlparse(raw_url or "")
    suffix = Path(unquote(parsed.path or "")).suffix.lower()
    if suffix in OFFICIAL_URL_EXTENSIONS:
        return True
    query = unquote(parsed.query or "").lower()
    return any(ext in query for ext in OFFICIAL_URL_EXTENSIONS)


def _is_public_ip(value: str) -> bool:
    try:
        ip = ipaddress.ip_address(value)
    except ValueError:
        return False
    return ip.is_global


def validate_public_extract_url(raw_url: str) -> str:
    url = (raw_url or "").strip()
    parsed = urlparse(url)
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname:
        raise HTTPException(status_code=400, detail={"error": "只接受公开 http(s) 文档 URL"})

    if not is_supported_official_url(url):
        raise HTTPException(
            status_code=400,
            detail={
                "error": f"URL 必须指向 MinerU 官方 URL API 支持的文件格式：{format_extensions(OFFICIAL_URL_EXTENSIONS)}",
            },
        )

    host = parsed.hostname
    try:
        literal_ip = ipaddress.ip_address(host)
        if not literal_ip.is_global:
            raise HTTPException(status_code=400, detail={"error": "拒绝内网、回环或保留地址 URL"})
    except ValueError:
        try:
            infos = socket.getaddrinfo(host, None)
        except socket.gaierror:
            raise HTTPException(status_code=400, detail={"error": "URL 域名无法解析"})
        resolved_ips = {info[4][0] for info in infos}
        if not resolved_ips or not all(_is_public_ip(ip) for ip in resolved_ips):
            raise HTTPException(status_code=400, detail={"error": "拒绝解析到内网、回环或保留地址的 URL"})
    return url


def env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def cloud_demo_settings() -> dict:
    return {
        "demo_access_code": os.environ.get("DEMO_ACCESS_CODE", "").strip(),
        "daily_limit": _env_int("DEMO_DAILY_LIMIT", 100),
        "upload_max_mb": _env_int("DEMO_UPLOAD_MAX_MB", 20),
        "require_access_code": env_bool("DEMO_REQUIRE_ACCESS_CODE", True),
        "allow_dry_run_without_code": env_bool("DEMO_ALLOW_DRY_RUN_WITHOUT_CODE", False),
        "mineru_token": os.environ.get("MINERU_API_TOKEN", os.environ.get("MINERU_OFFICIAL_TOKEN", "")).strip(),
        "mineru_base_url": os.environ.get("MINERU_OFFICIAL_BASE_URL", "https://mineru.net").rstrip("/"),
        "deepseek_key": os.environ.get("DEEPSEEK_API_KEY", os.environ.get("OPENAI_API_KEY", "")).strip(),
        "deepseek_base": os.environ.get("DEEPSEEK_API_BASE", PROVIDERS["deepseek"].base_url).rstrip("/"),
        "deepseek_model": os.environ.get("DEEPSEEK_MODEL", PROVIDERS["deepseek"].default_model),
    }


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except (ValueError, TypeError):
        return default


def _missing_keys_summary(cfg: dict) -> str:
    """Build a short hint about which cloud API keys are missing."""
    missing = []
    if not cfg["mineru_token"]:
        missing.append("MINERU_API_TOKEN")
    if not cfg["deepseek_key"]:
        missing.append("DEEPSEEK_API_KEY")
    return ("云端 key 状态: " + ", ".join(f"{k} 未配置" for k in missing)) if missing else "云端 key 均已配置"


def require_access(code: str | None, *, allow_unconfigured: bool = False) -> None:
    cfg = cloud_demo_settings()
    expected = cfg["demo_access_code"]
    unconfigured_dry_run_allowed = allow_unconfigured and cfg["allow_dry_run_without_code"]
    if cfg["require_access_code"] and not expected and not allow_unconfigured:
        raise HTTPException(
            status_code=503,
            detail={"error": f"DEMO_ACCESS_CODE 未配置，云端实时调用已锁定。{_missing_keys_summary(cfg)}"},
        )
    if cfg["require_access_code"] and not expected and allow_unconfigured and not unconfigured_dry_run_allowed:
        raise HTTPException(
            status_code=503,
            detail={
                "error": f"DEMO_ACCESS_CODE 未配置；如需本地 dry-run 验证，请设置 DEMO_ALLOW_DRY_RUN_WITHOUT_CODE=1。{_missing_keys_summary(cfg)}",
            },
        )
    if expected and (code or "").strip() != expected:
        raise HTTPException(status_code=403, detail={"error": "Demo access code required"})


def public_boundary() -> dict:
    return {
        "profile": "pure_cloud_demo",
        "allowed": ["public_document_url", "public_pdf_upload", "desensitized_question"],
        "supported_url_formats": format_extensions(OFFICIAL_URL_EXTENSIONS),
        "forbidden": [
            "patient_material",
            "private_enterprise_document",
            "confidential_document",
            "private_vector_index",
            "local_gpu_task",
            "frontend_api_key",
        ],
        "services": ["MinerU official API", "DeepSeek API"],
    }


def is_personal_medical_question(question: str) -> bool:
    lowered = question.lower()
    patterns = [
        "我该", "我要不要", "帮我诊断", "怎么吃药", "吃多少", "能不能停药",
        "胸痛", "呼吸困难", "my dose", "diagnose me", "should i take",
        "chest pain", "emergency",
    ]
    return any(pattern in lowered for pattern in patterns)
