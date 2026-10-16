"""DeepSeek API client — call chat completions with quota tracking.

Migrated from ``cloud_demo_standalone.app``.
"""

from __future__ import annotations

import httpx
from fastapi import HTTPException

from mineru_api.services.cloud_demo import cloud_demo_settings
from mineru_api.services.quota import consume_quota


def call_deepseek(
    messages: list[dict[str, str]],
    reason: str,
    *,
    temperature: float = 0.2,
) -> tuple[str, dict | None, dict]:
    """Call DeepSeek chat API and return (answer, usage, quota_after)."""
    cfg = cloud_demo_settings()
    if not cfg["deepseek_key"]:
        raise HTTPException(status_code=503, detail={"error": "DEEPSEEK_API_KEY 未配置"})
    quota_after = consume_quota(reason)
    try:
        with httpx.Client(timeout=60, trust_env=False) as client:
            resp = client.post(
                f"{cfg['deepseek_base']}/chat/completions",
                headers={
                    "Authorization": f"Bearer {cfg['deepseek_key']}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": cfg["deepseek_model"],
                    "messages": messages,
                    "temperature": temperature,
                },
            )
            data = resp.json() if resp.text else {}
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail={"error": "DeepSeek API 请求超时"})
    except Exception as exc:
        raise HTTPException(status_code=502, detail={"error": f"DeepSeek API 调用失败: {str(exc)[:180]}"})
    if resp.status_code >= 400:
        raise HTTPException(status_code=resp.status_code, detail={"error": data})
    answer = (((data.get("choices") or [{}])[0]).get("message") or {}).get("content", "")
    return answer, data.get("usage"), quota_after
