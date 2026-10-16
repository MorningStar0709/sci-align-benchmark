"""MinerU-core — 轻量 LLM HTTP 客户端。

提供最小化的 OpenAI-compatible API 调用能力，供 pipeline 等包使用，
避免它们直接依赖 mineru-benchmark 的 LLM 调用层。
"""

from __future__ import annotations

import json
import time
from typing import Any

from mineru_core.settings import PROVIDERS


def call_openai_direct(
    api_key: str,
    system_prompt: str,
    user_prompt: str,
    model: str = PROVIDERS["deepseek"].default_model,
    base_url: str = PROVIDERS["deepseek"].base_url,
    timeout: int = 120,
    max_retries: int = 3,
    max_tokens: int = 8192,
) -> dict[str, Any] | None:
    """调用 OpenAI-compatible API，带重试。

    这是一个最小化实现，不包含 JSON 校验、诊断日志等 benchmark 专用特性。
    如需高级功能（provider routing、diagnostic logging、JSON validation），
    请直接使用 ``mineru_benchmark.generator.api``。

    Args:
        api_key: API 密钥
        system_prompt: 系统提示词
        user_prompt: 用户提示词
        model: 模型名称
        base_url: API base URL
        timeout: 单次请求超时（秒）
        max_retries: 最大重试次数
        max_tokens: 最大生成 token 数

    Returns:
        解析后的 JSON 对象，或 None（所有重试失败后）
    """
    import httpx

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.1,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
    }

    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            with httpx.Client(timeout=timeout) as client:
                resp = client.post(
                    f"{base_url.rstrip('/')}/chat/completions",
                    headers=headers,
                    json=payload,
                )
                resp.raise_for_status()
                data = resp.json()
                content = data["choices"][0]["message"]["content"]
                if content:
                    return json.loads(content)
                return None
        except Exception as e:
            last_error = e
            status = getattr(e, "response", None)
            status_code = getattr(status, "status_code", None) if status else None
            if status_code == 429:
                print(
                    f"[llm_client] 429 rate limited ({model}), "
                    f"retry {attempt + 1}/{max_retries}",
                    flush=True,
                )
            else:
                print(
                    f"[llm_client] Error ({model}, attempt {attempt + 1}/{max_retries}): "
                    f"{type(e).__name__}: {e}",
                    flush=True,
                )
            if attempt < max_retries - 1:
                time.sleep(2**attempt)

    print(
        f"[llm_client] All {max_retries} attempts failed ({model}). "
        f"Last error: {last_error}",
        flush=True,
    )
    return None


__all__ = ["call_openai_direct"]
