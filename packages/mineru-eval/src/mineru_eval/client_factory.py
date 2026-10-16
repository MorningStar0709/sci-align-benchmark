"""Client factory for LLM API calls — supports local (llama.cpp) and remote providers."""

from __future__ import annotations

import os

import mineru_core  # noqa: F401 — trigger _load_dotenv() to load .env API keys

import httpx


def _resolve_by_provider(provider: str, fallback_key=None) -> str:
    """Resolve API key by provider configuration."""
    from mineru_core.settings import get_provider
    cfg = get_provider(provider)
    for key in cfg.api_key_env.split("|"):
        val = os.environ.get(key.strip())
        if val:
            return val
    return fallback_key


def _resolve_by_url(base_url, fallback_key=None) -> str:
    """Resolve API key by URL pattern matching."""
    base = str(base_url or "").lower()
    if "localhost" in base or "127.0.0.1" in base:
        return "no-key-required"
    if "minimaxi.com" in base and "/anthropic" in base:
        return os.environ.get("MINIMAX_API_KEY") or fallback_key
    if "xiaomimimo.com" in base:
        return os.environ.get("MIMO_API_KEY") or fallback_key
    if "deepseek.com" in base:
        return fallback_key or os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if "aliyuncs.com" in base:
        return os.environ.get("QWEN_API_KEY") or os.environ.get("DASHSCOPE_API_KEY") or fallback_key
    return fallback_key or os.environ.get("OPENAI_API_KEY")


def resolve_api_key(base_url, explicit_key=None, fallback_key=None, provider=None):
    """Resolve API key by provider name or URL pattern.

    Args:
        base_url: Provider base URL (used for URL-pattern matching).
        explicit_key: If set, returned directly.
        fallback_key: Fallback key if env var not found.
        provider: Provider name (e.g. "deepseek", "mimo", "minimax").
                  Takes precedence over URL-pattern matching.
    """
    if explicit_key:
        return explicit_key
    if provider:
        return _resolve_by_provider(provider, fallback_key)
    return _resolve_by_url(base_url, fallback_key)


# ---------------------------------------------------------------------------
# Thinking-aware proxy: auto-injects correct thinking params based on model
# ---------------------------------------------------------------------------

from mineru_tools.llm_client import thinking_extra_body as _thinking_extra_body  # noqa: E402


class _ThinkingCompletions:
    """Proxy for client.chat.completions that auto-injects thinking params."""

    def __init__(self, inner):
        self._inner = inner

    def create(self, **kwargs):
        model = kwargs.get("model", "")
        if "extra_body" not in kwargs:
            extra = _thinking_extra_body(model)
            if extra:
                kwargs["extra_body"] = extra
        return self._inner.create(**kwargs)


class _ThinkingProxyClient:
    """Wraps OpenAI client, auto-injects thinking on chat.completions.create.

    Rules:
    - If caller does NOT set extra_body → auto-inject thinking disable params
    - If caller already sets extra_body → do NOT override (preserve explicit control)
    - All other attributes proxy through to the underlying client
    """

    _inner = None  # class-level default prevents __getattr__ recursion on failed init

    def __init__(self, inner):
        self._inner = inner
        self.chat = type("Chat", (), {
            "completions": _ThinkingCompletions(inner.chat.completions)
        })()

    def __repr__(self):
        return f"_ThinkingProxyClient({self._inner!r})"

    def __getattr__(self, name):
        inner = self.__dict__.get("_inner")
        if inner is None:
            raise AttributeError(name)
        return getattr(inner, name)


def wrap_client(client):
    """Wrap any OpenAI client with thinking-aware proxy.

    Idempotent: passing an already-wrapped client returns it unchanged.

    Usage:
        client = wrap_client(OpenAI(...))
        client.chat.completions.create(model="qwen3.7-plus", ...)  # auto-injects thinking params
    """
    if isinstance(client, _ThinkingProxyClient):
        return client
    return _ThinkingProxyClient(client)


def create_client(api_key=None, base_url=None, model=None):
    """Create an LLM client based on provider URL.

    For localhost URLs, returns an OpenAI client pointing to llama.cpp server.
    For MiniMax Anthropic endpoints, returns an Anthropic client.
    For other URLs, returns an OpenAI-compatible client.

    Args:
        api_key: API key (optional for localhost).
        base_url: Provider base URL (defaults to DeepSeek via PROVIDERS).
        model: Model name (defaults to DeepSeek default via PROVIDERS).

    Returns:
        Tuple of (client, model_name, provider_type).
    """
    from mineru_core.settings import PROVIDERS
    base_url = base_url or PROVIDERS["deepseek"].base_url
    model = model or PROVIDERS["deepseek"].default_model
    if base_url is None or str(base_url).strip() == "":
        raise ValueError(
            "base_url is required. Provide a localhost URL for llama.cpp, "
            "a MiniMax Anthropic endpoint, or an OpenAI-compatible endpoint."
        )
    base_lower = str(base_url).lower()

    if "localhost" in base_lower or "127.0.0.1" in base_lower:
        # Use standard OpenAI client pointing to llama.cpp
        from openai import OpenAI, DefaultHttpxClient

        # Ensure base_url ends with /v1 for llama.cpp
        if not base_url.rstrip("/").endswith("/v1"):
            base_url = base_url.rstrip("/") + "/v1"
        # trust_env=False: 禁止读系统 HTTP_PROXY/HTTPS_PROXY 环境变量（本机 mihomo 代理 7890 间歇性挂起会拖死 API 请求）
        # 经验来源：MinerU docs/retrospectives 2026-05-09 "httpx 走系统代理导致 hang"、2026-05-10 "missing proxy=None"
        # limits=max_keepalive_connections=0: 禁连接复用，避免死连接被静默复用导致请求无响应（teacher 矩阵 82 分钟零进展事故）
        client = OpenAI(
            api_key="no-key-required",
            base_url=base_url,
            timeout=120,
            max_retries=0,
            http_client=DefaultHttpxClient(proxy=None, trust_env=False, limits=httpx.Limits(max_keepalive_connections=0)),
        )
        return client, model or "qwen3.5-9b", "openai"  # no wrap — llama.cpp doesn't understand provider-specific extra_body

    if "minimaxi.com" in base_lower and "/anthropic" in base_lower:
        from anthropic import Anthropic

        if api_key is None:
            api_key = os.environ.get("MINIMAX_API_KEY", "")
        if not api_key:
            raise ValueError("Anthropic API key required. Set MINIMAX_API_KEY env var or pass --target-api-key.")
        client = Anthropic(api_key=api_key, base_url=base_url, timeout=120, http_client=httpx.Client(proxy=None, trust_env=False, limits=httpx.Limits(max_keepalive_connections=0)))
        return client, model, "anthropic"

    from openai import OpenAI, DefaultHttpxClient

    if api_key is None:
        api_key = resolve_api_key(base_url)
    if not api_key:
        raise ValueError("API key is required. Set via --api-key or provider env var (DEEPSEEK_API_KEY/OPENAI_API_KEY for DeepSeek).")
    # trust_env=False: 禁读系统代理 env（mimo/deepseek 均国内直连 0.1-0.3s；代理 7890 间歇故障会 hang）
    # 与 MiniMax 分支 httpx.Client(proxy=None, trust_env=False) 对齐；proxy=None 单独用不生效（httpx 仍读 env）
    # limits=max_keepalive_connections=0: 禁连接复用（keepalive 死连接复用嫌疑：19:04 进程 82 分钟零进展）
    client = OpenAI(api_key=api_key, base_url=base_url, timeout=120, max_retries=0, http_client=DefaultHttpxClient(proxy=None, trust_env=False, limits=httpx.Limits(max_keepalive_connections=0)))
    return wrap_client(client), model, "openai"
