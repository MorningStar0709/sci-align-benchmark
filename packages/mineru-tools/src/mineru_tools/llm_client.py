"""Unified local LLM client — wraps llama.cpp's OpenAI-compatible API.

Replaces the Ollama-specific client code. All local LLM calls (chat + embedding)
route through this module, providing a single point of configuration.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import httpx


# -- Configuration --

_DEFAULT_CHAT_PORT = 8080
_DEFAULT_EMBED_PORT = 8081


def _chat_base_url() -> str:
    """Chat model base URL (llama.cpp OpenAI-compatible endpoint)."""
    port = os.environ.get("LLAMA_CHAT_PORT", str(_DEFAULT_CHAT_PORT))
    return f"http://localhost:{port}/v1"


def _embed_base_url() -> str:
    """Embedding model base URL (llama.cpp OpenAI-compatible endpoint)."""
    port = os.environ.get("LLAMA_EMBED_PORT", str(_DEFAULT_EMBED_PORT))
    return f"http://localhost:{port}/v1"


def thinking_extra_body(model: str) -> dict:
    """Return the correct extra_body dict to disable thinking mode for a given model.

    Different providers use different parameter formats:
    - Qwen (Alibaba Bailian): `{"enable_thinking": False}`
    - DeepSeek / MiMo:        `{"thinking": {"type": "disabled"}}`
    - Others (no thinking):   `{}` (no extra_body needed)
    """
    if model.startswith("qwen"):
        return {"enable_thinking": False}
    elif model.startswith(("mimo", "deepseek")):
        return {"thinking": {"type": "disabled"}}
    return {}


# -- Data classes --


@dataclass
class ChatResponse:
    """Normalized chat completion response."""

    content: str
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class EmbedResponse:
    """Normalized embedding response."""

    embeddings: list[list[float]]
    raw: dict[str, Any] = field(default_factory=dict)


# -- Client --


class LocalLLMClient:
    """Stateless HTTP client for llama.cpp OpenAI-compatible server."""

    def __init__(self, timeout: float = 120.0) -> None:
        self._timeout = timeout
        self._http = httpx.Client(timeout=timeout, trust_env=False)

    # -- Chat --

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        model: str = "qwen3.5-9b",
        temperature: float = 0.1,
        max_tokens: int = 4096,
        response_format: str | None = None,
        thinking: bool = False,
    ) -> ChatResponse:
        """Send a chat completion request.

        Args:
            messages: OpenAI-format messages list.
            model: Model name (informational; llama.cpp uses the loaded model).
            temperature: Sampling temperature.
            max_tokens: Max tokens to generate.
            response_format: "json" to request JSON output, None for text.
            thinking: Enable Qwen3 thinking mode (default: disabled).

        Returns:
            ChatResponse with content and raw response data.

        Raises:
            httpx.HTTPStatusError: If the server returns a non-2xx response.
        """
        url = f"{_chat_base_url()}/chat/completions"
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
            "chat_template_kwargs": {"enable_thinking": thinking},
        }
        if response_format == "json":
            payload["response_format"] = {"type": "json_object"}

        resp = self._http.post(url, json=payload)
        resp.raise_for_status()
        data = resp.json()

        choices = data.get("choices", [])
        content = ""
        if choices:
            msg = choices[0].get("message", {})
            content = msg.get("content", "")

        return ChatResponse(content=content, raw=data)

    # -- Embedding --

    def embed(
        self,
        texts: list[str],
        *,
        model: str = "qwen3-embedding-4b",
    ) -> EmbedResponse:
        """Send an embedding request.

        Args:
            texts: List of strings to embed.
            model: Model name (informational; llama.cpp uses the loaded model).

        Returns:
            EmbedResponse with embeddings list and raw response data.

        Raises:
            httpx.HTTPStatusError: If the server returns a non-2xx response.
        """
        url = f"{_embed_base_url()}/embeddings"
        payload: dict[str, Any] = {
            "model": model,
            "input": texts,
        }
        resp = self._http.post(url, json=payload)
        resp.raise_for_status()
        data = resp.json()

        embeddings = [item.get("embedding", []) for item in data.get("data", [])]
        return EmbedResponse(embeddings=embeddings, raw=data)

    # -- Health --

    def is_chat_healthy(self) -> bool:
        """Check if chat server is reachable."""
        try:
            port = os.environ.get("LLAMA_CHAT_PORT", str(_DEFAULT_CHAT_PORT))
            resp = self._http.get(f"http://localhost:{port}/health", timeout=5)
            return resp.status_code == 200
        except Exception:
            return False

    def is_embed_healthy(self) -> bool:
        """Check if embedding server is reachable."""
        try:
            port = os.environ.get("LLAMA_EMBED_PORT", str(_DEFAULT_EMBED_PORT))
            resp = self._http.get(f"http://localhost:{port}/health", timeout=5)
            return resp.status_code == 200
        except Exception:
            return False

    def close(self) -> None:
        """Close the underlying HTTP client."""
        self._http.close()

    def __enter__(self) -> "LocalLLMClient":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
