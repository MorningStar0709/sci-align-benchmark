"""llama.cpp server lifecycle management.

llama.cpp servers are managed externally (started via docker-start.sh or direct command).
This module provides health checks and server status queries.
"""

from __future__ import annotations

import os

import httpx

# Settings keys that reference llama.cpp URLs
_LLM_SETTINGS_KEYS = [
    "llm_chat_url",
    "llm_embed_url",
]


def chat_base_url() -> str:
    """Get chat server base URL."""
    port = os.environ.get("LLAMA_CHAT_PORT", "8080")
    return f"http://localhost:{port}"


def embed_base_url() -> str:
    """Get embedding server base URL."""
    port = os.environ.get("LLAMA_EMBED_PORT", "8081")
    return f"http://localhost:{port}"


def is_chat_running(timeout: float = 5.0) -> bool:
    """Check if chat server is responding."""
    try:
        resp = httpx.get(f"{chat_base_url()}/health", timeout=timeout)
        return resp.status_code == 200
    except Exception:
        return False


def is_embed_running(timeout: float = 5.0) -> bool:
    """Check if embedding server is responding."""
    try:
        resp = httpx.get(f"{embed_base_url()}/health", timeout=timeout)
        return resp.status_code == 200
    except Exception:
        return False
