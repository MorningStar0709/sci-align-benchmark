"""Shared helpers for cross_audit and cross_arbitrate modules.

Extracted from cross_audit.py to avoid duplication between audit and arbitrate.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from mineru_eval.client_factory import wrap_client
from mineru_generator.api import PROVIDER_CONFIG, PROVIDER_ENV_MAP, resolve_model

ROOT = Path(__file__).resolve().parents[5]


# ---------------------------------------------------------------------------
# Paper loading
# ---------------------------------------------------------------------------


def load_paper_registry(registry_path: Path) -> dict[str, dict]:
    """Load paper_registry.json, index by arxiv_id."""
    with open(registry_path, encoding="utf-8") as f:
        data = json.load(f)
    papers = data.get("papers", data if isinstance(data, list) else [])
    return {p["arxiv_id"]: p for p in papers}


def read_paper_text(paper_path: str, corpus_dir: Path, max_chars: int = 6000, offset: int = 0) -> str:
    """Read full_text.md for a paper, with sliding-window offset."""
    full_path = (corpus_dir / paper_path / "full_text.md").resolve()
    if not str(full_path).startswith(str(corpus_dir.resolve())):
        return ""
    if not full_path.exists():
        return ""
    text = full_path.read_text(encoding="utf-8", errors="ignore")
    start = min(offset, max(0, len(text) - max_chars))
    return text[start : start + max_chars]


def _load_paper_text(q, corpus_dir, paper_registry):
    """Load paper text for a question, trying paper_source path then registry."""
    paper_source = q.get("paper_source", {})
    arxiv_id = paper_source.get("arxiv_id", "")
    paper_path_str = paper_source.get("paper_path", "")
    text_offset = paper_source.get("text_offset", 0)
    paper_text = ""
    if paper_path_str:
        paper_text = read_paper_text(paper_path_str, corpus_dir, offset=text_offset)
    if not paper_text and arxiv_id and arxiv_id in paper_registry:
        paper_text = read_paper_text(paper_registry[arxiv_id]["paper_path"], corpus_dir, offset=text_offset)
    return paper_text


# ---------------------------------------------------------------------------
# API dispatch
# ---------------------------------------------------------------------------


def _get_api_client(provider: str, model_alias: str | None = None):
    """Get an OpenAI-compatible client for the given provider."""
    try:
        from openai import OpenAI
    except ImportError:
        raise SystemExit("openai package required: uv pip install openai")

    if provider not in PROVIDER_CONFIG:
        raise ValueError(f"Unknown provider: {provider}. Available: {list(PROVIDER_CONFIG.keys())}")

    config = PROVIDER_CONFIG[provider]
    model = resolve_model(provider, model_alias)

    # Resolve API key via env var fallback chain (e.g. DEEPSEEK_API_KEY|OPENAI_API_KEY)
    env_spec = PROVIDER_ENV_MAP[provider]
    api_key = ""
    for env_name in env_spec.split("|"):
        api_key = os.environ.get(env_name.strip(), "")
        if api_key:
            break

    if not api_key:
        raise SystemExit(f"No API key found for {provider}. Set one of [{env_spec}] env vars.")

    return wrap_client(OpenAI(api_key=api_key, base_url=config["base_url"])), model


# ---------------------------------------------------------------------------
# JSON sanitization
# ---------------------------------------------------------------------------


def _sanitize_json_escapes(s: str) -> str:
    r"""Fix invalid JSON escape sequences (e.g., LaTeX \mathbb -> \\mathbb).

    JSON only allows: \", \\, \/, \b, \f, \n, \r, \t, \uXXXX.
    LLM responses often contain LaTeX like \mathbb, \hat which have invalid escapes.
    """
    valid_esc = set('"\\/bfnrtu')
    result = []
    i = 0
    while i < len(s):
        if s[i] == '\\' and i + 1 < len(s):
            next_char = s[i + 1]
            if next_char not in valid_esc:
                result.append('\\\\')
                i += 1
                continue
            if next_char == 'u':
                if i + 5 >= len(s):
                    result.append('\\\\')
                    i += 1
                    continue
                hex_chars = s[i + 2:i + 6]
                if not all(c in '0123456789abcdefABCDEF' for c in hex_chars):
                    result.append('\\\\')
                    i += 1
                    continue
                result.append(s[i:i + 6])
                i += 6
                continue
            if next_char in 'bfnrt' and i + 2 < len(s) and s[i + 2].isalpha():
                result.append('\\\\')
                i += 1
                continue
            result.append(s[i])
            result.append(next_char)
            i += 2
            continue
        else:
            result.append(s[i])
            i += 1
    return ''.join(result)


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------


def write_json_atomic(path: Path, data: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def save_checkpoint(cp_path: Path, results: list[dict], start_time: float) -> None:
    elapsed = time.time() - start_time
    done = len(results)
    rate = done / elapsed * 60 if elapsed > 0 else 0
    write_json_atomic(cp_path, {
        "completed": done,
        "elapsed_sec": round(elapsed, 0),
        "rate_qpm": round(rate, 1),
        "results": results,
    })


def load_checkpoint(checkpoint_path: Path) -> list[dict]:
    if not checkpoint_path.exists():
        return []
    with open(checkpoint_path, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("results", [])


def _resolve_path(p):
    """Resolve a path relative to ROOT if not absolute."""
    p = Path(p)
    return p if p.is_absolute() else ROOT / p
