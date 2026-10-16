"""Demo utility functions — file checks, network probes, runtime helpers.

Migrated from ``controlsci.api.demo_endpoints``.
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import HTTPException

from mineru_core.paths import PROJECT_ROOT

ROOT = PROJECT_ROOT


def read_json(rel_path: str) -> dict:
    """Read a JSON file relative to PROJECT_ROOT."""
    filepath = ROOT / rel_path
    if not filepath.exists():
        raise HTTPException(status_code=503, detail={"error": f"file missing: {rel_path}"})
    return json.loads(filepath.read_text(encoding="utf-8"))


def file_ok(*parts: str) -> bool:
    """Check if a file exists relative to PROJECT_ROOT."""
    return (ROOT / Path(*parts)).exists()


def public_medical_evidence_ok() -> bool:
    """Check public medical evidence bundle availability."""
    bundle = ROOT / "docs" / "submissions" / "data_trace_bundle" / "09_medical_rag"
    required = [
        "medical_rag_eval.json",
        "medical_rag_eval_zh_ask.json",
        "vision_chunks_manifest.json",
        "vision_descriptions_qwen35.jsonl",
    ]
    return all((bundle / name).exists() for name in required)


def local_llm_tags(timeout: float = 0.5) -> tuple[bool, list[str]]:
    """Check if llama.cpp chat server is healthy, return available models."""
    try:
        import httpx
        from mineru_core.settings import get_settings
        url = f"{get_settings().llm_chat_url}/health"
        r = httpx.get(url, timeout=timeout, trust_env=False)
        if r.status_code == 200:
            return (True, ["qwen3.5-9b", "qwen3-embedding-4b"])
    except Exception:
        pass
    return (False, [])


def docker_ok() -> bool:
    """Check if Docker is available."""
    try:
        import subprocess
        r = subprocess.run(["docker", "info"], capture_output=True, text=True, timeout=5)
        return r.returncode == 0
    except Exception:
        return False
