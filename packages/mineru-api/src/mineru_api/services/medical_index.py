"""Medical index helpers — index discovery, metadata, and root resolution.

Extracted from ``controlsci.api.medical_rag``.
"""

from __future__ import annotations

import json
from pathlib import Path

from mineru_core.paths import PROJECT_ROOT
from mineru_core.settings import get_settings

LOCAL_MEDICAL_ROOT = PROJECT_ROOT / "data" / "sources_medical"
DEMO_MEDICAL_ROOT = PROJECT_ROOT / "data" / "demo_cloud" / "medical"

INDEX_ALIASES: dict[str, str] = {
    "qwen": "index",
    "qwen3": "index",
    "qwen3_embedding": "index",
    "bge_small": "index_bge_small",
    "bge-small": "index_bge_small",
    "bge_m3": "index_bge_m3",
    "bge-m3": "index_bge_m3",
}


def medical_root() -> Path:
    """Resolve the active medical data root based on profile."""
    profile = get_settings().cloud_demo_profile
    return DEMO_MEDICAL_ROOT if profile == "cloud_demo" else LOCAL_MEDICAL_ROOT


def resolve_index_dir(index_id: str | None = None) -> Path:
    """Resolve an index directory by alias, defaulting to the primary index."""
    root = medical_root()
    resolved = INDEX_ALIASES.get(index_id or "", index_id or "")
    if not resolved:
        return root / "index"
    return root / resolved


def index_meta(index_id: str | None = None) -> dict:
    """Read index metadata from ``index_meta.json``."""
    meta_path = resolve_index_dir(index_id) / "index_meta.json"
    if meta_path.exists():
        try:
            return json.loads(meta_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def list_available_indexes() -> list[dict]:
    """Scan medical root for available index directories and their metadata."""
    from mineru_medical.embedding_providers import read_index_metadata

    root = medical_root()
    indexes = []
    if not root.exists():
        return indexes

    for sub in root.iterdir():
        if sub.is_dir() and sub.name.startswith("index"):
            meta = index_meta(sub.name)
            indexes.append(
                {
                    "id": sub.name,
                    "display": meta.get("display_name", sub.name),
                    "dimension": meta.get("dimension"),
                    "chunk_count": meta.get("chunk_count"),
                    "exists": True,
                    "meta": read_index_metadata(str(sub)),
                }
            )
    return sorted(indexes, key=lambda x: x["id"])


__all__ = [
    "medical_root",
    "resolve_index_dir",
    "index_meta",
    "list_available_indexes",
]
