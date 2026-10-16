"""Pydantic models for medical RAG endpoints.

Migrated from ``controlsci.api.medical_rag``.
"""

from __future__ import annotations

from pydantic import BaseModel


class SynthesizeRequest(BaseModel):
    query: str
    doc_ids: list[int] = []
    k: int = 3
    synthesis_model: str = "qwen3.5:9b"
    mode: str = "hybrid"
    index_id: str = "qwen"


class AskRequest(BaseModel):
    question: str
    k: int = 3
    synthesis_model: str = "qwen3.5:9b"
    mode: str = "hybrid"
    index_id: str = "bge_m3"
    session_id: str | None = None
    history: list[dict] = []
    triage_model: str | None = None
    runtime_config: dict | None = None


class HealthResponse(BaseModel):
    status: str
    timestamp: str
    profile: str
    components: dict
    issues: list[str]


class SearchResultItem(BaseModel):
    rank: int
    chunk_id: str
    rrf_score: float
    vision_score: float | None = None
    is_vision: bool = False
    medical_label: str
    medical_parent: str | None
    source_file: str
    source_section: str
    text_preview: str
    text_full: str


class SearchResponse(BaseModel):
    query: str
    search_query: str | None = None
    search_queries: list[str] = []
    query_language: str | None = None
    query_rewrites: list[dict] = []
    rewrite_method: str | None = None
    top_k: int
    mode: str = "hybrid"
    vision_enabled: bool = False
    count: int
    results: list[SearchResultItem]
