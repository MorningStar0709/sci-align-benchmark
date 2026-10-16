"""Medical RAG controller — health, search, synthesize, ask, report endpoints.

Migrated from ``controlsci.api.medical_rag``.
"""

from __future__ import annotations

import json
import re
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Query

from mineru_core.paths import PROJECT_ROOT
from mineru_core.settings import get_settings

from mineru_api.schemas.medical import (
    AskRequest,
    SearchResponse,
    SearchResultItem,
    SynthesizeRequest,
)
from mineru_api.services.medical_index import (
    medical_root,
    resolve_index_dir,
    list_available_indexes,
)

router = APIRouter(tags=["medical-rag"])

REPORT_DIR = PROJECT_ROOT / "benchmark" / "eval" / "results" / "medical"
TRACE_JSONL = PROJECT_ROOT / "benchmark" / "eval" / "results" / "medical_rag_live_traces.jsonl"


def _health():
    """Build health check payload for medical RAG."""
    md = medical_root()
    issues = []
    available = len(list(md.glob("index*"))) if md.exists() else 0
    components = {
        "medical_root": {"path": str(md), "exists": md.exists()},
        "indexes": {"available": available, "detail": list_available_indexes()},
    }
    if not md.exists():
        issues.append(f"医疗数据目录不存在: {md}")
    return {
        "status": "ok" if not issues else "degraded",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "profile": get_settings().cloud_demo_profile,
        "components": components,
        "issues": issues,
    }


# Public alias for CLI and other modules
health = _health


@router.get("/api/evidence/search", response_model=SearchResponse)
def evidence_search(
    q: str = Query(..., description="检索查询"),
    k: int = Query(5, description="返回数量"),
    mode: str = Query("hybrid", description="dense | bm25 | hybrid | vision"),
    index_id: str = Query("bge_m3", description="qwen | bge_small | bge_m3"),
    vision: bool = Query(False, description="启用视觉检索"),
):
    from mineru_medical.indexing import load_manifest, search_hybrid

    manifest = load_manifest(str(medical_root() / "chunks"))
    results_raw = search_hybrid(q, k=k, mode=mode, index_root=resolve_index_dir(index_id), vision=vision)
    results = []
    for item in results_raw:
        chunk_id = item.get("chunk_id", "")
        chunk = manifest.get(chunk_id, {})
        results.append(SearchResultItem(
            rank=item["rank"],
            chunk_id=chunk_id,
            rrf_score=item.get("rrf_score", 0.0),
            vision_score=item.get("vision_score"),
            is_vision=item.get("is_vision", False),
            medical_label=chunk.get("label", ""),
            medical_parent=chunk.get("parent"),
            source_file=chunk.get("source", ""),
            source_section=chunk.get("section", ""),
            text_preview=item["text_preview"],
            text_full=item["text_full"],
        ))
    return SearchResponse(
        query=q, top_k=k, mode=mode, vision_enabled=vision,
        count=len(results), results=results,
    )


@router.post("/api/evidence/synthesize")
def evidence_synthesize(req: SynthesizeRequest):
    from mineru_medical.indexing import load_manifest, search_hybrid
    from mineru_medical.synthesis import synthesize_evidence

    manifest = load_manifest(str(medical_root() / "chunks"))
    results_raw = search_hybrid(req.query, k=req.k, mode=req.mode, index_root=resolve_index_dir(req.index_id))
    chunks = []
    for item in results_raw:
        chunk_id = item.get("chunk_id", "")
        chunk = manifest.get(chunk_id, {})
        chunks.append({
            "chunk_id": chunk_id,
            "text": item["text_full"],
            "source": chunk.get("source", ""),
            "section": chunk.get("section", ""),
            "label": chunk.get("label", ""),
            "score": item.get("rrf_score", 0),
        })

    synthesis = synthesize_evidence(req.query, chunks, model=req.synthesis_model)
    task_id = uuid.uuid4().hex[:12]
    report_path = REPORT_DIR / f"synthesis_{task_id}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(synthesis, ensure_ascii=False, indent=2), encoding="utf-8")

    return {"task_id": task_id, "query": req.query, "synthesis": synthesis, "chunks": chunks}


@router.post("/api/medical-rag/ask")
def medical_rag_ask(req: AskRequest):
    from mineru_medical.indexing import load_manifest, search_hybrid
    from mineru_medical.synthesis import synthesize_evidence

    manifest = load_manifest(str(medical_root() / "chunks"))
    results_raw = search_hybrid(req.question, k=req.k, mode=req.mode, index_root=resolve_index_dir(req.index_id))
    chunks = []
    for item in results_raw:
        chunk_id = item.get("chunk_id", "")
        chunk = manifest.get(chunk_id, {})
        chunks.append({
            "chunk_id": chunk_id,
            "text": item["text_full"],
            "source": chunk.get("source", ""),
            "section": chunk.get("section", ""),
            "score": item.get("rrf_score", 0),
        })

    synthesis = synthesize_evidence(req.question, chunks, model=req.synthesis_model)

    trace = {
        "question": req.question,
        "k": req.k,
        "mode": req.mode,
        "index_id": req.index_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "chunk_count": len(chunks),
        "synthesis": synthesis,
    }
    TRACE_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with open(TRACE_JSONL, "a", encoding="utf-8") as f:
        f.write(json.dumps(trace, ensure_ascii=False) + "\n")

    return {
        "answer": synthesis.get("answer", ""),
        "citations": synthesis.get("citations", []),
        "citation_coverage": synthesis.get("citation_coverage", 0),
        "confidence": synthesis.get("confidence", ""),
        "chunks": chunks,
    }


@router.get("/api/evidence/report/{task_id}")
def evidence_report(task_id: str):
    if not re.fullmatch(r"[a-f0-9]{6,32}", task_id):
        raise HTTPException(status_code=400, detail={"error": "task_id 格式不合法"})
    path = REPORT_DIR / f"synthesis_{task_id}.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail={"error": f"报告不存在: {task_id}"})
    return json.loads(path.read_text(encoding="utf-8"))
