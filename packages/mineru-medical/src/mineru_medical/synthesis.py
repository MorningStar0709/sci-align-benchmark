"""Synthesis module — RAG evidence synthesis for medical queries."""

from __future__ import annotations

from mineru_api.services.rag_synthesis import (
    _build_evidence_cards,
    _build_rag_prompt,
    _call_local_llm_chat,
    _classify_question_profile,
    _normalize_rag_answer,
)


def _library_safe_model(model: str) -> str:
    """Validate model choice without leaking HTTPException into library context.

    Replaces the old _clean_model_choice which raises fastapi.HTTPException
    for cloud API models — inappropriate for a library function that may be
    called from CLI, notebooks, or other non-FastAPI contexts.
    """
    model = (model or "").strip()
    if not model or model in {"local", "replay"}:
        return "qwen3.5:9b"
    cloud_models = {"deepseek", "openai", "minimax", "mimo"}
    if model in cloud_models:
        raise ValueError(
            f"Medical RAG synthesis must use a local model, "
            f"cannot send medical context to cloud API. Got: {model}"
        )
    return model


def synthesize_evidence(query: str, chunks: list[dict], model: str = "qwen3.5:9b") -> dict:
    """Synthesize evidence from pre-retrieved chunks using the old pipeline.

    Args:
        query: Original user question.
        chunks: Pre-formatted evidence chunks with text/source/section/score.
        model: Local LLM model name for synthesis.

    Returns:
        Dict with answer, citations, citation_coverage, confidence.
    """
    model = _library_safe_model(model)

    # Build evidence items for synthesis pipeline
    evidence_items = []
    for chunk in chunks:
        text = chunk.get("text", "")
        evidence_items.append({
            "chunk_id": chunk.get("chunk_id", ""),
            "source_file": chunk.get("source", "unknown"),
            "source_section": chunk.get("section", ""),
            "medical_label": chunk.get("label", ""),
            "rrf_score": chunk.get("score", 0.0),
            "text": text,
            "prompt_text": text[:1800],
        })

    prompt_items = [{**item, "text": item.get("prompt_text", "")} for item in evidence_items]
    question_profile = _classify_question_profile(query)
    evidence_cards = _build_evidence_cards(prompt_items, question_profile)
    messages = _build_rag_prompt(
        query, prompt_items,
        search_query=query,
        question_profile=question_profile,
        evidence_cards=evidence_cards,
    )
    rag_raw, _ = _call_local_llm_chat(model, messages)
    rag_answer = _normalize_rag_answer(
        rag_raw, evidence_items,
        question_profile=question_profile,
        evidence_cards=evidence_cards,
        original_query=query,
    )
    return {
        "answer": rag_answer.get("answer", ""),
        "citations": rag_answer.get("citations", []),
        "citation_coverage": rag_answer.get("citation_coverage", 0),
        "confidence": rag_answer.get("confidence", ""),
    }
