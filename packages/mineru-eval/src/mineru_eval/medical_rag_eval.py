"""Reproducible evaluation loop for the local medical RAG track.

The default run evaluates retrieval only. Add ``--with-synthesis`` when a
local Ollama chat model should generate evidence-bounded answers as well.
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from mineru_core.paths import PROJECT_ROOT
from mineru_medical.embedding_providers import LOCAL_DEFAULT_MODEL, embed_texts, read_index_metadata
from mineru_medical.indexing import _load_chunk_texts, _lazy_import_faiss, load_manifest


@dataclass(frozen=True)
class EvalCase:
    case_id: str
    query: str
    target_labels: tuple[str, ...]
    target_terms: tuple[str, ...]
    focus: str


DEFAULT_CASES = [
    EvalCase(
        "primary_endpoint_safety",
        "primary endpoint adverse events",
        ("primary_outcome", "secondary_outcome", "outcome_measures", "adverse_events"),
        ("primary endpoint", "adverse event", "adverse events"),
        "Endpoint and safety retrieval",
    ),
    EvalCase(
        "survival_endpoint",
        "overall survival progression free survival",
        ("primary_outcome", "secondary_outcome", "outcome_measures", "statistical_analysis"),
        ("overall survival", "progression-free survival", "progression free survival"),
        "Survival endpoint retrieval",
    ),
    EvalCase(
        "eligibility",
        "inclusion criteria randomized controlled trial",
        ("population", "_methods_other", "_randomization_other"),
        ("inclusion criteria", "exclusion criteria", "eligible"),
        "Eligibility criteria retrieval",
    ),
    EvalCase(
        "safety_sae",
        "safety endpoint serious adverse event",
        ("adverse_events", "secondary_outcome", "_data_collection_other"),
        ("serious adverse event", "safety", "sae"),
        "Safety event retrieval",
    ),
    EvalCase(
        "closed_loop_diabetes",
        "closed loop insulin hypoglycaemia primary endpoint",
        ("primary_outcome", "outcome_measures", "_methods_other"),
        ("closed-loop", "hypoglycaemia", "hypoglycemia", "primary endpoint"),
        "Domain-specific endpoint retrieval",
    ),
    EvalCase(
        "chemo_toxicity",
        "chemotherapy dose reduction treatment delay adverse events",
        ("intervention", "secondary_outcome", "adverse_events"),
        ("dose reduction", "treatment delay", "adverse events", "toxicit"),
        "Intervention and toxicity retrieval",
    ),
    EvalCase(
        "kaplan_meier",
        "Kaplan Meier survival curve median overall survival",
        ("statistical_analysis", "_statistical_analysis_other", "outcome_measures"),
        ("kaplan", "median overall survival", "survival"),
        "Statistical method retrieval",
    ),
    EvalCase(
        "itt_population",
        "statistical analysis intention to treat population",
        ("statistical_analysis", "_statistical_analysis_other", "population"),
        ("intention-to-treat", "intention to treat", "statistical analysis"),
        "Analysis population retrieval",
    ),
]

ZH_ASK_CASES = [
    EvalCase(
        "zh_primary_endpoint_safety",
        "主要终点和安全性证据",
        ("primary_outcome", "secondary_outcome", "outcome_measures", "adverse_events"),
        ("primary endpoint", "adverse event", "adverse events", "safety"),
        "中文 Ask: 主要终点与安全性",
    ),
    EvalCase(
        "zh_survival_endpoint",
        "总生存和无进展生存证据",
        ("primary_outcome", "secondary_outcome", "outcome_measures", "statistical_analysis"),
        ("overall survival", "progression-free survival", "progression free survival"),
        "中文 Ask: OS/PFS 证据",
    ),
    EvalCase(
        "zh_chemo_toxicity",
        "化疗剂量减少和治疗延迟的不良事件证据",
        ("intervention", "secondary_outcome", "adverse_events"),
        ("chemotherapy", "dose reduction", "treatment delay", "adverse events", "toxicit"),
        "中文 Ask: 化疗剂量与毒性",
    ),
    EvalCase(
        "zh_serious_adverse_event",
        "严重不良事件的安全性终点证据",
        ("adverse_events", "secondary_outcome", "_data_collection_other"),
        ("serious adverse event", "safety", "endpoint"),
        "中文 Ask: 严重不良事件",
    ),
    EvalCase(
        "zh_itt_population",
        "意向治疗分析人群的统计分析证据",
        ("statistical_analysis", "_statistical_analysis_other", "population"),
        ("intention-to-treat", "intention to treat", "statistical analysis"),
        "中文 Ask: ITT 分析人群",
    ),
    EvalCase(
        "zh_closed_loop_mechanism",
        "为什么诊断后48个月内胰岛素需求上升支持早期使用自适应闭环系统？",
        ("primary_outcome", "outcome_measures", "_methods_other"),
        ("closed-loop", "closed loop", "insulin", "hypoglycaemia", "hypoglycemia", "48"),
        "中文 Ask: 闭环胰岛素机制解释",
    ),
    EvalCase(
        "zh_eligibility",
        "随机对照试验的纳入排除标准",
        ("population", "_methods_other", "_randomization_other"),
        ("randomized controlled trial", "inclusion criteria", "exclusion criteria", "eligible"),
        "中文 Ask: 纳入排除标准",
    ),
]

CASE_SETS = {
    "en": DEFAULT_CASES,
    "zh_ask": ZH_ASK_CASES,
}

INDEX_ALIASES = {
    "qwen": "index",
    "bge_small": "index_bge_small",
    "bge_m3": "index_bge_m3",
}


def _load_bm25(index_dir: Path):
    with open(index_dir / "bm25.pkl", "rb") as f:
        return pickle.load(f)


def _case_to_dict(case: EvalCase) -> dict:
    return {
        "case_id": case.case_id,
        "query": case.query,
        "target_labels": list(case.target_labels),
        "target_terms": list(case.target_terms),
        "focus": case.focus,
    }


def _result_item(idx: int, score: float, mode: str, manifest: dict, texts: list[str], rank: int) -> dict:
    chunk = manifest["chunks"][idx]
    text = texts[idx] if idx < len(texts) else ""
    return {
        "rank": rank,
        "manifest_index": idx,
        "chunk_id": chunk.get("chunk_id", ""),
        "score": round(float(score), 6),
        "mode": mode,
        "medical_label": chunk.get("medical_label", ""),
        "medical_parent": chunk.get("medical_parent"),
        "source_section": chunk.get("source_section", ""),
        "source_file": str(chunk.get("filepath") or ""),
        "preview": text[:220].replace("\n", " "),
        "_text": text,
    }


def _bm25_search(query: str, bm25_data: dict, manifest: dict, texts: list[str], k: int) -> list[dict]:
    scores = np.asarray(bm25_data["bm25"].get_scores(query.lower().split()))
    top = np.argsort(scores)[-k:][::-1]
    return [_result_item(int(idx), float(scores[int(idx)]), "bm25", manifest, texts, rank + 1) for rank, idx in enumerate(top)]


def _dense_search(query_emb: np.ndarray, faiss_index, manifest: dict, texts: list[str], k: int) -> list[dict]:
    faiss = _lazy_import_faiss()
    emb = query_emb.reshape(1, -1).astype(np.float32)
    faiss.normalize_L2(emb)
    scores, indices = faiss_index.search(emb, k)
    items = []
    for rank, (idx, score) in enumerate(zip(indices[0], scores[0])):
        if idx >= 0:
            items.append(_result_item(int(idx), float(score), "dense", manifest, texts, rank + 1))
    return items


def _hybrid_search(dense_items: list[dict], bm25_items: list[dict], manifest: dict, texts: list[str], k: int, rrf_k: int = 60) -> list[dict]:
    scores: dict[int, float] = {}
    for rank, item in enumerate(dense_items):
        idx = item["manifest_index"]
        scores[idx] = scores.get(idx, 0.0) + 1.0 / (rrf_k + rank + 1)
    for rank, item in enumerate(bm25_items):
        idx = item["manifest_index"]
        scores[idx] = scores.get(idx, 0.0) + 1.0 / (rrf_k + rank + 1)
    ranked = sorted(scores.items(), key=lambda kv: -kv[1])[:k]
    return [_result_item(idx, score, "hybrid", manifest, texts, rank + 1) for rank, (idx, score) in enumerate(ranked)]


def _matches_case(case: EvalCase, item: dict) -> dict:
    label_hit = item.get("medical_label") in set(case.target_labels)
    text = f"{item.get('medical_label', '')} {item.get('source_section', '')} {item.get('preview', '')}".lower()
    term_hits = [term for term in case.target_terms if term.lower() in text]
    return {
        "label_hit": label_hit,
        "term_hits": term_hits,
        "hit": bool(label_hit or term_hits),
    }


def _score_case(case: EvalCase, items: list[dict]) -> dict:
    matches = [_matches_case(case, item) for item in items]
    first_hit_rank = next((i + 1 for i, match in enumerate(matches) if match["hit"]), None)
    label_hit_rank = next((i + 1 for i, match in enumerate(matches) if match["label_hit"]), None)
    return {
        "hit_at_k": first_hit_rank is not None,
        "first_hit_rank": first_hit_rank,
        "label_hit_at_k": label_hit_rank is not None,
        "first_label_hit_rank": label_hit_rank,
        "matched_terms": sorted({term for match in matches for term in match["term_hits"]}),
    }


def _strip_private_text(items: list[dict]) -> list[dict]:
    public_items = []
    for item in items:
        public = dict(item)
        public.pop("_text", None)
        public_items.append(public)
    return public_items


def _api_search_case(case: EvalCase, index_id: str, manifest: dict, texts: list[str], k: int) -> tuple[list[dict], dict]:
    """已停用：依赖的 API 侧查询改写与多路融合实现随重构消失，无后继。"""
    raise RuntimeError(
        "--case-set zh_ask 检索分支停用：mineru_api.medical_rag 已随 API 层 controllers/services "
        "重构移除，_rewrite_query_for_retrieval 与 _search_multi_query 无后继实现。"
        "可用的现成件：mineru_medical.indexing.search_hybrid（混合检索）与 "
        "mineru_eval.track3_zh_ask_robustness.rewrite_query（查询改写），"
        "两者调用契约与原实现不同，需重设计后再接回 —— 见 docs/reports/backlog-post-linux-cutover.md"
    )


def _synthesize(case: EvalCase, items: list[dict], model: str, search_query: str | None = None) -> dict:
    from mineru_api.services.rag_synthesis import (
        _build_evidence_cards,
        _build_rag_prompt,
        _call_local_llm_chat,
        _classify_question_profile,
        _normalize_rag_answer,
    )

    evidence_items = []
    for item in items:
        evidence_items.append(
            {
                "chunk_id": item["chunk_id"],
                "source_file": Path(item.get("source_file") or "").stem or "unknown",
                "source_section": item.get("source_section", ""),
                "medical_label": item.get("medical_label", ""),
                "medical_parent": item.get("medical_parent"),
                "rrf_score": item.get("score", 0.0),
                "text": item.get("_text", ""),
                "text_preview": item.get("preview", ""),
            }
        )

    question_profile = _classify_question_profile(case.query)
    evidence_cards = _build_evidence_cards(evidence_items, question_profile)
    messages = _build_rag_prompt(case.query, evidence_items, search_query, question_profile, evidence_cards)
    raw, meta = _call_local_llm_chat(model, messages)
    raw["query"] = case.query
    raw["original_query"] = case.query
    normalized = _normalize_rag_answer(
        raw,
        evidence_items,
        question_profile=question_profile,
        evidence_cards=evidence_cards,
        original_query=case.query,
    )
    return {
        "model": meta.get("model", model),
        "rag_mode": "evidence_structured",
        "question_type": question_profile["question_type"],
        "answer_strategy": question_profile["answer_strategy"],
        "evidence_cards_count": len(evidence_cards),
        "evidence_card_roles": sorted({card["role"] for card in evidence_cards}),
        "direct_source_cards": sum(1 for card in evidence_cards if card.get("support_level") == "direct"),
        "reasoning_steps": len(normalized["reasoning_trace"]),
        "answer": normalized["answer"],
        "citations": normalized["citations"],
        "reasoning_trace": normalized["reasoning_trace"],
        "claim_count": normalized["claim_count"],
        "supported_claims": normalized["supported_claims"],
        "unsupported_claims": normalized["unsupported_claims"],
        "citation_coverage": normalized["citation_coverage"],
        "confidence": normalized["confidence"],
        "abstain": normalized["abstain"],
        "abstain_reason": normalized["abstain_reason"],
        "evidence_sufficiency": normalized["evidence_sufficiency"],
        "answer_audit": normalized["answer_audit"],
    }


def _count_rag_cases(per_case: dict, key_path: list[str], target_value) -> int:
    """Count cases matching a nested key path and target value."""
    count = 0
    for result in per_case.values():
        obj = result
        for k in key_path:
            obj = (obj or {}).get(k)
        if obj == target_value:
            count += 1
    return count


def _compute_index_stats(per_case: dict) -> dict:
    """Compute per-index aggregate statistics."""
    total = len(per_case)
    hit_count = sum(1 for result in per_case.values() if result["metrics"]["hit_at_k"])
    label_hit_count = sum(1 for result in per_case.values() if result["metrics"]["label_hit_at_k"])
    ranks = [result["metrics"]["first_hit_rank"] for result in per_case.values() if result["metrics"]["first_hit_rank"]]
    citation_coverages = [
        result["synthesis"]["citation_coverage"]
        for result in per_case.values()
        if result.get("synthesis") and result["synthesis"].get("citation_coverage") is not None
    ]
    structured_cases = _count_rag_cases(per_case, ["synthesis", "rag_mode"], "evidence_structured")
    mechanism_cases = _count_rag_cases(per_case, ["synthesis", "question_type"], "mechanism_explanation")
    audit_passed_cases = _count_rag_cases(per_case, ["synthesis", "answer_audit", "status"], "passed")
    query_languages = sorted({result.get("query_language") for result in per_case.values() if result.get("query_language")})
    bridged_cases = sum(1 for result in per_case.values() if result.get("rewrite_method") not in {None, "identity"})
    multi_query_cases = sum(1 for result in per_case.values() if len(result.get("search_queries") or []) > 1)
    return {
        "total": total,
        "hit_count": hit_count,
        "label_hit_count": label_hit_count,
        "ranks": ranks,
        "citation_coverages": citation_coverages,
        "structured_cases": structured_cases,
        "mechanism_cases": mechanism_cases,
        "audit_passed_cases": audit_passed_cases,
        "query_languages": query_languages,
        "bridged_cases": bridged_cases,
        "multi_query_cases": multi_query_cases,
    }


def _summarize_index(per_case: dict) -> dict:
    s = _compute_index_stats(per_case)
    return {
        "cases": s["total"],
        "hit_at_k": s["hit_count"],
        "hit_rate": round(s["hit_count"] / s["total"], 4) if s["total"] else 0,
        "label_hit_at_k": s["label_hit_count"],
        "label_hit_rate": round(s["label_hit_count"] / s["total"], 4) if s["total"] else 0,
        "mean_first_hit_rank": round(sum(s["ranks"]) / len(s["ranks"]), 3) if s["ranks"] else None,
        "mean_citation_coverage": round(sum(s["citation_coverages"]) / len(s["citation_coverages"]), 4) if s["citation_coverages"] else None,
        "query_languages": s["query_languages"],
        "bridged_cases": s["bridged_cases"],
        "multi_query_cases": s["multi_query_cases"],
        "structured_rag_cases": s["structured_cases"],
        "mechanism_cases": s["mechanism_cases"],
        "answer_audit_passed_cases": s["audit_passed_cases"],
    }


def evaluate(
    index_dirs: list[Path],
    output_json: Path,
    k: int,
    with_synthesis: bool,
    synthesis_model: str,
    limit_cases: int | None = None,
    case_set: str = "en",
    trace_jsonl: Path | None = None,
) -> dict:
    manifest_path = PROJECT_ROOT / "data" / "sources_medical" / "chunks" / "chunks_manifest.json"
    manifest = load_manifest(manifest_path)
    texts = _load_chunk_texts(manifest["chunks"])
    faiss = _lazy_import_faiss()
    selected_cases = CASE_SETS[case_set]
    cases = selected_cases[:limit_cases] if limit_cases else selected_cases

    result = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "manifest": str(manifest_path),
        "k": k,
        "with_synthesis": with_synthesis,
        "synthesis_model": synthesis_model if with_synthesis else None,
        "case_set": case_set,
        "trace_jsonl": str(trace_jsonl) if trace_jsonl else None,
        "cases": [_case_to_dict(case) for case in cases],
        "indexes": [],
        "results": {},
        "summary": {},
    }
    trace_records = []

    for index_dir in index_dirs:
        meta = read_index_metadata(index_dir)
        provider = meta.get("embedding_provider", "local")
        model = meta.get("embedding_model", LOCAL_DEFAULT_MODEL)
        label = meta.get("index_id") or index_dir.name
        print(f"[Medical RAG Eval] {label}: {provider}:{model}", flush=True)

        t0 = time.time()
        query_embeddings = None
        if case_set == "en":
            query_embeddings = embed_texts(
                [case.query for case in cases],
                provider,
                model,
                batch_size=min(8, len(cases)),
                progress_label=f"eval_{label}",
            )
        embed_seconds = time.time() - t0
        bm25_data = _load_bm25(index_dir)
        faiss_index = faiss.read_index(str(index_dir / "medical.index"))

        index_info = {
            "label": label,
            "index_dir": str(index_dir),
            "embedding_provider": provider,
            "embedding_model": model,
            "embedding_dim": meta.get("embedding_dim"),
            "query_embed_seconds": round(embed_seconds, 3),
        }
        result["indexes"].append(index_info)

        per_case = {}
        for case_idx, case in enumerate(cases):
            query_info = {
                "original_query": case.query,
                "search_query": case.query,
                "search_queries": [case.query],
                "query_language": "en",
                "query_rewrites": [],
                "rewrite_method": "identity",
                "vision_enabled": False,
            }
            if case_set == "zh_ask":
                hybrid_items, query_info = _api_search_case(case, label, manifest, texts, k)
            else:
                dense_items = _dense_search(query_embeddings[case_idx], faiss_index, manifest, texts, k * 2)
                bm25_items = _bm25_search(case.query, bm25_data, manifest, texts, k * 2)
                hybrid_items = _hybrid_search(dense_items, bm25_items, manifest, texts, k)
            metrics = _score_case(case, hybrid_items)
            item_result = {
                "case": _case_to_dict(case),
                "search_query": query_info["search_query"],
                "search_queries": query_info["search_queries"],
                "query_language": query_info["query_language"],
                "query_rewrites": query_info["query_rewrites"],
                "rewrite_method": query_info["rewrite_method"],
                "vision_enabled": query_info["vision_enabled"],
                "metrics": metrics,
                "top_results": _strip_private_text(hybrid_items),
            }
            if with_synthesis:
                print(f"[Medical RAG Eval] synthesize {label}/{case.case_id}", flush=True)
                item_result["synthesis"] = _synthesize(case, hybrid_items, synthesis_model, query_info["search_query"])
            per_case[case.case_id] = item_result
            trace_records.append(
                {
                    "generated_at": result["generated_at"],
                    "case_set": case_set,
                    "index": label,
                    "case": _case_to_dict(case),
                    "search_query": item_result["search_query"],
                    "search_queries": item_result["search_queries"],
                    "query_language": item_result["query_language"],
                    "query_rewrites": item_result["query_rewrites"],
                    "rewrite_method": item_result["rewrite_method"],
                    "metrics": item_result["metrics"],
                    "top_results": item_result["top_results"],
                    "synthesis": item_result.get("synthesis"),
                }
            )

        result["results"][label] = per_case
        result["summary"][label] = _summarize_index(per_case)

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    output_json.with_suffix(".md").write_text(_render_markdown(result), encoding="utf-8")
    if trace_jsonl:
        trace_jsonl.parent.mkdir(parents=True, exist_ok=True)
        trace_jsonl.write_text(
            "\n".join(json.dumps(record, ensure_ascii=False) for record in trace_records) + "\n",
            encoding="utf-8",
        )
    return result


def _render_markdown(data: dict) -> str:
    lines = [
        "# Medical RAG Evaluation",
        "",
        f"- Cases: {len(data['cases'])}",
        f"- Case set: {data.get('case_set', 'en')}",
        f"- Top-k: {data['k']}",
        f"- Synthesis: {'enabled' if data['with_synthesis'] else 'disabled'}",
        "",
        "## Summary",
        "",
        "| Index | Provider | Model | Hit@K | Label Hit@K | Mean Hit Rank | Citation Coverage | Structured | Audit Pass | Mechanism | Bridged | Multi-query | Query Embed Sec |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    index_by_label = {item["label"]: item for item in data["indexes"]}
    for label, summary in data["summary"].items():
        index = index_by_label[label]
        coverage = summary["mean_citation_coverage"]
        coverage_text = "" if coverage is None else f"{coverage:.2f}"
        rank = summary["mean_first_hit_rank"]
        rank_text = "" if rank is None else f"{rank:.2f}"
        lines.append(
            f"| {label} | {index['embedding_provider']} | {index['embedding_model']} | "
            f"{summary['hit_at_k']}/{summary['cases']} | {summary['label_hit_at_k']}/{summary['cases']} | "
            f"{rank_text} | {coverage_text} | {summary['structured_rag_cases']} | {summary['answer_audit_passed_cases']} | {summary['mechanism_cases']} | "
            f"{summary['bridged_cases']} | {summary['multi_query_cases']} | {index['query_embed_seconds']} |"
        )

    lines.extend(["", "## Case Details", ""])
    for case in data["cases"]:
        lines.append(f"### {case['case_id']}: {case['query']}")
        lines.append("")
        lines.append("| Index | Hit | Label Hit | Search Query | Top-1 | Label | Preview |")
        lines.append("|---|---|---|---|---|---|---|")
        for index in data["indexes"]:
            label = index["label"]
            case_result = data["results"][label][case["case_id"]]
            top = case_result["top_results"][0]
            metrics = case_result["metrics"]
            preview = top["preview"].replace("|", "/")
            search_query = case_result.get("search_query", case["query"]).replace("|", "/")
            lines.append(
                f"| {label} | {metrics['hit_at_k']} | {metrics['label_hit_at_k']} | "
                f"{search_query} | `{top['chunk_id']}` | {top['medical_label']} | {preview} |"
            )
        lines.append("")
    return "\n".join(lines)


def _resolve_indexes(values: list[str]) -> list[Path]:
    resolved = []
    for value in values:
        dirname = INDEX_ALIASES.get(value, value)
        path = Path(dirname)
        if not path.is_absolute():
            path = PROJECT_ROOT / "data" / "sources_medical" / dirname
        resolved.append(path)
    return resolved


def main():
    parser = argparse.ArgumentParser(description="Run a reproducible local medical RAG evaluation loop")
    parser.add_argument("--indexes", nargs="+", default=["qwen", "bge_small", "bge_m3"], help="Index aliases or directories")
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--with-synthesis", action="store_true", help="Also call the local chat model for answer synthesis")
    parser.add_argument("--synthesis-model", default="qwen3.5:9b")
    parser.add_argument("--limit-cases", type=int, default=None, help="Run only the first N built-in cases")
    parser.add_argument("--case-set", choices=sorted(CASE_SETS), default="en", help="Built-in evaluation case set")
    parser.add_argument("--trace-jsonl", default=None, help="Optional per-case trace JSONL output path")
    parser.add_argument("--output", default=str(PROJECT_ROOT / "benchmark" / "eval" / "results" / "medical_rag_eval.json"))
    args = parser.parse_args()

    output = Path(args.output)
    trace_jsonl = Path(args.trace_jsonl) if args.trace_jsonl else None
    evaluate(_resolve_indexes(args.indexes), output, args.k, args.with_synthesis, args.synthesis_model, args.limit_cases, args.case_set, trace_jsonl)
    print(f"[Medical RAG Eval] wrote {output}", flush=True)
    print(f"[Medical RAG Eval] wrote {output.with_suffix('.md')}", flush=True)
    if trace_jsonl:
        print(f"[Medical RAG Eval] wrote {trace_jsonl}", flush=True)


if __name__ == "__main__":
    main()
