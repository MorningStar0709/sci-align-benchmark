"""D1 MedBench Sciverse 三维对比评测 Wrapper

路线 A: 不修改 kb_quality.py，独立 wrapper 脚本。
对每个 MedBench 问题执行三路检索 + 三维 Judge 评分：
  - LOCAL: 本地 Hybrid FAISS+BM25 检索（复用 kb_quality.search_hybrid）
  - SCIVERSE: Sciverse agentic-search 检索
  - FUSED: 本地 + Sciverse 简单拼接（先到先得 top-k）

Judge 复用 kb_quality.judge_single_result + 医学四维评分体系。
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]  # restored Phase 5

from mineru_medical.kb_quality import (
    search_hybrid,
    judge_single_result,
)
from mineru_eval.client_factory import create_client
from mineru_core.mineru_client import SciverseClient

BENCH_DIR = ROOT / "data" / "sources_medical" / "medbench" / "medbench_9b_probe"
INDEX_DIR = ROOT / "data" / "sources_medical" / "index"
MANIFEST_PATH = ROOT / "data" / "sources_medical" / "chunks" / "chunks_manifest.json"
OUTPUT_DIR = ROOT / "benchmark" / "eval" / "results" / "medical"
OUTPUT_PATH = OUTPUT_DIR / "medbench_sciverse_report.json"

SUBSET_FILES = [
    "medbench_9b_probe_MedDiag.json",
    "medbench_9b_probe_MedExam.json",
    "medbench_9b_probe_MedLitQA.json",
    "medbench_9b_probe_MedReportQC.json",
    "medbench_9b_probe_MedRxPlan.json",
    "medbench_9b_probe_MedTreat.json",
]

LOCAL_LLM_BASE_URL = "http://localhost:8080/v1"
DEFAULT_JUDGE_MODEL = "qwen3.5-9b"

MAX_SCIVERSE_PREVIEW_CHARS = 1500
MAX_RESULT_CHARS = 3000


def load_medbench_questions(subset_filter=None, max_questions=None):
    questions = []
    for fname in SUBSET_FILES:
        subset = fname.replace("medbench_9b_probe_", "").replace(".json", "")
        if subset_filter and subset not in subset_filter:
            continue

        path = BENCH_DIR / fname
        if not path.exists():
            print(f"  SKIP {fname}: not found")
            continue

        data = json.loads(path.read_text(encoding="utf-8"))
        for item in data.get("per_question", []):
            questions.append({
                "query_id": item["query_id"],
                "subset": subset,
                "question": item["question"],
                "generated_answer": item.get("generated_answer", ""),
                "retrieval_count": item.get("retrieval_count", 0),
                "retrieval_pmcids": item.get("retrieval_pmcids", []),
            })

    if max_questions:
        questions = questions[:max_questions]

    return questions


def search_sciverse(query, top_k=5):
    client = SciverseClient()
    try:
        result = client.agentic_search(query=query, top_k=top_k)
    except Exception as e:
        return {"error": str(e)[:200], "hits": []}

    hits = result.get("hits", [])
    sciverse_results = []
    for h in hits:
        doc_id = h.get("doc_id", "")[:20]
        chunk = h.get("chunk", "")
        title = h.get("title", "")
        sciverse_results.append({
            "chunk_id": f"sciverse:{doc_id}",
            "score": h.get("score", 0),
            "text": (title + "\n\n" + chunk)[:MAX_SCIVERSE_PREVIEW_CHARS],
            "title": title[:120],
            "source_type": h.get("source_type", ""),
            "page_no": h.get("page_no"),
            "_source": "sciverse",
        })
    return {"error": None, "hits": sciverse_results}


def search_local(question, top_k=5):
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    try:
        results = search_hybrid(question, str(INDEX_DIR), manifest, k=top_k)
    except Exception as e:
        return {"error": str(e)[:200], "hits": []}

    local_results = []
    for r in results:
        chunk_id = r.get("chunk_id", "")
        label = r.get("medical_label", "")
        parent = r.get("medical_parent", "")
        text = r.get("text", "")
        local_results.append({
            "chunk_id": chunk_id,
            "rrf_score": r.get("_rrf_score", 0),
            "text": text[:MAX_RESULT_CHARS],
            "label": label,
            "medical_parent": parent,
            "_source": "local",
        })
    return {"error": None, "hits": local_results}


def fuse_results(local_hits, sciverse_hits, top_k=5):
    merged = []
    seen_texts = set()

    local_items = local_hits.get("hits", [])
    sciverse_items = sciverse_hits.get("hits", [])

    for item in local_items:
        key = item["text"][:100]
        if key not in seen_texts:
            seen_texts.add(key)
            merged.append(item)

    for item in sciverse_items:
        key = item["text"][:100]
        if key not in seen_texts:
            seen_texts.add(key)
            merged.append(item)

    return merged[:top_k]


def chunk_to_judge_item(chunk):
    source = chunk.get("_source", "")
    if source == "sciverse":
        return {
            "chunk_id": chunk.get("chunk_id", ""),
            "text": chunk.get("text", ""),
            "label": f"sciverse:{chunk.get('source_type','')}",
            "medical_parent": chunk.get("title", ""),
            "rrf_score": chunk.get("score", 0),
        }
    else:
        return {
            "chunk_id": chunk.get("chunk_id", ""),
            "text": chunk.get("text", ""),
            "label": chunk.get("label", "unknown"),
            "medical_parent": chunk.get("medical_parent", ""),
            "rrf_score": chunk.get("rrf_score", 0),
        }


def run_judge_for_query(question_text, local_chunks, sciverse_chunks, fused_chunks,
                        client, model_name, provider, top_k_judge=3):
    results = {"local": [], "sciverse": [], "fused": []}

    for source_name, chunks in [("local", local_chunks), ("sciverse", sciverse_chunks), ("fused", fused_chunks)]:
        for rank, chunk in enumerate(chunks[:top_k_judge]):
            item = chunk_to_judge_item(chunk)
            try:
                result = judge_single_result(
                    question_text, item,
                    client, model_name,
                    provider=provider,
                )
            except Exception as e:
                result = {
                    "score": 0.0,
                    "reason": f"judge error: {str(e)[:200]}",
                    "dimensions": {"relevance": 0.0, "completeness": 0.0, "traceability": 0.0, "accuracy": 0.0},
                    "issues": [f"judge_error: {str(e)[:100]}"],
                }
            results[source_name].append({
                "rank": rank + 1,
                "chunk_id": item["chunk_id"],
                "source": source_name,
                "judge": result,
            })

    return results


def _parse_args():
    """Parse and apply profile defaults."""
    parser = argparse.ArgumentParser(description="D1: MedBench + Sciverse \u4e09\u7ef4\u5bf9\u6bd4\u8bc4\u6d4b")
    parser.add_argument("--profile", choices=["smoke", "report"], default="smoke")
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--judge-base-url", default=LOCAL_LLM_BASE_URL)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--top-k-judge", type=int, default=3)
    parser.add_argument("--subset", nargs="*", default=None)
    parser.add_argument("--max-questions", type=int, default=None)
    parser.add_argument("--output", default=str(OUTPUT_PATH))
    args = parser.parse_args()

    if args.profile == "smoke":
        args.max_questions = args.max_questions or 3
        args.top_k = args.top_k or 3
        args.top_k_judge = args.top_k_judge or 1
    else:
        args.max_questions = args.max_questions or 25
        args.top_k = args.top_k or 5
        args.top_k_judge = args.top_k_judge or 3
    return args


def _process_single_question(q, args, client, model_name, provider, acc):
    """Process one question: search + judge + accumulate scores."""
    question_text = q["question"]
    qid = q["query_id"]
    subset = q["subset"]
    print(f"\n  [{acc['index'] + 1}/{acc['total']}] {subset}:{qid}")

    t0 = time.time()
    local_result = search_local(question_text, top_k=args.top_k)
    dt_local = time.time() - t0
    print(f"    LOCAL: {len(local_result['hits'])} hits ({dt_local:.1f}s)")

    t0 = time.time()
    sciverse_result = search_sciverse(question_text, top_k=args.top_k)
    dt_sciverse = time.time() - t0
    print(f"    SCIVERSE: {len(sciverse_result['hits'])} hits ({dt_sciverse:.1f}s)")

    fused_chunks = fuse_results(local_result, sciverse_result, top_k=args.top_k)
    print(f"    FUSED: {len(fused_chunks)} hits (local={len(local_result['hits'])} + sciverse={len(sciverse_result['hits'])})")

    overlap_ids = set()
    for h in sciverse_result.get("hits", []):
        if h.get("text", "")[:100] in {lh.get("text", "")[:100] for lh in local_result.get("hits", [])}:
            overlap_ids.add(h.get("chunk_id", "")[:20])

    print(f"    Judge: evaluating {args.top_k_judge} chunks \u00d7 3 sources...")
    judge_results = run_judge_for_query(
        question_text,
        local_result.get("hits", []),
        sciverse_result.get("hits", []),
        fused_chunks,
        client, model_name, provider,
        top_k_judge=args.top_k_judge,
    )

    entry = {
        "query_id": qid,
        "subset": subset,
        "question": question_text[:500],
        "generated_answer": q.get("generated_answer", "")[:500],
        "retrieval_timing": {"local_s": round(dt_local, 2), "sciverse_s": round(dt_sciverse, 2)},
        "overlap_local_sciverse": len(overlap_ids),
        "retrieval": {
            "local": {
                "count": len(local_result.get("hits", [])),
                "error": local_result.get("error"),
                "chunk_ids": [h["chunk_id"][:40] for h in local_result.get("hits", [])],
            },
            "sciverse": {
                "count": len(sciverse_result.get("hits", [])),
                "error": sciverse_result.get("error"),
                "chunk_ids": [h["chunk_id"][:40] for h in sciverse_result.get("hits", [])],
            },
            "fused": {
                "count": len(fused_chunks),
                "chunk_ids": [h.get("chunk_id", "")[:40] for h in fused_chunks],
            },
        },
        "judge": judge_results,
    }

    for src in ["local", "sciverse", "fused"]:
        scores = [j["judge"].get("score", 0) for j in judge_results.get(src, [])]
        entry[f"{src}_mean_score"] = round(float(np.mean(scores)), 4) if scores else 0.0
        acc[f"all_{src}_scores"].extend(scores)

    local_m = entry.get("local_mean_score", 0)
    sciv_m = entry.get("sciverse_mean_score", 0)
    fused_m = entry.get("fused_mean_score", 0)
    print(f"    => LOCAL={local_m:.3f} SCIVERSE={sciv_m:.3f} FUSED={fused_m:.3f}")

    return entry


def main():
    args = _parse_args()

    print("=" * 64)
    print("D1: MedBench + Sciverse \u4e09\u7ef4\u5bf9\u6bd4\u8bc4\u6d4b")
    print(f"  profile={args.profile} n_q={args.max_questions} top_k={args.top_k} top_k_judge={args.top_k_judge}")
    print(f"  judge: {args.judge_model} @ {args.judge_base_url}")
    print("=" * 64)

    questions = load_medbench_questions(
        subset_filter=args.subset,
        max_questions=args.max_questions,
    )
    print(f"\n[Phase 1] \u52a0\u8f7d {len(questions)} \u4e2a MedBench \u95ee\u9898")

    print(f"\n[Phase 2] \u521d\u59cb\u5316 Judge ({args.judge_model})...")
    client, model_name, provider = create_client(
        base_url=args.judge_base_url,
        model=args.judge_model,
    )
    print(f"  Judge ready: {model_name} (provider={provider})")

    acc = {
        "index": 0,
        "total": len(questions),
        "all_local_scores": [],
        "all_sciverse_scores": [],
        "all_fused_scores": [],
    }
    t_start = time.time()
    total_judges = len(questions) * args.top_k_judge * 3
    per_question_results = []

    for qi, q in enumerate(questions):
        acc["index"] = qi
        entry = _process_single_question(q, args, client, model_name, provider, acc)
        per_question_results.append(entry)

    elapsed = time.time() - t_start
    all_local_scores = acc["all_local_scores"]
    all_sciverse_scores = acc["all_sciverse_scores"]
    all_fused_scores = acc["all_fused_scores"]

    print(f"\n{'=' * 64}")
    print(f"  D1 COMPLETE ({elapsed:.0f}s, {total_judges} judge calls)")
    print(f"  LOCAL avg:   {np.mean(all_local_scores):.4f} (n={len(all_local_scores)})")
    print(f"  SCIVERSE avg: {np.mean(all_sciverse_scores):.4f} (n={len(all_sciverse_scores)})")
    print(f"  FUSED avg:   {np.mean(all_fused_scores):.4f} (n={len(all_fused_scores)})")

    report = {
        "meta": {
            "title": "MedBench + Sciverse \u4e09\u7ef4\u5bf9\u6bd4\u8bc4\u6d4b",
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "profile": args.profile,
            "judge_model": args.judge_model,
            "judge_provider": provider,
            "n_questions": len(questions),
            "top_k": args.top_k,
            "top_k_judge": args.top_k_judge,
            "elapsed_seconds": round(elapsed, 1),
        },
        "summary": {
            "local": {
                "mean_score": round(float(np.mean(all_local_scores)), 4),
                "n": len(all_local_scores),
            },
            "sciverse": {
                "mean_score": round(float(np.mean(all_sciverse_scores)), 4),
                "n": len(all_sciverse_scores),
            },
            "fused": {
                "mean_score": round(float(np.mean(all_fused_scores)), 4),
                "n": len(all_fused_scores),
            },
            "delta_fused_vs_local": round(float(np.mean(all_fused_scores) - np.mean(all_local_scores)), 4) if all_local_scores and all_fused_scores else 0,
            "delta_sciverse_vs_local": round(float(np.mean(all_sciverse_scores) - np.mean(all_local_scores)), 4) if all_local_scores and all_sciverse_scores else 0,
        },
        "per_question": per_question_results,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n  Report saved: {args.output}")
    print(f"{'=' * 64}")


if __name__ == "__main__":
    main()
