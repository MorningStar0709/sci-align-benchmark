"""知识库自评测引擎 — MedBench 风格 QA 对 + 检索对比 + 多模型 Judge 矩阵

评估层级:
  Level 1: 检索质量    Hybrid vs Dense-only        → 证明 Hybrid 的必要性
  Level 2: 微调影响    QLoRA 微调前 vs 微调后        → 需微调模型就绪 (标注 planned)
  Level 3: 答案质量    14-Judge 矩阵 (3模型×4维)    → 跨模型一致性验证
  Level 4: 标准对齐    MedBench 风格 QA 对自动评测  → 与赛题评测方式对齐

输出:
  output_dir/kb_quality_report.json  (通过 --output 指定)

用法:
  .venv/bin/python -m mineru_medical.kb_quality
  .venv/bin/python -m mineru_medical.kb_quality --n-queries 20 --top-k 3
  .venv/bin/python -m mineru_medical.kb_quality --skip-local --dry-run
"""

import argparse
import hashlib
import json
import os
import random
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from mineru_eval.client_factory import create_client
from mineru_eval.judge import (
    MEDICAL_JUDGE_SYSTEM_PROMPT,
    MEDICAL_USER_PROMPT_TEMPLATE,
    _parse_judge_result,
)
from mineru_core.paths import PROJECT_ROOT
from mineru_core.settings import PROVIDERS
from mineru_medical.indexing import _load_chunk_texts, load_manifest, search_hybrid

VALID_SCORES = [0.0, 0.25, 0.5, 0.75, 1.0]
SMOKE_LOCAL_CANDIDATES = ["gemma3-4b", "qwen3.5-9b"]
REPORT_LOCAL_MODELS = [
    "gemma3-4b",
    "llama3.2-3b",
    "qwen3.5-2b",
    "qwen3.5-4b",
    "qwen3.5-9b",
    "qwen3.5-35b",
]
MEDICAL_LABELS_PRIORITY = [
    "results", "methods", "discussion", "population",
    "intervention", "study_design", "statistical_analysis",
    "introduction", "front_matter",
]
DEFAULT_QUERY_GEN_PROMPT = """你是一位临床医学研究员。请根据以下医学文献片段，生成一个可被检索系统回答的临床问题。

要求:
- 问题应聚焦于该片段所涉及的临床主题（如疗效、安全性、剂量、亚组差异等）
- 问题应像医生/研究人员在文献检索时提出的自然查询
- 问题长度 10-40 字
- 仅返回问题本身，不要解释、编号或额外文字

医学文献片段:
{chunk_text}

临床问题:"""


def _stable_id(index, label):
    return hashlib.md5(f"{index}:{label}".encode()).hexdigest()[:8]


def sample_chunks(manifest, n=30, seed=42):
    chunks = manifest["chunks"]
    random.seed(seed)

    by_label = defaultdict(list)
    for i, c in enumerate(chunks):
        label = c.get("medical_label", "other")
        parent = c.get("medical_parent") or "none"
        if label == "other":
            continue
        key = label if parent == "none" else f"{parent}>{label}"
        by_label[key].append((i, c))

    selected = []
    all_keys = sorted(by_label.keys(), key=lambda k: -len(by_label[k]))
    if not all_keys:
        random.shuffle(chunks)
        selected = [(i, chunks[i]) for i in range(min(n, len(chunks)))]
        return selected

    per_class = max(1, n // max(1, len(all_keys)))
    for key in all_keys:
        pool = by_label[key]
        pick = min(per_class, len(pool))
        picked = random.sample(pool, pick)
        for idx, chunk in picked:
            selected.append((idx, chunk))

    random.shuffle(selected)
    return selected[:n]


def generate_queries(sampled_chunks, client, model_name, n_queries=None):
    queries = []
    target = n_queries or len(sampled_chunks)

    for idx, chunk in sampled_chunks[:target]:
        chunk_text = chunk.get("text", "")
        if not chunk_text:
            fpath = PROJECT_ROOT / chunk.get("filepath", "")
            if fpath.exists():
                chunk_text = fpath.read_text(encoding="utf-8").strip()
        if not chunk_text:
            continue

        snippet = chunk_text[:2000]

        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "你是一位临床医学研究员，只输出临床问题本身。"},
                    {"role": "user", "content": DEFAULT_QUERY_GEN_PROMPT.format(chunk_text=snippet)},
                ],
                temperature=0.7,
                max_tokens=128,
            )
            question = response.choices[0].message.content.strip().strip('"').strip("'")
        except Exception as e:
            print(f"  [WARN] 查询生成失败 (chunk {chunk.get('chunk_id', idx)}): {e}", flush=True)
            question = chunk.get("source_section", "clinical question")[:80]

        if not question or len(question) < 5:
            question = chunk.get("source_section", "clinical question")[:80]

        queries.append({
            "query_id": _stable_id(idx, chunk.get("medical_label", "gen")),
            "question": question,
            "source_chunk_id": chunk.get("chunk_id", ""),
            "source_label": chunk.get("medical_label", ""),
            "source_parent": chunk.get("medical_parent"),
            "source_pmcid": chunk.get("filename", ""),
        })

    return queries


def dense_only_search(query, index_dir, manifest, k=5,
                      embedding_provider="local", embedding_model="qwen3-embedding-4b"):
    import faiss
    from mineru_medical.embedding_providers import embed_texts

    dense_path = Path(index_dir) / "medical.index"
    if not dense_path.exists():
        raise FileNotFoundError(f"FAISS 索引不存在: {dense_path}")

    index = faiss.read_index(str(dense_path))
    q_emb = embed_texts([query], provider=embedding_provider, model=embedding_model)
    q_emb_norm = q_emb.copy()
    faiss.normalize_L2(q_emb_norm)
    scores, indices = index.search(q_emb_norm, k)

    results = []
    texts = _load_chunk_texts(manifest["chunks"])
    for rank, (idx, score) in enumerate(zip(indices[0], scores[0])):
        if idx < 0 or idx >= len(manifest["chunks"]):
            continue
        chunk = dict(manifest["chunks"][idx])
        chunk["_dense_score"] = round(float(score), 6)
        chunk["_rank"] = rank + 1
        if texts:
            chunk["text"] = texts[idx]
        results.append(chunk)
    return results


def run_retrieval_comparison(queries, index_dir, manifest, top_k=5,
                             embedding_provider="local", embedding_model="qwen3-embedding-4b"):
    comparison = []
    for qi, q in enumerate(queries):
        question = q["question"]
        print(f"  [{qi+1}/{len(queries)}] 检索: {question[:60]}...", flush=True)

        try:
            hybrid_results = search_hybrid(question, index_dir, manifest, k=top_k)
        except Exception as e:
            print(f"    [WARN] Hybrid 检索失败: {e}", flush=True)
            hybrid_results = []

        try:
            dense_results = dense_only_search(question, index_dir, manifest, k=top_k,
                                                embedding_provider=embedding_provider,
                                                embedding_model=embedding_model)
        except Exception as e:
            print(f"    [WARN] Dense-only 检索失败: {e}", flush=True)
            dense_results = []

        hybrid_ids = [r.get("chunk_id", "") for r in hybrid_results]
        dense_ids = [r.get("chunk_id", "") for r in dense_results]
        overlap = set(hybrid_ids) & set(dense_ids)

        comparison.append({
            "query_id": q["query_id"],
            "question": question,
            "source_chunk_id": q["source_chunk_id"],
            "source_label": q["source_label"],
            "hybrid": {
                "chunk_ids": hybrid_ids,
                "count": len(hybrid_results),
                "details": [
                    {"chunk_id": r.get("chunk_id", ""), "rrf_score": r.get("_rrf_score", 0),
                     "label": r.get("medical_label", ""), "text_snippet": r.get("text", "")[:200],
                     "filepath": r.get("filepath", ""),
                     "medical_parent": r.get("medical_parent")}
                    for r in hybrid_results
                ],
            },
            "dense_only": {
                "chunk_ids": dense_ids,
                "count": len(dense_results),
                "details": [
                    {"chunk_id": r.get("chunk_id", ""), "cosine_score": r.get("_dense_score", 0),
                     "label": r.get("medical_label", ""), "text_snippet": r.get("text", "")[:200],
                     "filepath": r.get("filepath", ""),
                     "medical_parent": r.get("medical_parent")}
                    for r in dense_results
                ],
            },
            "overlap_count": len(overlap),
            "overlap_ratio": round(len(overlap) / max(1, len(hybrid_ids), len(dense_ids)), 4),
        })
    return comparison


def _resolve_chunk_text(chunk):
    text = chunk.get("text", "")
    if not text or len(text.strip()) < 100:
        fpath = PROJECT_ROOT / chunk.get("filepath", "")
        if fpath.is_file():
            text = fpath.read_text(encoding="utf-8").strip()
    if not text:
        text = chunk.get("text_snippet", "")
    return text


def judge_single_result(question, chunk, client, model_name, provider="openai"):
    chunk_text = _resolve_chunk_text(chunk)
    if not chunk_text or len(chunk_text.strip()) < 20:
        return {
            "score": 0.0,
            "reason": "chunk text empty or too short",
            "dimensions": {"relevance": 0.0, "completeness": 0.0, "traceability": 0.0, "accuracy": 0.0},
            "issues": ["empty_or_short_chunk"],
        }

    medical_label = chunk.get("label") or chunk.get("medical_label", "unknown")
    medical_parent = chunk.get("medical_parent") or "无"

    user_prompt = MEDICAL_USER_PROMPT_TEMPLATE.format(
        question=question,
        chunk_text=chunk_text[:3000],
        medical_label=medical_label,
        medical_parent=medical_parent,
    )

    try:
        if provider == "anthropic":
            response = client.messages.create(
                model=model_name,
                system=MEDICAL_JUDGE_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_prompt}],
                temperature=0.0,
                max_tokens=1024,
            )
            raw = ""
            for block in response.content:
                if hasattr(block, "text") and block.text:
                    raw += block.text
            if not raw.strip():
                return {
                    "score": 0.0,
                    "reason": "Anthropic returned no text content blocks",
                    "dimensions": {"relevance": 0.0, "completeness": 0.0, "traceability": 0.0, "accuracy": 0.0},
                    "issues": ["empty_anthropic_response"],
                }
        else:
            kwargs = {
                "model": model_name,
                "messages": [
                    {"role": "system", "content": MEDICAL_JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": 0.0,
                "max_tokens": 512,
            }
            response = client.chat.completions.create(**kwargs)
            raw = response.choices[0].message.content
    except Exception as e:
        return {
            "score": 0.0,
            "reason": f"API call failed: {str(e)[:200]}",
            "dimensions": {"relevance": 0.0, "completeness": 0.0, "traceability": 0.0, "accuracy": 0.0},
            "issues": [f"api_error: {str(e)[:100]}"],
        }

    result = _parse_judge_result(raw, VALID_SCORES)
    return result


def _judge_one_model(question, hitem, model_cfg):
    try:
        result = judge_single_result(
            question, hitem,
            model_cfg["client"], model_cfg["model"],
            provider=model_cfg.get("provider", "openai"),
        )
    except Exception as e:
        result = {
            "score": 0.0,
            "reason": f"judge error: {str(e)[:200]}",
            "dimensions": {"relevance": 0.0, "completeness": 0.0, "traceability": 0.0, "accuracy": 0.0},
            "issues": [f"judge_error: {str(e)[:100]}"],
        }
    return model_cfg["name"], result


def _judge_api_models(queries, retrieval_comparison, api_models, top_k_judge, n_workers, total_tasks):
    """Run API model judging in parallel. Returns results_map."""
    results_map = {}
    print(f"  API pool: {n_workers} workers × {len(api_models)} models", flush=True)
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {}
        for qi, q in enumerate(queries):
            comp = retrieval_comparison[qi]
            for ri, hitem in enumerate(comp["hybrid"]["details"][:top_k_judge]):
                for mc in api_models:
                    f = pool.submit(_judge_one_model, q["question"], hitem, mc)
                    futures[f] = (qi, q, ri, hitem)
        api_done = 0
        for future in as_completed(futures):
            qi, q, ri, hitem = futures[future]
            name, result = future.result()
            key = (qi, ri)
            if key not in results_map:
                results_map[key] = {
                    "query_id": q["query_id"],
                    "question": q["question"],
                    "chunk_id": hitem["chunk_id"],
                    "rank": ri + 1,
                    "medical_label": hitem.get("label", ""),
                }
            results_map[key][name] = result
            api_done += 1
            if api_done % 20 == 0:
                print(f"  [API] {api_done}/{total_tasks} ({api_done*100//total_tasks}%)", flush=True)
    return results_map


def _judge_local_models(queries, retrieval_comparison, local_models, results_map, top_k_judge):
    """Run local model judging serially. Returns judge_matrix."""
    judge_matrix = []
    for qi, q in enumerate(queries):
        comp = retrieval_comparison[qi]
        for ri, hitem in enumerate(comp["hybrid"]["details"][:top_k_judge]):
            key = (qi, ri)
            judge_row = results_map.get(key, {
                "query_id": q["query_id"],
                "question": q["question"],
                "chunk_id": hitem["chunk_id"],
                "rank": ri + 1,
                "medical_label": hitem.get("label", ""),
            })
            for mc in local_models:
                name, result = _judge_one_model(q["question"], hitem, mc)
                judge_row[name] = result
            judge_matrix.append(judge_row)
            idx = len(judge_matrix)
            print(f"  Judge [{idx}] query={judge_row['query_id']} chunk={judge_row['chunk_id']} [OK]", flush=True)
    return judge_matrix


def run_judge_matrix(queries, retrieval_comparison, models_config, top_k_judge=3):
    api_models = [m for m in models_config if m.get("provider") != "local"]
    local_models = [m for m in models_config if m.get("provider") == "local"]

    total_tasks = len(queries) * top_k_judge * len(api_models)
    if api_models:
        n_workers = min(16, total_tasks)
        print(f"  API pool: {n_workers} workers × {len(api_models)} models", flush=True)

    results_map = {}
    if api_models:
        results_map = _judge_api_models(queries, retrieval_comparison, api_models, top_k_judge, n_workers, total_tasks)

    judge_matrix = _judge_local_models(queries, retrieval_comparison, local_models, results_map, top_k_judge)

    return judge_matrix


def compute_statistics(judge_matrix, models_config):
    model_names = [m["name"] for m in models_config]
    stats = {"per_model": {}, "overall": {}}

    for mn in model_names:
        scores = []
        dim_scores = defaultdict(list)
        for row in judge_matrix:
            entry = row.get(mn, {})
            s = entry.get("score", 0.0)
            scores.append(s)
            dims = entry.get("dimensions", {})
            for dk, dv in dims.items():
                dim_scores[dk].append(float(dv))

        stats["per_model"][mn] = {
            "mean_score": round(np.mean(scores), 4) if scores else 0.0,
            "median_score": round(np.median(scores), 4) if scores else 0.0,
            "std_score": round(np.std(scores), 4) if scores else 0.0,
            "count": len(scores),
            "dimensions": {
                dk: {
                    "mean": round(np.mean(dv), 4),
                    "std": round(np.std(dv), 4),
                }
                for dk, dv in dim_scores.items() if dv
            },
        }

    all_scores = []
    for mn in model_names:
        for row in judge_matrix:
            entry = row.get(mn, {})
            all_scores.append(entry.get("score", 0.0))

    stats["overall"] = {
        "mean_score": round(np.mean(all_scores), 4) if all_scores else 0.0,
        "median_score": round(np.median(all_scores), 4) if all_scores else 0.0,
        "std_score": round(np.std(all_scores), 4) if all_scores else 0.0,
        "total_judgements": len(all_scores),
    }

    return stats


def compute_retrieval_stats(retrieval_comparison):
    hybrid_counts = [c["hybrid"]["count"] for c in retrieval_comparison]
    dense_counts = [c["dense_only"]["count"] for c in retrieval_comparison]
    overlaps = [c["overlap_count"] for c in retrieval_comparison]
    overlap_ratios = [c["overlap_ratio"] for c in retrieval_comparison]

    return {
        "hybrid_mean_hits": round(np.mean(hybrid_counts), 2),
        "dense_mean_hits": round(np.mean(dense_counts), 2),
        "mean_overlap": round(np.mean(overlaps), 2),
        "mean_overlap_ratio": round(np.mean(overlap_ratios), 4),
        "total_queries": len(retrieval_comparison),
    }


def generate_report(
    queries,
    retrieval_comparison,
    judge_matrix,
    models_config,
    output_path,
    elapsed_s,
    profile="report",
    manifest_path=None,
    index_dir=None,
):
    retrieval_stats = compute_retrieval_stats(retrieval_comparison)
    judge_stats = compute_statistics(judge_matrix, models_config)

    model_names = [m["name"] for m in models_config]

    report = {
        "meta": {
            "title": "医疗 RAG 知识库自评测报告",
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "elapsed_seconds": round(elapsed_s, 1),
            "profile": profile,
            "total_queries": len(queries),
            "total_judge_entries": len(judge_matrix),
            "judge_models": model_names,
            "chunk_source": str(manifest_path) if manifest_path is not None else "data/sources_medical/chunks/chunks_manifest.json",
            "index_source": str(index_dir) if index_dir is not None else "data/sources_medical/index/",
        },
        "retrieval_comparison": {
            "summary": retrieval_stats,
            "per_query": retrieval_comparison,
        },
        "judge_matrix": {
            "summary": judge_stats,
            "per_entry": judge_matrix,
        },
        "queries": queries,
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    tmp_path = str(output_path) + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, str(output_path))
    print(f"\n[Report] {output_path} ({os.path.getsize(output_path) / 1024:.0f} KB)", flush=True)

    print_summary(retrieval_stats, judge_stats, model_names)
    return report


def print_summary(retrieval_stats, judge_stats, model_names):
    print("\n" + "=" * 60)
    print("  知识库自评测报告摘要")
    print("=" * 60)

    print(f"\n  [Level 1] 检索质量 (n={retrieval_stats['total_queries']})")
    print(f"    Hybrid 平均命中:     {retrieval_stats['hybrid_mean_hits']}")
    print(f"    Dense-only 平均命中: {retrieval_stats['dense_mean_hits']}")
    print(f"    平均重叠数:          {retrieval_stats['mean_overlap']}")
    print(f"    平均重叠率:          {retrieval_stats['mean_overlap_ratio']:.2%}")

    print("\n  [Level 3] Judge 矩阵评分")
    for mn in model_names:
        pm = judge_stats["per_model"].get(mn, {})
        dims = pm.get("dimensions", {})
        print(f"    {mn}:")
        print(f"      score μ={pm.get('mean_score', '-')} σ={pm.get('std_score', '-')} (n={pm.get('count', '-')})")
        for dk, dv in dims.items():
            print(f"        {dk}: μ={dv['mean']} σ={dv['std']}")

    print("\n  [Overall]")
    ov = judge_stats.get("overall", {})
    print(f"    总体平均分: {ov.get('mean_score', '-')}")
    print(f"    总评测次数: {ov.get('total_judgements', '-')}")

    print("\n  [Level 2] QLoRA 微调前后对比: 待微调模型就绪后补充")
    print("  [Level 4] MedBench 对齐: 待下载 MedBench 测试集后补充")
    print("=" * 60)


API_MODELS = [
    {"name": "ds-v4-flash",  "base_url": PROVIDERS["deepseek"].base_url,  "model": "deepseek-v4-flash",       "key_env": PROVIDERS["deepseek"].api_key_env},
    {"name": "ds-v4-pro",    "base_url": PROVIDERS["deepseek"].base_url,  "model": "deepseek-v4-pro",         "key_env": PROVIDERS["deepseek"].api_key_env},
    {"name": "mimo-v2-flash","base_url": PROVIDERS["mimo"].base_url,      "model": "mimo-v2-flash",           "key_env": PROVIDERS["mimo"].api_key_env},
    {"name": "mimo-v2-pro",  "base_url": PROVIDERS["mimo"].base_url,      "model": "mimo-v2-pro",             "key_env": PROVIDERS["mimo"].api_key_env},
    {"name": "mimo-v2.5",    "base_url": PROVIDERS["mimo"].base_url,      "model": "mimo-v2.5",               "key_env": PROVIDERS["mimo"].api_key_env},
    {"name": "mimo-v2.5-pro","base_url": PROVIDERS["mimo"].base_url,      "model": "mimo-v2.5-pro",           "key_env": PROVIDERS["mimo"].api_key_env},
    {"name": "mm-m2.5",      "base_url": PROVIDERS["minimax"].base_url,   "model": "MiniMax-M2.5-highspeed", "key_env": PROVIDERS["minimax"].api_key_env},
    {"name": "mm-m2.7",      "base_url": PROVIDERS["minimax"].base_url,   "model": "MiniMax-M2.7-highspeed", "key_env": PROVIDERS["minimax"].api_key_env},
]


def _register_api_models():
    models = []
    for cfg in API_MODELS:
        key = os.environ.get(cfg["key_env"])
        if not key:
            print(f"  [WARN] {cfg['key_env']} 未设置，跳过 {cfg['name']}", flush=True)
            continue
        try:
            client, model, provider = create_client(api_key=key, base_url=cfg["base_url"], model=cfg["model"])
            entry = {"name": cfg["name"], "client": client, "model": model, "provider": provider}
            models.append(entry)
            print(f"  {cfg['name']} 就绪 (provider={provider})", flush=True)
        except Exception as e:
            print(f"  [WARN] {cfg['name']} 不可用: {e}", flush=True)
    return models


def _register_local_models(model_names):
    """Register judge models via llama.cpp OpenAI-compatible API."""
    from openai import OpenAI, DefaultHttpxClient

    port = os.environ.get("LLAMA_CHAT_PORT", "8080")
    base_url = f"http://localhost:{port}/v1"
    models = []
    for m in model_names:
        m = m.strip()
        if not m:
            continue
        try:
            client = OpenAI(
                api_key="no-key-required",
                base_url=base_url,
                timeout=120,
                max_retries=0,
                http_client=DefaultHttpxClient(proxy=None, trust_env=False),
            )
            models.append({"name": m, "client": client, "model": m, "provider": "openai"})
            print(f"  LLM {m} ready (provider=openai)", flush=True)
        except Exception as e:
            print(f"  [WARN] LLM {m} unavailable: {e}", flush=True)
    return models


def _resolve_local_list(raw):
    if not raw or raw.strip() == "":
        return []
    return [m.strip() for m in raw.split(",") if m.strip()]


def _available_local_models(timeout_s=2):
    """Check if llama.cpp chat server is healthy, return available model names."""
    from mineru_tools.llm_client import LocalLLMClient

    client = LocalLLMClient(timeout=float(timeout_s))
    if client.is_chat_healthy():
        return {"qwen3.5-9b", "qwen3-embedding-4b"}
    return set()


def _resolve_profile(args):
    profile = args.profile.lower()
    if profile == "smoke":
        args.n_queries = args.n_queries if args.n_queries is not None else 1
        args.top_k = args.top_k if args.top_k is not None else 3
        args.top_k_judge = args.top_k_judge if args.top_k_judge is not None else 1
        args.skip_api = args.skip_api or not args.include_api
        args.skip_query_gen = True if args.skip_query_gen is None else args.skip_query_gen
        if args.local_models is None:
            available = _available_local_models()
            selected = next((m for m in SMOKE_LOCAL_CANDIDATES if m in available), None)
            args.local_models = selected or SMOKE_LOCAL_CANDIDATES[0]
    else:
        args.n_queries = args.n_queries if args.n_queries is not None else 25
        args.top_k = args.top_k if args.top_k is not None else 5
        args.top_k_judge = args.top_k_judge if args.top_k_judge is not None else 3
        args.skip_query_gen = False if args.skip_query_gen is None else args.skip_query_gen
        if args.local_models is None:
            args.local_models = ",".join(REPORT_LOCAL_MODELS)
    return args


def _validate_kb_paths(manifest_path, index_dir, output_path):
    """Validate that all required paths exist; print init info."""
    print(f"[Init] manifest={manifest_path}", flush=True)
    print(f"[Init] index={index_dir}", flush=True)
    print(f"[Init] output={output_path}", flush=True)

    if not manifest_path.exists():
        print(f"ERROR: manifest \u4e0d\u5b58\u5728: {manifest_path}", flush=True)
        sys.exit(1)
    dense_path = index_dir / "medical.index"
    sparse_path = index_dir / "bm25.pkl"
    if not dense_path.exists():
        print(f"ERROR: FAISS \u7d22\u5f15\u4e0d\u5b58\u5728: {dense_path}", flush=True)
        sys.exit(1)
    if not sparse_path.exists():
        print(f"ERROR: BM25 \u7d22\u5f15\u4e0d\u5b58\u5728: {sparse_path}", flush=True)
        sys.exit(1)
    return dense_path, sparse_path


def _run_dry_check(dense_path, sparse_path, index_dir, manifest):
    """Run dry-run availability check."""
    print("\n[Dry-Run] index availability check passed [OK]", flush=True)
    print(f"  FAISS: {dense_path.stat().st_size / 1024:.0f} KB", flush=True)
    print(f"  BM25:  {sparse_path.stat().st_size / 1024:.0f} KB", flush=True)
    try:
        test_results = search_hybrid("primary endpoint", index_dir, manifest, k=3)
        print(f"  \u6d4b\u8bd5\u68c0\u7d22 'primary endpoint' \u2192 {len(test_results)} results:", flush=True)
        for r in test_results:
            print(f"    {r.get('chunk_id')} rrf={r.get('_rrf_score')} label={r.get('medical_label')}", flush=True)
    except Exception as e:
        print(f"  [WARN] 测试检索失败 (可能 llama.cpp embedding 服务未运行): {e}", flush=True)
        print("  [INFO] 索引文件完整，检索通路需 llama.cpp embedding 服务 (port 8081) 运行后方可验证", flush=True)


def main():
    parser = argparse.ArgumentParser(description="医疗 RAG 知识库自评测")
    parser.add_argument("--profile", choices=["smoke", "report"], default="smoke",
                        help="评测策略: smoke=高可用最小真实测试; report=报告级多 Judge 矩阵")
    parser.add_argument("--manifest", default=str(PROJECT_ROOT / "data" / "sources_medical" / "chunks" / "chunks_manifest.json"))
    parser.add_argument("--index-dir", default=str(PROJECT_ROOT / "data" / "sources_medical" / "index"))
    parser.add_argument("--output", default=str(PROJECT_ROOT / "benchmark" / "eval" / "results" / "medical" / "kb_quality_report.json"))
    parser.add_argument("--n-queries", type=int, default=None, help="生成的评估查询数量 (默认随 profile)")
    parser.add_argument("--top-k", type=int, default=None, help="检索返回 top-k (默认随 profile)")
    parser.add_argument("--top-k-judge", type=int, default=None, help="每查询 Judge 评估 top-k 结果 (默认随 profile)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-api", action="store_true", help="跳过 API Judge (仅用本地 LLM)")
    parser.add_argument("--include-api", action="store_true", help="smoke profile 中显式启用 API Judge")
    parser.add_argument("--skip-local", action="store_true", help="跳过本地 LLM Judge (仅用 API)")
    parser.add_argument("--local-models", default=None, help="本地 LLM 模型列表，逗号分隔 (默认随 profile)")
    parser.add_argument("--skip-query-gen", action=argparse.BooleanOptionalAction, default=None,
                        help="跳过 LLM 查询生成，使用 section 标题作为查询")
    parser.add_argument("--dry-run", action="store_true", help="仅验证配置和索引可用性，不运行完整评测")
    parser.add_argument("--embedding-provider", default="local", choices=["local", "ollama", "hf", "hf_local", "transformers"],
                        help="嵌入向量 Provider (默认 local)")
    parser.add_argument("--embedding-model", default="qwen3-embedding-4b",
                        help="嵌入模型名 (local) 或 HuggingFace 路径 (hf)")

    args = parser.parse_args()
    args = _resolve_profile(args)

    manifest_path = Path(args.manifest)
    index_dir = Path(args.index_dir)
    output_path = Path(args.output)

    dense_path, sparse_path = _validate_kb_paths(manifest_path, index_dir, output_path)
    print(f"[Init] profile={args.profile} n_queries={args.n_queries} top_k={args.top_k}", flush=True)

    manifest = load_manifest(manifest_path)
    print(f"[Init] {manifest['total_chunks']} chunks loaded", flush=True)

    if args.dry_run:
        _run_dry_check(dense_path, sparse_path, index_dir, manifest)
        return

    t_start = time.time()

    print("\n[Phase 1] 采样 chunk...", flush=True)
    sampled = sample_chunks(manifest, n=args.n_queries, seed=args.seed)
    print(f"  采样 {len(sampled)} chunks (来自 {len(set(c.get('medical_label','') for _,c in sampled))} 个章节标签)", flush=True)

    if args.skip_query_gen:
        print("[Phase 2] 使用 section 标题作为查询 (跳过 LLM 生成)", flush=True)
        queries = []
        for idx, chunk in sampled:
            queries.append({
                "query_id": _stable_id(idx, chunk.get("medical_label", "gen")),
                "question": chunk.get("source_section", "clinical question")[:120],
                "source_chunk_id": chunk.get("chunk_id", ""),
                "source_label": chunk.get("medical_label", ""),
                "source_parent": chunk.get("medical_parent"),
                "source_pmcid": chunk.get("filename", ""),
            })
    else:
        print("[Phase 2] 生成 MedBench 风格临床查询...", flush=True)
        gen_client, gen_model, _ = create_client(
            api_key=os.environ.get("DEEPSEEK_API_KEY"),
            base_url=PROVIDERS["deepseek"].base_url,
            model=PROVIDERS["deepseek"].default_model,
        )
        queries = generate_queries(sampled, gen_client, gen_model, n_queries=args.n_queries)
        print(f"  生成 {len(queries)} 条查询", flush=True)
        for q in queries[:3]:
            print(f"    [{q['query_id']}] {q['question'][:80]} (label={q['source_label']})", flush=True)

    print(f"\n[Phase 3] 检索对比 (Hybrid vs Dense-only, k={args.top_k})...", flush=True)
    retrieval_comparison = run_retrieval_comparison(queries, index_dir, manifest, top_k=args.top_k,
                                                     embedding_provider=args.embedding_provider,
                                                     embedding_model=args.embedding_model)

    print("\n[Phase 4] Judge 矩阵评分...", flush=True)
    models_config = []

    if not args.skip_api:
        api_models = _register_api_models()
        models_config.extend(api_models)

    if not args.skip_local:
        local_list = _resolve_local_list(args.local_models)
        if local_list:
            local_models = _register_local_models(local_list)
            models_config.extend(local_models)

    if len(models_config) < 1:
        print("ERROR: 无可用 Judge 模型，请检查 API Key 和 Ollama", flush=True)
        sys.exit(1)

    print(f"  Judge 模型: {[m['name'] for m in models_config]}", flush=True)

    judge_matrix = run_judge_matrix(queries, retrieval_comparison, models_config, top_k_judge=args.top_k_judge)

    elapsed = time.time() - t_start
    print(f"\n[Phase 5] 生成报告... ({elapsed:.0f}s)", flush=True)
    generate_report(
        queries,
        retrieval_comparison,
        judge_matrix,
        models_config,
        output_path,
        elapsed,
        profile=args.profile,
        manifest_path=manifest_path,
        index_dir=index_dir,
    )

    print(f"\nDone [OK] ({elapsed:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
