"""A3: Chunk 级冗余检测 — MiniBatchKMeans 分桶 + 组内 cosine 去重"""

import sys

try:
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.preprocessing import normalize
except ImportError as e:
    print(f"缺少依赖: {e}")
    print("请安装: .venv/bin/python -m pip install scikit-learn")
    sys.exit(1)

import json
import time
from collections import Counter, defaultdict

import numpy as np


def _discover_redundant_pairs(emb_norm, chunk_metas, labels, cos_threshold):
    """Find redundant chunk pairs within each cluster."""
    redundant_pairs = []
    checked_pairs = 0
    cluster_sizes = Counter(labels)
    t0 = time.time()

    for c_id in sorted(cluster_sizes.keys()):
        indices = np.where(labels == c_id)[0]
        if len(indices) < 2:
            continue
        cluster_emb = emb_norm[indices]
        cos_mat = cluster_emb @ cluster_emb.T
        for i in range(len(indices)):
            idx_i = int(indices[i])
            for j in range(i + 1, len(indices)):
                checked_pairs += 1
                cos_val = float(cos_mat[i, j])
                if cos_val >= cos_threshold:
                    idx_j = int(indices[j])
                    redundant_pairs.append({
                        "chunk_a_id": chunk_metas[idx_i]["chunk_id"],
                        "chunk_b_id": chunk_metas[idx_j]["chunk_id"],
                        "cosine_similarity": round(cos_val, 4),
                        "same_doc": chunk_metas[idx_i]["filename"] == chunk_metas[idx_j]["filename"],
                    })

    elapsed = time.time() - t0
    return redundant_pairs, checked_pairs, elapsed


def _print_redundancy_report(redundant_pairs, chunk_metas, cos_threshold, checked_pairs, elapsed):
    """Print redundancy analysis report."""
    print(f"  \u904d\u5386 {checked_pairs} \u5bf9 ({elapsed:.1f}s), \u53d1\u73b0 {len(redundant_pairs)} \u5bf9\u5197\u4f59 (cos \u2265 {cos_threshold})", flush=True)

    redundant_pairs.sort(key=lambda x: x["cosine_similarity"], reverse=True)

    same_doc = [p for p in redundant_pairs if p["same_doc"]]
    cross_doc = [p for p in redundant_pairs if not p["same_doc"]]
    print(f"    \u540c\u6587\u6863\u5185\u5197\u4f59: {len(same_doc)} \u5bf9", flush=True)
    print(f"    \u8de8\u6587\u6863\u5197\u4f59: {len(cross_doc)} \u5bf9", flush=True)

    doc_redundancy = defaultdict(lambda: {"total": 0, "redundant_chunks": set()})
    for m in chunk_metas:
        doc_redundancy[m["filename"]]["total"] += 1
    chunk_id_to_meta = {m["chunk_id"]: m for m in chunk_metas}
    for p in same_doc:
        fn = chunk_id_to_meta[p["chunk_a_id"]]["filename"]
        doc_redundancy[fn]["redundant_chunks"].add(p["chunk_a_id"])

    high_redundancy_docs = sorted(
        [(fn, v) for fn, v in doc_redundancy.items() if v["total"] >= 10 and len(v["redundant_chunks"]) > 0],
        key=lambda x: len(x[1]["redundant_chunks"]) / x[1]["total"],
        reverse=True,
    )[:10]

    print("\n  Top-10 \u5197\u4f59\u7387\u6700\u9ad8\u6587\u6863:", flush=True)
    for fn, v in high_redundancy_docs:
        unique_count = len(v["redundant_chunks"])
        rate = unique_count / v["total"] * 100
        print(f"    {fn[:60]}: {unique_count}/{v['total']} ({rate:.1f}%)", flush=True)

    if cross_doc:
        print("\n  Top-10 \u8de8\u6587\u6863\u5197\u4f59\u5bf9:", flush=True)
        for pair in cross_doc[:10]:
            same_flag = "\u540c\u6587\u6863" if pair["same_doc"] else "\u8de8\u6587\u6863"
            print(f"    cos={pair['cosine_similarity']:.4f} [{same_flag}] "
                  f"{pair['chunk_a_id'][:50]} \u2194 {pair['chunk_b_id'][:50]}", flush=True)

    return same_doc, cross_doc, high_redundancy_docs


def analysis_a3_redundancy(embeddings, chunk_metas, output_dir, cos_threshold=0.95):
    """A3: Chunk \u7ea7\u5197\u4f59\u68c0\u6d4b — MiniBatchKMeans \u5206\u6876 + \u7ec4\u5185 cosine \u53bb\u91cd"""
    print("\n" + "=" * 60)
    print(f"A3: Chunk \u7ea7\u5197\u4f59\u68c0\u6d4b (cos > {cos_threshold})")
    print("=" * 60)

    n = embeddings.shape[0]
    n_clusters = min(500, max(2, n // 50))
    if n_clusters > n:
        n_clusters = max(1, n)
    print(f"  L2 \u5f52\u4e00\u5316 \u2192 MiniBatchKMeans k={n_clusters}...", flush=True)

    emb_norm = normalize(embeddings, norm="l2")
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42,
                              batch_size=200, n_init=3, max_iter=100)
    labels = kmeans.fit_predict(emb_norm)

    cluster_sizes = Counter(labels)
    print(f"  \u805a\u7c7b\u5b8c\u6210: {len(cluster_sizes)} \u4e2a\u975e\u7a7a\u7c07, "
           f"max_size={max(cluster_sizes.values())}, mean_size={n/len(cluster_sizes):.1f}", flush=True)

    redundant_pairs, checked_pairs, elapsed = _discover_redundant_pairs(emb_norm, chunk_metas, labels, cos_threshold)
    same_doc, cross_doc, high_redundancy_docs = _print_redundancy_report(
        redundant_pairs, chunk_metas, cos_threshold, checked_pairs, elapsed
    )

    result = {
        "analysis": "A3_redundancy",
        "cosine_threshold": cos_threshold,
        "n_chunks": n,
        "n_clusters": n_clusters,
        "checked_pairs": checked_pairs,
        "redundant_pairs_total": len(redundant_pairs),
        "same_doc_redundant": len(same_doc),
        "cross_doc_redundant": len(cross_doc),
        "top_same_doc_sample": [
            {k: v for k, v in p.items() if k != "same_doc"}
            for p in same_doc[:10]
        ],
        "top_cross_doc_sample": [
            {k: v for k, v in p.items() if k != "same_doc"}
            for p in cross_doc[:10]
        ],
        "high_redundancy_docs": [
            {"filename": fn, "total_chunks": v["total"],
             "redundant_chunk_count": len(v["redundant_chunks"]),
             "redundancy_rate": round(len(v["redundant_chunks"]) / v["total"], 4)}
            for fn, v in high_redundancy_docs
        ],
        "interpretation": (
            f"\u5728 {n} chunks \u4e2d\u53d1\u73b0 {len(redundant_pairs)} \u5bf9 cos \u2265 {cos_threshold} \u7684\u8fd1\u4f3c\u91cd\u590d\u3002"
            f"\u540c\u6587\u6863\u5185 {len(same_doc)} \u5bf9 \u2192 \u53ef\u80fd\u4e3a\u516c\u5f0f/\u5217\u8868\u91cd\u590d\uff08\u4f4e\u9891\uff0c\u22641% \u5c5e\u6b63\u5e38\uff09\uff1b"
            f"\u8de8\u6587\u6863 {len(cross_doc)} \u5bf9 \u2192 \u53ef\u80fd\u4e3a\u5f15\u7528\u6587\u5b57\u91cd\u53e0\u6216\u7ffb\u8bd1\u4e00\u81f4\uff08\u9700\u4eba\u5de5\u62bd\u67e5\uff09\u3002"
        ),
    }

    json_path = output_dir / "a3_redundancy.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"  JSON \u5df2\u4fdd\u5b58: {json_path}", flush=True)
    return result
