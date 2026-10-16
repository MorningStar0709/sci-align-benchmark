#!/usr/bin/env python
"""Task 2c: Embedding model analysis

2c-1: Semantic clustering validation — PCA 2D scatter plot, colored by A/B/C/D
2c-2: Embedding arbitration vs LLM arbitration consistency — Spearman ρ
2c-3: Failure case embedding clustering — KMeans to discover knowledge blind spots

Prerequisites: llama.cpp embedding server running on port 8081
Time: embedding computation ~4min + analysis ~1min = ~5min
"""

import argparse
import json
import sys
import time
from pathlib import Path
from collections import defaultdict

import numpy as np

from mineru_tools.llm_client import LocalLLMClient

# llama.cpp embedding server (OpenAI-compatible /v1/embeddings endpoint)
_llm_client = LocalLLMClient()
EMBED_MODEL = "qwen3-embedding-4b"
BATCH_SIZE = 10
EMBED_DIM = 2560


def get_embeddings(texts, progress_label="embedding"):
    """Batch get embedding vectors via llama.cpp, returns (n, 2560) ndarray."""
    if not texts:
        print(f"  [{progress_label}] Empty input, returning empty array", flush=True)
        return np.array([], dtype=np.float32)
    n = len(texts)
    all_emb = []
    t0 = time.time()
    for i in range(0, n, BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        result = _llm_client.embed(batch, model=EMBED_MODEL)
        if not result.embeddings:
            raise RuntimeError(f"LLM returned empty embeddings for batch starting at {i}")
        all_emb.extend(result.embeddings)
        done = min(i + BATCH_SIZE, n)
        if done % 50 == 0 or done >= n:
            elapsed = time.time() - t0
            rate = done / elapsed if elapsed > 0 else 0
            print(f"  [{progress_label}] {done}/{n} ({done/n*100:.0f}%) — {rate:.1f} items/s", flush=True)
    elapsed = time.time() - t0
    print(f"  [{progress_label}] Completed {n} items, total time {elapsed:.1f}s, avg {n/elapsed:.1f} items/s", flush=True)
    return np.array(all_emb, dtype=np.float32)


def load_core_questions(core_path):
    """加载 core.json，返回 (questions_list, id->question map)"""
    with open(core_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    questions = data.get("questions", data) if isinstance(data, dict) else data
    id_map = {q["id"]: q for q in questions}
    return questions, id_map


def load_reports(reports_dir):
    """加载所有 *_report.json（跳过 chunk/cross_domain/progress），返回 {model: report}"""
    reports = {}
    for p in sorted(Path(reports_dir).glob("*_report.json")):
        name = p.name
        if "_chunk_" in name or "cross_domain" in name:
            continue
        if "leaderboard" in name or "kappa" in name:
            continue
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        records = data.get("records", [])
        if records:
            reports[data.get("model", p.stem.replace("_report", ""))] = data
    return reports


def analysis_2c1(core_path, output_base):
    """2c-1: 四维语义聚类验证"""
    print("\n" + "=" * 60)
    print("2c-1: 四维语义聚类验证 (PCA 2D)")
    print("=" * 60)

    from sklearn.decomposition import PCA
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    questions, _ = load_core_questions(core_path)
    dim_order = ["A", "B", "C", "D"]
    dim_colors = {"A": "#E74C3C", "B": "#3498DB", "C": "#2ECC71", "D": "#F39C12"}
    dim_labels_cn = {"A": "Conceptual", "B": "Reasoning", "C": "Sensitivity", "D": "Design"}

    texts = [f"{q['question']}\n{q['answer']}" for q in questions]
    dims = [q["dimension"] for q in questions]

    print(f"加载 {len(questions)} 题, 维度分布: { {d: dims.count(d) for d in dim_order} }")

    embeddings = get_embeddings(texts, progress_label="2c-1 embed")

    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(embeddings)
    evr = pca.explained_variance_ratio_

    print(f"PCA 解释方差: PC1={evr[0]:.3f}, PC2={evr[1]:.3f}, 合计={sum(evr):.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # 左: 四维着色散点
    ax = axes[0]
    for d in dim_order:
        mask = np.array([di == d for di in dims])
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            c=dim_colors[d], label=f"{d}: {dim_labels_cn[d]}", alpha=0.7, s=20, edgecolors="none",
        )
    ax.set_xlabel(f"PC1 ({evr[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({evr[1]*100:.1f}%)")
    ax.set_title("4-Dimension Semantic PCA Projection\n(500 Questions × qwen3-embedding:4b)")
    ax.legend(markerscale=2, fontsize=9)
    ax.grid(True, alpha=0.3)

    # 右: 各维质心 + 95% 椭圆
    ax = axes[1]
    from matplotlib.patches import Ellipse
    from scipy.stats import chi2
    chi2_scale = np.sqrt(chi2.ppf(0.95, df=2))
    for d in dim_order:
        mask = np.array([di == d for di in dims])
        pts = coords[mask]
        centroid = pts.mean(axis=0)
        cov = np.cov(pts.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
        width, height = 2 * chi2_scale * np.sqrt(eigenvalues)
        ell = Ellipse(
            xy=centroid, width=width, height=height, angle=angle,
            edgecolor=dim_colors[d], facecolor=dim_colors[d], alpha=0.12, linewidth=1.5,
        )
        ax.add_patch(ell)
        ax.scatter(*centroid, c=dim_colors[d], s=100, marker="X", edgecolors="black", linewidths=0.8, zorder=5)
        ax.annotate(
            f"{d}", centroid, textcoords="offset points", xytext=(8, 8),
            fontsize=11, fontweight="bold", color=dim_colors[d],
        )
    ax.set_xlabel(f"PC1 ({evr[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({evr[1]*100:.1f}%)")
    ax.set_title("4-Dimension Centroids + 95% Confidence Ellipses")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    png_path = f"{output_base}.png"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  PCA 图已保存: {png_path}")

    # 质心距离矩阵
    centroids = {}
    for d in dim_order:
        mask = [di == d for di in dims]
        centroids[d] = coords[mask].mean(axis=0)

    centroid_distances = {}
    for i, d1 in enumerate(dim_order):
        for d2 in dim_order[i + 1 :]:
            dist = float(np.linalg.norm(centroids[d1] - centroids[d2]))
            centroid_distances[f"{d1}-{d2}"] = round(dist, 4)

    result = {
        "analysis": "2c-1_four_dimension_pca",
        "n_questions": len(questions),
        "dimension_counts": {d: dims.count(d) for d in dim_order},
        "pca_variance_explained": [round(float(v), 4) for v in evr],
        "centroid_2d": {d: [round(float(v), 4) for v in centroids[d]] for d in dim_order},
        "centroid_distances": centroid_distances,
        "interpretation": (
            "四维在嵌入空间中部分重叠（尤其 A/B），反映 L1→L2 的科学发现连续谱；"
            "C/D 维度有独立聚集趋势，说明 sensitivity 和 design 有独特语义特征。"
        ),
    }

    json_path = f"{output_base}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"  JSON 已保存: {json_path}")

    return result


def _compute_cross_judge_correlation(reports_dir, cross_judge_path):
    """Compute cross_judge embedding vs LLM score correlation."""
    from scipy.stats import spearmanr

    with open(cross_judge_path, "r", encoding="utf-8") as f:
        cj = json.load(f)
    source_model = cj["meta"]["source_model"]
    cj_records = [r for r in cj["records"] if r.get("score", -1) >= 0]

    report_path = Path(reports_dir) / f"{source_model}_report.json"
    if not report_path.exists():
        print(f"  \u26a0 \u672a\u627e\u5230\u6e90\u6a21\u578b\u62a5\u544a {report_path}\uff0c\u8df3\u8fc7 cross_judge \u5bf9\u6bd4", flush=True)
        return source_model, {}

    with open(report_path, "r", encoding="utf-8") as f:
        source_report = json.load(f)
    report_map = {r["id"]: r for r in source_report["records"]}

    q_agg = {}
    for rec in cj_records:
        qid = rec["question_id"]
        if qid not in report_map:
            continue
        if qid not in q_agg:
            mr = report_map[qid]
            q_agg[qid] = {
                "model_answer": mr.get("model_answer", ""),
                "expected_answer": mr.get("expected_answer", ""),
                "judge_scores": [],
                "judges": [],
            }
        q_agg[qid]["judge_scores"].append(rec["score"])
        q_agg[qid]["judges"].append(rec["judge"])

    if not q_agg:
        return source_model, {"note": "cross_judge \u65e0\u5339\u914d\u5230 report \u4e2d\u7684\u9898\u76ee"}

    qids_ordered = sorted(q_agg.keys())
    all_texts = []
    for qid in qids_ordered:
        all_texts.append(q_agg[qid]["model_answer"])
        all_texts.append(q_agg[qid]["expected_answer"])
    print(f"  \u5d4c\u5165 {len(all_texts)} \u6761\u6587\u672c ({len(qids_ordered)} \u9898\u76ee \u00d7 \u6a21\u578b\u56de\u7b54+\u6807\u51c6\u7b54\u6848)...", flush=True)
    all_emb = get_embeddings(all_texts, progress_label="2c-2 cross-judge embed")

    cos_vals = []
    avg_scores = []
    for i, qid in enumerate(qids_ordered):
        cos = float(np.dot(all_emb[i * 2], all_emb[i * 2 + 1]))
        cos_vals.append(cos)
        avg_scores.append(np.mean(q_agg[qid]["judge_scores"]))

    if len(set(avg_scores)) >= 2:
        rho, pv = spearmanr(avg_scores, cos_vals)
        cj_correlations = {
            "spearman_rho": round(float(rho), 4),
            "p_value": round(float(pv), 4),
            "n_questions": len(qids_ordered),
        }
        print(f"  [cross_judge question-level] Spearman \u03c1={rho:.4f}, p={pv:.4f}, n={len(qids_ordered)} questions", flush=True)
        from collections import defaultdict
        judge_counts = defaultdict(int)
        for qid in qids_ordered:
            for j in q_agg[qid]["judges"]:
                judge_counts[j] += 1
        cj_correlations["judge_sample_counts"] = dict(judge_counts)
    else:
        cj_correlations = {"spearman_rho": None, "p_value": None, "n_questions": len(qids_ordered), "note": "\u5206\u6570\u65b9\u5dee\u4e0d\u8db3"}

    return source_model, cj_correlations


def _compute_full_report_correlation(reports, preferred):
    """Compute full-report embedding vs score correlation for preferred models."""
    from scipy.stats import spearmanr
    from collections import defaultdict

    print(f"\n  \u627e\u5230 {len(reports)} \u4e2a\u6a21\u578b\u62a5\u544a\uff0c\u9009\u4ee3\u8868\u6027\u6a21\u578b\u505a\u5168\u91cf cos vs score \u5bf9\u6bd4...", flush=True)

    full_correlations = {}
    for model_name in preferred:
        if model_name not in reports:
            continue
        rep = reports[model_name]
        records = rep["records"]
        print(f"\n  --- {model_name} ({len(records)} records) ---", flush=True)

        answers_texts = []
        for r in records:
            answers_texts.append(r.get("model_answer", ""))
            answers_texts.append(r.get("expected_answer", ""))
        print(f"  \u5d4c\u5165 {len(answers_texts)} \u6761\u6587\u672c...", flush=True)
        all_emb = get_embeddings(answers_texts, progress_label=f"2c-2 {model_name}")

        cos_vals = []
        scores = []
        for i, r in enumerate(records):
            v1 = all_emb[i * 2]
            v2 = all_emb[i * 2 + 1]
            cos_vals.append(float(np.dot(v1, v2)))
            scores.append(r["score"])

        if len(set(scores)) >= 2:
            rho, pv = spearmanr(scores, cos_vals)
            full_correlations[model_name] = {
                "spearman_rho": round(float(rho), 4),
                "p_value": round(float(pv), 4),
                "n": len(records),
            }
            print(f"  [{model_name}] Spearman \u03c1={rho:.4f}, p={pv:.4f}, n={len(records)}", flush=True)

            bins = defaultdict(list)
            for s, c in zip(scores, cos_vals):
                bins[s].append(c)
            print("  \u5404\u5206\u6570\u6863 cos_sim \u5747\u503c:", flush=True)
            for s_val in sorted(bins):
                print(f"    score={s_val}: cos_mean={np.mean(bins[s_val]):.4f}, n={len(bins[s_val])}", flush=True)
        break

    return full_correlations


def analysis_2c2(reports_dir, cross_judge_path, output_base):
    """2c-2: \u5d4c\u5165\u4ef2\u88c1 vs LLM \u4ef2\u88c1\u4e00\u81f4\u6027"""
    print("\n" + "=" * 60)
    print("2c-2: \u5d4c\u5165\u4ef2\u88c1 vs LLM \u4ef2\u88c1\u4e00\u81f4\u6027 (Spearman \u03c1)")
    print("=" * 60)

    source_model, cj_correlations = _compute_cross_judge_correlation(reports_dir, cross_judge_path)

    reports = load_reports(reports_dir)
    preferred = ["qwen3.5:9b", "deepseek-v4-flash"]
    full_correlations = _compute_full_report_correlation(reports, preferred)

    result = {
        "analysis": "2c-2_embedding_vs_llm_correlation",
        "cross_judge": {
            "source_model": source_model,
            "description": "\u4f7f\u7528 cross_judge_kappa \u6570\u636e, question-level embedding cos \u4e0e\u591a judge \u5747\u503c score \u7684 Spearman \u76f8\u5173\u6027",
            "correlation": cj_correlations,
        },
        "full_report": full_correlations,
        "interpretation": (
            "\u82e5 \u03c1 > 0.5: \u5d4c\u5165 cos \u4e0e LLM \u5224\u5206\u4e2d\u7b49\u4ee5\u4e0a\u76f8\u5173 \u2192 embedding \u5feb\u7b5b\u6709\u4e00\u5b9a\u53c2\u8003\u4ef7\u503c; "
            "\u82e5 \u03c1 < 0.3: \u5d4c\u5165 cos \u4e0e LLM \u5224\u5206\u5f31\u76f8\u5173 \u2192 LLM \u4ef2\u88c1\u4e0d\u53ef\u66ff\u4ee3, "
            "\u9a8c\u8bc1\u4e86\u5d4c\u5165\u6a21\u578b\u300c\u65e0\u6cd5\u533a\u5206\u6982\u5ff5\u6b63\u786e\u6027\u300d\u7684\u7ed3\u8bba (cos\u22480.97 \u7684\u7b54\u6848\u53ef\u80fd\u5f97 0 \u5206)\u3002"
        ),
    }

    json_path = f"{output_base}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n  JSON \u5df2\u4fdd\u5b58: {json_path}")

    return result


def _build_kmeans_clusters(embeddings, texts, meta):
    """Perform KMeans clustering and build cluster summaries."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    if len(texts) < 3:
        print("  \u26a0 \u4f4e\u5206\u9898\u4e0d\u8db3 3 \u9898\uff0c\u8df3\u8fc7\u805a\u7c7b", flush=True)
        return None, 0, 0

    k_range = range(2, min(8, len(texts)))
    best_k = 3
    best_score = -1
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(embeddings)
        if len(set(labels)) < 2:
            continue
        sil = silhouette_score(embeddings, labels)
        print(f"  k={k}: silhouette={sil:.4f}", flush=True)
        if sil > best_score:
            best_score = sil
            best_k = k

    km = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    labels = km.fit_predict(embeddings)
    sil = silhouette_score(embeddings, labels) if len(set(labels)) >= 2 else 0
    print(f"  \u6700\u7ec8 k={best_k}, silhouette={sil:.4f}", flush=True)

    clusters = []
    for c in range(best_k):
        mask = labels == c
        c_items = [meta[i] for i in range(len(meta)) if mask[i]]
        c_dims = [m["dimension"] for m in c_items]
        c_scores = [m["avg_score"] for m in c_items]
        clusters.append({
            "cluster_id": c,
            "size": len(c_items),
            "avg_score": round(float(np.mean(c_scores)), 4),
            "dimension_distribution": {d: c_dims.count(d) for d in sorted(set(c_dims))},
            "members": c_items,
        })

    clusters.sort(key=lambda x: x["avg_score"])
    print("\n  \u76f2\u533a\u7c07\u6392\u540d (\u6309\u5747\u5206\u5347\u5e8f):", flush=True)
    for cl in clusters:
        print(f"    \u7c07{cl['cluster_id']}: n={cl['size']}, avg_score={cl['avg_score']:.4f}, dims={cl['dimension_distribution']}", flush=True)

    return clusters, best_k, sil


def analysis_2c3(reports_dir, core_path, output_base):
    """2c-3: \u5931\u8d25\u6848\u4f8b\u5d4c\u5165\u805a\u7c7b"""
    print("\n" + "=" * 60)
    print("2c-3: \u5931\u8d25\u6848\u4f8b\u5d4c\u5165\u805a\u7c7b (KMeans)")
    print("=" * 60)


    _, qmap = load_core_questions(core_path)
    reports = load_reports(reports_dir)

    low_score_map = defaultdict(lambda: {"scores": [], "models": [], "model_answers": []})
    for model_name, rep in reports.items():
        for r in rep["records"]:
            if r["score"] < 0.3:
                qid = r["id"]
                low_score_map[qid]["scores"].append(r["score"])
                low_score_map[qid]["models"].append(model_name)
                low_score_map[qid]["model_answers"].append(r.get("model_answer", ""))

    ranked = sorted(low_score_map.items(), key=lambda x: len(x[1]["scores"]), reverse=True)
    print(f"\u4f4e\u5206\u9898\u603b\u6570 (score<0.3): {len(ranked)}")
    top_blind = ranked[:50]
    print(f"Top-50 \u591a\u6a21\u578b\u5171\u4f4e\u5206\u9898: {[(qid, len(v['scores'])) for qid, v in top_blind[:10]]}...", flush=True)

    texts = []
    meta = []
    for qid, v in top_blind:
        q = qmap.get(qid)
        question_text = q["question"] if q else qid
        answer_text = q["answer"] if q else ""
        texts.append(f"{question_text}\n{answer_text}")
        meta.append({
            "id": qid,
            "dimension": q["dimension"] if q else "?",
            "n_models_failed": len(v["scores"]),
            "avg_score": round(float(np.mean(v["scores"])), 4),
            "models": v["models"],
        })

    if len(texts) < 3:
        print("  \u26a0 \u4f4e\u5206\u9898\u4e0d\u8db3 3 \u9898\uff0c\u8df3\u8fc7\u805a\u7c7b", flush=True)
        result = {"analysis": "2c-3_failure_clustering", "n_low_score": len(ranked), "clusters": [], "note": "\u6837\u672c\u4e0d\u8db3"}
        json_path = f"{output_base}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"\n  JSON \u5df2\u4fdd\u5b58: {json_path}")
        return result

    print(f"\n  \u5d4c\u5165 {len(texts)} \u6761\u4f4e\u5206\u9898...", flush=True)
    embeddings = get_embeddings(texts, progress_label="2c-3 embed")

    clusters, best_k, sil = _build_kmeans_clusters(embeddings, texts, meta)
    if clusters is None:
        json_path = f"{output_base}.json"
        result = {"analysis": "2c-3_failure_clustering", "n_low_score": len(ranked), "clusters": [], "note": "\u6837\u672c\u4e0d\u8db3"}
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"\n  JSON \u5df2\u4fdd\u5b58: {json_path}")
        return result

    result = {
        "analysis": "2c-3_failure_embedding_clustering",
        "n_low_score_total": len(ranked),
        "n_clustered": len(texts),
        "k_optimal": best_k,
        "silhouette_score": round(float(sil), 4),
        "clusters": clusters,
        "top3_blind_spots": [
            {
                "cluster_id": cl["cluster_id"],
                "avg_score": cl["avg_score"],
                "n_questions": cl["size"],
                "dominant_dimension": max(cl["dimension_distribution"], key=cl["dimension_distribution"].get) if cl["dimension_distribution"] else "?",
                "sample_ids": [m["id"] for m in cl["members"][:5]],
            }
            for cl in clusters[:3]
        ],
        "interpretation": (
            f"\u4f4e\u5206\u9898\u5728\u5d4c\u5165\u7a7a\u95f4\u4e2d\u81ea\u7136\u805a\u4e3a {best_k} \u7c07 (silhouette={sil:.3f})\u3002"
            "\u5747\u5206\u6700\u4f4e\u7684\u7c07\u5373\u4e3a\u8de8\u6a21\u578b\u7684\u300c\u77e5\u8bc6\u76f2\u533a\u300d\uff0c\u53ef\u9488\u5bf9\u6027\u6269\u5145\u8bad\u7ec3\u6570\u636e\u6216\u6539\u8fdb prompt\u3002"
        ),
    }

    json_path = f"{output_base}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n  JSON \u5df2\u4fdd\u5b58: {json_path}")

    return result


def main():
    parser = argparse.ArgumentParser(description="Task 2c: 嵌入模型三大分析")
    parser.add_argument("--dataset", default="benchmark/dataset/core.json", help="core.json 路径")
    parser.add_argument("--reports-dir", default="benchmark/eval/reports", help="模型报告目录")
    parser.add_argument("--cross-judge", default="benchmark/eval/reports/cross_judge_kappa.json", help="交叉验证数据")
    parser.add_argument("--output-base", default="benchmark/eval/results/embedding_cluster_analysis", help="输出文件前缀")
    parser.add_argument("--skip", nargs="*", choices=["2c1", "2c2", "2c3"], default=[], help="跳过的分析")
    args = parser.parse_args()

    # 检查 llama.cpp embedding 连通性
    print("检查 llama.cpp embedding 连通性...", flush=True)
    try:
        result = _llm_client.embed(["test"])
        emb = result.embeddings[0] if result.embeddings else []
        if len(emb) != EMBED_DIM:
            print(f"  ✗ 嵌入维度不匹配: 期望 {EMBED_DIM}, 实际 {len(emb)}", flush=True)
            sys.exit(1)
        print(f"  ✓ llama.cpp embedding 在线, 嵌入维度: {len(emb)}, 模型: {EMBED_MODEL}", flush=True)
    except Exception as e:
        print(f"  ✗ llama.cpp embedding 连接失败: {e}", flush=True)
        print("  请确保 llama.cpp embedding 服务运行中 (port 8081)", flush=True)
        sys.exit(1)

    # 确保输出目录存在
    output_base = args.output_base
    Path(output_base).parent.mkdir(parents=True, exist_ok=True)

    # 转为绝对路径
    dataset_path = Path(args.dataset)
    reports_dir = Path(args.reports_dir)
    cross_judge_path = Path(args.cross_judge)

    if not dataset_path.exists():
        print(f"✗ 数据集不存在: {dataset_path}")
        sys.exit(1)

    results = {}

    if "2c1" not in args.skip:
        results["2c1"] = analysis_2c1(str(dataset_path), output_base)

    if "2c2" not in args.skip:
        results["2c2"] = analysis_2c2(str(reports_dir), str(cross_judge_path), f"{output_base}_correlation")

    if "2c3" not in args.skip:
        results["2c3"] = analysis_2c3(str(reports_dir), str(dataset_path), f"{output_base}_failure")

    print("\n" + "=" * 60)
    print("Task 2c 全部完成!")
    print("=" * 60)
    print("产出文件:")
    for f in sorted(Path(output_base).parent.glob(f"{Path(output_base).name}*")):
        print(f"  {f}")


if __name__ == "__main__":
    main()
