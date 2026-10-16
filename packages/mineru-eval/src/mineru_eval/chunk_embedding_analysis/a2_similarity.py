"""A2: 文档间相似度矩阵"""

import sys

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import normalize
    from scipy.cluster.hierarchy import dendrogram, linkage
    from scipy.spatial.distance import squareform
except ImportError as e:
    print(f"缺少依赖: {e}")
    print("请安装: .venv/bin/python -m pip install scikit-learn matplotlib scipy")
    sys.exit(1)

import numpy as np
import json
from collections import defaultdict

from .core import _configure_plot_style


def analysis_a2_doc_similarity(embeddings, chunk_metas, doc_map, output_dir):
    """A2: 文档间相似度矩阵"""
    print("\n" + "=" * 60)
    print("A2: 文档间相似度矩阵 (362 docs)")
    print("=" * 60)

    _configure_plot_style()

    # 按文档聚合 centroid
    doc_embeddings = defaultdict(list)
    for i, m in enumerate(chunk_metas):
        doc_embeddings[m["filename"]].append(embeddings[i])

    doc_filenames = sorted(doc_embeddings.keys())
    doc_centroids = np.array([np.mean(doc_embeddings[fn], axis=0) for fn in doc_filenames])
    n_docs = len(doc_filenames)
    print(f"  {n_docs} 个文档, centroid 矩阵: {doc_centroids.shape}", flush=True)

    doc_centroids_norm = normalize(doc_centroids, norm="l2")

    cos_sim = doc_centroids_norm @ doc_centroids_norm.T
    cos_sim = np.clip(cos_sim, -1.0, 1.0)
    cos_dist = 1.0 - cos_sim
    np.fill_diagonal(cos_dist, 0.0)

    # 层次聚类
    Z = linkage(squareform(cos_dist), method="ward")

    # 短标签: 取文件名前30字符 + type 标记
    short_labels = []
    for fn in doc_filenames:
        doc = doc_map.get(fn, {})
        dtype = "T" if doc.get("type") == "textbook" else "A"
        label = fn[:28] + "…" if len(fn) > 28 else fn
        short_labels.append(f"[{dtype}] {label}")

    fig, axes = plt.subplots(1, 2, figsize=(28, 14))

    # 左: 热力图 (下采样显示概览，全量存单独文件)
    ax = axes[0]
    im = ax.imshow(cos_sim, cmap="RdYlBu_r", aspect="auto", vmin=0.5, vmax=1.0)
    plt.colorbar(im, ax=ax, label="cosine similarity", shrink=0.8)
    ax.set_title(f"Inter-Document Cosine Similarity ({n_docs}×{n_docs})")
    ax.set_xlabel("Document Index")
    ax.set_ylabel("Document Index")

    # 右: 树状图
    ax = axes[1]
    dendrogram(Z, labels=short_labels, leaf_font_size=3, ax=ax,
               orientation="right", color_threshold=0.3 * max(Z[:, 2]))
    ax.set_title("Document Hierarchical Clustering (Ward linkage on cosine distance)")
    ax.set_xlabel("Merge Distance")

    plt.tight_layout()
    png_path = output_dir / "a2_doc_similarity.png"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  相似度图已保存: {png_path}", flush=True)

    # 全尺寸热力图单独保存
    fig2, ax2 = plt.subplots(1, 1, figsize=(24, 22))
    im2 = ax2.imshow(cos_sim, cmap="RdYlBu_r", aspect="auto", vmin=0.5, vmax=1.0)
    plt.colorbar(im2, ax=ax2, label="cosine similarity")
    ax2.set_title(f"Inter-Document Cosine Similarity — Full ({n_docs}×{n_docs})")
    ax2.set_xticks(range(n_docs))
    ax2.set_yticks(range(n_docs))
    ax2.set_xticklabels(short_labels, rotation=90, fontsize=2.5)
    ax2.set_yticklabels(short_labels, fontsize=2.5)
    full_heatmap_path = output_dir / "a2_doc_similarity_full.png"
    fig2.savefig(full_heatmap_path, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"  全尺寸热力图已保存: {full_heatmap_path}", flush=True)

    # Top-20 最相似文档对 (排除自相似)
    triu_indices = np.triu_indices(n_docs, k=1)
    triu_vals = cos_sim[triu_indices]
    top_n = min(20, len(triu_vals))
    top_indices = np.argpartition(triu_vals, -top_n)[-top_n:]
    top_indices = top_indices[np.argsort(triu_vals[top_indices])[::-1]]

    top_pairs = []
    for idx in top_indices:
        i, j = triu_indices[0][idx], triu_indices[1][idx]
        doc_i = doc_map.get(doc_filenames[i], {})
        doc_j = doc_map.get(doc_filenames[j], {})
        top_pairs.append({
            "rank": len(top_pairs) + 1,
            "doc_a": doc_filenames[i],
            "doc_a_id": doc_i.get("id", ""),
            "doc_a_type": doc_i.get("type", ""),
            "doc_a_title": doc_i.get("title", ""),
            "doc_b": doc_filenames[j],
            "doc_b_id": doc_j.get("id", ""),
            "doc_b_type": doc_j.get("type", ""),
            "doc_b_title": doc_j.get("title", ""),
            "cosine_similarity": round(float(cos_sim[i, j]), 4),
        })

    print("\n  Top-20 最相似文档对:", flush=True)
    for pair in top_pairs[:10]:
        print(f"    #{pair['rank']}: cos={pair['cosine_similarity']:.4f}  "
              f"[{pair['doc_a_type']}] {pair['doc_a'][:40]} ↔ [{pair['doc_b_type']}] {pair['doc_b'][:40]}", flush=True)

    # 统计: textbook-textbook, arxiv-arxiv, cross-type 相似度分布
    tt_sims, aa_sims, ta_sims = [], [], []
    for i in range(n_docs):
        for j in range(i + 1, n_docs):
            ti = doc_map.get(doc_filenames[i], {}).get("type", "")
            tj = doc_map.get(doc_filenames[j], {}).get("type", "")
            val = float(cos_sim[i, j])
            if ti == "textbook" and tj == "textbook":
                tt_sims.append(val)
            elif ti == "arxiv" and tj == "arxiv":
                aa_sims.append(val)
            else:
                ta_sims.append(val)

    sim_stats = {
        "textbook_textbook": {"mean": round(np.mean(tt_sims), 4), "std": round(np.std(tt_sims), 4), "n": len(tt_sims)} if tt_sims else {},
        "arxiv_arxiv": {"mean": round(np.mean(aa_sims), 4), "std": round(np.std(aa_sims), 4), "n": len(aa_sims)} if aa_sims else {},
        "cross_type": {"mean": round(np.mean(ta_sims), 4), "std": round(np.std(ta_sims), 4), "n": len(ta_sims)} if ta_sims else {},
    }
    print("\n  同类/跨类相似度统计:", flush=True)
    for k, v in sim_stats.items():
        if v:
            print(f"    {k}: mean={v['mean']:.4f} ± {v['std']:.4f} (n={v['n']})", flush=True)

    result = {
        "analysis": "A2_doc_similarity",
        "n_docs": n_docs,
        "similarity_stats": sim_stats,
        "top20_pairs": top_pairs,
        "interpretation": (
            "教材-教材相似度若显著高于 arXiv-ArXiv，说明教材内容更同质化；"
            "跨类型相似度若与同类相当，说明语料主题覆盖连续。"
        ),
    }

    json_path = output_dir / "a2_doc_similarity.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"  JSON 已保存: {json_path}", flush=True)
    return result
