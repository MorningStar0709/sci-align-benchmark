"""A4: Train/Eval 分布一致性"""

import sys

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
except ImportError as e:
    print(f"缺少依赖: {e}")
    print("请安装: .venv/bin/python -m pip install scikit-learn matplotlib")
    sys.exit(1)

import json

import numpy as np

from .core import _configure_plot_style


def _rbf_mmd(X, Y, sigma=None):
    """RBF kernel Maximum Mean Discrepancy with median heuristic for sigma."""
    if sigma is None:
        XX = np.dot(X, X.T)
        distances = (np.diag(XX).reshape(-1, 1) + np.diag(XX).reshape(1, -1) - 2 * XX)
        sigma = np.median(np.sqrt(np.maximum(distances[np.triu_indices(len(X), k=1)], 0)))
        if sigma == 0:
            sigma = 1.0

    gamma = 1.0 / (2 * sigma * sigma)

    def kernel(A, B):
        AA = np.sum(A * A, axis=1).reshape(-1, 1)
        BB = np.sum(B * B, axis=1).reshape(1, -1)
        dist2 = np.maximum(AA + BB - 2 * np.dot(A, B.T), 0)
        return np.exp(-gamma * dist2)

    K_XX = kernel(X, X)
    K_YY = kernel(Y, Y)
    K_XY = kernel(X, Y)

    mmd = np.mean(K_XX) + np.mean(K_YY) - 2 * np.mean(K_XY)
    return mmd, sigma


def analysis_a4_train_eval(embeddings, chunk_metas, output_dir):
    """A4: Train/Eval 分布一致性"""
    print("\n" + "=" * 60)
    print("A4: Train/Eval 分布一致性")
    print("=" * 60)

    _configure_plot_style()

    train_mask = np.array([m["set"] == "train" for m in chunk_metas])
    eval_mask = np.array([m["set"] == "eval" for m in chunk_metas])

    n_train = train_mask.sum()
    n_eval = eval_mask.sum()
    print(f"  Train: {n_train}, Eval: {n_eval}", flush=True)

    # PCA
    pca = PCA(n_components=2, random_state=42, svd_solver="auto")
    coords = pca.fit_transform(embeddings)
    evr = pca.explained_variance_ratio_

    train_emb = embeddings[train_mask]
    eval_emb = embeddings[eval_mask]
    train_centroid = train_emb.mean(axis=0)
    eval_centroid = eval_emb.mean(axis=0)

    centroid_cos = float(np.dot(train_centroid, eval_centroid) /
                         (np.linalg.norm(train_centroid) * np.linalg.norm(eval_centroid)))
    print(f"  Train/Eval 质心 cosine: {centroid_cos:.6f}", flush=True)

    # MMD
    sample_size = min(1000, n_train, n_eval)
    rng = np.random.RandomState(42)
    train_sample = train_emb[rng.choice(n_train, sample_size, replace=False)]
    eval_sample = eval_emb[rng.choice(n_eval, sample_size, replace=False)]

    mmd_val, sigma_val = _rbf_mmd(train_sample, eval_sample)
    print(f"  MMD (RBF, n={sample_size}, σ={sigma_val:.2f}): {mmd_val:.6f}", flush=True)

    # Permutation test for MMD significance
    combined = np.vstack([train_sample, eval_sample])
    n_perm = 200
    perm_mmds = []
    for _ in range(n_perm):
        perm_idx = rng.permutation(2 * sample_size)
        perm_train = combined[perm_idx[:sample_size]]
        perm_eval = combined[perm_idx[sample_size:]]
        perm_mmd, _ = _rbf_mmd(perm_train, perm_eval, sigma_val)
        perm_mmds.append(perm_mmd)

    perm_mmds = np.array(perm_mmds)
    p_value = float(np.mean(perm_mmds >= mmd_val))
    print(f"  Permutation test (n={n_perm}): p={p_value:.4f}")
    if p_value < 0.05:
        print("  ⚠ Train/Eval 分布在 MMD 意义上存在显著差异 (p<0.05)", flush=True)
    else:
        print("  ✓ Train/Eval 分布在 MMD 意义上无显著差异", flush=True)

    # 可视化
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # 左: PCA split 着色
    ax = axes[0]
    split_colors = {"train": "#3498DB", "eval": "#E74C3C"}
    for split_name, mask in [("train", train_mask), ("eval", eval_mask)]:
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   c=split_colors[split_name],
                   label=f"{split_name} (n={mask.sum()})",
                   alpha=0.12, s=3, edgecolors="none")
    ax.set_xlabel(f"PC1 ({evr[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({evr[1]*100:.1f}%)")
    ax.set_title("Train/Eval Distribution Comparison (PCA 2D)")
    ax.legend(markerscale=8, fontsize=10)
    ax.grid(True, alpha=0.3)

    # 右: MMD permutation 分布
    ax = axes[1]
    ax.hist(perm_mmds, bins=30, alpha=0.7, color="#95A5A6", edgecolor="white",
            label=f"Permutation null (n={n_perm})")
    ax.axvline(mmd_val, color="#E74C3C", linewidth=2.5, linestyle="--",
               label=f"Observed MMD={mmd_val:.4f}")
    ax.set_xlabel("MMD")
    ax.set_ylabel("Frequency")
    ax.set_title(f"MMD Permutation Test (p={p_value:.3f})")
    ax.legend(fontsize=9)

    plt.tight_layout()
    png_path = output_dir / "a4_train_eval.png"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  图已保存: {png_path}", flush=True)

    result = {
        "analysis": "A4_train_eval_consistency",
        "n_train": int(n_train),
        "n_eval": int(n_eval),
        "split_ratio_actual": round(float(n_train) / (n_train + n_eval), 4),
        "centroid_cosine": round(centroid_cos, 4),
        "mmd_rbf": round(float(mmd_val), 6),
        "mmd_sigma": round(float(sigma_val), 2),
        "mmd_p_value": round(p_value, 4),
        "mmd_permutation_n": n_perm,
        "mmd_significant": p_value < 0.05,
        "interpretation": (
            f"质心 cosine = {centroid_cos:.4f} (≈1.0 说明分布一致)。"
            f"MMD = {mmd_val:.4f}, permutation p = {p_value:.4f}"
            f"{' → 差异显著' if p_value < 0.05 else ' → 无显著差异'}。"
        ),
    }

    json_path = output_dir / "a4_train_eval.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"  JSON 已保存: {json_path}", flush=True)
    return result
