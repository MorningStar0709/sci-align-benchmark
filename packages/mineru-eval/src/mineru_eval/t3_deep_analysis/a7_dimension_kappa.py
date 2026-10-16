"""A7: Dimension Agreement (κ) — Kendall's W / ICC / inter-dimension consistency matrix."""

import numpy as np
from scipy import stats
from scipy.stats import spearmanr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from mineru_eval.t3_deep_analysis.core import (
    MODEL_SHORT, MODEL_FAMILY, FAMILY_COLORS,
    DIMENSIONS, OUTPUT_DIR,
    _save_fig, _sig_stars,
)


def _compute_kendall_w(dim_rank_matrix: np.ndarray, n_models: int, n_dims: int) -> tuple:
    """Compute Kendall's W and Friedman test."""
    Ri = dim_rank_matrix.sum(axis=0)
    R_mean = Ri.mean()
    S = ((Ri - R_mean) ** 2).sum()
    W = 12 * S / (n_models ** 2 * (n_dims ** 3 - n_dims))
    W = min(W, 1.0)
    chi2 = n_models * (n_dims - 1) * W
    p = 1 - stats.chi2.cdf(chi2, df=n_dims - 1)
    return W, chi2, p


def _compute_pairwise_spearman(by_dim: dict, models_sorted: list) -> dict:
    """Compute dimension pairwise Spearman correlations."""
    dim_pairwise = {}
    for i, d1 in enumerate(DIMENSIONS):
        for j, d2 in enumerate(DIMENSIONS):
            if j <= i:
                continue
            s1 = [np.mean(by_dim[m][d1]) if by_dim[m][d1] else 0 for m in models_sorted]
            s2 = [np.mean(by_dim[m][d2]) if by_dim[m][d2] else 0 for m in models_sorted]
            if len(set(s1)) > 1 and len(set(s2)) > 1:
                r, p = spearmanr(s1, s2)
            else:
                r, p = 0.0, 1.0
            dim_pairwise[f"{d1}↔{d2}"] = {"spearman_rho": round(float(r), 4), "p": round(float(p), 4)}
    return dim_pairwise


def _compute_icc(icc_matrix: np.ndarray, n_models: int, n_dims: int) -> float:
    """Compute ICC(2,1)."""
    ms_between = n_dims * np.var(icc_matrix.mean(axis=1), ddof=1)
    ms_within = np.mean([np.var(icc_matrix[i, :], ddof=1) for i in range(n_models)])
    denom = ms_between + (n_dims - 1) * ms_within
    return (ms_between - ms_within) / denom if denom > 0 else 0


def _plot_dimension_heatmap(models_sorted, by_dim, names, n_models, n_dims, W_kendall):
    """Chart 1: Dimension Ranking Heatmap."""
    print("  Chart 1: Dimension Ranking Heatmap...")
    fig, ax = plt.subplots(figsize=(10, 8))
    rank_pct = np.zeros((n_models, n_dims))
    for j, d in enumerate(DIMENSIONS):
        scores_dim = [np.mean(by_dim[m][d]) if by_dim[m][d] else 0 for m in models_sorted]
        max_s = max(scores_dim) if scores_dim else 1
        for i in range(n_models):
            rank_pct[i, j] = scores_dim[i] / max_s if max_s > 0 else 0

    annot = np.empty_like(rank_pct, dtype=object)
    for i in range(n_models):
        for j in range(n_dims):
            val = np.mean(by_dim[models_sorted[i]][DIMENSIONS[j]]) if by_dim[models_sorted[i]][DIMENSIONS[j]] else 0
            annot[i, j] = f"{val:.4f}"

    sns.heatmap(rank_pct, annot=annot, fmt="", cmap="RdYlGn", vmin=0.5, vmax=1.0,
                xticklabels=DIMENSIONS, yticklabels=names,
                linewidths=0.5, linecolor="white",
                cbar_kws={"label": "Score / Max per Dim"},
                annot_kws={"fontsize": 9}, ax=ax)
    ax.set_title(f"A7: Dimension Score Matrix (Kendall's W={W_kendall:.3f})", fontsize=12, fontweight="bold")
    ax.set_xlabel("Dimension", fontsize=11)
    ax.set_ylabel("Model", fontsize=11)
    fig.tight_layout()
    _save_fig(fig, "a7_dimension_agreement.png")


def _plot_concordance_radar(models_sorted, by_dim, names, n_dims, W_kendall):
    """Chart 2: Dimension Concordance Radar."""
    print("  Chart 2: Dimension Concordance Radar...")
    angles = np.linspace(0, 2 * np.pi, n_dims, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw={"projection": "polar"})
    for i, m in enumerate(models_sorted):
        vals = [np.mean(by_dim[m][d]) if by_dim[m][d] else 0 for d in DIMENSIONS]
        vals += vals[:1]
        clr = FAMILY_COLORS.get(MODEL_FAMILY[m], "#888")
        ax.plot(angles, vals, color=clr, linewidth=1.2, alpha=0.6, label=names[i])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(DIMENSIONS, fontsize=11)
    ax.set_ylim(0, 0.85)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8])
    ax.set_title(f"A7: Dimension Concordance (W={W_kendall:.3f})", fontsize=13, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0), fontsize=8)
    ax.grid(alpha=0.4)
    fig.tight_layout()
    _save_fig(fig, "a7_concordance_radar.png")


def analysis_a7_dimension_kappa(all_scores, by_dim, data):
    """A7: \u7ef4\u5ea6\u95f4\u4e00\u81f4\u6027\u5206\u6790 — Kendall's W / ICC / \u7ef4\u95f4 Spearman"""
    print("\n" + "=" * 60)
    print("A7: \u7ef4\u5ea6 \u03ba \u4e00\u81f4\u6027 (Dimension Agreement)")
    print("=" * 60)

    models_sorted = sorted(data.keys(), key=lambda m: data[m].get("overall_score", 0), reverse=True)
    names = [MODEL_SHORT[m] for m in models_sorted]

    n_models = len(models_sorted)
    n_dims = len(DIMENSIONS)

    dim_rank_matrix = np.zeros((n_models, n_dims))
    for i, m in enumerate(models_sorted):
        scores_dim = [np.mean(by_dim[m][d]) if by_dim[m][d] else 0 for d in DIMENSIONS]
        ranks = stats.rankdata([-s for s in scores_dim], method="average")
        dim_rank_matrix[i, :] = ranks

    W_kendall, friedman_chi2, friedman_p = _compute_kendall_w(dim_rank_matrix, n_models, n_dims)
    print(f"  Kendall's W = {W_kendall:.4f} (Friedman \u03c7\u00b2={friedman_chi2:.2f}, p={friedman_p:.4f})")
    if W_kendall > 0.7:
        print("    \u2192 High concordance: models rank consistently across dimensions")
    elif W_kendall > 0.4:
        print("    \u2192 Moderate concordance")
    else:
        print("    \u2192 Low concordance: dimension rankings differ substantially")

    dim_pairwise = _compute_pairwise_spearman(by_dim, models_sorted)
    print("  Dimension pairwise Spearman:")
    for k, v in dim_pairwise.items():
        print(f"    {k}: \u03c1={v['spearman_rho']:.4f} {_sig_stars(v['p'])}")

    icc_matrix = np.zeros((n_models, n_dims))
    for i, m in enumerate(models_sorted):
        for j, d in enumerate(DIMENSIONS):
            icc_matrix[i, j] = np.mean(by_dim[m][d]) if by_dim[m][d] else 0

    icc_2_1 = _compute_icc(icc_matrix, n_models, n_dims)
    print(f"  ICC(2,1): {icc_2_1:.4f}")
    if icc_2_1 > 0.75:
        print("    \u2192 Excellent reliability")
    elif icc_2_1 > 0.5:
        print("    \u2192 Good reliability")
    else:
        print("    \u2192 Poor reliability")

    _plot_dimension_heatmap(models_sorted, by_dim, names, n_models, n_dims, W_kendall)
    _plot_concordance_radar(models_sorted, by_dim, names, n_dims, W_kendall)

    result_a7 = {
        "analysis": "A7_dimension_agreement",
        "n_models": n_models,
        "n_dimensions": n_dims,
        "kendall_W": round(float(W_kendall), 4),
        "friedman_chi2": round(float(friedman_chi2), 2),
        "friedman_p": round(float(friedman_p), 4),
        "icc_2_1": round(float(icc_2_1), 4),
        "dimension_pairwise_spearman": dim_pairwise,
        "interpretation": (
            "Kendall's W \u63a5\u8fd1 1 \u8bf4\u660e\u6a21\u578b\u5728\u5404\u7ef4\u5ea6\u7684\u6392\u540d\u9ad8\u5ea6\u4e00\u81f4\uff1b"
            "ICC(2,1) > 0.75 \u8868\u793a\u7ef4\u5ea6\u8bc4\u5206\u53ef\u9760\u6027\u4f18\u79c0\u3002"
        ),
    }
    import json
    json_path = OUTPUT_DIR / "a7_dimension_agreement.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result_a7, f, ensure_ascii=False, indent=2)
    print(f"  JSON: {json_path}")

    return result_a7
