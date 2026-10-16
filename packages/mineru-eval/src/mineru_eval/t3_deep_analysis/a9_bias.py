"""A9: Length Bias — model_answer length vs score correlation."""

import numpy as np
from scipy.stats import pearsonr, spearmanr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mineru_eval.t3_deep_analysis.core import (
    MODEL_SHORT, MODEL_FAMILY, FAMILY_COLORS,
    DIMENSIONS, OUTPUT_DIR,
    _save_fig, _sig_stars,
)


def _compute_bias_metrics(all_scores, all_lengths, models_sorted, data):
    """计算各模型 + 全局 + 分维度的长度-评分相关性。"""
    # Per-model correlation
    len_corrs = []
    for m in models_sorted:
        lens = all_lengths[m]
        scores = all_scores[m]
        if np.std(lens) > 0 and np.std(scores) > 0:
            r_pearson, p_pearson = pearsonr(lens, scores)
            r_spearman, p_spearman = spearmanr(lens, scores)
        else:
            r_pearson, p_pearson = 0.0, 1.0
            r_spearman, p_spearman = 0.0, 1.0
        len_corrs.append({
            "model": m, "name": MODEL_SHORT[m],
            "mean_length": int(lens.mean()),
            "median_length": int(np.median(lens)),
            "std_length": int(lens.std(ddof=1)),
            "min_length": int(lens.min()), "max_length": int(lens.max()),
            "pearson_r": round(float(r_pearson), 4),
            "pearson_p": round(float(p_pearson), 4),
            "spearman_rho": round(float(r_spearman), 4),
            "spearman_p": round(float(p_spearman), 4),
        })

    print(f"\n{'Model':<16} {'AvgLen':>7} {'Pearson r':>10} {'p':>8} {'Spearman ρ':>10} {'p':>8}")
    print("-" * 63)
    for lc in len_corrs:
        print(f"{lc['name']:<16} {lc['mean_length']:>7} {lc['pearson_r']:>10.4f} "
              f"{_sig_stars(lc['pearson_p']):>8} {lc['spearman_rho']:>10.4f} {_sig_stars(lc['spearman_p']):>8}")

    # Global all-model pooled
    all_lens_flat = np.concatenate([all_lengths[m] for m in models_sorted])
    all_scores_flat = np.concatenate([all_scores[m] for m in models_sorted])
    global_r, global_p = pearsonr(all_lens_flat, all_scores_flat)
    global_rho, global_rho_p = spearmanr(all_lens_flat, all_scores_flat)
    print("\n  Global (all models pooled):")
    print(f"    Pearson r={global_r:.4f} {_sig_stars(global_p)}")
    print(f"    Spearman ρ={global_rho:.4f} {_sig_stars(global_rho_p)}")

    # By dimension
    dimension_corrs = {}
    print("\n  Length-Score correlation by dimension:")
    for d in DIMENSIONS:
        dim_lens = np.concatenate([
            all_lengths[m][[rec.get("dimension", "") == d for rec in data[m].get("records", [])]]
            for m in models_sorted if all_lengths[m].size > 0
        ])
        dim_scores = np.concatenate([
            all_scores[m][[rec.get("dimension", "") == d for rec in data[m].get("records", [])]]
            for m in models_sorted if all_scores[m].size > 0
        ])
        if len(dim_lens) > 1 and np.std(dim_lens) > 0 and np.std(dim_scores) > 0:
            dr, dp = pearsonr(dim_lens, dim_scores)
        else:
            dr, dp = 0.0, 1.0
        dimension_corrs[d] = {"pearson_r": round(float(dr), 4), "p": round(float(dp), 4)}
        print(f"    Dim {d}: r={dr:.4f} {_sig_stars(dp)} (n={len(dim_lens)})")

    return len_corrs, global_r, global_p, global_rho, global_rho_p, dimension_corrs, all_lens_flat, all_scores_flat



def _plot_length_vs_score(models_sorted, all_scores, all_lengths, len_corrs, colors, names):
    """Chart 1: Length vs Score Scatter (per model)"""
    print("  Chart 1: Length vs Score Scatter...")
    n_sub = len(models_sorted)
    n_cols = 3
    n_rows = (n_sub + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
    axes_flat = axes.flatten() if n_sub > 1 else [axes]

    for idx, m in enumerate(models_sorted):
        ax = axes_flat[idx]
        lens = all_lengths[m]
        scores = all_scores[m]
        ax.scatter(lens, scores, c=colors[idx], alpha=0.3, s=8, edgecolors="none")
        if len(lens) > 1:
            z = np.polyfit(lens, scores, 1)
            p_fit = np.poly1d(z)
            x_line = np.linspace(lens.min(), lens.max(), 100)
            ax.plot(x_line, p_fit(x_line), color=colors[idx], linewidth=1.5, linestyle="--")

        lc = len_corrs[idx]
        ax.text(0.95, 0.05, f"r={lc['pearson_r']:.3f}\nρ={lc['spearman_rho']:.3f}",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=8, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        ax.set_xlabel("Answer Length (chars)", fontsize=8)
        ax.set_ylabel("Score", fontsize=8)
        ax.set_title(names[idx], fontsize=9, fontweight="bold")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(alpha=0.3)

    for idx in range(len(models_sorted), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle("A9: Answer Length vs Score (per Model)", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save_fig(fig, "a9_length_vs_score.png")



def _plot_length_distribution(all_lens_flat, all_scores_flat, models_sorted, all_lengths, colors, names, global_r):
    """Chart 2: Global Length Distribution"""
    print("  Chart 2: Global Length Distribution...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    ax = axes[0]
    for i, m in enumerate(models_sorted):
        ax.hist(all_lengths[m], bins=30, alpha=0.4, color=colors[i], label=names[i])
    ax.set_xlabel("Answer Length (chars)", fontsize=11)
    ax.set_ylabel("Frequency", fontsize=11)
    ax.set_title("A9: Answer Length Distribution (by Model)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1]
    length_bins = [0, 500, 1000, 1500, 2000, 3000, 5000, 10000]
    length_labels = ["<500", "500-1k", "1k-1.5k", "1.5k-2k", "2k-3k", "3k-5k", ">5k"]
    bin_means = []
    bin_counts = []
    for i in range(len(length_bins) - 1):
        mask = (all_lens_flat >= length_bins[i]) & (all_lens_flat < length_bins[i + 1])
        bin_means.append(all_scores_flat[mask].mean() if mask.sum() > 0 else 0)
        bin_counts.append(mask.sum())
    bin_means[-1] = all_scores_flat[all_lens_flat >= length_bins[-2]].mean()

    bars = ax.bar(length_labels, bin_means, alpha=0.7, color="#3498DB", edgecolor="white")
    for i, (b, c) in enumerate(zip(bin_means, bin_counts)):
        ax.text(i, b + 0.01, f"n={c}", ha="center", fontsize=8)

    ax.set_xlabel("Answer Length (chars)", fontsize=11)
    ax.set_ylabel("Mean Score", fontsize=11)
    ax.set_title(f"A9: Mean Score by Length Bin (Global r={global_r:.3f})", fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _save_fig(fig, "a9_length_distribution.png")



def _plot_length_ratio(models_sorted, data, all_lengths, all_scores, colors, names):
    """Chart 3: Expected vs Actual Length Ratio"""
    print("  Chart 3: Expected vs Actual Length Ratio...")
    exp_lens_by_model = {}
    for m in models_sorted:
        exp_lens = []
        for rec in data[m].get("records", []):
            exp_lens.append(len(rec.get("expected_answer", "")))
        exp_lens_by_model[m] = np.array(exp_lens)

    fig, ax = plt.subplots(figsize=(11, 7))
    for i, m in enumerate(models_sorted):
        ratio = all_lengths[m] / np.maximum(exp_lens_by_model[m], 1)
        ratio = np.clip(ratio, 0, 10)
        scores = all_scores[m]
        ax.scatter(ratio, scores, c=colors[i], alpha=0.2, s=5, label=names[i])

    ax.axvline(1.0, color="gray", linestyle="--", alpha=0.5, label="Model=Expected length")
    ax.set_xlabel("Model Answer Length / Expected Answer Length", fontsize=11)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_xlim(0, 8)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("A9: Length Ratio (Model/Expected) vs Score", fontsize=13, fontweight="bold")
    ax.legend(fontsize=7, ncol=2, loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _save_fig(fig, "a9_length_ratio.png")



def _generate_bias_report(len_corrs, global_r, global_p, global_rho, global_rho_p, dimension_corrs, all_scores, models_sorted):
    """构建并保存 A9 报告 JSON。"""
    result = {
        "analysis": "A9_length_bias",
        "n_models": len(models_sorted),
        "n_per_model": len(all_scores[models_sorted[0]]),
        "global_pearson_r": round(float(global_r), 4),
        "global_pearson_p": round(float(global_p), 4),
        "global_spearman_rho": round(float(global_rho), 4),
        "global_spearman_p": round(float(global_rho_p), 4),
        "per_model": len_corrs,
        "dimension_corrs": dimension_corrs,
        "interpretation": (
            "若 Pearson r 显著为正 → 存在长度偏见（Judge 倾向于给长答案高分）。"
            "若 r ≈ 0 → 无系统性长度偏见。"
        ),
    }
    import json
    json_path = OUTPUT_DIR / "a9_length_bias.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"  JSON: {json_path}")
    return result



def analysis_a9_length_bias(all_scores, all_lengths, by_dim, data):
    """A9: 长度偏见分析 — model_answer 长度与评分的相关性"""
    print("\n" + "=" * 60)
    print("A9: 长度偏见 (Length Bias)")
    print("=" * 60)

    models_sorted = sorted(data.keys(), key=lambda m: data[m].get("overall_score", 0), reverse=True)
    names = [MODEL_SHORT[m] for m in models_sorted]
    families = [MODEL_FAMILY[m] for m in models_sorted]
    colors = [FAMILY_COLORS[f] for f in families]

    len_corrs, global_r, global_p, global_rho, global_rho_p, dimension_corrs, all_lens_flat, all_scores_flat = \
        _compute_bias_metrics(all_scores, all_lengths, models_sorted, data)

    _plot_length_vs_score(models_sorted, all_scores, all_lengths, len_corrs, colors, names)
    _plot_length_distribution(all_lens_flat, all_scores_flat, models_sorted, all_lengths, colors, names, global_r)
    _plot_length_ratio(models_sorted, data, all_lengths, all_scores, colors, names)

    result_a9 = _generate_bias_report(len_corrs, global_r, global_p, global_rho, global_rho_p, dimension_corrs, all_scores, models_sorted)
    return result_a9
