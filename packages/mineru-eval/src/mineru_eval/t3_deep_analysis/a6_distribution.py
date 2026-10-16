"""A6: Score Distribution — distribution shape / skewness / kurtosis / bimodality detection."""

import numpy as np
from scipy import stats
from scipy.stats import kurtosis, skew

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mineru_eval.t3_deep_analysis.core import (
    MODEL_SHORT, MODEL_FAMILY, FAMILY_COLORS, OUTPUT_DIR,
    _save_fig,
)


def analysis_a6_distribution(all_scores):
    """A6: 评分分布深度分析 — 直方图/KDE/偏度/峰度/双峰检测"""
    print("\n" + "=" * 60)
    print("A6: 评分分布 (Score Distribution)")
    print("=" * 60)

    models_sorted = sorted(all_scores.keys(), key=lambda m: all_scores[m].mean(), reverse=True)
    names = [MODEL_SHORT[m] for m in models_sorted]
    families = [MODEL_FAMILY[m] for m in models_sorted]
    colors = [FAMILY_COLORS[f] for f in families]

    dist_stats = []
    for m in models_sorted:
        s = all_scores[m]
        sk = float(skew(s))
        ku = float(kurtosis(s, fisher=True))
        qs = np.percentile(s, [25, 50, 75])
        iqr = qs[2] - qs[0]

        try:
            from diptest import diptest
            dip_stat, dip_p = diptest(s)
        except ImportError:
            dip_stat, dip_p = None, None

        dist_stats.append({
            "model": m, "name": MODEL_SHORT[m],
            "mean": float(s.mean()), "std": float(s.std(ddof=1)),
            "skewness": round(sk, 4),
            "kurtosis": round(ku, 4),
            "q25": round(float(qs[0]), 4),
            "median": round(float(qs[1]), 4),
            "q75": round(float(qs[2]), 4),
            "iqr": round(float(iqr), 4),
            "dip_stat": round(float(dip_stat), 4) if dip_stat is not None else None,
            "dip_p": round(float(dip_p), 4) if dip_p is not None else None,
            "min": float(s.min()), "max": float(s.max()),
            "n_zeros": int((s == 0).sum()),
            "pct_zeros": round(float((s == 0).mean() * 100), 2),
        })

    print(f"\n{'Model':<16} {'Mean':>6} {'Skew':>7} {'Kurt':>7} {'Q25':>6} {'Median':>6} {'Q75':>6} {'Zeros%':>7}")
    print("-" * 72)
    for st in dist_stats:
        print(f"{st['name']:<16} {st['mean']:>6.4f} {st['skewness']:>7.2f} {st['kurtosis']:>7.2f} "
              f"{st['q25']:>6.4f} {st['median']:>6.4f} {st['q75']:>6.4f} {st['pct_zeros']:>6.1f}%")

    # Chart 1: Density Ridges
    print("\n  Chart 1: Density Ridges...")
    fig, ax = plt.subplots(figsize=(12, 8))
    y_offsets = list(range(len(models_sorted)))
    for i, (m, y) in enumerate(zip(models_sorted, y_offsets)):
        s = all_scores[m]
        kde = stats.gaussian_kde(s)
        x_grid = np.linspace(-0.05, 1.05, 300)
        density = kde(x_grid)
        density = density / density.max() * 0.7
        ax.fill_between(x_grid, y + density, y, alpha=0.6, color=colors[i])
        ax.plot(x_grid, y + density, color=colors[i], linewidth=1.2)
        ax.axhline(y, color="gray", linewidth=0.3, alpha=0.4)

    ax.set_yticks(y_offsets)
    ax.set_yticklabels(names, fontsize=10)
    ax.set_xlim(-0.05, 1.05)
    ax.set_xlabel("Score", fontsize=11)
    ax.set_title("A6: Score Distribution Density Ridges (by Model)", fontsize=13, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    _save_fig(fig, "a6_density_ridges.png")

    # Chart 2: Skewness vs Mean
    print("  Chart 2: Skewness vs Mean...")
    fig, ax = plt.subplots(figsize=(10, 7))
    means_arr = [st["mean"] for st in dist_stats]
    skews_arr = [st["skewness"] for st in dist_stats]
    for i, m in enumerate(models_sorted):
        ax.scatter(means_arr[i], skews_arr[i], s=120, c=colors[i],
                   edgecolors="black", linewidths=0.5, zorder=5, label=names[i])
        ax.annotate(names[i], (means_arr[i], skews_arr[i]),
                    textcoords="offset points", xytext=(5, 5), fontsize=8)

    ax.axhline(0, color="gray", linestyle="--", alpha=0.4)
    ax.set_xlabel("Mean Score", fontsize=11)
    ax.set_ylabel("Skewness", fontsize=11)
    ax.set_title("A6: Score Skewness vs Mean (by Model)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _save_fig(fig, "a6_skewness_vs_mean.png")

    # Chart 3: ECDF
    print("  Chart 3: ECDF (Cumulative Distribution)...")
    fig, ax = plt.subplots(figsize=(12, 7))
    for i, m in enumerate(models_sorted):
        s = sorted(all_scores[m])
        n = len(s)
        y = np.arange(1, n + 1) / n
        ax.plot(s, y, color=colors[i], linewidth=1.5, label=names[i], alpha=0.85)

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Score", fontsize=11)
    ax.set_ylabel("Cumulative Proportion", fontsize=11)
    ax.set_title("A6: ECDF — Empirical Cumulative Distribution (by Model)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=8, loc="lower right", ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _save_fig(fig, "a6_ecdf.png")

    result_a6 = {
        "analysis": "A6_score_distribution",
        "n_models": len(models_sorted),
        "n_per_model": len(all_scores[models_sorted[0]]),
        "model_stats": dist_stats,
        "interpretation": (
            "负偏态 (skewness < 0) 表示高分集中；正偏态表示低分集中。"
            "峰度 > 0 表示极端值多。双峰 (dip test p<0.05) 暗示能力分化。"
        ),
    }
    import json
    json_path = OUTPUT_DIR / "a6_score_distribution.json"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result_a6, f, ensure_ascii=False, indent=2)
    print(f"  JSON: {json_path}")

    return result_a6
