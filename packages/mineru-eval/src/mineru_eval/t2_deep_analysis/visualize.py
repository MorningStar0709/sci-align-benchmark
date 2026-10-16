"""
T2: 9-Model Deep Analysis — visualization (7 charts).
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from .core import (
    MODEL_SHORT, MODEL_FAMILY, FAMILY_COLORS,
    DIMENSIONS, DIFFICULTIES, _save_fig,
)


def fig_overall_bar(all_scores: dict, dunn_df):
    """Fig 1: Overall scores with 95% CI + significance annotations."""
    models = sorted(all_scores.keys(), key=lambda m: all_scores[m].mean(), reverse=True)
    means = [all_scores[m].mean() for m in models]
    ses = [all_scores[m].std(ddof=1) / np.sqrt(len(all_scores[m])) for m in models]
    cis = [1.96 * se for se in ses]
    labels = [MODEL_SHORT.get(m, m) for m in models]
    families = [MODEL_FAMILY.get(m, "Other") for m in models]
    colors = [FAMILY_COLORS.get(f, "#888") for f in families]

    n = len(models)
    fig, ax = plt.subplots(figsize=(12, 6))
    y_pos = range(n)

    bars = ax.barh(y_pos, means, xerr=cis, color=colors, alpha=0.85,
                   capsize=3, edgecolor="white", linewidth=0.8)

    for i, (m, ci) in enumerate(zip(means, cis)):
        ax.text(m + ci + 0.003, i, f"{m:.4f}", va="center", fontsize=9, fontweight="bold")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlim(0, max(means) * 1.18)
    ax.set_xlabel("Overall Score", fontsize=11)
    ax.set_title("ControlSci: 9-Model Overall Scores (95% CI)", fontsize=13, fontweight="bold")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)

    legend_patches = [plt.Rectangle((0, 0), 1, 1, color=FAMILY_COLORS[f], alpha=0.85, label=f)
                      for f in ["MiMo", "DeepSeek", "Qwen", "MiniMax"]]
    ax.legend(handles=legend_patches, loc="lower right", fontsize=9, title="Family")

    fig.tight_layout()
    return _save_fig(fig, "fig1_overall_bar.png")


def fig_dimension_heatmap(dimension_means: dict, data: dict, per_dim_stats: dict):
    """Fig 2: Model × Dimension heatmap."""
    models = sorted(data.keys(), key=lambda m: data[m].get("overall_score", 0), reverse=True)
    mat = np.zeros((len(models), len(DIMENSIONS)))
    annot = []
    for i, m in enumerate(models):
        row = []
        for j, d in enumerate(DIMENSIONS):
            val = dimension_means[m].get(d, 0)
            mat[i, j] = val
            row.append(f"{val:.4f}")
        annot.append(row)

    labels = [MODEL_SHORT.get(m, m) for m in models]

    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(mat, annot=annot, fmt="", cmap="RdYlGn", vmin=0.15, vmax=0.80,
                xticklabels=DIMENSIONS, yticklabels=labels,
                linewidths=0.5, linecolor="white", cbar_kws={"label": "Score"},
                annot_kws={"fontsize": 9}, ax=ax)

    ax.set_title("Model × Dimension Score Heatmap", fontsize=13, fontweight="bold")
    ax.set_xlabel("Dimension", fontsize=11)
    ax.set_ylabel("Model", fontsize=11)

    for i, m in enumerate(models):
        best_dim = max(dimension_means[m], key=dimension_means[m].get)
        worst_dim = min(dimension_means[m], key=dimension_means[m].get)
        j_best = DIMENSIONS.index(best_dim)
        j_worst = DIMENSIONS.index(worst_dim)
        ax.add_patch(plt.Rectangle((j_best, i), 1, 1, fill=False, edgecolor="blue",
                                     lw=2.5, linestyle="-"))
        ax.add_patch(plt.Rectangle((j_worst, i), 1, 1, fill=False, edgecolor="red",
                                     lw=2.5, linestyle="--"))

    sig_notes = []
    for d in DIMENSIONS:
        if d in per_dim_stats:
            s = per_dim_stats[d]
            sig_notes.append(f"{d}: H={s['H']:.1f}, p={s['p']:.4f}")
    if sig_notes:
        fig.text(0.5, -0.02, "K-W: " + " | ".join(sig_notes),
                 ha="center", fontsize=8, style="italic")

    fig.tight_layout()
    return _save_fig(fig, "fig2_dimension_heatmap.png")


def fig_difficulty_boxplot(by_diff: dict, data: dict):
    """Fig 3: Difficulty-level boxplot (all models)."""
    models = sorted(data.keys(), key=lambda m: data[m].get("overall_score", 0), reverse=True)
    labels = [MODEL_SHORT.get(m, m) for m in models]

    n_rows = 3
    n_cols = 3
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 14), sharey=True)
    fig.suptitle("Score Distribution by Difficulty Level × Model", fontsize=14, fontweight="bold")

    for idx, model in enumerate(models):
        row, col = divmod(idx, n_cols)
        ax = axes[row, col]
        data_list = [by_diff[model].get(lvl, [0]) for lvl in DIFFICULTIES]

        bp = ax.boxplot(data_list, tick_labels=DIFFICULTIES, patch_artist=True,
                         widths=0.5, showmeans=True,
                         meanprops={"marker": "D", "markerfacecolor": "red", "markersize": 5})
        box_colors = ["#c6dbef", "#6baed6", "#3182bd", "#08519c"]
        for patch, color in zip(bp["boxes"], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_title(labels[idx], fontsize=10, fontweight="bold")
        ax.set_ylim(-0.05, 1.1)
        ax.grid(axis="y", alpha=0.3)
        if col == 0:
            ax.set_ylabel("Score", fontsize=9)

    fig.tight_layout()
    return _save_fig(fig, "fig3_difficulty_boxplot.png")


def fig_cohens_d_heatmap(d_mat: np.ndarray, names: list, models: list):
    """Fig 4: Cohen's d pairwise matrix."""
    mask = np.triu(np.ones_like(d_mat, dtype=bool), k=1)

    fig, ax = plt.subplots(figsize=(11, 9))
    annot = np.empty_like(d_mat, dtype=object)
    for i in range(len(names)):
        for j in range(len(names)):
            annot[i, j] = f"{d_mat[i, j]:.3f}" if i != j else "—"

    sns.heatmap(d_mat, annot=annot, fmt="", cmap="RdBu_r", center=0,
                vmin=-0.8, vmax=0.8, mask=mask,
                xticklabels=names, yticklabels=names,
                linewidths=0.3, cbar_kws={"label": "Cohen's d"},
                annot_kws={"fontsize": 8}, ax=ax)

    ax.set_title("Pairwise Cohen's d Effect Sizes (Row > Column = Row wins)",
                 fontsize=12, fontweight="bold")
    ax.tick_params(axis="x", rotation=45)
    ax.tick_params(axis="y", rotation=0)
    fig.tight_layout()
    return _save_fig(fig, "fig4_cohens_d.png")


def fig_correlation_clustermap(corr: np.ndarray, names: list):
    """Fig 5: Spearman correlation + hierarchical clustering."""
    cg = sns.clustermap(corr, annot=True, fmt=".3f", cmap="coolwarm",
                         vmin=0.6, vmax=1.0, method="average",
                         xticklabels=names, yticklabels=names,
                         figsize=(10, 9), linewidths=0.3,
                         annot_kws={"fontsize": 9},
                         dendrogram_ratio=0.12,
                         cbar_kws={"label": "Spearman ρ"})
    cg.ax_heatmap.set_title("Model Score Correlation + Hierarchical Clustering",
                             fontsize=12, fontweight="bold", pad=20)
    cg.ax_heatmap.tick_params(axis="x", rotation=45)
    cg.fig.tight_layout()
    return _save_fig(cg.fig, "fig5_correlation_cluster.png")


def fig_per_difficulty_line(by_diff: dict, data: dict):
    """Fig 6: Difficulty-sensitivity curves."""
    models = sorted(data.keys(), key=lambda m: data[m].get("overall_score", 0), reverse=True)
    families = [MODEL_FAMILY.get(m, "Other") for m in models]
    labels = [MODEL_SHORT.get(m, m) for m in models]

    fig, ax = plt.subplots(figsize=(11, 7))

    for m, fam, lbl in zip(models, families, labels):
        means = [np.mean(by_diff[m].get(lvl, [0])) for lvl in DIFFICULTIES]
        color = FAMILY_COLORS.get(fam, "#888")
        lw = 2.0 if m in ("mimo-v2-flash", "deepseek-v4-flash", "qwen3.5:9b") else 1.2
        ls = "-" if lw > 1.5 else "--"
        ax.plot(DIFFICULTIES, means, marker="o", label=lbl, color=color,
                linewidth=lw, linestyle=ls, markersize=6 if lw > 1.5 else 4)

    ax.set_ylim(0, 1)
    ax.set_xlabel("Difficulty Level", fontsize=11)
    ax.set_ylabel("Mean Score", fontsize=11)
    ax.set_title("Difficulty-Sensitivity Curves: How Scores Decay with Difficulty",
                 fontsize=12, fontweight="bold")
    ax.legend(loc="lower left", fontsize=8, ncol=2, title="Model", title_fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return _save_fig(fig, "fig6_difficulty_curves.png")


def fig_model_radar(data: dict):
    """Fig 7: Radar profile for each model (3×3 subplot)."""
    models = sorted(data.keys(), key=lambda m: data[m].get("overall_score", 0), reverse=True)
    labels = [MODEL_SHORT.get(m, m) for m in models]

    angles = np.linspace(0, 2 * np.pi, len(DIMENSIONS), endpoint=False).tolist()
    angles += angles[:1]

    fig, axes = plt.subplots(3, 3, figsize=(15, 15), subplot_kw={"projection": "polar"})
    fig.suptitle("Model Dimension Profiles (Radar)", fontsize=14, fontweight="bold")

    for idx, (model, ax) in enumerate(zip(models, axes.flat)):
        dim_scores = data[model].get("dimension_scores", {})
        values = [dim_scores.get(d, 0) for d in DIMENSIONS]
        values += values[:1]

        family = MODEL_FAMILY.get(model, "Other")
        color = FAMILY_COLORS.get(family, "#888")

        ax.fill(angles, values, alpha=0.25, color=color)
        ax.plot(angles, values, color=color, linewidth=2)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(DIMENSIONS, fontsize=9)
        ax.set_ylim(0, 0.85)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8])
        ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8"], fontsize=7)
        ax.set_title(labels[idx], fontsize=10, fontweight="bold", pad=15)
        ax.grid(alpha=0.4)

    fig.tight_layout()
    return _save_fig(fig, "fig7_model_radar.png")
