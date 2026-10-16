"""A8: Difficulty Matrix — Model×Difficulty heatmap / item difficulty / decay patterns."""

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from mineru_eval.t3_deep_analysis.core import (
    MODEL_SHORT, MODEL_FAMILY, FAMILY_COLORS,
    DIFFICULTIES, OUTPUT_DIR,
    _save_fig,
)


def _build_diff_matrix(models_sorted, by_diff):
    """Build Model × Difficulty mean score matrix."""
    matrix = np.zeros((len(models_sorted), len(DIFFICULTIES)))
    for i, m in enumerate(models_sorted):
        for j, lvl in enumerate(DIFFICULTIES):
            vals = by_diff[m].get(lvl, [])
            matrix[i, j] = np.mean(vals) if vals else 0.0
    return matrix


def _compute_item_difficulty(all_scores, models_sorted):
    """Compute item difficulty (p-value) distribution across models."""
    avg_scores = np.mean([all_scores[m] for m in models_sorted], axis=0)
    bins = {
        "very_hard (0.0-0.2)": int((avg_scores < 0.2).sum()),
        "hard (0.2-0.4)": int(((avg_scores >= 0.2) & (avg_scores < 0.4)).sum()),
        "medium (0.4-0.6)": int(((avg_scores >= 0.4) & (avg_scores < 0.6)).sum()),
        "easy (0.6-0.8)": int(((avg_scores >= 0.6) & (avg_scores < 0.8)).sum()),
        "very_easy (0.8-1.0)": int((avg_scores >= 0.8).sum()),
    }
    return avg_scores, bins


def _compute_decay_info(models_sorted, diff_matrix, names):
    """Compute score decay L1→L4 for each model."""
    decay_info = []
    for i, m in enumerate(models_sorted):
        l1 = diff_matrix[i, 0] if diff_matrix[i, 0] > 0 else 0.001
        l4 = diff_matrix[i, -1]
        absolute_decay = l1 - l4
        decay_info.append({
            "model": m, "name": names[i], "L1": float(diff_matrix[i, 0]),
            "L2": float(diff_matrix[i, 1]), "L3": float(diff_matrix[i, 2]),
            "L4": float(diff_matrix[i, 3]),
            "absolute_decay": round(float(absolute_decay), 4),
            "relative_decay": round(float(absolute_decay / l1), 4),
            "decay_per_step": round(float(absolute_decay / 3), 4),
        })
    return decay_info


def _print_diff_matrix(models_sorted, names, diff_matrix):
    """Print the Model × Difficulty matrix to stdout."""
    print("\n  Model \u00d7 Difficulty Mean Score Matrix:")
    header = f"{'Model':<16}" + "".join(f"{lvl:>8}" for lvl in DIFFICULTIES)
    print(f"  {header}")
    print(f"  {'-' * (16 + 8 * len(DIFFICULTIES))}")
    for i, m in enumerate(models_sorted):
        vals = "".join(f"{diff_matrix[i, j]:>8.4f}" for j in range(len(DIFFICULTIES)))
        print(f"  {names[i]:<16}{vals}")


def _print_item_difficulty(bins, avg_scores):
    """Print item difficulty distribution."""
    print("\n  Item difficulty (avg score across models):")
    for k, v in bins.items():
        print(f"    {k}: {v} items ({v/len(avg_scores)*100:.1f}%)")


def _print_decay_info(decay_info):
    """Print decay info table to stdout."""
    ARROW = "\u2192"
    DELTA = "\u0394"
    print(f"\n  Score Decay L1{ARROW}L4:")
    header2 = f"{'Model':<16} {'L1':>7} {'L4':>7} {'Abs'+DELTA:>7} {'Rel'+DELTA:>7} {'/step':>7}"
    print(f"  {header2}")
    print(f"  {'-' * (16 + 7 * 5)}")
    for di in decay_info:
        print(f"  {di['name']:<16} {di['L1']:>7.4f} {di['L4']:>7.4f} "
              f"{di['absolute_decay']:>7.4f} {di['relative_decay']:>7.1%} {di['decay_per_step']:>7.4f}")


def analysis_a8_difficulty_matrix(all_scores, by_diff, by_dim, data):
    """A8: 难度矩阵 — 模型×难度/题目难度分析/衰退模式"""
    print("\n" + "=" * 60)
    print("A8: 难度矩阵 (Difficulty Matrix)")
    print("=" * 60)

    models_sorted = sorted(data.keys(), key=lambda m: data[m].get("overall_score", 0), reverse=True)
    names = [MODEL_SHORT[m] for m in models_sorted]

    # ── 1. Model × Difficulty mean score matrix ──
    diff_matrix = _build_diff_matrix(models_sorted, by_diff)
    _print_diff_matrix(models_sorted, names, diff_matrix)

    # ── 2. Item difficulty (p-value) distribution ──
    avg_scores, difficulty_bins = _compute_item_difficulty(all_scores, models_sorted)
    _print_item_difficulty(difficulty_bins, avg_scores)

    # ── 3. Score deterioration velocity ──
    decay_info = _compute_decay_info(models_sorted, diff_matrix, names)
    _print_decay_info(decay_info)

    # ── Chart 1: Model × Difficulty Heatmap ──
    print("\n  Chart 1: Model × Difficulty Heatmap...")
    fig, ax = plt.subplots(figsize=(10, 8))
    annot = np.empty_like(diff_matrix, dtype=object)
    for i in range(len(models_sorted)):
        for j in range(len(DIFFICULTIES)):
            annot[i, j] = f"{diff_matrix[i, j]:.4f}"
    sns.heatmap(diff_matrix, annot=annot, fmt="", cmap="YlOrRd",
                vmin=0.0, vmax=0.85,
                xticklabels=DIFFICULTIES, yticklabels=names,
                linewidths=0.5, linecolor="white",
                cbar_kws={"label": "Mean Score"},
                annot_kws={"fontsize": 9}, ax=ax)
    ax.set_title("A8: Model × Difficulty Level (Mean Score)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Difficulty Level", fontsize=11)
    ax.set_ylabel("Model", fontsize=11)
    fig.tight_layout()
    _save_fig(fig, "a8_difficulty_matrix.png")

    # ── Chart 2: Decay Curves (model family grouped) ──
    print("  Chart 2: Decay Curves...")
    fig, ax = plt.subplots(figsize=(11, 7))
    for i, m in enumerate(models_sorted):
        family = MODEL_FAMILY.get(m, "Other")
        color = FAMILY_COLORS.get(family, "#888")
        lw = 2.0 if names[i] in ["MiMo-v2-flash", "DS-v4-flash", "Qwen3.5:9b"] else 1.2
        ls = "-" if lw > 1.5 else "--"
        ax.plot(DIFFICULTIES, diff_matrix[i, :], marker="o", label=names[i],
                color=color, linewidth=lw, linestyle=ls, markersize=6 if lw > 1.5 else 4)

    ax.set_ylim(0, 1)
    ax.set_xlabel("Difficulty Level", fontsize=11)
    ax.set_ylabel("Mean Score", fontsize=11)
    ax.set_title("A8: Score Decay Curves (L1→L4)", fontsize=13, fontweight="bold")
    ax.legend(loc="lower left", fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _save_fig(fig, "a8_decay_curves.png")

    # ── Chart 3: Item Difficulty Histogram ──
    print("  Chart 3: Item Difficulty Histogram...")
    fig, ax = plt.subplots(figsize=(10, 6))
    bins = np.linspace(0, 1, 21)
    ax.hist(avg_scores, bins=bins, color="#3498DB", alpha=0.7, edgecolor="white",
            linewidth=0.5)
    ax.axvline(avg_scores.mean(), color="#E74C3C", linestyle="--", linewidth=2,
               label=f"Mean difficulty={avg_scores.mean():.3f}")
    ax.axvline(0.2, color="gray", linestyle=":", alpha=0.5)
    ax.axvline(0.8, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("Item Difficulty (avg score across 9 models)", fontsize=11)
    ax.set_ylabel("Number of Items", fontsize=11)
    ax.set_title("A8: Item Difficulty Distribution", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _save_fig(fig, "a8_item_difficulty_hist.png")

    result_a8 = {
        "analysis": "A8_difficulty_matrix",
        "n_models": len(models_sorted),
        "n_per_model": len(all_scores[models_sorted[0]]),
        "difficulty_matrix": {
            "models": names,
            "difficulty_levels": DIFFICULTIES,
            "values": [[round(float(diff_matrix[i, j]), 4) for j in range(len(DIFFICULTIES))]
                       for i in range(len(models_sorted))],
        },
        "item_difficulty_bins": difficulty_bins,
        "mean_item_difficulty": round(float(avg_scores.mean()), 4),
        "decay_details": decay_info,
        "most_stable": min(decay_info, key=lambda x: x["absolute_decay"]),
        "most_sensitive": max(decay_info, key=lambda x: x["absolute_decay"]),
        "interpretation": (
            "Item difficulty (p-value) 表示所有模型在该题上的平均得分。"
            "<0.2 为非常困难，>0.8 为非常容易。"
            "Decay 越大表示模型对难度变化越敏感。"
        ),
    }
    import json
    json_path = OUTPUT_DIR / "a8_difficulty_matrix.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result_a8, f, ensure_ascii=False, indent=2)
    print(f"  JSON: {json_path}")

    return result_a8
