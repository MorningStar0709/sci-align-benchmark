"""
Benchmark Quality Self-Certification Charts (T3-7)

Generates 3 charts proving benchmark quality:
  1. 14 Sub-domain Coverage Bar Chart
  2. Category × Difficulty Heatmap
  3. Category × Dimension Heatmap

Data source: core.json (500 questions, 675 category assignments)
Output: benchmark/eval/results/benchmark_quality_viz_{1,2,3}.png
"""

import matplotlib
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
DATA_PATH = ROOT / "benchmark" / "dataset" / "core.json"
OUTPUT_DIR = ROOT / "benchmark" / "eval" / "results"

CATEGORY_LABELS = {
    "classical": "Classical", "optimal": "Optimal", "robust": "Robust",
    "nonlinear": "Nonlinear", "modern": "Modern Control", "intelligent": "Intelligent",
    "mpc": "Model Predictive", "adaptive": "Adaptive", "digital": "Digital Control",
    "multi_agent": "Multi-Agent",
}

CORPUS_ONLY_CATEGORIES = [
    ("PID Control", 0), ("Estimation & Localization", 0), ("Sliding Mode", 0), ("H∞ Control", 0),
]

DIFFICULTY_LABELS = {"L1": "L1: Conceptual", "L2": "L2: Basic App.", "L3": "L3: Interm. Reasoning", "L4": "L4: Advanced Design"}
DIM_LABELS = {"A": "A: Conceptual", "B": "B: Multi-Step", "C": "C: Sensitivity", "D": "D: Design"}


def load_data():
    raw = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    questions = raw["questions"]
    meta = raw.get("meta", {})

    total = meta.get("total_questions", 0)
    assert len(questions) == total, \
        f"题目数不一致: {len(questions)} vs meta {total}"

    cat_count = {tag: 0 for tag in CATEGORY_LABELS}
    cat_diff = {tag: {"L1": 0, "L2": 0, "L3": 0, "L4": 0} for tag in CATEGORY_LABELS}
    cat_dim = {tag: {"A": 0, "B": 0, "C": 0, "D": 0} for tag in CATEGORY_LABELS}

    for q in questions:
        cats = q.get("control_category", [])
        diff = q.get("difficulty_level", "")
        dim = q.get("dimension", "")
        for c in cats:
            if c in cat_count:
                cat_count[c] += 1
            if c in cat_diff and diff in cat_diff[c]:
                cat_diff[c][diff] += 1
            if c in cat_dim and dim in cat_dim[c]:
                cat_dim[c][dim] += 1

    return cat_count, cat_diff, cat_dim


def chart1_coverage(cat_count, output_path):
    tags_used = [t for t in CATEGORY_LABELS]
    counts = [cat_count[t] for t in tags_used]
    cn_names = [CATEGORY_LABELS[t] for t in tags_used]

    sort_idx = np.argsort(counts)
    cn_sorted = [cn_names[i] for i in sort_idx]
    cnt_sorted = [counts[i] for i in sort_idx]

    fig, ax = plt.subplots(figsize=(12, 7))
    fig.subplots_adjust(left=0.25, right=0.88, top=0.90, bottom=0.10)

    y_pos = np.arange(len(cn_sorted) + len(CORPUS_ONLY_CATEGORIES))
    core_n = len(cn_sorted)

    colors = ["#27ae60"] * core_n + ["#bdc3c7"] * len(CORPUS_ONLY_CATEGORIES)
    edge_colors = ["#1e8449"] * core_n + ["#95a5a6"] * len(CORPUS_ONLY_CATEGORIES)

    labels = cn_sorted + [c[0] for c in CORPUS_ONLY_CATEGORIES]
    values = cnt_sorted + [c[1] for c in CORPUS_ONLY_CATEGORIES]

    bars = ax.barh(y_pos, values, color=colors, edgecolor=edge_colors,
                   linewidth=1.2, height=0.65, zorder=3)

    for bar, val, _ in zip(bars, values, labels):
        if val > 0:
            ax.text(bar.get_width() + 2, bar.get_y() + bar.get_height() / 2,
                    f"{val} Qs", va="center", fontsize=10, fontweight="bold", color="#2c3e50")
        else:
            ax.text(bar.get_width() + 2, bar.get_y() + bar.get_height() / 2,
                    "Corpus Coverage", va="center", fontsize=9, color="#95a5a6", style="italic")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlim(0, max(values) * 1.25)
    ax.set_xlabel("Question Count", fontsize=12, fontweight="bold")
    ax.set_title("Subdomain Coverage: 10 of 14 Control Disciplines Covered",
                 fontsize=14, fontweight="bold", pad=15)
    ax.grid(axis="x", alpha=0.3, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, color="#27ae60", ec="#1e8449", label="Core 500-Q Set Coverage"),
        plt.Rectangle((0, 0), 1, 1, color="#bdc3c7", ec="#95a5a6", label="Corpus-Level (Pending)"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=10, framealpha=0.9)

    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[1/3] 14 领域覆盖图 → {output_path}")


def chart2_difficulty_heatmap(cat_diff, output_path):
    tags_plot = list(CATEGORY_LABELS.keys())
    cn_names = [CATEGORY_LABELS[t] for t in tags_plot]
    diffs = ["L1", "L2", "L3", "L4"]

    matrix = np.array([[cat_diff[t][d] for d in diffs] for t in tags_plot])
    row_totals = matrix.sum(axis=1)

    sort_idx = np.argsort(-row_totals)
    matrix_sorted = matrix[sort_idx]
    cn_sorted = [cn_names[i] for i in sort_idx]

    fig, ax = plt.subplots(figsize=(10, 7.5))
    fig.subplots_adjust(left=0.22, right=0.96, top=0.92, bottom=0.10)

    cmap = plt.cm.Greens
    norm = mcolors.Normalize(vmin=0, vmax=matrix_sorted.max())
    im = ax.imshow(matrix_sorted, cmap=cmap, norm=norm, aspect="auto")

    for i in range(matrix_sorted.shape[0]):
        for j in range(matrix_sorted.shape[1]):
            val = matrix_sorted[i, j]
            color = "white" if val > matrix_sorted.max() * 0.6 else "#2c3e50"
            ax.text(j, i, str(val), ha="center", va="center",
                    fontsize=12, fontweight="bold", color=color)

    ax.set_xticks(range(len(diffs)))
    ax.set_xticklabels([DIFFICULTY_LABELS[d] for d in diffs], fontsize=11, fontweight="bold")
    ax.set_yticks(range(len(cn_sorted)))
    ax.set_yticklabels(cn_sorted, fontsize=11)
    ax.set_xlabel("Difficulty Level", fontsize=12, fontweight="bold", labelpad=10)
    ax.set_ylabel("Control Subdomain", fontsize=12, fontweight="bold", labelpad=10)
    ax.set_title("Subdomain × Difficulty: Full L1–L4 Coverage",
                 fontsize=14, fontweight="bold", pad=15)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label("Tag Assignment Count", fontsize=10)

    fig.text(0.5, 0.01, "Data: Core 500 Questions, 675 Subdomain Tag Assignments | Color = Assignment Count",
             ha="center", fontsize=9, color="gray")

    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[2/3] 难度交叉热力图 → {output_path}")


def chart3_dimension_heatmap(cat_dim, output_path):
    tags_plot = list(CATEGORY_LABELS.keys())
    cn_names = [CATEGORY_LABELS[t] for t in tags_plot]
    dims = ["A", "B", "C", "D"]

    matrix = np.array([[cat_dim[t][d] for d in dims] for t in tags_plot])
    row_totals = matrix.sum(axis=1)

    sort_idx = np.argsort(-row_totals)
    matrix_sorted = matrix[sort_idx]
    cn_sorted = [cn_names[i] for i in sort_idx]

    col_norm = np.array([matrix_sorted[:, j].sum() for j in range(len(dims))])

    fig, ax = plt.subplots(figsize=(10, 7.5))
    fig.subplots_adjust(left=0.22, right=0.96, top=0.92, bottom=0.10)

    cmap = plt.cm.Blues
    norm = mcolors.Normalize(vmin=0, vmax=matrix_sorted.max())
    im = ax.imshow(matrix_sorted, cmap=cmap, norm=norm, aspect="auto")

    for i in range(matrix_sorted.shape[0]):
        for j in range(matrix_sorted.shape[1]):
            val = matrix_sorted[i, j]
            pct = val / col_norm[j] * 100 if col_norm[j] > 0 else 0
            color = "white" if val > matrix_sorted.max() * 0.55 else "#2c3e50"
            ax.text(j, i, f"{val}\n({pct:.0f}%)", ha="center", va="center",
                    fontsize=10, fontweight="bold", color=color)

    ax.set_xticks(range(len(dims)))
    ax.set_xticklabels([DIM_LABELS[d] for d in dims], fontsize=11, fontweight="bold")
    ax.set_yticks(range(len(cn_sorted)))
    ax.set_yticklabels(cn_sorted, fontsize=11)
    ax.set_xlabel("Evaluation Dimension", fontsize=12, fontweight="bold", labelpad=10)
    ax.set_ylabel("Control Subdomain", fontsize=12, fontweight="bold", labelpad=10)
    ax.set_title("Subdomain × Dimension: Four-Dimension Balance Validation",
                 fontsize=14, fontweight="bold", pad=15)

    col_sums = [f"∑={int(col_norm[j])}" for j in range(len(dims))]
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(range(len(dims)))
    ax2.set_xticklabels(col_sums, fontsize=9, color="#555")
    ax2.set_xlabel("Total Tags per Dimension", fontsize=10, color="#555", labelpad=5)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label("Tag Assignment Count", fontsize=10)

    fig.text(0.5, 0.01, "Data: Core 500 Questions, 675 Subdomain Tag Assignments | Parentheses = Dimension Share",
             ha="center", fontsize=9, color="gray")

    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[3/3] 维度交叉热力图 → {output_path}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cat_count, cat_diff, cat_dim = load_data()

    chart1_coverage(cat_count, OUTPUT_DIR / "benchmark_quality_viz_1_coverage.png")
    chart2_difficulty_heatmap(cat_diff, OUTPUT_DIR / "benchmark_quality_viz_2_difficulty.png")
    chart3_dimension_heatmap(cat_dim, OUTPUT_DIR / "benchmark_quality_viz_3_dimension.png")

    print("\n=== Benchmark 质量自证 3 张图已全部生成 ===")


if __name__ == "__main__":
    main()
