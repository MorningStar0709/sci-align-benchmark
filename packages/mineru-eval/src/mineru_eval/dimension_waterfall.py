"""
D-Dimension Waterfall Chart — Task 1.3 Phase 6

Shows D-dimension (开放设计) scores across 9 models sorted high→low,
with adjacent-model gaps annotated. The 58-point range (0.736→0.156)
demonstrates superior discriminative power of the D dimension.

Data source: leaderboard.json (9 models × 500 questions)
"""

import matplotlib
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
LEADERBOARD_PATH = ROOT / "benchmark" / "eval" / "results" / "leaderboard.json"


def load_d_scores():
    data = json.loads(LEADERBOARD_PATH.read_text(encoding="utf-8"))
    entries = []
    for m in data["models"]:
        dim_scores = m.get("dimension_scores", {})
        d_score = dim_scores.get("D", 0)
        entries.append((m["model"], d_score))
    entries.sort(key=lambda x: x[1], reverse=True)
    return entries


def build_gradient_colors(n, high_color="#2ecc71", low_color="#e74c3c"):
    return [mcolors.to_hex(c) for c in mcolors.LinearSegmentedColormap.from_list(
        "d_grad", [low_color, high_color])(np.linspace(0, 1, n))]


DISPLAY_NAMES = {
    "mimo-v2-flash":            "MiMo-v2-flash",
    "qwen3.5:9b":               "Qwen3.5-9B",
    "qwen3.7-plus":             "Qwen3.7-plus",
    "qwen3.5-plus":             "Qwen3.5-plus",
    "deepseek-v4-flash":        "DeepSeek-v4-flash",
    "deepseek-v4-pro":          "DeepSeek-v4-pro",
    "mimo-v2.5-pro":            "MiMo-v2.5-pro",
    "mimo-v2-pro":              "MiMo-v2-pro",
    "mimo-v2.5":                "MiMo-v2.5",
}


def _safe_round_d(v):
    """round(v*100, 1) with epsilon to defeat banker's rounding on .x25/.x75"""
    return round(v * 100 + 1e-9, 1)


def make_waterfall(models, scores):
    n = len(models)
    colors = build_gradient_colors(n, high_color="#e74c3c", low_color="#2ecc71")
    displayed = [_safe_round_d(s) for s in scores]
    labels = [DISPLAY_NAMES.get(m, m) for m in models]
    gap_58 = displayed[0] - displayed[-1]

    fig, ax = plt.subplots(figsize=(11, 7))
    fig.subplots_adjust(left=0.22, right=0.92, top=0.88, bottom=0.12)

    y_positions = np.arange(n - 1, -1, -1)
    bars = ax.barh(y_positions, displayed, color=colors,
                   edgecolor="white", linewidth=0.8, height=0.60)

    for bar, val in zip(bars, displayed):
        ax.text(bar.get_width() + 0.8, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}", va="center", fontsize=10.5, fontweight="bold",
                color="#2c3e50")

    for i in range(n - 1):
        y_mid = (y_positions[i] + y_positions[i + 1]) / 2
        gap = displayed[i] - displayed[i + 1]
        ax.annotate(f"↓ {gap:.1f}",
                    xy=((displayed[i] + displayed[i + 1]) / 2, y_mid),
                    fontsize=8, color="#7f8c8d", ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="#f8f9fa",
                              edgecolor="#dee2e6", alpha=0.9))

    ax.annotate(f"Range = {gap_58:.0f} pts",
                xy=(gap_58 / 2, -1.3), xycoords="data",
                fontsize=12, fontweight="bold", color="#c0392b",
                ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="#fdecea",
                          edgecolor="#e74c3c", alpha=0.95),
                annotation_clip=False)

    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels, fontsize=10.5, fontfamily="monospace",
                       ha="right")
    ax.tick_params(axis="y", pad=8)
    ax.set_xlim(0, 96)
    ax.set_ylim(-1.8, n - 0.5)
    ax.set_xlabel("D-Dimension Score (0-100)", fontsize=11, color="#555", labelpad=8)
    ax.set_title("D-Dimension (Open Design) Score Distribution",
                 fontsize=14, fontweight="bold", pad=24, color="#2c3e50")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#dee2e6")
    ax.spines["bottom"].set_color("#dee2e6")
    ax.tick_params(axis="x", colors="#888")
    ax.grid(axis="x", alpha=0.3, color="#dee2e6")

    fig.text(0.5, 0.02, "Data: leaderboard.json | 9 models × 500 questions | D-Dimension (Open Design) | Judge: deepseek-v4-flash",
             ha="center", fontsize=8, color="#666")

    output_dir = ROOT / "benchmark" / "eval" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "dimension_waterfall.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"D-Dimension waterfall chart saved: {output_path}")
    return output_path


if __name__ == "__main__":
    entries = load_d_scores()
    models, scores = zip(*entries)
    print(f"Loaded {len(entries)} models | D range: {scores[-1]*100:.1f} ~ {scores[0]*100:.1f} = {scores[0]*100 - scores[-1]*100:.0f} pts gap")
    make_waterfall(list(models), list(scores))
