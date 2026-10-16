"""Medical RAG visualization: 4 charts for track 3 report.
Output: docs/assets/medical/*.png

Charts:
  1. rag_hybrid_vs_dense.png — Hybrid vs Dense-only per-query hit comparison
  2. rag_judge_heatmap.png — Judge score heatmap (14 models x 4 dimensions)
  3. section_distribution.png — Medical section label distribution (L1 hierarchy)
  4. rag_dimension_radar.png — 4-dimension radar: API group vs Local group
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import json
from collections import Counter

from mineru_core.paths import PROJECT_ROOT

# ── Paths (constants only, no I/O) ──
BASE = PROJECT_ROOT
KB_REPORT = BASE / "benchmark" / "eval" / "results" / "medical" / "kb_quality_report.json"
MANIFEST = BASE / "data" / "sources_medical" / "chunks" / "chunks_manifest.json"
OUT_DIR = BASE / "docs" / "assets" / "medical"


def _load_data() -> tuple[dict, list]:
    """Load report and manifest JSON. Creates OUT_DIR on first call."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(KB_REPORT, "r", encoding="utf-8") as f:
        report = json.load(f)
    if MANIFEST.exists():
        with open(MANIFEST, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    else:
        manifest = []
    return report, manifest

# ── Color palette (consistent with system_architecture) ──
C_API = "#4285F4"
C_LOCAL = "#34A853"
C_ACCENT = "#1A73E8"
C_TEXT = "#202124"
C_TEXT_SUB = "#5F6368"
C_BG = "#FAFBFC"
C_BORDER = "#DADCE0"
C_GOLD = "#F9AB00"
C_RED = "#EA4335"

# Model grouping
API_MODELS = ["ds-v4-flash", "ds-v4-pro", "mimo-v2-flash", "mimo-v2-pro",
              "mimo-v2.5", "mimo-v2.5-pro", "mm-m2.5", "mm-m2.7"]
LOCAL_MODELS = ["gemma3:4b", "llama3.2:3b", "qwen3.5:2b", "qwen3.5:4b",
                "qwen3.5:9b", "qwen3.5:35b"]

# Helper: short model labels
MODEL_SHORT = {
    "ds-v4-flash": "DS v4-Flash", "ds-v4-pro": "DS v4-Pro",
    "mimo-v2-flash": "MiMo Flash", "mimo-v2-pro": "MiMo Pro",
    "mimo-v2.5": "MiMo 2.5", "mimo-v2.5-pro": "MiMo 2.5 Pro",
    "mm-m2.5": "MM M2.5", "mm-m2.7": "MM M2.7",
    "gemma3:4b": "Gemma3 4B", "llama3.2:3b": "Llama3.2 3B",
    "qwen3.5:2b": "Qwen3.5 2B", "qwen3.5:4b": "Qwen3.5 4B",
    "qwen3.5:9b": "Qwen3.5 9B", "qwen3.5:35b": "Qwen3.5 35B",
}

# =====================================================================
# Chart 1: Hybrid vs Dense-only per-query comparison
# =====================================================================
def chart_hybrid_vs_dense(report: dict):
    rc = report.get("retrieval_comparison", {})
    per_query = rc.get("per_query", [])
    if not per_query:
        print("  ⚠ No per_query data, skipping chart 1")
        return

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.8), facecolor=C_BG)
    fig.suptitle("Hybrid vs Dense-only: Per-Query Retrieval Hit Analysis",
                 fontsize=15, fontweight="bold", color=C_TEXT, y=0.97)

    # Sub-plot A: Hit count bar (Hybrid vs Dense for each query)
    ax = axes[0]
    query_ids = [f"Q{i:02d}" for i in range(1, len(per_query) + 1)]
    hybrid_hits = []
    dense_hits = []
    for pq in per_query:
        src_label = pq.get("source_label", "")
        h_hit = sum(1 for d in pq.get("hybrid", {}).get("details", [])
                    if d.get("label", "") == src_label)
        d_hit = sum(1 for d in pq.get("dense_only", {}).get("details", [])
                    if d.get("label", "") == src_label)
        hybrid_hits.append(h_hit)
        dense_hits.append(d_hit)

    x = np.arange(len(query_ids))
    w = 0.35
    ax.set_axisbelow(True)
    bars1 = ax.bar(
        x - w/2,
        hybrid_hits,
        w,
        label="Hybrid",
        color=C_API,
        alpha=0.9,
        edgecolor="white",
        linewidth=0.5,
        zorder=3,
    )
    bars2 = ax.bar(
        x + w/2,
        dense_hits,
        w,
        label="Dense-only",
        color=C_LOCAL,
        alpha=0.9,
        edgecolor="white",
        linewidth=0.5,
        zorder=3,
    )

    total_h = sum(hybrid_hits)
    total_d = sum(dense_hits)
    fig.legend(
        [bars1, bars2],
        ["Hybrid", "Dense-only"],
        fontsize=8,
        loc="upper left",
        bbox_to_anchor=(0.06, 0.88),
        ncol=2,
        frameon=True,
        edgecolor=C_BORDER,
    )
    fig.text(
        0.5,
        0.875,
        f"Total correct source-label hits: Hybrid {total_h} | Dense {total_d}",
        fontsize=9,
        fontweight="bold",
        ha="center",
        va="center",
        color=C_TEXT,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#B8C0CC", alpha=0.98),
    )

    ax.set_title("Correct Source-Label Hits per Query", fontsize=10, fontweight="bold", color=C_TEXT, pad=10)
    ax.set_xlabel("Query", fontsize=9, color=C_TEXT_SUB)
    ax.set_ylabel("Correct label hits (0-5)", fontsize=9, color=C_TEXT_SUB)
    ax.set_xticks(x)
    ax.set_xticklabels(query_ids, fontsize=7, rotation=0)
    ax.set_ylim(0, 5.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3, linestyle="--", zorder=0)

    # Sub-plot B: Overlap ratio histogram
    ax = axes[1]
    overlaps = []
    for pq in per_query:
        h_ids = set(pq.get("hybrid", {}).get("chunk_ids", []))
        d_ids = set(pq.get("dense_only", {}).get("chunk_ids", []))
        union = h_ids | d_ids
        if union:
            overlaps.append(len(h_ids & d_ids) / len(union))
        else:
            overlaps.append(0.0)

    ax.set_axisbelow(True)
    ax.hist(
        overlaps,
        bins=10,
        range=(0, 1),
        color=C_GOLD,
        alpha=0.85,
        edgecolor="white",
        linewidth=0.8,
        zorder=3,
    )
    mean_overlap = np.mean(overlaps)
    ax.axvline(mean_overlap, color=C_RED, linestyle="--", linewidth=1.5, zorder=4)
    ax.set_title("Top-k Result Overlap Distribution", fontsize=10, fontweight="bold", color=C_TEXT, pad=10)
    ax.set_xlabel("Overlap ratio: intersection / union", fontsize=9, color=C_TEXT_SUB)
    ax.set_ylabel("Query count", fontsize=9, color=C_TEXT_SUB)
    ax.text(
        0.98,
        0.96,
        f"Mean overlap = {mean_overlap:.2f}",
        transform=ax.transAxes,
        fontsize=9,
        ha="right",
        va="top",
        color=C_TEXT_SUB,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor=C_BORDER),
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3, linestyle="--", zorder=0)

    plt.tight_layout(rect=[0, 0, 1, 0.82])
    path = OUT_DIR / "rag_hybrid_vs_dense.png"
    plt.savefig(path, dpi=200, bbox_inches="tight", facecolor=C_BG, edgecolor="none")
    plt.close()
    print(f"  OK {path.name}")


# =====================================================================
# Chart 2: Judge score heatmap (14 models x 4 dimensions)
# =====================================================================
def chart_judge_heatmap(report: dict):
    per_model = report.get("judge_matrix", {}).get("summary", {}).get("per_model", {})
    if not per_model:
        print("  ⚠ No per_model data, skipping chart 2")
        return

    dims = ["relevance", "completeness", "traceability", "accuracy"]
    dim_labels = ["Relevance", "Completeness", "Traceability", "Accuracy"]

    api_order = [m for m in API_MODELS if m in per_model]
    local_order = [m for m in LOCAL_MODELS if m in per_model]

    def _matrix(model_order):
        data = np.zeros((len(model_order), len(dims)))
        for i, m in enumerate(model_order):
            dim_data = per_model[m].get("dimensions", {})
            for j, d in enumerate(dims):
                data[i, j] = dim_data.get(d, {}).get("mean", 0)
        return data

    api_data = _matrix(api_order)
    local_data = _matrix(local_order)

    fig = plt.figure(figsize=(13.8, 6.8), facecolor=C_BG)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 0.045, 0.85], wspace=0.22)
    ax_api = fig.add_subplot(gs[0, 0])
    cax = fig.add_subplot(gs[0, 1])
    ax_local = fig.add_subplot(gs[0, 2])
    fig.suptitle("Medical RAG Judge Scores: API vs Local Models",
                 fontsize=15, fontweight="bold", color=C_TEXT, y=0.965)

    overall = report.get("judge_matrix", {}).get("summary", {}).get("overall", {})
    stats_text = (
        f"Total judgements: {overall.get('total_judgements', 0):,}    "
        f"Mean: {overall.get('mean_score', 0):.3f}    "
        f"Median: {overall.get('median_score', 0):.2f}"
    )
    fig.text(
        0.5,
        0.885,
        stats_text,
        fontsize=9.5,
        fontweight="bold",
        color=C_TEXT,
        ha="center",
        va="center",
        bbox=dict(boxstyle="round,pad=0.28", facecolor="white", edgecolor="#B8C0CC", alpha=0.98),
    )

    cmap = plt.cm.YlOrRd
    norm = mcolors.Normalize(vmin=0, vmax=1)

    def _draw(ax, data, model_order, title, labels_on_right=False):
        im = ax.imshow(data, cmap=cmap, norm=norm, aspect="auto")
        ax.set_title(title, fontsize=11, fontweight="bold", color=C_TEXT, pad=10)
        ax.set_xticks(np.arange(len(dims)))
        ax.set_yticks(np.arange(len(model_order)))
        ax.set_xticklabels(dim_labels, fontsize=9.5)
        ax.set_yticklabels([MODEL_SHORT.get(m, m) for m in model_order], fontsize=9.2, color=C_TEXT)
        if labels_on_right:
            ax.yaxis.tick_right()
            ax.tick_params(axis="y", labelright=True, labelleft=False, pad=6)
        else:
            ax.tick_params(axis="y", labelleft=True, labelright=False, pad=6)
        ax.tick_params(top=False, bottom=True, labeltop=False, labelbottom=True)
        plt.setp(ax.get_xticklabels(), rotation=0, ha="center")

        for i in range(len(model_order)):
            for j in range(len(dims)):
                val = data[i, j]
                text_color = "white" if val >= 0.5 else "#111827"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=9, color=text_color, fontweight="bold")

        for spine in ax.spines.values():
            spine.set_color("#111827")
            spine.set_linewidth(0.9)
        return im

    im = _draw(ax_api, api_data, api_order, f"API Models ({len(api_order)})")
    _draw(ax_local, local_data, local_order, f"Local Models ({len(local_order)})", labels_on_right=True)

    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("Mean score", fontsize=9.5, color=C_TEXT)
    cbar.ax.tick_params(colors=C_TEXT)

    fig.subplots_adjust(left=0.11, right=0.88, top=0.78, bottom=0.12, wspace=0.22)
    path = OUT_DIR / "rag_judge_heatmap.png"
    plt.savefig(path, dpi=200, bbox_inches="tight", facecolor=C_BG, edgecolor="none")
    plt.close()
    print(f"  OK {path.name}")


# =====================================================================
# Chart 3: Medical section distribution (L1 labels)
# =====================================================================
def chart_section_distribution(manifest: dict):
    chunks = manifest.get("chunks", []) if isinstance(manifest, dict) else manifest
    if not chunks:
        print("  ⚠ No chunks data, skipping chart 3")
        return

    # Count L1 labels (medical_parent is None)
    l1_counter = Counter()
    for c in chunks:
        parent = c.get("medical_parent")
        label = c.get("medical_label", "unknown")
        if parent is None and not label.startswith("_"):
            l1_counter[label] += 1

    # Sort by count descending
    sorted_items = sorted(l1_counter.items(), key=lambda x: -x[1])

    # Only top 12, group rest as "Other"
    top_n = 12
    if len(sorted_items) > top_n:
        main = dict(sorted_items[:top_n])
        other_count = sum(v for _, v in sorted_items[top_n:])
        main["_other"] = other_count
    else:
        main = dict(sorted_items)

    labels = list(main.keys())
    values = list(main.values())

    # Clean up label names for display
    def _fmt_label(l):
        if l == "_other":
            return "Other (minor)"
        return l.replace("_", " ").strip().title()
    display_labels = [_fmt_label(l) for l in labels]

    # Colors
    n = len(labels)
    colors = plt.cm.Set3(np.linspace(0, 1, n))
    np.random.seed(42)
    np.random.shuffle(colors)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5), facecolor=C_BG,
                                    gridspec_kw={"width_ratios": [1.2, 1]})
    fig.suptitle("Medical Section Distribution (3,348 Chunks, L1 Medical Labels)",
                 fontsize=14, fontweight="bold", color=C_TEXT, y=0.97)

    # Sub-plot A: Horizontal bar chart
    ax = ax1
    y_pos = np.arange(len(labels))
    bars = ax.barh(y_pos, values, color=colors, edgecolor="white", linewidth=0.8, height=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(display_labels, fontsize=9)
    ax.set_xlabel("Chunk count", fontsize=10, color=C_TEXT_SUB)
    ax.invert_yaxis()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", alpha=0.3, linestyle="--")

    # Annotate values
    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 5, bar.get_y() + bar.get_height() / 2,
                str(val), va="center", fontsize=8, color=C_TEXT_SUB)

    # Sub-plot B: Pie chart
    ax = ax2
    wedges, texts, autotexts = ax.pie(
        values, labels=None, autopct="%1.1f%%",
        colors=colors, startangle=90,
        textprops={"fontsize": 7},
        pctdistance=0.85,
        wedgeprops={"edgecolor": "white", "linewidth": 1}
    )
    # Legend outside
    legend_labels = [f"{l} ({v})" for l, v in zip(display_labels, values)]
    ax.legend(wedges, legend_labels, title="Sections",
              loc="center left", bbox_to_anchor=(1, 0.5),
              fontsize=7, title_fontsize=8)

    ax.set_title("Proportion", fontsize=10, color=C_TEXT_SUB)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    path = OUT_DIR / "section_distribution.png"
    plt.savefig(path, dpi=200, bbox_inches="tight", facecolor=C_BG, edgecolor="none")
    plt.close()
    print(f"  ✅ {path.name}")


def _group_model_dimensions(per_model, dims):
    """Group per-model dimension data by API vs LOCAL and compute means + top3."""
    api_dims = {d: [] for d in dims}
    local_dims = {d: [] for d in dims}
    for m in API_MODELS:
        if m in per_model:
            dd = per_model[m].get("dimensions", {})
            for d in dims:
                api_dims[d].append(dd.get(d, {}).get("mean", 0))
    for m in LOCAL_MODELS:
        if m in per_model:
            dd = per_model[m].get("dimensions", {})
            for d in dims:
                local_dims[d].append(dd.get(d, {}).get("mean", 0))
    api_means = [np.mean(api_dims[d]) for d in dims]
    local_means = [np.mean(local_dims[d]) for d in dims]

    top3_models = sorted(per_model.keys(), key=lambda m: per_model[m].get("mean_score", 0), reverse=True)[:3]
    top3_data = {
        m: [per_model[m].get("dimensions", {}).get(d, {}).get("mean", 0) for d in dims]
        for m in top3_models
    }
    return api_means, local_means, top3_data


# =====================================================================
# Chart 4: Radar chart — API group vs Local group dimension averages
# =====================================================================
def chart_dimension_radar(report: dict):
    per_model = report.get("judge_matrix", {}).get("summary", {}).get("per_model", {})
    if not per_model:
        print("  \u26a0 No per_model data, skipping chart 4")
        return

    dims = ["relevance", "completeness", "traceability", "accuracy"]
    dim_labels = ["Relevance", "Completeness", "Traceability", "Accuracy"]

    api_means, local_means, top3_data = _group_model_dimensions(per_model, dims)

    # Radar setup
    n_dims = len(dims)
    angles = np.linspace(0, 2 * np.pi, n_dims, endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    fig, ax = plt.subplots(figsize=(8.4, 7.2), subplot_kw={"projection": "polar"},
                           facecolor=C_BG)
    fig.suptitle("Judge Score Dimensions: API vs Local Models",
                 fontsize=14, fontweight="bold", color=C_TEXT, y=0.965)
    fig.subplots_adjust(left=0.06, right=0.70, top=0.86, bottom=0.12)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([])
    ax.tick_params(axis='x', pad=0)
    ax.set_ylim(0, 0.70)
    ax.set_yticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    ax.set_yticklabels(["0.1", "0.2", "0.3", "0.4", "0.5", "0.6"],
                        fontsize=7, color=C_TEXT_SUB)
    ax.grid(True, alpha=0.3, linestyle="--")

    label_positions = [
        ("Relevance", angles[0], 0.73, "center", "bottom"),
        ("Traceability", angles[2], 0.73, "center", "top"),
        ("Accuracy", angles[3], 0.72, "right", "center"),
    ]
    for label, angle, radius, ha, va in label_positions:
        ax.text(angle, radius, label, fontsize=10.5, fontweight="bold",
                ha=ha, va=va, color="black", clip_on=False)
    legend_left_x = 0.98
    table_left_x = 0.98
    table_width = 0.34
    ax.text(angles[1], 0.705, "Completeness", fontsize=10.5,
            fontweight="bold", ha="left", va="center",
            color="black", clip_on=False)

    # Plot API group
    values = api_means + api_means[:1]
    ax.plot(angles, values, "o-", linewidth=2.5, color=C_API, label="API Group (8)", alpha=0.9)
    ax.fill(angles, values, alpha=0.1, color=C_API)

    # Plot Local group
    values = local_means + local_means[:1]
    ax.plot(angles, values, "s-", linewidth=2.5, color=C_LOCAL, label="Local Group (6)", alpha=0.9)
    ax.fill(angles, values, alpha=0.1, color=C_LOCAL)

    dash_styles = [(6, 3), (3, 3), (1, 2)]
    colors_top = ["#FF7043", "#7E57C2", "#66BB6A"]
    for idx, (m, mdata) in enumerate(top3_data.items()):
        vals = mdata + mdata[:1]
        ax.plot(angles, vals, linewidth=1.0, color=colors_top[idx],
                alpha=0.45, label=f"  {MODEL_SHORT.get(m, m)}",
                dashes=dash_styles[idx])

    ax.legend(loc="lower left",
              bbox_to_anchor=(legend_left_x, 0.86), fontsize=7.3, title="Model Group",
              title_fontsize=8.3, frameon=True, borderaxespad=0.0,
              borderpad=0.35, handlelength=2.0, handletextpad=0.55,
              labelspacing=0.35)

    table_rows = [
        [label, f"{api_means[i]:.2f}", f"{local_means[i]:.2f}"]
        for i, label in enumerate(dim_labels)
    ]
    table = ax.table(
        cellText=table_rows,
        colLabels=["Dimension", "API", "Local"],
        cellLoc="center",
        colLoc="center",
        colWidths=[0.54, 0.23, 0.23],
        bbox=[table_left_x, -0.06, table_width, 0.22],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(6.6)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor(C_BORDER)
        cell.set_linewidth(0.6)
        cell.set_facecolor("white")
        cell.set_alpha(0.92)
        cell.PAD = 0.055
        if row == 0:
            cell.set_text_props(weight="bold", color=C_TEXT)
            cell.set_facecolor(C_BG)
        elif col == 0:
            cell.set_text_props(ha="left", color=C_TEXT_SUB)
        else:
            cell.set_text_props(color=C_TEXT_SUB)

    path = OUT_DIR / "rag_dimension_radar.png"
    plt.savefig(path, dpi=200, bbox_inches="tight", facecolor=C_BG, edgecolor="none")
    plt.close()
    print(f"  ✅ {path.name}")


def main():
    print("Generating Medical RAG visualizations...")
    report, manifest = _load_data()
    chart_hybrid_vs_dense(report)
    chart_judge_heatmap(report)
    chart_section_distribution(manifest)
    chart_dimension_radar(report)
    print(f"\nAll charts saved to {OUT_DIR}")


# =====================================================================
# Main
# =====================================================================
if __name__ == "__main__":
    main()
