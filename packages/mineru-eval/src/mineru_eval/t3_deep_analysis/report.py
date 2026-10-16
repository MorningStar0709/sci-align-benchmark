"""T3 Deep Analysis — Markdown report generation."""

from mineru_eval.t3_deep_analysis.core import (
    DIMENSIONS,
    _sig_stars,
)


def render_markdown(data, a6_result, a7_result, a8_result, a9_result):
    md = []
    md.append("# T3: 深度分析扩展 — A6~A9")
    md.append("")
    md.append("**Generated:** 2026-05-09 | **Models:** 9 | **Questions per model:** 500")
    md.append("")
    md.append("---")
    md.append("")

    # ── A6 ──
    md.append("## A6: 评分分布 (Score Distribution)")
    md.append("")
    md.append("| Model | Mean | Std | Skew | Kurt | Q25 | Median | Q75 | IQR | Zeros% |")
    md.append("|-------|------|-----|------|------|-----|--------|-----|-----|--------|")
    for st in a6_result["model_stats"]:
        md.append(f"| {st['name']} | {st['mean']:.4f} | {st['std']:.4f} | "
                  f"{st['skewness']:+.2f} | {st['kurtosis']:+.2f} | {st['q25']:.4f} | "
                  f"{st['median']:.4f} | {st['q75']:.4f} | {st['iqr']:.4f} | {st['pct_zeros']:.1f}% |")
    md.append("")

    md.append("### Distribution Charts")
    md.append("")
    md.append("- **A6 Density Ridges**: `a6_density_ridges.png` — Per-model score density curves")
    md.append("- **A6 Skewness vs Mean**: `a6_skewness_vs_mean.png` — Distribution shape vs central tendency")
    md.append("- **A6 ECDF**: `a6_ecdf.png` — Empirical cumulative distribution functions")
    md.append("")

    # ── A7 ──
    md.append("## A7: 维度 κ 一致性 (Dimension Agreement)")
    md.append("")
    md.append(f"**Kendall's W (Coefficient of Concordance):** {a7_result['kendall_W']:.4f}")
    md.append("")
    md.append(f"- Friedman χ² = {a7_result['friedman_chi2']:.2f}, p = {a7_result['friedman_p']:.4f} {_sig_stars(a7_result['friedman_p'])}")
    md.append(f"- ICC(2,1) = {a7_result['icc_2_1']:.4f}")
    md.append("")

    md.append("### Pairwise Dimension Spearman ρ")
    md.append("")
    md.append("| Dimension Pair | ρ | Significance |")
    md.append("|---------------|----|-------------|")
    for k, v in a7_result["dimension_pairwise_spearman"].items():
        md.append(f"| {k} | {v['spearman_rho']:.4f} | {_sig_stars(v['p'])} |")
    md.append("")

    md.append("### Charts")
    md.append("- **A7 Dimension Agreement**: `a7_dimension_agreement.png` — Score matrix normalized by dimension")
    md.append("- **A7 Concordance Radar**: `a7_concordance_radar.png` — Polar plot of dimension profiles")
    md.append("")

    # ── A8 ──
    md.append("## A8: 难度矩阵 (Difficulty Matrix)")
    md.append("")
    dm = a8_result["difficulty_matrix"]
    md.append("| Model | L1 | L2 | L3 | L4 | AbsΔ | RelΔ |")
    md.append("|-------|----|----|----|----|------|------|")
    for i in range(len(dm["models"])):
        vals = dm["values"][i]
        decay = a8_result["decay_details"][i]
        md.append(f"| {dm['models'][i]} | {vals[0]:.4f} | {vals[1]:.4f} | "
                  f"{vals[2]:.4f} | {vals[3]:.4f} | "
                  f"{decay['absolute_decay']:.4f} | {decay['relative_decay']:.1%} |")
    md.append("")

    md.append(f"**Most stable:** {a8_result['most_stable']['name']} (decay={a8_result['most_stable']['absolute_decay']:.4f})")
    md.append("")
    md.append(f"**Most sensitive:** {a8_result['most_sensitive']['name']} (decay={a8_result['most_sensitive']['absolute_decay']:.4f})")
    md.append("")

    md.append("### Item Difficulty Distribution")
    md.append("")
    md.append(f"Mean item difficulty (all models): **{a8_result['mean_item_difficulty']:.4f}**")
    md.append("")
    md.append("| Category | Count | % |")
    md.append("|----------|-------|---|")
    for k, v in a8_result["item_difficulty_bins"].items():
        md.append(f"| {k} | {v} | {v/500*100:.1f}% |")
    md.append("")

    md.append("### Charts")
    md.append("- **A8 Difficulty Matrix**: `a8_difficulty_matrix.png` — Model × Difficulty level heatmap")
    md.append("- **A8 Decay Curves**: `a8_decay_curves.png` — Score decay L1→L4 per model")
    md.append("- **A8 Item Difficulty Histogram**: `a8_item_difficulty_hist.png` — Distribution of item p-values")
    md.append("")

    # ── A9 ──
    md.append("## A9: 长度偏见 (Length Bias)")
    md.append("")
    md.append(f"**Global Pearson r:** {a9_result['global_pearson_r']:.4f} {_sig_stars(a9_result['global_pearson_p'])}")
    md.append(f"**Global Spearman ρ:** {a9_result['global_spearman_rho']:.4f} {_sig_stars(a9_result['global_spearman_p'])}")
    md.append("")

    md.append("| Model | AvgLen | Min | Max | Pearson r | p | Spearman ρ | p |")
    md.append("|-------|--------|-----|-----|-----------|----|------------|----|")
    for lc in a9_result["per_model"]:
        md.append(f"| {lc['name']} | {lc['mean_length']} | {lc['min_length']} | "
                  f"{lc['max_length']} | {lc['pearson_r']:.4f} | {_sig_stars(lc['pearson_p'])} | "
                  f"{lc['spearman_rho']:.4f} | {_sig_stars(lc['spearman_p'])} |")
    md.append("")

    md.append("### Dimension Breakdown")
    md.append("")
    for d in DIMENSIONS:
        if d in a9_result.get("dimension_corrs", {}):
            dc = a9_result["dimension_corrs"][d]
            md.append(f"- **Dim {d}**: Pearson r = {dc['pearson_r']:.4f} {_sig_stars(dc['p'])}")
    md.append("")

    md.append("### Charts")
    md.append("- **A9 Length vs Score**: `a9_length_vs_score.png` — Scatter plot per model with regression line")
    md.append("- **A9 Length Distribution**: `a9_length_distribution.png` — Histogram + binned mean score")
    md.append("- **A9 Length Ratio**: `a9_length_ratio.png` — Model/Expected length ratio vs score")
    md.append("")

    # ── Methodology ──
    md.append("## Methodology Notes")
    md.append("")
    md.append("- **A6**: Skewness = Fisher-Pearson standardized moment; Kurtosis = Fisher (excess, 0=normal). Dip test requires `scipy.stats` or `diptest` package.")
    md.append("- **A7**: Kendall's W measures rank concordance across dimensions (0=no agreement, 1=perfect). ICC(2,1) = two-way random effects, single rater.")
    md.append("- **A8**: Item difficulty = mean score across all 9 models (classical test theory p-value). Decay = L1 - L4 absolute difference.")
    md.append("- **A9**: Length = len(model_answer) in characters. Both Pearson (linear) and Spearman (monotonic) correlations reported.")
    md.append("")

    return "\n".join(md)
