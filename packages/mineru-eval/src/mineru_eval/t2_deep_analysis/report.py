"""
T2: 9-Model Deep Analysis — Markdown report generation.
"""

from __future__ import annotations

from collections import Counter

import numpy as np

from .core import MODEL_SHORT, MODEL_FAMILY, DIMENSIONS, DIFFICULTIES, _sig_stars, _d_interpret


def _render_stats_section(md, kw_result, dunn_df, names_d, d_mat):
    """Section 2: Kruskal-Wallis, Dunn's Post-Hoc, Cohen's d."""
    md.append("## 2. Statistical Inference")
    md.append("")

    if kw_result:
        kw = kw_result
        md.append("**Kruskal-Wallis Test** (non-parametric ANOVA):")
        md.append(f"- H = {kw['H']:.2f}, df = {kw['df']}, p = {kw['p']:.6f} {_sig_stars(kw['p'])}")
        md.append(f"- n = {kw['n_models']} models \u00d7 {kw['n_total'] // kw['n_models']} records = {kw['n_total']} total")
        conclusion = "significant differences exist among models" if kw["p"] < 0.05 else "no significant differences (p \u2265 0.05)"
        md.append(f"- Conclusion: **{conclusion}**.")
        md.append("")

    md.append("### 2.1 Dunn's Post-Hoc (Bonferroni corrected)")
    md.append("")
    if dunn_df is not None:
        md.append("Significant pairwise comparisons (p < 0.05):")
        md.append("")
        md.append("| Comparison | p-value | Significance |")
        md.append("|-----------|---------|-------------|")
        sig_count = 0
        n_models = len(dunn_df)
        for i in range(n_models):
            for j in range(i + 1, n_models):
                p_val = dunn_df.iloc[i, j]
                if p_val < 0.05:
                    sig_count += 1
                    md.append(f"| {dunn_df.index[i]} vs {dunn_df.columns[j]} | {p_val:.6f} | {_sig_stars(p_val)} |")
        md.append("")
        md.append(f"**{sig_count}** significant pairs out of {n_models * (n_models - 1) // 2} total comparisons.")
        md.append("")
    else:
        md.append("*Dunn's test skipped \u2014 fewer than 3 models loaded.*")
        md.append("")

    md.append("### 2.2 Cohen's d Effect Sizes (Top vs Others)")
    md.append("")
    md.append(f"Reference model: **{names_d[0]}** (rank 1)")
    md.append("")
    md.append("| vs Model | Cohen's d | Interpretation |")
    md.append("|----------|----------|----------------|")
    for j in range(1, len(names_d)):
        d_val = d_mat[0, j]
        md.append(f"| {names_d[j]} | {d_val:+.3f} | {_d_interpret(d_val)} |")
    md.append("")


def _render_dimension_section(md, models_sorted, data, per_dim_kw):
    """Section 3: Per-dimension analysis."""
    md.append("## 3. Per-Dimension Analysis")
    md.append("")
    md.append("### 3.1 Dimension-wise Kruskal-Wallis")
    md.append("")
    md.append("| Dimension | H | p | Best | Best Mean | Worst | Worst Mean | Spread |")
    md.append("|-----------|----|----|------|-----------|-------|-----------|--------|")
    for d in DIMENSIONS:
        if d in per_dim_kw:
            s = per_dim_kw[d]
            md.append(f"| {d} | {s['H']:.1f} | {s['p']:.4f} {_sig_stars(s['p'])} | "
                      f"{s['best_model']} | {s['best_mean']:.4f} | "
                      f"{s['worst_model']} | {s['worst_mean']:.4f} | {s['spread']:.4f} |")
    md.append("")

    md.append("### 3.2 Model Dimension Profiles")
    md.append("")
    for m in models_sorted:
        ds = data[m].get("dimension_scores", {})
        all_dims = [(d, ds.get(d, 0)) for d in DIMENSIONS]
        best = max(all_dims, key=lambda x: x[1])
        worst = min(all_dims, key=lambda x: x[1])
        md.append(f"- **{MODEL_SHORT[m]}** (Overall: {data[m].get('overall_score', 0):.4f}): "
                  f"Best in {best[0]} ({best[1]:.4f}), weakest in {worst[0]} ({worst[1]:.4f}) "
                  f"| Spread: {best[1] - worst[1]:.4f}")
    md.append("")


def _render_difficulty_section(md, models_sorted, by_diff, per_diff_kw):
    """Section 4: Difficulty-level analysis with decay rates."""
    md.append("## 4. Difficulty-Level Analysis")
    md.append("")
    md.append("### 4.1 Difficulty-wise Kruskal-Wallis")
    md.append("")
    md.append("| Difficulty | H | p | Best | Worst | Spread |")
    md.append("|-----------|----|----|------|-------|--------|")
    for lvl in DIFFICULTIES:
        if lvl in per_diff_kw:
            s = per_diff_kw[lvl]
            md.append(f"| {lvl} | {s['H']:.1f} | {s['p']:.4f} {_sig_stars(s['p'])} | "
                      f"{s['best_model']} | {s['worst_model']} | {s['spread']:.4f} |")
    md.append("")

    md.append("### 4.2 Difficulty Decay Rates")
    md.append("")
    md.append("| Model | L1 Mean | L4 Mean | Decay (L1\u2192L4) | Sensitivity |")
    md.append("|-------|---------|---------|---------------|-------------|")
    decay_rates = []
    for m in models_sorted:
        l1_m = np.mean(by_diff[m].get("L1", [0]))
        l4_m = np.mean(by_diff[m].get("L4", [0]))
        decay = l1_m - l4_m
        sensitivity = decay / l1_m if l1_m > 0 else 0
        decay_rates.append((m, decay, sensitivity))
        md.append(f"| {MODEL_SHORT[m]} | {l1_m:.4f} | {l4_m:.4f} | {decay:+.4f} | {sensitivity:.1%} |")
    md.append("")

    most_stable = min(decay_rates, key=lambda x: x[1])
    most_sensitive = max(decay_rates, key=lambda x: x[1])
    md.append(f"- **Most stable** across difficulty: {MODEL_SHORT[most_stable[0]]} (decay = {most_stable[1]:+.4f})")
    md.append(f"- **Most sensitive** to difficulty: {MODEL_SHORT[most_sensitive[0]]} (decay = {most_sensitive[1]:+.4f})")
    md.append("")


def _render_correlation_issues(md, models_sorted, data, names_corr, corr):
    """Sections 5-6: Score correlation and issue patterns."""
    md.append("## 5. Model Score Correlation")
    md.append("")
    md.append(f"Mean pairwise Spearman \u03c1: **{np.mean(corr[np.triu_indices_from(corr, k=1)]):.4f}**")
    md.append("")
    md.append("| Model Pair | \u03c1 | Strength |")
    md.append("|-----------|------|----------|")
    for i in range(len(names_corr)):
        for j in range(i + 1, len(names_corr)):
            r = corr[i, j]
            strength = "Very Strong" if r > 0.8 else "Strong" if r > 0.6 else "Moderate" if r > 0.4 else "Weak"
            md.append(f"| {names_corr[i]} \u2194 {names_corr[j]} | {r:.3f} | {strength} |")
    md.append("")

    md.append("## 6. Cross-Model Issue Patterns")
    md.append("")
    all_issues = Counter()
    model_issue_counts = {}
    for m in models_sorted:
        issues = Counter()
        for rec in data[m].get("records", []):
            for iss in (rec.get("issues") or []):
                issues[iss] += 1
        model_issue_counts[m] = issues
        all_issues.update(issues)

    md.append("### 6.1 Top-10 Issues (Aggregated)")
    md.append("")
    md.append("| Rank | Issue | Total Count | Models Affected |")
    md.append("|------|-------|-------------|-----------------|")
    for rank, (iss, cnt) in enumerate(all_issues.most_common(10), 1):
        affected = sum(1 for m in models_sorted if iss in model_issue_counts[m])
        md.append(f"| {rank} | {iss} | {cnt} | {affected}/9 |")
    md.append("")


def render_markdown(data, all_scores, by_dim, by_diff, dimension_means,
                    kw_result, dunn_df, d_mat, names_d, models_d,
                    corr, names_corr, per_dim_kw, per_diff_kw):
    """Generate comprehensive Markdown report."""
    md = []
    md.append("# T2: 9-Model Deep Analysis Report")
    md.append("")
    md.append("**Generated:** 2026-05-06 | **Models:** 9 | **Questions per model:** 500")
    md.append("")
    md.append("---")
    md.append("")

    # ── Executive Summary ──
    md.append("## 1. Executive Summary")
    md.append("")

    models_sorted = sorted(data.keys(), key=lambda m: data[m].get("overall_score", 0), reverse=True)

    md.append("| Rank | Model | Overall | A | B | C | D | Family |")
    md.append("|------|-------|---------|----|----|----|----|--------|")
    for rank, m in enumerate(models_sorted, 1):
        d = data[m]
        ov = d.get("overall_score", 0)
        ds = d.get("dimension_scores", {})
        fam = MODEL_FAMILY.get(m, "—")
        md.append(f"| {rank} | {MODEL_SHORT.get(m, m)} | **{ov:.4f}** | "
                  f"{ds.get('A', 0):.4f} | {ds.get('B', 0):.4f} | "
                  f"{ds.get('C', 0):.4f} | {ds.get('D', 0):.4f} | {fam} |")
    md.append("")

    top = models_sorted[0]
    bottom = models_sorted[-1]
    top_ov = data[top].get("overall_score", 0)
    bottom_ov = data[bottom].get("overall_score", 0)
    md.append(f"**Winner:** {MODEL_SHORT[top]} ({top_ov:.4f}) "
              f"\u2014 {top_ov - bottom_ov:.4f} ahead of last place ({bottom_ov:.4f}).")
    md.append("")

    _render_stats_section(md, kw_result, dunn_df, names_d, d_mat)
    _render_dimension_section(md, models_sorted, data, per_dim_kw)
    _render_difficulty_section(md, models_sorted, by_diff, per_diff_kw)
    _render_correlation_issues(md, models_sorted, data, names_corr, corr)

    # ── Charts ──
    md.append("## 7. Generated Charts")
    md.append("")
    charts = [
        ("Overall Score Bar Chart (95% CI)", "fig1_overall_bar.png"),
        ("Model × Dimension Heatmap", "fig2_dimension_heatmap.png"),
        ("Difficulty-Level Boxplots", "fig3_difficulty_boxplot.png"),
        ("Cohen's d Pairwise Matrix", "fig4_cohens_d.png"),
        ("Score Correlation + Hierarchical Clustering", "fig5_correlation_cluster.png"),
        ("Difficulty-Sensitivity Curves", "fig6_difficulty_curves.png"),
        ("Model Dimension Radar Profiles", "fig7_model_radar.png"),
    ]
    for title, fname in charts:
        md.append(f"- **{title}**: `{fname}`")
    md.append("")

    # ── Methodology ──
    md.append("## 8. Methodology Notes")
    md.append("")
    md.append("- **Kruskal-Wallis**: Non-parametric alternative to one-way ANOVA; does not assume normality.")
    md.append("- **Dunn's Post-hoc**: Multiple comparison correction via Bonferroni; identifies _which_ pairs differ.")
    md.append("- **Cohen's d**: Pooled standard deviation; thresholds: small (0.2), medium (0.5), large (0.8).")
    md.append("- **Spearman ρ**: Rank correlation; robust to non-linear monotonic relationships.")
    md.append("- **Hierarchical Clustering**: Average linkage on (1 − ρ) distance matrix.")
    md.append("- **95% CI**: ±1.96 × SE; assumes asymptotic normality (n=500, CLT applies).")
    md.append("")

    return "\n".join(md)
