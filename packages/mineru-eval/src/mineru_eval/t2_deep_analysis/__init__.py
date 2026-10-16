"""
T2: 9-Model Deep Analysis — 统计推断 + 可视化

重构子包：
  core.py      — 常量 + 数据加载
  stats.py     — 统计检验 (Kruskal-Wallis, Dunn's, Cohen's d, Spearman)
  visualize.py — 7 张图表
  report.py    — Markdown 报告生成
  __init__.py  — main 入口 + re-export
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from mineru_eval.t2_deep_analysis import (
    core as _t2_core,
    stats as _t2_stats_mod,
    visualize as _t2_viz_mod,
    report as _t2_report_mod,
)
from mineru_eval.t2_deep_analysis.core import (
    ROOT, CORE_9, MODEL_SHORT, MODEL_FAMILY, FAMILY_COLORS,
    DIMENSIONS, DIFFICULTIES, RNG, OUTPUT_DIR,
    load_core_9, extract_records,
    _save_fig, _sig_stars, _d_interpret, _model_rank_desc,
)
from mineru_eval.t2_deep_analysis.stats import (
    kruskal_wallis_test, dunn_posthoc_test, cohens_d_matrix,
    spearman_correlation, per_dimension_kruskal, per_difficulty_kruskal,
)
from mineru_eval.t2_deep_analysis.visualize import (
    fig_overall_bar, fig_dimension_heatmap, fig_difficulty_boxplot,
    fig_cohens_d_heatmap, fig_correlation_clustermap,
    fig_per_difficulty_line, fig_model_radar,
)
from mineru_eval.t2_deep_analysis.report import render_markdown


def _build_t2_parser() -> argparse.ArgumentParser:
    """Build argument parser for T2 deep analysis."""
    parser = argparse.ArgumentParser(description="T2: 9-Model Deep Analysis")
    parser.add_argument("--reports", default=str(ROOT / "benchmark" / "eval" / "reports"),
                        help="Directory containing *_report.json files")
    parser.add_argument("--output-dir", default=str(_t2_core.OUTPUT_DIR),
                        help="Output directory for charts and report")
    parser.add_argument("--no-charts", action="store_true", help="Skip chart generation")
    parser.add_argument("--report-only", action="store_true", help="Only generate the Markdown report")
    return parser


def _run_t2_analysis(args, output_dir, data):
    """Run extract_records, statistical tests, viz, and report."""
    print("\n[2/5] Extracting records...")
    all_scores, by_dim, by_diff, dimension_means = extract_records(data)
    for m in sorted(data.keys(), key=lambda x: all_scores[x].mean(), reverse=True):
        print(f"  {MODEL_SHORT[m]}: overall={all_scores[m].mean():.4f}, "
              f"n={len(all_scores[m])}")

    # 3. Statistical tests
    print("\n[3/5] Running statistical tests...")
    kw = kruskal_wallis_test(all_scores)
    if kw:
        print(f"  Kruskal-Wallis: H={kw['H']:.2f}, p={kw['p']:.6f} {_sig_stars(kw['p'])}")

    dunn_df = dunn_posthoc_test(all_scores)
    if dunn_df is not None:
        sig_pairs = 0
        for i in range(len(dunn_df)):
            for j in range(i + 1, len(dunn_df)):
                if dunn_df.iloc[i, j] < 0.05:
                    sig_pairs += 1
        print(f"  Dunn's post-hoc: {sig_pairs} significant pairs (Bonferroni corrected)")

    d_mat, names_d, models_d = cohens_d_matrix(all_scores)
    print(f"  Cohen's d (top vs #2): {d_mat[0, 1]:+.3f} ({_d_interpret(d_mat[0, 1])})")

    corr, names_corr = spearman_correlation(all_scores)
    mean_corr = np.mean(corr[np.triu_indices_from(corr, k=1)])
    print(f"  Mean Spearman ρ: {mean_corr:.4f}")

    per_dim_kw = per_dimension_kruskal(by_dim)
    for d in DIMENSIONS:
        if d in per_dim_kw:
            s = per_dim_kw[d]
            print(f"  K-W Dim {d}: H={s['H']:.1f}, p={s['p']:.4f} {_sig_stars(s['p'])}, "
                  f"best={s['best_model']}({s['best_mean']:.3f}), spread={s['spread']:.3f}")

    per_diff_kw = per_difficulty_kruskal(by_diff)
    for lvl in DIFFICULTIES:
        if lvl in per_diff_kw:
            s = per_diff_kw[lvl]
            print(f"  K-W {lvl}: H={s['H']:.1f}, p={s['p']:.4f} {_sig_stars(s['p'])}, "
                  f"best={s['best_model']}, spread={s['spread']:.3f}")

    # 4. Visualizations
    if not args.report_only:
        print("\n[4/5] Generating charts...")
        if not args.no_charts:
            fig_overall_bar(all_scores, dunn_df)
            fig_dimension_heatmap(dimension_means, data, per_dim_kw)
            fig_difficulty_boxplot(by_diff, data)
            fig_cohens_d_heatmap(d_mat, names_d, models_d)
            fig_correlation_clustermap(corr, names_corr)
            fig_per_difficulty_line(by_diff, data)
            fig_model_radar(data)
            print("  All 7 charts generated.")
        else:
            print("  Charts skipped (--no-charts).")
    else:
        print("\n[4/5] Charts skipped (--report-only).")

    # 5. Report
    print("\n[5/5] Generating Markdown report...")
    md = render_markdown(data, all_scores, by_dim, by_diff, dimension_means,
                          kw, dunn_df, d_mat, names_d, models_d,
                          corr, names_corr, per_dim_kw, per_diff_kw)
    report_path = output_dir / "t2_deep_analysis.md"
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path.write_text(md, encoding="utf-8")
    print(f"  Report: {report_path}")

    print(f"\n{'=' * 60}")
    print("T2 analysis complete.")
    print(f"Output: {output_dir}")
    print(f"{'=' * 60}")


def main():
    args = _build_t2_parser().parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _t2_core.OUTPUT_DIR = output_dir
    for _mod in (_t2_stats_mod, _t2_viz_mod, _t2_report_mod):
        _mod.OUTPUT_DIR = output_dir

    print("=" * 60)
    print("T2: 9-Model Deep Analysis")
    print("=" * 60)

    print("\n[1/5] Loading 9 model reports...")
    data = load_core_9(args.reports)
    if len(data) < 3:
        print(f"ERROR: Only {len(data)} models loaded, need ≥ 3.  Exiting.")
        return 1
    print(f"  Loaded {len(data)} models: {', '.join(MODEL_SHORT.get(m, m) for m in data)}")

    _run_t2_analysis(args, output_dir, data)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
