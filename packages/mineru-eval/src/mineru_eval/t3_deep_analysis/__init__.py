"""T3: 深度分析扩展 — A6~A9
A6: 评分分布 (Score Distribution) — 分布形状/偏度/峰度/双峰检测
A7: 维度 κ (Dimension Agreement) — Kendall's W / ICC / 维度间一致性矩阵
A8: 难度矩阵 (Difficulty Matrix) — 模型×难度热力图/题目难度分析/衰退模式
A9: 长度偏见 (Length Bias) — model_answer 长度与评分相关性

Produce:
  - 10 PNG charts in results/analysis/
  - 4 JSON result files
  - Comprehensive Markdown section in t3_deep_analysis.md
"""

from __future__ import annotations

from pathlib import Path

from mineru_eval.t3_deep_analysis import (
    core as _t3_core,
    a6_distribution as _a6_mod,
    a7_dimension_kappa as _a7_mod,
    a8_matrix as _a8_mod,
    a9_bias as _a9_mod,
)
from mineru_eval.t3_deep_analysis.core import (
    ROOT, CORE_9, MODEL_SHORT, MODEL_FAMILY, FAMILY_COLORS,
    DIMENSIONS, DIFFICULTIES, RNG, OUTPUT_DIR,
    load_core_9, extract_records,
    _save_fig, _sig_stars,
)
from mineru_eval.t3_deep_analysis.a6_distribution import analysis_a6_distribution
from mineru_eval.t3_deep_analysis.a7_dimension_kappa import analysis_a7_dimension_kappa
from mineru_eval.t3_deep_analysis.a8_matrix import analysis_a8_difficulty_matrix
from mineru_eval.t3_deep_analysis.a9_bias import analysis_a9_length_bias
from mineru_eval.t3_deep_analysis.report import render_markdown


def main():
    import argparse
    parser = argparse.ArgumentParser(description="T3: Deep Analysis Extension — A6~A9")
    parser.add_argument("--reports", default=str(ROOT / "benchmark" / "eval" / "reports"),
                        help="Directory containing *_report.json files")
    parser.add_argument("--output-dir", default=str(_t3_core.OUTPUT_DIR),
                        help="Output directory for charts and report")
    parser.add_argument("--skip", nargs="*", choices=["A6", "A7", "A8", "A9"], default=[],
                        help="Skip specific analyses")
    args = parser.parse_args()

    # Allow CLI override of OUTPUT_DIR (must propagate to all submodules)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _t3_core.OUTPUT_DIR = output_dir
    for _mod in (_a6_mod, _a7_mod, _a8_mod, _a9_mod):
        _mod.OUTPUT_DIR = output_dir

    print("=" * 60)
    print("T3: 深度分析扩展 — A6~A9")
    print("=" * 60)

    # 1. Load data
    print("\n[1/4] Loading 9 model reports...")
    data = load_core_9(args.reports)
    if len(data) < 3:
        print(f"ERROR: Only {len(data)} models loaded, need ≥ 3. Exiting.")
        return 1
    print(f"  Loaded {len(data)} models: {', '.join(MODEL_SHORT.get(m, m) for m in data)}")

    # 2. Extract records
    print("\n[2/4] Extracting records...")
    all_scores, all_lengths, all_records, by_dim, by_diff, length_by_dim, dimension_means = \
        extract_records(data)

    results = {}

    # 3. A6-A9 analyses
    print("\n[3/4] Running analyses A6~A9...")

    if "A6" not in args.skip:
        print("\n" + "-" * 40)
        results["A6"] = analysis_a6_distribution(all_scores)

    if "A7" not in args.skip:
        print("\n" + "-" * 40)
        results["A7"] = analysis_a7_dimension_kappa(all_scores, by_dim, data)

    if "A8" not in args.skip:
        print("\n" + "-" * 40)
        results["A8"] = analysis_a8_difficulty_matrix(all_scores, by_diff, by_dim, data)

    if "A9" not in args.skip:
        print("\n" + "-" * 40)
        results["A9"] = analysis_a9_length_bias(all_scores, all_lengths, by_dim, data)

    # 4. Report
    print("\n[4/4] Generating Markdown report...")
    md = render_markdown(data,
                         results.get("A6", {"model_stats": []}),
                         results.get("A7", {"kendall_W": 0, "friedman_chi2": 0, "friedman_p": 1,
                                             "icc_2_1": 0, "dimension_pairwise_spearman": {}}),
                         results.get("A8", {"difficulty_matrix": {"models": [], "difficulty_levels": [],
                                                                    "values": []},
                                             "item_difficulty_bins": {}, "mean_item_difficulty": 0,
                                             "decay_details": [], "most_stable": {}, "most_sensitive": {}}),
                         results.get("A9", {"global_pearson_r": 0, "global_pearson_p": 1,
                                             "global_spearman_rho": 0, "global_spearman_p": 1,
                                             "per_model": [], "dimension_corrs": {}}))
    report_path = output_dir / "t3_deep_analysis.md"
    report_path.write_text(md, encoding="utf-8")
    print(f"  Report: {report_path}")

    print(f"\n{'=' * 60}")
    print("T3: A6~A9 analysis complete.")
    lists = [f"benchmark/eval/results/analysis/a{i}_{name}.png"
             for i, name in [("6", "density_ridges"), ("6", "skewness_vs_mean"), ("6", "ecdf"),
                             ("7", "dimension_agreement"), ("7", "concordance_radar"),
                             ("8", "difficulty_matrix"), ("8", "decay_curves"), ("8", "item_difficulty_hist"),
                             ("9", "length_vs_score"), ("9", "length_distribution"), ("9", "length_ratio")]]
    print(f"Charts: {len(lists)} PNGs + 4 JSONs + 1 Markdown")
    print(f"Output: {output_dir}")
    print(f"{'=' * 60}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
