"""
T2: 9-Model Deep Analysis — 旧单块文件 → 薄兼容层 (deprecated)

内容已重构至 t2_deep_analysis/ 子包：
  core.py      — 常量 + 数据加载
  stats.py     — 统计检验
  visualize.py — 7 张图表
  report.py    — Markdown 报告生成
  __init__.py  — main 入口 + re-export

请改用:
  from mineru_eval.t2_deep_analysis import main
  python -m mineru_benchmark.eval.t2_deep_analysis --help
"""

import warnings

# Re-export everything from the new package for backwards compatibility.
from mineru_eval.t2_deep_analysis.core import (
    ROOT, CORE_9, MODEL_SHORT, MODEL_FAMILY, FAMILY_COLORS,
    DIMENSIONS, DIFFICULTIES, RNG, OUTPUT_DIR,
    load_core_9, extract_records, _save_fig, _sig_stars,
    _d_interpret, _model_rank_desc,
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
from mineru_eval.t2_deep_analysis import main

warnings.warn(
    "t2_deep_analysis.py is deprecated. Use "
    "'from mineru_eval.t2_deep_analysis import ...' or "
    "'python -m mineru_benchmark.eval.t2_deep_analysis' instead.",
    DeprecationWarning,
    stacklevel=2,
)

if __name__ == "__main__":
    raise SystemExit(main())
