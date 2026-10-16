"""
T3: 深度分析扩展 — A6~A9  旧单块文件 → 薄兼容层

内容已重构至 t3_deep_analysis/ 子包：
  core.py     — 常量 + 数据加载
  a6_distribution.py  — A6 评分分布
  a7_dimension_kappa.py — A7 维度 κ 一致性
  a8_matrix.py   — A8 难度矩阵
  a9_bias.py     — A9 长度偏见
  report.py      — Markdown 报告生成
  __init__.py    — main 入口 + re-export
"""

# Re-export everything from the new package for backwards compatibility.
from mineru_eval.t3_deep_analysis.core import (
    ROOT, CORE_9, MODEL_SHORT, MODEL_FAMILY, FAMILY_COLORS,
    DIMENSIONS, DIFFICULTIES, RNG, OUTPUT_DIR,
    load_core_9, extract_records, _save_fig, _sig_stars,
)
from mineru_eval.t3_deep_analysis.a6_distribution import analysis_a6_distribution
from mineru_eval.t3_deep_analysis.a7_dimension_kappa import analysis_a7_dimension_kappa
from mineru_eval.t3_deep_analysis.a8_matrix import analysis_a8_difficulty_matrix
from mineru_eval.t3_deep_analysis.a9_bias import analysis_a9_length_bias
from mineru_eval.t3_deep_analysis.report import render_markdown
from mineru_eval.t3_deep_analysis import main

if __name__ == "__main__":
    raise SystemExit(main())
