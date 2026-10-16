"""
Task 2d: 28K Chunk 全量嵌入分析 — 旧单块文件 → 薄兼容层 (deprecated)

内容已重构至 chunk_embedding_analysis/ 子包：
  core.py          — 常量 + 共享工具函数
  a1_pca.py        — A1: 语料全局语义空间 PCA
  a2_similarity.py — A2: 文档间相似度矩阵
  a3_redundancy.py — A3: Chunk 级冗余检测
  a4_train_eval.py — A4: Train/Eval 分布一致性
  __init__.py      — main 入口 + re-export

请改用:
  from mineru_eval.chunk_embedding_analysis import main
  python -m mineru_benchmark.eval.chunk_embedding_analysis --help
"""

import warnings

# Re-export everything from the new package for backwards compatibility.
from mineru_eval.chunk_embedding_analysis.core import (
    ROOT, PROJECT_ROOT, EMBED_MODEL, EMBED_DIM,
    BATCH_SIZE, DEFAULT_MANIFEST, DEFAULT_METADATA, DEFAULT_OUTPUT_DIR,
    get_embeddings, load_manifest, load_metadata, load_chunk_texts,
    ensure_output_dir, _configure_plot_style,
)
from mineru_eval.chunk_embedding_analysis.a1_pca import analysis_a1_pca
from mineru_eval.chunk_embedding_analysis.a2_similarity import analysis_a2_doc_similarity
from mineru_eval.chunk_embedding_analysis.a3_redundancy import analysis_a3_redundancy
from mineru_eval.chunk_embedding_analysis.a4_train_eval import analysis_a4_train_eval
from mineru_eval.chunk_embedding_analysis import main

warnings.warn(
    "chunk_embedding_analysis.py is deprecated. Use "
    "'from mineru_eval.chunk_embedding_analysis import ...' or "
    "'python -m mineru_benchmark.eval.chunk_embedding_analysis' instead.",
    DeprecationWarning,
    stacklevel=2,
)

if __name__ == "__main__":
    raise SystemExit(main())
