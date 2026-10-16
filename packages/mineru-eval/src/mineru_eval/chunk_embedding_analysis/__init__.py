"""
chunk_embedding_analysis — 28K Chunk 全量嵌入分析

重构子包：
  core.py          — 常量 + 共享工具函数
  a1_pca.py        — A1: 语料全局语义空间 PCA
  a2_similarity.py — A2: 文档间相似度矩阵
  a3_redundancy.py — A3: Chunk 级冗余检测
  a4_train_eval.py — A4: Train/Eval 分布一致性
"""

from __future__ import annotations

import sys
from pathlib import Path

from mineru_eval.chunk_embedding_analysis import (
    core as _core_mod,
    a1_pca as _a1_mod,
    a2_similarity as _a2_mod,
    a3_redundancy as _a3_mod,
    a4_train_eval as _a4_mod,
)

# ── Re-export all public symbols ──────────────────────────────────
from mineru_eval.chunk_embedding_analysis.core import (
    ROOT,
    PROJECT_ROOT,
    EMBED_MODEL,
    EMBED_DIM,
    BATCH_SIZE,
    DEFAULT_MANIFEST,
    DEFAULT_METADATA,
    DEFAULT_OUTPUT_DIR,
    get_embeddings,
    load_manifest,
    load_metadata,
    load_chunk_texts,
    ensure_output_dir,
    _configure_plot_style,
)

from mineru_eval.chunk_embedding_analysis.a1_pca import analysis_a1_pca
from mineru_eval.chunk_embedding_analysis.a2_similarity import analysis_a2_doc_similarity
from mineru_eval.chunk_embedding_analysis.a3_redundancy import analysis_a3_redundancy
from mineru_eval.chunk_embedding_analysis.a4_train_eval import analysis_a4_train_eval


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Task 2d: 28K Chunk 全量嵌入分析")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST), help="chunks_manifest.json 路径")
    parser.add_argument("--metadata", default=str(DEFAULT_METADATA), help="corpus metadata.json 路径")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="输出目录")
    parser.add_argument("--max-chunks", type=int, default=0, help="限制加载 chunk 数量 (0=全量, >0=快速测试)")
    parser.add_argument("--skip", nargs="*", choices=["A1", "A2", "A3", "A4"], default=[], help="跳过的分析")
    parser.add_argument("--skip-embed", action="store_true", help="跳过嵌入计算，使用缓存")
    parser.add_argument("--cos-threshold", type=float, default=0.95, help="A3 冗余检测 cosine 阈值")
    args = parser.parse_args()

    output_dir = ensure_output_dir(args.output_dir)
    max_chunks = args.max_chunks if args.max_chunks > 0 else None

    print("=" * 60)
    print("Task 2d: 28K Chunk 全量嵌入分析")
    print("=" * 60)

    manifest_path = Path(args.manifest)
    metadata_path = Path(args.metadata)

    if not manifest_path.exists():
        print(f"✗ manifest 不存在: {manifest_path}")
        sys.exit(1)
    if not metadata_path.exists():
        print(f"✗ metadata 不存在: {metadata_path}")
        sys.exit(1)

    print("\n配置:")
    print(f"  manifest: {manifest_path}")
    print(f"  metadata: {metadata_path}")
    print(f"  output: {output_dir}")
    print(f"  max_chunks: {max_chunks or '全量'}")
    print(f"  skip: {args.skip or '无'}")
    print(f"  skip_embed: {args.skip_embed}")
    print(f"  cos_threshold: {args.cos_threshold}")

    import numpy as np

    chunks, manifest_summary = load_manifest(manifest_path, max_chunks)
    doc_map, meta_summary = load_metadata(metadata_path)
    print(f"\nManifest: {manifest_summary['total_chunks']} chunks (train={manifest_summary['train_chunks']}, eval={manifest_summary['eval_chunks']})")
    print(f"Metadata: {meta_summary['total_docs']} docs (textbooks={meta_summary['textbooks']}, arxiv={meta_summary['arxiv_papers']})")

    texts, _, chunk_metas = load_chunk_texts(chunks)
    if not texts:
        print("✗ 无有效 chunk 文本可加载，请检查 corpus/chunks/ 目录")
        sys.exit(1)

    cache_path = output_dir / "embeddings_cache.npy"
    if args.skip_embed:
        embeddings = get_embeddings([], progress_label="embed", cache_path=str(cache_path))
        if embeddings.size == 0:
            print("✗ --skip-embed 但缓存不可用，请先完成初始嵌入")
            sys.exit(1)
    else:
        print("\n检查 llama.cpp embedding 连通性...", flush=True)
        try:
            from mineru_tools.llm_client import LocalLLMClient
            _check_client = LocalLLMClient()
            result = _check_client.embed(["test"])
            emb = result.embeddings[0] if result.embeddings else []
            if len(emb) != EMBED_DIM:
                print(f"  ✗ 嵌入维度不匹配: 期望 {EMBED_DIM}, 实际 {len(emb)}", flush=True)
                sys.exit(1)
            print(f"  ✓ llama.cpp embedding 在线, 嵌入维度: {len(emb)}, 模型: {EMBED_MODEL}", flush=True)
        except Exception as e:
            print(f"  ✗ llama.cpp embedding 连接失败: {e}", flush=True)
            print("  请确保 llama.cpp embedding 服务运行中 (port 8081)", flush=True)
            sys.exit(1)

        print(f"\n开始嵌入计算 ({len(texts)} chunks)...", flush=True)
        embeddings = get_embeddings(texts, progress_label="28K embed", cache_path=str(cache_path))

    print(f"\n嵌入矩阵: {embeddings.shape}", flush=True)

    results = {}

    if "A1" not in args.skip:
        results["A1"] = analysis_a1_pca(embeddings, chunk_metas, doc_map, output_dir)

    if "A2" not in args.skip:
        results["A2"] = analysis_a2_doc_similarity(embeddings, chunk_metas, doc_map, output_dir)

    if "A3" not in args.skip:
        results["A3"] = analysis_a3_redundancy(embeddings, chunk_metas, output_dir, args.cos_threshold)

    if "A4" not in args.skip:
        results["A4"] = analysis_a4_train_eval(embeddings, chunk_metas, output_dir)

    print("\n" + "=" * 60)
    print("Task 2d 全部完成!")
    print("=" * 60)
    print(f"产出目录: {output_dir}")
    for f in sorted(output_dir.glob("*")):
        size_kb = f.stat().st_size / 1024
        print(f"  {f.name} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
