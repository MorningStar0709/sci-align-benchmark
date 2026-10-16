"""Hybrid 医疗文献检索索引: FAISS Dense + BM25 Sparse → RRF 合并

前置条件:
  - llama.cpp qwen3-embedding-4b 运行中 (http://localhost:8081)
  - chunks manifest 就绪 (data/sources_medical/chunks/chunks_manifest.json)
  - pip install faiss-cpu rank-bm25

约定路径:
  - manifest: data/sources_medical/chunks/chunks_manifest.json
  - FAISS:    data/sources_medical/index/medical.index
  - BM25:     data/sources_medical/index/bm25.pkl
  - 嵌入缓存:  data/sources_medical/index/embeddings_cache.npy
"""

import argparse
import hashlib
import json
import pickle
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from mineru_core.paths import PROJECT_ROOT
from mineru_medical.embedding_providers import (
    LOCAL_DEFAULT_MODEL,
    embed_texts,
    provider_slug,
    read_index_metadata,
    write_index_metadata,
)


def _lazy_import_faiss():
    try:
        import faiss
        return faiss
    except ImportError:
        print("缺少 faiss-cpu，请执行: .venv/bin/python -m pip install faiss-cpu")
        sys.exit(1)


def _lazy_import_bm25():
    try:
        from rank_bm25 import BM25Okapi
        return BM25Okapi
    except ImportError:
        print("缺少 rank-bm25，请执行: .venv/bin/python -m pip install rank-bm25")
        sys.exit(1)


def load_manifest(manifest_path):
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    if not manifest.get("chunks"):
        raise ValueError(f"manifest 未包含 chunks 字段: {manifest_path}")
    return manifest


def _load_chunk_texts(chunks):
    texts = []
    for c in chunks:
        fpath = PROJECT_ROOT / c.get("filepath", "")
        if fpath.exists():
            texts.append(fpath.read_text(encoding="utf-8").strip())
        else:
            texts.append("")
    return texts


def _file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def build_dense_index(texts, output_path, cache_path=None, embedding_provider="local", embedding_model=LOCAL_DEFAULT_MODEL, batch_size=16):
    faiss = _lazy_import_faiss()

    print(f"[Dense] 获取 {len(texts)} 条嵌入向量: {embedding_provider}:{embedding_model}", flush=True)
    t0 = time.time()
    embeddings = embed_texts(
        texts,
        embedding_provider,
        embedding_model,
        batch_size=batch_size,
        progress_label="medical_embed",
        cache_path=cache_path,
    )
    elapsed = time.time() - t0
    print(f"[Dense] 嵌入完成: {embeddings.shape}, 耗时 {elapsed:.1f}s", flush=True)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    embeddings_norm = embeddings.copy()
    faiss.normalize_L2(embeddings_norm)
    index.add(embeddings_norm)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(output_path))
    print(f"[Dense] FAISS 索引已写入: {output_path} ({index.ntotal} 条, dim={dim})", flush=True)
    return index, embeddings


def build_sparse_index(texts, output_path):
    BM25Okapi = _lazy_import_bm25()

    if not texts or all(not t for t in texts):
        raise ValueError("build_sparse_index: texts 为空或全为空，无法构建 BM25 索引")

    print(f"[Sparse] 构建 BM25 索引 ({len(texts)} 条)...", flush=True)
    t0 = time.time()
    tokenized = [text.lower().split() for text in texts]
    bm25 = BM25Okapi(tokenized)
    elapsed = time.time() - t0
    print(f"[Sparse] BM25 构建完成, 耗时 {elapsed:.1f}s", flush=True)

    data = {"bm25": bm25, "tokenized": tokenized, "texts": texts}
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(data, f)
    print(f"[Sparse] BM25 索引已写入: {output_path}", flush=True)
    return bm25, tokenized


def build_hybrid_index(
    manifest_path,
    output_dir,
    cache_path=None,
    embedding_provider="local",
    embedding_model=LOCAL_DEFAULT_MODEL,
    batch_size=16,
    max_chunks=0,
):
    manifest = load_manifest(manifest_path)
    chunks = manifest["chunks"]
    if max_chunks and max_chunks > 0:
        chunks = chunks[:max_chunks]
    texts = _load_chunk_texts(chunks)

    if not texts or all(not t.strip() for t in texts):
        raise ValueError("manifest 中无可用的 chunk text")

    print(f"[Hybrid] 开始构建: {len(chunks)} chunks → FAISS + BM25", flush=True)

    dense_path = Path(output_dir) / "medical.index"
    sparse_path = Path(output_dir) / "bm25.pkl"

    _, embeddings = build_dense_index(
        texts,
        dense_path,
        cache_path=cache_path,
        embedding_provider=embedding_provider,
        embedding_model=embedding_model,
        batch_size=batch_size,
    )
    build_sparse_index(texts, sparse_path)
    meta = {
        "index_id": Path(output_dir).name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "embedding_provider": embedding_provider,
        "embedding_model": embedding_model,
        "embedding_dim": int(embeddings.shape[1]),
        "chunk_count": len(chunks),
        "manifest_path": str(Path(manifest_path)),
        "manifest_sha256": _file_sha256(manifest_path),
        "dense_index": str(dense_path),
        "embedding_cache": str(cache_path) if cache_path else "",
        "bm25_index": str(sparse_path),
    }
    write_index_metadata(Path(output_dir) / "index_meta.json", meta)

    print("[Hybrid] 索引构建完成 OK", flush=True)


def search_hybrid(query, index_dir, manifest, k=5, rrf_k=60, texts_override=None):
    faiss = _lazy_import_faiss()
    import numpy as np

    index_dir = Path(index_dir)
    dense_path = index_dir / "medical.index"
    sparse_path = index_dir / "bm25.pkl"
    meta = read_index_metadata(index_dir)
    embedding_provider = meta.get("embedding_provider", "local")
    embedding_model = meta.get("embedding_model", LOCAL_DEFAULT_MODEL)

    if not dense_path.exists() or not sparse_path.exists():
        raise FileNotFoundError(f"索引文件缺失: dense={dense_path.exists()}, sparse={sparse_path.exists()}")

    index = faiss.read_index(str(dense_path))
    q_emb = embed_texts([query], embedding_provider, embedding_model, batch_size=1, progress_label="query_embed")
    q_emb_norm = q_emb.copy()
    faiss.normalize_L2(q_emb_norm)
    dense_scores, dense_indices = index.search(q_emb_norm, k * 2)

    with open(sparse_path, "rb") as f:
        bm25_data = pickle.load(f)
    tokenized_query = query.lower().split()
    sparse_scores = np.array(bm25_data["bm25"].get_scores(tokenized_query))
    sparse_top_k = np.argsort(sparse_scores)[-k * 2 :][::-1]

    rrf_scores = {}
    for rank, idx in enumerate(dense_indices[0]):
        rrf_scores[int(idx)] = rrf_scores.get(int(idx), 0) + 1.0 / (rrf_k + rank + 1)
    for rank, idx in enumerate(sparse_top_k):
        rrf_scores[int(idx)] = rrf_scores.get(int(idx), 0) + 1.0 / (rrf_k + rank + 1)

    ranked = sorted(rrf_scores.items(), key=lambda x: -x[1])[:k]
    results = []
    if texts_override is not None:
        texts = texts_override
    elif manifest["chunks"] and "filepath" in manifest["chunks"][0]:
        texts = _load_chunk_texts(manifest["chunks"])
    else:
        texts = []
    for idx, score in ranked:
        if idx < len(manifest["chunks"]):
            chunk = dict(manifest["chunks"][idx])
            chunk["_rrf_score"] = round(score, 6)
            if texts:
                chunk["text"] = texts[idx]
            results.append(chunk)
    return results


def main():
    parser = argparse.ArgumentParser(description="构建 Hybrid 医疗文献检索索引 (FAISS + BM25 + RRF)")
    parser.add_argument(
        "--manifest",
        default=str(PROJECT_ROOT / "data" / "sources_medical" / "chunks" / "chunks_manifest.json"),
        help="chunks manifest JSON 路径",
    )
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "data" / "sources_medical" / "index"),
        help="索引输出目录",
    )
    parser.add_argument(
        "--embed-cache",
        default=None,
        help="嵌入缓存文件路径",
    )
    parser.add_argument("--embedding-provider", default="local", choices=["local", "ollama", "hf", "hf_local", "transformers"], help="嵌入 provider")
    parser.add_argument("--embedding-model", default=LOCAL_DEFAULT_MODEL, help="嵌入模型名或本地 HuggingFace 路径")
    parser.add_argument("--batch-size", type=int, default=16, help="嵌入 batch size")
    parser.add_argument("--max-chunks", type=int, default=0, help="Limit indexed chunks for smoke tests (0 = all)")
    parser.add_argument(
        "--query",
        default=None,
        help="构建完成后执行一次测试检索（可选）",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="检索返回 top-k (默认 5)",
    )
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        print(f"manifest 不存在: {manifest_path}")
        print("请先完成任务 2: 医疗章节本体切片")
        sys.exit(1)

    cache_path = args.embed_cache
    if cache_path is None:
        slug = provider_slug(args.embedding_provider, args.embedding_model)
        cache_path = str(Path(args.output) / f"embeddings_{slug}.npy")

    build_hybrid_index(
        manifest_path,
        args.output,
        cache_path=cache_path,
        embedding_provider=args.embedding_provider,
        embedding_model=args.embedding_model,
        batch_size=args.batch_size,
        max_chunks=args.max_chunks,
    )

    if args.query:
        print(f"\n[测试检索] query={args.query!r}, k={args.k}", flush=True)
        manifest = load_manifest(manifest_path)
        results = search_hybrid(args.query, args.output, manifest, k=args.k)
        for i, r in enumerate(results):
            text_preview = r.get("text", "")[:120].replace("\n", " ")
            print(f"  #{i+1} rrf={r.get('_rrf_score')} | label={r.get('medical_label','?')} | {text_preview}...")
    else:
        print("[跳过] 未指定 --query，不执行测试检索")


if __name__ == "__main__":
    main()
