"""chunk_embedding_analysis — 常量 + 共享工具函数"""

import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from mineru_tools.llm_client import LocalLLMClient

ROOT = PROJECT_ROOT = Path(__file__).resolve().parents[5]

# llama.cpp embedding server (OpenAI-compatible /v1/embeddings endpoint)
_llm_client = LocalLLMClient()
EMBED_MODEL = "qwen3-embedding-4b"
EMBED_DIM = 2560
BATCH_SIZE = 20
DEFAULT_MANIFEST = PROJECT_ROOT / "corpus" / "chunks" / "chunks_manifest.json"
DEFAULT_METADATA = PROJECT_ROOT / "corpus" / "metadata.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "benchmark" / "eval" / "results" / "chunk_embedding"


def get_embeddings(texts, progress_label="embed", cache_path=None):
    """Batch get embedding vectors via llama.cpp, with disk cache for resume.

    Returns (n, 2560) float32 ndarray.
    """
    if not texts:
        if cache_path and Path(cache_path).exists():
            print(f"  [{progress_label}] Loading from cache: {cache_path}", flush=True)
            cached = np.load(cache_path)
            print(f"  [{progress_label}] Cache hit {cached.shape[0]} items", flush=True)
            return cached
        return np.array([], dtype=np.float32)

    if cache_path and Path(cache_path).exists():
        print(f"  [{progress_label}] Loading from cache: {cache_path}", flush=True)
        cached = np.load(cache_path)
        if cached.shape[0] == len(texts):
            print(f"  [{progress_label}] Cache hit {cached.shape[0]} items", flush=True)
            return cached
        print(f"  [{progress_label}] Cache dimension mismatch ({cached.shape[0]} vs {len(texts)}), recomputing",
              flush=True)

    n = len(texts)
    all_emb = []
    t0 = time.time()
    for i in range(0, n, BATCH_SIZE):
        batch = texts[i: i + BATCH_SIZE]
        result = _llm_client.embed(batch, model=EMBED_MODEL)
        if not result.embeddings:
            raise RuntimeError(f"LLM returned empty embeddings for batch starting at {i}")
        all_emb.extend(result.embeddings)
        done = min(i + BATCH_SIZE, n)
        if done % 500 == 0 or done >= n:
            elapsed = time.time() - t0
            rate = done / elapsed if elapsed > 0 else 0
            print(f"  [{progress_label}] {done}/{n} ({done/n*100:.0f}%) — {rate:.1f} items/s",
                  flush=True)

    elapsed = time.time() - t0
    result = np.array(all_emb, dtype=np.float32)
    print(f"  [{progress_label}] Completed {n} items, total time {elapsed:.1f}s, avg {n/elapsed:.1f} items/s",
          flush=True)

    if cache_path:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, result)
        print(f"  [{progress_label}] Cache saved: {cache_path}", flush=True)

    return result


def load_manifest(manifest_path, max_chunks=None):
    """加载 chunk manifest，返回 (chunks_list, summary_dict)"""
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    chunks = manifest["chunks"]
    if max_chunks and max_chunks < len(chunks):
        chunks = chunks[:max_chunks]

    summary = {
        "total_chunks": manifest.get("total_chunks", len(chunks)),
        "train_chunks": manifest.get("train_chunks", 0),
        "eval_chunks": manifest.get("eval_chunks", 0),
        "target_tokens": manifest.get("target_tokens"),
        "min_tokens": manifest.get("min_tokens"),
        "max_tokens": manifest.get("max_tokens"),
    }
    return chunks, summary


def load_metadata(metadata_path):
    """加载 corpus metadata，返回 {filename: doc_info}"""
    with open(metadata_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    doc_map = {}
    for doc in meta.get("docs", []):
        doc_map[doc["filename"]] = doc
    return doc_map, {
        "total_docs": meta.get("total_docs"),
        "textbooks": meta.get("textbooks"),
        "arxiv_papers": meta.get("arxiv_papers"),
    }


def load_chunk_texts(chunks):
    """从 manifest chunk 条目加载所有文本。
    返回 (texts: list[str], ids: list[str], metas: list[dict])
    """
    texts, ids, metas = [], [], []
    skipped = 0
    for item in chunks:
        filepath = PROJECT_ROOT / item["filepath"]
        if not filepath.exists():
            skipped += 1
            continue
        text = filepath.read_text(encoding="utf-8").strip()
        if not text:
            skipped += 1
            continue
        texts.append(text)
        ids.append(item["chunk_id"])
        metas.append({
            "chunk_id": item["chunk_id"],
            "filename": item["filename"],
            "set": item["set"],
            "source_section": item.get("source_section", ""),
            "estimated_tokens": item.get("estimated_tokens", 0),
        })

    if skipped:
        print(f"  ⚠ 跳过 {skipped} 个不可读/空 chunk", flush=True)
    print(f"  加载 {len(texts)} 个有效 chunk", flush=True)
    return texts, ids, metas


def ensure_output_dir(output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _configure_plot_style():
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
