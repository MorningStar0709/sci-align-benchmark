"""Local embedding providers for medical RAG indexes."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np


LOCAL_DEFAULT_MODEL = "qwen3-embedding-4b"


def provider_slug(provider: str, model: str) -> str:
    text = f"{provider}_{model}".lower()
    for ch in [":", "/", "\\", " ", "."]:
        text = text.replace(ch, "_")
    while "__" in text:
        text = text.replace("__", "_")
    return text.strip("_")


def _cache_meta_path(cache_path: str | Path) -> Path:
    p = Path(cache_path)
    return p.with_name(p.stem + ".meta.json")


def _write_cache_meta(cache_path: str | Path, dim: int) -> None:
    meta_path = _cache_meta_path(cache_path)
    meta_path.write_text(json.dumps({"dim": dim}, ensure_ascii=False), encoding="utf-8")


def embed_texts(
    texts: list[str],
    provider: str,
    model: str,
    *,
    batch_size: int = 16,
    cache_path: str | Path | None = None,
    progress_label: str = "embed",
) -> np.ndarray:
    provider = (provider or "local").strip().lower()
    # Backward compat: "ollama" → "local"
    if provider == "ollama":
        provider = "local"
    model = (model or LOCAL_DEFAULT_MODEL).strip()

    if cache_path and Path(cache_path).exists():
        cached = np.load(cache_path)
        if cached.shape[0] == len(texts):
            # 验证嵌入维度（通过 sidecar 元数据）
            meta_path = _cache_meta_path(cache_path)
            if meta_path.exists():
                try:
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                    expected_dim = meta.get("dim")
                    if expected_dim is not None and cached.shape[1] != expected_dim:
                        print(f"  [{progress_label}] 缓存维度 {cached.shape[1]} ≠ 期望 {expected_dim}，重新计算", flush=True)
                        del cached
                    else:
                        print(f"  [{progress_label}] 缓存命中: {cache_path} {cached.shape}", flush=True)
                        return cached.astype(np.float32, copy=False)
                except (json.JSONDecodeError, OSError):
                    pass
            else:
                # 没有 sidecar 元数据时，按文件名判断（已含 provider+model slug）
                print(f"  [{progress_label}] 缓存命中: {cache_path} {cached.shape}", flush=True)
                return cached.astype(np.float32, copy=False)
        print(f"  [{progress_label}] 缓存条数不匹配，重新计算: {cached.shape[0]} vs {len(texts)}", flush=True)

    if provider == "local":
        embeddings = _embed_local(texts, model, batch_size=batch_size, progress_label=progress_label)
    elif provider in {"hf", "hf_local", "transformers"}:
        embeddings = _embed_hf(texts, model, batch_size=batch_size, progress_label=progress_label)
    else:
        raise ValueError(f"Unsupported embedding provider: {provider}")

    if cache_path:
        cache_path = Path(cache_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, embeddings)
        _write_cache_meta(cache_path, int(embeddings.shape[1]))
        print(f"  [{progress_label}] 缓存已保存: {cache_path} (dim={embeddings.shape[1]})", flush=True)
    return embeddings


def _embed_local(texts: list[str], model: str, *, batch_size: int, progress_label: str) -> np.ndarray:
    """Embed texts via llama.cpp embedding server."""
    from mineru_tools.llm_client import LocalLLMClient

    client = LocalLLMClient(timeout=180.0)
    all_emb = []
    t0 = time.time()
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        result = client.embed(batch, model=model)
        all_emb.extend(result.embeddings)
        _print_progress(progress_label, min(i + batch_size, len(texts)), len(texts), t0)
    return np.asarray(all_emb, dtype=np.float32)


def _embed_hf(texts: list[str], model: str, *, batch_size: int, progress_label: str) -> np.ndarray:
    import torch
    import torch.nn.functional as F
    from transformers import AutoModel, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True, trust_remote_code=True)
    net = AutoModel.from_pretrained(model, local_files_only=True, trust_remote_code=True)
    net.to(device)
    net.eval()

    all_emb = []
    t0 = time.time()
    with torch.inference_mode():
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            inputs = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=8192 if "bge-m3" in model.lower() else 512,
                return_tensors="pt",
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            output = net(**inputs)
            emb = output.last_hidden_state[:, 0]
            emb = F.normalize(emb, p=2, dim=1)
            all_emb.append(emb.cpu().numpy().astype(np.float32))
            _print_progress(progress_label, min(i + batch_size, len(texts)), len(texts), t0)
    return np.concatenate(all_emb, axis=0)


def _print_progress(label: str, done: int, total: int, start: float) -> None:
    if done >= total or done % 250 == 0:
        elapsed = time.time() - start
        rate = done / elapsed if elapsed > 0 else 0.0
        print(f"  [{label}] {done}/{total} ({done / total * 100:.0f}%) - {rate:.1f}条/s", flush=True)


def write_index_metadata(path: str | Path, metadata: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


def read_index_metadata(index_dir: str | Path) -> dict:
    path = Path(index_dir) / "index_meta.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))
