"""全量数据集 Embedding 扫描 — 六组文本提取 + BGE-M3 编码落盘（幂等续跑）。

计划依据: docs/plans/dataset-embedding-scan-plan.md
- 文本提取口径（§二 冻结）: 题库/医疗题库取 question 字段; 训练数据取 messages 中
  role=user 的 content（全部 user 消息以换行拼接，实测每行恰好 1 条）
- 不加 instruct 前缀; normalize_embeddings=True; fp16; batch 32（OOM 降 8）
- embedder 名称/维度/判据阈值原文强制写入 meta，供日后更强 embedder（如
  Qwen3-Embedding-4B）复跑作 robustness 对照（§一 关键约束）

用法:
    .venv/bin/python scripts/embed_dataset.py --dry-run      # 只打印六组计数
    .venv/bin/python scripts/embed_dataset.py                # 全量编码（幂等续跑）
    .venv/bin/python scripts/embed_dataset.py --only core_v3 # 补跑单组
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from typing import TYPE_CHECKING

if TYPE_CHECKING:    # 注解用，运行期由函数内 import 提供
    import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "benchmark" / "eval" / "results" / "question_embedding"

# ── 冻结常量（计划 §一 判据表 + §二 口径；开跑后禁止回调）──
MODEL_NAME = "BAAI/bge-m3"
EMBED_DIM = 1024
BATCH_SIZE = 32
BATCH_SIZE_OOM = 8
MAX_TOKENS = 8192
NORMALIZE_EMBEDDINGS = True
PRECISION = "float16"
L1_ALERT = 0.90
L2_REDLINE = 0.95
R1_REDUNDANCY = 0.95
L2_REDLINE_RATIO = 0.01


@dataclass(frozen=True)
class TextSource:
    name: str
    path: str
    kind: str  # "questions" | "messages"
    extract_rule: str


SOURCES = [
    TextSource("core_v3", "benchmark/dataset/core_v3.json", "questions",
               "questions[].question，id=questions[].id"),
    TextSource("medical_benchmark", "benchmark/dataset/medical_benchmark.json", "questions",
               "questions[].question，id=questions[].id"),
    TextSource("train_v3", "data/finetune/control_sci/train_v3.jsonl", "messages",
               "messages 中 role=user 的 content（多条以换行拼接），id=行号(1-based)"),
    TextSource("cot_train_final", "data/finetune/control_sci/cot_train_final.jsonl", "messages",
               "messages 中 role=user 的 content（多条以换行拼接），id=行号(1-based)"),
    TextSource("medical_train", "data/finetune/medical/train.jsonl", "messages",
               "messages 中 role=user 的 content（多条以换行拼接），id=行号(1-based)"),
    TextSource("sciverse_train", "data/finetune/sciverse/train.jsonl", "messages",
               "messages 中 role=user 的 content（多条以换行拼接），id=行号(1-based)"),
]


def extract(source: TextSource) -> tuple[list[str], list[str]]:
    """提取 (texts, ids)。口径冻结于 SOURCES.extract_rule，不入模型。"""
    path = PROJECT_ROOT / source.path
    if source.kind == "questions":
        data = json.loads(path.read_text(encoding="utf-8"))
        questions = data["questions"] if isinstance(data, dict) else data
        return [q["question"] for q in questions], [str(q["id"]) for q in questions]
    texts: list[str] = []
    ids: list[str] = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        row = json.loads(line)
        user_parts = [m["content"] for m in row["messages"] if m.get("role") == "user"]
        if not user_parts:
            raise ValueError(f"{source.name} 行 {i} 无 role=user 消息，提取口径失效")
        texts.append("\n".join(user_parts))
        ids.append(str(i))
    return texts, ids


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_write(path: Path, write_fn) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    write_fn(tmp)
    os.replace(tmp, path)


def already_done(source: TextSource) -> tuple[bool, str]:
    npy = OUTPUT_DIR / f"embeddings_{source.name}.npy"
    meta = OUTPUT_DIR / f"{source.name}.meta.json"
    if not npy.exists() or not meta.exists():
        return False, "产物缺失"
    m = json.loads(meta.read_text(encoding="utf-8"))
    n_texts = len(extract(source)[0])
    if m.get("n") != n_texts:
        return False, f"meta n={m.get('n')} != 当前提取 {n_texts}"
    return True, "meta n 匹配"


def embed_texts(model, texts: list[str], batch_size: int) -> "np.ndarray":
    import numpy as np

    embs = []
    for i in range(0, len(texts), batch_size):
        embs.append(model.encode(
            texts[i : i + batch_size],
            normalize_embeddings=NORMALIZE_EMBEDDINGS,
            precision=PRECISION,
        ))
    return np.vstack(embs)


def count_truncated(model, texts: list[str], max_tokens: int) -> int:
    """token 数 > max_tokens 的条数（encode 内部自动截断，此处仅统计声明）。"""
    lengths = model.tokenizer(texts, truncation=False, return_length=True)["length"]
    return int(sum(1 for length in lengths if length > max_tokens))


def embed_source(source: TextSource) -> None:
    """单组编码落盘（幂等：meta n 匹配则跳过）。"""
    texts, ids = extract(source)
    done, reason = already_done(source)
    if done:
        print(f"[SKIP] {source.name}: {len(texts)} texts（{reason}）")
        return
    import numpy as np
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(MODEL_NAME)
    n_truncated = count_truncated(model, texts, MAX_TOKENS)
    try:
        emb = embed_texts(model, texts, BATCH_SIZE)
    except RuntimeError as exc:
        if "out of memory" not in str(exc).lower():
            raise
        print(f"[WARN] {source.name}: batch {BATCH_SIZE} OOM，降为 {BATCH_SIZE_OOM} 重试")
        emb = embed_texts(model, texts, BATCH_SIZE_OOM)
    if emb.shape != (len(texts), EMBED_DIM):
        raise RuntimeError(f"{source.name}: shape {emb.shape} 与期望 {(len(texts), EMBED_DIM)} 不符")
    if np.isnan(emb).any():
        raise RuntimeError(f"{source.name}: 嵌入含 NaN")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    atomic_write(OUTPUT_DIR / f"embeddings_{source.name}.npy", lambda p: np.save(p, emb))
    meta = {
        "model": MODEL_NAME,
        "dim": EMBED_DIM,
        "n": len(texts),
        "ids": ids,
        "source_path": source.path,
        "source_sha256": file_sha256(PROJECT_ROOT / source.path),
        "extract_rule": source.extract_rule,
        "truncated_n": n_truncated,
        "device": str(model.device),
        "judgment": {"L1_alert": L1_ALERT, "L2_redline": L2_REDLINE,
                     "L2_redline_ratio": L2_REDLINE_RATIO, "R1_redundancy": R1_REDUNDANCY},
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    atomic_write(OUTPUT_DIR / f"{source.name}.meta.json",
                 lambda p: p.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"))
    print(f"[DONE] {source.name}: {len(texts)} texts -> embeddings_{source.name}.npy"
          f"（truncated_n={n_truncated}）")


def main() -> int:
    ap = argparse.ArgumentParser(description="六组文本提取 + BGE-M3 编码落盘（幂等续跑）")
    ap.add_argument("--dry-run", action="store_true", help="只打印六组文本计数，不加载模型")
    ap.add_argument("--only", choices=[s.name for s in SOURCES], help="只处理单组（断点续跑）")
    args = ap.parse_args()

    selected = [s for s in SOURCES if args.only is None or s.name == args.only]
    for source in selected:
        texts, ids = extract(source)
        if len(texts) != len(ids):
            raise RuntimeError(f"{source.name}: texts/ids 数量不一致")
        if args.dry_run:
            print(f"{source.name}: {len(texts)} texts / {len(ids)} ids | {source.path}")
            continue
        embed_source(source)
    return 0


if __name__ == "__main__":
    sys.exit(main())
