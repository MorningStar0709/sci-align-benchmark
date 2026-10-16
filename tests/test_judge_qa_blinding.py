"""T12 盲化回归（重新锚定版）

原卡锚点 `EXCLUDED_SOURCE_MARKERS` / `scan_candidates` 在现版 `humaneval/pipeline/build_pool.py`
（analysis.json → pool.json 的渲染视图构建器）中已不存在，故回归改为直接锁人类池的两条不变量：
1. 问卷池文件与其 display_sha sidecar 的校验和不变（机制验证不得震动人类池）
2. `judge_qa/` 的题号与文本不得出现在问卷池里
"""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
POOL = ROOT / "humaneval" / "pool" / "pool.json"
POOL_SHA = ROOT / "humaneval" / "pool" / "pool.json.sha256"
JUDGE_QA = ROOT / "benchmark" / "eval" / "judge_qa"

# 2026-09-27 冻结值：人类池在机制验证开工前的状态
FROZEN_POOL_SHA256 = "f6678e198ce87c895e20bd12f2d244c0b1387e207fc12e14ca19feca9202027d"
FROZEN_DISPLAY_SHA = "dd2e336e539ea813b7ea445ebdf8ae5210ceb3434e0ddb2352eda921a8420986"


def test_pool_file_and_sidecar_unchanged():
    """问卷池未被震动：文件 sha256 与 display_sha sidecar 均等于冻结值"""
    actual = hashlib.sha256(POOL.read_bytes()).hexdigest()
    assert actual == FROZEN_POOL_SHA256, f"pool.json 被改动: {actual}"
    assert POOL_SHA.read_text(encoding="utf-8").strip() == FROZEN_DISPLAY_SHA


def test_judge_qa_ids_absent_from_pool():
    """judge_qa 的题号与文本不得泄漏进问卷池"""
    pool = json.loads(POOL.read_text(encoding="utf-8"))
    ids = {it.get("item_id", "") for it in pool["items"]}
    assert all(not i.startswith("CS-EVO-") for i in ids), "judge_qa 题号出现在问卷池"

    pool_texts = {it.get("question", "") for it in pool["items"]}
    pool_texts |= {it.get("model_answer", "") for it in pool["items"]}
    leaked = pool_texts & _judge_qa_texts()
    assert not leaked, f"judge_qa 文本泄漏进问卷池: {len(leaked)} 条"


def _judge_qa_texts() -> set[str]:
    """读 judge_qa 扰动集里的文本（兼容裸 list 与 dict 两种外形）"""
    path = JUDGE_QA / "perturbation_set.json"
    if not path.is_file():
        return set()
    data = json.loads(path.read_text(encoding="utf-8"))
    items = data.get("samples", data.get("items", [])) if isinstance(data, dict) else data
    return {s["text"] for s in items if isinstance(s, dict) and s.get("text")}
