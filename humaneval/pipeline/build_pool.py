#!/usr/bin/env python3
"""从分析视图构建渲染视图 pool.json。

输入：analysis.json（含 agent 分与溯源字段的完整分析视图）
输出：pool.json（只含展示字段的渲染视图，供采集服务使用）

约束：
1. 渲染视图必须剥离所有 _DISPLAY_FORBIDDEN 字段（agent_score, score, reason 等）
2. 渲染视图字段顺序与 app.py 的 _RENDER_KEYS + _RENDER_HTML 对齐
3. 输出文件写入前计算 display_sha，与 pool.json.sha256 比对（若存在）
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

# 与 server/app.py 的 _DISPLAY_FORBIDDEN 故意各写一份：审计侧与被审计侧共用
# 一个常量的话，一次改错就能同时拆掉两道门。
_DISPLAY_FORBIDDEN = frozenset({
    "agent_score", "score", "reason", "issues", "origin_id", "difficulty_level",
    "judge1_score", "judge2_score", "avg_score", "model", "judge_model",
    "model_source", "source_file", "source_ref",
})

# 渲染视图保留字段：与 app.py 的 _RENDER_KEYS + _RENDER_HTML 对齐
_RENDER_KEYS = ("item_id", "dimension")
_RENDER_HTML = {"question": "question_html", "expected_answer": "expected_html",
                "model_answer": "answer_html"}
# 附加字段：is_attention_check 用于注意力检查题标记
_EXTRA_KEYS = ("is_attention_check",)

# pool_id._CORE 的镜像：display_sha 计算用
_CORE = ("item_id", "dimension", "question", "expected_answer", "model_answer")


def _display_sha(items) -> str:
    """渲染视图身份摘要：按文件顺序取 _CORE + is_attention_check。"""
    rows = [{f: it.get(f) for f in _CORE + ("is_attention_check",)} for it in items]
    canon = json.dumps(rows, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()


def _strip_forbidden(item: dict) -> dict:
    """从分析视图项剥离禁止字段，返回渲染视图项。"""
    out = {}
    for k in _RENDER_KEYS:
        if k in item:
            out[k] = item[k]
    for src, dst in _RENDER_HTML.items():
        if src in item:
            out[dst] = item[src]
    for k in _EXTRA_KEYS:
        if k in item:
            out[k] = item[k]
    return out


def build_pool(analysis_path: Path, output_path: Path, *, force: bool = False) -> int:
    """从 analysis.json 构建 pool.json。

    Returns:
        0 on success, 1 on validation failure.
    """
    if not analysis_path.is_file():
        print(f"错误：分析视图不存在 {analysis_path}", file=sys.stderr)
        return 1

    doc = json.loads(analysis_path.read_text(encoding="utf-8"))
    items = doc.get("items", [])
    if not items:
        print(f"错误：分析视图 {analysis_path} 中没有 items", file=sys.stderr)
        return 1

    # 检查是否含禁止字段（警告，不阻止——分析视图本来就该有）
    sample_keys = set()
    for it in items[:10]:
        sample_keys.update(it.keys())
    found_forbidden = _DISPLAY_FORBIDDEN & sample_keys
    if found_forbidden:
        print(f"提示：分析视图含禁止字段（将被剥离）：{sorted(found_forbidden)}")

    # 剥离禁止字段，构建渲染视图
    display_items = [_strip_forbidden(it) for it in items]

    # 验证：渲染视图不得含任何禁止字段
    for it in display_items:
        hit = _DISPLAY_FORBIDDEN & set(it.keys())
        if hit:
            print(f"错误：剥离后仍含禁止字段 {sorted(hit)}", file=sys.stderr)
            return 1

    # 验证：必需字段存在
    for i, it in enumerate(display_items):
        for k in _RENDER_KEYS:
            if k not in it:
                print(f"错误：第 {i} 项缺少必需字段 {k}", file=sys.stderr)
                return 1

    # 计算 display_sha
    sha = _display_sha(display_items)

    # 若已存在 pool.json.sha256，比对
    sha_path = output_path.with_suffix(".json.sha256")
    if sha_path.exists() and not force:
        stored = sha_path.read_text(encoding="utf-8").strip()
        if stored != sha:
            print("错误：display_sha 不匹配", file=sys.stderr)
            print(f"  存储: {stored[:16]}...", file=sys.stderr)
            print(f"  计算: {sha[:16]}...", file=sys.stderr)
            print("  使用 --force 覆盖", file=sys.stderr)
            return 1
        print(f"display_sha 验证通过: {sha[:16]}...")

    # 写入
    output_doc = {"items": display_items}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output_doc, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8"
    )
    sha_path.write_text(sha + "\n", encoding="utf-8")

    print(f"已构建 {output_path}: {len(display_items)} 题, sha={sha[:16]}...")
    return 0


def main():
    p = argparse.ArgumentParser(description="从分析视图构建渲染视图 pool.json")
    p.add_argument("analysis", type=Path, help="分析视图路径（含 agent 分）")
    p.add_argument("-o", "--output", type=Path, default=Path("pool/pool.json"),
                   help="输出路径（默认 pool/pool.json）")
    p.add_argument("--force", action="store_true",
                   help="强制覆盖，不验证已有 sha256")
    args = p.parse_args()
    sys.exit(build_pool(args.analysis, args.output, force=args.force))


if __name__ == "__main__":
    main()
