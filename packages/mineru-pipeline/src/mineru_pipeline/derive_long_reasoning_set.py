"""S2 派生数据集 — 长推导链集

从 core.json 中筛选 C 维（敏感性分析与方案对比）+ 推理步数 ≥3 的题目，
导出为长推导链分析专用数据集。

产出:
  - dataset/long_reasoning_set.json    包含完整 question 字段
  - dataset/long_reasoning_set.jsonl   每行一条 (轻量格式)
"""

import argparse
import json
from pathlib import Path

# Phase 5: restored from new package location
ROOT = Path(__file__).resolve().parents[4]  # packages/mineru-pipeline/src/mineru_pipeline/ → project root

from mineru_core.utils import load_json, write_json_atomic

DEFAULT_INPUT = ROOT / "benchmark" / "dataset" / "core.json"
DEFAULT_OUTPUT_JSON = ROOT / "dataset" / "long_reasoning_set.json"
DEFAULT_OUTPUT_JSONL = ROOT / "dataset" / "long_reasoning_set.jsonl"


def derive_long_reasoning(core_data, min_steps=3):
    questions = core_data.get("questions", [])
    derived = []
    for q in questions:
        dim = q.get("dimension", "")
        steps = q.get("reasoning_steps", [])
        if dim == "C" and len(steps) >= min_steps:
            derived.append(q)
    return derived


def main():
    parser = argparse.ArgumentParser(description="从 core.json 派生长推导链数据集 (C维 + ≥3步推理链)")
    parser.add_argument("--input", "-i", default=str(DEFAULT_INPUT), help="core.json 路径")
    parser.add_argument("--output-json", default=str(DEFAULT_OUTPUT_JSON), help="JSON 输出路径")
    parser.add_argument("--output-jsonl", default=str(DEFAULT_OUTPUT_JSONL), help="JSONL 输出路径")
    parser.add_argument("--min-steps", type=int, default=3, help="最少推理步数 (默认 3)")
    parser.add_argument("--no-json", action="store_true", help="跳过 JSON 输出")
    parser.add_argument("--no-jsonl", action="store_true", help="跳过 JSONL 输出")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(f"输入文件不存在: {input_path}")

    core_data = load_json(input_path)
    derived = derive_long_reasoning(core_data, min_steps=args.min_steps)

    dim_counts = {}
    step_lens = []
    for q in derived:
        dim_counts[q["dimension"]] = dim_counts.get(q["dimension"], 0) + 1
        step_lens.append(len(q.get("reasoning_steps", [])))

    meta = core_data.get("meta", {})
    package = {
        "meta": {
            "derived_from": meta.get("project", "ControlSci"),
            "source_version": meta.get("version", "unknown"),
            "source_updated": meta.get("updated", "unknown"),
            "filter": f"dimension=C, reasoning_steps>={args.min_steps}",
            "total_derived": len(derived),
            "dimension_counts": dim_counts,
            "step_length_stats": {
                "min": min(step_lens) if step_lens else 0,
                "max": max(step_lens) if step_lens else 0,
                "mean": round(sum(step_lens) / len(step_lens), 2) if step_lens else 0,
            },
        },
        "questions": derived,
    }

    if not args.no_json:
        write_json_atomic(args.output_json, package)
        print(f"[OK] JSON 已写入: {args.output_json} ({len(derived)} 题)")

    if not args.no_jsonl:
        output_jsonl = Path(args.output_jsonl)
        output_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with output_jsonl.open("w", encoding="utf-8") as f:
            for q in derived:
                f.write(json.dumps(q, ensure_ascii=False) + "\n")
        print(f"[OK] JSONL 已写入: {output_jsonl} ({len(derived)} 行)")

    print(f"  派生完成: {len(derived)} 题 (C维, ≥{args.min_steps}步推理链)")
    if step_lens:
        print(f"  步数范围: {min(step_lens)}–{max(step_lens)}, 均值: {sum(step_lens)/len(step_lens):.1f}")


if __name__ == "__main__":
    main()
