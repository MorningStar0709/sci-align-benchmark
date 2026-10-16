r"""S2 派生数据集 — 公式验证集

从 core.json 的 question/answer 字段中提取所有 LaTeX 公式，
结构化为独立公式条目，用于公式级质量验证。

提取格式:
  - $$...$$            显示公式 (display_dollar)
  - \[...\]            显示公式 (display_bracket)
   - $...$              行内公式 (inline_dollar)
   - \(...\)            行内公式 (inline_paren)
   - \begin{env}...\end{env}  环境公式 (environment)

产出:
  - dataset/formula_validation_set.json    含 meta + formulas 完整结构
  - dataset/formula_validation_set.jsonl   每行一条公式 (轻量格式)
"""

import argparse
import json
import re
from pathlib import Path
from collections import Counter

# Phase 5: restored from new package location
ROOT = Path(__file__).resolve().parents[4]  # packages/mineru-pipeline/src/mineru_pipeline/ → project root

from mineru_core.utils import load_json, write_json_atomic

DEFAULT_INPUT = ROOT / "benchmark" / "dataset" / "core.json"
DEFAULT_OUTPUT_JSON = ROOT / "dataset" / "formula_validation_set.json"
DEFAULT_OUTPUT_JSONL = ROOT / "dataset" / "formula_validation_set.jsonl"

PATTERNS = [
    ("display_dollar", re.compile(r'\$\$(.*?)\$\$', re.DOTALL)),
    ("display_bracket", re.compile(r'\\\[(.*?)\\\]', re.DOTALL)),
    ("inline_dollar", re.compile(r'(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)')),
    ("inline_paren", re.compile(r'\\\((.+?)\\\)')),
    ("environment", re.compile(
        r'\\begin\{((?:equation|align|gather|multline|flalign|eqnarray)\**)\}(.*?)\\end\{\1\}',
        re.DOTALL
    )),
]

def extract_formulas(questions):
    """从题目列表中提取所有公式，返回公式条目列表"""
    formulas = []
    for q in questions:
        qid = q["id"]
        dim = q.get("dimension", "")
        for field in ("question", "answer"):
            text = q.get(field, "") or ""
            for ftype, pattern in PATTERNS:
                for idx, m in enumerate(pattern.finditer(text)):
                    body = m.group(1) if ftype == "environment" else (
                        m.group(1) if ftype in ("display_dollar", "display_bracket", "inline_dollar", "inline_paren")
                        else m.group(0)
                    )
                    body = body.strip()
                    if not body:
                        continue
                    formulas.append({
                        "formula_id": None,  # will be assigned
                        "source_question_id": qid,
                        "dimension": dim,
                        "field": field,
                        "formula_type": ftype,
                        "formula_text": body,
                        "formula_index": idx,
                    })
    return formulas


def assign_ids(formulas):
    """为公式条目分配 FM-XXXXX 格式 ID"""
    for i, fm in enumerate(formulas):
        fm["formula_id"] = f"FM-{i+1:05d}"


def compute_stats(formulas, questions):
    """计算统计信息"""
    type_dist = Counter(f["formula_type"] for f in formulas)
    field_dist = Counter(f["field"] for f in formulas)
    dim_dist = Counter(f["dimension"] for f in formulas)

    lengths = [len(f["formula_text"]) for f in formulas]
    src_qs = set(f["source_question_id"] for f in formulas)
    zero_fm_qs = len(questions) - len(src_qs)

    return {
        "total_formulas": len(formulas),
        "source_questions": len(src_qs),
        "questions_without_formulas": zero_fm_qs,
        "formula_length_stats": {
            "min": min(lengths) if lengths else 0,
            "max": max(lengths) if lengths else 0,
            "mean": round(sum(lengths) / len(lengths), 1) if lengths else 0,
        },
        "type_distribution": dict(type_dist),
        "field_distribution": dict(field_dist),
        "dimension_distribution": dict(dim_dist),
    }


def main():
    parser = argparse.ArgumentParser(
        description="从 core.json 提取 LaTeX 公式，生成公式验证集"
    )
    parser.add_argument("--input", "-i", default=str(DEFAULT_INPUT), help="core.json 路径")
    parser.add_argument("--output-json", default=str(DEFAULT_OUTPUT_JSON), help="JSON 输出路径")
    parser.add_argument("--output-jsonl", default=str(DEFAULT_OUTPUT_JSONL), help="JSONL 输出路径")
    parser.add_argument("--no-json", action="store_true", help="跳过 JSON 输出")
    parser.add_argument("--no-jsonl", action="store_true", help="跳过 JSONL 输出")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(f"输入文件不存在: {input_path}")

    core_data = load_json(input_path)
    questions = core_data.get("questions", [])
    meta = core_data.get("meta", {})

    formulas = extract_formulas(questions)
    assign_ids(formulas)
    stats = compute_stats(formulas, questions)

    package = {
        "meta": {
            "derived_from": meta.get("project", "ControlSci"),
            "source_version": meta.get("version", "unknown"),
            "source_updated": meta.get("updated", "unknown"),
            "filter": "question + answer fields, all LaTeX delimiters ($, $$, \\(, \\[, \\begin)",
            "requirement": ">= 500 formulas for quality validation",
            **stats,
        },
        "formulas": formulas,
    }

    if not args.no_json:
        write_json_atomic(args.output_json, package)
        print(f"[OK] JSON 已写入: {args.output_json} ({len(formulas)} 条)")

    if not args.no_jsonl:
        output_jsonl = Path(args.output_jsonl)
        output_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with output_jsonl.open("w", encoding="utf-8") as f:
            for fm in formulas:
                f.write(json.dumps(fm, ensure_ascii=False) + "\n")
        print(f"[OK] JSONL 已写入: {output_jsonl} ({len(formulas)} 行)")

    print(f"  提取完成: {len(formulas)} 条公式, 来源 {stats['source_questions']} 题")
    print(f"  类型分布: {stats['type_distribution']}")
    if lengths := [len(f["formula_text"]) for f in formulas]:
        print(f"  长度范围: {min(lengths)}–{max(lengths)}, 均值: {sum(lengths)/len(lengths):.0f} 字符")
    if len(formulas) >= 500:
        print("  ✅ 满足 ≥500 条要求")
    else:
        print(f"  ⚠️  未达到 500 条 (仅 {len(formulas)} 条)")


if __name__ == "__main__":
    main()
