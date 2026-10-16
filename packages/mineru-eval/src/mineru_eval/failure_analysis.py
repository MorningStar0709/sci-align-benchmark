"""
Failure Case Analysis — 失败案例分析
=====================================
基于 9 模型评测 reports，提取 score=0.0 的低分题，
用 DeepSeek API 做四维度错误分类，输出结构化分析报告。

用法:
    .venv/bin/python benchmark/eval/failure_analysis.py \
        --reports-dir benchmark/eval/reports \
        --output benchmark/eval/results/failure_analysis.json \
        --top-n 5
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import mineru_core  # noqa: F401 — trigger _load_dotenv() to load .env API keys

from mineru_core.settings import PROVIDERS

from openai import OpenAI, DefaultHttpxClient
from mineru_eval.client_factory import wrap_client


# ============================================================
# Config
# ============================================================
# API key with pipe-delimited fallback support
_key_spec = PROVIDERS["deepseek"].api_key_env
API_KEY = next((os.environ.get(k.strip()) for k in _key_spec.split("|") if os.environ.get(k.strip())), None) or os.environ.get("OPENAI_API_KEY", "")
BASE_URL = PROVIDERS["deepseek"].base_url
MODEL = PROVIDERS["deepseek"].default_model
MAX_TOKENS = 4096
TIMEOUT = 120

ERROR_CATEGORIES = {
    "概念混淆": {
        "code": "concept_confusion",
        "description": "模型对控制科学术语/定义的理解偏差，回答与标准答案语义相近但关键细节错误",
        "dimension_hint": "A",
    },
    "推理断裂": {
        "code": "reasoning_break",
        "description": "模型推导过程存在逻辑跳跃或因果链断裂，未能完成有效的多步推理",
        "dimension_hint": "B",
    },
    "条件盲区": {
        "code": "condition_blindness",
        "description": "模型忽略或误用了题目中的关键约束条件，导致在正确思路下得出错误结论",
        "dimension_hint": "C",
    },
    "设计发散": {
        "code": "design_divergence",
        "description": "模型在开放性设计/方案题目中偏离评分标准，方案不完整或未覆盖关键评分点",
        "dimension_hint": "D",
    },
}


# ============================================================
# Classification prompt
# ============================================================
CLASSIFY_SYSTEM_PROMPT = """你是一位控制科学评测专家，正在分析大模型在控制科学题目上的失败原因。

你的任务：阅读一道题目的标准答案和模型回答，判断模型的错误属于以下哪一类，并给出诊断理由。

## 四类错误定义

1. **概念混淆 (concept_confusion)**：模型对控制科学术语/定义的理解有偏差。回答中使用了错误的概念、混淆了相似术语、或对定义的理解与标准答案不一致。尽管语义上可能与正确答案接近，但关键细节出错。

2. **推理断裂 (reasoning_break)**：模型的推导过程存在逻辑跳跃或因果链断裂。可能给出了正确的起始条件，但中间推理步骤缺失或错误，未能完成有效的多步推理。

3. **条件盲区 (condition_blindness)**：模型忽略或误用了题目中的关键约束条件（如系统参数范围、初始条件、边界条件等），导致在部分正确思路下得出错误结论。

4. **设计发散 (design_divergence)**：针对开放性设计/方案题，模型的方案偏离了评分标准的核心要求，方案不完整、遗漏关键组件、或未能覆盖评分标准中的主要评分点。

## 输出格式

请严格输出 JSON（不要包含 markdown 代码块标记）:
{
    "error_category": "concept_confusion|reasoning_break|condition_blindness|design_divergence",
    "error_category_cn": "概念混淆|推理断裂|条件盲区|设计发散",
    "diagnosis": "用1-3句话描述具体的错误诊断理由，中文",
    "confidence": 0.0-1.0
}
"""


def classify_error(client, model, question_id, dimension, expected_answer, model_answer, judge_reason, issues, max_retries=3):
    """调用 DeepSeek API 对单题做错误分类"""
    user_prompt = f"""## 题目信息
- 题目ID: {question_id}
- 评测维度: {dimension}

## 标准答案
{expected_answer[:2000]}

## 模型回答
{model_answer[:2000]}

## Judge 评分理由
{judge_reason[:1000]}

## Judge 发现的问题
{json.dumps(issues[:5], ensure_ascii=False) if issues else "无"}

请根据以上信息，判断该模型的错误属于哪一类。"""

    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": CLASSIFY_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
                max_tokens=MAX_TOKENS,
                response_format={"type": "json_object"},
            )
            raw = resp.choices[0].message.content
            result = json.loads(raw)
            # Validate
            valid_codes = {c["code"] for c in ERROR_CATEGORIES.values()}
            if result.get("error_category") not in valid_codes:
                result["error_category"] = "concept_confusion"
                result["error_category_cn"] = "概念混淆"
            token_used = resp.usage.total_tokens if resp.usage else 0
            return result, token_used, None
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                err_msg = f"API 分类失败: {str(e)[:200]}"
                return {
                    "error_category": "concept_confusion",
                    "error_category_cn": "概念混淆",
                    "diagnosis": err_msg,
                    "confidence": 0.0,
                }, 0, str(e)


# ============================================================
# Main logic
# ============================================================
def load_reports(reports_dir):
    """加载所有主 report JSON（排除 chunk 和 cross_domain）"""
    reports_dir = Path(reports_dir)
    report_files = sorted(reports_dir.glob("*_report.json"))
    main_reports = [
        f for f in report_files
        if "_chunk_" not in f.name and "cross_domain" not in f.name
    ]
    all_records = []
    for rf in main_reports:
        data = json.loads(rf.read_text(encoding="utf-8"))
        model_name = data.get("model", rf.stem.replace("_report", ""))
        for r in data.get("records", []):
            r["_model"] = model_name
            r["_report_file"] = rf.name
            all_records.append(r)
    return all_records


def aggregate_by_question(records):
    """按题目 ID 聚合，返回每个题目的失败模型列表和维度"""
    by_id = defaultdict(list)
    for r in records:
        if r.get("score", 1) == 0.0:
            by_id[r["id"]].append(r)
    return by_id


def select_candidates(by_question, max_per_dim=12):
    """按维度分层抽样：选失败模型数最多的题目。
    若某维度候选不足 max_per_dim，全部纳入。"""
    dim_questions = defaultdict(list)
    for qid, recs in by_question.items():
        dim = recs[0].get("dimension", "?")
        dim_questions[dim].append((qid, recs))

    candidates = []
    for dim in ["A", "B", "C", "D"]:
        items = dim_questions.get(dim, [])
        # 按失败模型数降序
        items.sort(key=lambda x: len(x[1]), reverse=True)
        selected = items[:max_per_dim]
        for qid, recs in selected:
            best_rec = recs[0]  # 取任意一个模型记录
            candidates.append({
                "id": qid,
                "dimension": dim,
                "fail_count": len(recs),
                "failed_models": [r["_model"] for r in recs],
                "expected_answer": best_rec.get("expected_answer", ""),
                "sample_model": best_rec["_model"],
                "sample_answer": best_rec.get("model_answer", ""),
                "sample_reason": best_rec.get("reason", ""),
                "sample_issues": best_rec.get("issues", []),
            })
    return candidates


def _resample_short_categories(client, classified, by_question, top_n, total_tokens, api_failures):
    """若某错误类别不足 top_n 例，从对应 hint 维度追加候选并重新分类。"""
    by_category = defaultdict(list)
    for c in classified:
        cat = c["classification"].get("error_category", "concept_confusion")
        by_category[cat].append(c)

    classified_ids = {c["id"] for c in classified}
    additional_total = 0

    for cat_code, cat_info in ERROR_CATEGORIES.items():
        code = cat_info["code"]
        hint_dim = cat_info["dimension_hint"]
        current = len(by_category.get(code, []))

        if current >= top_n:
            continue

        needed = top_n - current
        print(f"\n  ⚠ 类别 '{cat_code}' 仅 {current} 例 (需 ≥{top_n})，从维度{hint_dim}追加 {needed} 题")

        # Find untested candidates from hint dimension
        dim_candidates = []
        for qid, recs in by_question.items():
            if qid in classified_ids:
                continue
            if recs[0].get("dimension") != hint_dim:
                continue
            dim_candidates.append((qid, recs))

        dim_candidates.sort(key=lambda x: len(x[1]), reverse=True)

        new_matches = 0
        for qid, recs in dim_candidates[:needed * 2]:  # oversample for robustness
            if new_matches >= needed:
                break

            best_rec = recs[0]
            c = {
                "id": qid,
                "dimension": hint_dim,
                "fail_count": len(recs),
                "failed_models": [r["_model"] for r in recs],
                "expected_answer": best_rec.get("expected_answer", ""),
                "sample_model": best_rec["_model"],
                "sample_answer": best_rec.get("model_answer", ""),
                "sample_reason": best_rec.get("reason", ""),
                "sample_issues": best_rec.get("issues", []),
            }

            print(f"    [{len(classified) + 1}] {qid} (维度{hint_dim}, {len(recs)}模型翻车)...", end=" ")
            result, tokens, err = classify_error(
                client, MODEL, qid, hint_dim,
                c["expected_answer"], c["sample_answer"],
                c["sample_reason"], c["sample_issues"],
            )
            total_tokens += tokens
            classified_ids.add(qid)

            if err:
                api_failures += 1
                print("⚠ API失败")
                c["classification"] = {
                    "error_category": code,
                    "error_category_cn": cat_code,
                    "diagnosis": f"API 重采样失败: {err[:200]}",
                    "confidence": 0.0,
                }
                new_matches += 1  # count failures toward target
            else:
                c["classification"] = result
                if result.get("error_category") == code:
                    new_matches += 1
                print(f"→ {result['error_category_cn']} (conf={result['confidence']:.2f})")

            c["api_tokens"] = tokens
            classified.append(c)
            additional_total += 1
            time.sleep(0.3)

        print(f"    类别 '{cat_code}' 追加后: {current + new_matches} 例")


    return classified, total_tokens, api_failures, additional_total


def run_analysis(reports_dir, output_path, top_n=5, max_candidates_per_dim=12):
    reports_dir = Path(reports_dir)
    output_path = Path(output_path)

    print(f"[1/5] 加载 reports 目录: {reports_dir}")
    all_records = load_reports(reports_dir)
    print(f"  加载 {len(all_records)} 条记录")

    print("[2/5] 聚合 score=0.0 失败题")
    by_q = aggregate_by_question(all_records)
    print(f"  共 {len(by_q)} 个唯一题目存在 score=0.0 记录")

    print(f"[3/5] 按维度抽样候选题目 (每维最多{max_candidates_per_dim}题)")
    candidates = select_candidates(by_q, max_per_dim=max_candidates_per_dim)
    print(f"  选出 {len(candidates)} 条候选")
    for dim in ["A", "B", "C", "D"]:
        dim_count = sum(1 for c in candidates if c["dimension"] == dim)
        print(f"    维度 {dim}: {dim_count} 题")

    print(f"[4/5] DeepSeek API 错误分类 (共 {len(candidates)} 题)")
    client = wrap_client(OpenAI(
        api_key=API_KEY,
        base_url=BASE_URL,
        timeout=TIMEOUT,
        max_retries=0,
        http_client=DefaultHttpxClient(proxy=None, trust_env=False),
    ))  # trust_env=False 禁系统代理 env

    total_tokens = 0
    api_failures = 0
    classified = []

    for i, c in enumerate(candidates):
        qid = c["id"]
        dim = c["dimension"]
        print(f"  [{i+1}/{len(candidates)}] {qid} (维度{dim}, {c['fail_count']}模型翻车)...", end=" ")
        result, tokens, err = classify_error(
            client, MODEL,
            qid, dim,
            c["expected_answer"],
            c["sample_answer"],
            c["sample_reason"],
            c["sample_issues"],
        )
        total_tokens += tokens
        if err:
            api_failures += 1
            print(f"⚠ API失败: {err[:80]}")
        else:
            print(f"→ {result['error_category_cn']} (conf={result['confidence']:.2f}, {tokens}tok)")

        c["classification"] = result
        c["api_tokens"] = tokens
        classified.append(c)
        time.sleep(0.3)  # 避免触发限流

    print(f"\n  API 调用完成: {len(classified)} 次, 失败 {api_failures}, 总 token: {total_tokens}")

    # Resampling: ensure each error category has at least top_n cases
    classified, total_tokens, api_failures, additional_candidates = _resample_short_categories(
        client, classified, by_q, top_n, total_tokens, api_failures
    )
    all_classified = classified

    print("[5/5] 精选典型案例并生成报告")
    by_category = defaultdict(list)
    for c in classified:
        cat = c["classification"].get("error_category", "concept_confusion")
        by_category[cat].append(c)

    # Select top-N per category
    top_cases = {}
    for cat_code, cat_info in ERROR_CATEGORIES.items():
        cat_name = cat_info["code"]
        items = by_category.get(cat_name, [])
        # Sort by confidence desc, then fail_count desc
        items.sort(key=lambda x: (x["classification"].get("confidence", 0), x["fail_count"]), reverse=True)
        selected = items[:top_n]
        top_cases[cat_name] = {
            "category_cn": cat_code,
            "description": cat_info["description"],
            "count_total": len(items),
            "cases": [
                {
                    "id": c["id"],
                    "dimension": c["dimension"],
                    "fail_count": c["fail_count"],
                    "failed_models": c["failed_models"],
                    "sample_model": c["sample_model"],
                    "expected_answer": c["expected_answer"][:1500],
                    "sample_answer": c["sample_answer"][:1500],
                    "judge_reason": c["sample_reason"][:800],
                    "issues": c["sample_issues"][:5],
                    "diagnosis": c["classification"].get("diagnosis", ""),
                    "confidence": c["classification"].get("confidence", 0),
                }
                for c in selected
            ],
        }

    # Build output JSON
    output = {
        "meta": {
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "reports_dir": str(reports_dir),
            "total_records": len(all_records),
            "total_score_zero": sum(1 for r in all_records if r.get("score", 1) == 0.0),
            "unique_failed_questions": len(by_q),
            "candidates_sampled": len(candidates),
            "api_calls": len(classified),
            "api_failures": api_failures,
            "total_tokens": total_tokens,
            "judge_model": MODEL,
        },
        "category_stats": {
            cat_info["code"]: {
                "category_cn": cat_code,
                "total_classified": len(by_category.get(cat_info["code"], [])),
                "selected_cases": len(top_cases.get(cat_info["code"], {}).get("cases", [])),
            }
            for cat_code, cat_info in ERROR_CATEGORIES.items()
        },
        "top_cases": top_cases,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  JSON 已保存: {output_path}")

    # Generate Markdown
    md_path = output_path.with_suffix(".md")
    generate_markdown(output, md_path)
    print(f"  Markdown 已保存: {md_path}")

    return output


def generate_markdown(output, md_path):
    """生成人类可读的 Markdown 报告"""
    meta = output["meta"]
    lines = [
        "# 失败案例分析报告",
        "",
        f"> 生成时间: {meta['generated_at']} | Judge: {meta['judge_model']}",
        f"> 数据源: {meta['reports_dir']} | 总记录: {meta['total_records']} | score=0.0: {meta['total_score_zero']}",
        f"> 唯一失败题: {meta['unique_failed_questions']} | 抽样分析: {meta['candidates_sampled']} | API 调用: {meta['api_calls']} | Token: {meta['total_tokens']:,}",
        "",
        "---",
        "",
        "## 概览",
        "",
    ]

    # Summary table
    lines.append("| 错误类别 | 总分类型 | 精选案例 | 说明 |")
    lines.append("|:--|--:|--:|:--|")
    for cat_code, cat_info in ERROR_CATEGORIES.items():
        code = cat_info["code"]
        stats = output["category_stats"].get(code, {})
        total = stats.get("total_classified", 0)
        selected = stats.get("selected_cases", 0)
        lines.append(f"| {cat_code} | {total} | {selected} | {cat_info['description']} |")
    lines.append("")

    # Detail sections
    top_cases = output.get("top_cases", {})
    for cat_code, cat_info in ERROR_CATEGORIES.items():
        code = cat_info["code"]
        section = top_cases.get(code, {})
        cases = section.get("cases", [])
        if not cases:
            continue

        lines.append("---")
        lines.append(f"## {cat_code} — {cat_info['description']}")
        lines.append("")

        for idx, case in enumerate(cases, 1):
            lines.append(f"### 案例 {cat_code[0]}{idx}: {case['id']}")
            lines.append("")
            lines.append(f"- **维度**: {case['dimension']} | **翻车模型数**: {case['fail_count']}")
            lines.append(f"- **失败模型**: {', '.join(case['failed_models'][:8])}")
            if len(case['failed_models']) > 8:
                lines.append(f"  ...等 {len(case['failed_models'])} 个模型")
            lines.append(f"- **示例模型**: {case['sample_model']} | **置信度**: {case['confidence']:.2f}")
            lines.append("")
            lines.append(f"**诊断**: {case['diagnosis']}")
            lines.append("")

            if case.get("issues"):
                lines.append("**Judge 发现的问题**:")
                for issue in case["issues"][:3]:
                    lines.append(f"- {issue}")
                lines.append("")

            lines.append("<details>")
            lines.append("<summary>标准答案 vs 模型回答（点击展开）</summary>")
            lines.append("")
            lines.append("**标准答案**:")
            lines.append("")
            lines.append(case["expected_answer"][:2000])
            lines.append("")
            lines.append("**模型回答** (" + case["sample_model"] + "):")
            lines.append("")
            lines.append(case["sample_answer"][:2000])
            lines.append("")
            lines.append("</details>")
            lines.append("")

    # Cross-dimension summary
    lines.append("---")
    lines.append("## 跨维度分析摘要")
    lines.append("")

    # Which dimensions have most failures
    dim_counts = defaultdict(int)
    for cat_code in ERROR_CATEGORIES:
        code = ERROR_CATEGORIES[cat_code]["code"]
        for case in top_cases.get(code, {}).get("cases", []):
            dim_counts[case["dimension"]] += 1

    lines.append("### 失败维度分布")
    lines.append("")
    for dim in ["A", "B", "C", "D"]:
        count = dim_counts.get(dim, 0)
        dim_names = {"A": "定义与概念", "B": "推理与推导", "C": "条件与约束", "D": "设计与方案"}
        lines.append(f"- **维度 {dim}** ({dim_names.get(dim, '?')}): {count} 例")

    md_path.write_text("\n".join(lines), encoding="utf-8")


# ============================================================
# CLI
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="失败案例分析 — 四维度错误分类")
    parser.add_argument("--reports-dir", default="benchmark/eval/reports", help="reports 目录路径")
    parser.add_argument("--output", default="benchmark/eval/results/failure_analysis.json", help="输出 JSON 路径")
    parser.add_argument("--top-n", type=int, default=5, help="每类精选案例数")
    parser.add_argument("--max-candidates-per-dim", type=int, default=12, help="每维度最大候选数")
    args = parser.parse_args()

    if not Path(args.reports_dir).exists():
        print(f"错误: reports 目录不存在: {args.reports_dir}")
        sys.exit(1)

    run_analysis(args.reports_dir, args.output, args.top_n, args.max_candidates_per_dim)


if __name__ == "__main__":
    main()
