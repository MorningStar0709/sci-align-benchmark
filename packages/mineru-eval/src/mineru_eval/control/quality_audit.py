"""S2* AI质量审计 — 四维批量审计 benchmark 题目质量 (Control domain).

Moved from quality_audit.py. Imports _extract_json_block from _shared.json_utils.
"""

import argparse
import json
import os
import random
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]

from mineru_eval._shared.json_utils import _extract_json_block
from mineru_eval.utils import load_json, write_json_atomic

DEFAULT_CORE = ROOT / "benchmark" / "dataset" / "core.json"
DEFAULT_OUTPUT = ROOT / "benchmark" / "eval" / "reports" / "quality_audit.json"
DEFAULT_SAMPLE = 50

_deepseek_key = None


def _get_deepseek_key():
    global _deepseek_key
    if _deepseek_key is None:
        _deepseek_key = os.getenv("DEEPSEEK_API_KEY", "")
    return _deepseek_key


AUDIT_SYSTEM_PROMPT = """你是控制科学领域资深命题审核专家。请对以下题目的质量进行四维审计。

## 评分维度
1. **clarity (题目清晰度, 1-5)**: 题目表述是否明确、无歧义？术语是否定义清晰？5=完全清晰无歧义，1=严重模糊
2. **correctness (答案正确性, 1-5)**: 标准答案是否技术上正确？数学推导/公式是否有误？5=完全正确，1=严重错误
3. **difficulty_alignment (难度对齐, 1-5)**: 标注的 L1/L2/L3 是否与实际难度匹配？5=完全匹配，1=严重偏差
4. **dimension_fit (维度归类, 1-5)**: 题目归类到 A/B/C/D 维度是否准确？该维度标签的标准定义请自行理解。5=完全契合，1=归类错误

## A/B/C/D 维度参考
- A (概念定义与数学表达): 考察基础概念、定义、公式
- B (多步推理与计算求解): 需要多步推导或计算
- C (敏感性分析与方案对比): 分析参数变化影响或对比不同方案
- D (完整控制方案设计与综合评估): 设计完整控制系统并综合评估

## 输出格式
严格输出如下 JSON，不要输出其他内容:
{
  "clarity": {"score": <1-5>, "reason": "<一句话理由>", "flag": "<none|review|critical>"},
  "correctness": {"score": <1-5>, "reason": "<一句话理由>", "flag": "<none|review|critical>"},
  "difficulty_alignment": {"score": <1-5>, "reason": "<一句话理由>", "flag": "<none|review|critical>"},
  "dimension_fit": {"score": <1-5>, "reason": "<一句话理由>", "flag": "<none|review|critical>"}
}"""


def build_audit_prompt(q):
    dimension = q.get("dimension", "?")
    difficulty = q.get("difficulty_level", "?")
    question_text = q.get("question", "")
    answer = q.get("answer", "")
    steps = q.get("reasoning_steps", [])
    category = q.get("control_category", [])

    steps_text = "\n".join(f"  {i + 1}. {s}" for i, s in enumerate(steps)) if steps else "（无推理步骤）"
    return f"""## 题目信息
- 题目ID: {q.get('id', '?')}
- 标注维度: {dimension}
- 标注难度: {difficulty}
- 控制分类: {', '.join(category) if category else '无'}

## 题目内容
{question_text}

## 标准答案
{answer}

## 推理步骤
{steps_text}

请进行四维质量审计。"""


def audit_one(client, model, q, retries=2):
    user_prompt = build_audit_prompt(q)
    for attempt in range(retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": AUDIT_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.1,
                max_tokens=2000,
            )
            raw = resp.choices[0].message.content.strip()
            result = _parse_audit_result(raw)
            result["question_id"] = q.get("id", "?")
            result["dimension"] = q.get("dimension", "")
            result["difficulty"] = q.get("difficulty_level", "")
            return result
        except Exception as e:
            if attempt == retries:
                return {
                    "question_id": q.get("id", "?"),
                    "dimension": q.get("dimension", ""),
                    "difficulty": q.get("difficulty_level", ""),
                    "error": str(e),
                    "clarity": {"score": 0, "reason": f"API error: {e}", "flag": "critical"},
                    "correctness": {"score": 0, "reason": "", "flag": "critical"},
                    "difficulty_alignment": {"score": 0, "reason": "", "flag": "critical"},
                    "dimension_fit": {"score": 0, "reason": "", "flag": "critical"},
                }
            time.sleep(1 * (attempt + 1))


def _parse_audit_result(raw):
    json_str = _extract_json_block(raw)
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        return {
            "clarity": {"score": 0, "reason": f"parse error: {raw[:200]}", "flag": "critical"},
            "correctness": {"score": 0, "reason": "", "flag": "critical"},
            "difficulty_alignment": {"score": 0, "reason": "", "flag": "critical"},
            "dimension_fit": {"score": 0, "reason": "", "flag": "critical"},
        }


def compute_summary(results):
    dims = ["clarity", "correctness", "difficulty_alignment", "dimension_fit"]
    summary = {}
    flags_all = {"none": 0, "review": 0, "critical": 0}
    for d in dims:
        scores = [r.get(d, {}).get("score", 0) for r in results if r.get(d, {}).get("score", 0) > 0]
        flags = Counter()
        for r in results:
            f = r.get(d, {}).get("flag", "critical")
            flags[f] += 1
            flags_all[f] += 1
        summary[d] = {
            "n_valid": len(scores),
            "mean": round(sum(scores) / len(scores), 2) if scores else None,
            "min": min(scores) if scores else None,
            "max": max(scores) if scores else None,
            "flags": dict(flags),
        }
    summary["overall"] = {
        "mean": round(sum(
            summary[d]["mean"] for d in dims if summary[d]["mean"] is not None
        ) / max(1, sum(1 for d in dims if summary[d]["mean"] is not None)), 2),
        "total_flags": dict(flags_all),
    }
    return summary


def main():
    parser = argparse.ArgumentParser(description="S2* AI质量审计 — 四维批量审计 benchmark 题目")
    parser.add_argument("--core", "-i", default=str(DEFAULT_CORE), help="core.json 路径")
    parser.add_argument("--output", "-o", default=str(DEFAULT_OUTPUT), help="输出路径")
    parser.add_argument("--sample", "-n", type=int, default=DEFAULT_SAMPLE, help="采样题数 (0=全量)")
    parser.add_argument("--seed", type=int, default=42, help="采样种子")
    parser.add_argument("--dimension", default=None, help="限定维度 A/B/C/D")
    parser.add_argument("--api-key", default=None, help="DeepSeek API Key")
    parser.add_argument("--model", default="deepseek-v4-flash", help="审计模型")
    args = parser.parse_args()

    core_path = Path(args.core)
    if not core_path.exists():
        raise SystemExit(f"core.json 不存在: {core_path}")

    core_data = load_json(core_path)
    questions = core_data.get("questions", [])

    if args.dimension:
        questions = [q for q in questions if q.get("dimension") == args.dimension.upper()]

    if args.sample and args.sample < len(questions):
        rng = random.Random(args.seed)
        questions = rng.sample(questions, args.sample)

    api_key = args.api_key or _get_deepseek_key()
    if not api_key:
        raise SystemExit("需要 DeepSeek API Key")

    from mineru_core.settings import PROVIDERS
    from openai import OpenAI, DefaultHttpxClient
    from mineru_eval.client_factory import wrap_client
    client = wrap_client(OpenAI(
        api_key=api_key, base_url=PROVIDERS["deepseek"].base_url,
        timeout=120, max_retries=0,
        http_client=DefaultHttpxClient(proxy=None, trust_env=False),
    ))

    print(f"审计 {len(questions)} 题 (模型: {args.model})")
    results = []
    for i, q in enumerate(questions):
        print(f"[{i + 1}/{len(questions)}] {q.get('id', '?')}  (dim={q.get('dimension', '')}, diff={q.get('difficulty_level', '')})", flush=True)
        result = audit_one(client, args.model, q)
        results.append(result)
        c = result.get("clarity", {}).get("score", "?")
        o = result.get("correctness", {}).get("score", "?")
        d = result.get("difficulty_alignment", {}).get("score", "?")
        f = result.get("dimension_fit", {}).get("score", "?")
        print(f"  clarity={c} correctness={o} diff_align={d} dim_fit={f}", flush=True)
        if i < len(questions) - 1:
            time.sleep(0.3)

    summary = compute_summary(results)
    report = {
        "meta": {
            "project": "ControlSci S2* AI质量审计",
            "audit_model": args.model,
            "sample_size": len(results),
            "total_population": len(core_data.get("questions", [])),
            "seed": args.seed,
            "dimension_filter": args.dimension or "all",
        },
        "summary": summary,
        "results": results,
    }

    write_json_atomic(args.output, report)
    print(f"\n[OK] 审计报告已写入: {args.output}")
    for d in ["clarity", "correctness", "difficulty_alignment", "dimension_fit"]:
        s = summary[d]
        print(f"  {d}: mean={s['mean']}, range={s['min']}–{s['max']}, flags={s['flags']}")
    print(f"  overall mean: {summary['overall']['mean']}")


if __name__ == "__main__":
    main()
