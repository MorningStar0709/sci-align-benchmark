"""Control science domain scorers — AScorer, BScorer, CScorer, DScorer, GenericScorer, RubricScorer.

Extracted from judge.py. Imports shared utilities from _shared and prompts from control.prompts.
"""

import json

from mineru_eval._shared.judge_base import _base_judge_call
from mineru_eval._shared.json_utils import _coerce_float, _extract_json_block
from mineru_eval._shared.scoring_base import (
    _rubric_judge_batch,
    _rubric_judge_single,
)
from mineru_eval.control.prompts import (
    A_SYSTEM_PROMPT,
    A_USER_PROMPT_TEMPLATE,
    B_SYSTEM_PROMPT,
    B_USER_PROMPT_TEMPLATE,
    C_SYSTEM_PROMPT,
    C_USER_PROMPT_TEMPLATE,
    D_SYSTEM_PROMPT,
    D_USER_PROMPT_TEMPLATE,
)

# NOTE: 所有 scorer 函数签名中的 `disable_thinking` 参数已废弃。
# 代理层 (wrap_client) 现在统一自动注入 thinking 禁用参数，此参数无效。
# 保留仅为向后兼容，后续版本可移除。


class AScorer:
    """答案语义等价性评分器 (Dimension A)."""

    VERSION = "1.1"
    METHOD = "llm_judge"

    @staticmethod
    def judge(question, expected_answer, model_answer, client, model_name,
              disable_thinking: bool = False):
        return _base_judge_call(
            A_SYSTEM_PROMPT, A_USER_PROMPT_TEMPLATE,
            question, expected_answer, model_answer, client, model_name,
        )


class BScorer:
    """推理步骤逻辑一致性评分器 (Dimension B)."""

    VERSION = "1.1"
    METHOD = "llm_judge"

    @staticmethod
    def judge(question, expected_answer, model_answer, client, model_name,
              disable_thinking: bool = False):
        return _base_judge_call(
            B_SYSTEM_PROMPT, B_USER_PROMPT_TEMPLATE,
            question, expected_answer, model_answer, client, model_name,
        )


class CScorer:
    """敏感性分析与方案对比评分器 (Dimension C)."""

    VERSION = "1.0"
    METHOD = "llm_judge"

    @staticmethod
    def judge(question, expected_answer, model_answer, client, model_name,
              disable_thinking: bool = False):
        return _base_judge_call(
            C_SYSTEM_PROMPT, C_USER_PROMPT_TEMPLATE,
            question, expected_answer, model_answer, client, model_name,
        )


class DScorer:
    """完整控制方案设计与综合评估评分器 (Dimension D)."""

    VERSION = "1.0"
    METHOD = "llm_judge"

    @staticmethod
    def judge(question, expected_answer, model_answer, client, model_name,
              disable_thinking: bool = False):
        return _base_judge_call(
            D_SYSTEM_PROMPT, D_USER_PROMPT_TEMPLATE,
            question, expected_answer, model_answer, client, model_name,
        )


class GenericScorer:
    """通用 LLM-as-Judge 评分器，用于 C/D 维回退评分。"""

    VERSION = "1.1"
    METHOD = "llm_judge"

    @staticmethod
    def judge(question, expected_answer, model_answer, client, model_name,
              disable_thinking: bool = False):
        system_prompt = """你是一个控制科学答案评分员。判断模型回答与标准答案是否对应同一结论。
评分规则：
- 1.0: 完全正确，核心结论一致
- 0.75: 基本正确，有少量表述差异
- 0.5: 部分正确，存在明显遗漏
- 0.25: 有正确元素但整体不完整
- 0.0: 错误或空白

输出格式：{"score": <分数>, "reason": "<理由>", "issues": ["<问题>"]}
分数只能是 0.0, 0.25, 0.5, 0.75, 1.0 之一。"""

        user_prompt = f"""## 问题
{question}

## 标准答案
{expected_answer}

## 模型回答
{model_answer}

请评分。"""

        kwargs = dict(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,
            max_tokens=2000,
        )
        response = client.chat.completions.create(**kwargs)
        raw = response.choices[0].message.content.strip()
        json_str = _extract_json_block(raw)
        try:
            result = json.loads(json_str)
        except (json.JSONDecodeError, ValueError):
            return {"score": 0.0, "reason": f"Parse error: {raw[:200]}", "issues": ["judge parse failure"]}
        valid = {0.0, 0.25, 0.5, 0.75, 1.0}
        score = _coerce_float(result.get("score", 0.0), default=0.0)
        if score not in valid:
            score = min(valid, key=lambda v: abs(v - score))
        issues = result.get("issues", [])
        if isinstance(issues, str):
            issues = [issues]
        elif not isinstance(issues, list):
            issues = []
        return {"score": score, "reason": result.get("reason", ""), "issues": issues}


# Backward-compatible alias (kept for legacy imports)
generic_judge = GenericScorer.judge


class RubricScorer:
    """开放性设计评分器 (Dimension D) — 逐 rubric 项加权评分。"""

    VERSION = "1.1"
    METHOD = "llm_judge_rubric"

    @staticmethod
    def judge(question, expected_answer, model_answer, client, model_name, rubric=None,
              disable_thinking: bool = False):
        if not rubric:
            return generic_judge(question, expected_answer, model_answer, client, model_name,
                                 disable_thinking=disable_thinking)

        rubric_items = []
        for name, item in rubric.items():
            if not isinstance(item, dict):
                continue
            rubric_items.append({
                "name": name,
                "description": item.get("description", ""),
                "weight": float(item.get("weight", 1)),
            })

        batch_result = _rubric_judge_batch(question, rubric_items, model_answer, client, model_name,
                                           disable_thinking=disable_thinking)

        details = []
        total_weight = 0.0
        weighted_sum = 0.0

        if batch_result:
            for item in rubric_items:
                name = item["name"]
                weight = item["weight"]
                detail = batch_result.get(name, {"score": 0.0, "reason": "Not evaluated"})
                score = detail["score"]
                details.append({
                    "name": name,
                    "score": score,
                    "weight": weight,
                    "reason": detail.get("reason", ""),
                })
                total_weight += weight
                weighted_sum += score * weight
        else:
            for item in rubric_items:
                name = item["name"]
                weight = item["weight"]
                result = _rubric_judge_single(question, item, model_answer, client, model_name,
                                              disable_thinking=disable_thinking)
                score = result["score"]
                details.append({
                    "name": name,
                    "score": score,
                    "weight": weight,
                    "reason": result.get("reason", ""),
                })
                total_weight += weight
                weighted_sum += score * weight

        final_score = round(weighted_sum / total_weight, 4) if total_weight > 0 else 0.0
        reasons = [f"{d['name']}: {d['score']}" for d in details]
        return {
            "score": final_score,
            "reason": "Rubric weighted score: " + "; ".join(reasons),
            "issues": [],
            "rubric_details": details,
        }
