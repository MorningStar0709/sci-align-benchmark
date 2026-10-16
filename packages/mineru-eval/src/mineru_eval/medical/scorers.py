"""Medical domain scorers — MedicalScorer and scoring utilities.

Extracted from medical_judge.py. Imports shared utilities from _shared and prompts from medical.prompts.
"""

import logging

from mineru_eval._shared.judge_base import _base_judge_call
from mineru_eval._shared.json_utils import _parse_judge_result
from mineru_eval.medical.prompts import MEDICAL_PROMPTS

_logger = logging.getLogger(__name__)

VALID_MEDICAL_SCORES = {0.0, 0.25, 0.5, 0.75, 1.0}


def _greedy_parse(raw):
    """Parse medical judge result with greedy JSON extraction."""
    return _parse_judge_result(raw, valid_scores=VALID_MEDICAL_SCORES, greedy=True)


# ---------------------------------------------------------------------------
# MedicalScorer
# ---------------------------------------------------------------------------


class MedicalScorer:
    """医学跨域 QA 评分器 — identify/model/transfer x L1-L4 共 12 套量规。

    与 AScorer/BScorer/CScorer/DScorer 同级，但独立实现。
    """

    VERSION = "2.0"
    METHOD = "medical_llm_judge"

    @staticmethod
    def judge(question, expected_answer, model_answer, dimension, client, model_name,
              difficulty="L2", disable_thinking: bool = False):
        key = (dimension, difficulty)
        if key not in MEDICAL_PROMPTS:
            fallback_key = (dimension, "L2")
            if fallback_key not in MEDICAL_PROMPTS:
                return {
                    "score": 0.0,
                    "reason": f"Unknown medical dimension/difficulty: {dimension}/{difficulty}",
                    "issues": [f"unknown medical dimension/difficulty: {dimension}/{difficulty}"],
                }
            _logger.warning(
                "Medical judge fallback: (%s, %s) not found, using (%s, L2)",
                dimension, difficulty, dimension
            )
            key = fallback_key

        system_prompt, user_template = MEDICAL_PROMPTS[key]
        return _base_judge_call(
            system_prompt, user_template,
            question, expected_answer, model_answer, client, model_name,
            parse_fn=_greedy_parse,
        )


# ---------------------------------------------------------------------------
# Anthropic-compatible scoring (for arbitration Judge)
# ---------------------------------------------------------------------------


def _score_via_anthropic_medical(client, model_name, question, expected, model_answer,
                                 dimension, difficulty="L2"):
    """Score via Anthropic or OpenAI-compatible client (for arbitration Judge)."""
    key = (dimension, difficulty)
    if key not in MEDICAL_PROMPTS:
        fallback_key = (dimension, "L2")
        if fallback_key not in MEDICAL_PROMPTS:
            return {"score": 0.0,
                    "reason": f"Unknown medical dimension/difficulty: {dimension}/{difficulty}",
                    "issues": [f"unknown medical dimension/difficulty: {dimension}/{difficulty}"]}
        key = fallback_key

    sys_p, usr_tpl = MEDICAL_PROMPTS[key]

    if hasattr(client, 'messages') and hasattr(client.messages, 'create'):
        resp = client.messages.create(
            model=model_name, max_tokens=3000,
            system=sys_p, messages=[{"role": "user", "content": usr_tpl.format(
                question=question, expected_answer=expected, model_answer=model_answer,
            )}],
        )
        text = "".join(b.text for b in resp.content if hasattr(b, "text") and b.text).strip()
    else:
        user_prompt = usr_tpl.format(
            question=question, expected_answer=expected, model_answer=model_answer,
        )
        resp = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": sys_p},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,
            max_tokens=3000,
        )
        text = (resp.choices[0].message.content or "").strip()

    return _greedy_parse(text)


# ---------------------------------------------------------------------------
# Public routing function
# ---------------------------------------------------------------------------


def score_medical_answer(question, expected, model_answer, dimension,
                         judge_client, judge_model, difficulty="L2",
                         disable_thinking: bool = False):
    """医学域评分路由 — 等价于控制科学的 score_model_answer。"""
    return MedicalScorer.judge(
        question, expected, model_answer, dimension,
        judge_client, judge_model,
        difficulty=difficulty,
        disable_thinking=disable_thinking,
    )
