"""Medical domain scoring helpers — single-question scoring + record building.

Thin wrappers over _shared generic functions with medical-domain callbacks.
No cross-domain imports from control science.
"""


from mineru_eval._shared.call_utils import call_target_model
from mineru_eval._shared.eval_scoring_base import _generic_score_one_question
from mineru_eval._shared.report_base import build_eval_record
from mineru_eval._shared.scoring_base import _dual_judge_score
from mineru_eval.medical.prompts import MEDICAL_TARGET_SYSTEM_PROMPT


def _get_scorer_version() -> str:
    """Lazy import to get MedicalScorer.VERSION without circular import."""
    from mineru_eval.medical.scorers import MedicalScorer
    return MedicalScorer.VERSION


def _call_medical_target(question_text, target_client, target_model, provider_kind,
                         *, system_prompt=None, max_tokens: int = 4000):
    """Call target model with medical-domain system prompt.

    Args:
        system_prompt: 覆盖 medical 系统提示词（默认 None = MEDICAL_TARGET_SYSTEM_PROMPT）。
        max_tokens: 输出 token 预算（默认 4000 保持历史行为；补充实验预算覆盖入口）。
    """
    return call_target_model(
        question_text, target_client, target_model,
        provider_kind=provider_kind,
        system_prompt=system_prompt or MEDICAL_TARGET_SYSTEM_PROMPT,
        max_tokens=max_tokens,
    )


# ---------------------------------------------------------------------------
# Single-question scoring (thin wrapper over _generic_score_one_question)
# ---------------------------------------------------------------------------


def _score_medical_question(
    q, question_text, expected,
    target_client, target_model, provider_kind,
    judge_client, judge_model,
    *, dual_judge=False, judge2_client=None, judge2_model=None,
    arb_client=None, arb_model=None, retries=3,
    system_prompt=None, max_tokens: int = 4000,
    all_dimensions=False,
) -> tuple[str, dict, bool]:
    """Call target model and score via medical judge.

    Returns:
        Tuple of (model_answer, result_dict, is_failure).

    Args:
        system_prompt: 覆盖 medical 系统提示词（默认 None = MEDICAL_TARGET_SYSTEM_PROMPT）。
        max_tokens: 输出 token 预算（默认 4000 保持历史行为；补充实验预算覆盖入口）。
        all_dimensions: 兼容 model_evaluate 调用契约（medical 域 dual judge 恒全维度，
            该参数仅接收不生效）。
    """
    from mineru_eval.medical.scorers import score_medical_answer, _score_via_anthropic_medical

    difficulty = q.get("difficulty_level", "L2")
    use_dual = dual_judge and judge2_client and judge2_model

    def _medical_target_call(q_text, client, model):
        return _call_medical_target(q_text, client, model, provider_kind,
                                    system_prompt=system_prompt, max_tokens=max_tokens)

    def _medical_score_fn(question, exp, m_answer, dimension, client, model):
        return score_medical_answer(
            question, exp, m_answer, dimension,
            client, model, difficulty=difficulty,
        )

    def _medical_dual_judge(question, exp, m_answer, dimension,
                            j1c, j1m, j2c, j2m, ac, am):
        def _ms(question, expected, model_answer, dimension, client, model, **kw):
            return score_medical_answer(
                question, expected, model_answer, dimension,
                client, model, difficulty=difficulty,
                disable_thinking=kw.get("disable_thinking", False),
            )
        def _ma(client, model_name, question, expected, model_answer, dimension, **kw):
            return _score_via_anthropic_medical(
                client, model_name, question, expected, model_answer, dimension,
                difficulty=difficulty,
            )
        return _dual_judge_score(
            question, exp, m_answer, dimension,
            j1c, j1m, j2c, j2m, ac, am,
            score_fn=_ms, arb_fn=_ma,
        )

    return _generic_score_one_question(
        q, question_text, expected,
        target_client, target_model,
        judge_client, judge_model,
        target_call_fn=_medical_target_call,
        score_fn=_medical_score_fn,
        dual_judge=use_dual,
        dual_judge_fn=_medical_dual_judge if use_dual else None,
        judge2_client=judge2_client, judge2_model=judge2_model,
        arb_client=arb_client, arb_model=arb_model,
        retries=retries,
    )


# ---------------------------------------------------------------------------
# Record building (thin wrapper over build_eval_record)
# ---------------------------------------------------------------------------


def _build_medical_eval_record(q, model_answer, result):
    """Build a result record dict from medical scoring output."""
    return build_eval_record(
        q, model_answer, result,
        scoring_mode="medical_llm_judge",
        scorer_version=_get_scorer_version(),
        extra_fields={
            "paper_source": q.get("paper_source"),
            "source_ref": q.get("source_ref", ""),
            "consistency_status": q.get("consistency_status", ""),
            "lint_issues": q.get("lint_issues") or None,
        },
    )
