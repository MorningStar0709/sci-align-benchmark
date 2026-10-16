"""Control domain scoring helpers — single-question scoring + record building.

Thin wrappers over _shared generic functions with control-domain callbacks.
"""


from mineru_eval._shared.eval_scoring_base import _generic_score_one_question
from mineru_eval._shared.report_base import build_eval_record
from mineru_eval._shared.scoring_base import _dual_judge_score

SCORER_VERSION = "2.0"
SCORER_METHOD = "llm_judge"


def _score_one_question(
    q, question_text, expected,
    target_client, target_model, provider_kind,
    judge_client, judge_model,
    *, dual_judge=False, judge2_client=None, judge2_model=None,
    arb_client=None, arb_model=None, retries=3,
    all_dimensions: bool = False,
) -> tuple[str, dict, bool]:
    """Call target model and score via judge. Returns (model_answer, result, is_failure)."""
    from mineru_eval._shared.call_utils import call_target_model
    from mineru_eval.control.evaluate import score_model_answer

    dimension = q.get("dimension", "?")
    use_dual = (
        dual_judge and judge2_client and judge2_model
        and (dimension in ("A", "B", "C", "D") if all_dimensions else dimension in ("C", "D"))
    )

    return _generic_score_one_question(
        q, question_text, expected,
        target_client, target_model,
        judge_client, judge_model,
        target_call_fn=lambda q_text, client, model: call_target_model(
            q_text, client, model, provider_kind=provider_kind,
        ),
        score_fn=score_model_answer,
        dual_judge=use_dual,
        dual_judge_fn=_dual_judge_score,
        judge2_client=judge2_client, judge2_model=judge2_model,
        arb_client=arb_client, arb_model=arb_model,
        retries=retries,
        score_kwargs={"rubric": q.get("rubric")},
    )


def _build_eval_record(q, model_answer, result):
    """Build a result record dict from scoring output."""
    return build_eval_record(
        q, model_answer, result,
        scoring_mode="llm_judge",
        scorer_version=SCORER_VERSION,
    )
