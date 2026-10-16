"""Shared single-question scoring orchestration.

Eliminates duplicated target-call + judge-scoring logic between
control/eval_scoring.py and medical/scoring.py.
"""

from functools import partial

from mineru_eval._shared.call_utils import call_target_model, call_with_retries


def _generic_score_one_question(
    q, question_text, expected,
    target_client, target_model,
    judge_client, judge_model,
    *,
    target_call_fn=None,
    score_fn=None,
    dual_judge=False,
    dual_judge_fn=None,
    judge2_client=None, judge2_model=None,
    arb_client=None, arb_model=None,
    retries=3,
    score_kwargs=None,
):
    """Generic single-question scoring: target call + judge scoring.

    Args:
        q: Question dict.
        question_text: Question text.
        expected: Expected answer text.
        target_client: Target model client.
        target_model: Target model name.
        judge_client: Judge model client.
        judge_model: Judge model name.
        target_call_fn: ``callable(question, client, model) -> model_answer``.
            Default: ``call_target_model`` (control-domain system prompt).
        score_fn: ``callable(question, expected, model_answer, dimension,
            client, model) -> result_dict``.  Used for single-judge path.
            Default: lazy-imports ``score_model_answer`` from control domain.
        dual_judge: Whether to use dual-judge path.
        dual_judge_fn: ``callable(question, expected, model_answer, dimension,
            j1_client, j1_model, j2_client, j2_model, arb_client, arb_model)
            -> result_dict``.  Used when *dual_judge* is True.
        judge2_client: Second judge client (for dual-judge path).
        judge2_model: Second judge model name.
        arb_client: Arbiter client (for dual-judge path).
        arb_model: Arbiter model name.
        retries: Number of retries for each LLM call.
        score_kwargs: Extra keyword arguments forwarded to *score_fn* and
            *dual_judge_fn* (e.g. ``{"rubric": ...}`` for control domain).

    Returns:
        tuple: ``(model_answer, result_dict, is_failure)``
    """
    dimension = q.get("dimension", "?")

    _sk = score_kwargs or {}

    # ── Target model call ──
    if target_call_fn is None:
        target_call_fn = lambda q_text, client, model: call_target_model(
            q_text, client, model,
        )

    try:
        target_call = partial(target_call_fn, question_text, target_client, target_model)
        model_answer = call_with_retries("target", target_call, retries=retries)
    except Exception as e:
        print(f"  ERROR calling target model after retries: {e}", flush=True)
        model_answer = ""

    # ── Judge scoring ──
    use_dual = dual_judge and dual_judge_fn is not None

    if use_dual:
        dj_label = getattr(dual_judge_fn, "__name__", "dual-judge")
        if "medical" in dj_label:
            print("  dual-judge medical scoring...", end=" ", flush=True)
        else:
            print("  dual-judge scoring...", end=" ", flush=True)
        try:
            if model_answer:
                dj_call = partial(
                    dual_judge_fn,
                    question_text, expected, model_answer, dimension,
                    judge_client, judge_model,
                    judge2_client, judge2_model,
                    arb_client, arb_model,
                    **_sk,
                )
                result = call_with_retries("dual-judge", dj_call, retries=retries)
            else:
                result = {"score": 0.0, "reason": "Target model failed", "issues": ["target failure"],
                          "judge1_score": 0.0, "judge2_score": 0.0, "avg_score": 0.0,
                          "score_diff": 0.0, "disputed": False, "arbitrated": False}
        except Exception as e:
            print(f"ERROR dual-judge: {e}", flush=True)
            result = {"score": 0.0, "reason": f"Dual-judge error: {e}", "issues": ["scoring failure"],
                      "judge1_score": 0.0, "judge2_score": 0.0, "avg_score": 0.0,
                      "score_diff": 0.0, "disputed": False, "arbitrated": False,
                      "_used_dual": True}
            return model_answer, result, True
        result["_used_dual"] = True
        return model_answer, result, False
    else:
        if score_fn is None:
            from mineru_eval.control.evaluate import score_model_answer
            score_fn = score_model_answer

        print("  scoring...", end=" ", flush=True)
        try:
            if model_answer:
                judge_call = partial(
                    score_fn,
                    question_text, expected, model_answer, dimension,
                    judge_client, judge_model,
                    **_sk,
                )
                result = call_with_retries("judge", judge_call, retries=retries)
            else:
                result = {"score": 0.0, "reason": "Target model failed after retries", "issues": ["target failure"]}
        except Exception as e:
            print(f"ERROR scoring after retries: {e}", flush=True)
            result = {"score": 0.0, "reason": f"Scoring error after retries: {e}", "issues": ["scoring failure"]}
            result["_used_dual"] = False
            return model_answer, result, True
        is_failure = bool(result.get("issues") and any("failure" in str(issue).lower() for issue in result.get("issues", [])))
        result["_used_dual"] = False
        return model_answer, result, is_failure
