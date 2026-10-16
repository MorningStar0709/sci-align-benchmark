"""Shared evaluation runner — domain-agnostic evaluation orchestration.

Extracted from evaluate.py. The scoring callback (score_one_fn) is injected
by the caller, making this module domain-agnostic.
"""

import time
from datetime import datetime
from functools import partial
from pathlib import Path

from mineru_eval._shared.call_utils import call_with_retries, format_eta
from mineru_eval._shared.report_base import build_model_report, load_existing_records, select_questions
from mineru_eval.utils import append_jsonl, write_json_atomic


def _init_eval_session(data, output_path, limit, balanced_mini, resume, retry_failed):
    """Initialize evaluation session: load questions, resume state, paths."""
    questions = data.get("questions", [])
    questions = select_questions(questions, limit=limit, balanced_mini=balanced_mini)
    selected_ids = [q.get("id", f"idx-{i}") for i, q in enumerate(questions)]
    records = load_existing_records(output_path, selected_ids, retry_failed=retry_failed) if (resume and output_path) else []
    completed_ids = {record.get("id") for record in records}
    total = len(questions)
    started_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    session_start = time.time()
    initial_completed = len(records)
    status_path = Path(output_path).with_suffix(Path(output_path).suffix + ".status.json") if output_path else None
    progress_path = Path(output_path).with_suffix(Path(output_path).suffix + ".progress.jsonl") if output_path else None
    if records:
        print(f"Resume enabled: loaded {len(records)} existing records from {output_path}", flush=True)
    return questions, records, completed_ids, total, started_at, session_start, initial_completed, status_path, progress_path


def _save_eval_progress(output_path, status_path, progress_path, records, target_model, judge_model,
                         total, session_start, initial_completed, started_at, consecutive_failures, qid):
    """Save partial evaluation progress and status."""
    if not output_path:
        return
    elapsed_sec = max(time.time() - session_start, 0.001)
    session_completed = len(records) - initial_completed
    rate_q_per_min = session_completed / (elapsed_sec / 60) if session_completed > 0 else 0.0
    remaining = max(total - len(records), 0)
    eta_min = remaining / rate_q_per_min if rate_q_per_min > 0 else None
    partial_report = build_model_report(records, target_model, judge_model)
    partial_report["complete"] = len(records) >= total
    partial_report["started_at"] = started_at
    partial_report["run_metrics"] = {
        "session_completed": session_completed,
        "elapsed_sec": round(elapsed_sec, 2),
        "rate_q_per_min": round(rate_q_per_min, 4),
        "eta_min": round(eta_min, 2) if eta_min is not None else None,
        "eta_at": format_eta(eta_min),
    }
    write_json_atomic(output_path, partial_report)
    if status_path:
        status_item = {
            "model": target_model,
            "judge_model": judge_model,
            "status": "running" if len(records) < total else "completed",
            "completed": len(records),
            "total": total,
            "percent": round(len(records) / total * 100, 2) if total else 100.0,
            "remaining": remaining,
            "session_completed": session_completed,
            "elapsed_sec": round(elapsed_sec, 2),
            "rate_q_per_min": round(rate_q_per_min, 4),
            "eta_min": round(eta_min, 2) if eta_min is not None else None,
            "eta_at": format_eta(eta_min),
            "last_id": qid,
            "consecutive_failures": consecutive_failures,
            "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "progress_log": progress_path.name if progress_path else None,
        }
        write_json_atomic(status_path, status_item)
        if progress_path:
            append_jsonl(progress_path, status_item)


def _score_or_fallback(question_text, expected, model_answer, dimension, q, judge_client, judge_model,
                       retries, consecutive_failures, disable_thinking: bool = False,
                       *, score_fn=None):
    """Score the model answer or produce fallback result on failure.

    If score_fn is None, lazily imports score_model_answer from control.evaluate.
    """
    if score_fn is None:
        from mineru_eval.control.evaluate import score_model_answer
        score_fn = score_model_answer

    print("  scoring...", end=" ", flush=True)
    try:
        if model_answer:
            judge_call = partial(
                score_fn,
                question_text,
                expected,
                model_answer,
                dimension,
                judge_client,
                judge_model,
                rubric=q.get("rubric"),
                disable_thinking=disable_thinking,
            )
            result = call_with_retries("judge", judge_call, retries=retries)
        else:
            result = {"score": 0.0, "reason": "Target model failed after retries", "issues": ["target failure"]}
    except Exception as e:
        print(f"ERROR scoring after retries: {e}", flush=True)
        result = {"score": 0.0, "reason": f"Scoring error after retries: {e}", "issues": ["scoring failure"]}
        consecutive_failures += 1
    else:
        if result.get("issues") and any("failure" in str(issue).lower() for issue in result.get("issues", [])):
            consecutive_failures += 1
        else:
            consecutive_failures = 0
    return result, consecutive_failures


def model_evaluate(
    data,
    target_client,
    target_model,
    judge_client,
    judge_model,
    limit=0,
    provider_kind="openai",
    output_path=None,
    resume=False,
    retry_failed=False,
    retries=3,
    max_consecutive_failures=5,
    balanced_mini=False,
    dual_judge=False,
    judge2_client=None,
    judge2_model=None,
    arb_client=None,
    arb_model=None,
    score_one_fn=None,
    build_record_fn=None,
    scorer_version="",
    scorer_method="",
    all_dimensions_dual=False,
):
    """Run evaluation loop with injected scoring callbacks.

    Args:
        score_one_fn: Callable that scores one question. Signature must match
            _score_one_question(q, question_text, expected, target_client, target_model,
            provider_kind, judge_client, judge_model, ...) -> (model_answer, result, is_failure).
        build_record_fn: Callable that builds a record dict from (q, model_answer, result).
        scorer_version: Version string for the report.
        scorer_method: Method string for the report.
    """
    # Lazy import default scoring functions from control domain
    if score_one_fn is None:
        from mineru_eval.control.eval_scoring import _score_one_question as score_one_fn
    if build_record_fn is None:
        from mineru_eval.control.eval_scoring import _build_eval_record as build_record_fn

    questions, records, completed_ids, total, started_at, session_start, initial_completed, status_path, progress_path = \
        _init_eval_session(data, output_path, limit, balanced_mini, resume, retry_failed)
    consecutive_failures = 0
    stopped_early = False

    for i, q in enumerate(questions):
        qid = q.get("id", "?")
        if qid in completed_ids:
            print(f"[{i + 1}/{total}] {qid} already completed; skipping", flush=True)
            continue

        dimension = q.get("dimension", "?")
        difficulty = q.get("difficulty_level", "?")
        question_text = q.get("question", "")
        expected = q.get("answer", "")
        print(f"[{i + 1}/{total}] {qid} ({dimension}, {difficulty}) - calling target model...", flush=True)

        model_answer, result, is_failure = score_one_fn(
            q, question_text, expected,
            target_client, target_model, provider_kind,
            judge_client, judge_model,
            dual_judge=dual_judge, judge2_client=judge2_client, judge2_model=judge2_model,
            arb_client=arb_client, arb_model=arb_model, retries=retries,
            all_dimensions=all_dimensions_dual,
        )
        if is_failure:
            consecutive_failures += 1
        else:
            consecutive_failures = 0

        print(f"score={result['score']}", flush=True)

        record = build_record_fn(q, model_answer, result)
        records.append(record)
        completed_ids.add(qid)

        _save_eval_progress(output_path, status_path, progress_path, records, target_model, judge_model,
                            total, session_start, initial_completed, started_at, consecutive_failures, qid)

        if consecutive_failures >= max_consecutive_failures:
            stopped_early = True
            print(
                f"Stopping early after {consecutive_failures} consecutive failures. "
                "Use --resume --retry-failed after fixing the issue.",
                flush=True,
            )
            break

    report = build_model_report(records, target_model, judge_model,
                                scorer_version=scorer_version, scorer_method=scorer_method)
    report["complete"] = len(records) >= total
    if not report["complete"]:
        reason = "max_consecutive_failures" if stopped_early else "not_all_questions_completed"
        report["incomplete_reason"] = reason
        report["completed"] = len(records)
        report["total"] = total
    report["started_at"] = started_at
    return report
