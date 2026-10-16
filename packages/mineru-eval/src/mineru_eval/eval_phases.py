"""Evaluation phase orchestration — GPU generation (Phase 1) + API judge scoring (Phase 2).

Extracted from eval_adapter.py to reduce cyclomatic complexity.
"""

from __future__ import annotations

import concurrent.futures
import time
from functools import partial
from pathlib import Path

from mineru_eval.utils import write_json_atomic


def _adapter_imports():
    """Lazy import from mineru_eval.eval_adapter to avoid circular import."""
    from mineru_eval.eval_adapter import (
        SCORER_VERSION,
        _judge_one,
        _save_progress,
        call_target_model_local,
        call_with_retries,
        format_eta,
    )
    return SCORER_VERSION, _judge_one, _save_progress, call_target_model_local, call_with_retries, format_eta


# ---------------------------------------------------------------------------
# Phase 1 helpers — GPU generation
# ---------------------------------------------------------------------------
def _load_recovery_answers(recovery_path: Path | None) -> dict[str, str]:
    """Load previously generated answers from a recovery file."""
    recovered: dict[str, str] = {}
    if recovery_path and recovery_path.is_file():
        try:
            from mineru_eval.utils import load_json
            for item in load_json(recovery_path):
                recovered[item["id"]] = item["model_answer"]
            recovery_path.unlink()
            print(f"[Phase 1] Recovered {len(recovered)} answers from previous interrupted run", flush=True)
        except (ValueError, KeyError, OSError):
            pass
    return recovered


def _save_recovery_progress(recovery_path: Path, pending: list[tuple[dict, str]]) -> None:
    """Atomically save partial generation results for crash recovery."""
    recovery_data = [
        {"id": q.get("id", "?"), "model_answer": ans}
        for q, ans in pending
    ]
    write_json_atomic(recovery_path, recovery_data)


def _generate_one_answer(q, pending, remaining, model, tokenizer,
                         max_answer_tokens, retries, gen_count, gen_start,
                         recovered_count, system_prompt=None) -> tuple[str, float]:
    """Generate a single answer on GPU. Returns (model_answer, q_start_time)."""
    _, _, _, call_target_model_local, call_with_retries, _ = _adapter_imports()
    qid = q.get("id", "?")
    dimension = q.get("dimension", "?")
    difficulty = q.get("difficulty_level", "?")
    question_text = q.get("question", "")
    print(
        f"  [{len(pending) + 1}/{remaining}] "
        f"{qid} ({dimension}, {difficulty}) generating...",
        flush=True,
    )
    q_start = time.time()
    try:
        gen_call = partial(call_target_model_local, question_text, model, tokenizer,
                           max_new_tokens=max_answer_tokens, strip_think=True,
                           system_prompt=system_prompt)
        model_answer = call_with_retries("target", gen_call, retries=retries)
    except Exception as e:
        print(f"    ERROR generating: {e}", flush=True)
        model_answer = ""
    return model_answer, q_start


def _report_generation_eta(gen_count, remaining, recovered_count,
                           gen_start, q_start) -> None:
    """Report generation progress and ETA."""
    _, _, _, _, _, format_eta = _adapter_imports()
    if gen_count < 2:
        return
    elapsed = time.time() - gen_start
    rate_qpm = gen_count / (elapsed / 60)
    remaining_gen = max(0, remaining - gen_count - recovered_count)
    eta_min = remaining_gen / rate_qpm if rate_qpm > 0 else None
    last_sec = time.time() - q_start
    print(
        f"    ── {gen_count}/{remaining - recovered_count} gen'd, "
        f"last {last_sec:.0f}s, {rate_qpm:.1f} qpm, "
        f"ETA {format_eta(eta_min)}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Phase 1 — GPU generation (serial with crash recovery)
# ---------------------------------------------------------------------------
def _phase1_generate(
    questions: list,
    completed_ids: set,
    model,
    tokenizer,
    recovery_path: Path | None,
    max_answer_tokens: int,
    retries: int,
    system_prompt: str | None = None,
) -> list[tuple[dict, str]]:
    """Generate answers serially on GPU, with recovery from prior crash.

    Returns list of (question_dict, model_answer) tuples.
    On KeyboardInterrupt, saves partial progress to recovery_path.

    Args:
        system_prompt: 覆盖目标模型 system prompt（默认 None = 内置，补充实验注入点）。
    """
    pending: list[tuple[dict, str]] = []
    remaining = sum(1 for q in questions if q.get("id", "?") not in completed_ids)
    print(f"\n[Phase 1] Generating answers for {remaining} pending questions...", flush=True)
    gen_start = time.time()
    gen_count = 0  # track only actual GPU generations for accurate ETA

    recovered = _load_recovery_answers(recovery_path)

    try:
        for q in questions:
            qid = q.get("id", "?")
            if qid in completed_ids:
                continue
            if qid in recovered:
                pending.append((q, recovered[qid]))
                continue

            model_answer, q_start = _generate_one_answer(q, pending, remaining, model, tokenizer,
                                                          max_answer_tokens, retries, gen_count, gen_start,
                                                          len(recovered), system_prompt=system_prompt)
            pending.append((q, model_answer))
            gen_count += 1
            if recovery_path and gen_count % 10 == 0:
                _save_recovery_progress(recovery_path, pending)
            if gen_count >= 2:
                _report_generation_eta(gen_count, remaining, len(recovered), gen_start, q_start)
    except (Exception, KeyboardInterrupt):
        if recovery_path and pending:
            _save_recovery_progress(recovery_path, pending)
            print(f"\n[Phase 1] Interrupted! Saved {len(pending)} generated answers. Re-run to resume.", flush=True)
        raise

    return pending


# ---------------------------------------------------------------------------
# Phase 2 helper — parallel judge scoring
# ---------------------------------------------------------------------------
def _phase2_score(
    pending: list[tuple[dict, str]],
    *,
    judge_client,
    judge_model: str,
    retries: int,
    disable_thinking: bool,
    parallel: int,
    max_consecutive_failures: int,
    records: list[dict],
    completed_ids: set[str],
    adapter_name: str,
    output_path: Path | None,
    status_path: Path | None,
    total: int,
    session_start: float,
    initial_completed: int,
    started_at: str,
    score_fn=None,
) -> tuple[list[dict], bool]:
    """Score pending answers via parallel judge API workers.

    Returns (records, stopped_early). On crash, saves progress before re-raise.
    """
    import threading

    SCORER_VERSION, _judge_one, _save_progress, _, _, format_eta = _adapter_imports()
    print(f"\n[Phase 2] Scoring {len(pending)} answers via {parallel} parallel judge workers...", flush=True)
    judge_start = time.time()
    write_lock = threading.Lock()
    cf_count = [0]
    completed_count = len(records)
    stopped_early = False
    last_qid = "?"

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=parallel) as executor:
            futures = {}
            for q, answer in pending:
                future = executor.submit(
                    _judge_one, q, answer, judge_client, judge_model, retries,
                    disable_thinking=disable_thinking,
                    score_fn=score_fn,
                )
                futures[future] = q

            for future in concurrent.futures.as_completed(futures):
                q = futures[future]
                qid = q.get("id", "?")
                last_qid = qid
                try:
                    record = future.result()
                    issues = record.get("issues", []) or []
                    is_failure = bool(issues) and any(
                        "failure" in str(issue).lower() for issue in issues
                    )
                except Exception as e:
                    print(f"  [{qid}] JUDGE EXCEPTION: {e}", flush=True)
                    record = {
                        "id": qid,
                        "dimension": q.get("dimension", "?"),
                        "difficulty_level": q.get("difficulty_level", "?"),
                        "score": 0.0,
                        "reason": f"Judge API error: {e}",
                        "issues": ["judge failure"],
                        "model_answer": "",
                        "expected_answer": q.get("answer", ""),
                        "scoring_mode": "llm_judge",
                        "scorer_version": SCORER_VERSION,
                    }
                    is_failure = True

                with write_lock:
                    cf_count[0] = cf_count[0] + 1 if is_failure else 0
                    records.append(record)
                    completed_ids.add(qid)
                    completed_count += 1
                    elapsed = time.time() - judge_start
                    cf_snapshot = cf_count[0]

                rate = (completed_count - initial_completed) / (elapsed / 60) if elapsed > 0 else 0.0
                remaining_q = max(total - completed_count, 0)
                eta_min = remaining_q / rate if rate > 0 else None

                print(
                    f"  [{completed_count}/{total}] {qid} "
                    f"score={record.get('score', 'N/A')} "
                    f"({completed_count - initial_completed} in {elapsed:.0f}s, "
                    f"{rate:.1f} qpm, ETA {format_eta(eta_min)})",
                    flush=True,
                )

                # Save progress every 5 questions (error handler below ensures no data loss)
                if completed_count % 5 == 0:
                    _save_progress(
                        output_path, status_path, records, adapter_name, judge_model,
                        total, session_start, initial_completed, started_at,
                        cf_snapshot, qid,
                    )

                if cf_snapshot >= max_consecutive_failures:
                    stopped_early = True
                    print(
                        f"\nStopping early after {cf_snapshot} consecutive failures. "
                        f"Use --resume --retry-failed.",
                        flush=True,
                    )
                    for f in futures:
                        f.cancel()
                    break
    except (Exception, KeyboardInterrupt):
        if records:
            _save_progress(
                output_path, status_path, records, adapter_name, judge_model,
                total, session_start, initial_completed, started_at,
                cf_count[0], last_qid,
            )
        raise

    judge_sec = time.time() - judge_start
    print(f"\n[Phase 2] Judge scoring: {completed_count - initial_completed} records in {judge_sec:.1f}s", flush=True)
    return records, stopped_early


# ---------------------------------------------------------------------------
# Parallel evaluation orchestrator
# ---------------------------------------------------------------------------
def _evaluate_parallel(
    questions: list,
    completed_ids: set,
    records: list,
    model,
    tokenizer,
    judge_client,
    judge_model: str,
    adapter_name: str,
    *,
    output_path: Path | None,
    status_path: Path | None,
    total: int,
    started_at: str,
    session_start: float,
    initial_completed: int,
    retries: int,
    max_consecutive_failures: int,
    parallel: int,
    disable_thinking: bool = False,
    max_answer_tokens: int = 2048,
    score_fn=None,
    answers_cache: Path | None = None,
    system_prompt: str | None = None,
) -> dict:
    """Two-phase parallel evaluation: GPU generation (serial) + API judge (parallel).

    Phases are delegated to _phase1_generate and _phase2_score to keep
    this orchestrator's complexity under the threshold.

    Args:
        system_prompt: 覆盖目标模型 system prompt（默认 None = 内置，补充实验注入点）。
    """
    from mineru_eval.evaluate import build_model_report

    # ── Answers cache: load if exists, skip GPU inference ──
    # Supports two formats:
    #   1. Output JSON: {"records": [{"id": ..., "model_answer": ...}, ...]}
    #   2. Cache list:  [{"id": ..., "model_answer": ...}, ...]
    cached_answers: dict[str, str] = {}
    if answers_cache and answers_cache.is_file():
        try:
            from mineru_eval.utils import load_json
            data = load_json(answers_cache)
            if isinstance(data, dict):
                records_list = data.get("records", [])
            elif isinstance(data, list):
                records_list = data
            else:
                records_list = []
            if not isinstance(records_list, list):
                print(f"[Cache] WARNING: unexpected format in {answers_cache}, skipping cache", flush=True)
                records_list = []
            for item in records_list:
                if isinstance(item, dict) and "id" in item and "model_answer" in item:
                    answer = item["model_answer"]
                    if isinstance(answer, str) and answer:
                        cached_answers[item["id"]] = answer
            if cached_answers:
                print(f"[Cache] Loaded {len(cached_answers)} answers from {answers_cache}", flush=True)
            else:
                print(f"[Cache] WARNING: {answers_cache} contained no valid answers, falling back to GPU", flush=True)
        except Exception as e:
            print(f"[Cache] Failed to load {answers_cache}: {e}", flush=True)

    recovery_path = (
        output_path.with_suffix(".phase1_recovery.json")
        if output_path else None
    )

    # ── Phase 1: GPU generation (or load from cache) ──
    gen_start = time.time()

    if cached_answers:
        # Build pending list from cached answers; cache-missing questions
        # (e.g. empty model_answer filtered at load) fall back to GPU generation
        # instead of being silently dropped — otherwise complete=True is unreachable.
        pending = []
        remaining = sum(1 for q in questions if q.get("id", "?") not in completed_ids)
        gen_count = 0
        gen_start_fb = time.time()
        for q in questions:
            qid = q.get("id", "?")
            if qid in completed_ids:
                continue
            if qid in cached_answers:
                pending.append((q, cached_answers[qid]))
                continue
            model_answer, _ = _generate_one_answer(
                q, pending, remaining, model, tokenizer,
                max_answer_tokens, retries, gen_count, gen_start_fb, 0,
                system_prompt=system_prompt,
            )
            pending.append((q, model_answer))
            gen_count += 1
            if gen_count >= 2:
                _report_generation_eta(gen_count, remaining, 0, gen_start_fb, time.time())
        gen_sec = time.time() - gen_start
        print(f"[Phase 1] Loaded {len(pending)} answers (cache + GPU 补齐)", flush=True)
    else:
        pending = _phase1_generate(
            questions, completed_ids, model, tokenizer,
            recovery_path, max_answer_tokens, retries,
            system_prompt=system_prompt,
        )
        gen_sec = time.time() - gen_start
        print(f"[Phase 1] Generated {len(pending)} answers in {gen_sec:.1f}s", flush=True)
        # Answers are saved in the output JSON's records[].model_answer field

    if not pending:
        report = build_model_report(records, adapter_name, judge_model)
        report["complete"] = True
        report["started_at"] = started_at
        total_sec_early = time.time() - session_start
        report["run_metrics"] = {
            "parallel_workers": parallel,
            "gen_phase_sec": round(gen_sec, 1),
            "judge_phase_sec": 0.0,
            "total_sec": round(total_sec_early, 1),
            "rate_q_per_min": round(
                (len(records) - initial_completed) / (total_sec_early / 60), 4
            ) if total_sec_early > 0 else 0.0,
        }
        return report

    import torch
    torch.cuda.empty_cache()

    # ── Phase 2: Parallel judge scoring ──
    judge_start = time.time()
    records, stopped_early = _phase2_score(
        pending,
        judge_client=judge_client,
        judge_model=judge_model,
        retries=retries,
        disable_thinking=disable_thinking,
        parallel=parallel,
        max_consecutive_failures=max_consecutive_failures,
        records=records,
        completed_ids=completed_ids,
        adapter_name=adapter_name,
        output_path=output_path,
        status_path=status_path,
        total=total,
        session_start=session_start,
        initial_completed=initial_completed,
        started_at=started_at,
        score_fn=score_fn,
    )

    # ── Build final report ──
    total_sec = time.time() - session_start
    report = build_model_report(records, adapter_name, judge_model)
    report["complete"] = len(records) >= total
    if not report["complete"]:
        report["incomplete_reason"] = (
            "max_consecutive_failures" if stopped_early else "interrupted"
        )
        report["completed"] = len(records)
        report["total"] = total
    report["started_at"] = started_at
    report["run_metrics"] = {
        "parallel_workers": parallel,
        "gen_phase_sec": round(gen_sec, 1),
        "judge_phase_sec": round(time.time() - judge_start, 1),
        "total_sec": round(total_sec, 1),
        "rate_q_per_min": round(
            (len(records) - initial_completed) / (total_sec / 60), 4
        ) if total_sec > 0 else 0.0,
    }

    # Persist Phase 1 answers as audit trail (separate from judge-scored output)
    if recovery_path and recovery_path.is_file():
        audit_path = recovery_path.with_suffix(".phase1_answers_audit.json")
        recovery_path.replace(audit_path)

    return report
