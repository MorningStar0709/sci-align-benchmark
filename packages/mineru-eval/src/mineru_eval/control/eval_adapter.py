"""Evaluate FR-LoRA adapters via local inference + LLM-as-Judge (Control domain).

Moved from eval_adapter.py. Imports from _shared and control.evaluate.
"""

from __future__ import annotations

import os
import time
from datetime import datetime
from functools import partial
from pathlib import Path

# ── Must be set BEFORE any ML imports that transitively load bitsandbytes ──
os.environ.setdefault("BNB_CUDA_VERSION", "130")  # Windows CUDA 13.2 compat

# Project root: control/eval_adapter.py is at packages/mineru-eval/src/mineru_eval/control/
ROOT = Path(__file__).resolve().parents[5]

import mineru_core  # noqa: F401 — trigger _load_dotenv() to load .env API keys

# Reuse existing infrastructure
from mineru_eval._shared.call_utils import TARGET_SYSTEM_PROMPT, call_with_retries, format_eta
from mineru_eval._shared.eval_runner import _score_or_fallback
from mineru_eval._shared.report_base import build_model_report, load_existing_records, select_questions
from mineru_eval.eval_phases import _evaluate_parallel
from mineru_eval.utils import load_json, write_json_atomic

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SCORER_VERSION = "1.1"


def _make_dual_judge_score_fn(
    dual_judge: bool,
    judge2_client,
    judge2_model: str | None,
    arb_client,
    arb_model: str | None,
    *,
    all_dimensions: bool = False,
):
    """构建双 Judge 分发函数，透传给 _score_or_fallback / _judge_one。

    C/D 维度 → _dual_judge_score；A/B 维度 → score_model_answer（单 Judge）。
    当 all_dimensions=True 时，ABCD 全维度均走 Dual Judge。
    当 dual_judge=False 时返回 None，_score_or_fallback 将使用默认的 score_model_answer。
    """
    if not dual_judge:
        return None

    if not judge2_client or not judge2_model:
        import warnings
        warnings.warn(
            "dual_judge=True 但 judge2_client/judge2_model 未完整提供，"
            "C/D 维度将回退为单 Judge 评分。",
            UserWarning, stacklevel=2,
        )
        return None

    from mineru_eval._shared.scoring_base import _dual_judge_score, _score_via_openai
    from mineru_eval.control.evaluate import score_model_answer

    def _dispatch(question, expected, model_answer, dimension, client, model,
                 *, rubric=None, disable_thinking=False):
        use_dual = (
            dimension in ("A", "B", "C", "D") if all_dimensions
            else dimension in ("C", "D")
        ) and judge2_client and judge2_model
        if use_dual:
            return _dual_judge_score(
                question, expected, model_answer, dimension,
                client, model, judge2_client, judge2_model,
                arb_client=arb_client, arb_model=arb_model,
                arb_fn=_score_via_openai,
                disable_thinking=disable_thinking, rubric=rubric,
            )
        return score_model_answer(
            question, expected, model_answer, dimension, client, model,
            rubric=rubric, disable_thinking=disable_thinking,
        )

    return _dispatch


# ---------------------------------------------------------------------------
# Local model inference
# ---------------------------------------------------------------------------
def call_target_model_local(
    question: str,
    model,
    tokenizer,
    max_new_tokens: int = 2048,
    temperature: float = 0.3,
    strip_think: bool = False,
    system_prompt: str | None = None,
) -> str:
    """Generate answer using a local PeftModel.

    Args:
        strip_think: If True, strip <think>...</think> prefix from output.
            Kept False by default for backward compatibility.
        system_prompt: Override system prompt (default None = TARGET_SYSTEM_PROMPT，
            空串同样回退内置——与 call_utils.call_target_model 的 or 语义一致)。
            补充实验 2×2 格子③④的告知措辞注入点（决策报告 §3.4）。
    """
    import torch

    sys_p = system_prompt or TARGET_SYSTEM_PROMPT
    messages = [
        {"role": "system", "content": sys_p},
        {"role": "user", "content": question},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            top_k=50,
            top_p=0.95,
        )

    input_len = inputs["input_ids"].shape[1]
    generated = outputs[0][input_len:]
    answer = tokenizer.decode(generated, skip_special_tokens=True).strip()

    # 兜底：如果模型输出包含 </think>，只取之后的部分作为正式回答
    # （防止 thinking 泄漏到最终评分中，仅 strip_think=True 时生效）
    if strip_think and '</think>' in answer:
        answer = answer.split('</think>')[-1].strip()
    return answer


def _judge_one(
    q: dict,
    model_answer: str,
    judge_client,
    judge_model: str,
    retries: int,
    disable_thinking: bool = False,
    *,
    score_fn=None,
) -> dict:
    """Score a single question via judge API — thread-safe, no GPU."""
    question_text = q.get("question", "")
    expected = q.get("answer", "")
    dimension = q.get("dimension", "?")

    result, _ = _score_or_fallback(
        question_text,
        expected,
        model_answer,
        dimension,
        q,
        judge_client,
        judge_model,
        retries,
        consecutive_failures=0,
        disable_thinking=disable_thinking,
        score_fn=score_fn,
    )

    issues = result.get("issues", []) or []
    if not model_answer and "target failure" not in issues:
        issues.append("target failure")

    qid = q.get("id", "?")
    # Pass through dual judge metadata fields for _compute_dual_judge_stats
    dj_fields = {}
    for k in ("judge1_score", "judge2_score", "avg_score", "score_diff",
              "disputed", "arbitrated"):
        if k in result:
            dj_fields[k] = result[k]
    return {
        "id": qid,
        "dimension": dimension,
        "difficulty_level": q.get("difficulty_level", "?"),
        "score": result.get("score", 0.0),
        "reason": result.get("reason", ""),
        "issues": issues,
        "model_answer": model_answer,
        "expected_answer": expected,
        "scoring_mode": "llm_judge",
        "scorer_version": SCORER_VERSION,
        **dj_fields,
        **(
            {"rubric_details": result["rubric_details"]}
            if "rubric_details" in result
            else {}
        ),
    }


# ---------------------------------------------------------------------------
# Progress saving
# ---------------------------------------------------------------------------
def _save_progress(
    output_path: Path,
    status_path: Path,
    records: list,
    adapter_name: str,
    judge_model: str,
    total: int,
    session_start: float,
    initial_completed: int,
    started_at: str,
    consecutive_failures: int,
    qid: str,
):
    """Save intermediate results and status."""
    if not output_path:
        return
    elapsed_sec = max(time.time() - session_start, 0.001)
    session_completed = len(records) - initial_completed
    rate = session_completed / (elapsed_sec / 60) if session_completed > 0 else 0.0
    remaining = max(total - len(records), 0)
    eta_min = remaining / rate if rate > 0 else None

    partial_report = build_model_report(records, adapter_name, judge_model)
    partial_report["complete"] = len(records) >= total
    partial_report["started_at"] = started_at
    partial_report["run_metrics"] = {
        "session_completed": session_completed,
        "elapsed_sec": round(elapsed_sec, 2),
        "rate_q_per_min": round(rate, 4),
        "eta_min": round(eta_min, 2) if eta_min is not None else None,
        "eta_at": format_eta(eta_min),
    }
    write_json_atomic(output_path, partial_report)

    if status_path:
        item = {
            "model": adapter_name,
            "judge_model": judge_model,
            "status": "running" if len(records) < total else "completed",
            "completed": len(records),
            "total": total,
            "percent": round(len(records) / total * 100, 2) if total else 100.0,
            "remaining": remaining,
            "session_completed": session_completed,
            "elapsed_sec": round(elapsed_sec, 2),
            "rate_q_per_min": round(rate, 4),
            "eta_min": round(eta_min, 2) if eta_min is not None else None,
            "eta_at": format_eta(eta_min),
            "last_id": qid,
            "consecutive_failures": consecutive_failures,
            "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        write_json_atomic(status_path, item)


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------
def _select_eval_questions(
    data: dict,
    *,
    fixed_questions_path: Path | None = None,
    limit: int = 0,
    balanced_mini: bool = False,
) -> list[dict]:
    """Select questions from data or a fixed question file."""
    if fixed_questions_path:
        if not fixed_questions_path.is_file():
            raise FileNotFoundError(f"Fixed questions file not found: {fixed_questions_path}")
        if limit > 0:
            print(f"[WARNING] --limit={limit} ignored when --fixed-questions is used", flush=True)
        if balanced_mini:
            print("[WARNING] --balanced-mini ignored when --fixed-questions is used", flush=True)
        try:
            questions = load_json(fixed_questions_path)
        except Exception as e:
            raise ValueError(f"Failed to load fixed questions from {fixed_questions_path}: {e}") from e
        if isinstance(questions, dict):
            questions = questions.get("questions", questions.get("records", questions))
        if not isinstance(questions, list):
            raise ValueError(f"Fixed questions file must contain a list, got {type(questions).__name__}")
        print(f"Using fixed question set: {len(questions)} questions from {fixed_questions_path}", flush=True)
        return questions
    return select_questions(data.get("questions", []), limit=limit, balanced_mini=balanced_mini)


def _init_adapter_eval(data, fixed_questions_path, limit, balanced_mini,
                       output_path, resume, retry_failed):
    """Initialize adapter evaluation."""
    questions = _select_eval_questions(
        data,
        fixed_questions_path=fixed_questions_path,
        limit=limit,
        balanced_mini=balanced_mini,
    )
    selected_ids = [q.get("id", "?") for q in questions]
    records = (
        load_existing_records(output_path, selected_ids, retry_failed=retry_failed)
        if (resume and output_path)
        else []
    )
    completed_ids = {r.get("id") for r in records}
    total = len(questions)
    started_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    session_start = time.time()
    initial_completed = len(records)
    output = Path(output_path) if output_path else None
    status_path = (
        output.with_suffix(output.suffix + ".status.json") if output else None
    )
    if records:
        print(f"Resume enabled: loaded {len(records)} existing records", flush=True)
    return questions, records, completed_ids, total, started_at, session_start, initial_completed, output, status_path


def evaluate_adapter(
    data: dict,
    model,
    tokenizer,
    judge_client,
    judge_model: str,
    adapter_name: str,
    *,
    limit: int = 0,
    balanced_mini: bool = False,
    fixed_questions_path: Path | None = None,
    output_path: Path | None = None,
    resume: bool = False,
    retry_failed: bool = False,
    retries: int = 3,
    max_consecutive_failures: int = 5,
    parallel: int = 1,
    disable_thinking: bool = False,
    max_answer_tokens: int = 2048,
    dual_judge: bool = False,
    judge2_client=None,
    judge2_model: str | None = None,
    arb_client=None,
    arb_model: str | None = None,
    answers_cache: Path | None = None,
    all_dimensions_dual: bool = False,
    external_score_fn=None,
    system_prompt: str | None = None,
) -> dict:
    """Run LLM-as-Judge evaluation on a local PeftModel adapter.

    Args:
        external_score_fn: If provided, use this scoring function instead of the
            default dual-judge dispatch. Used for medical domain evaluation.
        system_prompt: Override target model system prompt (default None = built-in
            TARGET_SYSTEM_PROMPT). 补充实验 2×2 格子③④告知措辞注入（决策报告 §3.4）。
    """
    questions, records, completed_ids, total, started_at, session_start, initial_completed, output, status_path = \
        _init_adapter_eval(data, fixed_questions_path, limit, balanced_mini,
                           output_path, resume, retry_failed)
    consecutive_failures = 0
    stopped_early = False

    # Build scoring function: external (medical) or dual-judge dispatch (control)
    if external_score_fn is not None:
        score_fn = external_score_fn
    else:
        score_fn = _make_dual_judge_score_fn(
            dual_judge, judge2_client, judge2_model, arb_client, arb_model,
            all_dimensions=all_dimensions_dual,
        )

    # ── Parallel judge path (API I/O bound) ──
    if parallel > 1:
        return _evaluate_parallel(
            questions,
            completed_ids,
            records,
            model,
            tokenizer,
            judge_client,
            judge_model,
            adapter_name,
            output_path=output,
            status_path=status_path,
            total=total,
            started_at=started_at,
            session_start=session_start,
            initial_completed=initial_completed,
            retries=retries,
            max_consecutive_failures=max_consecutive_failures,
            parallel=parallel,
            disable_thinking=disable_thinking,
            max_answer_tokens=max_answer_tokens,
            score_fn=score_fn,
            answers_cache=answers_cache,
            system_prompt=system_prompt,
        )

    # ── Serial path (existing) ──
    # Load cached answers for serial path (same logic as _evaluate_parallel)
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

    try:
        for i, q in enumerate(questions):
            qid = q.get("id", "?")
            if qid in completed_ids:
                print(
                    f"[{i + 1}/{total}] {qid} already completed; skipping",
                    flush=True,
                )
                continue

            dimension = q.get("dimension", "?")
            difficulty = q.get("difficulty_level", "?")
            question_text = q.get("question", "")
            expected = q.get("answer", "")

            # Use cached answer if available, skip GPU inference
            if qid in cached_answers:
                model_answer = cached_answers[qid]
                print(
                    f"[{i + 1}/{total}] {qid} ({dimension}, {difficulty}) "
                    f"- from cache",
                    flush=True,
                )
            else:
                print(
                    f"[{i + 1}/{total}] {qid} ({dimension}, {difficulty}) "
                    f"- generating...",
                    flush=True,
                )
                # --- Local inference ---
                try:
                    gen_call = partial(
                        call_target_model_local, question_text, model, tokenizer,
                        max_new_tokens=max_answer_tokens,
                        strip_think=True,
                        system_prompt=system_prompt,
                    )
                    model_answer = call_with_retries(
                        "target", gen_call, retries=retries
                    )
                except Exception as e:
                    print(f"  ERROR generating after retries: {e}", flush=True)
                    model_answer = ""

            # --- Judge scoring ---
            result, consecutive_failures = _score_or_fallback(
                question_text,
                expected,
                model_answer,
                dimension,
                q,
                judge_client,
                judge_model,
                retries,
                consecutive_failures,
                disable_thinking=disable_thinking,
                score_fn=score_fn,
            )

            print(f"score={result.get('score', 'N/A')}", flush=True)

            issues = result.get("issues", []) or []
            if not model_answer and "target failure" not in issues:
                issues.append("target failure")

            record = {
                "id": qid,
                "dimension": dimension,
                "difficulty_level": difficulty,
                "score": result.get("score", 0.0),
                "reason": result.get("reason", ""),
                "issues": issues,
                "model_answer": model_answer,
                "expected_answer": expected,
                "scoring_mode": "llm_judge",
                "scorer_version": SCORER_VERSION,
            }
            if "rubric_details" in result:
                record["rubric_details"] = result["rubric_details"]
            records.append(record)
            completed_ids.add(qid)

            # Save progress every 5 questions
            if len(records) % 5 == 0:
                _save_progress(
                    output,
                    status_path,
                    records,
                    adapter_name,
                    judge_model,
                    total,
                    session_start,
                    initial_completed,
                    started_at,
                    consecutive_failures,
                    qid,
                )

            import torch
            torch.cuda.empty_cache()

            if consecutive_failures >= max_consecutive_failures:
                stopped_early = True
                print(
                    f"Stopping early after {consecutive_failures} "
                    f"consecutive failures. Use --resume --retry-failed "
                    f"after fixing the issue.",
                    flush=True,
                )
                break
    except (Exception, KeyboardInterrupt):
        if records:
            _save_progress(
                output,
                status_path,
                records,
                adapter_name,
                judge_model,
                total,
                session_start,
                initial_completed,
                started_at,
                consecutive_failures,
                qid,
            )
        raise

    report = build_model_report(records, adapter_name, judge_model)
    report["complete"] = len(records) >= total
    if not report["complete"]:
        reason = (
            "max_consecutive_failures" if stopped_early
            else "not_all_questions_completed"
        )
        report["incomplete_reason"] = reason
        report["completed"] = len(records)
        report["total"] = total
    report["started_at"] = started_at
    return report


# ---------------------------------------------------------------------------
# CLI (delegated to eval_cli.py)
# ---------------------------------------------------------------------------
def parse_args(argv: list[str] | None = None):
    """Backward-compatible alias — delegates to eval_cli.parse_args."""
    from mineru_eval.eval_cli import parse_args as _parse_args
    return _parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Backward-compatible alias — delegates to eval_cli.main."""
    from mineru_eval.eval_cli import main as _main
    return _main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
