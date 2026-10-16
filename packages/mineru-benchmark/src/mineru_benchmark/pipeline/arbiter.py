"""Arbitration: resolve needs_review benchmark items via DeepSeek.
Supports checkpoint/resume: auto-detects arbitrate.checkpoint.json for partial progress.
"""
import json
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

from mineru_generator.distribution import DIMENSION_CYCLE
from mineru_generator.api import call_openai_direct
from mineru_generator.prompts import ARBITER_SYSTEM_PROMPT

CHECKPOINT_INTERVAL = 50
VALID_CHOICES = {"round_1", "round_2", "both_equivalent", "both_wrong"}
ARBITER_MAX_TOKENS = 1024


def _checkpoint_path(output_path):
    return Path(output_path).with_suffix(".arbitrate.checkpoint.json")


def _load_checkpoint(cp_path):
    if cp_path.exists():
        data = json.loads(cp_path.read_text(encoding="utf-8"))
        questions_map = data.get("questions_map", {})
        if not isinstance(questions_map, dict):
            questions_map = {}
        return set(data.get("done_ids", [])), questions_map, set(data.get("failed_ids", []))
    return set(), {}, set()


def _write_json_atomic(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.replace(path)


def _save_checkpoint(cp_path, done_ids, questions_map, failed_ids=None):
    failed_ids = failed_ids or set()
    _write_json_atomic(
        cp_path,
        {
            "done_ids": sorted(done_ids),
            "failed_ids": sorted(failed_ids),
            "questions_map": questions_map,
        },
    )
    print(f"  [Arbitrate] Checkpoint saved ({len(done_ids)} done, {len(failed_ids)} failed)", flush=True)


def _cleanup_checkpoint(cp_path):
    if cp_path.exists():
        cp_path.unlink()


def _diag_path(output_path):
    out = Path(output_path)
    return str(out.with_suffix(".arbitrate.diag.logl"))


def _ensure_runtime_dependencies():
    try:
        import openai  # noqa: F401
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Missing Python dependency 'openai' in the active environment. "
            "Install it with: .venv/bin/python -m pip install openai"
        ) from exc


def _backup_path(path):
    path = Path(path)
    candidate = path.with_suffix(".pre_arbitrate.bak.json")
    if not candidate.exists():
        return candidate
    index = 1
    while True:
        candidate = path.with_suffix(f".pre_arbitrate.{index}.bak.json")
        if not candidate.exists():
            return candidate
        index += 1


def _build_dimension_counts(questions):
    counts = {dimension: 0 for dimension in DIMENSION_CYCLE}
    for question in questions:
        dimension = question.get("dimension")
        if dimension in counts:
            counts[dimension] += 1
    return counts


def _apply_c_pair_discards(questions_map):
    """Discard both C siblings when either side is marked discarded."""
    discarded_ids = {
        qid for qid, question in questions_map.items()
        if question.get("consistency_status") == "reviewed_discarded"
    }
    changed = True
    while changed:
        changed = False
        for qid, question in questions_map.items():
            if question.get("dimension") != "C":
                continue
            sibling_id = question.get("sibling_id")
            if qid in discarded_ids or sibling_id in discarded_ids:
                if qid not in discarded_ids:
                    discarded_ids.add(qid)
                    question["consistency_status"] = "reviewed_discarded"
                    changed = True
                sibling = questions_map.get(sibling_id)
                if sibling and sibling_id not in discarded_ids:
                    discarded_ids.add(sibling_id)
                    sibling["consistency_status"] = "reviewed_discarded"
                    changed = True
    return discarded_ids


def _eligible_items(items, questions_map, done_ids):
    skipped = 0
    pending = []
    for item in items:
        qid = item.get("id")
        if not qid:
            print("[Arbitrate] Skipping malformed review item without id.", flush=True)
            skipped += 1
            continue
        if qid in done_ids:
            continue
        if qid not in questions_map:
            print(f"[Arbitrate] Skipping review item for missing question id: {qid}", flush=True)
            done_ids.add(qid)
            skipped += 1
            continue
        if questions_map[qid].get("consistency_status") != "needs_review":
            done_ids.add(qid)
            skipped += 1
            continue
        pending.append(item)
    return pending, skipped


def _call_arbiter(item, output_path, api_key, max_tokens=ARBITER_MAX_TOKENS):
    qid = item["id"]
    question = item.get("question", "")
    a1 = item.get("round_1_answer", "")
    a2 = item.get("round_2_answer", "")

    user_prompt = f"问题是：{question}\n\nRound 1 答案：{a1}\n\nRound 2 答案：{a2}"
    result = call_openai_direct(
        api_key,
        ARBITER_SYSTEM_PROMPT,
        user_prompt,
        timeout=30,
        diag_path=_diag_path(output_path),
        diag_ctx={"q_index": qid, "provider": "deepseek", "round": "arbitrate", "schema": "arbiter"},
        max_tokens=max_tokens,
    )
    choice = result.get("choice") if isinstance(result, dict) else None
    if choice not in VALID_CHOICES:
        raise RuntimeError(f"Arbitration failed for {qid}: invalid or missing choice.")
    return qid, choice


def _apply_choice(question, choice):
    if choice == "both_equivalent":
        question["consistency_status"] = "auto_passed"
    elif choice in ("round_1", "round_2"):
        question["consistency_status"] = "reviewed_kept"
    elif choice == "both_wrong":
        question["consistency_status"] = "reviewed_discarded"


def _finalize_arbitration(questions_map, output_path, benchmark_data, cp, done_ids, failed_ids, checkpoint_failed_ids):
    """应用 C-pair 丢弃规则，保存最终结果并清理 checkpoint。"""
    discarded_ids = _apply_c_pair_discards(questions_map)
    kept_questions = [
        q for q in questions_map.values()
        if q.get("id") not in discarded_ids and q.get("consistency_status") != "reviewed_discarded"
    ]
    discarded_count = len(questions_map) - len(kept_questions)

    benchmark_data["questions"] = kept_questions
    benchmark_data["meta"]["total_questions"] = len(kept_questions)
    benchmark_data["meta"]["dimensions"] = _build_dimension_counts(kept_questions)

    out = Path(output_path)
    if out.exists():
        backup = _backup_path(out)
        out.rename(backup)
        print(f"  [Arbitrate] Pre-arbitrate backup: {backup.name}", flush=True)

    _write_json_atomic(output_path, benchmark_data)

    passed = sum(1 for q in kept_questions if q.get("consistency_status") == "auto_passed")
    kept = sum(1 for q in kept_questions if q.get("consistency_status") == "reviewed_kept")
    print(f"[Arbitrate] Done. auto_passed={passed}, reviewed_kept={kept}, discarded={discarded_count}, final={len(kept_questions)}", flush=True)

    unresolved_failed_ids = failed_ids | checkpoint_failed_ids
    if unresolved_failed_ids:
        print(
            f"[Arbitrate] WARNING: {len(unresolved_failed_ids)} item(s) still need review after API failures: "
            f"{', '.join(sorted(unresolved_failed_ids)[:10])}",
            flush=True,
        )
    if discarded_count > 0:
        print(f"[Arbitrate] Use --supplement {discarded_count} to refill.", flush=True)

    if unresolved_failed_ids:
        _save_checkpoint(cp, done_ids, questions_map, unresolved_failed_ids)
    else:
        _cleanup_checkpoint(cp)


def _run_arbitration_loop(pending_items, output_path, api_key, max_tokens, workers, questions_map, done_ids, failed_ids, checkpoint_failed_ids, cp, processed):
    """Run the multi-threaded arbitration loop with checkpoint support."""
    item_iter = iter(pending_items)
    futures = {}

    with ThreadPoolExecutor(max_workers=workers) as executor:
        def submit_next():
            try:
                item = next(item_iter)
            except StopIteration:
                return False
            future = executor.submit(_call_arbiter, item, output_path, api_key, max_tokens)
            futures[future] = item.get("id")
            return True

        for _ in range(workers):
            if not submit_next():
                break

        while futures:
            done, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in done:
                qid = futures.pop(future)
                try:
                    result_qid, choice = future.result()
                except Exception as exc:
                    failed_ids.add(qid)
                    print(
                        f"[Arbitrate] WARNING: {qid} failed after retries: {exc}. "
                        "Leaving it as needs_review and continuing.",
                        flush=True,
                    )
                    submit_next()
                    continue

                _apply_choice(questions_map[result_qid], choice)
                done_ids.add(result_qid)
                failed_ids.discard(result_qid)
                checkpoint_failed_ids.discard(result_qid)
                processed += 1

                if processed % CHECKPOINT_INTERVAL == 0:
                    _save_checkpoint(cp, done_ids, questions_map, failed_ids | checkpoint_failed_ids)
                submit_next()

    return done_ids, failed_ids, checkpoint_failed_ids



def arbitrate_questions(benchmark_path, review_path, output_path, api_key, workers=3, max_tokens=ARBITER_MAX_TOKENS):
    _ensure_runtime_dependencies()
    cp = _checkpoint_path(output_path)
    done_ids, checkpoint_map, checkpoint_failed_ids = _load_checkpoint(cp)
    is_resume = bool(done_ids)
    failed_ids = set()

    print(f"[Arbitrate] Reading {review_path}...", flush=True)
    with open(review_path, "r", encoding="utf-8") as f:
        review_data = json.load(f)
    with open(benchmark_path, "r", encoding="utf-8") as f:
        benchmark_data = json.load(f)

    items = review_data.get("items", [])
    if not items:
        print("[Arbitrate] No items to arbitrate.", flush=True)
        return

    workers = max(1, int(workers or 1))
    if is_resume:
        print(f"[Arbitrate] Resuming from checkpoint ({len(done_ids)} already done, {len(items) - len(done_ids)} remaining, workers={workers}, max_tokens={max_tokens})...", flush=True)
    else:
        print(f"[Arbitrate] Arbitrating {len(items)} items via DeepSeek (workers={workers}, max_tokens={max_tokens})...", flush=True)

    questions_map = {
        q["id"]: q for q in benchmark_data["questions"]
        if isinstance(q, dict) and q.get("id")
    }
    if checkpoint_map:
        questions_map.update({
            qid: question for qid, question in checkpoint_map.items()
            if isinstance(question, dict)
        })

    pending_items, skipped = _eligible_items(items, questions_map, done_ids)
    processed = len(done_ids)
    if skipped:
        _save_checkpoint(cp, done_ids, questions_map, failed_ids)
    if pending_items:
        print(f"[Arbitrate] Pending API calls: {len(pending_items)}", flush=True)

    done_ids, failed_ids, checkpoint_failed_ids = _run_arbitration_loop(
        pending_items, output_path, api_key, max_tokens, workers,
        questions_map, done_ids, failed_ids, checkpoint_failed_ids, cp, processed,
    )

    _finalize_arbitration(questions_map, output_path, benchmark_data, cp, done_ids, failed_ids, checkpoint_failed_ids)
