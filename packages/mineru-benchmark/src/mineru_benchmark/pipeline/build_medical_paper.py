"""Medical paper mode — serial and concurrent generation from medical papers.

Extracted from build_medical.py to reduce cyclomatic complexity and separate
paper-mode orchestration from the main CLI entry point.
"""

import logging
import random
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, wait as _fut_wait, FIRST_COMPLETED
from dataclasses import dataclass
from pathlib import Path

from mineru_benchmark.pipeline._build_common import (
    DEFAULT_SEED,
    _paper_id, load_json, write_json_atomic,
    read_paper_text, _paper_tier, _paper_step,
    build_paper_source, _finalize_gen_output, _gen_reset_if_needed, _gen_checkpoint_and_status,
    write_outputs, _make_diag_ctx, _record_quality, _make_single_question,
)
from mineru_benchmark.pipeline.medical_lint import lint_medical_batch
from mineru_generator.api import call_llm
from mineru_generator.medical_distribution import (
    pick_next_medical_dim_and_diff,
    precompute_medical_dim_plan,
)
from mineru_generator.medical_prompts import (
    build_medical_system_prompt, build_medical_user_prompt,
)
from mineru_generator.quality import QualityMonitor


@dataclass
class MedicalTaskSlot:
    """Pre-allocated slot for one medical benchmark question."""
    global_index: int
    paper: dict
    paper_text: str
    dim: str  # identify/model/transfer
    difficulty: str  # L1/L2/L3/L4 from dim_plan
    text_offset: int = 0


# ---------------------------------------------------------------------------
# Medical paper mode: serial
# ---------------------------------------------------------------------------


def _paper_mode_serial(args, papers, corpus_dir, output_path, review_path,
                       diag_path, start_index, resume_questions, total_limit,
                       provider, model, api_key, registry):
    """Serial medical paper-based generation with quality monitoring."""
    log_path = output_path.with_suffix(".log")
    logging.basicConfig(filename=str(log_path), level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s", encoding="utf-8")
    logger = logging.getLogger(__name__)
    logger.info(f"Medical paper mode: split={args.paper_split}, papers={len(papers)}, limit={total_limit}")

    rng = random.Random(args.seed)
    rng.shuffle(papers)

    questions = list(resume_questions) if resume_questions else []
    review_items = []
    index = max(args.start_index, start_index)
    paper_idx = 0
    consecutive_failures = 0
    start_time = time.time()
    token_usage = [0, 0]
    questions_since_reset = 0

    quality = QualityMonitor(window_size=15, consecutive_threshold=3)
    system_prompt = build_medical_system_prompt()

    # Initialize paper caps
    paper_caps = {}
    paper_steps = {}
    paper_idx_to_key = {}
    for idx, p in enumerate(papers):
        pid = _paper_id(p) or f"__no_id_{idx}__"
        paper_idx_to_key[idx] = pid
        paper_path = p.get("paper_path", "")
        full_path = corpus_dir / paper_path / "full_text.md" if paper_path else None
        txt_len = len(full_path.read_text(encoding="utf-8", errors="ignore")) if full_path and full_path.exists() else 0
        cap = _paper_tier(txt_len) if txt_len > 0 else 12
        paper_caps[pid] = cap
        paper_steps[pid] = _paper_step(txt_len, cap) if txt_len > 0 else 3000

    paper_q_counts = Counter()
    for q in questions:
        ps = q.get("paper_source", {})
        if ps.get("arxiv_id"):
            paper_q_counts[ps["arxiv_id"]] += 1

    while len(questions) < total_limit and paper_idx < len(papers) * 3:
        # Select next paper
        if paper_idx >= len(papers) * 3:
            break
        failure_threshold = min(max(20, total_limit // 5), len(papers))
        if consecutive_failures >= failure_threshold:
            print(f"[medical-paper] Too many failures ({consecutive_failures}), stopping.")
            break

        paper = min(papers, key=lambda p: paper_q_counts.get(paper_idx_to_key.get(papers.index(p), _paper_id(p)), 0))
        _pidx = papers.index(paper)
        pid = paper_idx_to_key.get(_pidx, _paper_id(paper))
        cap = paper_caps.get(pid, 12)
        if paper_q_counts.get(pid, 0) >= cap:
            remaining = [(i, p) for i, p in enumerate(papers)
                         if paper_q_counts.get(paper_idx_to_key.get(i, _paper_id(p)), 0)
                         < paper_caps.get(paper_idx_to_key.get(i, _paper_id(p)), 12) * 2]
            if not remaining:
                break
            _ri, paper = min(remaining, key=lambda t: paper_q_counts.get(paper_idx_to_key.get(t[0], _paper_id(t[1])), 0))
            pid = paper_idx_to_key.get(_ri, _paper_id(paper))
            cap = paper_caps.get(pid, 12)

        paper_path_str = paper.get("paper_path", "")
        q_index = paper_q_counts.get(pid, 0)
        step = paper_steps.get(pid, 3000)
        text_offset = q_index * step
        paper_text = read_paper_text(paper_path_str, corpus_dir, max_chars=6000, offset=text_offset)
        if not paper_text:
            paper_idx += 1
            consecutive_failures += 1
            continue

        paper_reg = registry.get(pid, paper)
        ps = build_paper_source(paper_reg, corpus_dir, text_offset=text_offset)
        dim, difficulty = pick_next_medical_dim_and_diff(questions, total_limit)
        user_prompt = build_medical_user_prompt(paper_text, pid, dim, difficulty=difficulty)

        print(f"[medical-paper] q{index}: dim={dim}, diff={difficulty}, paper={pid} "
              f"(#{paper_q_counts.get(pid, 0)+1}/{cap})", flush=True)
        logger.info(f"q{index}: dim={dim}, diff={difficulty}, paper={pid}")

        # Generate question
        diag_ctx_1 = _make_diag_ctx(index, provider, 1)
        q1 = call_llm(provider, api_key, system_prompt, user_prompt, model=model,
                      _usage_acc=token_usage, diag_path=diag_path, diag_ctx=diag_ctx_1)
        q1_ok = _record_quality(quality, q1)
        if not q1_ok:
            consecutive_failures += 1
            paper_idx += 1
            continue

        diag_ctx_2 = _make_diag_ctx(index, provider, 2)
        q2 = call_llm(provider, api_key, system_prompt, user_prompt, max_retries=2, model=model,
                      _usage_acc=token_usage, diag_path=diag_path, diag_ctx=diag_ctx_2)
        q2_ok = _record_quality(quality, q2)

        qid = f"CS-MED-{index:05d}"
        final_q, review_items = _make_single_question(
            qid, q1, q2, q2_ok, paper, dim, difficulty, provider, review_items)
        final_q["paper_source"] = ps
        final_q["source_ref"] = pid
        final_q["scoring_reason"] = q1.get("scoring_reason", "")
        final_q["scoring_issues"] = q1.get("scoring_issues", [])
        questions.append(final_q)
        index += 1
        paper_q_counts[pid] += 1
        paper_idx += 1
        consecutive_failures = 0
        questions_since_reset += 1

        questions_since_reset, system_prompt, quality = _gen_reset_if_needed(
            questions_since_reset, quality, args.reset_every, logger, system_prompt, len(questions),
            prompt_builder_fn=build_medical_system_prompt)

        if len(questions) % 10 == 0:
            qs = quality.summary()
            dim_keys = sorted(set(q.get("dimension", "") for q in questions)) or ["identify","model","transfer"]
            dim_counts = {k: 0 for k in dim_keys}
            for q in questions:
                d = q.get("dimension", "")
                if d in dim_counts:
                    dim_counts[d] += 1
            _gen_checkpoint_and_status(output_path, review_path, questions, review_items, total_limit,
                                        index, paper_idx, provider, model,
                                        start_time, consecutive_failures, token_usage, quality,
                                        questions_since_reset, dim_counts, logger)

    questions, review_items = _finalize_gen_output(
        output_path, review_path, questions, review_items, index, paper_idx,
        provider, model, start_time, total_limit, consecutive_failures,
        token_usage, quality, questions_since_reset)

    data = write_outputs(questions, output_path, review_path,
                         source=f"AI-generated via {provider}/{model} from medical papers ({args.paper_split})",
                         review_items=review_items)

    print(f"\n[medical-paper] Running medical lint on {len(questions)} questions...")
    lint_result = lint_medical_batch(questions)
    print(f"  Lint: {lint_result['passed']} passed, {lint_result['failed']} failed "
          f"({lint_result['pass_rate']:.1f}%)")
    source = f"AI-generated via {provider}/{model} from medical papers ({args.paper_split})"
    if lint_result['failed'] > 0:
        for idx in lint_result['flagged_indices']:
            if idx < len(questions):
                issues = lint_result['details'][idx].get('issues', [])
                questions[idx]['consistency_status'] = 'needs_review'
                questions[idx].setdefault('lint_issues', []).extend(issues)
        source = f"AI-generated via {provider}/{model} (linted, {args.paper_split})"
        print(f"  {lint_result['failed']} questions marked as needs_review after lint")

    write_outputs(questions, output_path, review_path, source=source, review_items=review_items)
    print(f"\nMedical Benchmark saved: {output_path}", flush=True)
    print(f"Manual review saved: {review_path}", flush=True)


# ---------------------------------------------------------------------------
# Medical paper mode: concurrent
# ---------------------------------------------------------------------------


def _worker_medical_paper_mode(
    worker_id: int,
    tasks: list[MedicalTaskSlot],
    provider: str,
    model: str,
    api_key: str,
    output_dir: Path,
    reset_every: int,
    diag_path: str | None,
) -> dict:
    """Generate medical questions from pre-allocated TaskSlots in a single worker thread."""
    import logging as _logging
    questions: list[dict] = []
    review_items: list[dict] = []
    consecutive_failures = 0
    start_time = time.time()
    token_usage = [0, 0]
    questions_since_reset = 0
    failed_count = 0

    quality = QualityMonitor(window_size=15, consecutive_threshold=3)
    system_prompt = build_medical_system_prompt()

    cp_path = output_dir / f"worker_{worker_id}.checkpoint.json"
    done_ids: set[str] = set()
    if cp_path.exists():
        cp_data = load_json(cp_path)
        questions = cp_data.get("questions", [])
        review_items = cp_data.get("review_items", [])
        token_usage = cp_data.get("token_usage", [0, 0])
        failed_count = cp_data.get("failed_count", 0)
        done_ids = {q["id"] for q in questions if "id" in q}
        print(f"[Medical-W{worker_id}] Resumed {len(questions)} questions from checkpoint", flush=True)

    failure_threshold = min(max(20, len(tasks) // 5), len(tasks))

    for task in tasks:
        ps = build_paper_source(task.paper, output_dir, text_offset=task.text_offset)
        pid = _paper_id(task.paper)
        qid = f"CS-MED-{task.global_index:05d}"

        if qid in done_ids:
            continue

        if consecutive_failures >= failure_threshold:
            print(f"[Medical-W{worker_id}] Too many failures ({consecutive_failures}), stopping.", flush=True)
            break

        dim = task.dim
        difficulty = task.difficulty
        paper_text = task.paper_text

        user_prompt = build_medical_user_prompt(paper_text, pid, dim, difficulty=difficulty)
        print(f"[Medical-W{worker_id}] q{task.global_index}: dim={dim}, diff={difficulty}, paper={pid}", flush=True)

        diag_ctx_1 = _make_diag_ctx(task.global_index, provider, 1)
        q1 = call_llm(provider, api_key, system_prompt, user_prompt, model=model,
                      _usage_acc=token_usage, diag_path=diag_path, diag_ctx=diag_ctx_1)
        q1_ok = _record_quality(quality, q1)

        if not q1_ok:
            consecutive_failures += 1
            failed_count += 1
            continue

        diag_ctx_2 = _make_diag_ctx(task.global_index, provider, 2)
        q2 = call_llm(provider, api_key, system_prompt, user_prompt, max_retries=2, model=model,
                      _usage_acc=token_usage, diag_path=diag_path, diag_ctx=diag_ctx_2)
        q2_ok = _record_quality(quality, q2)

        final_q, review_items = _make_single_question(
            qid, q1, q2, q2_ok, task.paper, dim, difficulty, provider, review_items)
        final_q["paper_source"] = ps
        final_q["source_ref"] = pid
        final_q["scoring_reason"] = q1.get("scoring_reason", "")
        final_q["scoring_issues"] = q1.get("scoring_issues", [])
        questions.append(final_q)

        consecutive_failures = 0
        questions_since_reset += 1

        questions_since_reset, system_prompt, quality = _gen_reset_if_needed(
            questions_since_reset, quality, reset_every,
            _logging.getLogger(__name__), system_prompt, len(questions),
            prompt_builder_fn=build_medical_system_prompt)

        if len(questions) % 10 == 0:
            qs = quality.summary()
            elapsed = time.time() - start_time
            rate = len(questions) / elapsed * 60 if elapsed > 0 else 0
            write_json_atomic(cp_path, {
                "worker_id": worker_id,
                "questions": questions,
                "review_items": review_items,
                "token_usage": token_usage,
                "failed_count": failed_count,
                "elapsed_sec": round(elapsed, 0),
                "rate_qpm": round(rate, 1),
                "quality": qs,
            })
            dim_keys = sorted(set(q.get("dimension", "") for q in questions)) or ["identify","model","transfer"]
            dim_counts = {k: 0 for k in dim_keys}
            for q in questions:
                d = q.get("dimension", "")
                if d in dim_counts:
                    dim_counts[d] += 1
            dc_str = " ".join(f"{d}={dim_counts[d]}" for d in dim_keys)
            eta = (len(tasks) - len(questions)) / max(rate, 0.01) if rate > 0 else 0
            print(f"  [Medical-W{worker_id}] cp ({len(questions)}/{len(tasks)}) {rate:.1f}qpm ETA:{eta:.0f}min | {dc_str}", flush=True)

    write_json_atomic(cp_path, {
        "worker_id": worker_id,
        "questions": questions,
        "review_items": review_items,
        "token_usage": token_usage,
        "failed_count": failed_count,
        "completed": True,
    })

    return {
        "worker_id": worker_id,
        "questions": questions,
        "review_items": review_items,
        "token_usage": token_usage,
        "failed_count": failed_count,
    }


def _paper_mode_concurrent(args, papers, corpus_dir, output_path, review_path,
                           diag_path, provider, model, api_key, total_limit):
    """Concurrent medical paper-based generation with worker pool."""
    print(f"[medical-paper] Concurrent mode: {args.workers} workers")
    dim_plan = precompute_medical_dim_plan(total_limit)

    # Assign tasks
    worker_tasks = _assign_medical_tasks(papers, dim_plan, args.workers, corpus_dir,
                                          max_per_paper=12, seed=args.seed, start_index=args.start_index)
    total_slots = sum(len(t) for t in worker_tasks)
    print(f"[medical-paper] {total_slots} slots across {args.workers} workers")

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = []
        wid_for_future = {}
        for wid, tasks in enumerate(worker_tasks):
            if not tasks:
                continue
            worker_diag = str(output_path.parent / f"diag_w{wid}.logl") if diag_path else None
            future = executor.submit(
                _worker_medical_paper_mode,
                wid, tasks, provider, model, api_key,
                output_path.parent, args.reset_every, worker_diag)
            futures.append(future)
            wid_for_future[future] = wid

        all_questions = []
        all_review = []
        global_start = time.time()
        pending = set(futures)
        last_progress_print = 0.0

        while pending:
            # Wait up to 30 seconds for any future to complete
            done, pending = _fut_wait(pending, timeout=30, return_when=FIRST_COMPLETED)

            # Poll checkpoint files for global progress
            global_done = 0
            global_rate = 0.0
            for wid in range(args.workers):
                cp = output_path.parent / f"worker_{wid}.checkpoint.json"
                if cp.exists():
                    try:
                        cp_data = load_json(cp)
                        global_done += len(cp_data.get("questions", []))
                        if cp_data.get("rate_qpm", 0) > 0:
                            global_rate += cp_data["rate_qpm"]
                    except Exception:
                        pass

            elapsed = time.time() - global_start
            if global_rate <= 0 and elapsed > 0:
                global_rate = global_done / (elapsed / 60) if global_done > 0 else 0.0
            remaining = max(total_slots - global_done, 0)
            eta_min = remaining / global_rate if global_rate > 0 else 0

            # Print progress at most every 60 seconds
            now = time.time()
            if now - last_progress_print >= 60:
                print(f"[medical-paper] Global: {global_done}/{total_slots} "
                      f"({global_rate:.1f} qpm) ETA:{eta_min:.0f}min | "
                      f"{len(pending)} worker(s) pending", flush=True)
                last_progress_print = now

            # Collect results from completed futures
            for future in done:
                wid = wid_for_future.get(future, -1)
                try:
                    result = future.result()
                    n_q = len(result.get("questions", []))
                    print(f"[medical-paper] Worker {result.get('worker_id', wid)} finished: {n_q} questions", flush=True)
                    all_questions.extend(result["questions"])
                    all_review.extend(result.get("review_items", []))
                except Exception as e:
                    import traceback
                    print(f"[medical-paper] Worker {wid} raised: {e}", flush=True)
                    traceback.print_exc()
                    cp = output_path.parent / f"worker_{wid}.checkpoint.json"
                    if cp.exists():
                        try:
                            cp_data = load_json(cp)
                            recovered = cp_data.get("questions", [])
                            all_questions.extend(recovered)
                            all_review.extend(cp_data.get("review_items", []))
                            print(f"[medical-paper] Recovered {len(recovered)} questions from crashed worker {wid}", flush=True)
                        except Exception as load_err:
                            print(f"[medical-paper] Failed to load checkpoint for worker {wid}: {load_err}", flush=True)

    all_questions.sort(key=lambda q: q.get("id", ""))

    print(f"\n[medical-paper] Running medical lint on {len(all_questions)} questions...")
    lint_result = lint_medical_batch(all_questions)
    print(f"  Lint: {lint_result['passed']} passed, {lint_result['failed']} failed "
          f"({lint_result['pass_rate']:.1f}%)")
    if lint_result['failed'] > 0:
        for idx in lint_result['flagged_indices']:
            if idx < len(all_questions):
                issues = lint_result['details'][idx].get('issues', [])
                all_questions[idx]['consistency_status'] = 'needs_review'
                all_questions[idx].setdefault('lint_issues', []).extend(issues)

    source = f"AI-generated via {provider}/{model} ({args.workers} workers, {args.paper_split})"
    write_outputs(all_questions, output_path, review_path, source=source, review_items=all_review)

    for wid in range(args.workers):
        cp = output_path.parent / f"worker_{wid}.checkpoint.json"
        if cp.exists():
            cp.unlink()

    print(f"\nMedical Benchmark saved: {output_path}", flush=True)
    print(f"Manual review saved: {review_path}", flush=True)


def _assign_medical_tasks(papers, dim_plan, n_workers, corpus_dir, max_per_paper=12, seed=DEFAULT_SEED, start_index=1):
    """Distribute medical papers and dimension slots across N workers."""
    rng = random.Random(seed)
    rng.shuffle(papers)

    paper_ids = []
    id_to_paper = {}
    paper_caps = {}
    paper_steps = {}
    for idx, p in enumerate(papers):
        key = _paper_id(p) or f"__no_id_{idx}__"
        while key in id_to_paper:
            key = f"__no_medical_{idx}___{len(id_to_paper)}"
        paper_ids.append(key)
        id_to_paper[key] = p
        paper_path_str = p.get("paper_path", "")
        full_path = corpus_dir / paper_path_str / "full_text.md" if paper_path_str else None
        txt_len = len(full_path.read_text(encoding="utf-8", errors="ignore")) if full_path and full_path.exists() else 0
        cap = _paper_tier(txt_len) if txt_len > 0 else max_per_paper
        paper_caps[key] = cap
        paper_steps[key] = _paper_step(txt_len, cap)

    paper_counts = {k: 0 for k in paper_ids}
    slots = []
    slot_idx = 0
    paper_idx = 0
    max_cycles = sum(paper_caps.values()) * 2

    while slot_idx < len(dim_plan) and paper_idx < max_cycles:
        paper_key = min(paper_counts, key=lambda k: paper_counts[k])
        cap = paper_caps.get(paper_key, max_per_paper)
        if paper_counts[paper_key] >= cap:
            remaining = [k for k in paper_counts if paper_counts[k] < paper_caps.get(k, max_per_paper) * 2]
            if not remaining:
                break
            paper_key = min(remaining, key=lambda k: paper_counts[k])
            cap = paper_caps.get(paper_key, max_per_paper)

        paper = id_to_paper.get(paper_key, {})
        paper_path_str = paper.get("paper_path", "")
        q_index = paper_counts[paper_key]
        step = paper_steps.get(paper_key, 3000)
        text_offset = q_index * step
        paper_text = read_paper_text(paper_path_str, corpus_dir, max_chars=6000, offset=text_offset)
        if not paper_text:
            paper_idx += 1
            paper_counts[paper_key] += cap
            continue

        dim, difficulty = dim_plan[slot_idx]
        slots.append(MedicalTaskSlot(
            global_index=start_index + slot_idx,
            paper=dict(paper),
            paper_text=paper_text,
            text_offset=text_offset,
            dim=dim,
            difficulty=difficulty,
        ))
        paper_counts[paper_key] += 1
        slot_idx += 1
        paper_idx += 1

    worker_tasks = [[] for _ in range(n_workers)]
    for i, slot in enumerate(slots):
        worker_tasks[i % n_workers].append(slot)
    return worker_tasks
