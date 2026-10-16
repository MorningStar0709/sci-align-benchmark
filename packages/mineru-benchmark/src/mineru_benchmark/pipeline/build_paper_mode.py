"""Control domain paper-based benchmark generation.

Extracted from build_benchmark.py to separate paper mode logic from chunk mode.
Contains mock generation, serial/concurrent API generation, and paper selection helpers.
"""
import random
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from mineru_benchmark.pipeline._build_common import (
    ROOT, TaskSlot,
    _paper_id, load_json, write_json_atomic,
    read_paper_text, _paper_tier, _paper_step,
    build_paper_source, _finalize_gen_output, _gen_reset_if_needed, _gen_checkpoint_and_status,
    write_outputs, _make_diag_ctx, _record_quality, _make_single_question,
    make_question_from_llm, default_rubric,
)
from mineru_benchmark.pipeline.answer_lint import lint_batch
from mineru_generator.api import call_llm
from mineru_generator.distribution import (
    DIMENSION_CYCLE, DIFFICULTY_BY_DIMENSION,
    pick_next_dim_and_diff, precompute_dim_plan,
)
from mineru_generator.prompts import build_system_prompt, build_user_prompt
from mineru_generator.quality import QualityMonitor


# ---------------------------------------------------------------------------
# Mock generation
# ---------------------------------------------------------------------------


def _paper_mock_generate(papers, limit, start_index, registry_path, corpus_dir):
    """Generate mock questions from papers without API calls."""
    import mineru_benchmark.pipeline._build_common as _common
    rng = random.Random(_common.RUN_SEED)
    rng.shuffle(papers)

    registry = {}
    if registry_path and registry_path.exists():
        reg_data = load_json(registry_path)
        for p in reg_data.get("papers", []):
            registry[_paper_id(p)] = p

    questions = []
    index = start_index
    paper_idx = 0
    dim_cycle = list(DIMENSION_CYCLE)
    paper_q_counts = Counter()
    max_per_paper = 12

    while len(questions) < limit:
        if paper_idx >= len(papers) * 3:
            break
        paper = min(papers, key=lambda p: paper_q_counts.get(_paper_id(p), 0))
        pid = _paper_id(paper)
        if paper_q_counts.get(pid, 0) >= max_per_paper:
            remaining = [p for p in papers if paper_q_counts.get(_paper_id(p), 0) < max_per_paper * 2]
            if not remaining:
                break
            paper = min(remaining, key=lambda p: paper_q_counts.get(_paper_id(p), 0))
            pid = _paper_id(paper)

        dim = dim_cycle[len(questions) % len(dim_cycle)]
        difficulty_opts = DIFFICULTY_BY_DIMENSION[dim]
        difficulty = difficulty_opts[len(questions) % len(difficulty_opts)]

        paper_reg = registry.get(pid, paper)
        ps = build_paper_source(paper_reg, corpus_dir)

        qid = f"CS-EVO-{index:05d}"
        q = {
            "id": qid,
            "dimension": dim,
            "difficulty_level": difficulty,
            "control_category": [paper.get("control_category", "classical")],
            "question": f"基于论文 '{paper.get('title', pid)}' 中的方法，分析{dim}维度的相关问题。",
            "answer": f"该论文主要研究了{paper.get('title', '相关主题')}。答案应涵盖建模、分析和验证。",
            "reasoning_steps": ["识别论文主题", "抽取核心方法", "分析理论依据"],
            "source_ref": pid,
            "model_source": "mock",
            "sensitivity_dimension": "parameter" if dim == "C" else None,
            "sibling_id": None,
            "rubric": default_rubric(paper.get("title", "相关")) if dim == "D" else None,
            "consistency_status": "auto_passed",
            "paper_source": ps,
            "scoring_reason": f"Mock题：考察{pid}论文中{dim}维度的理解",
            "scoring_issues": ["Mock数据，非实际API生成"],
        }
        questions.append(q)
        paper_q_counts[pid] += 1
        index += 1
        paper_idx += 1

    return questions


# ---------------------------------------------------------------------------
# C-dimension pair processing (paper mode)
# ---------------------------------------------------------------------------


def _paper_process_c_pair(index, provider, api_key, system_prompt, paper_text, paper,
                          dim, difficulty, q1, q2, q2_ok, ps,
                          questions, review_items, token_usage, diag_path, quality, model=None):
    """Generate C-dim sibling pair for paper mode. Returns (questions, review_items, new_index)."""
    pid = _paper_id(paper)
    qid_a = f"CS-EVO-{index:05d}"
    qid_b = f"CS-EVO-{index + 1:05d}"

    c2_prompt = build_user_prompt(
        paper_text, pid, dim, difficulty,
        override_desc=(
            "生成一道与上一题配对的C维度变体题。\n"
            "保持相同的基准条件，但改变sensitivity_dimension方向"
            "（如果上题是parameter就改为environment，反之亦然）。\n"
            "题目必须是单一、聚焦的问题，一次只分析一个不同的变化因素。\n"
        ),
    )

    diag_ctx_1b = _make_diag_ctx(index + 1, provider, "1b")
    q1b = call_llm(provider, api_key, system_prompt, c2_prompt, model=model,
                   _usage_acc=token_usage, diag_path=diag_path, diag_ctx=diag_ctx_1b)
    q1b_ok = _record_quality(quality, q1b)

    if q1b_ok:
        q1["sibling_id"] = qid_b
        q1b["sibling_id"] = qid_a
        diag_ctx_2b = _make_diag_ctx(index + 1, provider, "2b")
        q2b = call_llm(provider, api_key, system_prompt, c2_prompt, max_retries=1, model=model,
                       _usage_acc=token_usage, diag_path=diag_path, diag_ctx=diag_ctx_2b)
        q2b_ok = _record_quality(quality, q2b)

        for base, r1, r2_data, r2_ok, qid in [
            (q1, q1, q2, q2_ok, qid_a),
            (q1b, q1b, q2b, q2b_ok, qid_b),
        ]:
            self_consistent = r2_ok and (r1.get("answer", "").strip() == r2_data.get("answer", "").strip() if r2_ok else False)
            final_q = make_question_from_llm(r1, qid, paper, dim, difficulty, provider=provider, paper_source=ps)
            final_q["source_ref"] = pid
            final_q["sibling_id"] = qid_b if base is q1 else qid_a
            final_q["consistency_status"] = "auto_passed" if self_consistent else "needs_review"
            if not self_consistent and r2_data:
                review_items.append({
                    "id": qid, "dimension": dim, "difficulty": difficulty,
                    "question": r1.get("question", ""),
                    "round_1_answer": r1.get("answer", ""),
                    "round_2_answer": r2_data.get("answer", ""),
                    "verdict": None,
                })
            questions.append(final_q)
        new_index = index + 2
    else:
        self_consistent = q2_ok and (q1.get("answer", "").strip() == q2.get("answer", "").strip() if q2_ok else False)
        final_q = make_question_from_llm(q1, qid_a, paper, dim, difficulty, provider=provider, paper_source=ps)
        final_q["source_ref"] = pid
        final_q["consistency_status"] = "auto_passed" if self_consistent else "needs_review"
        questions.append(final_q)
        new_index = index + 1

    return questions, review_items, new_index


# ---------------------------------------------------------------------------
# Paper selection helpers
# ---------------------------------------------------------------------------


def _init_paper_caps(papers, corpus_dir, questions):
    """Compute per-paper capacity and initialize question counts."""
    paper_caps: dict[str, int] = {}
    paper_steps: dict[str, int] = {}
    paper_idx_to_key: dict[int, str] = {}
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
    return paper_caps, paper_steps, paper_idx_to_key, paper_q_counts


def _paper_select_next(papers, paper_idx, paper_idx_to_key, paper_q_counts, paper_caps, consecutive_failures, total_limit):
    """Select the next paper for question generation. Returns (paper, arxiv_id, cap) or None."""
    if paper_idx >= len(papers) * 3:
        print(f"[paper-mode] Cycled through all papers {paper_idx // len(papers)}x, stopping")
        return None
    failure_threshold = min(max(20, total_limit // 5), len(papers))
    if consecutive_failures >= failure_threshold:
        print(f"[paper-mode] Too many failures ({consecutive_failures}), stopping.")
        return None
    default_max = 12
    paper = min(papers, key=lambda p: paper_q_counts.get(paper_idx_to_key.get(papers.index(p), _paper_id(p)), 0))
    _pidx = papers.index(paper)
    pid = paper_idx_to_key.get(_pidx, _paper_id(paper))
    cap = paper_caps.get(pid, default_max)
    if paper_q_counts.get(pid, 0) >= cap:
        remaining = [(i, p) for i, p in enumerate(papers)
                     if paper_q_counts.get(paper_idx_to_key.get(i, _paper_id(p)), 0)
                     < paper_caps.get(paper_idx_to_key.get(i, _paper_id(p)), default_max) * 2]
        if not remaining:
            return None
        _ri, paper = min(remaining, key=lambda t: paper_q_counts.get(paper_idx_to_key.get(t[0], _paper_id(t[1])), 0))
        pid = paper_idx_to_key.get(_ri, _paper_id(paper))
        cap = paper_caps.get(pid, default_max)
    return paper, pid, cap


def _paper_generate_one_question(index, paper, arxiv_id, paper_text, ps, dim, difficulty,
                                  provider, api_key, model, system_prompt, user_prompt,
                                  questions, review_items, total_limit, quality, token_usage, diag_path, logger):
    """Generate one question via LLM and process it.

    Returns (questions, review_items, index, added) on success, or None on failure.
    added = 2 for C-dimension pair, 1 for single question.
    """
    diag_ctx_1 = _make_diag_ctx(index, provider, 1)
    q1 = call_llm(provider, api_key, system_prompt, user_prompt, model=model,
                  _usage_acc=token_usage, diag_path=diag_path, diag_ctx=diag_ctx_1)
    q1_ok = _record_quality(quality, q1)
    if not q1_ok:
        return None
    diag_ctx_2 = _make_diag_ctx(index, provider, 2)
    q2 = call_llm(provider, api_key, system_prompt, user_prompt, max_retries=2, model=model,
                  _usage_acc=token_usage, diag_path=diag_path, diag_ctx=diag_ctx_2)
    q2_ok = _record_quality(quality, q2)
    if dim == "C" and len(questions) + 2 <= total_limit:
        questions, review_items, index = _paper_process_c_pair(
            index, provider, api_key, system_prompt, paper_text, paper, dim, difficulty,
            q1, q2, q2_ok, ps, questions, review_items, token_usage, diag_path, quality, model)
        return questions, review_items, index, 2
    qid = f"CS-EVO-{index:05d}"
    final_q, review_items = _make_single_question(
        qid, q1, q2, q2_ok, paper, dim, difficulty, provider, review_items)
    final_q["paper_source"] = ps
    final_q["source_ref"] = arxiv_id
    final_q["scoring_reason"] = q1.get("scoring_reason", "")
    final_q["scoring_issues"] = q1.get("scoring_issues", [])
    questions.append(final_q)
    index += 1
    return questions, review_items, index, 1


# ---------------------------------------------------------------------------
# Concurrent worker
# ---------------------------------------------------------------------------


def _worker_generate_paper_mode(
    worker_id: int,
    tasks: list[TaskSlot],
    provider: str,
    model: str,
    api_key: str,
    output_dir: Path,
    reset_every: int,
    diag_path: str | None,
) -> dict:
    """Generate questions from pre-allocated TaskSlots in a single worker thread.

    Each worker owns its own QualityMonitor, system_prompt, and counters.
    Writes per-worker checkpoint to ``output_dir / worker_{worker_id}.checkpoint.json``.

    Returns a dict with keys: questions, review_items, token_usage, failed_count.
    """
    import logging as _logging
    questions: list[dict] = []
    review_items: list[dict] = []
    consecutive_failures = 0
    start_time = time.time()
    token_usage = [0, 0]
    questions_since_reset = 0
    failed_count = 0

    quality = QualityMonitor(window_size=15, consecutive_threshold=3)
    system_prompt = build_system_prompt()

    cp_path = output_dir / f"worker_{worker_id}.checkpoint.json"
    done_ids: set[str] = set()
    if cp_path.exists():
        cp_data = load_json(cp_path)
        questions = cp_data.get("questions", [])
        review_items = cp_data.get("review_items", [])
        token_usage = cp_data.get("token_usage", [0, 0])
        failed_count = cp_data.get("failed_count", 0)
        done_ids = {q["id"] for q in questions if "id" in q}
        print(f"[W{worker_id}] Resumed {len(questions)} questions from checkpoint", flush=True)

    failure_threshold = min(max(20, len(tasks) // 5), len(tasks))

    for task in tasks:
        ps = build_paper_source(task.paper, output_dir, text_offset=task.text_offset)
        pid = _paper_id(task.paper)
        qid = f"CS-EVO-{task.global_index:05d}"

        if qid in done_ids:
            continue

        if consecutive_failures >= failure_threshold:
            print(f"[W{worker_id}] Too many failures ({consecutive_failures}), stopping.", flush=True)
            break

        dim = task.dim
        difficulty = task.difficulty
        paper_text = task.paper_text

        user_prompt = build_user_prompt(paper_text, pid, dim, difficulty)
        print(f"[W{worker_id}] q{task.global_index}: dim={dim}, diff={difficulty}, paper={pid}", flush=True)

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
        if dim == "C":
            final_q["sibling_id"] = None

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
            prompt_builder_fn=build_system_prompt)

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
            dim_keys = sorted(set(q.get("dimension", "") for q in questions)) or ["A","B","C","D"]
            dim_counts = {k: 0 for k in dim_keys}
            for q in questions:
                d = q.get("dimension", "")
                if d in dim_counts:
                    dim_counts[d] += 1
            dc_str = " ".join(f"{d}={dim_counts[d]}" for d in dim_keys)
            eta = (len(tasks) - len(questions)) / max(rate, 0.01) if rate > 0 else 0
            print(f"  [W{worker_id}] cp ({len(questions)}/{len(tasks)}) {rate:.1f}qpm ETA:{eta:.0f}min | {dc_str}", flush=True)

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
                           diag_path, provider, model, api_key, total_limit, assign_tasks_fn):
    """Concurrent paper-based generation with worker pool."""
    print(f"[paper-mode] Concurrent mode: {args.workers} workers")
    dim_plan = precompute_dim_plan(total_limit)
    worker_tasks = assign_tasks_fn(papers, dim_plan, args.workers, corpus_dir,
                                max_per_paper=12, seed=args.seed, start_index=args.start_index)
    total_slots = sum(len(t) for t in worker_tasks)
    print(f"[paper-mode] {total_slots} slots across {args.workers} workers")

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = []
        wid_for_future: dict = {}
        for wid, tasks in enumerate(worker_tasks):
            if not tasks:
                continue
            worker_diag = str(output_path.parent / f"diag_w{wid}.logl") if diag_path else None
            future = executor.submit(
                _worker_generate_paper_mode,
                wid, tasks, provider, model, api_key,
                output_path.parent, args.reset_every, worker_diag)
            futures.append(future)
            wid_for_future[future] = wid

        all_questions: list[dict] = []
        all_review: list[dict] = []
        for i, future in enumerate(futures):
            try:
                result = future.result()
            except Exception as e:
                import traceback
                wid = wid_for_future.get(future, i)
                print(f"[paper-mode] Worker {wid} raised: {e}", flush=True)
                traceback.print_exc()
                cp = output_path.parent / f"worker_{wid}.checkpoint.json"
                if cp.exists():
                    try:
                        cp_data = load_json(cp)
                        recovered = cp_data.get("questions", [])
                        all_questions.extend(recovered)
                        all_review.extend(cp_data.get("review_items", []))
                        print(f"[paper-mode] Recovered {len(recovered)} questions from crashed worker {wid}", flush=True)
                    except Exception as load_err:
                        print(f"[paper-mode] Failed to load checkpoint for worker {wid}: {load_err}", flush=True)
                continue
            n_q = len(result.get("questions", []))
            print(f"[paper-mode] Worker {result.get('worker_id', i)} finished: {n_q} questions", flush=True)
            all_questions.extend(result["questions"])
            all_review.extend(result.get("review_items", []))

    all_questions.sort(key=lambda q: q.get("id", ""))

    print(f"\n[paper-mode] Running answer lint on {len(all_questions)} questions...")
    lint_result = lint_batch(all_questions)
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

    print(f"\nBenchmark saved: {output_path}", flush=True)
    print(f"Manual review saved: {review_path}", flush=True)


# ---------------------------------------------------------------------------
# Serial paper mode
# ---------------------------------------------------------------------------


def _paper_mode_serial(args, papers, corpus_dir, output_path, review_path,
                       diag_path, start_index, resume_questions, total_limit,
                       provider, model, api_key, registry):
    """Serial paper-based generation with quality monitoring and checkpoint."""
    import logging

    log_path = output_path.with_suffix(".log")
    logging.basicConfig(filename=str(log_path), level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s", encoding="utf-8")
    logger = logging.getLogger(__name__)
    logger.info(f"Paper mode: split={args.paper_split}, papers={len(papers)}, limit={total_limit}")

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
    system_prompt = build_system_prompt()

    paper_caps, paper_steps, paper_idx_to_key, paper_q_counts = _init_paper_caps(papers, corpus_dir, questions)

    while len(questions) < total_limit and paper_idx < len(papers) * 3:
        selection = _paper_select_next(papers, paper_idx, paper_idx_to_key, paper_q_counts, paper_caps, consecutive_failures, total_limit)
        if selection is None:
            break
        paper, arxiv_id, cap = selection

        paper_path_str = paper.get("paper_path", "")
        q_index = paper_q_counts.get(arxiv_id, 0)
        step = paper_steps.get(arxiv_id, 3000)
        text_offset = q_index * step
        paper_text = read_paper_text(paper_path_str, corpus_dir, max_chars=6000, offset=text_offset)
        if not paper_text:
            paper_idx += 1
            consecutive_failures += 1
            continue

        paper_reg = registry.get(arxiv_id, paper)
        ps = build_paper_source(paper_reg, corpus_dir, text_offset=text_offset)
        dim, difficulty = pick_next_dim_and_diff(questions, total_limit)
        user_prompt = build_user_prompt(paper_text, arxiv_id, dim, difficulty)

        print(f"[paper-mode] q{index}: dim={dim}, diff={difficulty}, paper={arxiv_id} "
              f"(#{paper_q_counts.get(arxiv_id, 0)+1}/{cap})", flush=True)
        logger.info(f"q{index}: dim={dim}, diff={difficulty}, paper={arxiv_id}")

        result = _paper_generate_one_question(
            index, paper, arxiv_id, paper_text, ps, dim, difficulty,
            provider, api_key, model, system_prompt, user_prompt,
            questions, review_items, total_limit, quality, token_usage, diag_path, logger)
        if result is None:
            consecutive_failures += 1
            paper_idx += 1
            continue
        questions, review_items, index, added = result
        paper_q_counts[arxiv_id] += added
        paper_idx += 1
        consecutive_failures = 0
        questions_since_reset += 1

        questions_since_reset, system_prompt, quality = _gen_reset_if_needed(
            questions_since_reset, quality, args.reset_every, logger, system_prompt, len(questions),
            prompt_builder_fn=build_system_prompt)

        if len(questions) % 10 == 0:
            qs = quality.summary()
            dim_keys = sorted(set(q.get("dimension", "") for q in questions)) or ["A","B","C","D"]
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
                         source=f"AI-generated via {provider}/{model} from processed papers ({args.paper_split})",
                         review_items=review_items)

    print(f"\n[paper-mode] Running answer lint on {len(questions)} questions...")
    lint_result = lint_batch(questions)
    print(f"  Lint: {lint_result['passed']} passed, {lint_result['failed']} failed "
          f"({lint_result['pass_rate']:.1f}%)")
    if lint_result['failed'] > 0:
        for idx in lint_result['flagged_indices']:
            if idx < len(questions):
                issues = lint_result['details'][idx].get('issues', [])
                questions[idx]['consistency_status'] = 'needs_review'
                questions[idx].setdefault('lint_issues', []).extend(issues)
        source = f"AI-generated via {provider}/{model} (linted, {args.paper_split})"
        print(f"  {lint_result['failed']} questions marked as needs_review after lint")
    else:
        source = f"AI-generated via {provider}/{model} from processed papers ({args.paper_split})"

    write_outputs(questions, output_path, review_path, source=source, review_items=review_items)
    print(f"\nBenchmark saved: {output_path}", flush=True)
    print(f"Manual review saved: {review_path}", flush=True)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run_paper_mode(args, corpus_dir, output_path, review_path, diag_path,
                   start_index, resume_questions, total_limit,
                   load_paper_data_fn, resolve_credentials_fn, assign_tasks_fn):
    """Paper-based generation mode for control domain."""
    papers, registry = load_paper_data_fn(args, corpus_dir, domain='control')
    print(f"[paper-mode] {args.paper_split}: {len(papers)} papers")
    print(f"[paper-mode] Start index: {args.start_index}, limit: {total_limit}")

    if args.mock:
        registry_path = Path(args.registry) if args.registry else ROOT / "benchmark" / "dataset" / "paper_registry.json"
        questions = _paper_mock_generate(papers, total_limit, args.start_index, registry_path, corpus_dir)
        data = write_outputs(questions, output_path, review_path,
                             source=f"Mock-generated from processed papers ({args.paper_split})")
        print(f"\n[Mock] {len(questions)} questions generated", flush=True)
        print(f"Benchmark saved: {output_path}", flush=True)
        return

    provider, model, api_key = resolve_credentials_fn(args)
    print(f"[paper-mode] Provider: {provider}, Model: {model}")

    if args.workers > 1 and not args.resume:
        _paper_mode_concurrent(args, papers, corpus_dir, output_path, review_path,
                               diag_path, provider, model, api_key, total_limit, assign_tasks_fn)
        return

    if args.workers > 1 and args.resume:
        print("[paper-mode] WARNING: --resume is not yet supported with --workers > 1, falling back to serial.", flush=True)

    _paper_mode_serial(args, papers, corpus_dir, output_path, review_path,
                       diag_path, start_index, resume_questions, total_limit,
                       provider, model, api_key, registry)
