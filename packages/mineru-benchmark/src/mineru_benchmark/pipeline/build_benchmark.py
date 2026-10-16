"""Build ControlSci Sci-Align benchmark questions from corpus chunks.

Control domain entry point — pure control science with A/B/C/D dimensions
and L1-L4 difficulty levels. No domain branching.

Responsibilities:
  - CLI parsing & orchestration (main)
  - Corpus loading / chunk selection
  - Fulltext paper loading for paper_source tracing
  - Mock question generation
  - Question generation loop (API mode) with quality monitoring + periodic reset
  - Checkpoint / output persistence
  - Cross-model audit dispatch
"""
import argparse
import random
import time
from datetime import date  # noqa: F401  (used by backup naming in write_outputs)
from pathlib import Path

from mineru_benchmark.pipeline._build_common import (
    ROOT, DEFAULT_SEED, TaskSlot,
    _paper_id, first_env, load_corpus, read_chunk_text, read_paper_text, _paper_tier, _paper_step,
    _finalize_gen_output, _gen_reset_if_needed, _gen_checkpoint_and_status,
    write_outputs, _make_diag_ctx, _record_quality, _make_single_question,
    make_question_from_llm, default_rubric, _resolve_credentials, _resolve_run_state,
    _print_dry_run, _load_paper_data,
)
from mineru_benchmark.pipeline.arbiter import arbitrate_questions
from mineru_benchmark.pipeline.build_paper_mode import run_paper_mode
from mineru_generator.api import call_llm
from mineru_core.settings import PROVIDERS
from mineru_generator.distribution import (
    DIMENSION_CYCLE, DIFFICULTY_BY_DIMENSION,
    normalize_categories, pick_next_dim_and_diff,
)
from mineru_generator.prompts import build_system_prompt, build_user_prompt
from mineru_generator.quality import QualityMonitor


DEFAULT_CORPUS = ROOT / "corpus"
DEFAULT_OUTPUT = ROOT / "benchmark" / "dataset" / "benchmark.json"
DEFAULT_REVIEW = ROOT / "benchmark" / "dataset" / "manual_review.json"


def assign_tasks(
    papers: list[dict],
    dim_plan: list[tuple[str, str]],
    n_workers: int,
    corpus_dir: Path,
    max_per_paper: int = 12,
    seed: int = DEFAULT_SEED,
    start_index: int = 1,
) -> list[list[TaskSlot]]:
    """Distribute papers and dimension slots across N workers.

    Uses per-paper caps derived from text length (tiered: short=5,
    standard=8, long=12, book=20) and per-paper step sizes so each
    paper's question windows span the entire text.
    """
    rng = random.Random(seed)
    rng.shuffle(papers)

    # Build unique paper IDs and pre-compute per-paper caps + step sizes
    paper_ids: list[str] = []
    id_to_paper: dict[str, dict] = {}
    paper_caps: dict[str, int] = {}
    paper_steps: dict[str, int] = {}
    for idx, p in enumerate(papers):
        key = _paper_id(p) or f"__no_id_{idx}__"
        while key in id_to_paper:
            key = f"__no_arxiv_{idx}___{len(id_to_paper)}"
        paper_ids.append(key)
        id_to_paper[key] = p
        # Estimate text length for tier-based cap + step
        paper_path_str = p.get("paper_path", "")
        full_path = corpus_dir / paper_path_str / "full_text.md" if paper_path_str else None
        # Use char count (not st_size) for cross-platform accuracy (UTF-8 CJK ≈ 3 bytes/char)
        txt_len = len(full_path.read_text(encoding="utf-8", errors="ignore")) if full_path and full_path.exists() else 0
        cap = _paper_tier(txt_len) if txt_len > 0 else max_per_paper
        paper_caps[key] = cap
        paper_steps[key] = _paper_step(txt_len, cap)

    paper_counts: dict[str, int] = {k: 0 for k in paper_ids}
    slots: list[TaskSlot] = []
    slot_idx = 0
    paper_idx = 0
    max_cycles = sum(paper_caps.values()) * 2  # safety: per-paper caps × 2 relax window

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
        step = paper_steps.get(paper_key, 3000)  # per-paper dynamic step
        text_offset = q_index * step
        paper_text = read_paper_text(paper_path_str, corpus_dir, max_chars=6000, offset=text_offset)
        if not paper_text:
            paper_idx += 1
            paper_counts[paper_key] += cap  # mark as exhausted
            continue

        dim, difficulty = dim_plan[slot_idx]
        slots.append(TaskSlot(
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

    worker_tasks: list[list[TaskSlot]] = [[] for _ in range(n_workers)]
    for i, slot in enumerate(slots):
        worker_tasks[i % n_workers].append(slot)
    return worker_tasks


def select_chunks(corpus_dir, docs, chunks, category=None, min_tokens=80, seed=DEFAULT_SEED):
    selected = []
    for chunk in chunks:
        if chunk.get("estimated_tokens", 0) < min_tokens:
            continue
        doc = docs.get(chunk.get("doc_id"), {})
        cats = normalize_categories(doc.get("control_category"))
        if category and category not in cats:
            continue
        text = read_chunk_text(corpus_dir, chunk)
        if not text:
            continue
        selected.append({**chunk, "categories": cats, "text": text})
    random.Random(seed).shuffle(selected)
    return selected


def _process_dim_c_pair(index, current_provider, system_prompt, chunk_text, chunk, dim, difficulty,
                         q1, q2, q2_ok, questions, review_items, token_usage, diag_path, quality, active_api_key, active_model=None):
    """Handle Dimension C sibling pair generation. Returns (questions, review_items, new_index)."""
    qid_a = f"CS-EVO-{index:05d}"
    qid_b = f"CS-EVO-{index + 1:05d}"
    c2_prompt = build_user_prompt(
        chunk_text, chunk.get("chunk_id", ""), dim, difficulty,
        override_desc="生成一道与上一题配对的C维度变体题。\n保持相同的基准条件，但改变sensitivity_dimension方向（如果上题是parameter就改为environment，反之亦然）。\n题目必须是单一、聚焦的问题，一次只分析一个不同的变化因素。\n"
    )
    diag_ctx_1b = _make_diag_ctx(index + 1, current_provider, "1b")
    q1b = call_llm(current_provider, active_api_key, system_prompt, c2_prompt, model=active_model, _usage_acc=token_usage, diag_path=diag_path, diag_ctx=diag_ctx_1b)
    q1b_ok = _record_quality(quality, q1b)
    if q1b_ok:
        q1["sibling_id"] = qid_b
        q1b["sibling_id"] = qid_a
        diag_ctx_2b = _make_diag_ctx(index + 1, current_provider, "2b")
        q2b = call_llm(current_provider, active_api_key, system_prompt, c2_prompt, max_retries=1, model=active_model, _usage_acc=token_usage, diag_path=diag_path, diag_ctx=diag_ctx_2b)
        q2b_ok = _record_quality(quality, q2b)
        for (base, r1, r2_data, r2_ok) in [(q1, q1, q2, q2_ok), (q1b, q1b, q2b, q2b_ok)]:
            question_id = qid_a if base is q1 else qid_b
            self_consistent = False
            if r2_ok:
                self_consistent = (r1.get("answer", "").strip() == r2_data.get("answer", "").strip())
            final_q = make_question_from_llm(r1, question_id, chunk, dim, difficulty, provider=current_provider)
            final_q["consistency_status"] = "auto_passed" if self_consistent else "needs_review"
            if not self_consistent and r2_data:
                review_items.append({
                    "id": question_id, "dimension": dim, "difficulty": difficulty,
                    "question": r1.get("question", ""),
                    "round_1_answer": r1.get("answer", ""), "round_1_steps": r1.get("reasoning_steps", []),
                    "round_2_answer": r2_data.get("answer", ""), "round_2_steps": r2_data.get("reasoning_steps", []),
                    "verdict": None,
                })
            questions.append(final_q)
        new_index = index + 2
    else:
        self_consistent = False
        if q2_ok:
            self_consistent = (q1.get("answer", "").strip() == q2.get("answer", "").strip())
        final_q = make_question_from_llm(q1, qid_a, chunk, dim, difficulty, provider=current_provider)
        final_q["consistency_status"] = "auto_passed" if self_consistent else "needs_review"
        if not self_consistent and q2:
            review_items.append({
                "id": qid_a, "dimension": dim, "difficulty": difficulty,
                "question": q1.get("question", ""),
                "round_1_answer": q1.get("answer", ""), "round_1_steps": q1.get("reasoning_steps", []),
                "round_2_answer": q2.get("answer", ""), "round_2_steps": q2.get("reasoning_steps", []),
                "verdict": None,
            })
        questions.append(final_q)
        new_index = index + 1
    return questions, review_items, new_index


def generate_api_questions(chunks, limit, api_key, model, corpus_dir, provider="deepseek",
                            start_index=1, start_chunk=0, resume_questions=None, output_path=None, review_path=None,
                            reset_every=15, diag_path=None):
    import logging
    if output_path:
        log_path = output_path.with_suffix(".log")
    else:
        log_path = ROOT / "benchmark" / "generation.log"
    logging.basicConfig(filename=str(log_path), level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s", encoding="utf-8")
    logger = logging.getLogger(__name__)

    print(f"[API] Starting: limit={limit}, chunks={len(chunks)}, provider={provider}, reset_every={reset_every}", flush=True)
    logger.info(f"Start: limit={limit}, chunks={len(chunks)}, provider={provider}, reset_every={reset_every}")

    quality = QualityMonitor(window_size=15, consecutive_threshold=3)
    questions = list(resume_questions) if resume_questions else []
    review_items = []
    index = start_index
    chunk_index = start_chunk
    consecutive_failures = 0
    questions_since_reset = 0
    start_time = time.time()
    current_provider = provider
    active_api_key = api_key
    active_model = model
    token_usage = [0, 0]

    system_prompt = build_system_prompt()

    while len(questions) < limit and chunk_index < len(chunks):
        failure_threshold = min(max(20, limit // 5), len(chunks))
        if consecutive_failures >= failure_threshold:
            print(f"[API] Too many failures ({consecutive_failures}), stopping.", flush=True)
            logger.warning(f"Stopped early after {consecutive_failures} failures")
            break

        dim, difficulty = pick_next_dim_and_diff(questions, limit)
        chunk = chunks[chunk_index]
        chunk_text = chunk.get("text", "")
        if not chunk_text:
            chunk_index += 1
            consecutive_failures += 1
            continue

        user_prompt = build_user_prompt(chunk_text, chunk.get("chunk_id", ""), dim, difficulty)

        dim_counts = {}
        for q in questions:
            d = q.get("dimension")
            if not d:
                continue
            dim_counts[d] = dim_counts.get(d, 0) + 1

        print(f"[API] q{index}: dim={dim}, diff={difficulty}, chunk={chunk_index}, reset_ct={questions_since_reset}", flush=True)
        logger.info(f"q{index}: dim={dim}, diff={difficulty}, chunk={chunk_index}, provider={current_provider}")

        diag_ctx_1 = _make_diag_ctx(index, current_provider, 1)
        q1 = call_llm(current_provider, active_api_key, system_prompt, user_prompt, model=active_model, _usage_acc=token_usage, diag_path=diag_path, diag_ctx=diag_ctx_1)
        q1_ok = _record_quality(quality, q1)
        if q1:
            print(f"  [API] Round1 validated OK, q[:50]={q1.get('question','')[:50]}", flush=True)
        else:
            print("  [API] Round1 returned None", flush=True)

        if not q1_ok:
            chunk_index += 1
            consecutive_failures += 1
            continue

        diag_ctx_2 = _make_diag_ctx(index, current_provider, 2)
        q2 = call_llm(current_provider, active_api_key, system_prompt, user_prompt, max_retries=2, model=active_model, _usage_acc=token_usage, diag_path=diag_path, diag_ctx=diag_ctx_2)
        q2_ok = _record_quality(quality, q2)

        if dim == "C" and len(questions) + 2 <= limit:
            questions, review_items, index = _process_dim_c_pair(
                index, current_provider, system_prompt, chunk_text, chunk, dim, difficulty,
                q1, q2, q2_ok, questions, review_items, token_usage, diag_path, quality, active_api_key, active_model)
        else:
            qid = f"CS-EVO-{index:05d}"
            final_q, review_items = _make_single_question(qid, q1, q2, q2_ok, chunk, dim, difficulty, current_provider, review_items)
            questions.append(final_q)
            index += 1

        chunk_index += 1
        consecutive_failures = 0
        questions_since_reset += 1

        questions_since_reset, system_prompt, quality = _gen_reset_if_needed(
            questions_since_reset, quality, reset_every, logger, system_prompt, len(questions),
            prompt_builder_fn=build_system_prompt)

        _gen_checkpoint_and_status(output_path, review_path, questions, review_items, limit,
                                    index, chunk_index, current_provider, active_model,
                                    start_time, consecutive_failures, token_usage, quality,
                                    questions_since_reset, dim_counts, logger)

    return _finalize_gen_output(output_path, review_path, questions, review_items, index, chunk_index,
                                 current_provider, active_model, start_time, limit, consecutive_failures,
                                 token_usage, quality, questions_since_reset)


def make_question(qid, dimension, difficulty, chunk, sibling_id=None):
    section = chunk.get("source_section") or chunk.get("chunk_id")
    categories = normalize_categories(chunk.get("categories"))
    topic = section.replace("#", "").strip()[:80] or chunk.get("filename", "控制系统")
    source_ref = chunk.get("chunk_id")

    base = {
        "id": qid,
        "dimension": dimension,
        "difficulty_level": difficulty,
        "control_category": categories,
        "source_ref": source_ref,
        "consistency_status": "auto_passed",
        "sensitivity_dimension": None,
        "sibling_id": None,
        "rubric": None,
        "paper_source": {},
        "scoring_reason": "Mock-generated placeholder",
        "scoring_issues": ["Mock数据，非实际API生成"],
    }

    if dimension == "A":
        base.update(
            {
                "question": f"结合教材片段\u201c{topic}\u201d，说明其中涉及的核心控制概念，并给出必要的数学表达。",
                "answer": f"该片段围绕{topic}展开。答案应给出概念定义、关键变量含义，并写出与该概念对应的核心表达式或判据。",
                "reasoning_steps": ["识别片段主题", "抽取核心概念", "说明数学表达", "解释变量或判据含义"],
            }
        )
    elif dimension == "B":
        base.update(
            {
                "question": f"基于教材片段\u201c{topic}\u201d，构造一个多步推理问题：从系统描述出发推导控制结论。",
                "answer": "应先建立模型或方程，再应用对应控制理论判据，最后给出稳定性、性能或控制律结论。",
                "reasoning_steps": ["建立系统描述", "选择适用判据或方法", "执行代数或逻辑推导", "给出结论并解释"],
            }
        )
    elif dimension == "C":
        base.update(
            {
                "question": f"在\u201c{topic}\u201d场景下，如果关键参数或约束发生变化，原控制结论是否保持？请说明。",
                "answer": "需要比较变化前后的模型、参数或约束，判断控制性能或稳定性结论是否改变，并说明敏感性来源。",
                "reasoning_steps": ["确定基准条件", "引入条件变化", "比较判据或性能指标", "解释结论变化"],
                "sensitivity_dimension": "parameter",
                "sibling_id": sibling_id,
            }
        )
    else:
        base.update(
            {
                "question": f"围绕\u201c{topic}\u201d设计一个完整控制方案，要求说明建模、控制器设计、验证指标和工程约束。",
                "answer": "可行方案应覆盖系统建模、控制方法选择、参数整定或优化、仿真/实验验证以及工程约束处理。",
                "reasoning_steps": ["分析控制目标", "建立模型", "选择控制方法", "设计验证流程", "讨论工程约束"],
                "rubric": default_rubric(topic),
            }
        )
    return base


def generate_mock_questions(chunks, limit):
    questions = []
    index = 1
    chunk_index = 0
    group_index = 0
    while len(questions) < limit and chunks:
        dimension = DIMENSION_CYCLE[group_index % len(DIMENSION_CYCLE)]
        difficulty_options = DIFFICULTY_BY_DIMENSION[dimension]
        difficulty = difficulty_options[group_index % len(difficulty_options)]
        chunk = chunks[chunk_index % len(chunks)]

        if dimension == "C" and len(questions) + 2 <= limit:
            qid_a = f"CS-EVO-{index:05d}"
            qid_b = f"CS-EVO-{index + 1:05d}"
            questions.append(make_question(qid_a, "C", difficulty, chunk, sibling_id=qid_b))
            questions.append(make_question(qid_b, "C", difficulty, chunk, sibling_id=qid_a))
            index += 2
        else:
            if dimension == "C":
                dimension = "D"
                difficulty = DIFFICULTY_BY_DIMENSION[dimension][group_index % len(DIFFICULTY_BY_DIMENSION[dimension])]
            qid = f"CS-EVO-{index:05d}"
            questions.append(make_question(qid, dimension, difficulty, chunk))
            index += 1
        chunk_index += 1
        group_index += 1
    return questions[:limit]


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main() -> None:
    args = _parse_args()

    if args.arbitrate:
        _run_benchmark_arbitrate(args)
        return

    corpus_dir = Path(args.corpus).resolve()
    output_path = Path(args.output).resolve()
    review_path = Path(args.manual_review_output).resolve()

    diag_path = None
    if args.diag_stem:
        diag_path = str(ROOT / "benchmark" / "logs" / "diag" / f"generation_diag_{args.diag_stem}.logl")

    start_index, start_chunk, resume_questions, total_limit = _resolve_run_state(args, output_path, review_path)

    if args.audit:
        print("[WARN] --audit not yet integrated; run cross_audit.py standalone after generation.", flush=True)
    if args.arbitrate_contested:
        print("[WARN] --arbitrate-contested not yet implemented; run arbitration separately.", flush=True)

    if args.paper_split:
        run_paper_mode(args, corpus_dir, output_path, review_path, diag_path,
                       start_index, resume_questions, total_limit,
                       load_paper_data_fn=_load_paper_data,
                       resolve_credentials_fn=_resolve_credentials,
                       assign_tasks_fn=assign_tasks)
        return

    docs, chunks = load_corpus(corpus_dir)
    selected = select_chunks(corpus_dir, docs, chunks, category=args.category, seed=args.seed)
    if not selected:
        raise SystemExit("No usable chunks found for the requested filters.")

    if args.dry_run:
        _print_dry_run(args, corpus_dir, selected, total_limit, diag_path, resume_questions)
        return

    provider, model, api_key = _resolve_credentials(args)
    print(f"[DEBUG] selected {len(selected)} chunks, provider={provider}, model={model}, limit={total_limit}, seed={args.seed}, reset_every={args.reset_every}", flush=True)

    if args.mock:
        questions = generate_mock_questions(selected, total_limit)
        data = write_outputs(
            questions, output_path, review_path,
            source=f"Mock-generated from ControlSci corpus chunks; model target={model}",
        )
    else:
        _run_api_generation(args, selected, total_limit, api_key, model, corpus_dir, provider,
                            start_index, start_chunk, resume_questions,
                            output_path, review_path, diag_path)

    print(f"Benchmark saved: {output_path}", flush=True)
    print(f"Manual review saved: {review_path}", flush=True)


def _parse_args():
    parser = argparse.ArgumentParser(description="Build ControlSci Sci-Align benchmark questions from corpus chunks.")
    parser.add_argument("--corpus", default=str(DEFAULT_CORPUS), help="Corpus root directory.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Benchmark output JSON path.")
    parser.add_argument("--manual-review-output", default=str(DEFAULT_REVIEW), help="Manual review JSON path.")
    parser.add_argument("--model", default=None, help="Model alias or full ID. Empty = provider default.")
    parser.add_argument("--api-key", default=None, help="API key. Falls back to provider env var.")
    parser.add_argument("--category", default=None, help="Optional normalized control category filter, e.g. classical.")
    parser.add_argument("--limit", type=int, default=10, help="Number of questions to generate.")
    parser.add_argument("--mock", action="store_true", help="Generate deterministic mock questions without API calls.")
    parser.add_argument("--dry-run", action="store_true", help="Print generation plan without writing files.")
    parser.add_argument("--provider", default="deepseek", choices=["deepseek", "qwen", "mimo"],
                        help="Provider to use for generation.")
    parser.add_argument("--arbitrate", action="store_true",
                        help="Run arbitration pass: reads existing benchmark+review, resolves needs_review via DeepSeek.")
    parser.add_argument("--arbitrate-workers", type=int, default=3,
                        help="Parallel arbitration API calls. Default 3 balances speed and stability.")
    parser.add_argument("--arbitrate-max-tokens", type=int, default=1024,
                        help="Max tokens for each arbitration response. Default 1024.")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from checkpoint files.")
    parser.add_argument("--supplement", type=int, default=0,
                        help="Generate N additional questions and append to existing benchmark.json.")
    parser.add_argument("--reset-every", type=int, default=15,
                        help="Rebuild OpenAI client and system prompt every N questions (default: 15).")
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of concurrent API workers (default 1 = serial).")
    parser.add_argument("--diag-stem", default=None,
                        help="Stem name for diagnostic log file.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help="Random seed for chunk ordering and reproducible sampling metadata.")
    parser.add_argument("--registry", default=None,
                        help="Path to paper_registry.json for paper_source tracing.")
    parser.add_argument("--paper-split", default=None, choices=["group_a", "group_b"],
                        help="Use paper_split.json groups: group_a or group_b.")
    parser.add_argument("--start-index", type=int, default=1,
                        help="Starting question ID index.")
    parser.add_argument("--audit", action="store_true",
                        help="Run cross-model audit after generation (requires --auditor).")
    parser.add_argument("--auditor", default=None, choices=["deepseek", "mimo", "qwen"],
                        help="Auditor model for cross-audit.")
    parser.add_argument("--arbitrate-contested", action="store_true",
                        help="Run arbitration on contested questions (uses auditor provider).")
    parser.add_argument("--split-path", default=None,
                        help="Path to paper_split.json.")
    args = parser.parse_args()

    import mineru_benchmark.pipeline._build_common as common
    common.RUN_SEED = args.seed

    return args


def _run_benchmark_arbitrate(args):
    output_path = Path(args.output).resolve()
    review_path = Path(args.manual_review_output).resolve()
    if not output_path.exists():
        raise SystemExit(f"Benchmark file not found: {output_path}")
    if not review_path.exists():
        raise SystemExit(f"Manual review file not found: {review_path}")
    api_key = args.api_key or first_env(PROVIDERS["deepseek"].api_key_env)
    if not api_key:
        raise SystemExit("Arbitration mode requires --api-key or DEEPSEEK_API_KEY/OPENAI_API_KEY (uses DeepSeek).")
    arbitrate_questions(
        str(output_path), str(review_path), str(output_path), api_key,
        workers=args.arbitrate_workers,
        max_tokens=args.arbitrate_max_tokens,
    )


def _run_api_generation(args, selected, total_limit, api_key, model, corpus_dir, provider,
                         start_index, start_chunk, resume_questions,
                         output_path, review_path, diag_path):
    print("[DEBUG] calling generate_api_questions...", flush=True)
    questions, review_items = generate_api_questions(
        selected, total_limit, api_key, model, corpus_dir, provider=provider,
        start_index=start_index, start_chunk=start_chunk,
        resume_questions=resume_questions,
        output_path=output_path, review_path=review_path,
        reset_every=args.reset_every,
        diag_path=diag_path,
    )
    print(f"[DEBUG] generate_api_questions returned {len(questions)} questions", flush=True)
    data = write_outputs(
        questions, output_path, review_path,
        source=f"AI-generated via {provider}/{model} from ControlSci corpus chunks",
        review_items=review_items,
    )
    cp_out = output_path.with_suffix(".checkpoint.json")
    cp_rev = review_path.with_suffix(".checkpoint.json")
    if cp_out.exists():
        cp_out.unlink()
    if cp_rev.exists():
        cp_rev.unlink()
    needs_review = sum(1 for q in questions if q.get("consistency_status") == "needs_review")
    print(f"Questions needing review: {needs_review}/{data['meta']['total_questions']}")


if __name__ == "__main__":
    main()
