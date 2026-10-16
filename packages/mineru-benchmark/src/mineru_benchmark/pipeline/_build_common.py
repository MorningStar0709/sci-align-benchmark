"""Shared infrastructure for benchmark generation (domain-agnostic).

Extracted from build_benchmark.py to be shared by both control and medical domains.
Contains IO utilities, checkpoint management, metadata generation, and quality monitoring helpers.
"""
import json
import os
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from mineru_generator.api import validate_api_question


ROOT = Path(__file__).resolve().parents[5]
DEFAULT_SEED = 42
RUN_SEED = DEFAULT_SEED


# ---------------------------------------------------------------------------
# Shared data structures
# ---------------------------------------------------------------------------


@dataclass
class TaskSlot:
    """Pre-allocated slot for one benchmark question.

    Contains all read-only inputs a worker thread needs to generate a single
    question — no shared mutable references.
    """
    global_index: int       # 1-based global question index
    paper: dict             # paper metadata dict (arxiv_id, title, authors, year, paper_path, control_category)
    paper_text: str         # windowed full_text.md content
    dim: str                # pre-computed dimension (A/B/C/D for control, identify/model/transfer for medical)
    difficulty: str         # pre-computed difficulty level (L1-L4)
    text_offset: int = 0    # char offset used to extract paper_text (for audit reproduction)


# ---------------------------------------------------------------------------
# IO / Utility functions
# ---------------------------------------------------------------------------


def _paper_id(paper: dict) -> str:
    """Extract paper ID regardless of domain (arxiv_id for control, pmc_id for medical)."""
    return paper.get("arxiv_id") or paper.get("pmc_id") or ""


def first_env(env_spec: str) -> str:
    for env_name in env_spec.split("|"):
        value = os.environ.get(env_name.strip(), "")
        if value:
            return value
    return ""


def load_json(path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json_atomic(path, data):
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.replace(path)


def load_corpus(corpus_dir):
    metadata_path = corpus_dir / "metadata.json"
    manifest_path = corpus_dir / "chunks" / "chunks_manifest.json"
    if not metadata_path.exists():
        raise SystemExit(f"Missing corpus metadata: {metadata_path}")
    if not manifest_path.exists():
        raise SystemExit(f"Missing chunks manifest: {manifest_path}")
    metadata = load_json(metadata_path)
    manifest = load_json(manifest_path)
    docs = {doc["id"]: doc for doc in metadata.get("docs", [])}
    chunks = manifest.get("chunks", [])
    return docs, chunks


def read_chunk_text(corpus_dir, chunk):
    filepath = Path(chunk["filepath"])
    path = filepath if filepath.is_absolute() else ROOT / filepath
    if not path.exists():
        path = corpus_dir / filepath.name
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8", errors="ignore").strip()
    return text[:2400]


def load_fulltext_corpus(registry_path, corpus_dir):
    """Load paper_registry.json and build paper index.

    Returns dict[paper_id, paper_entry] where each entry contains
    paper_id, title, authors, year, control_category, paper_path.
    """
    if not registry_path.exists():
        return {}
    data = load_json(registry_path)
    papers = data.get("papers", data if isinstance(data, list) else [])
    return {_paper_id(p): p for p in papers}


def read_paper_text(paper_path_str, corpus_dir, max_chars=6000, offset=0):
    """Read full_text.md for a paper, truncated to a sliding window.

    Each paper question advances by ``max_chars // 2`` (50% overlap) so
    adjacent questions see different sections without hard sentence breaks.
    """
    full_path = corpus_dir / paper_path_str / "full_text.md"
    if not full_path.exists():
        return ""
    text = full_path.read_text(encoding="utf-8", errors="ignore")
    start = min(offset, max(0, len(text) - max_chars))
    return text[start : start + max_chars]


def _paper_tier(text_length: int) -> int:
    """Return per-paper question cap based on text length.

    Short paper:   <30K chars  → 5  questions (no over-sampling)
    Standard:      30-200K     → 8  questions (balanced coverage)
    Long/survey:   200-500K    → 12 questions (thorough coverage)
    Book/textbook: >500K       → 20 questions (chapter-level diversity)
    """
    if text_length < 30_000:
        return 5
    elif text_length < 200_000:
        return 8
    elif text_length < 500_000:
        return 12
    else:
        return 20


def _paper_step(text_length: int, paper_cap: int, window: int = 6000) -> int:
    """Dynamic step size so the last question window reaches paper end."""
    if text_length <= window or paper_cap <= 1:
        return 0
    return max(1, (text_length - window) // (paper_cap - 1))


def build_paper_source(paper_entry, corpus_dir, text_offset=0):
    """Build paper_source dict from paper registry entry."""
    return {
        "arxiv_id": _paper_id(paper_entry),
        "title": paper_entry.get("title", ""),
        "authors": paper_entry.get("authors", []),
        "year": paper_entry.get("year"),
        "paper_path": paper_entry.get("paper_path", ""),
        "text_offset": text_offset,
    }


# ---------------------------------------------------------------------------
# Checkpoint / Status management
# ---------------------------------------------------------------------------


def build_meta(questions, source, all_dimensions=None):
    """Build metadata dict. If all_dimensions is given, pre-fill zero counts."""
    if all_dimensions:
        dimensions = {d: 0 for d in all_dimensions}
    else:
        dimensions = {}
    for question in questions:
        dimension = question.get("dimension")
        if dimension:
            dimensions[dimension] = dimensions.get(dimension, 0) + 1
    is_mock = "mock" in source.lower()
    return {
        "project": "ControlSci — 控制科学结构化语料库与 Sci-Align 跨模态对齐评测基准",
        "version": "1.0-mock" if is_mock else "1.0",
        "updated": str(date.today()),
        "total_questions": len(questions),
        "dimensions": dimensions,
        "source": source,
        "seed": RUN_SEED,
    }


def save_checkpoint(output_path, review_path, questions, review_items, index, chunk_index,
                    provider, model, source, start_time, limit=0, consecutive_failures=0,
                    token_usage=None, quality_summary=None, questions_since_reset=0):
    cp_out = output_path.with_suffix(".checkpoint.json")
    cp_rev = review_path.with_suffix(".checkpoint.json")
    data = {
        "meta": build_meta(questions, source),
        "state": {
            "next_index": index,
            "chunk_index": chunk_index,
            "provider": provider,
            "model": model,
            "seed": RUN_SEED,
            "consecutive_failures": consecutive_failures,
            "questions_since_reset": questions_since_reset,
        },
        "questions": questions,
    }
    write_json_atomic(cp_out, data)
    review = {"total": len(review_items), "items": review_items}
    write_json_atomic(cp_rev, review)
    elapsed = time.time() - start_time
    rate = len(questions) / elapsed * 60 if elapsed > 0 else 0
    remaining = max(limit - len(questions), 0)
    eta_min = remaining / rate if rate > 0 else 0

    status = {
        "progress": f"{len(questions)}/{limit}",
        "percent": round(len(questions) / max(limit, 1) * 100, 1),
        "rate": round(rate, 1),
        "eta_min": round(eta_min, 0),
        "dimensions": build_meta(questions, source)["dimensions"],
        "consecutive_failures": consecutive_failures,
        "questions_since_reset": questions_since_reset,
        "last_question_time": time.strftime("%H:%M:%S"),
        "provider": provider,
        "seed": RUN_SEED,
        "status": "running",
    }
    if token_usage:
        status["token_usage"] = {"prompt": token_usage[0], "completion": token_usage[1]}
    if quality_summary:
        status["quality"] = quality_summary
    status_path = output_path.parent / "generation_status.json"
    write_json_atomic(status_path, status)
    return rate, eta_min


def _write_light_status(status_path, questions, limit, consecutive_failures, token_usage,
                        start_time, provider, quality_summary=None, questions_since_reset=0):
    elapsed = time.time() - start_time
    total_q = len(questions)
    rate = total_q / elapsed * 60 if elapsed > 0 else 0
    eta_min = (max(limit - total_q, 0)) / rate if rate > 0 else 0
    dim_counts = {}
    for q in questions:
        d = q.get("dimension", "")
        if d:
            dim_counts[d] = dim_counts.get(d, 0) + 1
    s = {
        "progress": f"{total_q}/{limit}",
        "percent": round(total_q / max(limit, 1) * 100, 1),
        "rate": round(rate, 1),
        "eta_min": round(eta_min, 0),
        "dimensions": dim_counts,
        "consecutive_failures": consecutive_failures,
        "questions_since_reset": questions_since_reset,
        "token_usage": {"prompt": token_usage[0], "completion": token_usage[1]},
        "last_question_time": time.strftime("%H:%M:%S"),
        "provider": provider,
        "seed": RUN_SEED,
        "status": "running",
    }
    if quality_summary:
        s["quality"] = quality_summary
    write_json_atomic(status_path, s)


def _finalize_gen_output(output_path, review_path, questions, review_items, index, chunk_index,
                          current_provider, active_model, start_time, limit, consecutive_failures,
                          token_usage, quality, questions_since_reset):
    """Final checkpoint save and status output for generation."""
    import logging
    logger = logging.getLogger(__name__)
    reached_limit = len(questions) >= limit

    if output_path and review_path:
        qs = quality.summary()
        save_checkpoint(
            output_path, review_path, questions, review_items,
            index, chunk_index, current_provider, active_model,
            f"AI-generated via {current_provider}/{active_model}",
            start_time, limit=limit, consecutive_failures=consecutive_failures,
            token_usage=token_usage, quality_summary=qs, questions_since_reset=questions_since_reset,
        )
        elapsed = time.time() - start_time
        logger.info(f"Done: {len(questions)}/{limit} in {elapsed:.0f}s")
        status_path = output_path.parent / "generation_status.json"
        final_dim = build_meta(questions, f"AI-generated via {current_provider}/{active_model}")["dimensions"]
        final_status = "completed" if reached_limit else "stopped_early"
        write_json_atomic(status_path, {
            "progress": f"{len(questions)}/{limit}",
            "percent": round(len(questions) / max(limit, 1) * 100, 1),
            "rate": round(len(questions) / elapsed * 60, 1) if elapsed > 0 else 0,
            "eta_min": 0,
            "dimensions": final_dim,
            "consecutive_failures": consecutive_failures,
            "questions_since_reset": questions_since_reset,
            "token_usage": {"prompt": token_usage[0], "completion": token_usage[1]},
            "elapsed_sec": round(elapsed, 0),
            "provider": current_provider,
            "seed": RUN_SEED,
            "status": final_status,
            "quality": qs,
            "last_updated": time.strftime("%H:%M:%S"),
        })
    return questions[:limit], review_items


def _gen_reset_if_needed(questions_since_reset, quality, reset_every, logger, system_prompt,
                         len_questions, prompt_builder_fn=None):
    """Reset system prompt and quality monitor if threshold reached.

    Args:
        prompt_builder_fn: Callable that returns a fresh system prompt.
            If None, falls back to mineru_generator.prompts.build_system_prompt().
    """
    if questions_since_reset >= reset_every or quality.should_reset():
        trigger = "quality" if quality.should_reset() else "periodic"
        print(f"  [Reset] {trigger} reset after {questions_since_reset} questions since last reset", flush=True)
        logger.info(f"{trigger.capitalize()} reset at {len_questions} questions, {questions_since_reset} since last reset")
        if prompt_builder_fn is not None:
            system_prompt = prompt_builder_fn()
        else:
            from mineru_generator.prompts import build_system_prompt
            system_prompt = build_system_prompt()
        quality.reset()
    return questions_since_reset, system_prompt, quality


def _gen_checkpoint_and_status(output_path, review_path, questions, review_items, limit,
                                index, chunk_index, current_provider, active_model,
                                start_time, consecutive_failures, token_usage, quality,
                                questions_since_reset, dim_counts, logger):
    """Save checkpoint and write generation status."""
    if output_path and review_path and len(questions) % 10 == 0:
        qs = quality.summary()
        rate, eta_min = save_checkpoint(
            output_path, review_path, questions, review_items,
            index, chunk_index, current_provider, active_model,
            f"AI-generated via {current_provider}/{active_model}",
            start_time, limit=limit, consecutive_failures=consecutive_failures,
            token_usage=token_usage, quality_summary=qs, questions_since_reset=questions_since_reset,
        )
        dim_counts_str = " ".join(f"{d}={dim_counts.get(d,0)}" for d in sorted(dim_counts))
        print(f"  ✓ checkpoint ({len(questions)}/{limit}) {rate:.1f}题/min ETA:{eta_min:.0f}min | {dim_counts_str}", flush=True)
        logger.info(f"Checkpoint: {len(questions)}/{limit}, rate={rate:.1f}, eta={eta_min:.0f}min")

    if output_path and review_path and len(questions) > 0:
        status_path = output_path.parent / "generation_status.json"
        _write_light_status(status_path, questions, limit, consecutive_failures, token_usage,
                            start_time, current_provider,
                            quality_summary=quality.summary(), questions_since_reset=questions_since_reset)


# ---------------------------------------------------------------------------
# Output / Metadata
# ---------------------------------------------------------------------------


def write_outputs(questions, output_path, review_path, source, review_items=None):
    data = {"meta": build_meta(questions, source), "questions": questions}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        backup = output_path.with_suffix(f".{date.today().strftime('%Y%m%d_%H%M%S')}.bak.json")
        output_path.rename(backup)
        print(f"  Backup saved: {backup.name}", flush=True)
    write_json_atomic(output_path, data)
    if review_items:
        review = {"total": len(review_items), "items": review_items}
    else:
        review = {"total": 0, "items": []}
    review_path.parent.mkdir(parents=True, exist_ok=True)
    if review_path.exists():
        backup_rev = review_path.with_suffix(f".{date.today().strftime('%Y%m%d_%H%M%S')}.bak.json")
        review_path.rename(backup_rev)
    write_json_atomic(review_path, review)
    return data


# ---------------------------------------------------------------------------
# Quality monitoring helpers
# ---------------------------------------------------------------------------


def _make_diag_ctx(q_index, provider, round_label):
    return {"q_index": q_index, "provider": provider, "round": round_label}


def _record_quality(quality, llm_output):
    """Record one LLM response and return whether it passed validation."""
    ok = llm_output is not None and validate_api_question(llm_output)
    quality.check({
        "parse_success": llm_output is not None,
        "validation_passed": ok,
        "content_length": len(str(llm_output)) if llm_output else 0,
    })
    return ok


def _make_single_question(qid, q1, q2, q2_ok, chunk, dim, difficulty, current_provider, review_items):
    """Create a single question entry and optional review item."""
    self_consistent = False
    if q2_ok:
        self_consistent = (q1.get("answer", "").strip() == q2.get("answer", "").strip())
    final_q = make_question_from_llm(q1, qid, chunk, dim, difficulty, provider=current_provider)
    final_q["consistency_status"] = "auto_passed" if self_consistent else "needs_review"
    if not self_consistent and q2:
        review_items.append({
            "id": qid, "dimension": dim, "difficulty": difficulty,
            "question": q1.get("question", ""),
            "round_1_answer": q1.get("answer", ""), "round_1_steps": q1.get("reasoning_steps", []),
            "round_2_answer": q2.get("answer", ""), "round_2_steps": q2.get("reasoning_steps", []),
            "verdict": None,
        })
    return final_q, review_items


def default_rubric(topic):
    """Default scoring rubric for Dimension D (open design)."""
    return {
        "feasibility": {"max_score": 1, "weight": 0.2, "description": f"{topic}方案是否具有工程可实施性"},
        "method_choice": {"max_score": 1, "weight": 0.2, "description": f"控制方法选择是否适合{topic}问题"},
        "completeness": {"max_score": 1, "weight": 0.2, "description": "是否覆盖建模、设计、分析与验证流程"},
        "innovation": {"max_score": 1, "weight": 0.2, "description": "是否包含有价值的设计改进或工程权衡"},
        "clarity": {"max_score": 1, "weight": 0.2, "description": "数学表达和文字说明是否清晰准确"},
    }


def make_question_from_llm(llm_output, qid, chunk, dimension, difficulty, provider="unknown",
                           paper_source=None, valid_categories=None):
    """Build a question dict from LLM output.

    Args:
        valid_categories: Set of allowed control_category values.
            If None, uses control-domain defaults.
    """
    section = chunk.get("source_section") or chunk.get("chunk_id", "")
    raw_cats = llm_output.get("control_category", [])
    if valid_categories is None:
        valid_categories = {"classical", "modern", "nonlinear", "robust", "optimal",
                            "adaptive", "digital", "intelligent", "mpc", "multi_agent"}
    categories = [c for c in raw_cats if c in valid_categories][:3]
    if not categories and valid_categories:
        categories = [next(iter(valid_categories))]

    question = {
        "id": qid,
        "dimension": dimension,
        "difficulty_level": difficulty,
        "control_category": categories,
        "question": llm_output.get("question", ""),
        "answer": llm_output.get("answer", ""),
        "reasoning_steps": llm_output.get("reasoning_steps", []),
        "source_ref": chunk.get("chunk_id", section),
        "model_source": provider,
        "sensitivity_dimension": None,
        "sibling_id": None,
        "rubric": None,
        "consistency_status": "auto_passed",
        "paper_source": paper_source or {},
        "scoring_reason": llm_output.get("scoring_reason", ""),
        "scoring_issues": llm_output.get("scoring_issues", []),
    }

    if dimension == "C":
        question["sensitivity_dimension"] = llm_output.get("sensitivity_dimension", "parameter")

    if dimension == "D":
        question["rubric"] = llm_output.get("rubric") or default_rubric(section)

    return question


# ---------------------------------------------------------------------------
# Run state resolution
# ---------------------------------------------------------------------------


def _resolve_credentials(args):
    from mineru_generator.api import PROVIDER_ENV_MAP, resolve_model
    provider = args.provider
    model = resolve_model(provider, args.model)
    env_spec = PROVIDER_ENV_MAP[provider]
    api_key = args.api_key or first_env(env_spec)
    if not api_key:
        raise SystemExit(
            f"API mode requires --api-key or {env_spec.replace('|', '/')} env var for provider '{provider}'."
        )
    return provider, model, api_key


def _resolve_run_state(args, output_path, review_path):
    start_index = 1
    start_chunk = 0
    resume_questions = None
    if args.resume:
        cp_out = output_path.with_suffix(".checkpoint.json")
        cp_rev = review_path.with_suffix(".checkpoint.json")
        if not cp_out.exists() or not cp_rev.exists():
            raise SystemExit(f"Checkpoint files not found. Expected: {cp_out} and {cp_rev}")
        cp_data = load_json(cp_out)
        resume_questions = cp_data.get("questions", [])
        state = cp_data.get("state", {})
        checkpoint_seed = state.get("seed")
        if checkpoint_seed is not None and int(checkpoint_seed) != args.seed:
            raise SystemExit(f"Checkpoint seed={checkpoint_seed} does not match --seed {args.seed}. Use the original seed or start a new run.")
        start_index = state.get("next_index") or (len(resume_questions) + 1)
        start_chunk = state.get("chunk_index")
        if start_chunk is None:
            start_chunk = start_index - 1
        print(
            f"[Resume] Found {len(resume_questions)} checkpoint questions, "
            f"resuming from q{start_index}, chunk={start_chunk}",
            flush=True,
        )

    if args.supplement > 0:
        if not output_path.exists():
            raise SystemExit(f"Supplement mode requires existing benchmark.json: {output_path}")
        existing = load_json(output_path)
        resume_questions = existing.get("questions", [])
        start_index = len(resume_questions) + 1
        total_limit = len(resume_questions) + args.supplement
        print(f"[Supplement] Adding {args.supplement} to existing {len(resume_questions)} → total {total_limit}", flush=True)
    else:
        total_limit = args.limit

    return start_index, start_chunk, resume_questions, total_limit


def _print_dry_run(args, corpus_dir, selected, total_limit, diag_path, resume_questions):
    print(f"Corpus: {corpus_dir}", flush=True)
    print(f"Usable chunks: {len(selected)}", flush=True)
    requested = total_limit if args.supplement > 0 else args.limit
    print(f"Requested questions: {requested}", flush=True)
    print(f"Mode: {'mock' if args.mock else 'api'}, provider: {args.provider}", flush=True)
    print(f"Seed: {args.seed}", flush=True)
    print(f"Reset every: {args.reset_every} questions", flush=True)
    if diag_path:
        print(f"Diagnostic log: {diag_path}", flush=True)
    if args.resume:
        print(f"Resume: checkpoint={len(resume_questions) if resume_questions else 0} questions", flush=True)
    if args.supplement > 0:
        print(f"Supplement: +{args.supplement} to existing {len(resume_questions) if resume_questions else 0}", flush=True)


def _load_paper_data(args, corpus_dir, domain='control'):
    """Load paper split and registry from disk.

    Args:
        args: Parsed arguments
        corpus_dir: Corpus root directory
        domain: 'control' or 'medical' — selects default split/registry paths
    """
    default_split = "medical_split.json" if domain == "medical" else "paper_split.json"
    split_path = Path(args.split_path) if args.split_path else ROOT / "benchmark" / "dataset" / default_split
    if not split_path.exists():
        raise SystemExit(f"Paper split not found: {split_path}. Run classify_papers.py first.")

    split_data = load_json(split_path)
    papers = split_data.get(args.paper_split, [])
    if not papers:
        raise SystemExit(f"No papers in {args.paper_split}. Available: {list(split_data.keys())}")

    default_reg = "medical_registry.json" if domain == "medical" else "paper_registry.json"
    registry_path = Path(args.registry) if args.registry else ROOT / "benchmark" / "dataset" / default_reg
    registry = {}
    if registry_path.exists():
        reg_data = load_json(registry_path)
        for p in reg_data.get("papers", []):
            registry[_paper_id(p)] = p

    return papers, registry
