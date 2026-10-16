"""Build MedicalSci Sci-Align benchmark questions from medical papers.

Medical domain entry point — pure medical domain with identify/model/transfer
dimensions and fixed L2 difficulty. No domain branching.

Responsibilities:
  - CLI parsing & orchestration (main)
  - Medical paper loading
  - Medical mock question generation
  - Paper-mode orchestration (delegated to build_medical_paper)
"""
import argparse
import random
import sys
from collections import Counter
from datetime import date  # noqa: F401  (kept for consistency)
from pathlib import Path

from mineru_benchmark.pipeline._build_common import (
    ROOT, DEFAULT_SEED,
    _paper_id, load_json,
    build_paper_source, write_outputs,
    _resolve_credentials, _resolve_run_state,
    _load_paper_data,
)
from mineru_generator.medical_distribution import MEDICAL_DIMENSION_CYCLE

# Re-export: paper-mode orchestration (extracted to build_medical_paper.py)
# These symbols are imported for backward compatibility — external callers may import from build_medical.
from mineru_benchmark.pipeline.build_medical_paper import (  # noqa: F401
    MedicalTaskSlot,
    _paper_mode_serial,
    _worker_medical_paper_mode,
    _paper_mode_concurrent,
    _assign_medical_tasks,
)


# ---------------------------------------------------------------------------
# Medical mock generation
# ---------------------------------------------------------------------------


def _paper_mock_generate(papers, limit, start_index, registry_path, corpus_dir):
    """Generate mock medical questions from papers without API calls."""
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
    dim_cycle = list(MEDICAL_DIMENSION_CYCLE)
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
        difficulty = "L2"  # Fixed for medical

        paper_reg = registry.get(pid, paper)
        ps = build_paper_source(paper_reg, corpus_dir)

        qid = f"CS-MED-{index:05d}"
        q = {
            "id": qid,
            "dimension": dim,
            "difficulty_level": difficulty,
            "control_category": ["medical"],
            "question": f"基于论文 '{paper.get('title', pid)}' 中的方法，分析{dim}维度的相关问题。",
            "answer": f"该论文主要研究了{paper.get('title', '相关主题')}。答案应涵盖医学与控制理论的跨域分析。",
            "reasoning_steps": ["识别论文主题", "抽取核心方法", "分析跨域应用"],
            "source_ref": pid,
            "model_source": "mock",
            "sensitivity_dimension": None,
            "sibling_id": None,
            "rubric": None,
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
# Main entry point
# ---------------------------------------------------------------------------


def run_paper_mode(args, corpus_dir, output_path, review_path, diag_path,
                   start_index, resume_questions, total_limit):
    """Medical paper-based generation mode."""
    papers, registry = _load_paper_data(args, corpus_dir, domain='medical')
    print(f"[medical-paper] {args.paper_split}: {len(papers)} papers")
    print(f"[medical-paper] Start index: {args.start_index}, limit: {total_limit}")

    if args.mock:
        registry_path = Path(args.registry) if args.registry else ROOT / "benchmark" / "dataset" / "medical_registry.json"
        questions = _paper_mock_generate(papers, total_limit, args.start_index, registry_path, corpus_dir)
        data = write_outputs(questions, output_path, review_path,
                             source=f"Mock-generated from medical papers ({args.paper_split})")
        print(f"\n[Medical Mock] {len(questions)} questions generated", flush=True)
        print(f"Benchmark saved: {output_path}", flush=True)
        return

    provider, model, api_key = _resolve_credentials(args)
    print(f"[medical-paper] Provider: {provider}, Model: {model}")

    if args.workers > 1 and not args.resume:
        _paper_mode_concurrent(args, papers, corpus_dir, output_path, review_path,
                               diag_path, provider, model, api_key, total_limit)
        return

    if args.workers > 1 and args.resume:
        print("[medical-paper] WARNING: --resume is not yet supported with --workers > 1, falling back to serial.", flush=True)

    _paper_mode_serial(args, papers, corpus_dir, output_path, review_path,
                       diag_path, start_index, resume_questions, total_limit,
                       provider, model, api_key, registry)


def main() -> None:
    args = _parse_args()

    corpus_dir = Path(args.corpus).resolve()
    output_path = Path(args.output).resolve()
    review_path = Path(args.manual_review_output).resolve()

    diag_path = None
    if args.diag_stem:
        diag_path = str(ROOT / "benchmark" / "logs" / "diag" / f"generation_diag_{args.diag_stem}.logl")

    start_index, start_chunk, resume_questions, total_limit = _resolve_run_state(args, output_path, review_path)

    if args.paper_split:
        run_paper_mode(args, corpus_dir, output_path, review_path, diag_path,
                       start_index, resume_questions, total_limit)
        return

    print("[ERROR] Medical domain requires --paper-split (group_a or group_b)", flush=True)
    sys.exit(1)


def _parse_args():
    parser = argparse.ArgumentParser(description="Build MedicalSci Sci-Align benchmark questions from medical papers.")
    parser.add_argument("--corpus", default=str(ROOT / "corpus"), help="Corpus root directory.")
    parser.add_argument("--output", default=str(ROOT / "benchmark" / "dataset" / "medical_benchmark.json"),
                        help="Medical benchmark output JSON path.")
    parser.add_argument("--manual-review-output", default=str(ROOT / "benchmark" / "dataset" / "medical_manual_review.json"),
                        help="Manual review JSON path.")
    parser.add_argument("--model", default=None, help="Model alias or full ID.")
    parser.add_argument("--api-key", default=None, help="API key. Falls back to provider env var.")
    parser.add_argument("--limit", type=int, default=10, help="Number of questions to generate.")
    parser.add_argument("--mock", action="store_true", help="Generate deterministic mock questions without API calls.")
    parser.add_argument("--provider", default="deepseek", choices=["deepseek", "qwen", "mimo"],
                        help="Provider to use for generation.")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from checkpoint files.")
    parser.add_argument("--supplement", type=int, default=0,
                        help="Generate N additional questions and append to existing benchmark.")
    parser.add_argument("--reset-every", type=int, default=15,
                        help="Rebuild system prompt every N questions (default: 15).")
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of concurrent API workers (default 1 = serial).")
    parser.add_argument("--diag-stem", default=None,
                        help="Stem name for diagnostic log file.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help="Random seed for paper ordering.")
    parser.add_argument("--registry", default=None,
                        help="Path to medical_registry.json.")
    parser.add_argument("--paper-split", default=None, choices=["group_a", "group_b"],
                        help="Use medical_split.json groups: group_a or group_b.")
    parser.add_argument("--start-index", type=int, default=1,
                        help="Starting question ID index.")
    parser.add_argument("--split-path", default=None,
                        help="Path to medical_split.json.")
    args = parser.parse_args()

    import mineru_benchmark.pipeline._build_common as common
    common.RUN_SEED = args.seed

    return args


if __name__ == "__main__":
    main()
