"""Merge benchmark questions with dual-audit + Qwen arbitration.

Dual-audit mode (recommended):
    DS + MM independently audit all questions → disagreements go to Qwen.
    python -m mineru_benchmark.pipeline.merge_dual_audit --dual-audit

Single-source mode (legacy):
    python -m mineru_benchmark.pipeline.merge_dual_audit --skip-audit
"""
from __future__ import annotations

import json
from collections import Counter
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
DEFAULT_QUESTIONS = ROOT / "benchmark" / "dataset" / "core_v3.json"
DEFAULT_MIMO = ROOT / "benchmark" / "dataset" / "gen_mimo.json"
DEFAULT_DEEPSEEK = ROOT / "benchmark" / "dataset" / "gen_deepseek.json"
DEFAULT_AUDIT_BY_DS = ROOT / "benchmark" / "dataset" / "audit_by_ds.json"
DEFAULT_AUDIT_BY_MM = ROOT / "benchmark" / "dataset" / "audit_by_mimo.json"
DEFAULT_ARBITRATION = ROOT / "benchmark" / "dataset" / "audit_arbitration.json"
# Legacy single-audit defaults
DEFAULT_AUDIT_DS = ROOT / "benchmark" / "dataset" / "audit_deepseek_by_mimo.json"
DEFAULT_AUDIT_MM = ROOT / "benchmark" / "dataset" / "audit_mimo_by_deepseek.json"
DEFAULT_OUTPUT = ROOT / "benchmark" / "dataset" / "core_v3.json"


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

def load_questions(path: Path) -> list[dict]:
    """Load questions from a benchmark JSON file."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    questions = data.get("questions", data if isinstance(data, list) else [])
    return questions


def load_audit_results(path: Path) -> dict[str, dict] | None:
    """Load audit results, return {question_id: audit_result} or None if missing."""
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    results = data.get("results", [])
    return {r["id"]: r for r in results if "id" in r}


# ---------------------------------------------------------------------------
# Merge logic
# ---------------------------------------------------------------------------

def annotate_question(q: dict, audit_result: dict | None) -> dict:
    """Add audit metadata to a question (single-audit mode)."""
    q = dict(q)  # shallow copy
    if audit_result is None:
        q["audit_status"] = "not_audited"
    elif audit_result.get("pass") is True:
        q["audit_status"] = "passed"
        q["audit_evidence"] = audit_result.get("evidence", "")
        if audit_result.get("issues"):
            q["audit_issues"] = audit_result["issues"]
    elif audit_result.get("pass") is False:
        q["audit_status"] = "rejected"
        q["audit_evidence"] = audit_result.get("evidence", "")
        q["audit_issues"] = audit_result.get("issues", [])
    else:
        q["audit_status"] = "unknown"
        q["audit_issues"] = audit_result.get("issues", [])
    q["audit_confidence"] = audit_result.get("confidence", 0.0) if audit_result else 0.0
    return q


def annotate_question_dual(
    q: dict,
    ds_result: dict | None,
    mm_result: dict | None,
    arb_result: dict | None = None,
) -> dict:
    """Add dual-audit + optional arbitration metadata to a question."""
    q = dict(q)
    q["audit_ds"] = "passed" if ds_result and ds_result.get("pass") is True else (
        "rejected" if ds_result and ds_result.get("pass") is False else (
            "unknown" if ds_result else "not_audited"))
    q["audit_mm"] = "passed" if mm_result and mm_result.get("pass") is True else (
        "rejected" if mm_result and mm_result.get("pass") is False else (
            "unknown" if mm_result else "not_audited"))
    q["audit_ds_confidence"] = ds_result.get("confidence", 0.0) if ds_result else 0.0
    q["audit_mm_confidence"] = mm_result.get("confidence", 0.0) if mm_result else 0.0

    # Store per-auditor evidence
    if ds_result and ds_result.get("evidence"):
        q["audit_ds_evidence"] = ds_result["evidence"]
    if mm_result and mm_result.get("evidence"):
        q["audit_mm_evidence"] = mm_result["evidence"]
    if ds_result and ds_result.get("issues"):
        q["audit_ds_issues"] = ds_result["issues"]
    if mm_result and mm_result.get("issues"):
        q["audit_mm_issues"] = mm_result["issues"]

    if arb_result and arb_result.get("pass") is not None:
        q["audit_arb"] = "passed" if arb_result.get("pass") else "rejected"
        q["audit_arb_confidence"] = arb_result.get("confidence", 0.0)
        q["audit_arb_evidence"] = arb_result.get("evidence", "")
        if arb_result.get("issues"):
            q["audit_arb_issues"] = arb_result["issues"]
    elif arb_result:
        # Arbitration attempted but failed (pass=None)
        q["audit_arb"] = "error"
        if arb_result.get("issues"):
            q["audit_arb_issues"] = arb_result["issues"]

    return q


def merge_with_dual_audit(
    questions: list[dict],
    audit_ds: dict[str, dict] | None,
    audit_mm: dict[str, dict] | None,
    arbitration: dict[str, dict] | None = None,
) -> list[dict]:
    """Merge questions with dual-audit comparison + optional Qwen arbitration.

    Logic:
        Both pass       → passed
        Both reject     → rejected
        One pass + one reject → disputed (if no arbitration) / resolved (if arbitration)
        Missing audit   → not_audited (kept for review)
    """
    merged = []
    stats = {"passed": 0, "rejected": 0, "disputed": 0, "not_audited": 0, "unknown": 0}

    for q in questions:
        qid = q.get("id", "")
        ds_r = audit_ds.get(qid) if audit_ds else None
        mm_r = audit_mm.get(qid) if audit_mm else None
        arb_r = arbitration.get(qid) if arbitration else None

        annotated = annotate_question_dual(q, ds_r, mm_r, arb_r)

        # No audit data at all
        if ds_r is None and mm_r is None:
            annotated["audit_status"] = "not_audited"
            stats["not_audited"] += 1
            merged.append(annotated)
            continue

        ds_pass = ds_r.get("pass") if ds_r else None
        mm_pass = mm_r.get("pass") if mm_r else None

        # Both agree pass
        if ds_pass is True and mm_pass is True:
            annotated["audit_status"] = "passed"
            stats["passed"] += 1
        # Both agree reject
        elif ds_pass is False and mm_pass is False:
            annotated["audit_status"] = "rejected"
            stats["rejected"] += 1
        # Disagreement → check arbitration
        elif ds_pass is not None and mm_pass is not None and ds_pass != mm_pass:
            if arb_r and arb_r.get("pass") is not None:
                annotated["audit_status"] = "passed" if arb_r.get("pass") else "rejected"
                stats[annotated["audit_status"]] += 1
            else:
                annotated["audit_status"] = "disputed"
                stats["disputed"] += 1
        else:
            annotated["audit_status"] = "unknown"
            stats["unknown"] += 1

        merged.append(annotated)

    merged.sort(key=lambda q: q.get("id", ""))

    print(f"[merge] Dual-audit: passed={stats['passed']} rejected={stats['rejected']} "
          f"disputed={stats['disputed']} not_audited={stats['not_audited']} unknown={stats['unknown']}")
    return merged


def merge_dual_audit(
    mimo_qs: list[dict],
    deepseek_qs: list[dict],
    audit_ds: dict[str, dict] | None,
    audit_mm: dict[str, dict] | None,
    skip_audit: bool = False,
    min_confidence: float = 0.0,
) -> list[dict]:
    """Merge questions from both sources with optional audit filtering.

    min_confidence: only reject questions where auditor confidence >= this threshold.
        Questions with pass=False but confidence < threshold are kept as "disputed".
    """
    merged = []
    rejected = 0
    disputed = 0
    unknown = 0

    for q in deepseek_qs:
        qid = q.get("id", "")
        ar = audit_ds.get(qid) if audit_ds else None
        annotated = annotate_question(q, ar)
        if skip_audit or annotated["audit_status"] in ("not_audited", "passed"):
            merged.append(annotated)
        elif annotated["audit_status"] == "rejected":
            conf = annotated.get("audit_confidence", 0.0)
            if min_confidence > 0 and conf < min_confidence:
                annotated["audit_status"] = "disputed"
                disputed += 1
                merged.append(annotated)
            else:
                rejected += 1
        else:
            unknown += 1
            merged.append(annotated)  # keep unknowns for review

    for q in mimo_qs:
        qid = q.get("id", "")
        ar = audit_mm.get(qid) if audit_mm else None
        annotated = annotate_question(q, ar)
        if skip_audit or annotated["audit_status"] in ("not_audited", "passed"):
            merged.append(annotated)
        elif annotated["audit_status"] == "rejected":
            conf = annotated.get("audit_confidence", 0.0)
            if min_confidence > 0 and conf < min_confidence:
                annotated["audit_status"] = "disputed"
                disputed += 1
                merged.append(annotated)
            else:
                rejected += 1
        else:
            unknown += 1
            merged.append(annotated)

    # Sort by ID
    merged.sort(key=lambda q: q.get("id", ""))

    return merged


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def build_output(questions: list[dict], skip_audit: bool, dual_audit: bool = False) -> dict:
    """Build the core_v3.json output structure."""
    dim_counts = Counter(q.get("dimension", "?") for q in questions)
    diff_counts = Counter(q.get("difficulty_level", "?") for q in questions)
    source_counts = Counter(q.get("model_source", "?") for q in questions)
    audit_counts = Counter(q.get("audit_status", "not_audited") for q in questions)

    if skip_audit:
        audit_mode = "skipped"
    elif dual_audit:
        audit_mode = "dual-audit+arbitration"
    else:
        audit_mode = "cross-model"

    meta = {
        "project": "FR-LoRA — Format-Robust Low-Rank Adaptation",
        "source": "AI-generated via deepseek+mimo dual-blind + dual-audit + qwen-arbitration" if dual_audit else (
            "AI-generated via deepseek+mimo dual-blind (skip-audit)" if skip_audit else
            "AI-generated via deepseek+mimo dual-blind + cross-audit"),
        "version": "3.0",
        "updated": str(date.today()),
        "total_questions": len(questions),
        "dimensions": {k: dim_counts[k] for k in "ABCD" if k in dim_counts},
        "difficulty": dict(sorted(diff_counts.items())),
        "sources": dict(sorted(source_counts.items())),
        "audit_mode": audit_mode,
        "audit_summary": dict(sorted(audit_counts.items())),
    }
    return {"meta": meta, "questions": questions}


def print_report(data: dict) -> None:
    """Print merge summary to stdout."""
    meta = data["meta"]
    qs = data["questions"]

    print(f"\n{'='*60}")
    print("  core_v3.json  Merge Report")
    print(f"{'='*60}")
    print(f"  Total:    {meta['total_questions']} questions")
    print(f"  Audit:    {meta['audit_mode']}")
    print()

    print(f"  Dimensions:  A={meta['dimensions'].get('A',0)}  "
          f"B={meta['dimensions'].get('B',0)}  "
          f"C={meta['dimensions'].get('C',0)}  "
          f"D={meta['dimensions'].get('D',0)}")

    print(f"  Difficulty:  {meta['difficulty']}")
    print(f"  Sources:     {meta['sources']}")

    if meta["audit_mode"] != "skipped":
        print(f"  Audit:       {meta['audit_summary']}")

    # Check paper_source coverage
    ps = sum(1 for q in qs if q.get("paper_source"))
    to = sum(1 for q in qs if q.get("paper_source", {}).get("text_offset") is not None)
    print(f"\n  paper_source:  {ps}/{len(qs)} ({ps*100//max(len(qs),1)}%)")
    print(f"  text_offset:   {to}/{len(qs)} ({to*100//max(len(qs),1)}%)")

    # Dimension × Difficulty matrix
    matrix: dict[str, Counter] = {}
    for q in qs:
        dim = q.get("dimension", "?")
        diff = q.get("difficulty_level", "?")
        matrix.setdefault(dim, Counter())
        matrix[dim][diff] += 1

    print("\n  Dim × Diff Matrix:")
    diffs = ["L1", "L2", "L3", "L4"]
    header = f"  {'':>4}" + "".join(f"{d:>6}" for d in diffs) + f"{'Total':>7}"
    print(header)
    for dim in "ABCD":
        c = matrix.get(dim, Counter())
        row = f"  {dim:>4}"
        total = 0
        for d in diffs:
            v = c.get(d, 0)
            total += v
            row += f"{v:>6}"
        row += f"{total:>7}"
        print(row)
    print(f"{'='*60}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Merge dual-source benchmark questions into core_v3.json"
    )
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS,
                        help=f"Merged questions pool (default: {DEFAULT_QUESTIONS.name})")
    parser.add_argument("--mimo", type=Path, default=DEFAULT_MIMO,
                        help=f"MiMo questions — legacy mode (default: {DEFAULT_MIMO.name})")
    parser.add_argument("--deepseek", type=Path, default=DEFAULT_DEEPSEEK,
                        help=f"DeepSeek questions — legacy mode (default: {DEFAULT_DEEPSEEK.name})")
    # Dual-audit flags
    parser.add_argument("--audit-by-ds", type=Path, default=DEFAULT_AUDIT_BY_DS,
                        help="DS audit of all questions (default: audit_by_ds.json)")
    parser.add_argument("--audit-by-mm", type=Path, default=DEFAULT_AUDIT_BY_MM,
                        help="MM audit of all questions (default: audit_by_mimo.json)")
    parser.add_argument("--arbitration", type=Path, default=DEFAULT_ARBITRATION,
                        help="Qwen arbitration results (default: audit_arbitration.json)")
    # Legacy single-audit flags
    parser.add_argument("--audit-ds", type=Path, default=DEFAULT_AUDIT_DS,
                        help="[legacy] Audit of DeepSeek by MiMo")
    parser.add_argument("--audit-mm", type=Path, default=DEFAULT_AUDIT_MM,
                        help="[legacy] Audit of MiMo by DeepSeek")
    parser.add_argument("--output", "-o", type=Path, default=DEFAULT_OUTPUT,
                        help=f"Output path (default: {DEFAULT_OUTPUT.name})")
    parser.add_argument("--dual-audit", action="store_true",
                        help="Use dual-audit mode (DS+MM both audit all, Qwen arbitrates)")
    parser.add_argument("--skip-audit", action="store_true",
                        help="Skip audit filtering — merge all questions")
    parser.add_argument("--min-confidence", type=float, default=0.0,
                        help="[legacy] Min confidence to reject (default 0.0 = strict)")
    args = parser.parse_args()

    # Resolve paths
    questions_path = args.questions if args.questions.is_absolute() else ROOT / args.questions
    mimo_path = args.mimo if args.mimo.is_absolute() else ROOT / args.mimo
    ds_path = args.deepseek if args.deepseek.is_absolute() else ROOT / args.deepseek
    audit_by_ds_path = args.audit_by_ds if args.audit_by_ds.is_absolute() else ROOT / args.audit_by_ds
    audit_by_mm_path = args.audit_by_mm if args.audit_by_mm.is_absolute() else ROOT / args.audit_by_mm
    arbitration_path = args.arbitration if args.arbitration.is_absolute() else ROOT / args.arbitration
    audit_ds_path = args.audit_ds if args.audit_ds.is_absolute() else ROOT / args.audit_ds
    audit_mm_path = args.audit_mm if args.audit_mm.is_absolute() else ROOT / args.audit_mm
    output_path = args.output if args.output.is_absolute() else ROOT / args.output

    # --- Dual-audit mode ---
    if args.dual_audit:
        if not questions_path.exists():
            print(f"[ERROR] Questions file not found: {questions_path}")
            return 1
        questions = load_questions(questions_path)
        print(f"[merge] Questions: {len(questions)}")

        audit_ds = load_audit_results(audit_by_ds_path)
        audit_mm = load_audit_results(audit_by_mm_path)
        arbitration = load_audit_results(arbitration_path) if arbitration_path.exists() else None

        ds_status = f"{len(audit_ds)} results" if audit_ds else "NOT FOUND"
        mm_status = f"{len(audit_mm)} results" if audit_mm else "NOT FOUND"
        arb_status = f"{len(arbitration)} results" if arbitration else "NOT FOUND"
        print(f"[merge] Audit by DS: {ds_status}")
        print(f"[merge] Audit by MM: {mm_status}")
        print(f"[merge] Arbitration: {arb_status}")

        if audit_ds is None and audit_mm is None:
            print("[ERROR] No audit files found. Run cross_audit.py first.")
            return 1

        merged = merge_with_dual_audit(questions, audit_ds, audit_mm, arbitration)
        output_data = build_output(merged, skip_audit=False, dual_audit=True)

    # --- Skip-audit mode (merge from gen files) ---
    elif args.skip_audit:
        for p, label in [(mimo_path, "MiMo"), (ds_path, "DeepSeek")]:
            if not p.exists():
                print(f"[ERROR] {label} file not found: {p}")
                return 1
        mimo_qs = load_questions(mimo_path)
        deepseek_qs = load_questions(ds_path)
        print(f"[merge] Mode: skip-audit | MiMo: {len(mimo_qs)} | DeepSeek: {len(deepseek_qs)}")
        merged = merge_dual_audit(mimo_qs, deepseek_qs, None, None, skip_audit=True)
        output_data = build_output(merged, skip_audit=True)

    # --- Legacy single-audit mode ---
    else:
        for p, label in [(mimo_path, "MiMo"), (ds_path, "DeepSeek")]:
            if not p.exists():
                print(f"[ERROR] {label} file not found: {p}")
                return 1
        mimo_qs = load_questions(mimo_path)
        deepseek_qs = load_questions(ds_path)
        audit_ds = load_audit_results(audit_ds_path)
        audit_mm = load_audit_results(audit_mm_path)
        print(f"[merge] Mode: legacy | MiMo: {len(mimo_qs)} | DeepSeek: {len(deepseek_qs)}")
        if audit_ds is None and audit_mm is None:
            print("[WARNING] No audit files found. All questions will be marked as not_audited.")
        merged = merge_dual_audit(mimo_qs, deepseek_qs, audit_ds, audit_mm,
                                  False, args.min_confidence)
        output_data = build_output(merged, skip_audit=False)

    # Write
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"[merge] Output: {output_path}")
    print_report(output_data)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
