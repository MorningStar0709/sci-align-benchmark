"""Shared report building and question selection utilities.

Extracted from evaluate.py to break cross-domain dependencies.
"""

import time
from collections import Counter
from datetime import datetime
from pathlib import Path

from mineru_eval.utils import load_json


def build_eval_record(q, model_answer, result, *,
                     scoring_mode, scorer_version,
                     extra_fields=None):
    """Build a standard eval record dict — domain-agnostic core.

    Args:
        q: Question dict.
        model_answer: Model-generated answer text.
        result: Scoring result dict.
        scoring_mode: Scoring mode identifier (e.g. ``"llm_judge"``).
        scorer_version: Scorer version string.
        extra_fields: Additional keys to merge into the record
            (e.g. ``paper_source``, ``lint_issues``).
    """
    qid = q.get("id", "?")
    dimension = q.get("dimension", "?")
    difficulty = q.get("difficulty_level", "?")
    expected = q.get("answer", "")
    issues = result.get("issues", []) or []
    if not model_answer and "target failure" not in issues:
        issues.append("target failure")

    record = {
        "id": qid,
        "dimension": dimension,
        "difficulty_level": difficulty,
        "score": result["score"],
        "reason": result.get("reason", ""),
        "issues": issues,
        "model_answer": model_answer,
        "expected_answer": expected,
        "scoring_mode": scoring_mode,
        "scorer_version": scorer_version,
    }
    # Optional result-derived fields
    if "rubric_details" in result:
        record["rubric_details"] = result["rubric_details"]
    if result.get("_used_dual", False):
        for key in ("judge1_score", "judge2_score", "avg_score", "score_diff", "disputed", "arbitrated"):
            if key in result:
                record[key] = result[key]
    # Domain-specific extra fields
    if extra_fields:
        for key, value in extra_fields.items():
            if value is not None:
                record[key] = value
    return record


def _compute_dual_judge_stats(records: list[dict]) -> dict | None:
    """Compute dual-judge statistics from records with judge1_score field."""
    dual_records = [r for r in records if "judge1_score" in r]
    if not dual_records:
        return None
    disputed_count = sum(1 for r in dual_records if r.get("disputed"))
    arbitrated_count = sum(1 for r in dual_records if r.get("arbitrated"))
    diffs = [abs(r.get("score_diff", 0)) for r in dual_records]
    by_dim_disputed = Counter()
    for r in dual_records:
        if r.get("disputed"):
            by_dim_disputed[r.get("dimension", "?")] += 1
    return {
        "total_dual_judge": len(dual_records),
        "disputed": disputed_count,
        "arbitrated": arbitrated_count,
        "dispute_rate": round(disputed_count / len(dual_records), 4),
        "mean_score_diff": round(sum(diffs) / len(diffs), 4),
        "disputed_by_dimension": dict(by_dim_disputed),
    }


def build_model_report(records, target_model, judge_model,
                       scorer_version="", scorer_method=""):
    """Build an evaluation report dict from scored records.

    Args:
        records: List of scored record dicts.
        target_model: Name of the evaluated model.
        judge_model: Name of the judge model.
        scorer_version: Version string for the scorer (domain-specific).
        scorer_method: Method string for the scorer (domain-specific).
    """
    issues_counter = Counter()
    failure_counter = Counter()
    by_dimension = Counter()
    score_sums = Counter()

    for record in records:
        dimension = record.get("dimension", "?")
        score = float(record.get("score", 0.0) or 0.0)
        by_dimension[dimension] += 1
        score_sums[dimension] += score
        for issue in record.get("issues", []) or []:
            if issue:
                issues_counter[issue] += 1
                issue_text = str(issue).lower()
                if "failure" in issue_text or "error" in issue_text:
                    failure_counter[issue] += 1

    dimension_scores = {
        dimension: round(score_sums[dimension] / count, 4)
        for dimension, count in sorted(by_dimension.items())
        if count
    }
    overall = round(sum(float(record.get("score", 0.0) or 0.0) for record in records) / len(records), 4) if records else 0.0

    # ── 双 Judge 统计（仅当 records 中包含双 Judge 字段时） ──
    dual_judge_stats = _compute_dual_judge_stats(records)

    result = {
        "model": target_model,
        "judge_model": judge_model,
        "scorer_version": scorer_version,
        "scoring_mode": scorer_method,
        "total_questions": len(records),
        "overall_score": overall,
        "dimension_scores": dimension_scores,
        "records": records,
        "issues_summary": dict(issues_counter.most_common(20)),
        "failure_summary": dict(failure_counter.most_common()),
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    if dual_judge_stats:
        result["dual_judge_stats"] = dual_judge_stats
    return result


def load_existing_records(output_path, selected_ids, retry_failed=False):
    """Load previously computed evaluation records for resume support."""
    output_path = Path(output_path)
    if not output_path.exists():
        return []
    try:
        existing = load_json(output_path)
    except Exception as exc:
        backup_path = output_path.with_suffix(output_path.suffix + f".corrupt.{int(time.time())}.bak")
        output_path.replace(backup_path)
        print(f"Existing report could not be parsed; moved to {backup_path}: {exc}", flush=True)
        return []

    selected_ids = set(selected_ids)
    records = []
    seen = set()
    for record in existing.get("records", []):
        qid = record.get("id")
        if qid not in selected_ids or qid in seen:
            continue
        issues = [str(issue).lower() for issue in (record.get("issues") or [])]
        failed = any("failure" in issue or "error" in issue for issue in issues)
        if retry_failed and failed:
            continue
        records.append(record)
        seen.add(qid)
    return records


def select_questions(questions, limit=0, balanced_mini=False, preferred_dims=None):
    """Select questions from a list, optionally with balanced mini-batch sampling.

    Args:
        questions: List of question dicts.
        limit: Max questions to return (0 = all).
        balanced_mini: If True, sample evenly across dimensions.
        preferred_dims: Ordered list of dimension names for balanced sampling.
            Defaults to control domain ["A", "B", "C", "D"].
    """
    if not balanced_mini:
        return questions[:limit] if limit > 0 else questions

    if limit <= 0:
        raise SystemExit("--balanced-mini requires --limit > 0")

    if preferred_dims is None:
        preferred_dims = ["A", "B", "C", "D"]

    buckets = {}
    for q in questions:
        buckets.setdefault(str(q.get("dimension", "?")), []).append(q)

    ordered = [d for d in preferred_dims if d in buckets]
    ordered.extend(sorted(d for d in buckets if d not in ordered))
    if not ordered:
        return []

    selected = []
    cursor = {d: 0 for d in ordered}
    while len(selected) < limit:
        progressed = False
        for dim in ordered:
            idx = cursor[dim]
            if idx < len(buckets[dim]):
                selected.append(buckets[dim][idx])
                cursor[dim] += 1
                progressed = True
                if len(selected) >= limit:
                    break
        if not progressed:
            break
    return selected
