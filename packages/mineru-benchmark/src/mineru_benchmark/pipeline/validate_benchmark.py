import argparse
import json
import re
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[5]  # restored Phase 5
DEFAULT_INPUT = ROOT / "benchmark" / "dataset" / "benchmark.json"
DEFAULT_STATS = ROOT / "benchmark" / "dataset" / "benchmark_stats.json"

ALLOWED_DIMENSIONS = {"A", "B", "C", "D"}
ALLOWED_DIFFICULTIES = {"L1", "L2", "L3", "L4"}
ALLOWED_CATEGORIES = {
    "classical",
    "modern",
    "nonlinear",
    "robust",
    "optimal",
    "adaptive",
    "digital",
    "intelligent",
    "mpc",
    "multi_agent",
}
ALLOWED_SENSITIVITY = {"parameter", "environment", "constraint"}
ALLOWED_STATUS = {"auto_passed", "needs_review", "reviewed_kept", "reviewed_discarded"}
RUBRIC_KEYS = {"feasibility", "method_choice", "completeness", "innovation", "clarity"}
ID_RE = re.compile(r"^CS-EVO-\d{5}$")


def load_json(path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def add_error(errors, qid, field, message):
    errors.append({"id": qid or "<root>", "field": field, "message": message})


def validate_root(data, errors):
    if not isinstance(data, dict):
        add_error(errors, None, "$", "benchmark must be a JSON object")
        return []

    meta = data.get("meta")
    questions = data.get("questions")
    if not isinstance(meta, dict):
        add_error(errors, None, "meta", "meta must be an object")
    else:
        for field in ["project", "version", "updated", "total_questions", "dimensions", "source"]:
            if field not in meta:
                add_error(errors, None, f"meta.{field}", "missing required meta field")
        if isinstance(meta.get("total_questions"), int) and isinstance(questions, list):
            if meta["total_questions"] != len(questions):
                add_error(errors, None, "meta.total_questions", "does not match questions length")

    if not isinstance(questions, list):
        add_error(errors, None, "questions", "questions must be an array")
        return []
    return questions


def validate_categories(errors, qid, categories):
    if not isinstance(categories, list):
        add_error(errors, qid, "control_category", "must be an array")
        return
    if not 1 <= len(categories) <= 3:
        add_error(errors, qid, "control_category", "must contain 1 to 3 categories")
    if len(categories) != len(set(categories)):
        add_error(errors, qid, "control_category", "must not contain duplicates")
    for category in categories:
        if category not in ALLOWED_CATEGORIES:
            add_error(errors, qid, "control_category", f"unknown category: {category}")


def validate_rubric(errors, qid, rubric):
    if not isinstance(rubric, dict):
        add_error(errors, qid, "rubric", "D dimension requires a rubric object")
        return
    missing = RUBRIC_KEYS - set(rubric)
    extra = set(rubric) - RUBRIC_KEYS
    if missing:
        add_error(errors, qid, "rubric", f"missing rubric items: {sorted(missing)}")
    if extra:
        add_error(errors, qid, "rubric", f"unknown rubric items: {sorted(extra)}")
    for key in RUBRIC_KEYS & set(rubric):
        item = rubric[key]
        if not isinstance(item, dict):
            add_error(errors, qid, f"rubric.{key}", "must be an object")
            continue
        for field in ["max_score", "weight", "description"]:
            if field not in item:
                add_error(errors, qid, f"rubric.{key}.{field}", "missing required field")
        if "max_score" in item and not isinstance(item["max_score"], (int, float)):
            add_error(errors, qid, f"rubric.{key}.max_score", "must be numeric")
        if "weight" in item and not isinstance(item["weight"], (int, float)):
            add_error(errors, qid, f"rubric.{key}.weight", "must be numeric")
        if "description" in item and not isinstance(item["description"], str):
            add_error(errors, qid, f"rubric.{key}.description", "must be string")


def _validate_required_fields(errors, qid, question):
    """Validate required fields are present and string fields are non-empty."""
    required = [
        "id", "dimension", "difficulty_level", "control_category",
        "question", "answer", "reasoning_steps", "source_ref", "consistency_status",
        "paper_source", "scoring_reason", "scoring_issues",
    ]
    for field in required:
        if field not in question:
            add_error(errors, qid, field, "missing required field")

    for field in ["question", "answer", "source_ref"]:
        value = question.get(field)
        if not isinstance(value, str) or not value.strip():
            add_error(errors, qid, field, "must be a non-empty string")

    # paper_source validation
    ps = question.get("paper_source")
    if not isinstance(ps, dict):
        add_error(errors, qid, "paper_source", "must be an object with arxiv_id, title, paper_path")
    else:
        for ps_field in ["arxiv_id", "title", "paper_path"]:
            if not ps.get(ps_field):
                add_error(errors, qid, f"paper_source.{ps_field}", "missing or empty")

    # scoring_reason validation
    sr = question.get("scoring_reason")
    if not isinstance(sr, str) or not sr.strip():
        add_error(errors, qid, "scoring_reason", "must be a non-empty string")

    # scoring_issues validation
    si = question.get("scoring_issues")
    if not isinstance(si, list):
        add_error(errors, qid, "scoring_issues", "must be a list of strings")
    elif len(si) == 0:
        # Allow empty list (some dimensions may not have obvious issues)
        pass


def _validate_dimension_checks(errors, qid, question):
    """Validate dimension-specific fields (C: sensitivity+sibling, D: rubric)."""
    dimension = question.get("dimension")
    if dimension == "C":
        if question.get("sensitivity_dimension") not in ALLOWED_SENSITIVITY:
            add_error(errors, qid, "sensitivity_dimension", "C dimension requires parameter/environment/constraint")
        sibling_id = question.get("sibling_id")
        if sibling_id is not None and (not isinstance(sibling_id, str) or not ID_RE.fullmatch(sibling_id)):
            add_error(errors, qid, "sibling_id", "C dimension requires a valid sibling_id or null")
    elif question.get("sensitivity_dimension") not in (None, ""):
        add_error(errors, qid, "sensitivity_dimension", "only C dimension should set sensitivity_dimension")

    if dimension == "D":
        validate_rubric(errors, qid, question.get("rubric"))
    elif question.get("rubric") not in (None, {}):
        add_error(errors, qid, "rubric", "only D dimension should set rubric")


def validate_question(question, errors):
    if not isinstance(question, dict):
        add_error(errors, None, "questions[]", "question must be an object")
        return None

    qid = question.get("id")
    if not isinstance(qid, str) or not ID_RE.fullmatch(qid):
        add_error(errors, qid, "id", "must match CS-EVO-{NNNNN}")

    _validate_required_fields(errors, qid, question)

    dimension = question.get("dimension")
    if dimension not in ALLOWED_DIMENSIONS:
        add_error(errors, qid, "dimension", f"must be one of {sorted(ALLOWED_DIMENSIONS)}")
    if question.get("difficulty_level") not in ALLOWED_DIFFICULTIES:
        add_error(errors, qid, "difficulty_level", f"must be one of {sorted(ALLOWED_DIFFICULTIES)}")
    validate_categories(errors, qid, question.get("control_category"))

    steps = question.get("reasoning_steps")
    if not isinstance(steps, list) or not steps:
        add_error(errors, qid, "reasoning_steps", "must be a non-empty array")
    elif any(not isinstance(step, str) or not step.strip() for step in steps):
        add_error(errors, qid, "reasoning_steps", "all steps must be non-empty strings")

    if question.get("consistency_status") not in ALLOWED_STATUS:
        add_error(errors, qid, "consistency_status", f"must be one of {sorted(ALLOWED_STATUS)}")

    _validate_dimension_checks(errors, qid, question)
    return qid


def validate_cross_references(questions, errors):
    by_id = {q.get("id"): q for q in questions if isinstance(q, dict) and isinstance(q.get("id"), str)}
    ids = [q.get("id") for q in questions if isinstance(q, dict)]
    for qid, count in Counter(ids).items():
        if qid and count > 1:
            add_error(errors, qid, "id", "duplicate id")

    for question in questions:
        if not isinstance(question, dict) or question.get("dimension") != "C":
            continue
        qid = question.get("id")
        sibling_id = question.get("sibling_id")
        if sibling_id is None:
            continue  # concurrent-mode C questions have no sibling
        sibling = by_id.get(sibling_id)
        if not sibling:
            add_error(errors, qid, "sibling_id", "sibling question not found")
            continue
        if sibling.get("sibling_id") != qid:
            add_error(errors, qid, "sibling_id", "sibling reference is not bidirectional")
        if sibling.get("dimension") != "C":
            add_error(errors, qid, "sibling_id", "sibling must also be C dimension")


def build_stats(questions):
    dimension_counts = Counter(q.get("dimension") for q in questions if isinstance(q, dict))
    difficulty_counts = Counter(q.get("difficulty_level") for q in questions if isinstance(q, dict))
    status_counts = Counter(q.get("consistency_status") for q in questions if isinstance(q, dict))
    category_counts = Counter()
    paper_source_filled = 0
    scoring_reason_filled = 0
    for question in questions:
        if isinstance(question, dict):
            category_counts.update(question.get("control_category") or [])
            ps = question.get("paper_source", {})
            if isinstance(ps, dict) and ps.get("arxiv_id"):
                paper_source_filled += 1
            sr = question.get("scoring_reason", "")
            if isinstance(sr, str) and sr.strip():
                scoring_reason_filled += 1

    total = max(len(questions), 1)
    return {
        "total_questions": len(questions),
        "dimensions": dict(sorted(dimension_counts.items())),
        "difficulty_levels": dict(sorted(difficulty_counts.items())),
        "control_categories": dict(sorted(category_counts.items())),
        "consistency_status": dict(sorted(status_counts.items())),
        "paper_source_coverage": round(paper_source_filled / total * 100, 1),
        "scoring_reason_coverage": round(scoring_reason_filled / total * 100, 1),
    }


def validate_file(input_path):
    data = load_json(input_path)
    errors = []
    questions = validate_root(data, errors)
    for question in questions:
        validate_question(question, errors)
    validate_cross_references(questions, errors)
    stats = build_stats(questions)
    return errors, stats


def main():
    parser = argparse.ArgumentParser(description="Validate ControlSci Sci-Align benchmark JSON.")
    parser.add_argument("--input", "-i", default=str(DEFAULT_INPUT), help="Benchmark JSON path.")
    parser.add_argument("--stats-output", default=None, help="Optional benchmark_stats.json output path.")
    parser.add_argument("--max-errors", type=int, default=50, help="Maximum errors to print.")
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    if not input_path.exists():
        raise SystemExit(f"Input not found: {input_path}")

    errors, stats = validate_file(input_path)
    if args.stats_output:
        stats_output = Path(args.stats_output).resolve()
        stats_output.parent.mkdir(parents=True, exist_ok=True)
        stats_output.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Stats saved: {stats_output}")

    if errors:
        print(f"Validation failed: {len(errors)} error(s)")
        for error in errors[: args.max_errors]:
            print(f"- {error['id']} {error['field']}: {error['message']}")
        if len(errors) > args.max_errors:
            print(f"... {len(errors) - args.max_errors} more error(s)")
        raise SystemExit(1)

    print("Validation passed.")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
