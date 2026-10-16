"""Medical benchmark dataset validator — CS-MED-ID format verification.

Independent from control science validate_dataset.py.
Validates medical-specific fields: dimension, difficulty, paper_source, consistency.
"""

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path


# ---------------------------------------------------------------------------
# Validation rules
# ---------------------------------------------------------------------------

_ID_PATTERN = re.compile(r"^CS-MED-\d{5}$")
_VALID_DIMENSIONS = {"identify", "model", "transfer"}
_VALID_DIFFICULTIES = {"L1", "L2", "L3", "L4"}
_VALID_CONSISTENCY = {"auto_passed", "needs_review", "reviewed_kept", "reviewed_discarded"}


def _validate_question(q, index):
    """Validate a single medical benchmark question.

    Returns:
        list[str]: List of validation error messages (empty if valid).
    """
    errors = []

    # id
    qid = q.get("id", "")
    if not isinstance(qid, str) or not _ID_PATTERN.match(qid):
        errors.append(f"[{index}] id '{qid}' does not match CS-MED-XXXXX pattern")

    # dimension
    dim = q.get("dimension", "")
    if dim not in _VALID_DIMENSIONS:
        errors.append(f"[{index}] dimension '{dim}' not in {_VALID_DIMENSIONS}")

    # difficulty_level
    diff = q.get("difficulty_level", "")
    if diff not in _VALID_DIFFICULTIES:
        errors.append(f"[{index}] difficulty_level '{diff}' not in {_VALID_DIFFICULTIES}")

    # control_category - for medical domain, accept list with 'medical' or any control theory category
    cat = q.get("control_category", [])
    if not isinstance(cat, list) or len(cat) == 0:
        errors.append(f"[{index}] control_category must be a non-empty list")

    # paper_source
    ps = q.get("paper_source")
    if not isinstance(ps, dict):
        errors.append(f"[{index}] paper_source must be a dict, got {type(ps).__name__}")
    else:
        for key in ("arxiv_id", "title", "paper_path"):
            if key not in ps:
                errors.append(f"[{index}] paper_source missing key '{key}'")

    # answer
    answer = q.get("answer", "")
    if not isinstance(answer, str) or not answer.strip():
        errors.append(f"[{index}] answer must be a non-empty string")

    # reasoning_steps
    steps = q.get("reasoning_steps")
    if not isinstance(steps, list) or len(steps) == 0:
        errors.append(f"[{index}] reasoning_steps must be a non-empty list")

    # consistency_status (optional but validated if present)
    cs = q.get("consistency_status")
    if cs is not None and cs not in _VALID_CONSISTENCY:
        errors.append(f"[{index}] consistency_status '{cs}' not in {_VALID_CONSISTENCY}")

    return errors


# ---------------------------------------------------------------------------
# Batch validation
# ---------------------------------------------------------------------------


def validate_medical_dataset(questions):
    """Validate an entire medical benchmark dataset.

    Args:
        questions: List of question dicts.

    Returns:
        dict with keys:
            - total: int
            - valid: int
            - errors: list[str]
            - dimension_counts: Counter
            - difficulty_counts: Counter
    """
    all_errors = []
    dimension_counts = Counter()
    difficulty_counts = Counter()
    valid_count = 0

    for i, q in enumerate(questions):
        errs = _validate_question(q, i)
        if errs:
            all_errors.extend(errs)
        else:
            valid_count += 1
        dimension_counts[q.get("dimension", "?")] += 1
        difficulty_counts[q.get("difficulty_level", "?")] += 1

    return {
        "total": len(questions),
        "valid": valid_count,
        "errors": all_errors,
        "dimension_counts": dict(dimension_counts),
        "difficulty_counts": dict(difficulty_counts),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Medical benchmark dataset validator — CS-MED-ID format"
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to medical benchmark JSON file",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show per-question validation details",
    )
    args = parser.parse_args()

    path = Path(args.input)
    if not path.exists():
        print(f"ERROR: File not found: {path}", file=sys.stderr)
        sys.exit(1)

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    # Support both raw list and {"questions": [...]} format
    if isinstance(data, list):
        questions = data
    elif isinstance(data, dict):
        questions = data.get("questions", data.get("records", []))
    else:
        print("ERROR: Unexpected JSON structure", file=sys.stderr)
        sys.exit(1)

    result = validate_medical_dataset(questions)

    print("\n=== Medical Dataset Validation ===")
    print(f"File: {path}")
    print(f"Total questions: {result['total']}")
    print(f"Valid: {result['valid']}")
    print(f"Errors: {len(result['errors'])}")

    print("\nDimension distribution:")
    for dim, count in sorted(result["dimension_counts"].items()):
        print(f"  {dim}: {count}")

    print("\nDifficulty distribution:")
    for diff, count in sorted(result["difficulty_counts"].items()):
        print(f"  {diff}: {count}")

    if result["errors"]:
        print("\nValidation errors:")
        for err in result["errors"]:
            print(f"  - {err}")
        sys.exit(1)
    else:
        print("\nAll questions passed validation.")


if __name__ == "__main__":
    main()
