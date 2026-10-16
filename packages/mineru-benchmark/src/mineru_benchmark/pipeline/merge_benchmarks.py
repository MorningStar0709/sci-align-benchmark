"""Merge parallel benchmark checkpoints into a renumbered candidate pool."""
import argparse
import json
import re
from collections import Counter
from datetime import date
from pathlib import Path


ROOT = Path(__file__).resolve().parents[5]  # restored Phase 5
DEFAULT_OUTPUT = ROOT / "benchmark" / "dataset" / "benchmark_candidate.json"
DEFAULT_REVIEW = ROOT / "benchmark" / "dataset" / "manual_review_candidate.json"
DEFAULT_REPORT = ROOT / "benchmark" / "dataset" / "candidate_merge_report.json"


def _natural_key(p):
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', p.name)]


def load_json(path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    try:
        tmp_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp_path.replace(path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def default_inputs():
    benchmark_dir = ROOT / "benchmark" / "dataset"
    paths = []
    for pattern in [
        "benchmark.checkpoint.json",
        "benchmark_*part*.checkpoint.json",
        "benchmark_*part*.json",
    ]:
        paths.extend(sorted(benchmark_dir.glob(pattern), key=_natural_key))

    for pattern in [
        "benchmark_ds_p*.json",
        "benchmark_mm_p*.json",
        "benchmark_mimo_p*.json",
    ]:
        for p in sorted(benchmark_dir.glob(pattern), key=_natural_key):
            if not p.name.endswith(".checkpoint.json"):
                paths.append(p)

    for pattern in [
        "benchmark_ds_p*.checkpoint.json",
        "benchmark_mm_p*.checkpoint.json",
        "benchmark_mimo_p*.checkpoint.json",
    ]:
        paths.extend(sorted(benchmark_dir.glob(pattern), key=_natural_key))

    return unique_paths(paths)


def unique_paths(paths):
    seen = set()
    result = []
    for path in paths:
        p = Path(path).resolve()
        if p not in seen:
            seen.add(p)
            result.append(p)
    return result


def review_path_for(benchmark_path):
    name = benchmark_path.name
    if name.startswith("benchmark"):
        return benchmark_path.with_name(name.replace("benchmark", "manual_review", 1))
    return None


def normalize_review_item(item, new_id, question):
    return {
        "id": new_id,
        "dimension": question.get("dimension"),
        "difficulty": question.get("difficulty_level"),
        "question": question.get("question", item.get("question", "")),
        "round_1_answer": item.get("round_1_answer", ""),
        "round_1_steps": item.get("round_1_steps", []),
        "round_2_answer": item.get("round_2_answer", ""),
        "round_2_steps": item.get("round_2_steps", []),
        "verdict": item.get("verdict"),
    }


def load_review_items(benchmark_path):
    review_path = review_path_for(benchmark_path)
    if not review_path or not review_path.exists():
        return {}
    try:
        review_data = load_json(review_path)
    except (OSError, json.JSONDecodeError):
        return {}
    items = review_data.get("items", [])
    return {item.get("id"): item for item in items if isinstance(item, dict) and item.get("id")}


def question_fingerprint(question):
    return (
        (question.get("question") or "").strip(),
        (question.get("answer") or "").strip(),
        question.get("source_ref") or "",
        question.get("dimension") or "",
    )


def new_qid(index):
    return f"CS-EVO-{index:05d}"


def assign_sibling_ids(questions):
    id_to_q = {q["id"]: q for q in questions if q.get("id")}
    paired_ids = set()

    for q in questions:
        if q.get("dimension") != "C":
            q["sibling_id"] = None
            continue
        sid = q.get("sibling_id")
        if not sid or sid not in id_to_q or sid == q["id"]:
            continue
        target = id_to_q[sid]
        if target.get("dimension") != "C":
            continue
        if target.get("sibling_id") != q["id"]:
            continue
        if sid in paired_ids or q["id"] in paired_ids:
            continue
        paired_ids.add(q["id"])
        paired_ids.add(sid)

    pending = None
    orphan_ids = []
    pairs = len(paired_ids) // 2

    for q in questions:
        if q.get("dimension") != "C":
            continue
        if q["id"] in paired_ids:
            continue
        if pending is None:
            pending = q
            q["sibling_id"] = None
            continue
        pending["sibling_id"] = q["id"]
        q["sibling_id"] = pending["id"]
        pairs += 1
        pending = None

    if pending is not None:
        orphan_ids.append(pending["id"])
        pending["sibling_id"] = None

    return pairs, orphan_ids


def renumber_questions(questions, review_items):
    """Renumber question/review IDs after orphan C-dimension removal.

    NOTE: review_items here have already been normalized via normalize_review_item()
    in the merge loop, so we only need to remap the 'id' field. If normalize_review_item()
    adds new fields in the future, this function MUST be updated to match.
    """
    id_map = {question["id"]: new_qid(i) for i, question in enumerate(questions, start=1)}
    for question in questions:
        old_id = question["id"]
        sibling_id = question.get("sibling_id")
        question["id"] = id_map[old_id]
        question["sibling_id"] = id_map.get(sibling_id) if sibling_id else None

    normalized_reviews = []
    for item in review_items:
        old_id = item.get("id")
        if old_id not in id_map:
            continue
        updated = dict(item)
        updated["id"] = id_map[old_id]
        normalized_reviews.append(updated)

    return id_map, normalized_reviews


def build_meta(questions, sources, duplicates_removed):
    dimensions = {d: 0 for d in "ABCD"}
    for question in questions:
        dim = question.get("dimension")
        if dim in dimensions:
            dimensions[dim] += 1
    return {
        "project": "ControlSci — 控制科学结构化语料库与 Sci-Align 跨模态对齐评测基准",
        "version": "1.0-candidate",
        "updated": str(date.today()),
        "total_questions": len(questions),
        "dimensions": dimensions,
        "source": "Merged from parallel generation checkpoints; ids renumbered",
        "source_splits": sources,
        "duplicates_removed": duplicates_removed,
    }


def _merge_single_file(path, seen_fingerprints, original_id_counts, id_map, review_items, sources, questions):
    """Merge questions from a single benchmark file, deduplicating by fingerprint.
    Returns (added, duplicates_found).
    """
    try:
        data = load_json(path)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"  [skip] {path.name}: {exc}")
        return 0, 0

    file_questions = data.get("questions", [])
    if not isinstance(file_questions, list):
        print(f"  [skip] {path.name}: questions is not a list")
        return 0, 0

    reviews_by_id = load_review_items(path)
    added = 0
    duplicates = 0
    for question in file_questions:
        if not isinstance(question, dict):
            continue
        fingerprint = question_fingerprint(question)
        if fingerprint in seen_fingerprints:
            duplicates += 1
            continue
        seen_fingerprints.add(fingerprint)

        old_id = question.get("id") or ""
        original_id_counts[old_id] += 1
        new_id = new_qid(len(questions) + 1)
        id_map[f"{path.name}:{old_id}"] = new_id

        merged_question = dict(question)
        merged_question["id"] = new_id
        questions.append(merged_question)
        added += 1

        review_item = reviews_by_id.get(old_id)
        if review_item:
            review_items.append(normalize_review_item(review_item, new_id, merged_question))

    sources[path.name] = {"input_questions": len(file_questions), "added_questions": added}
    print(f"  [merge] {path.name}: {len(file_questions)} input, {added} added")
    return added, duplicates


def merge_benchmarks(input_paths, output_path=DEFAULT_OUTPUT, review_output=DEFAULT_REVIEW, report_output=DEFAULT_REPORT):
    questions = []
    review_items = []
    sources = {}
    duplicate_fingerprints = 0
    seen_fingerprints = set()
    id_map = {}
    original_id_counts = Counter()

    for path in unique_paths(input_paths):
        if not path.exists():
            print(f"  [skip] missing: {path}")
            continue
        before = len(questions)
        _, dup = _merge_single_file(path, seen_fingerprints, original_id_counts, id_map, review_items, sources, questions)
        duplicate_fingerprints += dup
        _ = before  # silence unused

    c_pairs, orphan_ids = assign_sibling_ids(questions)
    if orphan_ids:
        orphan_set = set(orphan_ids)
        questions = [question for question in questions if question.get("id") not in orphan_set]
        review_items = [item for item in review_items if item.get("id") not in orphan_set]
        final_id_map, review_items = renumber_questions(questions, review_items)
        for source_key, current_id in list(id_map.items()):
            if current_id in orphan_set:
                id_map[source_key] = None
            else:
                id_map[source_key] = final_id_map.get(current_id, current_id)

    meta = build_meta(questions, sources, duplicate_fingerprints)
    candidate = {"meta": meta, "questions": questions}
    review = {"total": len(review_items), "items": review_items}

    report = {
        "input_files": len(sources),
        "total_questions": len(questions),
        "dimensions": meta["dimensions"],
        "review_items": len(review_items),
        "c_pairs": c_pairs,
        "orphan_c_questions_removed": len(orphan_ids),
        "duplicate_fingerprints_removed": duplicate_fingerprints,
        "original_duplicate_ids": sum(1 for count in original_id_counts.values() if count > 1),
        "source_splits": sources,
        "id_map": id_map,
    }

    write_json(Path(output_path), candidate)
    write_json(Path(review_output), review)
    write_json(Path(report_output), report)

    print(f"\n  [done] candidate: {output_path} ({len(questions)} questions)")
    print(f"  [done] manual review: {review_output} ({len(review_items)} items)")
    print(f"  [done] report: {report_output}")
    return report


def main():
    parser = argparse.ArgumentParser(description="Merge parallel benchmark checkpoints into one candidate pool.")
    parser.add_argument("inputs", nargs="*", help="Input benchmark JSON/checkpoint files. Defaults to known benchmark checkpoints.")
    parser.add_argument("--output", "-o", default=str(DEFAULT_OUTPUT), help="Output candidate benchmark JSON.")
    parser.add_argument("--manual-review-output", default=str(DEFAULT_REVIEW), help="Output merged manual review JSON.")
    parser.add_argument("--report-output", default=str(DEFAULT_REPORT), help="Output merge report JSON.")
    args = parser.parse_args()

    input_paths = [Path(p) for p in args.inputs] if args.inputs else default_inputs()
    if not input_paths:
        raise SystemExit("No input benchmark files found.")
    merge_benchmarks(input_paths, args.output, args.manual_review_output, args.report_output)


if __name__ == "__main__":
    main()
