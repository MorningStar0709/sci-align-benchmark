"""
Build MiniMax self-correction candidates: filter low-score questions from
minimax-m2.5-highspeed_report.json, join with core.json for question text,
select 5 per dimension (A/B/C/D), output candidates JSON.

Usage:
    .venv/bin/python benchmark/eval/build_minimax_candidates.py
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]

from mineru_eval.utils import load_json, write_json_atomic

REPORT_PATH = ROOT / "benchmark" / "eval" / "reports" / "minimax-m2.5-highspeed_report.json"
CORE_PATH = ROOT / "benchmark" / "dataset" / "core.json"
CANDIDATES_PATH = ROOT / "benchmark" / "eval" / "results" / "self_correction_candidates_minimax.json"

SEED = 42
PER_DIM = 5


def _collect_candidates(report, questions_map, threshold):
    dim_map = {"A": [], "B": [], "C": [], "D": []}
    for r in report.get("records", []):
        score = r.get("score", 1.0)
        if score >= threshold:
            continue
        dim = r.get("dimension", "?")
        if dim not in dim_map:
            continue
        qid = r.get("id", "")
        core_item = questions_map.get(qid, {})
        question_text = core_item.get("question", "")
        if not question_text:
            continue
        dim_map[dim].append({
            "qid": qid,
            "dimension": dim,
            "difficulty_level": r.get("difficulty_level", "?"),
            "question_text": question_text,
            "expected_answer": r.get("expected_answer", ""),
            "judge_score": score,
            "judge_reason": r.get("reason", ""),
            "original_answer": r.get("model_answer", ""),
        })
    return dim_map


def filter_candidates():
    report = load_json(REPORT_PATH)
    core = load_json(CORE_PATH)

    questions_map = {q["id"]: q for q in core.get("questions", [])}
    records = report.get("records", [])
    print(f"Total records in report: {len(records)}")

    import random
    rng = random.Random(SEED)
    selected = []

    for dim in ["A", "B", "C", "D"]:
        threshold = 0.3
        while True:
            dim_map = _collect_candidates(report, questions_map, threshold)
            pool = dim_map[dim]
            if len(pool) >= PER_DIM:
                break
            threshold = round(threshold + 0.1, 1)
            if threshold >= 1.0:
                print(f"  WARNING: dimension {dim} only has {len(pool)} candidates even at threshold={threshold}")
                break

        rng.shuffle(pool)
        chosen = pool[:PER_DIM]
        selected.extend(chosen)
        print(f"  Dimension {dim}: threshold<{threshold}, pool={len(pool)}, selected={len(chosen)}")
        for c in chosen:
            print(f"    {c['qid']} ({c['difficulty_level']}) score={c['judge_score']}")

    output = {
        "meta": {
            "source_report": str(REPORT_PATH),
            "source_core": str(CORE_PATH),
            "model": "MiniMax-M2.5-highspeed",
            "total_candidates": len(selected),
            "dimensions": {dim: len([c for c in selected if c["dimension"] == dim]) for dim in ["A", "B", "C", "D"]},
            "seed": SEED,
            "generated_at": __import__("datetime").datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        },
        "candidates": selected,
    }

    write_json_atomic(CANDIDATES_PATH, output)
    print(f"\nOutput: {CANDIDATES_PATH} ({len(selected)} candidates)")


if __name__ == "__main__":
    filter_candidates()
