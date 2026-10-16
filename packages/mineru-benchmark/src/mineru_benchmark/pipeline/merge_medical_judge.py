"""Merge J1 (DeepSeek) and J2 (MiMo) judge results, arbitrate disputes with Qwen.

Usage:
    python -m mineru_benchmark.pipeline.merge_medical_judge
"""

import json
import time
from collections import Counter
from pathlib import Path

import mineru_core  # noqa: F401 — load .env

from mineru_core.settings import PROVIDERS
from mineru_eval.client_factory import create_client, resolve_api_key
from mineru_eval.medical.scorers import score_medical_answer
from mineru_eval.utils import load_json

ROOT = Path(__file__).resolve().parents[5]
BENCHMARK_PATH = ROOT / "benchmark" / "dataset" / "medical_benchmark.json"
J1_PATH = ROOT / "benchmark" / "dataset" / "medical_judge_deepseek.jsonl"
J2_PATH = ROOT / "benchmark" / "dataset" / "medical_judge_mimo.jsonl"
OUTPUT_PATH = ROOT / "benchmark" / "dataset" / "medical_dual_report.json"


def load_jsonl(path: Path) -> dict[str, dict]:
    """Load JSONL file into dict keyed by 'id'."""
    records = {}
    if not path.exists():
        raise FileNotFoundError(f"Missing: {path}")
    for line in path.read_text(encoding="utf-8").strip().split("\n"):
        if line.strip():
            r = json.loads(line)
            records[r["id"]] = r
    return records


def write_json_atomic(path: Path, data: dict):
    """Atomic JSON write: write to .tmp then rename."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    tmp.replace(path)


def snap_to_valid(value, valid_scores=(0.0, 0.25, 0.5, 0.75, 1.0)):
    """Snap float to nearest valid score."""
    return min(valid_scores, key=lambda v: abs(v - value))


def main():
    print("Loading judge results...")
    j1_records = load_jsonl(J1_PATH)
    j2_records = load_jsonl(J2_PATH)
    benchmark = load_json(BENCHMARK_PATH)
    questions = {q["id"]: q for q in benchmark.get("questions", [])}

    print(f"  J1 (DeepSeek): {len(j1_records)} records")
    print(f"  J2 (MiMo):     {len(j2_records)} records")
    print(f"  Benchmark:     {len(questions)} questions")

    # Find common IDs
    common_ids = set(j1_records.keys()) & set(j2_records.keys())
    only_j1 = set(j1_records.keys()) - set(j2_records.keys())
    only_j2 = set(j2_records.keys()) - set(j1_records.keys())

    if only_j1:
        print(f"  WARNING: {len(only_j1)} questions only in J1")
    if only_j2:
        print(f"  WARNING: {len(only_j2)} questions only in J2")

    print(f"  Common:        {len(common_ids)} questions")

    # Identify disputes (|score_diff| > 0.5)
    disputes = []
    aligned = []
    for qid in common_ids:
        r1, r2 = j1_records[qid], j2_records[qid]
        s1, s2 = r1.get("score", 0.0), r2.get("score", 0.0)
        diff = abs(s1 - s2)
        if diff > 0.5:
            disputes.append(qid)
        else:
            avg = snap_to_valid((s1 + s2) / 2.0)
            aligned.append({
                "id": qid,
                "score": avg,
                "judge1_score": s1,
                "judge2_score": s2,
                "avg_score": avg,
                "score_diff": diff,
                "disputed": False,
                "arbitrated": False,
                "reason": f"J1={s1} J2={s2} avg={avg}",
                "issues": [],
            })

    print(f"\nAgreement: {len(aligned)} questions")
    print(f"Disputes:  {len(disputes)} questions (|diff| > 0.5)")

    # Arbitrate disputes with Qwen
    if disputes:
        print(f"\nArbitrating {len(disputes)} disputes with Qwen...")
        qwen_cfg = PROVIDERS["qwen"]
        qwen_key = resolve_api_key(qwen_cfg.base_url, provider="qwen")
        qwen_client, qwen_model, _ = create_client(qwen_key, qwen_cfg.base_url, qwen_cfg.default_model)
        print(f"  Arbiter: {qwen_model}")

        arbitrated = []
        errors = 0
        t0 = time.time()

        for i, qid in enumerate(disputes, 1):
            q = questions.get(qid)
            if not q:
                print(f"  WARNING: {qid} not in benchmark, skipping")
                errors += 1
                continue

            r1, r2 = j1_records[qid], j2_records[qid]
            s1, s2 = r1.get("score", 0.0), r2.get("score", 0.0)
            diff = abs(s1 - s2)

            question_text = q["question"]
            expected = q["answer"]
            model_answer = q["answer"]
            dimension = q["dimension"]
            difficulty = q.get("difficulty_level", "L2")

            try:
                arb_result = score_medical_answer(
                    question_text, expected, model_answer,
                    dimension, qwen_client, qwen_model,
                    difficulty=difficulty,
                )
                arb_score = arb_result.get("score", 0.0)
                arbitrated.append({
                    "id": qid,
                    "score": arb_score,
                    "judge1_score": s1,
                    "judge2_score": s2,
                    "avg_score": snap_to_valid((s1 + s2) / 2.0),
                    "score_diff": diff,
                    "disputed": True,
                    "arbitrated": True,
                    "arbitrator_score": arb_score,
                    "reason": f"ARBITRATED: {arb_result.get('reason', '')} (J1={s1}, J2={s2})",
                    "issues": arb_result.get("issues", []),
                })
            except Exception as e:
                errors += 1
                # Fallback to average on arbitration failure
                avg = snap_to_valid((s1 + s2) / 2.0)
                arbitrated.append({
                    "id": qid,
                    "score": avg,
                    "judge1_score": s1,
                    "judge2_score": s2,
                    "avg_score": avg,
                    "score_diff": diff,
                    "disputed": True,
                    "arbitrated": False,
                    "reason": f"Arbitration failed ({e}), fallback to avg={avg}",
                    "issues": [f"arbitration failed: {e}"],
                })

            if i % 10 == 0:
                elapsed = time.time() - t0
                rate = i / max(elapsed, 0.1)
                eta = (len(disputes) - i) / max(rate, 0.01)
                print(f"  [{i}/{len(disputes)}] rate={rate:.1f}q/s eta={eta:.0f}s err={errors}")

        elapsed = time.time() - t0
        print(f"  Arbitration done: {len(arbitrated)} questions, {errors} errors, {elapsed:.1f}s")
    else:
        arbitrated = []

    # Merge all records
    all_records = aligned + arbitrated

    # Compute statistics
    scores = [r["score"] for r in all_records]
    overall = sum(scores) / len(scores) if scores else 0.0

    by_dim = Counter()
    score_sums = Counter()
    for r in all_records:
        q = questions.get(r["id"], {})
        dim = q.get("dimension", "?")
        by_dim[dim] += 1
        score_sums[dim] += r["score"]

    dim_scores = {dim: round(score_sums[dim] / by_dim[dim], 4) for dim in sorted(by_dim.keys())}

    dispute_rate = len(disputes) / len(all_records) if all_records else 0.0

    # Build report
    report = {
        "model": "mimo-v2.5",
        "judge1": "deepseek-v4-flash",
        "judge2": "mimo-v2.5",
        "arbitrator": "qwen3.7-plus",
        "total_questions": len(all_records),
        "overall_score": round(overall, 4),
        "dimension_scores": dim_scores,
        "dual_judge_stats": {
            "total": len(all_records),
            "agreed": len(aligned),
            "disputed": len(disputes),
            "arbitrated": len(arbitrated),
            "dispute_rate": round(dispute_rate, 4),
        },
        "records": all_records,
    }

    write_json_atomic(OUTPUT_PATH, report)
    print(f"\nReport saved: {OUTPUT_PATH}")
    print(f"  Overall score: {overall:.4f}")
    print(f"  Dimension scores: {dim_scores}")
    print(f"  Dispute rate: {dispute_rate:.1%}")


if __name__ == "__main__":
    main()
