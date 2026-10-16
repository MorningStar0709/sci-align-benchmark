"""合并医学域 A/B 两组出题结果到统一的 medical_benchmark.json。

Usage:
    python -m mineru_benchmark.pipeline.merge_medical_ab
"""

import json
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[5]
DATASET_DIR = ROOT / "benchmark" / "dataset"


def merge_medical_benchmark():
    """Merge medical_benchmark_a.json and medical_benchmark_b.json."""
    file_a = DATASET_DIR / "medical_benchmark_a.json"
    file_b = DATASET_DIR / "medical_benchmark_b.json"
    output = DATASET_DIR / "medical_benchmark.json"
    review_output = DATASET_DIR / "medical_manual_review.json"

    if not file_a.exists():
        print(f"ERROR: {file_a} not found")
        return
    if not file_b.exists():
        print(f"ERROR: {file_b} not found")
        return

    # Load A group
    with open(file_a, encoding="utf-8") as f:
        data_a = json.load(f)
    questions_a = data_a.get("questions", [])
    review_a = data_a.get("review_items", [])

    # Load B group
    with open(file_b, encoding="utf-8") as f:
        data_b = json.load(f)
    questions_b = data_b.get("questions", [])
    review_b = data_b.get("review_items", [])

    # Merge
    all_questions = questions_a + questions_b
    all_review = review_a + review_b

    # Sort by ID
    all_questions.sort(key=lambda q: q.get("id", ""))

    # Statistics
    from collections import Counter
    dim_counts = Counter(q.get("dimension", "?") for q in all_questions)
    diff_counts = Counter(q.get("difficulty_level", "?") for q in all_questions)
    cross_counts = Counter((q.get("dimension", "?"), q.get("difficulty_level", "?")) for q in all_questions)

    # Build merged output
    merged = {
        "questions": all_questions,
        "review_items": all_review,
        "metadata": {
            "source": f"Merged from {file_a.name} ({len(questions_a)} q) + {file_b.name} ({len(questions_b)} q)",
            "merged_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_questions": len(all_questions),
            "dimension_counts": dict(dim_counts),
            "difficulty_counts": dict(diff_counts),
        }
    }

    # Write merged output
    with open(output, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    # Write merged review
    review_data = {"review_items": all_review}
    with open(review_output, "w", encoding="utf-8") as f:
        json.dump(review_data, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*50}")
    print("医学域基准合并完成")
    print(f"{'='*50}")
    print(f"A组: {len(questions_a)} 题 ({file_a.name})")
    print(f"B组: {len(questions_b)} 题 ({file_b.name})")
    print(f"合并: {len(all_questions)} 题 → {output.name}")
    print("")
    print(f"维度分布: {dict(dim_counts)}")
    print(f"难度分布: {dict(diff_counts)}")
    print("")
    print("维度×难度交叉:")
    for (dim, diff), count in sorted(cross_counts.items()):
        print(f"  {dim} × {diff}: {count}")
    print("")
    print(f"Manual review: {review_output.name}")


if __name__ == "__main__":
    merge_medical_benchmark()
