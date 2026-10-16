"""Small-sample review consistency audit for Track 1 Sci-Align."""

import argparse
import json
import statistics
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


# ROOT = Path(__file__).resolve().parents[2]  # Phase 5: sys.path hack removed, use mineru_benchmark.agent.utils.ROOT instead


DIMENSIONS = ["A", "B", "C", "D"]
ANSWER_MODEL = "deepseek-v4-flash"


@dataclass
class JudgeRecord:
    source_file: str
    model: str
    judge_model: str
    scoring_mode: str
    score: Optional[float]
    dimension: str
    difficulty_level: str
    reason: str
    issues: List[object]


def load_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def questions(data: Dict[str, object]) -> List[Dict[str, object]]:
    items = data.get("questions", [])
    if not isinstance(items, list):
        raise ValueError("dataset.questions must be a list")
    return items


def path_text(path: Path) -> str:
    return str(path).replace("\\", "/")


def rate(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0


def numeric(value: object) -> Optional[float]:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def score_bucket(score: Optional[float]) -> str:
    if score is None:
        return "missing"
    if score >= 0.75:
        return "high"
    if score >= 0.4:
        return "medium"
    return "low"


def sample_by_dimension(items: List[Dict[str, object]], sample_size: int) -> List[Dict[str, object]]:
    buckets: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for item in items:
        buckets[str(item.get("dimension", "unknown"))].append(item)
    selected = []
    base = sample_size // len(DIMENSIONS)
    remainder = sample_size % len(DIMENSIONS)
    for idx, dimension in enumerate(DIMENSIONS):
        quota = base + int(idx < remainder)
        selected.extend(buckets.get(dimension, [])[:quota])
    return selected[:sample_size]


def dimension_distribution(items: List[Dict[str, object]]) -> Dict[str, int]:
    return dict(sorted(Counter(str(item.get("dimension", "unknown")) for item in items).items()))


def valid_judge_file(path: Path) -> bool:
    name = path.name
    if name.endswith(".status.json"):
        return False
    if "chunk" in name:
        return False
    if "consolidated" in name:
        return False
    return name.endswith("_judge.json") or name.endswith("_sample_judge.json")


def load_judge_records(judge_dir: Path, answer_model: str) -> Dict[str, List[JudgeRecord]]:
    mapped: Dict[str, List[JudgeRecord]] = defaultdict(list)
    for path in sorted(judge_dir.glob("*.json")):
        if not valid_judge_file(path):
            continue
        data = load_json(path)
        if str(data.get("model", "")) != answer_model:
            continue
        records = data.get("records", [])
        if not isinstance(records, list):
            continue
        for record in records:
            if not isinstance(record, dict):
                continue
            sample_id = str(record.get("id", ""))
            if not sample_id:
                continue
            mapped[sample_id].append(
                JudgeRecord(
                    source_file=path_text(path),
                    model=str(data.get("model", "")),
                    judge_model=str(data.get("judge_model", data.get("model", ""))),
                    scoring_mode=str(data.get("scoring_mode", record.get("scoring_mode", "unknown"))),
                    score=numeric(record.get("score")),
                    dimension=str(record.get("dimension", "")),
                    difficulty_level=str(record.get("difficulty_level", "")),
                    reason=str(record.get("reason", ""))[:280],
                    issues=record.get("issues", []) if isinstance(record.get("issues", []), list) else [],
                )
            )
    return mapped


def select_mapped_sample(items: List[Dict[str, object]], mapped: Dict[str, List[JudgeRecord]], sample_size: int) -> List[Dict[str, object]]:
    mapped_items = [item for item in items if str(item.get("id", "")) in mapped]
    selected = sample_by_dimension(mapped_items, sample_size)
    if len(selected) < sample_size:
        selected_ids = {item.get("id") for item in selected}
        for item in mapped_items:
            if item.get("id") not in selected_ids:
                selected.append(item)
            if len(selected) >= sample_size:
                break
    return selected[:sample_size]


def record_agreement(item: Dict[str, object], records: List[JudgeRecord]) -> Dict[str, object]:
    scores = [record.score for record in records if record.score is not None]
    buckets = Counter(score_bucket(score) for score in scores)
    majority_bucket, majority_count = buckets.most_common(1)[0] if buckets else ("missing", 0)
    dimension = str(item.get("dimension", ""))
    matching_dimensions = sum(1 for record in records if record.dimension == dimension)
    source_ref_present = bool(str(item.get("source_ref", "")).strip())
    reasoning_steps_present = isinstance(item.get("reasoning_steps"), list) and len(item.get("reasoning_steps", [])) > 0
    return {
        "sample_id": item.get("id"),
        "dimension": dimension,
        "difficulty_level": item.get("difficulty_level"),
        "judge_count": len(records),
        "score_mean": round(statistics.mean(scores), 4) if scores else None,
        "score_std": round(statistics.pstdev(scores), 4) if len(scores) > 1 else 0.0,
        "score_min": min(scores) if scores else None,
        "score_max": max(scores) if scores else None,
        "bucket_counts": dict(sorted(buckets.items())),
        "majority_bucket": majority_bucket,
        "bucket_agreement_rate": rate(majority_count, len(scores)),
        "dimension_label_agreement_rate": rate(matching_dimensions, len(records)),
        "source_support_proxy": {
            "source_ref_present": source_ref_present,
            "reasoning_steps_present": reasoning_steps_present,
            "note": "This proxy checks dataset traceability fields only; it does not re-read source documents.",
        },
        "judges": [asdict(record) for record in records],
    }


def aggregate_agreement(records: List[Dict[str, object]]) -> Dict[str, object]:
    if not records:
        return {
            "status": "degraded_no_judge_mapping",
            "mapped_sample_count": 0,
            "mean_bucket_agreement_rate": 0.0,
            "mean_dimension_label_agreement_rate": 0.0,
        }
    bucket_rates = [float(record["bucket_agreement_rate"]) for record in records]
    dimension_rates = [float(record["dimension_label_agreement_rate"]) for record in records]
    judge_counts = [int(record["judge_count"]) for record in records]
    source_present = sum(1 for record in records if record["source_support_proxy"]["source_ref_present"])
    reasoning_present = sum(1 for record in records if record["source_support_proxy"]["reasoning_steps_present"])
    return {
        "status": "ok",
        "mapped_sample_count": len(records),
        "mean_judge_count": round(statistics.mean(judge_counts), 4),
        "min_judge_count": min(judge_counts),
        "max_judge_count": max(judge_counts),
        "mean_bucket_agreement_rate": round(statistics.mean(bucket_rates), 4),
        "mean_dimension_label_agreement_rate": round(statistics.mean(dimension_rates), 4),
        "source_ref_present_rate": rate(source_present, len(records)),
        "reasoning_steps_present_rate": rate(reasoning_present, len(records)),
        "majority_bucket_distribution": dict(sorted(Counter(record["majority_bucket"] for record in records).items())),
    }


def run(dataset: str, judge_dir: str, sample_size: int, output: str) -> Dict[str, object]:
    dataset_path = Path(dataset)
    judge_path = Path(judge_dir)
    data = load_json(dataset_path)
    items = questions(data)
    mapped = load_judge_records(judge_path, ANSWER_MODEL)
    sample = select_mapped_sample(items, mapped, min(sample_size, len(items)))
    review_records = [record_agreement(item, mapped[str(item.get("id"))]) for item in sample]
    status = "ok" if review_records else "degraded_no_judge_mapping"
    payload = {
        "schema_version": "1.0",
        "experiment": "track1_sample_review_consistency",
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "status": status,
        "evaluation_mode": "stratified_small_sample_existing_judge_matrix_replay_no_new_judge_calls",
        "inputs": {
            "dataset": path_text(dataset_path),
            "judge_dir": path_text(judge_path),
            "sample_size_requested": sample_size,
            "answer_model_filter": ANSWER_MODEL,
        },
        "dataset_question_count": len(items),
        "mapped_question_count": len(mapped),
        "sample_size": len(sample),
        "dimension_sample_counts": dimension_distribution(sample),
        "judge_agreement": aggregate_agreement(review_records),
        "human_review": "not_performed",
        "review_records": review_records,
        "boundary_notes": [
            "This is a small-sample review consistency replay over existing judge_matrix artifacts; no new judge API or human review was performed.",
            "Agreement is computed over score buckets high/medium/low for the same answer model where judge records are available.",
            "source_support_proxy only checks source_ref and reasoning_steps presence in the dataset; it does not claim source-document rereading.",
            "The 30-record sample is a review example and must not be extrapolated as a full-dataset manual acceptance rate.",
        ],
    }
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(out)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Track 1 small-sample review consistency audit from existing judge artifacts.")
    parser.add_argument("--dataset", default="benchmark/dataset/core.json")
    parser.add_argument("--judge-dir", default="benchmark/eval/results/judge_matrix")
    parser.add_argument("--sample-size", type=int, default=30)
    parser.add_argument("--output", default="benchmark/eval/results/track1_sample_review_consistency.json")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    payload = run(args.dataset, args.judge_dir, args.sample_size, args.output)
    summary = {
        "status": payload["status"],
        "sample_size": payload["sample_size"],
        "dimension_sample_counts": payload["dimension_sample_counts"],
        "mapped_question_count": payload["mapped_question_count"],
        "judge_agreement": payload["judge_agreement"],
        "human_review": payload["human_review"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if payload["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
