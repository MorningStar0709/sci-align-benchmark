"""Dimension discriminability and difficulty calibration for Track 1 Sci-Align."""

import argparse
import json
import math
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ROOT = Path(__file__).resolve().parents[2]  # Phase 5: sys.path hack removed, use mineru_benchmark.agent.utils.ROOT instead


DIMENSIONS = ["A", "B", "C", "D"]
DIFFICULTY_LEVELS = ["L1", "L2", "L3", "L4"]


def load_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def questions(data: Dict[str, object]) -> List[Dict[str, object]]:
    items = data.get("questions", [])
    if not isinstance(items, list):
        raise ValueError("dataset.questions must be a list")
    return items


def leaderboard_models(data: Dict[str, object]) -> List[Dict[str, object]]:
    items = data.get("models", [])
    if not isinstance(items, list):
        raise ValueError("leaderboard.models must be a list")
    return items


def rate(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0


def round_float(value: float) -> float:
    return round(value, 4)


def mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def variance(values: List[float]) -> float:
    if not values:
        return 0.0
    avg = mean(values)
    return sum((value - avg) ** 2 for value in values) / len(values)


def std(values: List[float]) -> float:
    return math.sqrt(variance(values))


def numeric(value: object) -> Optional[float]:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def schema_gap(models: List[Dict[str, object]]) -> List[Dict[str, object]]:
    missing = []
    for model in models:
        model_name = str(model.get("model", "<missing-model>"))
        fields = []
        if not model.get("model"):
            fields.append("model")
        if numeric(model.get("overall_score")) is None:
            fields.append("overall_score")
        dimension_scores = model.get("dimension_scores")
        if not isinstance(dimension_scores, dict):
            fields.append("dimension_scores")
        else:
            for dimension in DIMENSIONS:
                if numeric(dimension_scores.get(dimension)) is None:
                    fields.append(f"dimension_scores.{dimension}")
        if fields:
            missing.append({"model": model_name, "missing_or_invalid_fields": fields})
    return missing


def rank_models(score_pairs: List[Tuple[str, float]]) -> List[Dict[str, object]]:
    ranked = sorted(score_pairs, key=lambda item: (-item[1], item[0]))
    return [{"rank": idx, "model": model, "score": round_float(score)} for idx, (model, score) in enumerate(ranked, 1)]


def rank_positions(ranking: List[Dict[str, object]]) -> Dict[str, int]:
    return {str(item["model"]): int(item["rank"]) for item in ranking}


def spearman_rank_correlation(left: Dict[str, int], right: Dict[str, int]) -> float:
    models = sorted(set(left) & set(right))
    n = len(models)
    if n < 2:
        return 1.0
    d2 = sum((left[model] - right[model]) ** 2 for model in models)
    return round_float(1 - (6 * d2) / (n * (n * n - 1)))


def pairwise_agreement(overall_scores: Dict[str, float], dimension_scores: Dict[str, float]) -> Dict[str, object]:
    models = sorted(set(overall_scores) & set(dimension_scores))
    concordant = 0
    discordant = 0
    tied = 0
    total = 0
    for idx, left in enumerate(models):
        for right in models[idx + 1 :]:
            total += 1
            overall_delta = overall_scores[left] - overall_scores[right]
            dimension_delta = dimension_scores[left] - dimension_scores[right]
            if overall_delta == 0 or dimension_delta == 0:
                tied += 1
            elif overall_delta * dimension_delta > 0:
                concordant += 1
            else:
                discordant += 1
    return {
        "pair_count": total,
        "concordant": concordant,
        "discordant": discordant,
        "tied": tied,
        "agreement_rate": rate(concordant, total),
    }


def dimension_statistics(models: List[Dict[str, object]]) -> Dict[str, object]:
    stats = {}
    for dimension in DIMENSIONS:
        score_pairs = []
        for model in models:
            dimension_scores = model.get("dimension_scores", {})
            score = numeric(dimension_scores.get(dimension)) if isinstance(dimension_scores, dict) else None
            if score is not None:
                score_pairs.append((str(model.get("model")), score))
        scores = [score for _, score in score_pairs]
        ranking = rank_models(score_pairs)
        stats[dimension] = {
            "model_count": len(score_pairs),
            "mean": round_float(mean(scores)),
            "variance": round_float(variance(scores)),
            "std": round_float(std(scores)),
            "min": round_float(min(scores)) if scores else 0.0,
            "max": round_float(max(scores)) if scores else 0.0,
            "range": round_float(max(scores) - min(scores)) if scores else 0.0,
            "top_model": ranking[0] if ranking else {},
            "bottom_model": ranking[-1] if ranking else {},
            "ranking": ranking,
            "scores_by_model": {model: round_float(score) for model, score in sorted(score_pairs)},
        }
    return stats


def rank_stability(models: List[Dict[str, object]], dimension_stats: Dict[str, object]) -> Dict[str, object]:
    overall_pairs = [(str(model.get("model")), float(model.get("overall_score"))) for model in models]
    overall_ranking = rank_models(overall_pairs)
    overall_positions = rank_positions(overall_ranking)
    overall_scores = {model: score for model, score in overall_pairs}
    by_dimension = {}
    for dimension in DIMENSIONS:
        ranking = dimension_stats[dimension]["ranking"]
        dimension_positions = rank_positions(ranking)
        dimension_scores = dimension_stats[dimension]["scores_by_model"]
        shifts = {}
        for model in sorted(overall_positions):
            shifts[model] = dimension_positions[model] - overall_positions[model]
        by_dimension[dimension] = {
            "spearman_rank_correlation_with_overall": spearman_rank_correlation(overall_positions, dimension_positions),
            "pairwise_agreement_with_overall": pairwise_agreement(overall_scores, dimension_scores),
            "max_abs_rank_shift": max(abs(value) for value in shifts.values()) if shifts else 0,
            "rank_shift_by_model": shifts,
            "ranking": ranking,
        }
    return {"overall_ranking": overall_ranking, "by_dimension": by_dimension}


def difficulty_distribution(items: List[Dict[str, object]]) -> Dict[str, object]:
    global_counts = Counter(str(item.get("difficulty_level", "unknown")) for item in items)
    dimension_counts: Dict[str, Counter] = defaultdict(Counter)
    for item in items:
        dimension = str(item.get("dimension", "unknown"))
        difficulty = str(item.get("difficulty_level", "unknown"))
        dimension_counts[dimension][difficulty] += 1
    by_dimension = {}
    for dimension in sorted(dimension_counts):
        total = sum(dimension_counts[dimension].values())
        by_dimension[dimension] = {
            "total": total,
            "counts": {level: dimension_counts[dimension].get(level, 0) for level in DIFFICULTY_LEVELS},
            "rates": {level: rate(dimension_counts[dimension].get(level, 0), total) for level in DIFFICULTY_LEVELS},
        }
    return {
        "global": {level: global_counts.get(level, 0) for level in DIFFICULTY_LEVELS},
        "global_rates": {level: rate(global_counts.get(level, 0), len(items)) for level in DIFFICULTY_LEVELS},
        "by_dimension": by_dimension,
    }


def gap_interpretation(dimension_stats: Dict[str, object], stability: Dict[str, object]) -> Dict[str, object]:
    range_order = sorted(DIMENSIONS, key=lambda dim: (-dimension_stats[dim]["range"], dim))
    std_order = sorted(DIMENSIONS, key=lambda dim: (-dimension_stats[dim]["std"], dim))
    ab_range = mean([dimension_stats["A"]["range"], dimension_stats["B"]["range"]])
    cd_range = mean([dimension_stats["C"]["range"], dimension_stats["D"]["range"]])
    ab_std = mean([dimension_stats["A"]["std"], dimension_stats["B"]["std"]])
    cd_std = mean([dimension_stats["C"]["std"], dimension_stats["D"]["std"]])
    c_range_delta = dimension_stats["C"]["range"] - ab_range
    d_range_delta = dimension_stats["D"]["range"] - ab_range
    c_rank_shift = stability["by_dimension"]["C"]["max_abs_rank_shift"]
    d_rank_shift = stability["by_dimension"]["D"]["max_abs_rank_shift"]
    if cd_range > ab_range and cd_std > ab_std:
        conclusion = "C/D dimensions separate model performance more strongly than A/B in the current leaderboard."
    elif dimension_stats["D"]["range"] > ab_range or dimension_stats["C"]["range"] > ab_range:
        conclusion = "At least one of C/D separates model performance more strongly than the A/B average in the current leaderboard."
    else:
        conclusion = "The current leaderboard does not support a stronger C/D separation claim than A/B."
    return {
        "range_order_desc": range_order,
        "std_order_desc": std_order,
        "ab_range_mean": round_float(ab_range),
        "cd_range_mean": round_float(cd_range),
        "ab_std_mean": round_float(ab_std),
        "cd_std_mean": round_float(cd_std),
        "c_vs_ab_range_delta": round_float(c_range_delta),
        "d_vs_ab_range_delta": round_float(d_range_delta),
        "c_max_abs_rank_shift": c_rank_shift,
        "d_max_abs_rank_shift": d_rank_shift,
        "conclusion": conclusion,
    }


def degraded_payload(leaderboard_path: Path, dataset_path: Path, output_path: Path, missing_fields: List[Dict[str, object]]) -> Dict[str, object]:
    payload = {
        "schema_version": "1.0",
        "experiment": "track1_dimension_discriminability",
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "status": "degraded_schema_gap",
        "evaluation_mode": "leaderboard_schema_gap_only_no_discriminability_claim",
        "inputs": {
            "leaderboard": str(leaderboard_path).replace("\\", "/"),
            "dataset": str(dataset_path).replace("\\", "/"),
        },
        "missing_fields": missing_fields,
        "interpretation_boundary": [
            "The leaderboard does not contain complete model, overall_score, and dimension_scores fields.",
            "No dimension discriminability metric is reported in degraded_schema_gap mode.",
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_suffix(output_path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(output_path)
    return payload


def run(leaderboard: str, dataset: str, output: str) -> Dict[str, object]:
    leaderboard_path = Path(leaderboard)
    dataset_path = Path(dataset)
    output_path = Path(output)
    leaderboard_data = load_json(leaderboard_path)
    dataset_data = load_json(dataset_path)
    models = leaderboard_models(leaderboard_data)
    items = questions(dataset_data)
    missing_fields = schema_gap(models)
    if missing_fields:
        return degraded_payload(leaderboard_path, dataset_path, output_path, missing_fields)
    dim_stats = dimension_statistics(models)
    stability = rank_stability(models, dim_stats)
    difficulty = difficulty_distribution(items)
    interpretation = gap_interpretation(dim_stats, stability)
    payload = {
        "schema_version": "1.0",
        "experiment": "track1_dimension_discriminability",
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "status": "ok",
        "evaluation_mode": "existing_leaderboard_dimension_score_analysis_no_new_model_eval",
        "inputs": {
            "leaderboard": str(leaderboard_path).replace("\\", "/"),
            "dataset": str(dataset_path).replace("\\", "/"),
        },
        "model_count": len(models),
        "question_count": len(items),
        "dimension_question_counts": {dimension: difficulty["by_dimension"].get(dimension, {}).get("total", 0) for dimension in DIMENSIONS},
        "dimension_stats": dim_stats,
        "rank_stability": stability,
        "difficulty_distribution": difficulty,
        "dimension_gap_interpretation": interpretation,
        "interpretation_boundary": [
            "This analysis summarizes existing leaderboard_complete.json and does not run new model inference.",
            "Dimension discriminability is measured by between-model score spread and ranking shifts, not by statistical significance testing over per-question predictions.",
            "Difficulty calibration is derived from dataset labels in core.json; it does not infer empirical item response difficulty from model-level aggregate scores.",
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_suffix(output_path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(output_path)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze Track 1 dimension discriminability from an existing leaderboard.")
    parser.add_argument("--leaderboard", default="benchmark/eval/results/leaderboard_complete.json")
    parser.add_argument("--dataset", default="benchmark/dataset/core.json")
    parser.add_argument("--output", default="benchmark/eval/results/track1_dimension_discriminability.json")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    payload = run(args.leaderboard, args.dataset, args.output)
    if payload["status"] != "ok":
        print(json.dumps({"status": payload["status"], "missing_fields": payload["missing_fields"]}, ensure_ascii=False, indent=2))
        return 1
    summary = {
        "status": payload["status"],
        "model_count": payload["model_count"],
        "question_count": payload["question_count"],
        "range_order_desc": payload["dimension_gap_interpretation"]["range_order_desc"],
        "std_order_desc": payload["dimension_gap_interpretation"]["std_order_desc"],
        "conclusion": payload["dimension_gap_interpretation"]["conclusion"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
