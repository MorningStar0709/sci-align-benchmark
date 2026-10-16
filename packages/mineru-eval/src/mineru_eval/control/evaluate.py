"""Control science evaluation entry point.

Extracted from evaluate.py — contains control-domain scoring functions and CLI.
"""

from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]

import mineru_core  # noqa: F401 — trigger _load_dotenv() to load .env API keys

from mineru_eval._shared.call_utils import TARGET_SYSTEM_PROMPT  # noqa: F401 — re-export
from mineru_eval._shared.report_base import build_model_report  # noqa: F401 — re-export
from mineru_eval.control.prompts import CONTROL_TERMS, _FORMULA_PATTERNS  # noqa: F401 — re-export
from mineru_eval.control.scorers import (  # noqa: F401 — re-export
    AScorer,
    BScorer,
    CScorer,
    DScorer,
    GenericScorer,
    RubricScorer,
)
from mineru_eval.control.eval_scoring import SCORER_VERSION, SCORER_METHOD  # noqa: F401 — re-export
from mineru_eval._shared.eval_runner import model_evaluate  # noqa: F401 — re-export

DEFAULT_INPUT = ROOT / "benchmark" / "dataset" / "sample_questions.json"
DEFAULT_OUTPUT = ROOT / "benchmark" / "dataset" / "sample_report.json"


# ── Reference mode (legacy, superseded by LLM-as-Judge model mode) ──


def score_a(question):
    answer = question.get("answer", "")
    has_formula = any(pattern.search(answer) for pattern in _FORMULA_PATTERNS)
    has_control_term = any(term in answer for term in CONTROL_TERMS)
    has_steps = len(question.get("reasoning_steps") or []) >= 2
    if has_formula and has_control_term and has_steps:
        return 1.0
    if (has_formula or has_control_term) and has_steps:
        return 0.7
    return 0.6 if has_steps else 0.3


def score_b(question):
    step_count = len(question.get("reasoning_steps") or [])
    if step_count >= 4:
        return 1.0
    if step_count >= 3:
        return 0.7
    if step_count >= 2:
        return 0.5
    return 0.2


def score_c(question):
    step_count = len(question.get("reasoning_steps") or [])
    has_pair = bool(question.get("sibling_id")) and bool(question.get("sensitivity_dimension"))
    return 0.5 if has_pair and step_count >= 3 else 0.25 if has_pair else 0.0


def score_d(question):
    rubric = question.get("rubric") or {}
    if not rubric:
        return 0.0
    weights = [item.get("weight", 0) for item in rubric.values() if isinstance(item, dict)]
    return round(sum(weights), 4)


def score_question(question):
    dimension = question.get("dimension")
    if dimension == "A":
        score = score_a(question)
    elif dimension == "B":
        score = score_b(question)
    elif dimension == "C":
        score = score_c(question)
    elif dimension == "D":
        score = score_d(question)
    else:
        score = 0.0
    return min(float(score), 1.0)


def evaluate(data, model):
    """Reference-mode evaluation (heuristic scoring, no LLM judge)."""
    records = []
    for question in data.get("questions", []):
        score = score_question(question)
        records.append(
            {
                "id": question.get("id"),
                "dimension": question.get("dimension"),
                "difficulty_level": question.get("difficulty_level"),
                "score": score,
                "scoring_mode": "reference_heuristic",
            }
        )

    by_dimension = Counter()
    score_sums = Counter()
    for record in records:
        dimension = record["dimension"]
        by_dimension[dimension] += 1
        score_sums[dimension] += record["score"]

    dimension_scores = {
        dimension: round(score_sums[dimension] / count, 4)
        for dimension, count in sorted(by_dimension.items())
        if count
    }
    overall = round(sum(record["score"] for record in records) / len(records), 4) if records else 0.0

    return {
        "model": model,
        "total_questions": len(records),
        "overall_score": overall,
        "dimension_scores": dimension_scores,
        "records": records,
    }


def score_model_answer(question, expected, model_answer, dimension, judge_client, judge_model,
                       rubric=None, disable_thinking: bool = False):
    """Score a model answer using the control-domain LLM-as-Judge pipeline."""
    if dimension == "A":
        return AScorer.judge(question, expected, model_answer, judge_client, judge_model,
                             disable_thinking=disable_thinking)
    if dimension == "B":
        return BScorer.judge(question, expected, model_answer, judge_client, judge_model,
                             disable_thinking=disable_thinking)
    if dimension == "D" and rubric:
        return RubricScorer.judge(question, expected, model_answer, judge_client, judge_model,
                                  rubric=rubric, disable_thinking=disable_thinking)
    if dimension == "C":
        return CScorer.judge(question, expected, model_answer, judge_client, judge_model,
                             disable_thinking=disable_thinking)
    if dimension == "D":
        return DScorer.judge(question, expected, model_answer, judge_client, judge_model,
                             disable_thinking=disable_thinking)
    return {"score": 0.0, "reason": "Unknown dimension", "issues": ["unknown dimension"]}


# ── CLI ──────────────────────────────────────────────────────────


def main():
    """Backward-compatible alias — delegates to evaluate_cli.main."""
    from mineru_eval.evaluate_cli import main as _main
    return _main()


if __name__ == "__main__":
    main()
