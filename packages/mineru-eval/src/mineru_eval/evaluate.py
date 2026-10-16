"""Thin backward-compatible shell — re-exports from _shared and control subpackages.

All implementation now lives in:
  - mineru_eval._shared.call_utils   (call_target_model, call_with_retries, format_eta, TARGET_SYSTEM_PROMPT)
  - mineru_eval._shared.report_base  (build_model_report, load_existing_records, select_questions)
  - mineru_eval._shared.eval_runner  (model_evaluate, _score_or_fallback, _init_eval_session, _save_eval_progress)
  - mineru_eval.control.evaluate     (score_a/b/c/d, score_question, evaluate, score_model_answer)
  - mineru_eval.control.eval_scoring (_score_one_question, _build_eval_record, SCORER_VERSION, SCORER_METHOD)
  - mineru_eval.control.scorers      (AScorer, BScorer, CScorer, DScorer, GenericScorer, RubricScorer)
  - mineru_eval.control.prompts      (CONTROL_TERMS, _FORMULA_PATTERNS)
"""

import re  # noqa: F401 — kept for any code that checks `re` availability
from collections import Counter  # noqa: F401 — used by render_md_report
from datetime import datetime  # noqa: F401 — used by render_md_report
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]

import mineru_core  # noqa: F401 — trigger _load_dotenv() to load .env API keys

# ── Re-exports from _shared ────────────────────────────────────────
from mineru_eval._shared.call_utils import (  # noqa: F401
    TARGET_SYSTEM_PROMPT,
    call_target_model,
    call_with_retries,
    format_eta,
)
from mineru_eval._shared.report_base import (  # noqa: F401
    build_model_report,
    load_existing_records,
    select_questions,
)
from mineru_eval._shared.eval_runner import (  # noqa: F401
    model_evaluate,
    _init_eval_session,
    _save_eval_progress,
    _score_or_fallback,
)
from mineru_eval.utils import write_json_atomic  # noqa: F401 — re-export for evaluate_cli

# ── Re-exports from control subpackage ──────────────────────────────
from mineru_eval.control.prompts import CONTROL_TERMS, _FORMULA_PATTERNS  # noqa: F401
from mineru_eval.control.scorers import (  # noqa: F401
    AScorer,
    BScorer,
    CScorer,
    DScorer,
    GenericScorer,
    RubricScorer,
)
from mineru_eval.control.eval_scoring import (  # noqa: F401
    SCORER_METHOD,
    SCORER_VERSION,
    _build_eval_record,
    _score_one_question,
)
from mineru_eval.control.evaluate import (  # noqa: F401
    evaluate,
    score_a,
    score_b,
    score_c,
    score_d,
    score_model_answer,
    score_question,
)

DEFAULT_INPUT = ROOT / "benchmark" / "dataset" / "sample_questions.json"
DEFAULT_OUTPUT = ROOT / "benchmark" / "dataset" / "sample_report.json"


# ── Report generation (inline md helper, delegates to report.py when available) ──


def render_md_report(report):
    """Quick Markdown rendering of a model evaluation report."""
    from mineru_eval.report import render_report as rp_render

    meta = {
        "title": "ControlSci Evaluation Report",
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "target_model": report.get("model", ""),
        "judge_model": report.get("judge_model", ""),
        "scoring_mode": report.get("scoring_mode", ""),
        "overall_score": report.get("overall_score", 0),
    }
    issues_counter = Counter(report.get("issues_summary", {}))
    return rp_render(meta, report.get("records", []), issues_counter)


# ── CLI ──────────────────────────────────────────────────────────


def main():
    """Backward-compatible alias — delegates to evaluate_cli.main."""
    from mineru_eval.evaluate_cli import main as _main
    return _main()


if __name__ == "__main__":
    main()
