"""Thin backward-compatible shell — re-exports from control.eval_scoring.

All implementation now lives in mineru_eval.control.eval_scoring.
"""

from mineru_eval.control.eval_scoring import (  # noqa: F401
    SCORER_METHOD,
    SCORER_VERSION,
    _build_eval_record,
    _score_one_question,
)
