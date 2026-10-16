"""Thin backward-compatible shell — re-exports from medical.scoring.

All implementation now lives in mineru_eval.medical.scoring.
"""

from mineru_eval.medical.scoring import (  # noqa: F401
    MEDICAL_TARGET_SYSTEM_PROMPT,
    _build_medical_eval_record,
    _call_medical_target,
    _score_medical_question,
)
