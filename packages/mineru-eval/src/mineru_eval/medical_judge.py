"""Thin backward-compatible shell — re-exports from medical.scorers and medical.prompts.

All implementation now lives in:
  - mineru_eval.medical.scorers  (MedicalScorer, score_medical_answer, _score_via_anthropic_medical, etc.)
  - mineru_eval.medical.prompts  (MEDICAL_PROMPTS, rubric builders)
"""

import logging

_logger = logging.getLogger(__name__)

# ── Re-exports from medical.scorers ─────────────────────────────────
from mineru_eval.medical.scorers import (  # noqa: F401
    VALID_MEDICAL_SCORES,
    MedicalScorer,
    _greedy_parse,
    _score_via_anthropic_medical,
    score_medical_answer,
)

# ── Re-exports from medical.prompts ─────────────────────────────────
from mineru_eval.medical.prompts import (  # noqa: F401
    MEDICAL_JUDGE_SYSTEM_PROMPT,
    MEDICAL_PROMPTS,
    MEDICAL_TARGET_SYSTEM_PROMPT,
    MEDICAL_USER_PROMPT_TEMPLATE,
    _build_identify_rubric,
    _build_model_rubric,
    _build_transfer_rubric,
)
