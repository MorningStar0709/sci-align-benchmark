"""Thin backward-compatible shell — re-exports from _shared.scoring_base.

All implementation now lives in mineru_eval._shared.scoring_base.
"""

from mineru_eval._shared.scoring_base import (  # noqa: F401
    UNIFIED_VALID_SCORES,
    _dual_judge_score,
    _format_rubric_item,
    _rubric_judge_batch,
    _rubric_judge_single,
    _score_via_anthropic,
    _snap_to_valid,
    RUBRIC_BATCH_SYSTEM_PROMPT,
)
