"""Thin backward-compatible shell — re-exports from control.eval_adapter.

All implementation now lives in mineru_eval.control.eval_adapter.
"""

from __future__ import annotations

import os
from pathlib import Path  # noqa: F401

# ── Must be set BEFORE any ML imports that transitively load bitsandbytes ──
os.environ.setdefault("BNB_CUDA_VERSION", "130")  # Windows CUDA 13.2 compat

import mineru_core  # noqa: F401 — trigger _load_dotenv() to load .env API keys

from mineru_eval.control.eval_adapter import (  # noqa: F401
    SCORER_VERSION,
    _init_adapter_eval,
    _judge_one,
    _save_progress,
    _select_eval_questions,
    call_target_model_local,
    evaluate_adapter,
    main,
    parse_args,
)
# eval_phases._adapter_imports() needs these two utilities
from mineru_eval._shared.call_utils import call_with_retries, format_eta  # noqa: F401


if __name__ == "__main__":
    raise SystemExit(main())
