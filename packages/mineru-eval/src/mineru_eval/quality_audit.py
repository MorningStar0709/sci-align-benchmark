"""Thin backward-compatible shell — re-exports from control.quality_audit.

All implementation now lives in mineru_eval.control.quality_audit.
"""

from mineru_eval.control.quality_audit import (  # noqa: F401
    AUDIT_SYSTEM_PROMPT,
    DEFAULT_CORE,
    DEFAULT_OUTPUT,
    DEFAULT_SAMPLE,
    _get_deepseek_key,
    _parse_audit_result,
    audit_one,
    build_audit_prompt,
    compute_summary,
    main,
)


if __name__ == "__main__":
    main()
