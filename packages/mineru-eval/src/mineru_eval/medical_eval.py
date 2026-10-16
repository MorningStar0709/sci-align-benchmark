"""Thin backward-compatible shell — re-exports from medical.evaluate.

All implementation now lives in mineru_eval.medical.evaluate.
"""

from mineru_eval.medical.evaluate import (  # noqa: F401
    DEFAULT_INPUT,
    DEFAULT_OUTPUT,
    main,
    medical_evaluate,
)


if __name__ == "__main__":
    main()
