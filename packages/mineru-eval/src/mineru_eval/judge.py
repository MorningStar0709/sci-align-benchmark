"""Thin backward-compatible shell — re-exports from _shared and control subpackages.

All implementation now lives in:
  - mineru_eval._shared.json_utils    (_extract_json_block, _parse_judge_result, _coerce_float)
  - mineru_eval._shared.scoring_base  (_dual_judge_score, _score_via_anthropic, _snap_to_valid, rubric helpers)
  - mineru_eval.control.scorers       (AScorer, BScorer, CScorer, DScorer, GenericScorer, RubricScorer)
  - mineru_eval.control.prompts       (A/B/C/D prompt constants)
  - mineru_eval.medical.prompts       (MEDICAL_JUDGE_SYSTEM_PROMPT, MEDICAL_USER_PROMPT_TEMPLATE)
"""

import argparse
import json

import mineru_core  # noqa: F401 — trigger _load_dotenv() to load .env API keys

from mineru_core.settings import PROVIDERS
from mineru_eval.client_factory import create_client

# ── Re-exports from _shared ────────────────────────────────────────
from mineru_eval._shared.json_utils import (  # noqa: F401
    _coerce_float,
    _extract_json_block,
    _parse_judge_result,
    VALID_SCORES,
)
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

# ── Re-exports from control subpackage ──────────────────────────────
from mineru_eval.control.prompts import (  # noqa: F401
    A_SYSTEM_PROMPT,
    A_USER_PROMPT_TEMPLATE,
    B_SYSTEM_PROMPT,
    B_USER_PROMPT_TEMPLATE,
    C_SYSTEM_PROMPT,
    C_USER_PROMPT_TEMPLATE,
    D_SYSTEM_PROMPT,
    D_USER_PROMPT_TEMPLATE,
)
from mineru_eval.control.scorers import (  # noqa: F401
    AScorer,
    BScorer,
    CScorer,
    DScorer,
    GenericScorer,
    RubricScorer,
    generic_judge,
)

# ── Re-exports from medical subpackage (for kb_quality.py cross-package) ──
from mineru_eval.medical.prompts import (  # noqa: F401
    MEDICAL_JUDGE_SYSTEM_PROMPT,
    MEDICAL_USER_PROMPT_TEMPLATE,
)


def create_judge_client(api_key=None, base_url=None, model=None):
    base_url = base_url or PROVIDERS["deepseek"].base_url
    model = model or PROVIDERS["deepseek"].default_model
    client, model_name, _ = create_client(api_key, base_url, model)
    return client, model_name


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description="LLM-as-Judge 评分器 — 控制科学基准测试"
    )
    parser.add_argument("--dimension", choices=["A", "B", "C", "D"], required=True,
                        help="评分维度")
    parser.add_argument("--question", required=True, help="原始问题文本")
    parser.add_argument("--expected", required=True, help="标准答案")
    parser.add_argument("--answer", required=True, help="模型回答")
    parser.add_argument("--api-key", default=None, help="API 密钥")
    parser.add_argument("--base-url", default=PROVIDERS["deepseek"].base_url, help="API 基础地址")
    parser.add_argument("--model", default=PROVIDERS["deepseek"].default_model, help="模型名称")
    return parser.parse_args()


def main():
    args = parse_args()
    client, model_name = create_judge_client(
        api_key=args.api_key, base_url=args.base_url, model=args.model,
    )
    scorer_map = {"A": AScorer, "B": BScorer, "C": CScorer, "D": DScorer}
    scorer = scorer_map.get(args.dimension, DScorer)
    result = scorer.judge(
        question=args.question, expected_answer=args.expected,
        model_answer=args.answer, client=client, model_name=model_name,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
