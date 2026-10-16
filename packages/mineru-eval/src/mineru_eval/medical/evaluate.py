"""Medical domain evaluation entry point.

Extracted from medical_eval.py. Uses _shared.eval_runner with medical-domain
scoring callbacks injected.
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]

import mineru_core  # noqa: F401 — trigger _load_dotenv() to load .env API keys

from mineru_core.settings import PROVIDERS
from mineru_eval.client_factory import create_client, resolve_api_key
from mineru_eval.medical.scorers import MedicalScorer
from mineru_eval.medical.scoring import _build_medical_eval_record, _score_medical_question
from mineru_eval.utils import load_json, write_json_atomic

DEFAULT_INPUT = ROOT / "benchmark" / "dataset" / "medical_benchmark.json"
DEFAULT_OUTPUT = ROOT / "benchmark" / "dataset" / "medical_report.json"


def medical_evaluate(
    data,
    target_client,
    target_model,
    judge_client,
    judge_model,
    *,
    limit=0,
    provider_kind="openai",
    output_path=None,
    resume=False,
    retry_failed=False,
    retries=3,
    max_consecutive_failures=5,
    dual_judge=False,
    judge2_client=None,
    judge2_model=None,
    arb_client=None,
    arb_model=None,
    system_prompt=None,
    max_tokens: int = 4000,
):
    """Run medical domain evaluation loop via shared runner with injected callbacks.

    Args:
        system_prompt: 覆盖 medical 目标模型系统提示词（默认 None = MEDICAL_TARGET_SYSTEM_PROMPT）。
        max_tokens: 目标模型输出 token 预算（默认 4000 保持历史行为；
            补充实验 medical 域预算覆盖入口，决策报告 §3.4 映射）。
    """
    from mineru_eval._shared.eval_runner import model_evaluate

    def _score_one(q, question_text, expected,
                   target_client, target_model, provider_kind,
                   judge_client, judge_model, **kwargs):
        return _score_medical_question(
            q, question_text, expected,
            target_client, target_model, provider_kind,
            judge_client, judge_model,
            system_prompt=system_prompt, max_tokens=max_tokens,
            **kwargs,
        )

    return model_evaluate(
        data,
        target_client,
        target_model,
        judge_client,
        judge_model,
        limit=limit,
        provider_kind=provider_kind,
        output_path=output_path,
        resume=resume,
        retry_failed=retry_failed,
        retries=retries,
        max_consecutive_failures=max_consecutive_failures,
        dual_judge=dual_judge,
        judge2_client=judge2_client,
        judge2_model=judge2_model,
        arb_client=arb_client,
        arb_model=arb_model,
        score_one_fn=_score_one,
        build_record_fn=_build_medical_eval_record,
        scorer_version=MedicalScorer.VERSION,
        scorer_method=MedicalScorer.METHOD,
    )


# ── CLI ──────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Medical domain benchmark evaluation — LLM-as-Judge with identify/model/transfer dimensions."
    )
    parser.add_argument("--input", "-i", default=str(DEFAULT_INPUT), help="Medical benchmark JSON path.")
    parser.add_argument("--output", "-o", default=str(DEFAULT_OUTPUT), help="Output report path.")

    parser.add_argument("--target-model", default="deepseek-v4-flash", help="Model to evaluate.")
    parser.add_argument("--judge-model", default="deepseek-v4-flash", help="Judge model for scoring.")
    parser.add_argument("--api-key", default=None, help="API key (fallback for target/judge).")
    parser.add_argument("--target-api-key", default=None, help="Override API key for target model.")
    parser.add_argument("--judge-api-key", default=None, help="Override API key for judge model.")
    parser.add_argument("--base-url", default=PROVIDERS["deepseek"].base_url,
                        help=f"API base URL (default: {PROVIDERS['deepseek'].base_url}).")
    parser.add_argument("--target-base-url", default=None, help="Override base URL for target model.")
    parser.add_argument("--judge-base-url", default=None, help="Override base URL for judge model.")

    parser.add_argument("--limit", type=int, default=0, help="Limit number of questions to evaluate.")
    parser.add_argument("--resume", action="store_true", help="Resume from existing report.")
    parser.add_argument("--retry-failed", action="store_true", help="Re-evaluate failed records when resuming.")
    parser.add_argument("--retries", type=int, default=3, help="Retry attempts for API calls.")
    parser.add_argument("--max-consecutive-failures", type=int, default=5,
                        help="Stop early after N consecutive failures.")

    parser.add_argument("--dual-judge", action="store_true",
                        help="Enable dual-judge scoring for all medical dimensions.")
    parser.add_argument("--judge2-model", default=None, help="Second judge model (enables dual-judge).")
    parser.add_argument("--judge2-base-url", default=None, help="Base URL for second judge.")
    parser.add_argument("--judge2-api-key", default=None, help="API key for second judge.")
    parser.add_argument("--arb-model", default=None, help="Arbitrator model (for dual-judge disputes).")
    parser.add_argument("--arb-base-url", default=None, help="Base URL for arbitrator.")
    parser.add_argument("--arb-api-key", default=None, help="API key for arbitrator.")

    parser.add_argument("--allow-incomplete", action="store_true",
                        help="Exit 0 even when not all questions completed.")
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()
    if not input_path.exists():
        raise SystemExit(f"Input not found: {input_path}")

    data = load_json(input_path)

    # ── Create clients ──
    target_base_url = args.target_base_url or args.base_url
    judge_base_url = args.judge_base_url or args.base_url
    target_api_key = args.target_api_key or args.api_key
    judge_api_key = args.judge_api_key or args.api_key

    target_api_key = resolve_api_key(target_base_url, explicit_key=target_api_key, fallback_key=args.api_key)
    target_client, target_model_name, provider_kind = create_client(
        target_api_key, target_base_url, args.target_model,
    )

    judge_api_key = resolve_api_key(judge_base_url, explicit_key=judge_api_key, fallback_key=args.api_key)
    judge_client, judge_model_name, _ = create_client(
        judge_api_key, judge_base_url, args.judge_model,
    )

    # ── Dual-judge setup ──
    judge2_client = judge2_model = arb_client = arb_model = None
    dual_judge = args.dual_judge or bool(args.judge2_model)

    if dual_judge and args.judge2_model:
        j2_base = args.judge2_base_url or args.base_url
        j2_key = resolve_api_key(j2_base, explicit_key=args.judge2_api_key, fallback_key=args.api_key)
        judge2_client, judge2_model, _ = create_client(j2_key, j2_base, args.judge2_model)
        print(f"Dual-judge enabled: J1={judge_model_name}, J2={judge2_model}")

    if args.arb_model:
        arb_base = args.arb_base_url or args.base_url
        arb_key = resolve_api_key(arb_base, explicit_key=args.arb_api_key, fallback_key=args.api_key)
        arb_client, arb_model, _ = create_client(arb_key, arb_base, args.arb_model)
        print(f"Arbitrator: {arb_model}")

    # ── Run evaluation ──
    report = medical_evaluate(
        data,
        target_client, target_model_name,
        judge_client, judge_model_name,
        limit=args.limit,
        provider_kind=provider_kind,
        output_path=str(output_path),
        resume=args.resume,
        retry_failed=args.retry_failed,
        retries=args.retries,
        max_consecutive_failures=args.max_consecutive_failures,
        dual_judge=dual_judge,
        judge2_client=judge2_client,
        judge2_model=judge2_model,
        arb_client=arb_client,
        arb_model=arb_model,
    )

    write_json_atomic(output_path, report)
    print(f"\nMedical evaluation report saved: {output_path}")
    print(f"Overall score: {report['overall_score']}")
    print(f"Dimension scores: {report['dimension_scores']}")

    if not report.get("complete", True) and not args.allow_incomplete:
        sys.exit(1)


if __name__ == "__main__":
    main()
