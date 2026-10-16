"""CLI entry point for ControlSci benchmark evaluation.

Extracted from evaluate.py to separate CLI argument parsing from
evaluation logic.
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]

import mineru_core  # noqa: F401 — trigger _load_dotenv() to load .env API keys

from mineru_core.settings import PROVIDERS
from mineru_eval.client_factory import create_client, resolve_api_key
from mineru_eval.evaluate import (
    DEFAULT_INPUT,
    DEFAULT_OUTPUT,
    evaluate,
    model_evaluate,
    render_md_report,
    write_json_atomic,
)
from mineru_eval.utils import load_json


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a ControlSci benchmark file. Supports reference (data integrity) and model (LLM-as-Judge) modes."
    )
    parser.add_argument("--input", "-i", default=str(DEFAULT_INPUT), help="Benchmark JSON path.")
    parser.add_argument("--output", "-o", default=str(DEFAULT_OUTPUT), help="Output report path.")
    parser.add_argument(
        "--mode",
        default="reference",
        choices=["reference", "model"],
        help="Evaluation mode: reference (data integrity check) or model (real LLM evaluation).",
    )

    parser.add_argument("--target-model", default="deepseek-v4-flash", help="Model to evaluate (model mode only).")
    parser.add_argument("--judge-model", default="deepseek-v4-flash", help="Judge model for LLM-as-Judge (model mode only).")
    parser.add_argument("--api-key", default=None, help="API key for model/judge calls (fallback for both target/judge).")
    parser.add_argument("--target-api-key", default=None, help="Override API key for target model.")
    parser.add_argument("--judge-api-key", default=None, help="Override API key for judge model.")
    parser.add_argument("--base-url", default=PROVIDERS["deepseek"].base_url, help=f"API base URL (default: {PROVIDERS['deepseek'].base_url}).")
    parser.add_argument("--target-base-url", default=None, help="Override base URL for target model.")
    parser.add_argument("--judge-base-url", default=None, help="Override base URL for judge model.")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of questions to evaluate (model mode only).")
    parser.add_argument("--balanced-mini", action="store_true",
                        help="With --limit, select questions round-robin across A/B/C/D instead of taking the first N.")
    parser.add_argument("--resume", action="store_true", help="Resume from an existing JSON report and skip completed question IDs.")
    parser.add_argument("--retry-failed", action="store_true", help="When resuming, recompute records marked with failure/error issues.")
    parser.add_argument("--retries", type=int, default=3, help="Retry attempts for target and judge API calls.")
    parser.add_argument("--max-consecutive-failures", type=int, default=5, help="Stop early after this many consecutive failed records.")
    parser.add_argument(
        "--dual-judge",
        action="store_true",
        help="Enable hierarchical dual-judge for C/D dimensions "
             "(Judge1: deepseek-v4-flash, Judge2: mimo-v2.5, Arbiter: qwen3.7-plus).",
    )
    parser.add_argument(
        "--all-dimensions-dual",
        action="store_true",
        help="With --dual-judge, enable dual-judge for ALL dimensions (A/B/C/D), not just C/D.",
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Exit 0 even when model-mode evaluation stops before all selected questions complete.",
    )
    parser.add_argument(
        "--format",
        default="json",
        choices=["json", "md"],
        help="Output format: json or md (model mode only).",
    )
    # Reference-mode only backward compat
    parser.add_argument("--model", dest="model_meta", default="reference-heuristic", help="[Reference mode only] Model name for metadata.")
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()
    if not input_path.exists():
        raise SystemExit(f"Input not found: {input_path}")

    data = load_json(input_path)

    if args.mode == "reference":
        report = evaluate(data, args.model_meta)
        write_json_atomic(output_path, report)
        print(f"Report saved: {output_path}")
        print(f"Overall score: {report['overall_score']}")
        return

    # ── Model mode ──
    target_base_url = args.target_base_url or args.base_url
    judge_base_url = args.judge_base_url or args.base_url
    target_api_key = resolve_api_key(target_base_url, explicit_key=args.target_api_key, fallback_key=args.api_key)
    judge_api_key = resolve_api_key(judge_base_url, explicit_key=args.judge_api_key, fallback_key=args.api_key)

    print(f"Target model: {args.target_model} @ {target_base_url}")
    print(f"Judge model: {args.judge_model} @ {judge_base_url}")
    print(f"Limit: {args.limit or 'unlimited'}")
    if args.balanced_mini:
        print("Selection: balanced-mini")

    target_client, target_model, target_provider_kind = create_client(target_api_key, target_base_url, args.target_model)
    judge_client, judge_model, _ = create_client(judge_api_key, judge_base_url, args.judge_model)

    # ── Dual-Judge 客户端初始化（仅 --dual-judge 时创建） ──
    dj2_client = dj2_model = arb_client = arb_model = None
    if args.dual_judge:
        mimo_base = PROVIDERS["mimo"].base_url
        mimo_key = resolve_api_key(mimo_base, provider="mimo")
        dj2_client, dj2_model, _ = create_client(mimo_key, mimo_base, "mimo-v2.5")
        qwen_base = PROVIDERS["qwen"].base_url
        qwen_key = resolve_api_key(qwen_base, provider="qwen")
        arb_client, arb_model, _ = create_client(qwen_key, qwen_base, "qwen3.7-plus")
        print(f"Dual-Judge: Judge2={dj2_model} @ {mimo_base}, Arbiter={arb_model} @ {qwen_base}")
    elif args.all_dimensions_dual:
        print(
            "[WARNING] --all-dimensions-dual requires --dual-judge, ignoring",
            file=sys.stderr,
        )

    state_output_path = output_path
    if args.format == "md" and output_path.suffix.lower() != ".json":
        state_output_path = output_path.with_suffix(".json")

    report = model_evaluate(
        data,
        target_client,
        target_model,
        judge_client,
        judge_model,
        limit=args.limit,
        provider_kind=target_provider_kind,
        output_path=state_output_path,
        resume=args.resume,
        retry_failed=args.retry_failed,
        retries=args.retries,
        max_consecutive_failures=args.max_consecutive_failures,
        balanced_mini=args.balanced_mini,
        dual_judge=args.dual_judge,
        judge2_client=dj2_client,
        judge2_model=dj2_model,
        arb_client=arb_client,
        arb_model=arb_model,
        all_dimensions_dual=args.all_dimensions_dual,
    )

    if args.format == "md":
        md_content = render_md_report(report)
        output_path = output_path.with_suffix(".md") if output_path.suffix == ".json" else output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(md_content, encoding="utf-8")
        print(f"Markdown report saved: {output_path}")
    else:
        write_json_atomic(output_path, report)
        print(f"JSON report saved: {output_path}")
        print(f"Overall score: {report['overall_score']}")

    if not report.get("complete", False) and not args.allow_incomplete:
        print(
            "Evaluation incomplete: "
            f"{report.get('completed', 0)}/{report.get('total', 0)} completed "
            f"({report.get('incomplete_reason', 'unknown')}). "
            "Use --resume --retry-failed after fixing the issue, or pass --allow-incomplete for exploratory runs.",
            file=sys.stderr,
        )
        raise SystemExit(2)


if __name__ == "__main__":
    main()
