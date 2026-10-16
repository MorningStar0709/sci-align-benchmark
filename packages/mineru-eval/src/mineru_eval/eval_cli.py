"""CLI entry point for FR-LoRA adapter evaluation.

Extracted from eval_adapter.py to separate CLI argument parsing from
evaluation logic.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# ── Must be set BEFORE any ML imports that transitively load bitsandbytes ──
import os
os.environ.setdefault("BNB_CUDA_VERSION", "130")  # Windows CUDA 13.2 compat

# Project root
ROOT = Path(__file__).resolve().parents[4]

import mineru_core  # noqa: F401 — trigger _load_dotenv() to load .env API keys
from mineru_core.settings import PROVIDERS

from mineru_eval.client_factory import create_client, resolve_api_key
from mineru_eval.eval_adapter import evaluate_adapter
from mineru_eval.utils import load_json, write_json_atomic


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a FR-LoRA adapter via local inference + DeepSeek judge."
    )
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument(
        "--adapter-path",
        type=Path,
        default=None,
        help="Path to LoRA adapter checkpoint directory",
    )
    model_group.add_argument(
        "--raw",
        action="store_true",
        help="Run base model without any LoRA adapter (pure baseline)",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "benchmark" / "dataset" / "core.json",
        help="Benchmark JSON path",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output report JSON path",
    )
    parser.add_argument(
        "--judge-model",
        default="deepseek-v4-flash",
        help="Judge model name (DeepSeek API)",
    )
    parser.add_argument(
        "--judge-base-url",
        default=PROVIDERS["deepseek"].base_url,
    )
    parser.add_argument("--judge-api-key", default=None)
    parser.add_argument(
        "--base-model-id",
        default=None,
        help="Override base model HF ID (default: from adapter_config.json)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit number of questions (0 = all)",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=1,
        metavar="N",
        help="Number of parallel judge API workers (1=serial, 2-8 recommended)",
    )
    parser.add_argument(
        "--balanced-mini",
        action="store_true",
        help="With --limit, select round-robin across A/B/C/D",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing output, skip completed IDs",
    )
    parser.add_argument(
        "--dual-judge",
        action="store_true",
        help="Enable hierarchical dual-judge for C/D dimensions "
             "(Judge1: deepseek-v4-flash, Judge2: mimo-v2.5, Arbiter: minimax-m2.5).",
    )
    parser.add_argument(
        "--all-dimensions-dual",
        action="store_true",
        help="With --dual-judge, enable dual-judge for ALL dimensions (A/B/C/D) "
             "instead of only C/D.",
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="When resuming, retry records with failure/error issues",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Retry attempts for local inference and judge API calls",
    )
    parser.add_argument(
        "--max-consecutive-failures",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--max-answer-tokens",
        type=int,
        default=2048,
        help="Max tokens for local model answer generation (default: 2048)",
    )
    parser.add_argument(
        "--fixed-questions",
        type=Path,
        default=None,
        help="Use pre-extracted question set JSON (bypasses select_questions for reproducibility)",
    )
    parser.add_argument(
        "--answers-cache",
        type=Path,
        default=None,
        help="Path to a previous run's output JSON to load answers from. "
             "If exists: load model_answer from records, skip GPU inference. "
             "If not: run GPU inference, answers saved in this run's output.",
    )
    parser.add_argument(
        "--domain",
        type=str,
        default="control",
        choices=["control", "medical"],
        help="Evaluation domain: 'control' for A/B/C/D dimensions, 'medical' for identify/model/transfer",
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Exit 0 even when not all questions completed",
    )
    parser.add_argument(
        "--target-system-prompt",
        type=str,
        default=None,
        help="Override target model system prompt (default None = built-in prompt). "
             "补充实验 2×2 格子③④告知措辞注入（决策报告 §3.4）.",
    )
    parser.add_argument(
        "--medical-max-tokens",
        type=int,
        default=None,
        help="Medical domain max answer tokens override (default None = use --max-answer-tokens). "
             "补充实验 medical 域预算覆盖（决策报告 §3.4 映射）.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.parallel < 1:
        print("[ERROR] --parallel must be >= 1", file=sys.stderr)
        return 1
    if args.parallel > 16:
        print(
            "[WARNING] --parallel > 16 may trigger API rate limits",
            file=sys.stderr,
        )

    # 1. Load benchmark data
    if not args.input.is_file():
        print(f"[ERROR] Input not found: {args.input}", file=sys.stderr)
        return 1
    data = load_json(args.input)
    print(f"Loaded {len(data.get('questions', []))} questions from {args.input}")

    # Detect judge endpoint for conditional extra_body
    judge_base_url_lower = args.judge_base_url.lower()
    disable_thinking = "deepseek.com" in judge_base_url_lower
    if disable_thinking:
        print("DeepSeek judge detected — disabling thinking mode via extra_body")

    # 2. Create DeepSeek judge client
    judge_api_key = resolve_api_key(
        args.judge_base_url, explicit_key=args.judge_api_key
    )
    judge_client, judge_model, _ = create_client(
        judge_api_key, args.judge_base_url, args.judge_model
    )
    print(f"Judge model: {judge_model} @ {args.judge_base_url}")

    # 2b. Dual-Judge 客户端初始化（仅 --dual-judge 时创建）
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

    # 3. Load model (PeftModel with adapter, or raw base model)
    base_model_id = args.base_model_id or "principled-intelligence/Qwen3.5-4B-text-only"
    model = None
    try:
        if args.raw:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch as _torch
            print(f"Loading raw base model: {base_model_id}")
            # 强制离线加载：模型已全部本地缓存，HF Hub 在国内不稳定（与 adapter_loader 同款防护，
            # 2026-08-09 网络抖动导致 raw_9b_med gen3 秒退后补上）。
            model = AutoModelForCausalLM.from_pretrained(
                base_model_id, torch_dtype=_torch.bfloat16, device_map="auto",
                trust_remote_code=True, local_files_only=True,
            )
            tokenizer = AutoTokenizer.from_pretrained(
                base_model_id, trust_remote_code=True, local_files_only=True,
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            model.eval()
            adapter_name = f"raw-{base_model_id.split('/')[-1]}"
        else:
            from mineru_eval.finetune.adapter_loader import load_peft_model

            if not args.adapter_path.is_dir():
                print(
                    f"[ERROR] Adapter not found: {args.adapter_path}",
                    file=sys.stderr,
                )
                return 1
            print(f"Loading adapter: {args.adapter_path}")
            model, tokenizer = load_peft_model(
                adapter_path=args.adapter_path,
                base_model_id=args.base_model_id,
                is_trainable=False,
            )
            model.eval()
            # Build a human-readable adapter name
            rel = (
                args.adapter_path.relative_to(ROOT)
                if args.adapter_path.is_relative_to(ROOT)
                else args.adapter_path
            )
            adapter_name = str(rel).replace("\\", "/")
        print(f"Model device: {model.device}")
        if not args.fixed_questions:
            print(f"Total questions for this run: {args.limit or 'all'}")

        # 3b. Build scoring function based on domain
        external_score_fn = None
        if args.domain == "medical":
            from mineru_eval.medical.scorers import score_medical_answer
            # Medical domain: use score_medical_answer with difficulty from question
            def _medical_score_wrapper(question, expected, model_answer, dimension,
                                       client, model, *, rubric=None, disable_thinking=False):
                # Extract difficulty from question dict (passed via rubric param workaround)
                # The actual difficulty is extracted in _score_or_fallback via q.get()
                return score_medical_answer(
                    question, expected, model_answer, dimension,
                    client, model, difficulty="L2",  # default, overridden by _score_or_fallback
                    disable_thinking=disable_thinking,
                )
            external_score_fn = _medical_score_wrapper
            print("Domain: medical (identify/model/transfer dimensions)")
        else:
            print("Domain: control (A/B/C/D dimensions)")

        # 2c. Medical 域预算覆盖：--medical-max-tokens 优先于 --max-answer-tokens
        # （补充实验 C 阶段 medical 域 8192 预算注入点，决策报告 §3.4 映射）
        answer_tokens = (
            args.medical_max_tokens
            if args.domain == "medical" and args.medical_max_tokens is not None
            else args.max_answer_tokens
        )

        # 4. Run evaluation
        report = evaluate_adapter(
            data,
            model,
            tokenizer,
            judge_client,
            judge_model,
            adapter_name,
            limit=args.limit,
            balanced_mini=args.balanced_mini,
            fixed_questions_path=args.fixed_questions,
            output_path=args.output,
            resume=args.resume,
            retry_failed=args.retry_failed,
            retries=args.retries,
            max_consecutive_failures=args.max_consecutive_failures,
            parallel=args.parallel,
            disable_thinking=disable_thinking,
            max_answer_tokens=answer_tokens,
            dual_judge=args.dual_judge,
            judge2_client=dj2_client,
            judge2_model=dj2_model,
            arb_client=arb_client,
            arb_model=arb_model,
            answers_cache=args.answers_cache,
            all_dimensions_dual=args.all_dimensions_dual,
            external_score_fn=external_score_fn,
            system_prompt=args.target_system_prompt,
        )

        # 5. Save report
        print(f"\nOverall score: {report.get('overall_score', 'N/A')}")
        print(f"Dimension scores: {report.get('dimension_scores', {})}")
        print(f"Complete: {report.get('complete', False)}")
        write_json_atomic(args.output, report)
        print(f"\nReport saved: {args.output}")

        if not report.get("complete", False) and not args.allow_incomplete:
            print(
                f"Incomplete: {report.get('completed', 0)}/{report.get('total', 0)} "
                f"({report.get('incomplete_reason', 'unknown')}). "
                f"Use --resume --retry-failed or --allow-incomplete.",
                file=sys.stderr,
            )
            return 2
        return 0
    finally:
        # GPU cleanup: release model reference and reclaim VRAM
        if model is not None:
            del model
        try:
            import torch as _torch_cleanup
            if _torch_cleanup.cuda.is_available():
                _torch_cleanup.cuda.empty_cache()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
