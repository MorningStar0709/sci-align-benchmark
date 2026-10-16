"""Orchestrate the evaluation pipeline after Phase C generation completes.

NOTE: This pipeline evaluates only 3 models (DeepSeek, MiniMax, MiMo) for
quick verification by default. Use --profile report or evaluate.py directly
for full/report-grade reproduction.

Usage:
    python benchmark/eval/run_eval_pipeline.py --benchmark benchmark/dataset/benchmark.json

Steps: arbitrate (1/4) → evaluate 3 models (2/4) → summary (3/4) → export (4/4)
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
PYTHON = sys.executable

import mineru_core  # noqa: F401 — trigger _load_dotenv()
from mineru_core.settings import PROVIDERS

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except AttributeError:
    pass

# Model definitions for evaluation
# WARNING: This pipeline evaluates only 3 models (subset).
# The full ControlSci Benchmark evaluates 9 models total.
# Use benchmark/eval/evaluate.py directly for multi-model runs.
TARGET_MODELS = [
    {"name": "DeepSeek",   "model": PROVIDERS["deepseek"].default_model,  "base_url": PROVIDERS["deepseek"].base_url,  "api_key_env": PROVIDERS["deepseek"].api_key_env},
    {"name": "Qwen",     "model": "qwen3.7-plus",                          "base_url": PROVIDERS["qwen"].base_url,       "api_key_env": PROVIDERS["qwen"].api_key_env},
    {"name": "MiMo",       "model": "mimo-v2.5-pro",                      "base_url": PROVIDERS["mimo"].base_url,      "api_key_env": PROVIDERS["mimo"].api_key_env},
]

JUDGE_CONFIG = {
    "model": PROVIDERS["deepseek"].default_model,
    "base_url": PROVIDERS["deepseek"].base_url,
    "api_key_env": PROVIDERS["deepseek"].api_key_env,
}


def step(msg):
    print(f"\n{'='*60}")
    print(f"  {msg}")
    print(f"{'='*60}")


SECRET_FLAGS = {"--api-key", "--target-api-key", "--judge-api-key", "--hf-token"}


def format_cmd(cmd):
    masked = []
    redact_next = False
    for arg in cmd:
        text = str(arg)
        if redact_next:
            masked.append("***")
            redact_next = False
            continue
        if text in SECRET_FLAGS:
            masked.append(text)
            redact_next = True
            continue
        masked.append(text)
    return " ".join(masked)


def run(cmd, cwd=None, env_extra=None):
    print(f"  $ {format_cmd(cmd)}", flush=True)
    full_env = os.environ.copy()
    if env_extra:
        full_env.update(env_extra)
    result = subprocess.run(cmd, cwd=cwd or str(ROOT), capture_output=True, text=True, env=full_env)
    for line in result.stdout.strip().split("\n"):
        if line.strip():
            print(f"  {line}", flush=True)
    if result.returncode != 0:
        for line in result.stderr.strip().split("\n"):
            if line.strip():
                print(f"  ! {line}", flush=True)
        print(f"  ❌ Failed (exit={result.returncode})", flush=True)
    else:
        print("  ✅ OK", flush=True)
    return result.returncode


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def first_env(env_spec):
    for env_name in str(env_spec or "").split("|"):
        value = os.environ.get(env_name.strip(), "")
        if value:
            return value
    return ""


def primary_env(env_spec):
    return str(env_spec or "").split("|")[0].strip()


def _parse_pipeline_args():
    """Parse and resolve pipeline arguments."""
    parser = argparse.ArgumentParser(description="Evaluation pipeline: arbitrate → evaluate 3 models → summary → export.")
    parser.add_argument("--profile", choices=["smoke", "report"], default="smoke")
    parser.add_argument("--benchmark", default=str(ROOT / "benchmark" / "dataset" / "benchmark.json"))
    parser.add_argument("--review", default=str(ROOT / "benchmark" / "dataset" / "manual_review.json"))
    parser.add_argument("--output-dir", default=str(ROOT / "benchmark" / "eval_output"))
    parser.add_argument("--skip-arbitrate", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--skip-export", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--push-to-hub", default=None)
    parser.add_argument("--hf-token", default=None)
    parser.add_argument("--supplement-target", type=int, default=500)
    args = parser.parse_args()
    if args.limit is None:
        args.limit = 10 if args.profile == "smoke" else 0
    benchmark_path = Path(args.benchmark).resolve()
    review_path = Path(args.review).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[EvalPipeline] profile={args.profile} limit={args.limit if args.limit > 0 else 'ALL'} output={output_dir}", flush=True)
    return args, benchmark_path, review_path, output_dir


def _step_arbitrate(args, benchmark_path, review_path):
    """Step 1: Run arbitration."""
    if not args.skip_arbitrate:
        step("Step 1/4: Arbitration")
        rc = run([PYTHON, "benchmark/pipeline/build_benchmark.py", "--arbitrate",
                  "--output", str(benchmark_path), "--manual-review-output", str(review_path)])
        if rc != 0:
            print("Arbitration failed. Continuing with un-arbitrated benchmark.", flush=True)
        post_count = len(load_json(str(benchmark_path)).get("questions", []))
        print(f"  Post-arbitration question count: {post_count}", flush=True)
    else:
        print("⏩ Skipping arbitration.", flush=True)


def _step_evaluate(args, benchmark_path, output_dir):
    """Step 2: Evaluate each target model."""
    if args.skip_eval:
        print("⏩ Skipping model evaluation.", flush=True)
        return
    for model_def in TARGET_MODELS:
        step(f"Step 2/4: Evaluate {model_def['name']}")
        output_path = output_dir / f"report_{model_def['name'].lower()}.json"
        target_api_key = first_env(model_def["api_key_env"])
        if not target_api_key:
            print(f"  ⚠️ No API key for {model_def['name']} ({model_def['api_key_env'].replace('|', '/')}), skipping...", flush=True)
            continue
        judge_api_key = first_env(JUDGE_CONFIG["api_key_env"])
        cmd = [PYTHON, "benchmark/eval/evaluate.py", "--mode", "model",
               "--input", str(benchmark_path), "--output", str(output_path),
               "--target-model", model_def["model"], "--target-base-url", model_def["base_url"],
               "--judge-model", JUDGE_CONFIG["model"], "--judge-base-url", JUDGE_CONFIG["base_url"],
               "--format", "json"]
        if args.limit > 0:
            cmd.extend(["--limit", str(args.limit)])
        env_extra = {}
        env_extra[primary_env(model_def["api_key_env"])] = target_api_key
        judge_env = primary_env(JUDGE_CONFIG["api_key_env"])
        if judge_env not in env_extra and judge_api_key:
            env_extra[judge_env] = judge_api_key
        rc = run(cmd, env_extra=env_extra)
        if rc == 0:
            print(f"  Report saved: {output_path}", flush=True)
        else:
            print(f"  ❌ {model_def['name']} evaluation failed, continuing...", flush=True)


def _step_summary(start_time, args, benchmark_path, output_dir):
    """Step 3: Generate comparison summary."""
    if args.skip_eval:
        return
    step("Step 3/4: Generate comparison summary")
    summary = {
        "pipeline_start": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time)),
        "profile": args.profile, "limit": args.limit,
        "benchmark_source": str(benchmark_path), "models_evaluated": [],
    }
    for model_def in TARGET_MODELS:
        json_path = output_dir / f"report_{model_def['name'].lower()}.json"
        if json_path.exists():
            report = load_json(str(json_path))
            summary["models_evaluated"].append({
                "name": model_def["name"],
                "overall_score": report.get("overall_score", 0),
                "total_questions": report.get("total_questions", 0),
                "dimension_scores": report.get("dimension_scores", {}),
            })
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"  Summary: {summary_path}", flush=True)
    for m in summary["models_evaluated"]:
        print(f"    {m['name']:>8s}: overall={m['overall_score']:.2f}, n={m['total_questions']}", flush=True)


def _step_export(args, benchmark_path, output_dir):
    """Step 4: Export to HuggingFace Hub."""
    if not args.push_to_hub:
        print("⏩ Skipping HF export. Pass --push-to-hub <repo> to enable.", flush=True)
        return
    step("Step 4/4: Export to HuggingFace Hub")
    hf_token = args.hf_token or os.environ.get("HF_TOKEN", "")
    cmd = [PYTHON, "benchmark/pipeline/export_dataset.py",
           "--input", str(benchmark_path), "--output", str(output_dir / "hf_export"),
           "--push-to-hub", args.push_to_hub]
    if hf_token:
        cmd.extend(["--hf-token", hf_token])
    run(cmd)



def main():
    args, benchmark_path, review_path, output_dir = _parse_pipeline_args()

    if not benchmark_path.exists():
        raise SystemExit(f"Benchmark not found: {benchmark_path}")

    start_time = time.time()
    _step_arbitrate(args, benchmark_path, review_path)
    _step_evaluate(args, benchmark_path, output_dir)
    _step_summary(start_time, args, benchmark_path, output_dir)
    _step_export(args, benchmark_path, output_dir)

    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"  Pipeline complete in {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"  Output: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
