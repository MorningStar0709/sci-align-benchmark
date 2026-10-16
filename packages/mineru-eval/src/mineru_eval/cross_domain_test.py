"""
Cross-Domain Generalization Experiment (S3*)

Runs small general-purpose llama.cpp models on Sci-Align core questions,
contrasting against the domain-tuned baseline (qwen3.5-9b, overall=0.6249).

Models:
  - qwen3.5-2b   (Qwen family size-scaling reference)
  - gemma3-4b    (Google general-purpose ceiling)

Judge: deepseek-v4-flash (same as all other evaluations for comparability)

优化 (2026-05-06):
  - API Key 默认已知可用值，不再依赖环境变量
  - 运行前 llama.cpp health 预检
  - --background 后台模式：start_new_session 脱离终端 + 日志重定向 + status.json 实时监控
"""

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]

import mineru_core  # noqa: F401 — trigger _load_dotenv()
from mineru_core.settings import PROVIDERS

from mineru_eval.utils import load_json, write_json_atomic

OLLAMA_BASE_URL = "http://localhost:8080/v1"  # legacy alias — actually llama.cpp

DEEPSEEK_KEY = os.getenv("DEEPSEEK_API_KEY", "")
JUDGE_MODEL = PROVIDERS["deepseek"].default_model
JUDGE_BASE_URL = PROVIDERS["deepseek"].base_url

CROSS_DOMAIN_MODELS = [
    {"name": "qwen3.5-2b",   "base_url": OLLAMA_BASE_URL, "family": "Qwen",   "size": "2B"},
    {"name": "gemma3-4b",    "base_url": OLLAMA_BASE_URL, "family": "Google", "size": "4B"},
]

BASELINE = {
    "model": "qwen3.5-9b",
    "overall_score": 0.6249,
    "dimension_scores": {"A": 0.5688, "B": 0.6104, "C": 0.6620, "D": 0.6586},
    "total_questions": 500,
    "complete": True,
    "family": "Qwen",
    "size": "9B",
    "note": "Domain-tuned baseline (llama.cpp, 9B params)",
}

DEFAULT_INPUT = str(ROOT / "benchmark" / "dataset" / "core.json")
DEFAULT_OUTPUT_DIR = str(ROOT / "benchmark" / "eval" / "reports")
STATUS_FILE = str(ROOT / "benchmark" / "eval" / "reports" / "cross_domain.status.json")
LOG_DIR = str(ROOT / "benchmark" / "eval" / "reports")

_MODEL_FAMILY_MAP = {m["name"]: (m["family"], m["size"]) for m in CROSS_DOMAIN_MODELS}


def write_status(data: dict):
    path = Path(STATUS_FILE)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _build_stratified_input(input_path: str, limit: int, output_dir: str) -> str:
    """Build a dimension-balanced + shuffled input file.

    从 core.json 的四维各随机抽 limit//4 题 (seed=42 可复现), 打乱后写入 temp JSON.
    """
    import random as _random

    data = load_json(input_path)
    questions = data.get("questions", [])
    buckets = {"A": [], "B": [], "C": [], "D": []}
    for q in questions:
        dim = q.get("dimension", "?")
        if dim in buckets:
            buckets[dim].append(q)
    per_dim = max(limit // 4, 1)
    _random.seed(42)
    selected = []
    for dim in ["A", "B", "C", "D"]:
        pool = buckets[dim]
        n = min(per_dim, len(pool))
        selected.extend(_random.sample(pool, n))
    _random.shuffle(selected)

    out_path = str(Path(output_dir) / f"cross_domain_balanced_{limit}.json")
    payload = {"meta": data.get("meta", {}), "questions": selected}
    write_json_atomic(out_path, payload)
    dims = {}
    for q in selected:
        d = q.get("dimension", "?")
        dims[d] = dims.get(d, 0) + 1
    print(f"  [BALANCED] {len(selected)} 题, seed=42, shuffled | dims: {dims}", flush=True)
    return out_path


def check_ollama_model(model_name: str, auto_pull: bool = True) -> bool:
    """Check if llama.cpp server is healthy (model availability is implicit — server serves the loaded model)."""
    from mineru_tools.llm_client import LocalLLMClient
    client = LocalLLMClient()
    if client.is_chat_healthy():
        print("  [llama.cpp] ✅ server healthy", flush=True)
        return True
    print("  [llama.cpp] ❌ server not reachable on port 8080", flush=True)
    return False


def precheck_ollama() -> bool:
    """Verify llama.cpp chat server is running and healthy."""
    print("[PRECHECK] llama.cpp 服务状态...", flush=True)
    from mineru_tools.llm_client import LocalLLMClient
    client = LocalLLMClient()
    if not client.is_chat_healthy():
        print("[PRECHECK] ❌ llama.cpp 不可用 (port 8080 无响应)", flush=True)
        return False
    print("[PRECHECK] ✅ llama.cpp 运行中", flush=True)
    return True


def run_single_model(model_cfg, judge_model, judge_base_url, input_path, output_dir,
                     limit=0, resume=False, retry_failed=False, retries=3, max_cf=5,
                     judge_api_key=None, background=False):
    model_name = model_cfg["name"]
    safe_name = model_name.replace(":", "-").replace("/", "-")
    output_path = str(Path(output_dir) / f"cross_domain_{safe_name}_report.json")

    cmd = [
        sys.executable,
        str(ROOT / "benchmark" / "eval" / "evaluate.py"),
        "--mode", "model",
        "--input", input_path,
        "--output", output_path,
        "--target-model", model_name,
        "--target-base-url", model_cfg["base_url"],
        "--judge-model", judge_model,
        "--judge-base-url", judge_base_url,
        "--judge-api-key", judge_api_key,
        "--retries", str(retries),
        "--max-consecutive-failures", str(max_cf),
    ]
    if limit > 0:
        cmd += ["--limit", str(limit)]
    if resume:
        cmd.append("--resume")
    if retry_failed:
        cmd.append("--retry-failed")

    if background:
        return _run_background(cmd, model_name, safe_name, output_path, model_cfg)

    print(f"\n{'='*60}")
    print(f"  Cross-Domain Model: {model_name} ({model_cfg['family']} {model_cfg['size']})")
    print(f"  Output: {output_path}")
    print(f"{'='*60}\n", flush=True)

    start = time.time()
    result = subprocess.run(cmd, cwd=str(ROOT))
    elapsed = time.time() - start

    mins = int(elapsed // 60)
    secs = int(elapsed % 60)
    if result.returncode == 0:
        print(f"\n[OK] {model_name} completed in {mins}m{secs}s → {output_path}", flush=True)
    else:
        print(f"\n[FAIL] {model_name} exit={result.returncode} after {mins}m{secs}s", flush=True)

    return {
        "model": model_name,
        "family": model_cfg["family"],
        "size": model_cfg["size"],
        "output_path": output_path,
        "exit_code": result.returncode,
        "elapsed_sec": elapsed,
    }


def _run_background(cmd, model_name, safe_name, output_path, model_cfg):
    """Launch model evaluation as a detached background process with log redirection."""
    log_path = str(ROOT / "benchmark" / "eval" / "reports" / f"cross_domain_{safe_name}.log")
    err_path = str(ROOT / "benchmark" / "eval" / "reports" / f"cross_domain_{safe_name}.err.log")
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)

    print(f"  [BG] 启动后台: {model_name} ({model_cfg['family']} {model_cfg['size']})", flush=True)
    print(f"  [BG] stdout → {log_path}", flush=True)
    print(f"  [BG] stderr → {err_path}", flush=True)

    popen_kwargs = {"start_new_session": True}  # 脱离控制终端，父会话退出不带走子进程

    with open(log_path, "w", encoding="utf-8") as stdout_f, open(err_path, "w", encoding="utf-8") as stderr_f:
        proc = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            stdout=stdout_f,
            stderr=stderr_f,
            **popen_kwargs,
        )

    write_status({
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "pid": proc.pid,
        "models": {
            model_name: {
                "status": "running",
                "output_path": output_path,
                "log_path": log_path,
                "err_path": err_path,
                "pid": proc.pid,
            }
        },
        "updated_at": datetime.now(timezone.utc).isoformat(),
    })

    print(f"  [BG] PID={proc.pid} 已启动，继续下一个模型...", flush=True)
    return {
        "model": model_name,
        "family": model_cfg["family"],
        "size": model_cfg["size"],
        "output_path": output_path,
        "exit_code": 0,
        "elapsed_sec": 0,
        "pid": proc.pid,
    }


def load_model_results(output_dir):
    results_dir = Path(output_dir)
    reports = []
    for path in sorted(results_dir.glob("cross_domain_*_report.json")):
        if path.name.endswith((".status.json", ".progress.jsonl")):
            continue
        try:
            data = load_json(path)
            model_name = data.get("model", path.stem)
            family, size = _MODEL_FAMILY_MAP.get(model_name, ("—", "—"))
            reports.append({
                "model": model_name,
                "family": family,
                "size": size,
                "overall_score": data.get("overall_score"),
                "total_questions": data.get("total_questions", 0),
                "dimension_scores": data.get("dimension_scores", {}),
                "complete": data.get("complete", False),
                "failure_summary": data.get("failure_summary", {}),
                "source_file": path.name,
            })
        except Exception as e:
            print(f"  Skipping {path.name}: {e}")
    return reports


def build_comparison(reports, baseline):
    all_dims = ["A", "B", "C", "D"]
    dim_labels = {
        "A": "概念定义与数学表达",
        "B": "多步推理与计算求解",
        "C": "敏感性分析与方案对比",
        "D": "完整控制方案设计与综合评估",
    }

    rows = [baseline]
    for r in sorted(reports, key=lambda x: x.get("overall_score") or 0, reverse=True):
        rows.append({
            "model": r["model"],
            "family": r.get("family", "—"),
            "size": r.get("size", "—"),
            "overall_score": r.get("overall_score"),
            "dimension_scores": r["dimension_scores"],
            "total_questions": r["total_questions"],
            "complete": r["complete"],
        })

    md_lines = []
    md_lines.append("# Cross-Domain Generalization Experiment (S3*)")
    md_lines.append("")
    md_lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    md_lines.append(f"**Judge:** {JUDGE_MODEL}")
    md_lines.append("**Dataset:** ControlSci core (500 questions, 125 per dimension)")
    md_lines.append("")
    md_lines.append("## Key Question")
    md_lines.append("")
    md_lines.append("> Do small **general-purpose** models (2B-4B params, not trained on control science)")
    md_lines.append("> perform significantly worse than a **domain-tuned** baseline (qwen3.5:9b)?")
    md_lines.append("> If so, this validates that the benchmark measures *domain knowledge* rather than generic reasoning.")
    md_lines.append("")
    md_lines.append("## Leaderboard")
    md_lines.append("")
    md_lines.append("| Rank | Model | Family | Size | Overall | A | B | C | D | Complete |")
    md_lines.append("|------|-------|--------|------|---------|---|---|---|---|----------|")

    for rank, r in enumerate(rows, 1):
        ov = r.get("overall_score")
        overall = f"{ov:.4f}" if isinstance(ov, (int, float)) else "-"
        dim_cells = []
        for d in all_dims:
            score = r.get("dimension_scores", {}).get(d, None)
            if isinstance(score, (int, float)):
                dim_cells.append(f"{score:.4f}")
            else:
                dim_cells.append("-")
        complete = "✅" if r.get("complete") else "❌"
        family = r.get("family", "—")
        size = r.get("size", "—")
        marker = " ⬅ baseline" if (r.get("model") or "").startswith("qwen3.5:9b") else ""
        md_lines.append(
            f"| {rank} | {r['model']}{marker} | {family} | {size} | {overall} | "
            + " | ".join(dim_cells)
            + f" | {complete} |"
        )

    md_lines.append("")

    if len(reports) >= 1:
        md_lines.append("## Gap Analysis vs Baseline")
        md_lines.append("")
        b_overall = baseline["overall_score"]
        md_lines.append("| Model | Δ Overall | Δ A | Δ B | Δ C | Δ D |")
        md_lines.append("|-------|-----------|---|---|---|---|")
        for r in sorted(reports, key=lambda x: x.get("overall_score") or 0, reverse=True):
            r_overall = r.get("overall_score")
            delta = r_overall - b_overall if isinstance(r_overall, (int, float)) else None
            delta_str = f"{delta:+.4f}" if isinstance(delta, (int, float)) else "-"
            dim_deltas = []
            for d in all_dims:
                r_dim = r.get("dimension_scores", {}).get(d, None)
                b_dim = baseline.get("dimension_scores", {}).get(d, None)
                if isinstance(r_dim, (int, float)) and isinstance(b_dim, (int, float)):
                    dim_deltas.append(f"{(r_dim - b_dim):+.4f}")
                else:
                    dim_deltas.append("-")
            md_lines.append(
                f"| {r['model']} | {delta_str} | " + " | ".join(dim_deltas) + " |"
            )
        md_lines.append("")

    md_lines.append("## Interpretation")
    md_lines.append("")
    md_lines.append("- **Large negative Δ** → general small models struggle with domain knowledge → the benchmark has strong construct validity")
    md_lines.append("- **Small Δ** → generic reasoning ability compensates for lack of domain training → the benchmark may conflate general intelligence with domain expertise")
    md_lines.append("- **Mixed Δ across dimensions** → some dimensions (e.g., A=concept recall) are more domain-specific than others (e.g., C=sensitivity analysis)")
    md_lines.append("")
    md_lines.append(f"*Cross-domain experiment completed with {len(reports)} models.*")

    return "\n".join(md_lines)


def _build_cross_domain_parser() -> argparse.ArgumentParser:
    """Build argument parser for cross-domain generalization experiment."""
    parser = argparse.ArgumentParser(
        description="S3* Cross-Domain Generalization Experiment — run 3 small Ollama models on Sci-Align core"
    )
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Benchmark JSON path")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Reports directory")
    parser.add_argument("--judge-model", default=JUDGE_MODEL)
    parser.add_argument("--judge-base-url", default=JUDGE_BASE_URL)
    parser.add_argument("--judge-api-key", default=DEEPSEEK_KEY, help="DeepSeek API key")
    parser.add_argument("--models", default="", help="Comma-separated model names to run (default: all 3)")
    parser.add_argument("--limit", type=int, default=0, help="Limit questions per model (0=all)")
    parser.add_argument("--resume", action="store_true", help="Resume from existing reports")
    parser.add_argument("--retry-failed", action="store_true", help="Retry previously failed records")
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--max-consecutive-failures", type=int, default=5)
    parser.add_argument("--compare-only", action="store_true", help="Only generate comparison from existing reports")
    parser.add_argument("--skip-precheck", action="store_true", help="Skip llama.cpp health check")
    parser.add_argument("--background", action="store_true", help="Launch models as detached background processes")
    return parser


def _select_cross_domain_models(args) -> list[dict] | None:
    """Select models to run based on CLI args. Returns None if precheck fails."""
    if not args.skip_precheck:
        if not precheck_ollama():
            print("\n[ABORT] 预检未通过。使用 --skip-precheck 跳过检查。", flush=True)
            return None
    selected = [m.strip() for m in args.models.split(",") if m.strip()] if args.models else None
    models_to_run = [
        m for m in CROSS_DOMAIN_MODELS
        if selected is None or m["name"] in selected
    ]
    if not models_to_run:
        print("No models selected. Available: qwen3.5:2b, gemma3:4b, llama3.2:3b")
        return None
    return models_to_run


def _print_run_summary(results, background: bool):
    """Print run summary table."""
    print(f"\n{'='*60}")
    print("  Run Summary")
    print(f"{'='*60}")
    for r in results:
        if background:
            status_str = f"BG PID={r.get('pid', '?')}"
        else:
            status_str = "OK" if r["exit_code"] == 0 else f"FAIL({r['exit_code']})"
        mins = int(r["elapsed_sec"] // 60)
        secs = int(r["elapsed_sec"] % 60)
        print(f"  [{status_str}] {r['model']} ({r['family']} {r['size']}) — {mins}m{secs}s")
    print()


def _run_cross_domain_models(args) -> list[dict] | None:
    """Run all selected models; return results or None if background mode."""
    models_to_run = _select_cross_domain_models(args)
    if models_to_run is None:
        return None

    total_models = len(models_to_run)
    input_path = args.input
    if args.limit > 0:
        input_path = _build_stratified_input(args.input, args.limit, args.output_dir)

    print("\nCross-Domain Generalization Experiment (S3*)")
    print(f"  Models: {[m['name'] for m in models_to_run]}")
    print(f"  Judge:  {args.judge_model} @ {args.judge_base_url}")
    print(f"  Input:  {input_path}")
    print(f"  Output: {args.output_dir}/")
    print(f"  Resume: {args.resume}, Retry-failed: {args.retry_failed}")
    print(f"  Background: {args.background}")
    print()

    results = []
    for i, model_cfg in enumerate(models_to_run):
        print(f"[{i+1}/{total_models}] Running {model_cfg['name']}...", flush=True)
        r = run_single_model(
            model_cfg=model_cfg,
            judge_model=args.judge_model,
            judge_base_url=args.judge_base_url,
            input_path=input_path,
            output_dir=args.output_dir,
            limit=args.limit,
            resume=args.resume,
            retry_failed=args.retry_failed,
            retries=args.retries,
            max_cf=args.max_consecutive_failures,
            judge_api_key=args.judge_api_key,
            background=args.background,
        )
        results.append(r)

        if args.background and len(models_to_run) > 1:
            overlap_msg = "下一个模型将并行启动"
            print(f"  [{i+1}/{total_models}] {model_cfg['name']} 后台 PID={r.get('pid','?')} — {overlap_msg}", flush=True)

    _print_run_summary(results, args.background)

    if args.background:
        print("[INFO] 模型已在后台运行。等待完成后运行 --compare-only 生成对比报告。")
        print(f"[INFO] 状态文件: {STATUS_FILE}")
        return None

    return results


def _save_comparison(args, reports: list[dict]):
    """Generate and save comparison report (MD + JSON)."""
    reports = sorted(reports, key=lambda r: r["overall_score"], reverse=True)

    comparison_md = build_comparison(reports, BASELINE)
    out_md = Path(args.output_dir) / "cross_domain_comparison.md"
    out_md.write_text(comparison_md, encoding="utf-8")
    print(f"\nComparison saved: {out_md}")

    comparison_json = {
        "experiment": "S3* Cross-Domain Generalization",
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "judge_model": args.judge_model,
        "baseline": BASELINE,
        "models": reports,
    }
    out_json = Path(args.output_dir) / "cross_domain_comparison.json"
    write_json_atomic(out_json, comparison_json)
    print(f"Comparison JSON saved: {out_json}")

    write_status({
        "status": "completed",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "models_count": len(reports),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    })

    print(comparison_md)


def main():
    parser = _build_cross_domain_parser()
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    if not args.compare_only:
        result = _run_cross_domain_models(args)
        if result is None:  # background or abort
            return

    reports = load_model_results(args.output_dir)
    if not reports:
        print("No cross_domain_*_report.json found. Run without --compare-only first.")
        return

    _save_comparison(args, reports)


if __name__ == "__main__":
    main()
