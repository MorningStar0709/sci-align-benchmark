"""Dual-Judge evaluation orchestrator.

Track A (API-only, no GPU): Re-score existing models with dual judge.
    Loads cached answers from run1 JSON → scores via DeepSeek + MiMo + Qwen arbiter.

Track B (GPU + API): Phase 2 new models with dual judge.
    run1: GPU inference + dual judge scoring.
    run2/3: Reuse run1 answers cache + dual judge scoring only.

Run both tracks: .venv/bin/python scripts/run_dual_judge_eval.py
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# ── Must be set BEFORE any ML imports ──
os.environ.setdefault("BNB_CUDA_VERSION", "130")

import mineru_core  # noqa: F401 — trigger _load_dotenv() to load .env API keys
from mineru_core.settings import PROVIDERS
from mineru_eval.client_factory import create_client, resolve_api_key
from mineru_eval._shared.scoring_base import _dual_judge_score, _score_via_openai
from mineru_eval.control.evaluate import score_model_answer
from mineru_eval.control.eval_adapter import SCORER_VERSION
from mineru_eval.utils import load_json, write_json_atomic

ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = ROOT / "benchmark" / "eval" / "rebuilt_quick_judge"
FIXED_Q = EVAL_DIR / "fixed_40.json"
PYTHON = Path(sys.executable)

PARALLEL = 4
RETRIES = 3
RUNS_PER_MODEL = 3

# ── Track A: API-only re-scoring (existing models, no GPU) ──
TRACK_A = [
    {
        "name": "mgpo_250",
        "adapter": str(ROOT / "models/probe_matrix/control_sci_rebuilt_250/mgpo/final"),
        "answers_cache": EVAL_DIR / "mgpo_250_40_run1.json",
    },
    {
        "name": "ftd_augment_mgpo_250",
        "adapter": str(ROOT / "models/probe_matrix/control_sci_rebuilt_250/ftd_augment_mgpo/final"),
        "answers_cache": EVAL_DIR / "ftd_augment_mgpo_250_40_run1.json",
    },
    {
        "name": "raw_40",
        "adapter": None,  # raw base model
        "answers_cache": EVAL_DIR / "raw_40_run1.json",
    },
]

# ── Track B: Phase 2 new models (GPU + API) ──
TRACK_B = [
    {
        "name": "mgpo_ar_250",
        "adapter": str(ROOT / "models/probe_matrix/control_sci_rebuilt_250/mgpo_ar/final"),
    },
    {
        "name": "mgpo_eg_250",
        "adapter": str(ROOT / "models/probe_matrix/control_sci_rebuilt_250/mgpo_eg/final"),
    },
    {
        "name": "mgpo_fa_250",
        "adapter": str(ROOT / "models/probe_matrix/control_sci_rebuilt_250/mgpo_fa/final"),
    },
]

try:
    from patrol import kill_tree
except ImportError:  # pytest 以 scripts.* 包式导入时
    from scripts.patrol import kill_tree



def _validate_track_paths(track: str = "both"):
    """Pre-flight check: ensure required paths exist for the selected track."""
    missing = []
    if not FIXED_Q.is_file():
        missing.append(f"FIXED_Q: {FIXED_Q}")
    if track in ("a", "both"):
        for cfg in TRACK_A:
            p = cfg.get("answers_cache")
            if p and not Path(p).is_file():
                missing.append(f"TRACK_A[{cfg['name']}].answers_cache: {p}")
    if track in ("b", "both"):
        for cfg in TRACK_B:
            adapter = cfg.get("adapter")
            if adapter and not Path(adapter).is_dir():
                missing.append(f"TRACK_B[{cfg['name']}].adapter: {adapter}")
    if not PYTHON.is_file():
        missing.append(f"PYTHON: {PYTHON}")
    if missing:
        print("\n[PRE-FLIGHT] Missing paths — abort before starting:", flush=True)
        for m in missing:
            print(f"  - {m}", flush=True)
        raise SystemExit(1)
    print("[PRE-FLIGHT] All paths validated.", flush=True)


# ============================================================================
# Track A: Standalone API-only dual-judge re-scoring
# ============================================================================

def _score_one(
    q: dict, model_answer: str, score_fn, judge_client, judge_model: str,
) -> dict:
    """Score a single question — thread-safe, no GPU. Retries transient API failures."""
    question_text = q.get("question", "")
    expected = q.get("answer", "")
    dimension = q.get("dimension", "?")

    result = None
    for attempt in range(RETRIES + 1):
        try:
            result = score_fn(
                question_text, expected, model_answer, dimension,
                judge_client, judge_model,
                disable_thinking=True,  # DeepSeek needs this
            )
            break
        except Exception as e:
            if attempt < RETRIES:
                time.sleep(2 ** attempt)  # exponential backoff: 1s, 2s, 4s
                continue
            result = {"score": 0.0, "reason": f"scoring_error (retries exhausted): {e}", "issues": ["judge failure"]}

    issues = result.get("issues", []) or []
    if not model_answer and "target failure" not in issues:
        issues.append("target failure")

    # Include dual judge metadata fields
    dj_fields = {}
    for k in ("judge1_score", "judge2_score", "avg_score", "score_diff",
              "disputed", "arbitrated"):
        if k in result:
            dj_fields[k] = result[k]

    return {
        "id": q.get("id", "?"),
        "dimension": dimension,
        "difficulty_level": q.get("difficulty_level", "?"),
        "score": result.get("score", 0.0),
        "reason": result.get("reason", ""),
        "issues": issues,
        "model_answer": model_answer,
        "expected_answer": expected,
        "scoring_mode": "llm_judge",
        "scorer_version": SCORER_VERSION,
        **dj_fields,
        **(
            {"rubric_details": result["rubric_details"]}
            if "rubric_details" in result else {}
        ),
    }


def _create_judge_clients():
    """Create DeepSeek + MiMo + Qwen arbiter clients (API-only, no GPU)."""
    # DeepSeek judge
    ds_base = PROVIDERS["deepseek"].base_url
    ds_key = resolve_api_key(ds_base, explicit_key=None)
    dj1_client, dj1_model, _ = create_client(ds_key, ds_base, "deepseek-v4-flash")

    # MiMo judge2
    mimo_base = PROVIDERS["mimo"].base_url
    mimo_key = resolve_api_key(mimo_base, provider="mimo")
    dj2_client, dj2_model, _ = create_client(mimo_key, mimo_base, "mimo-v2.5")

    # Qwen arbiter
    qwen_base = PROVIDERS["qwen"].base_url
    qwen_key = resolve_api_key(qwen_base, provider="qwen")
    arb_client, arb_model, _ = create_client(qwen_key, qwen_base, "qwen3.7-plus")

    print(f"  Judge1: {dj1_model} @ {ds_base}")
    print(f"  Judge2: {dj2_model} @ {mimo_base}")
    print(f"  Arbiter: {arb_model} @ {qwen_base}")

    return dj1_client, dj1_model, dj2_client, dj2_model, arb_client, arb_model


def _make_score_fn(dj1_client, dj1_model, dj2_client, dj2_model, arb_client, arb_model,
                   *, all_dimensions: bool = False):
    """Build dual judge dispatch function.

    When all_dimensions=True, all dimensions (A/B/C/D) use dual judge.
    When all_dimensions=False (default), only C/D use dual judge.
    """
    def dispatch(question, expected, model_answer, dimension, client, model,
                 *, disable_thinking=False):
        use_dual = (
            dimension in ("A", "B", "C", "D") if all_dimensions
            else dimension in ("C", "D")
        )
        if use_dual:
            return _dual_judge_score(
                question, expected, model_answer, dimension,
                client, model, dj2_client, dj2_model,
                arb_client=arb_client, arb_model=arb_model,
                arb_fn=_score_via_openai,
                disable_thinking=disable_thinking,
            )
        return score_model_answer(
            question, expected, model_answer, dimension, client, model,
            disable_thinking=disable_thinking,
        )
    return dispatch


def run_track_a(*, all_dimensions_dual: bool = False):
    """API-only dual-judge re-scoring for existing models. No GPU needed."""
    print(f"\n{'='*60}")
    print("  Track A: API-only Dual-Judge Re-scoring")
    print(f"  Models: {[m['name'] for m in TRACK_A]}")
    print(f"  Runs/model: {RUNS_PER_MODEL}, Parallel: {PARALLEL}")
    print(f"{'='*60}")

    # Create judge clients
    dj1_client, dj1_model, dj2_client, dj2_model, arb_client, arb_model = _create_judge_clients()
    score_fn = _make_score_fn(dj1_client, dj1_model, dj2_client, dj2_model, arb_client, arb_model,
                              all_dimensions=all_dimensions_dual)

    # Load fixed questions
    questions = load_json(FIXED_Q)
    print(f"  Loaded {len(questions)} questions from {FIXED_Q.name}")

    # ETA: estimate total API calls
    total_runs = len(TRACK_A) * RUNS_PER_MODEL
    total_q_per_run = len(questions)  # approximate
    est_calls = total_runs * total_q_per_run * 2  # 2 judges per question
    print(f"  Estimated ~{est_calls} API calls ({total_runs} runs x ~{total_q_per_run} q x 2 judges)")

    results = []
    t0 = time.time()
    completed_runs = 0

    try:
        for model_cfg in TRACK_A:
            name = model_cfg["name"]
            cache_path = model_cfg["answers_cache"]

            # Load cached answers
            cache_data = load_json(cache_path)
            if isinstance(cache_data, dict):
                cache_records = cache_data.get("records", [])
            else:
                cache_records = cache_data
            cached = {r["id"]: r["model_answer"] for r in cache_records
                      if isinstance(r, dict) and "id" in r and r.get("model_answer")}
            print(f"\n  [{name}] Loaded {len(cached)} cached answers from {cache_path.name}")

            # Build question-answer pairs
            qa_pairs = [(q, cached[q["id"]]) for q in questions if q["id"] in cached]
            missing = len(questions) - len(qa_pairs)
            if missing:
                print(f"  [{name}] WARNING: {missing} questions missing from cache!")

            for run_id in range(1, RUNS_PER_MODEL + 1):
                output_path = EVAL_DIR / f"{name}_dj_run{run_id}.json"
                if output_path.is_file():
                    print(f"  [SKIP] {name} run{run_id}: already exists")
                    results.append({"model": name, "run": run_id, "status": "skipped"})
                    completed_runs += 1
                    continue

                print(f"\n  [{name}] run {run_id}/{RUNS_PER_MODEL} — scoring {len(qa_pairs)} questions...")
                run_t0 = time.time()
                scored_records = []
                failures = 0

                with ThreadPoolExecutor(max_workers=PARALLEL) as pool:
                    futures = {
                        pool.submit(_score_one, q, ans, score_fn, dj1_client, dj1_model): q
                        for q, ans in qa_pairs
                    }
                    for fut in as_completed(futures):
                        try:
                            record = fut.result()
                            scored_records.append(record)
                            if record.get("issues"):
                                failures += 1
                        except Exception as e:
                            q = futures[fut]
                            scored_records.append({
                                "id": q["id"], "score": 0.0,
                                "reason": f"exception: {e}", "issues": ["judge failure"],
                            })
                            failures += 1

                run_sec = time.time() - run_t0
                avg_score = sum(r["score"] for r in scored_records) / max(len(scored_records), 1)
                rate = len(scored_records) / (run_sec / 60) if run_sec > 0 else 0
                print(f"  [DONE] {name} run{run_id}: avg={avg_score:.2f}, "
                      f"{len(scored_records)}/{len(qa_pairs)} scored in {run_sec:.0f}s ({rate:.1f} qpm), "
                      f"failures={failures}")

                # Build report
                report = {
                    "target_model": name,
                    "judge_model": dj1_model,
                    "judge2_model": dj2_model,
                    "scoring_mode": "dual_judge",
                    "scorer_version": SCORER_VERSION,
                    "complete": True,
                    "completed": len(scored_records),
                    "total": len(questions),
                    "avg_score": round(avg_score, 4),
                    "pass_rate": round(sum(1 for r in scored_records if r["score"] >= 0.5) / max(len(scored_records), 1), 4),
                    "records": sorted(scored_records, key=lambda r: r.get("id", "")),
                }

                # Dual judge stats
                dual_recs = [r for r in scored_records if "judge1_score" in r]
                if dual_recs:
                    disputed = sum(1 for r in dual_recs if r.get("disputed"))
                    arbitrated = sum(1 for r in dual_recs if r.get("arbitrated"))
                    report["dual_judge_stats"] = {
                        "total_dual_judge": len(dual_recs),
                        "disputed": disputed,
                        "arbitrated": arbitrated,
                        "dispute_rate": round(disputed / len(dual_recs), 4),
                        "mean_score_diff": round(
                            sum(abs(r.get("score_diff", 0)) for r in dual_recs) / len(dual_recs), 4
                        ),
                    }

                report["run_metrics"] = {
                    "parallel_workers": PARALLEL,
                    "judge_phase_sec": round(run_sec, 1),
                    "rate_q_per_min": round(rate, 4),
                }

                # Atomic write via shared utility
                write_json_atomic(output_path, report)
                print(f"  [SAVED] {output_path.name}")

                results.append({
                    "model": name, "run": run_id, "status": "ok",
                    "avg_score": round(avg_score, 4),
                })
                completed_runs += 1

                # ETA update
                elapsed = time.time() - t0
                if completed_runs > 0:
                    avg_per_run = elapsed / completed_runs
                    remaining_runs = total_runs - completed_runs
                    eta_min = (avg_per_run * remaining_runs) / 60
                    print(f"  [ETA] {eta_min:.1f}min remaining ({completed_runs}/{total_runs} runs done)",
                          flush=True)

    except KeyboardInterrupt:
        print(f"\n  [INTERRUPT] Track A interrupted after {completed_runs}/{total_runs} runs.",
              flush=True)
        print(f"  Completed results preserved: {len(results)} entries.", flush=True)
        raise  # re-raise so main() can exit instead of falling through to Track B

    sec = time.time() - t0
    print(f"\n  Track A complete in {sec/60:.1f}min")
    return results


# ============================================================================
# Track B: Phase 2 GPU + dual judge via eval_cli subprocess
# ============================================================================

def _run_eval_cli(adapter_path: str, output: Path, answers_cache: Path | None = None,
                  *, all_dimensions_dual: bool = False) -> int:
    """Run eval_cli.py as subprocess with streaming stdout. Returns exit code."""
    cmd = [
        str(PYTHON), "-u", "-m", "mineru_eval.eval_cli",
        "--input", str(ROOT / "benchmark" / "dataset" / "core_v3.json"),
        "--output", str(output),
        "--fixed-questions", str(FIXED_Q),
        "--adapter-path", str(adapter_path),
        "--dual-judge",
        "--parallel", str(PARALLEL),
        "--retries", str(RETRIES),
        "--max-consecutive-failures", "5",
    ]
    if all_dimensions_dual:
        cmd += ["--all-dimensions-dual"]
    if answers_cache and answers_cache.is_file():
        cmd += ["--answers-cache", str(answers_cache)]

    log_path = output.with_suffix(".log")
    print(f"  [CMD] eval_cli → {output.name}", flush=True)

    try:
        proc = subprocess.Popen(
            cmd, cwd=str(ROOT),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace",
            bufsize=1,
        )
        with open(log_path, "w", encoding="utf-8") as log_f:
            for line in proc.stdout:
                sys.stdout.write(f"    │ {line}")
                sys.stdout.flush()
                log_f.write(line)
                log_f.flush()
        proc.wait(timeout=3600)
        return proc.returncode
    except subprocess.TimeoutExpired:
        try:
            leftovers = kill_tree(proc.pid)   # 树杀：SIGKILL 只打直接子进程会留孤儿占显存
            if leftovers:
                print(f"  [ERROR] 进程树未清干净，显存可能仍被占用: {leftovers}", flush=True)
        except OSError:
            pass  # process may have already exited
        print(f"  [TIMEOUT] {output.name}", flush=True)
        return -1
    except Exception as e:
        print(f"  [ERROR] {output.name}: {e}", flush=True)
        return -1


def run_track_b(*, all_dimensions_dual: bool = False):
    """Phase 2 GPU + dual judge via eval_cli subprocess."""
    print(f"\n{'='*60}")
    print("  Track B: Phase 2 GPU + Dual-Judge")
    print(f"  Models: {[m['name'] for m in TRACK_B]}")
    print(f"  Runs/model: {RUNS_PER_MODEL}")
    if all_dimensions_dual:
        print("  All dimensions: Dual Judge (A/B/C/D)")
    print(f"{'='*60}")

    results = []
    t0 = time.time()

    for model_cfg in TRACK_B:
        name = model_cfg["name"]
        adapter = model_cfg["adapter"]
        prev_output = None

        for run_id in range(1, RUNS_PER_MODEL + 1):
            output_path = EVAL_DIR / f"{name}_dj_run{run_id}.json"
            if output_path.is_file():
                print(f"  [SKIP] {name} run{run_id}: already exists")
                results.append({"model": name, "run": run_id, "status": "skipped"})
                if run_id == 1:
                    prev_output = output_path
                continue

            # run2/3: reuse run1 answers (skip GPU)
            cache = prev_output if run_id > 1 and prev_output and prev_output.is_file() else None
            mode = "cache" if cache else "gpu"
            print(f"\n  [{name}] run {run_id}/{RUNS_PER_MODEL} ({mode})")

            rc = _run_eval_cli(adapter, output_path, answers_cache=cache,
                               all_dimensions_dual=all_dimensions_dual)
            status = "ok" if rc == 0 else f"fail(rc={rc})"
            results.append({"model": name, "run": run_id, "status": status})
            print(f"  [{status.upper()}] {name} run{run_id}")

            if run_id == 1 and rc == 0:
                prev_output = output_path
            elif run_id == 1:
                print("  [WARN] run1 failed, run2/3 will need GPU inference")

    sec = time.time() - t0
    print(f"\n  Track B complete in {sec/60:.1f}min")
    return results


# ============================================================================
# Main orchestrator
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Dual-Judge evaluation orchestrator")
    parser.add_argument("--track", choices=["a", "b", "both"], default="both",
                        help="Which track to run: a (API-only), b (GPU+API), both (default)")
    parser.add_argument("--all-dimensions-dual", action="store_true",
                        help="Enable dual-judge for ALL dimensions (A/B/C/D), not just C/D.")
    args = parser.parse_args()

    _validate_track_paths(args.track)
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    track_a_results = []
    track_b_results = []

    if args.track in ("a", "both"):
        track_a_results = run_track_a(all_dimensions_dual=args.all_dimensions_dual)

    if args.track in ("b", "both"):
        track_b_results = run_track_b(all_dimensions_dual=args.all_dimensions_dual)

    total_sec = time.time() - t_start
    all_results = track_a_results + track_b_results

    if not all_results:
        print("\nNo results to report.")
        return 0

    # Summary
    ok = sum(1 for r in all_results if r.get("status") in ("ok", "skipped"))
    fail = sum(1 for r in all_results if "fail" in str(r.get("status", "")))

    print(f"\n{'='*60}")
    print(f"  DUAL-JUDGE EVALUATION COMPLETE — {total_sec/60:.1f}min total")
    if track_a_results:
        print(f"  Track A (API-only): {len(track_a_results)} runs")
    if track_b_results:
        print(f"  Track B (GPU+API):  {len(track_b_results)} runs")
    print(f"  OK: {ok}/{len(all_results)}  Failed: {fail}/{len(all_results)}")
    print(f"{'='*60}")
    print(f"\n  {'Model':<25} {'Run':>4}  {'Status':<10}  {'Avg Score'}")
    print(f"  {'-'*55}")
    for r in sorted(all_results, key=lambda x: (x["model"], x["run"])):
        avg = r.get("avg_score", "")
        avg_str = f"{avg:.4f}" if isinstance(avg, float) else str(avg)
        print(f"  {r['model']:<25} run{r['run']}  {r['status']:<10}  {avg_str}")

    # Save summary
    track_label = args.track.upper()
    summary_path = EVAL_DIR / f"dual_judge_summary_track{track_label}.json"
    summary = {
        "track": args.track,
        "total_runs": len(all_results),
        "ok": ok, "failed": fail,
        "elapsed_min": round(total_sec / 60, 1),
        "track_a": track_a_results,
        "track_b": track_b_results,
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n  Summary: {summary_path}")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
