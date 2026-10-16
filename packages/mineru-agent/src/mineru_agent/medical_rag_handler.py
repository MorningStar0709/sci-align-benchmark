"""Medical RAG 全流程编排器

编排任务 2→3→4→5→6 的端到端管线:
  1. 医疗章节本体切片 (medical_section_detector + chunk_corpus)
  2. QLoRA 微调 (prepare_medical_instructions + train_qlora)
  3. Hybrid 索引构建 (FAISS + BM25)
  4. RAG 检索 + 评分矩阵评估
  5. 知识库自评测
  6. 跨文献证据合成 + QA 格式化
  7. 可视化

Usage:
  python -m mineru_benchmark.agent.medical_rag_handler \
      --input data/sources_medical --output benchmark/eval/results/medical
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]  # restored Phase 5

TAIL_CHARS = 800


def _get_python_cmd():
    """一律用当前解释器。

    原实现是 conda run 包装 + 可用性探测，与 utils._python_cmd 是同一层的第二份拷贝；
    本仓环境由 uv 管理 .venv、无 conda，包装恒回落 ⇒ 连同 _MRAG_CONDA_WRAPPED 哨兵一并删除。
    """
    return [sys.executable, "-B"]


def _run_py(script_args):
    return _get_python_cmd() + script_args


def _step_env():
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["CONDA_NO_PLUGINS"] = "true"
    env["PYTHONPATH"] = str(ROOT) + (
        os.pathsep + env.get("PYTHONPATH", "") if env.get("PYTHONPATH") else ""
    )
    return env


def _run_step(name, cmd, cwd=None, timeout=7200, extra_env=None):
    t0 = time.time()
    print(f"\n{'=' * 60}")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 开始: {name}")
    print(f"  cmd: {' '.join(str(c) for c in cmd)}")

    env = _step_env()
    if extra_env:
        env.update(extra_env)

    try:
        result = subprocess.run(
            cmd,
            cwd=cwd or str(ROOT),
            capture_output=True,
            text=True,
            timeout=timeout,
            encoding="utf-8",
            errors="replace",
            env=env,
            shell=False,
        )
        elapsed = time.time() - t0
        status = "OK" if result.returncode == 0 else "FAIL"
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {status} ({elapsed:.0f}s, rc={result.returncode})")
        if result.stdout:
            print(result.stdout[-TAIL_CHARS:] if len(result.stdout) > TAIL_CHARS else result.stdout)
        if result.returncode != 0 and result.stderr:
            print(f"  stderr: {result.stderr[-TAIL_CHARS:]}")
        return result.returncode == 0, {
            "step": name,
            "success": result.returncode == 0,
            "elapsed_s": round(elapsed, 1),
            "returncode": result.returncode,
            "stdout_tail": result.stdout[-TAIL_CHARS:] if result.stdout else "",
            "stderr_tail": result.stderr[-TAIL_CHARS:] if result.stderr else "",
        }
    except subprocess.TimeoutExpired:
        elapsed = time.time() - t0
        print(f"[{datetime.now().strftime('%H:%M:%S')}] TIMEOUT (>{timeout}s, elapsed {elapsed:.0f}s)")
        return False, {"step": name, "success": False, "elapsed_s": round(elapsed, 1), "error": f"Timeout >{timeout}s"}
    except Exception as e:
        elapsed = time.time() - t0
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ERROR: {e}")
        return False, {"step": name, "success": False, "elapsed_s": round(elapsed, 1), "error": str(e)[:500]}



def _skip_step(step_name, reason, required=False):
    msg = {
        "step": step_name,
        "success": False,
        "skipped": True,
        "required": required,
        "reason": reason,
    }
    if required:
        msg["required_missing"] = True
    return msg


def _fail_step(step_name, reason):
    return {"step": step_name, "success": False, "error": reason}


# ============================================================
# Extracted phase functions (reducing main() complexity)
# ============================================================


def _parse_mrag_args():
    """Parse CLI arguments for the Medical RAG pipeline."""
    parser = argparse.ArgumentParser(description="Medical RAG 全流程编排器")
    parser.add_argument("--profile", choices=["smoke", "report"], default="smoke",
                        help="运行策略: smoke=高可用最小真实测试; report=报告级全流程")
    parser.add_argument("--input", default=str(ROOT / "data" / "sources_medical"),
                        help="医疗文献数据根目录")
    parser.add_argument("--output", default=str(ROOT / "benchmark" / "eval" / "results" / "medical"),
                        help="评估结果输出目录")
    parser.add_argument("--chunks-dir", default="")
    parser.add_argument("--index-dir", default="")
    parser.add_argument("--qa-output", default="")
    parser.add_argument("--viz-output", default=str(ROOT / "docs" / "assets" / "medical"))
    parser.add_argument("--skip-train", action="store_true", help="跳过 QLoRA 微调")
    parser.add_argument("--skip-hybrid", action="store_true", help="跳过 Hybrid 索引构建")
    parser.add_argument("--skip-viz", action="store_true", help="跳过可视化")
    parser.add_argument("--allow-skips", action="store_true",
                        help="允许必需脚本缺失时仍以 0 退出")
    parser.add_argument("--max-chunks", type=int, default=0)
    parser.add_argument("--n-queries", type=int, default=0)
    parser.add_argument("--top-k-judge", type=int, default=0)
    parser.add_argument("--local-models", default="")
    parser.add_argument("--qa-limit-queries", type=int, default=0)
    parser.add_argument("--qa-allow-empty", action="store_true")
    parser.add_argument("--skip-query-gen", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--skip-api-judge", action="store_true")
    parser.add_argument("--skip-local-judge", action="store_true")
    return parser.parse_args()


def _resolve_mrag_paths(args):
    """Resolve all directory paths from args and validate inputs."""
    input_dir = Path(args.input).resolve()
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    md_dir = input_dir / "md"
    chunks_dir = Path(args.chunks_dir).resolve() if args.chunks_dir else input_dir / "chunks"
    index_dir = Path(args.index_dir).resolve() if args.index_dir else input_dir / "index"
    qa_output = Path(args.qa_output).resolve() if args.qa_output else input_dir / "qa"
    viz_output = Path(args.viz_output).resolve()
    source_list = input_dir / "source_list.json"
    manifest_path = chunks_dir / "chunks_manifest.json"

    if not md_dir.exists():
        print(f"[FATAL] MD 目录不存在: {md_dir}")
        return None
    if not source_list.exists():
        print(f"[WARNING] source_list.json 不存在: {source_list}")

    return {
        "input_dir": input_dir, "output_dir": output_dir,
        "md_dir": md_dir, "chunks_dir": chunks_dir, "index_dir": index_dir,
        "qa_output": qa_output, "viz_output": viz_output,
        "source_list": source_list, "manifest_path": manifest_path,
    }


def _resolve_mrag_profile(args):
    """Resolve smoke/report profile defaults."""
    smoke = args.profile == "smoke"
    return {
        "smoke_mode": smoke,
        "skip_train": args.skip_train or smoke,
        "skip_viz": args.skip_viz or smoke,
        "max_chunks": args.max_chunks if args.max_chunks > 0 else (3 if smoke else 0),
        "n_queries": args.n_queries if args.n_queries > 0 else (1 if smoke else 25),
        "top_k_judge": args.top_k_judge if args.top_k_judge > 0 else (1 if smoke else 3),
        "qa_limit_queries": args.qa_limit_queries if args.qa_limit_queries > 0 else (1 if smoke else 0),
        "qa_allow_empty": args.qa_allow_empty or smoke,
        "skip_query_gen": args.skip_query_gen if args.skip_query_gen is not None else smoke,
        "skip_api_judge": args.skip_api_judge or smoke,
    }


def _mrag_step1_chunk(paths, summary):
    """Step 1: 医疗章节本体切片"""
    step1_code = (
        "import json, os, sys;"
        "cfg = json.loads(os.environ['_MRAG_STEP1_CFG']);"
        "from pathlib import Path;"
        "from pipeline.chunk_corpus import build_chunks;"
        "build_chunks(Path(cfg['md_dir']), medical_mode=True, chunks_dir=Path(cfg['chunks_dir']))"
    )
    step1_env = {
        "_MRAG_STEP1_CFG": json.dumps({
            "root": str(ROOT), "md_dir": str(paths["md_dir"]),
            "chunks_dir": str(paths["chunks_dir"]),
        }, ensure_ascii=False),
    }
    ok, result = _run_step("Step1-chunk", _run_py(["-c", step1_code]), extra_env=step1_env)
    summary["steps"].append(result)
    manifest_mtime = paths["manifest_path"].stat().st_mtime if ok and paths["manifest_path"].exists() else None
    return ok, manifest_mtime


def _mrag_step2_qlora(step1_ok, skip_train, paths, summary):
    """Step 2: QLoRA 微调"""
    if not step1_ok:
        summary["steps"].append(_skip_step("Step2-qlora", "Step1 切片失败，跳过微调"))
        return
    if skip_train:
        summary["steps"].append(_skip_step("Step2-qlora", "--skip-train"))
        return

    ok, result = _run_step("Step2a-instructions", _run_py([
        "-m", "mineru_medical.instructions",
        "--md-dir", str(paths["md_dir"]),
        "--source-list", str(paths["source_list"]),
        "--output-dir", str(ROOT / "finetune" / "data" / "medical"),
    ]))
    summary["steps"].append(result)
    if not ok:
        summary["steps"].append(_skip_step("Step2b-qlora", "Step2a 失败，跳过微调训练"))
        return

    train_script = ROOT / "finetune" / "scripts" / "train_qlora.py"
    train_config = ROOT / "finetune" / "config" / "qlora_config.yaml"
    if train_script.exists() and train_config.exists():
        ok, result = _run_step("Step2b-qlora", _run_py([str(train_script), "--config", str(train_config)]), timeout=14400)
        summary["steps"].append(result)
    else:
        missing = []
        if not train_script.exists():
            missing.append("train_qlora.py")
        if not train_config.exists():
            missing.append("qlora_config.yaml")
        summary["steps"].append(_skip_step("Step2b-qlora", f"缺失: {', '.join(missing)}", required=True))


def _mrag_step3_hybrid_index(step1_ok, step1_manifest_mtime, paths, args, profile, summary):
    """Step 3: Hybrid 索引构建"""
    if not step1_ok:
        summary["steps"].append(_skip_step("Step3-hybrid-index", "Step1 切片失败，跳过索引构建"))
        return
    if not paths["manifest_path"].exists():
        summary["steps"].append(_fail_step("Step3-hybrid-index", f"manifest 不存在: {paths['manifest_path']}"))
        return

    if step1_manifest_mtime is not None:
        current_mtime = paths["manifest_path"].stat().st_mtime
        if abs(current_mtime - step1_manifest_mtime) > 1.0:
            print(f"[WARNING] manifest 在 Step1 后被修改 (mtime diff={current_mtime - step1_manifest_mtime:.1f}s)")

    if args.skip_hybrid:
        summary["steps"].append(_skip_step("Step3-hybrid-index", "--skip-hybrid: 使用已有索引"))
        return

    index_cmd = [
        "-m", "mineru_medical.indexing", "--manifest", str(paths["manifest_path"]),
        "--output", str(paths["index_dir"]),
    ]
    if profile["max_chunks"] > 0:
        index_cmd.extend(["--max-chunks", str(profile["max_chunks"])])
    ok, result = _run_step("Step3-hybrid-index", _run_py(index_cmd), timeout=3600)
    summary["steps"].append(result)


def _mrag_step4_5_eval_kb(step1_ok, paths, args, profile, summary):
    """Step 4-5: 知识库自评测"""
    if not step1_ok:
        summary["steps"].append(_skip_step("Step4-5-eval-kb", "Step1 切片失败，跳过知识库自评测"))
        return

    kb_cmd = [
        "-m", "mineru_medical.kb_quality", "--profile", args.profile,
        "--manifest", str(paths["manifest_path"]),
        "--index-dir", str(paths["index_dir"]),
        "--output", str(paths["output_dir"] / "kb_quality_report.json"),
        "--n-queries", str(profile["n_queries"]),
        "--top-k-judge", str(profile["top_k_judge"]),
    ]
    if args.local_models:
        kb_cmd.extend(["--local-models", args.local_models])
    if profile["skip_query_gen"]:
        kb_cmd.append("--skip-query-gen")
    if profile["skip_api_judge"]:
        kb_cmd.append("--skip-api")
    if args.skip_local_judge:
        kb_cmd.append("--skip-local")
    ok, result = _run_step("Step4-5-eval-kb", _run_py(kb_cmd))
    summary["steps"].append(result)


def _mrag_step6_qa_format(step1_ok, paths, profile, summary):
    """Step 6: 跨文献证据合成 + QA 格式化"""
    if not step1_ok:
        summary["steps"].append(_skip_step("Step6-qa-format", "Step1 切片失败，跳过 QA 格式化"))
        return

    qa_cmd = [
        "-m", "mineru_medical.qa_formatter", "--index-dir", str(paths["index_dir"]),
        "--manifest", str(paths["manifest_path"]), "--output", str(paths["qa_output"]),
        "--kb-report", str(paths["output_dir"] / "kb_quality_report.json"),
        "--limit-queries", str(profile["qa_limit_queries"]),
    ]
    if profile["qa_allow_empty"]:
        qa_cmd.append("--allow-empty")
    ok, result = _run_step("Step6-qa-format", _run_py(qa_cmd))
    summary["steps"].append(result)


def _mrag_step7_viz(step1_ok, skip_viz, paths, summary):
    """Step 7: 可视化"""
    if skip_viz:
        summary["steps"].append(_skip_step("Step7-viz", "--skip-viz/profile=smoke"))
        return
    if not step1_ok:
        summary["steps"].append(_skip_step("Step7-viz", "Step1 切片失败，跳过可视化"))
        return

    ok, result = _run_step("Step7-viz", _run_py([
        "-m", "mineru_medical.visualization",
        "--report", str(paths["output_dir"] / "kb_quality_report.json"),
        "--output", str(paths["viz_output"]),
    ]))
    summary["steps"].append(result)


def _mrag_summarize(summary, args, paths):
    """Compute and persist pipeline summary."""
    total = len(summary["steps"])
    succeeded = sum(1 for s in summary["steps"] if s.get("success"))
    skipped = sum(1 for s in summary["steps"] if s.get("skipped"))
    failed = sum(1 for s in summary["steps"] if not s.get("success") and not s.get("skipped"))
    required_skipped = sum(1 for s in summary["steps"] if s.get("skipped") and s.get("required_missing"))
    summary["total_steps"] = total
    summary["succeeded"] = succeeded
    summary["skipped"] = skipped
    summary["required_skipped"] = required_skipped
    summary["failed"] = failed
    summary["finished_at"] = datetime.now().isoformat()
    summary["overall_success"] = failed == 0 and (required_skipped == 0 or args.allow_skips)
    summary["allow_skips"] = bool(args.allow_skips)

    report_path = paths["output_dir"] / "pipeline_summary.json"
    _write_atomic(report_path, summary)

    print(f"\n{'=' * 60}")
    print(f"Medical RAG Pipeline: {succeeded}/{total} 成功, {skipped} 跳过, {failed} 失败")
    if required_skipped:
        print(f"[WARNING] {required_skipped} 个必需步骤被跳过；使用 --allow-skips 才会以 0 退出")
    print(f"报告已写入: {report_path}")
    return 0 if summary["overall_success"] else 1


def main():
    """Medical RAG 全流程编排器 — orchestrate all 7 steps."""
    args = _parse_mrag_args()
    profile = _resolve_mrag_profile(args)
    paths = _resolve_mrag_paths(args)
    if paths is None:
        return 1

    summary = {
        "pipeline": "medical_rag",
        "profile": args.profile,
        "started_at": datetime.now().isoformat(),
        "steps": [],
    }

    step1_ok, step1_manifest_mtime = _mrag_step1_chunk(paths, summary)
    _mrag_step2_qlora(step1_ok, profile["skip_train"], paths, summary)
    _mrag_step3_hybrid_index(step1_ok, step1_manifest_mtime, paths, args, profile, summary)
    _mrag_step4_5_eval_kb(step1_ok, paths, args, profile, summary)
    _mrag_step6_qa_format(step1_ok, paths, profile, summary)
    _mrag_step7_viz(step1_ok, profile["skip_viz"], paths, summary)

    return _mrag_summarize(summary, args, paths)


def _write_atomic(path, payload):
    try:
        fd, tmp = tempfile.mkstemp(suffix=".json", prefix="pipeline_", dir=str(path.parent))
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        os.replace(tmp, str(path))
    except Exception as e:
        print(f"[ERROR] 写入报告失败: {e}")


if __name__ == "__main__":
    sys.exit(main())
