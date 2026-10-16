"""Executor handlers — multi-judge, multi-format, and medical_rag intent handlers.

Extracted from executor.py to reduce file size and separate specialized
intent handlers from the core Executor class.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Tuple

from mineru_agent.utils import (
    ROOT,
    _python_cmd,
    _run_subprocess,
    _safe_cmd,
)
from mineru_core.settings import PROVIDERS


logger = logging.getLogger("mineru_agent.executor")  # 保持与原模块一致


def _parse_mrag_params(params: dict) -> dict:
    """Extract and normalize medical_rag parameters from a params dict."""
    def _to_bool(v, default=False):
        if isinstance(v, str):
            return v.lower() in ("true", "1", "yes")
        return bool(v) if v is not None else default

    profile = str(params.get("profile", "smoke") or "smoke").lower()
    if profile not in ("smoke", "report"):
        profile = "smoke"

    p = {
        "profile": profile,
        "input_dir": params.get("input_dir", str(ROOT / "data" / "sources_medical")),
        "output_dir": params.get("output_dir", str(ROOT / "benchmark" / "eval" / "results" / "medical")),
        "skip_train": _to_bool(params.get("skip_train", False)),
        "skip_hybrid": _to_bool(params.get("skip_hybrid", False)),
        "skip_viz": _to_bool(params.get("skip_viz", False)),
        "chunks_dir": str(params.get("chunks_dir", "") or ""),
        "index_dir": str(params.get("index_dir", "") or ""),
        "qa_output": str(params.get("qa_output", "") or ""),
        "viz_output": str(params.get("viz_output", "") or ""),
        "max_chunks": int(params.get("max_chunks", 0) or 0),
        "n_queries": int(params.get("n_queries", 0) or 0),
        "top_k_judge": int(params.get("top_k_judge", 0) or 0),
        "local_models": str(params.get("local_models", params.get("ollama_models", "")) or ""),
        "qa_limit_queries": int(params.get("qa_limit_queries", 0) or 0),
        "skip_query_gen": params.get("skip_query_gen", None),
        "skip_api_judge": _to_bool(params.get("skip_api_judge", False)),
        "skip_local_judge": _to_bool(params.get("skip_local_judge", params.get("skip_ollama_judge", False))),
        "qa_allow_empty": _to_bool(params.get("qa_allow_empty", False)),
    }
    # Re-normalize skip_query_gen: could be None, True, False, or string
    if isinstance(p["skip_query_gen"], str):
        p["skip_query_gen"] = p["skip_query_gen"].lower() in ("true", "1", "yes")
    return p


def _build_mrag_cmd(handler_path: Path, p: dict) -> list:
    """Build medical_rag subprocess command from parsed params."""
    cmd = _python_cmd([
        str(handler_path),
        "--profile", p["profile"],
        "--input", p["input_dir"],
        "--output", p["output_dir"],
    ])
    optional_str = [
        ("chunks_dir", "--chunks-dir"),
        ("index_dir", "--index-dir"),
        ("qa_output", "--qa-output"),
        ("viz_output", "--viz-output"),
        ("local_models", "--local-models"),
    ]
    for key, flag in optional_str:
        if p[key]:
            cmd.extend([flag, str(p[key])])
    optional_int = [
        ("max_chunks", "--max-chunks"),
        ("n_queries", "--n-queries"),
        ("top_k_judge", "--top-k-judge"),
        ("qa_limit_queries", "--qa-limit-queries"),
    ]
    for key, flag in optional_int:
        if p[key] > 0:
            cmd.extend([flag, str(p[key])])
    if p["skip_query_gen"] is True:
        cmd.append("--skip-query-gen")
    elif p["skip_query_gen"] is False:
        cmd.append("--no-skip-query-gen")
    optional_bool = [
        "skip_api_judge", "skip_local_judge", "qa_allow_empty",
        "skip_train", "skip_hybrid", "skip_viz",
    ]
    for key in optional_bool:
        if p[key]:
            cmd.append(f"--{key.replace('_', '-')}")
    return cmd


def exec_multi_judge_compare(params: dict) -> Tuple[int, str, str]:
    """Execute multi-judge comparison intent.

    Args:
        params: Step parameters from the plan.

    Returns:
        Tuple of (return_code, stdout, stderr).
    """
    models = params.get("models", ["deepseek-v4-flash", "mimo-v2-flash", "qwen3.5:9b"])
    api_judge = params.get("api_judge", "deepseek-v4-flash")
    local_judge = params.get("local_judge", "qwen3.5:9b")
    dataset = params.get("dataset", "benchmark/dataset/core.json")
    output = params.get("output", str(ROOT / "eval" / "results" / "judge_comparison.json"))
    max_questions = params.get("max_questions", params.get("limit", 0))

    output_path = Path(output)
    output_dir = output_path.parent

    judge_runs = [
        {"label": "api",   "judge_model": api_judge,   "judge_base_url": PROVIDERS["deepseek"].base_url},
        {"label": "local", "judge_model": local_judge, "judge_base_url": f"http://localhost:{os.environ.get('LLAMA_CHAT_PORT', '8080')}/v1",
         "judge_api_key": "local"},
    ]

    all_outputs = []
    all_errors = []
    global_rc = 0
    per_judge_rc = {}

    for run_cfg in judge_runs:
        run_rc = 0
        for model in models:
            output_file = output_dir / f"judge_{run_cfg['label']}_{model}.json"
            cmd = _python_cmd([
                "-m", "mineru_eval.evaluate",
                "--mode", "model",
                "--input", str(dataset),
                "--output", str(output_file),
                "--target-model", model,
                "--judge-model", run_cfg["judge_model"],
                "--judge-base-url", run_cfg["judge_base_url"],
                "--resume",
            ])
            if run_cfg.get("judge_api_key"):
                cmd.extend(["--judge-api-key", run_cfg["judge_api_key"]])
            if max_questions:
                cmd.extend(["--limit", str(max_questions)])

            logger.info("  → multi_judge_compare [%s/%s]: judge=%s",
                         run_cfg["label"], model, run_cfg["judge_model"])
            logger.info("  → cmd: %s", _safe_cmd(cmd))
            rc, stdout, stderr = _run_subprocess(cmd, timeout=14400)
            all_outputs.append(f"[{run_cfg['label']}/{model}] rc={rc}\n{stdout[-200:]}")
            if rc != 0:
                all_errors.append(f"[{run_cfg['label']}/{model}] {stderr[-200:]}")
                run_rc = 1
        per_judge_rc[run_cfg["label"]] = run_rc
        if run_rc != 0:
            global_rc = 1

    combined_stdout = "\n---\n".join(all_outputs)
    combined_stderr = "\n---\n".join(all_errors) if all_errors else ""

    logger.info("  → multi_judge_compare: 双路×%d模型评测完成, API rc=%s, Local rc=%s, output=%s",
                 len(models), per_judge_rc.get("api", "?"), per_judge_rc.get("local", "?"), output)
    return global_rc, combined_stdout, combined_stderr


def exec_multi_format_parse(params: dict) -> Tuple[int, str, str]:
    """Execute multi-format parse intent.

    Args:
        params: Step parameters from the plan.

    Returns:
        Tuple of (return_code, stdout, stderr).
    """
    files = params.get("files", [])
    input_dir = params.get("input_dir", "benchmark/agent/test_materials/")
    output_dir = params.get("output_dir", "benchmark/agent/test_materials/md/")
    lang = params.get("lang", "ch")
    parser_backend = params.get("parser_backend", params.get("backend", "local"))
    data_class = params.get("data_class", "private_enterprise")
    allow_cloud_upload = bool(params.get("allow_cloud_upload", False))

    script_path = ROOT / "tools" / "mineru_to_md.py"
    abs_input_dir = ROOT / input_dir
    abs_output = ROOT / output_dir
    abs_output.mkdir(parents=True, exist_ok=True)

    if files:
        resolved_files = []
        for f in files:
            p = Path(f)
            if not p.is_absolute():
                p = abs_input_dir / f
            resolved_files.append(str(p))
        cmd = _python_cmd([
            str(script_path),
            *resolved_files,
            "--output-dir", str(abs_output),
            "--lang", str(lang),
            "--backend", str(parser_backend),
            "--data-class", str(data_class),
            "--skip-existing",
            "--stats",
        ])
        if allow_cloud_upload:
            cmd.append("--allow-cloud-upload")
        logger.info("  → multi_format_parse: %d files resolved to %s", len(files), input_dir)
    else:
        cmd = _python_cmd([
            str(script_path),
            "--input-dir", str(abs_input_dir),
            "--recursive",
            "--output-dir", str(abs_output),
            "--lang", str(lang),
            "--backend", str(parser_backend),
            "--data-class", str(data_class),
            "--skip-existing",
            "--stats",
        ])
        if allow_cloud_upload:
            cmd.append("--allow-cloud-upload")
        logger.info("  → multi_format_parse: dir=%s via mineru_to_md.py", input_dir)

    logger.info("  → cmd: %s", _safe_cmd(cmd))
    return _run_subprocess(cmd, timeout=3600)


def exec_medical_rag(params: dict) -> Tuple[int, str, str]:
    """Execute medical_rag intent.

    Args:
        params: Step parameters from the plan.

    Returns:
        Tuple of (return_code, stdout, stderr).
    """
    p = _parse_mrag_params(params)
    handler_path = ROOT / "benchmark" / "agent" / "medical_rag_handler.py"
    if not handler_path.exists():
        status_msg = json.dumps({
            "intent": "medical_rag",
            "status": "skipped",
            "reason": f"medical_rag_handler.py 未创建 ({handler_path})",
            "hint": "Create benchmark/agent/medical_rag_handler.py to enable this intent",
        }, ensure_ascii=False)
        logger.warning("medical_rag_handler.py 不存在: %s", handler_path)
        return 0, status_msg, ""

    cmd = _build_mrag_cmd(handler_path, p)
    logger.info("  → medical_rag: profile=%s, input=%s, output=%s, skip_train=%s, skip_hybrid=%s, skip_viz=%s",
                 p["profile"], p["input_dir"], p["output_dir"], p["skip_train"], p["skip_hybrid"], p["skip_viz"])
    logger.info("  → cmd: %s", _safe_cmd(cmd))
    return _run_subprocess(cmd, timeout=54000)
