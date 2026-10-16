"""Executor — intent execution orchestration with 15+ handler functions.
"""

from __future__ import annotations

import json
import logging
import os
import textwrap
import time
from pathlib import Path
from typing import Optional, Tuple


from mineru_agent.intent_registry import IntentRegistry, PlanStep
from mineru_agent.utils import (
    _python_cmd,
    _run_subprocess,
    _safe_cmd,
)
from mineru_agent.executor_handlers import (
    exec_medical_rag,
    exec_multi_format_parse,
    exec_multi_judge_compare,
)
from mineru_core.log_schema import LogStep
from mineru_core.settings import PROVIDERS

ROOT = Path(__file__).resolve().parents[4]  # restored Phase 5

logger = logging.getLogger(__name__)


def _get_scheduler(local_mode: bool = False):
    """Lazy import resource_scheduler to avoid circular import with benchmark/agent/__init__.py."""
    from mineru_agent.resource_scheduler import get_global_scheduler
    return get_global_scheduler(local_mode=local_mode)


class Executor:
    """Layer 3 — intent execution orchestration with 15 dispatch handlers."""

    def __init__(self, scheduler=None, local_mode: bool = False,
                 registry: IntentRegistry = None):
        self._scheduler = scheduler or _get_scheduler(local_mode=local_mode)
        self._local_mode = local_mode
        self._registry = registry or IntentRegistry()

        self._dispatch_map = {
            "arxiv_search": self._exec_arxiv_search,
            "mineru_parse": self._exec_mineru_parse,
            "corpus_parse": self._exec_corpus_parse,
            "cross_modal_audit": self._exec_cross_modal_audit,
            "corpus_quality_report": self._exec_corpus_quality_report,
            "benchmark_build": self._exec_benchmark_build,
            "quality_arbitrate": self._exec_quality_arbitrate,
            "model_evaluate": self._exec_model_evaluate,
            "multi_judge_compare": self._exec_multi_judge_compare,
            "leaderboard_viz": self._exec_leaderboard_viz,
            "local_finetune": self._exec_local_finetune,
            "reproduce_all": self._exec_reproduce_all,
            "multi_format_parse": self._exec_multi_format_parse,
            "medical_rag": self._exec_medical_rag,
            "sciverse_search": self._exec_sciverse_search,
        }

    def execute(self, step: PlanStep, dry_run: bool = False,
                resolved=None) -> Tuple[LogStep, str, Optional[str]]:
        t0 = time.time()

        if dry_run:
            return self._dry_run_step(step, t0)

        intent_id = step.intent_id
        params = step.parameters

        if resolved is None:
            try:
                resolved = self._scheduler.resolve(intent_id, local_mode=self._local_mode)
            except Exception as e:
                error = f"资源解析失败: {e}"
                duration_ms = int((time.time() - t0) * 1000)
                log = LogStep(
                    step_id=step.rank,
                    step_name=f"{intent_id} ({step.intent_name})",
                    tool=step.tool,
                    input_summary=json.dumps(params, ensure_ascii=False)[:200],
                    output_summary="",
                    status="failed",
                    duration_ms=duration_ms,
                    error=error,
                )
                return log, "", error

        logger.info("  [ResourceScheduler] resolved: provider=%s, model=%s",
                     resolved.provider, resolved.model)

        handler = self._dispatch_map.get(intent_id)
        if handler is None:
            error = f"未知 intent_id: {intent_id}（dispatch_map 缺失）"
            duration_ms = int((time.time() - t0) * 1000)
            log = LogStep(
                step_id=step.rank,
                step_name=f"{intent_id} ({step.intent_name})",
                tool=step.tool,
                input_summary=json.dumps(params, ensure_ascii=False)[:200],
                output_summary="",
                status="failed",
                duration_ms=duration_ms,
                error=error,
            )
            return log, "", error

        try:
            rc, stdout, stderr = handler(params, resolved)
        except Exception as e:
            error = f"执行异常: {e}"
            duration_ms = int((time.time() - t0) * 1000)
            log = LogStep(
                step_id=step.rank,
                step_name=f"{intent_id} ({step.intent_name})",
                tool=step.tool,
                input_summary=json.dumps(params, ensure_ascii=False)[:200],
                output_summary="",
                status="failed",
                duration_ms=duration_ms,
                error=error,
            )
            return log, "", error

        duration_ms = int((time.time() - t0) * 1000)
        status = "success" if rc == 0 else "failed"
        output_summary = stdout[-500:] if stdout else (stderr[-500:] if stderr else "")
        error_msg = stderr[-300:] if rc != 0 and stderr else None

        log = LogStep(
            step_id=step.rank,
            step_name=f"{intent_id} ({step.intent_name})",
            tool=step.tool,
            input_summary=json.dumps(params, ensure_ascii=False)[:200],
            output_summary=output_summary,
            status=status,
            duration_ms=duration_ms,
            error=error_msg,
        )
        return log, stdout, error_msg

    def _dry_run_step(self, step: PlanStep, t0: float) -> Tuple[LogStep, str, Optional[str]]:
        logger.info("[DRY RUN] Step %d: %s (%s) → %s",
                     step.rank, step.intent_id, step.intent_name, step.tool)
        log = LogStep(
            step_id=step.rank,
            step_name=f"{step.intent_id} ({step.intent_name})",
            tool=step.tool,
            input_summary=json.dumps(step.parameters, ensure_ascii=False)[:200],
            output_summary="[dry run]",
            status="skipped",
            duration_ms=0,
        )
        return log, "[dry run]", None

    # ---------- intent handlers ----------

    def _exec_arxiv_search(self, params: dict, resolved=None) -> Tuple[int, str, str]:
        keywords = params.get("keywords", ["control barrier function", "differentiable MPC"])
        max_papers = params.get("max_papers", 5)
        output_dir = params.get("output_dir", "data/sources_flywheel/")

        if not keywords:
            logger.warning("  ⚠️ arxiv_search: keywords 为空，跳过")
            return 0, json.dumps({"status": "skipped", "reason": "empty keywords"}), ""

        script_path = ROOT / "data" / "sources_flywheel" / "search_and_download.py"
        if not script_path.exists():
            _run_subprocess(_python_cmd(["-c",
                f"import json,pathlib;"
                f"p=pathlib.Path('{output_dir}');"
                f"p.mkdir(parents=True,exist_ok=True);"
                f"(p/'arxiv_search_status.json').write_text("
                f"json.dumps({{'intent':'arxiv_search','status':'skipped',"
                f"'reason':'search_and_download.py not found'}},ensure_ascii=False),encoding='utf-8')"
            ]), timeout=30)
            return 0, json.dumps({"status": "skipped", "reason": "search_and_download.py not found"}), ""

        kw_str = ",".join(keywords[:3])
        cmd = _python_cmd([
            str(script_path),
            "--keywords", kw_str,
            "--max-papers", str(max_papers),
            "--output-dir", str(output_dir),
        ])
        logger.info("  → arxiv_search: %d papers via searching-arxiv-papers Skill, keywords=%s", max_papers, kw_str)
        logger.info("  → cmd: %s", _safe_cmd(cmd))
        return _run_subprocess(cmd, timeout=600)

    def _exec_mineru_parse(self, params: dict, resolved=None) -> Tuple[int, str, str]:
        source = params.get("source", "data/sources_flywheel/")
        output_dir = params.get("output_dir", "data/sources_flywheel/md/")
        lang = params.get("lang", "ch")
        parser_backend = params.get("parser_backend", params.get("backend", "local"))
        data_class = params.get("data_class", "private_enterprise")
        allow_cloud_upload = bool(params.get("allow_cloud_upload", False))

        script_path = ROOT / "tools" / "mineru_to_md.py"

        cmd = _python_cmd([
            str(script_path),
            "--input-dir", str(source),
            "--output-dir", str(output_dir),
            "--lang", str(lang),
            "--backend", str(parser_backend),
            "--data-class", str(data_class),
            "--skip-existing",
            "--stats",
        ])
        if allow_cloud_upload:
            cmd.append("--allow-cloud-upload")
        logger.info("  → mineru_parse: backend=%s, data_class=%s, src=%s, output=%s", parser_backend, data_class, source, output_dir)
        logger.info("  → cmd: %s", _safe_cmd(cmd))
        return _run_subprocess(cmd, timeout=3600)

    def _exec_corpus_parse(self, params: dict, resolved=None) -> Tuple[int, str, str]:
        source = params.get("source", "data/sources/")
        recursive = params.get("recursive", True)
        output_dir = params.get("output_dir", "data/processed/md/")
        max_files = params.get("max_files", 0)
        parser_backend = params.get("parser_backend", params.get("backend", "local"))
        data_class = params.get("data_class", "private_enterprise")
        allow_cloud_upload = bool(params.get("allow_cloud_upload", False))

        cmd = _python_cmd([
            str(ROOT / "tools" / "mineru_to_md.py"),
            "--input-dir", str(source),
            "--output-dir", str(output_dir),
            "--backend", str(parser_backend),
            "--data-class", str(data_class),
            "--skip-existing",
            "--stats",
        ])

        if recursive:
            cmd.append("--recursive")
        if max_files > 0:
            cmd.extend(["--max-files", str(max_files)])
        if allow_cloud_upload:
            cmd.append("--allow-cloud-upload")

        logger.info("  → corpus_parse: MinerU 文档解析, src=%s, output=%s, backend=%s, recursive=%s, max=%s",
                     source, output_dir, parser_backend, recursive, max_files or "all")
        logger.info("  → cmd: %s", _safe_cmd(cmd))
        return _run_subprocess(cmd, timeout=7200)

    def _exec_cross_modal_audit(self, params: dict, resolved=None) -> Tuple[int, str, str]:
        concurrency = params.get("concurrency", 5)
        max_images = params.get("max_images", 0)
        resume = params.get("resume", False)
        output_dir = params.get("output_dir", "")
        provider = params.get("provider", "")

        cmd = _python_cmd([
            "-m", "mineru_agent.visual_audit",
            "--workers", str(concurrency),
        ])
        if max_images > 0:
            cmd.extend(["--max-items", str(max_images)])
        if resume:
            cmd.append("--resume")
        if output_dir:
            cmd.extend(["--output-dir", str(output_dir)])
        if provider:
            cmd.extend(["--provider", provider])

        logger.info("  → cross_modal_audit: MiMo-V2.5 视觉审计, %d workers, max_items=%s, provider=%s",
                     concurrency, max_images or "all", provider or "mimo(default)")
        logger.info("  → cmd: %s", _safe_cmd(cmd))
        return _run_subprocess(cmd, timeout=7200)

    def _exec_corpus_quality_report(self, params: dict, resolved=None) -> Tuple[int, str, str]:
        source = params.get("source", "corpus/stats/")
        granularity = params.get("granularity", "full")
        output_format = params.get("output_format", "both")
        include_viz = params.get("include_visualizations", True)

        if resolved and resolved.provider == "local":
            return self._exec_corpus_quality_report_local(params)

        stats_path = Path(source)
        stats_files = list(stats_path.glob("*.json")) if stats_path.exists() else []

        template_args = json.dumps({
            "source": str(stats_path),
            "granularity": granularity,
            "format": output_format,
            "include_visualizations": include_viz,
            "stats_file_names": [f.name for f in stats_files[:20]],
        }, ensure_ascii=False)

        cmd = _python_cmd(["-c", textwrap.dedent("""\
            import json, sys
            from pathlib import Path
            t = json.loads(sys.argv[1])
            p = Path(t["source"])
            files = list(p.glob("*.json")) if p.exists() else []
            report = {
                "report_type": "corpus_quality",
                "granularity": t["granularity"],
                "format": t["format"],
                "source": str(p),
                "stats_files_found": len(files),
                "stats_files": [f.name for f in files[:20]],
                "visualizations": str(t["include_visualizations"]).lower(),
                "summary": "Corpus quality report generated via ControlSci Data Agent",
                "dimensions": [
                    "chunk_semantic_completeness", "formula_extraction_rate",
                    "image_match_rate", "cooccurrence_distribution",
                ],
                "status": "ok" if files else "warning: no stats files found",
            }
            print(json.dumps(report, ensure_ascii=False, indent=2))
        """), template_args])

        logger.info("  → corpus_quality_report: 质量报告生成, src=%s, granularity=%s, files=%d",
                     source, granularity, len(stats_files))
        return _run_subprocess(cmd, timeout=600)

    def _exec_corpus_quality_report_local(self, params: dict) -> Tuple[int, str, str]:
        source = params.get("source", "corpus/stats/")
        granularity = params.get("granularity", "full")
        output_format = params.get("output_format", "both")

        stats_path = Path(source)
        stats_files = list(stats_path.glob("*.json")) if stats_path.exists() else []

        stats_content = ""
        for f in stats_files[:5]:
            try:
                content = f.read_text(encoding="utf-8")[:2000]
                stats_content += f"\n--- {f.name} ---\n{content}\n"
            except Exception:
                pass

        prompt = json.dumps({
            "task": "corpus_quality_report",
            "source": str(stats_path),
            "granularity": granularity,
            "format": output_format,
            "stats_files_found": len(stats_files),
            "stats_files": [f.name for f in stats_files[:20]],
            "stats_preview": stats_content if stats_content else "(no stats data)",
        }, ensure_ascii=False)

        from mineru_tools.llm_client import LocalLLMClient

        try:
            client = LocalLLMClient(timeout=600)
            result = client.chat(
                messages=[
                    {"role": "system", "content": "你是语料质量分析专家。根据输入的统计数据生成 JSON 格式的语料质量报告。只输出 JSON，不要其他内容。"},
                    {"role": "user", "content": f"请分析以下语料统计数据并生成质量报告：\n{prompt}"},
                ],
                max_tokens=2048,
                response_format="json",
            )
            content = result.content
            logger.info("  → corpus_quality_report [local]: 模型=%s, stats_files=%d",
                         "qwen3.5-9b", len(stats_files))
            return 0, content, ""
        except Exception as e:
            error_msg = f"Local LLM quality report failed: {e}"
            logger.error("  → corpus_quality_report [local] FAILED: %s", e)
            return 1, "", error_msg

    def _exec_benchmark_build(self, params: dict, resolved=None) -> Tuple[int, str, str]:
        source = params.get("source", "corpus/")
        target_size = params.get("limit", params.get("target_size", 500))
        provider = params.get("provider", "")
        providers = params.get("providers", ["deepseek"])
        output = params.get("output", "benchmark/dataset/core.json")
        review_output = params.get("manual_review_output")
        seed = params.get("seed", 42)
        mock = bool(params.get("mock", False))
        dry_run = bool(params.get("dry_run", False))

        if provider:
            selected_provider = str(provider)
        elif isinstance(providers, list) and providers:
            selected_provider = str(providers[0])
        else:
            selected_provider = "deepseek"

        cmd = _python_cmd([
            "-m", "mineru_benchmark.pipeline.build_benchmark",
            "--corpus", str(source),
            "--provider", selected_provider,
            "--limit", str(target_size),
            "--output", str(output),
            "--seed", str(seed),
        ])
        if review_output:
            cmd.extend(["--manual-review-output", str(review_output)])
        if mock:
            cmd.append("--mock")
        if dry_run:
            cmd.append("--dry-run")

        logger.info("  → benchmark_build: 构建 %d 题评测数据集, provider=%s, output=%s, seed=%s",
                     target_size, selected_provider, output, seed)
        logger.info("  → cmd: %s", _safe_cmd(cmd))
        return _run_subprocess(cmd, timeout=14400)

    def _exec_quality_arbitrate(self, params: dict, resolved=None) -> Tuple[int, str, str]:
        consistency_filter = params.get("consistency_filter", True)
        output = params.get("output", "benchmark/dataset/arbitrated.json")

        output_path = Path(output)
        review_path = output_path.parent / "manual_review.json"

        cmd = _python_cmd([
            "-m", "mineru_benchmark.pipeline.build_benchmark",
            "--arbitrate",
            "--output", str(output_path),
            "--manual-review-output", str(review_path),
        ])

        filter_note = " + 一致性过滤" if consistency_filter else " (无过滤)"
        logger.info("  → quality_arbitrate: 双层 LLM 仲裁%s, output=%s, review=%s",
                     filter_note, output, str(review_path))
        logger.info("  → cmd: %s", _safe_cmd(cmd))
        return _run_subprocess(cmd, timeout=7200)

    def _exec_model_evaluate(self, params: dict, resolved=None) -> Tuple[int, str, str]:
        model = params.get("model", "deepseek-v4-flash")
        mode = params.get("mode", "model")
        dataset = params.get("dataset", str(ROOT / "benchmark" / "dataset" / "core_v3.json"))
        judge_provider = params.get("judge_provider", "auto")
        output_dir = params.get("output_dir", str(ROOT / "eval" / "results/"))
        max_questions = params.get("max_questions", 0)

        target_base_url, target_api_key_env = self._resolve_target(model)
        target_api_key = self._get_api_key(target_api_key_env) if target_api_key_env else ""

        if resolved and resolved.provider == "local":
            judge_model = "qwen3.5-9b"
            judge_base_url = f"http://localhost:{os.environ.get('LLAMA_CHAT_PORT', '8080')}/v1"
        else:
            judge_model, judge_base_url = {
                "auto":      (PROVIDERS["deepseek"].default_model, PROVIDERS["deepseek"].base_url),
                "deepseek":  (PROVIDERS["deepseek"].default_model, PROVIDERS["deepseek"].base_url),
                "local":     ("qwen3.5-9b", f"http://localhost:{os.environ.get('LLAMA_CHAT_PORT', '8080')}/v1"),
            }.get(judge_provider, (PROVIDERS["deepseek"].default_model, PROVIDERS["deepseek"].base_url))

        cmd = _python_cmd([
            "-m", "mineru_eval.evaluate",
            "--mode", mode,
            "--input", str(dataset),
            "--output", str(Path(output_dir) / f"{model}_report.json"),
            "--target-model", model,
            "--judge-model", judge_model,
            "--judge-base-url", judge_base_url,
            "--resume",
        ])
        if target_base_url:
            cmd.extend(["--target-base-url", target_base_url])
        if target_api_key:
            cmd.extend(["--target-api-key", target_api_key])
        if max_questions > 0:
            cmd.extend(["--limit", str(max_questions)])

        logger.info("  → model_evaluate: 评测 %s, mode=%s, base_url=%s",
                     model, mode, target_base_url or "(default)")
        logger.info("  → cmd: %s", _safe_cmd(cmd))
        return _run_subprocess(cmd, timeout=14400)

    @staticmethod
    def _resolve_target(model: str) -> Tuple[Optional[str], str]:
        m = model.lower()
        if "minimax" in m:
            return PROVIDERS["minimax"].base_url, PROVIDERS["minimax"].api_key_env
        if "mimo" in m:
            return PROVIDERS["mimo"].base_url, PROVIDERS["mimo"].api_key_env
        if "deepseek" in m:
            return PROVIDERS["deepseek"].base_url, PROVIDERS["deepseek"].api_key_env
        if ":" in model:
            return f"http://localhost:{os.environ.get('LLAMA_CHAT_PORT', '8080')}/v1", ""
        return None, "OPENAI_API_KEY"

    @staticmethod
    def _get_api_key(env_spec: str) -> str:
        for env_name in env_spec.split("|"):
            value = os.environ.get(env_name.strip(), "")
            if value:
                return value
        return ""

    def _exec_multi_judge_compare(self, params: dict, resolved=None) -> Tuple[int, str, str]:
        return exec_multi_judge_compare(params)

    def _exec_leaderboard_viz(self, params: dict, resolved=None) -> Tuple[int, str, str]:
        source = params.get("source", "benchmark/eval/results/")
        formats = params.get("formats", ["json", "html"])
        output_dir = params.get("output_dir", "benchmark/eval/results/")

        leaderboard_json = str(Path(output_dir) / "leaderboard.json")
        leaderboard_html = str(Path(output_dir) / "leaderboard.html")

        all_outputs = []
        final_rc = 0

        if "json" in formats:
            cmd_json = _python_cmd([
                "-m", "mineru_eval.summarize_multi",
                "--input", str(source),
                "--output", leaderboard_json,
            ])
            logger.info("  → leaderboard_viz [json]: summarize_multi.py → %s", leaderboard_json)
            logger.info("  → cmd: %s", _safe_cmd(cmd_json))
            rc, out, err = _run_subprocess(cmd_json, timeout=600)
            all_outputs.append(f"[json] rc={rc}\n{out[-200:]}")
            if rc != 0:
                final_rc = rc

        if "html" in formats:
            cmd_html = _python_cmd([
                "-m", "mineru_eval.report_viz",
                "--input", str(source),
                "--output", leaderboard_html,
            ])
            logger.info("  → leaderboard_viz [html]: report_viz.py → %s", leaderboard_html)
            logger.info("  → cmd: %s", _safe_cmd(cmd_html))
            rc, out, err = _run_subprocess(cmd_html, timeout=300)
            all_outputs.append(f"[html] rc={rc}\n{out[-200:]}")
            if rc != 0:
                final_rc = 1

        combined = "\n---\n".join(all_outputs)
        logger.info("  → leaderboard_viz: 完成, formats=%s", formats)
        return final_rc, combined, ""

    def _exec_multi_format_parse(self, params: dict, resolved=None) -> Tuple[int, str, str]:
        return exec_multi_format_parse(params)

    def _exec_local_finetune(self, params: dict, resolved=None) -> Tuple[int, str, str]:
        """已停用：原实现跑 finetune/scripts/train_qlora.py，该路径随重构消失。

        训练/微调的权威实现 2026-09-25 起在兄弟仓 Kiln。本仓候选
        `mineru_benchmark.qlora_train` 虽已可 import（sciverse 已声明），但它自带数据前置
        且无 resume/force 语义，不适合作为该 intent 的后继，故 intent 保持停用而非改指。
        """
        payload = json.dumps({
            "intent": "local_finetune",
            "status": "disabled",
            "reason": "训练管线已迁出本仓（Kiln）；本仓残留入口指向不存在的脚本",
            "successors": ["Kiln 仓的训练编排入口", "本仓仅保留评测/生成/判分链"],
        }, ensure_ascii=False)
        logger.warning("local_finetune 已停用：训练不在本仓执行")
        return 0, payload, ""

    def _exec_reproduce_all(self, params: dict, resolved=None) -> Tuple[int, str, str]:
        """已停用：原实现驱动 run_agent.ps1，该启动器已随 Windows 时代归档。"""
        payload = json.dumps({
            "intent": "reproduce_all",
            "status": "disabled",
            "reason": "run_agent.ps1 已随 Windows 启动器归档（_archive/win-launchers/README.md）",
            "params": {
                "steps": params.get("steps", []),
                "local_mode": params.get("local_mode", self._local_mode),
                "skip_existing": params.get("skip_existing", True),
                "output_dir": params.get("output_dir", "benchmark/reproduce/"),
            },
            "successors": [
                "python -m mineru_eval.eval_cli",
                "scripts/b1_verdict.py",
                "scripts/aggregate_n3_final.py",
            ],
        }, ensure_ascii=False)
        logger.warning("reproduce_all 已停用，请按 successors 逐步执行")
        return 0, payload, ""

    def _exec_medical_rag(self, params: dict, resolved=None) -> Tuple[int, str, str]:
        return exec_medical_rag(params)

    def _exec_sciverse_search(self, params: dict, resolved=None) -> Tuple[int, str, str]:
        from mineru_agent.search_handler import handle_sciverse_search

        result = handle_sciverse_search({
            "query": params.get("query", "model predictive control"),
            "top_k": int(params.get("top_k", 5)),
        })
        n_hits = result.get("total_hits", 0)
        titles = [h.get("title", "")[:60] for h in result.get("hits", [])[:3]]
        logger.info("  → sciverse_search: %d hits, top: %s", n_hits, titles)
        return (0, f"sciverse_search completed: {n_hits} hits", json.dumps(result, ensure_ascii=False, default=str))
