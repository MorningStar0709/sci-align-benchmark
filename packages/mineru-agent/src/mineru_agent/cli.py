"""Agent CLI — ControlSciAgentCLI orchestrator and argparse entry point.

Migrated from ``benchmark/agent/agent_cli.py``.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import textwrap
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import httpx

from mineru_agent.executor import Executor
from mineru_agent.intent_registry import IntentRegistry, PlanStep, VerifyResult
from mineru_agent.intent_router import IntentRouter
from mineru_agent.utils import (
    VERSION,
    _configure_stdio,
    _get_logs_dir,
    _setup_logging,
)
from mineru_agent.verifier import Verifier
from mineru_core.log_schema import ExecutionLog, LogStep
from mineru_core.settings import PROVIDERS

ROOT = Path(__file__).resolve().parents[4]  # restored Phase 5

# Ensure stdio is configured early
_configure_stdio()

logger = logging.getLogger(__name__)


# ============================================================
# ControlSciAgentCLI — main orchestrator
# ============================================================


class ControlSciAgentCLI:
    def __init__(self, local_mode: bool = False, dry_run: bool = False,
                 max_retries: int = 2, enable_verify: bool = True):
        self._local_mode = local_mode
        self._dry_run = dry_run
        self._max_retries = max_retries
        self._enable_verify = enable_verify
        self._registry = IntentRegistry()
        self._router = IntentRouter(self._registry)
        self._scheduler = self._get_scheduler(local_mode=local_mode)
        self._executor = Executor(scheduler=self._scheduler, local_mode=local_mode,
                                   registry=self._registry)
        self._verifier: Optional[Verifier] = None
        self._execution_log: Optional[ExecutionLog] = None
        self._env_checked: Optional[List[str]] = None
        _get_logs_dir().mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _get_scheduler(local_mode: bool = False):
        # Lazy import to avoid circular dependency at module level
        from mineru_agent.resource_scheduler import get_global_scheduler
        return get_global_scheduler(local_mode=local_mode)

    @property
    def _verifier_instance(self) -> Verifier:
        if self._verifier is None:
            self._verifier = Verifier()
        return self._verifier

    def _check_env(self) -> List[str]:
        if self._env_checked is not None:
            return self._env_checked
        warnings = []
        checks = [
            (PROVIDERS["deepseek"].api_key_env, "DeepSeek API"),
            (PROVIDERS["mimo"].api_key_env, "MiMo API"),
        ]
        for var_spec, name in checks:
            if not any(os.environ.get(var.strip()) for var in var_spec.split("|")):
                warnings.append(f"{name} Key 未设置 (环境变量 {var_spec.replace('|', '/')})")

        llama_port = os.environ.get("LLAMA_CHAT_PORT", "8080")
        try:
            resp = httpx.get(f"http://localhost:{llama_port}/health", timeout=3)
            if resp.status_code != 200:
                warnings.append(f"llama.cpp 服务无响应: port {llama_port}")
        except Exception as e:
            logger.debug("llama.cpp 健康检查异常（非关键）: %s", e)

        self._env_checked = warnings
        return warnings

    def run(self, user_input: str = None, intent_list: List[str] = None,
            intent_params: dict = None, expand_deps: bool = True) -> ExecutionLog:
        env_warnings = self._check_env()
        for w in env_warnings:
            logger.warning("⚠ %s", w)

        task_id = f"{datetime.now().strftime('agent-%Y%m%d-%H%M%S-%f')}-{uuid.uuid4().hex[:8]}"
        task_name = user_input[:80] if user_input else (
            ",".join(intent_list) if intent_list else "agent-task"
        )
        self._execution_log = ExecutionLog(task_id=task_id, task_name=task_name)
        self._execution_log.final_status = "running"
        t0 = time.time()

        plan = self._create_plan(user_input, intent_list, intent_params, expand_deps, t0)
        if plan is None:
            return self._execution_log

        logger.info("📋 执行计划 (%d 步):", len(plan))
        for s in plan:
            deps = f" (依赖: {', '.join(s.depends_on)})" if s.depends_on else ""
            logger.info("  %d. [%s] %s%s", s.rank, s.intent_id, s.description[:60], deps)

        self._execution_log.final_status = "running"
        all_passed = True
        any_failed = False

        for step in plan:
            logger.info("━" * 50)
            logger.info("▶ Step %d/%d: %s (%s)", step.rank, len(plan),
                         step.intent_id, step.intent_name)
            logger.info("  └ 工具: %s | 资源类型: %s", step.tool, step.resource_type)

            log_step, output, error = self._execute_with_retry(step)
            self._execution_log.add_step(log_step)

            status_icon = {"success": "✅", "failed": "❌", "timeout": "⏰", "skipped": "⏭"}.get(log_step.status, "❓")
            logger.info("%s Step %d 完成: %s (%dms)",
                         status_icon, step.rank, log_step.status, log_step.duration_ms)

            if log_step.status == "failed":
                any_failed = True
                all_passed = False
            elif log_step.status == "skipped":
                pass
            else:
                step_passed = self._execute_step_verify(step, output, error)
                if not step_passed:
                    all_passed = False

        self._execution_log.total_duration_ms = int((time.time() - t0) * 1000)
        if all_passed and not any_failed:
            self._execution_log.final_status = "completed"
        elif any_failed:
            self._execution_log.final_status = "failed"
        else:
            self._execution_log.final_status = "partial"

        return self._execution_log

    def _execute_with_retry(self, step: PlanStep) -> Tuple[LogStep, str, Optional[str]]:
        resolved = self._scheduler.resolve(step.intent_id, local_mode=self._local_mode)
        log_step, output, error = self._executor.execute(step, dry_run=self._dry_run, resolved=resolved)

        if log_step.status != "failed" or self._dry_run:
            return log_step, output, error

        if self._max_retries == 0:
            return log_step, output, error

        improved_params = {}
        self_improve_reason = ""
        score_before = 0.0
        if self._enable_verify:
            verify_result = self._verifier_instance.verify(step, output, error)
            score_before = verify_result.score
            logger.info("  ⚠️ 首次验证: score=%.1f issues=%s",
                         verify_result.score, verify_result.issues[:3] if verify_result.issues else [])
            if score_before < 0.5 and verify_result.correction_hint:
                improved_params, si_reason = self._self_improve_params(
                    step.intent_id, step.intent_name, step.description,
                    step.parameters, output, verify_result,
                )
                if improved_params:
                    self_improve_reason = si_reason
                    step.parameters = {**step.parameters, **improved_params}
                    logger.info("  📝 参数已更新: %s", json.dumps(improved_params, ensure_ascii=False))

        for attempt in range(1, self._max_retries + 1):
            logger.info("  🔄 重试 %d/%d: %s", attempt, self._max_retries, step.intent_id)
            time.sleep(2)

            resolved = self._scheduler.resolve(step.intent_id, local_mode=self._local_mode)
            fallback_provider = self._scheduler.get_fallback(resolved)
            if fallback_provider:
                logger.info("  🔽 降级至 provider=%s", fallback_provider.provider)
                resolved = fallback_provider

            log_step2, output2, error2 = self._executor.execute(step, dry_run=False, resolved=resolved)
            if log_step2.status == "success":
                if self._enable_verify and self_improve_reason:
                    verify2 = self._verifier_instance.verify(step, output2, error2)
                    delta = verify2.score - score_before
                    logger.info("  📈 Self-Improve Δ: %+.1f, reason=%s", delta, self_improve_reason)
                return log_step2, output2, error2
            log_step = log_step2
            output = output2
            error = error2

        return log_step, output, error

    def _create_plan(self, user_input, intent_list, intent_params, expand_deps, t0):
        """Create execution plan from user_input or intent_list. Returns None on error."""
        if user_input and not intent_list:
            if intent_params:
                logger.warning("⚠ --intent-params 仅在 --intents 模式下生效，当前被忽略")
            logger.info("🔮 Intent Router 正在将自然语言拆解为执行计划...")
            try:
                plan = self._router.plan(user_input)
                if expand_deps:
                    plan = self._expand_plan_dependencies(plan)
            except Exception as e:
                logger.error("❌ Intent Router 失败: %s", e)
                self._execution_log.final_status = "failed"
                self._execution_log.total_duration_ms = int((time.time() - t0) * 1000)
                return None
        elif intent_list:
            plan = self._intent_list_to_plan(intent_list, intent_params)
            if expand_deps:
                plan = self._expand_plan_dependencies(plan)
        else:
            logger.error("必须提供 user_input 或 intent_list")
            self._execution_log.final_status = "failed"
            self._execution_log.total_duration_ms = int((time.time() - t0) * 1000)
            return None
        return plan

    def _execute_step_verify(self, step, output, error):
        """Verify step output and attempt self-improve if needed. Returns True if passed."""
        if not (self._enable_verify and not self._dry_run):
            return True
        verify_result = self._verifier_instance.verify(step, output, error)
        v_icon = "✅" if verify_result.passed else "⚠️"
        logger.info("  %s 验证: score=%.1f %s",
                     v_icon, verify_result.score, verify_result.reason[:100])
        if verify_result.passed:
            return True

        score_before = verify_result.score
        if not verify_result.correction_hint:
            return False

        improved_params, si_reason = self._self_improve_params(
            step.intent_id, step.intent_name, step.description,
            step.parameters, output, verify_result,
        )
        if not improved_params:
            return False

        step.parameters = {**step.parameters, **improved_params}
        logger.info("  🔄 Self-Improve 重试: %s", si_reason)
        resolved2 = self._scheduler.resolve(step.intent_id, local_mode=self._local_mode)
        log_step2, output2, error2 = self._executor.execute(
            step, dry_run=False, resolved=resolved2)
        verify2 = self._verifier_instance.verify(step, output2, error2)
        delta = verify2.score - score_before
        logger.info("  📈 Self-Improve Δ: %+.1f, reason=%s", delta, si_reason)
        if verify2.passed:
            self._execution_log.steps[-1] = log_step2
            logger.info("  ✅ Self-Improve 后验证通过")
            return True
        return False

    def _self_improve_params(self, intent_id: str, intent_name: str, description: str,
                             original_params: dict, exec_output: str,
                             verify_result: 'VerifyResult') -> Tuple[dict, str]:
        from openai import OpenAI, DefaultHttpxClient
        from mineru_eval.client_factory import wrap_client

        api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("DEEPSEEK_API_KEY", "")
        if not api_key:
            logger.warning("  Self-Improve 跳过：无 API Key")
            return {}, "no-api-key"

        logger.info("  🔄 Self-Improve: 正在调用 DeepSeek 生成改进参数...")

        prompt = (
            f"执行意图 [{intent_id}] 第1次尝试未达预期。\n\n"
            f"意图名称: {intent_name}\n"
            f"描述: {description}\n"
            f"原始参数: {json.dumps(original_params, ensure_ascii=False)}\n"
            f"执行输出（截取前 1500 字符）:\n{exec_output[:1500]}\n"
            f"验证问题: {json.dumps(verify_result.issues, ensure_ascii=False)}\n"
            f"修正建议: {verify_result.correction_hint}\n\n"
            f"请给出改进后的参数 JSON（仅包含需要修改的字段，使用原始参数中的英文 key）：\n"
            f'{{"temperature": 0.3, "max_tokens": 8192, "reason": "调整原因"}}'
        )

        try:
            client = wrap_client(OpenAI(
                api_key=api_key,
                base_url=PROVIDERS["deepseek"].base_url,
                timeout=60,
                max_retries=0,
                http_client=DefaultHttpxClient(proxy=None, trust_env=False),
            ))
            response = client.chat.completions.create(
                model="deepseek-v4-flash",
                messages=[
                    {"role": "system", "content": "你是 ControlSci Data Agent 的参数优化器。根据验证反馈给出精确的参数调整。仅输出 JSON。"},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
                max_tokens=1024,
            )
            raw = response.choices[0].message.content.strip()
            from mineru_agent.utils import _extract_json_block
            json_str = _extract_json_block(raw)
            improved = json.loads(json_str)
        except Exception as e:
            logger.warning("  Self-Improve DeepSeek 调用失败: %s", e)
            return {}, f"deepseek-error: {e}"

        if not isinstance(improved, dict):
            logger.warning("  Self-Improve 返回非字典: %s", type(improved))
            return {}, "invalid-response-type"

        reason = improved.pop("reason", "")

        logger.info("  ✅ Self-Improve 建议: %s → %s", reason, json.dumps(improved, ensure_ascii=False))
        return improved, reason

    def _intent_list_to_plan(self, intent_ids: List[str],
                              intent_params: dict = None,
                              log_params: bool = True) -> List[PlanStep]:
        plan = []
        intents_info = self._registry.get_batch(intent_ids)
        for idx, pid in enumerate(intent_ids, 1):
            pid = pid.strip()
            intent_info = intents_info.get(pid)
            if intent_info is None:
                logger.warning("跳过未知 intent_id: %s", pid)
                continue
            params = {}
            if intent_params and pid in intent_params:
                raw = intent_params[pid]
                if isinstance(raw, dict):
                    params = dict(raw)
                    if log_params:
                        logger.info("  📝 [%s] 使用自定义参数: %s", pid,
                                     json.dumps(params, ensure_ascii=False))
                else:
                    logger.warning("  ⚠️ [%s] intent_params 值不是 dict (type=%s), 已跳过",
                                   pid, type(raw).__name__)
            plan.append(PlanStep(
                intent_id=pid,
                intent_name=intent_info.get("name", pid),
                description=intent_info.get("description", pid),
                tool=", ".join(intent_info.get("toolchain", [])),
                resource_type=intent_info.get("resource_type", "script"),
                depends_on=intent_info.get("depends_on", []),
                parameters=params,
                rank=idx,
            ))
        return plan

    def _expand_plan_dependencies(self, plan: List[PlanStep]) -> List[PlanStep]:
        requested_params = {step.intent_id: step.parameters for step in plan}
        expanded_ids: List[str] = []
        visiting: set[str] = set()

        def visit(intent_id: str):
            if intent_id in expanded_ids:
                return
            if intent_id in visiting:
                raise ValueError(f"Intent dependency cycle detected at {intent_id}")
            intent_info = self._registry.get(intent_id)
            if intent_info is None:
                raise ValueError(f"Unknown intent_id in dependency graph: {intent_id}")
            visiting.add(intent_id)
            for dep in intent_info.get("depends_on", []):
                visit(dep)
            visiting.remove(intent_id)
            expanded_ids.append(intent_id)

        for step in plan:
            visit(step.intent_id)

        expanded = self._intent_list_to_plan(
            expanded_ids,
            {pid: requested_params[pid] for pid in requested_params},
            log_params=False,
        )
        if [s.intent_id for s in expanded] != [s.intent_id for s in plan]:
            logger.info("  🧭 已补齐依赖 DAG: %s", " → ".join(s.intent_id for s in expanded))
        return expanded

    def save_log(self, path: str = None):
        if self._execution_log is None:
            logger.warning("无执行日志可保存")
            return ""
        path = path or str(_get_logs_dir() / f"{self._execution_log.task_id}.json")
        try:
            self._execution_log.save(path)
            logger.info("执行日志已保存: %s", path)
            return path
        except Exception as e:
            logger.error("执行日志保存失败（不覆盖主任务状态）: %s", e)
            return ""

    def print_summary(self):
        if self._execution_log is None:
            return
        log = self._execution_log
        print("\n" + "=" * 60)
        print("  ControlSci Data Agent — 执行摘要")
        print("=" * 60)
        print(f"  Task:       {log.task_name}")
        print(f"  Status:     {log.final_status}")
        print(f"  耗时:       {log.total_duration_ms / 1000:.1f}s")
        print(f"  步骤:       {len(log.steps)}")
        success_count = sum(1 for s in log.steps if s.status == "success")
        failed_count = sum(1 for s in log.steps if s.status == "failed")
        skipped_count = sum(1 for s in log.steps if s.status == "skipped")
        timeout_count = sum(1 for s in log.steps if s.status == "timeout")
        print(f"    [OK] 成功: {success_count}  [FAIL] 失败: {failed_count}  [SKIP] 跳过: {skipped_count}  [TIMEOUT] 超时: {timeout_count}")
        print("=" * 60)


# ============================================================
# CLI Entry Point
# ============================================================


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agent_cli.py",
        description="ControlSci Data Agent CLI — 科学文档跨模态语料智能体",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(f"""\
            示例:
              %(prog)s "审计 arXiv 论文的跨模态对齐质量"
              %(prog)s --intents corpus_parse,cross_modal_audit
              %(prog)s --intents cross_modal_audit,leaderboard_viz -P '{{"cross_modal_audit":{{"max_images":10}}}}'
              %(prog)s --dry-run "生成 500 题评测数据集并跑评测"
              %(prog)s --local "QLoRA 微调 qwen3.5 并验证 Perplexity"
              %(prog)s --list-intents

            版本: {VERSION}
        """),
    )

    parser.add_argument(
        "input", nargs="?",
        help="自然语言任务描述（由 Intent Router 自动拆解）",
    )
    parser.add_argument(
        "--intents", "-i",
        help="手动指定 intent 序列，逗号分隔（跳过 Intent Router）",
    )
    parser.add_argument(
        "--dry-run", "-n", action="store_true",
        help="干跑模式：仅生成计划并输出，不实际执行",
    )
    parser.add_argument(
        "--local", "-l", action="store_true",
        help="本地模式：优先使用 Ollama + 本地 MinerU，不依赖外部 API",
    )
    parser.add_argument(
        "--max-retries", type=int, default=2,
        help="每步最大重试次数（默认 2）",
    )
    parser.add_argument(
        "--no-verify", action="store_true",
        help="跳过 LLM 验证步骤（仅执行，不验证产出质量）",
    )
    parser.add_argument(
        "--output-log", "-o",
        help="执行日志 JSON 输出路径",
    )
    parser.add_argument(
        "--allow-partial", action="store_true",
        help="允许 partial 状态以 0 退出；默认 partial 返回 2，避免评审复现误判为完整成功。",
    )
    parser.add_argument(
        "--intent-params", "-P",
        help='per-intent 自定义参数 JSON，如 \'{"cross_modal_audit":{"max_images":10}}\'',
    )
    parser.add_argument(
        "--no-expand-deps", action="store_true",
        help="不自动补齐 intent 依赖；用于已有产物驱动的复现入口。",
    )
    parser.add_argument(
        "--list-intents", action="store_true",
        help="列出全部可用能力并退出",
    )
    parser.add_argument(
        "--version", "-V", action="version",
        version=f"ControlSci Data Agent CLI v{VERSION}",
    )

    return parser


def main():
    _setup_logging()
    parser = build_parser()
    args = parser.parse_args()

    if args.list_intents:
        registry = IntentRegistry()
        print(f"ControlSci Data Agent — {len(registry.list_intents())} 项可用能力:\n")
        for i in registry.list_intents():
            deps = i.get("depends_on", [])
            dep_str = f" 依赖: {', '.join(deps)}" if deps else ""
            rank = i.get("rank", "-")
            rank_str = str(rank).rjust(4)
            print(f"  {rank_str}. [{i['intent_id']:<25s}] {i.get('name', '')}")
            print(f"      工具: {', '.join(i.get('toolchain', ['-']))}{dep_str}")
            print(f"      资源: {i.get('resource_type', '-')}")
            print()
        return 0

    if not args.input and not args.intents:
        parser.print_help()
        print("\n❌ 请提供任务描述（自然语言）或使用 --intents 指定能力序列")
        return 1

    intent_list = None
    if args.intents:
        intent_list = [s.strip() for s in args.intents.split(",") if s.strip()]

    intent_params = None
    if args.intent_params:
        try:
            raw_intent_params = args.intent_params
            if raw_intent_params.startswith("@"):
                raw_intent_params = Path(raw_intent_params[1:]).read_text(encoding="utf-8")
            intent_params = json.loads(raw_intent_params)
            if not isinstance(intent_params, dict):
                logger.error("--intent-params 必须是 JSON 对象: {%s...}", str(args.intent_params)[:60])
                return 1
        except json.JSONDecodeError as e:
            logger.error("--intent-params JSON 解析失败: %s", e)
            return 1

    cli = ControlSciAgentCLI(
        local_mode=args.local,
        dry_run=args.dry_run,
        max_retries=args.max_retries,
        enable_verify=not args.no_verify,
    )

    try:
        exec_log = cli.run(
            user_input=args.input,
            intent_list=intent_list,
            intent_params=intent_params,
            expand_deps=not args.no_expand_deps,
        )
    except KeyboardInterrupt:
        logger.warning("用户中断执行")
        return 130
    except Exception as e:
        logger.error("执行异常: %s", e, exc_info=True)
        return 1

    log_path = cli.save_log(args.output_log)
    cli.print_summary()

    if exec_log.final_status == "completed":
        return 0
    elif exec_log.final_status == "partial":
        return 0 if args.allow_partial else 2
    else:
        return 1


if __name__ == "__main__":
    sys.exit(main())
