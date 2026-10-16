"""Phase 1e: 10 intent 逐个端到端干跑 + 日志验证

规则 10.1.3: 每个 intent 独立 try/except，一个异常不污染后续结果。

验证维度（每 intent 7 项）：
  1. Registry Check — intent_id 是否在 IntentRegistry 中
  2. Dispatch Check — Executor.dispatch_map 是否有对应 handler
  3. Schema Check  — parameters 是否符合 JSON Schema
  4. Dependency Check — depends_on 是否全部为有效 intent_id
  5. Executor Dry-Run — Executor.execute(dry_run=True) 产生有效 LogStep
  6. CLI Dry-Run — ControlSciAgentCLI --dry-run → ExecutionLog 完整
  7. Router Check (可选，需 API) — IntentRouter.plan() 从触发语产出合理计划

Usage:
  .venv/bin/python benchmark/agent/_verify_10_intents.py
  .venv/bin/python benchmark/agent/_verify_10_intents.py --skip-router
  .venv/bin/python benchmark/agent/_verify_10_intents.py --json-only -o verify_report.json
"""

import argparse
import json
import logging
import os
import sys
import textwrap
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ROOT = Path(__file__).resolve().parents[2]  # Phase 5: sys.path hack removed, use mineru_benchmark.agent.utils.ROOT instead

from mineru_agent.agent_cli import (
    IntentRegistry,
    IntentRouter,
    Executor,
    PlanStep,
    ControlSciAgentCLI,
)
from mineru_core.log_schema import LogStep

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("Verify10Intents")


INTENT_TRIGGERS: List[Tuple[str, str]] = [
    ("corpus_parse",           "解析这 23 本控制科学教材为结构化语料"),
    ("cross_modal_audit",      "审计这批 arXiv 论文的公式-图片跨模态对齐质量"),
    ("corpus_quality_report",  "生成语料质量全维度报告"),
    ("benchmark_build",        "基于语料生成 500 题评测数据集，覆盖 A/B/C/D 四维度"),
    ("quality_arbitrate",      "对生成题目做双层 LLM 仲裁确保答案准确性"),
    ("model_evaluate",         "在控制科学 benchmark 上评测 deepseek-v4-flash 模型"),
    ("multi_judge_compare",    "对比 API Judge 和本地 Ollama Judge 的评分差异"),
    ("leaderboard_viz",        "生成排行榜 JSON 和可视化 HTML 页面"),
    ("local_finetune",         "QLoRA 微调 qwen3.5:9b 并验证 Perplexity 改善"),
    ("reproduce_all",          "一键复现从语料解析到排行榜的完整评测流程"),
]


@dataclass
class VerifyItem:
    intent_id: str
    trigger: str
    checks: Dict[str, dict] = field(default_factory=dict)
    overall_pass: Optional[bool] = None
    error_summary: str = ""


@dataclass
class VerifyReport:
    schema_version: str = "1.0"
    verified_at: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    total: int = 0
    passed: int = 0
    failed: int = 0
    skipped: int = 0
    router_skipped: bool = False
    items: List[VerifyItem] = field(default_factory=list)


def _make_result(status: str, detail: str, duration_ms: int = 0) -> dict:
    return {"status": status, "detail": detail, "duration_ms": duration_ms}


def _check_registry(intent_id: str, registry: IntentRegistry) -> Tuple[bool, str]:
    entry = registry.get(intent_id)
    if entry is None:
        return False, f"intent_id '{intent_id}' 不在 IntentRegistry 中"
    required_fields = ["intent_id", "name", "toolchain", "resource_type", "parameters"]
    missing = [f for f in required_fields if f not in entry]
    if missing:
        return False, f"缺少字段: {missing}"
    return True, f"已注册 (resource_type={entry['resource_type']}, toolchain={entry.get('toolchain', [])})"


def _check_dispatch(intent_id: str, executor: Executor) -> Tuple[bool, str]:
    try:
        handler = executor._dispatch_map.get(intent_id)
    except AttributeError:
        return False, "executor._dispatch_map 不可访问（Executor 可能已重构）"
    if handler is None:
        return False, "dispatch_map 中缺少 handler"
    return True, f"handler={handler.__name__}"


def _check_schema(intent_id: str, registry: IntentRegistry) -> Tuple[bool, str]:
    entry = registry.get(intent_id)
    if entry is None:
        return False, "intent 未注册"
    params = entry.get("parameters", {})
    if not isinstance(params, dict):
        return False, "parameters 不是 dict"
    if "properties" not in params:
        return False, "parameters 缺少 properties 字段"

    props = params.get("properties", {})
    required = params.get("required", [])
    additional = params.get("additionalProperties", None)

    issues = []
    if additional is not False:
        issues.append(f"additionalProperties 不为 false (当前: {additional})")
    for r in required:
        if r not in props:
            issues.append(f"required 字段 '{r}' 不在 properties 中")
    if issues:
        return False, "; ".join(issues)
    return True, f"params={len(props)} required={required}"


def _check_dependencies(intent_id: str, registry: IntentRegistry) -> Tuple[bool, str]:
    entry = registry.get(intent_id)
    if entry is None:
        return False, "intent 未注册"
    deps = entry.get("depends_on", [])
    if not deps:
        return True, "无依赖"
    invalid = [d for d in deps if registry.get(d) is None]
    if invalid:
        return False, f"无效依赖: {invalid}"
    return True, f"依赖: {deps}"


def _check_dry_run(intent_id: str, registry: IntentRegistry,
                   executor: Executor) -> Tuple[bool, str, Optional[LogStep]]:
    entry = registry.get(intent_id)
    if entry is None:
        return False, "intent 未注册", None

    step = PlanStep(
        intent_id=intent_id,
        intent_name=entry.get("name", intent_id),
        description=entry.get("description", intent_id),
        tool=", ".join(entry.get("toolchain", [])),
        resource_type=entry.get("resource_type", "script"),
        depends_on=entry.get("depends_on", []),
        parameters={},
        rank=1,
    )

    log_step, output, error = executor.execute(step, dry_run=True)

    if log_step.status != "skipped":
        return False, f"dry_run 状态异常: expected='skipped', got='{log_step.status}'", log_step
    if log_step.step_id != 1:
        return False, f"step_id 异常: {log_step.step_id}", log_step
    if not log_step.step_name:
        return False, "step_name 为空", log_step
    return True, f"dry_run OK, step_name='{log_step.step_name}'", log_step


def _check_router(intent_id: str, trigger: str, registry: IntentRegistry,
                  router: IntentRouter, valid_ids: set = None) -> Tuple[bool, str]:
    try:
        plan = router.plan(trigger)
    except Exception as e:
        return False, f"IntentRouter.plan() 异常: {e}"

    if not plan:
        return False, "Router 返回空 plan"

    matched = [s for s in plan if s.intent_id == intent_id]
    if not matched:
        plan_ids = [s.intent_id for s in plan]
        return False, f"plan 中未找到目标 intent_id '{intent_id}', plan={plan_ids}"

    step = matched[0]
    valid_ids = valid_ids or set(registry.list_ids())
    invalid_deps = [d for d in step.depends_on if d not in valid_ids]
    if invalid_deps:
        return False, f"plan step 含无效依赖: {invalid_deps}"

    return True, f"plan 包含目标 intent (step={step.rank}, desc='{step.description[:50]}')"


def _check_cli_dry_run(intent_id: str, registry: IntentRegistry,
                       cli: ControlSciAgentCLI) -> Tuple[bool, str]:
    entry = registry.get(intent_id)
    if entry is None:
        return False, "intent 未注册"

    try:
        exec_log = cli.run(intent_list=[intent_id])
    except Exception as e:
        return False, f"CLI dry_run 异常: {e}"

    if exec_log.final_status not in ("completed", "skipped"):
        return False, f"exec_log 状态异常: {exec_log.final_status}"

    steps = exec_log.steps
    if not steps:
        return False, "exec_log 无 step 记录"

    step = steps[0]
    if step.status != "skipped":
        return False, f"step status 异常: expected='skipped', got='{step.status}'"

    return True, f"CLI dry_run OK (status={exec_log.final_status}, steps={len(steps)})"


def _wrap_check(checks: dict, key: str, label: str, fn, *args):
    t0 = time.time()
    try:
        result = fn(*args)
        ok = result[0]
        detail = result[1]
        checks[key] = _make_result("pass" if ok else "fail", detail, int((time.time() - t0) * 1000))
        logger.info("  %s %s: %s", "✅" if ok else "❌", label, detail)
    except Exception as e:
        checks[key] = _make_result("error", str(e))
        logger.error("  ❌ %s: 异常 — %s", label, e)


def run(all_intents: List[Tuple[str, str]], skip_router: bool = False,
        local_mode: bool = False) -> VerifyReport:
    if not all_intents:
        raise ValueError("all_intents 为空 — 没有可验证的 intent")

    registry = IntentRegistry()
    router = IntentRouter(registry)
    executor = Executor(local_mode=local_mode, registry=registry)
    cli = ControlSciAgentCLI(local_mode=local_mode, dry_run=True)
    report = VerifyReport()

    trigger_ids = set(pid for pid, _ in all_intents)
    registry_ids = set(registry.list_ids())
    only_in_trigger = trigger_ids - registry_ids
    only_in_registry = registry_ids - trigger_ids
    if only_in_registry or only_in_trigger:
        msgs = []
        if only_in_registry:
            msgs.append(f"capabilities.json 存在但验证表缺失: {sorted(only_in_registry)}")
        if only_in_trigger:
            msgs.append(f"验证表有但 capabilities.json 缺失: {sorted(only_in_trigger)}")
        logger.error("intent 列表不一致: %s", "; ".join(msgs))

    if skip_router:
        report.router_skipped = True
        logger.info("⏭️  跳过 IntentRouter 验证 (--skip-router)")
    else:
        api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("DEEPSEEK_API_KEY", "")
        if not api_key:
            logger.warning("⚠️  API Key 未设置，自动跳过 IntentRouter 验证")
            report.router_skipped = True
            skip_router = True

    print()
    print("=" * 72)
    print("  ControlSci Data Agent — 10 Intent 端到端验证")
    print("=" * 72)
    print(f"  Router: {'⏭️ 跳过' if skip_router else '🔮 启用'}")
    print(f"  Local:  {'🏠 本地模式' if local_mode else '☁️  API 模式'}")
    print(f"  时间:   {report.verified_at}")
    print("=" * 72)
    print()

    for idx, (intent_id, trigger) in enumerate(all_intents, 1):
        item = VerifyItem(intent_id=intent_id, trigger=trigger)
        logger.info("━" * 60)
        logger.info("[%2d/%d] %s — %s", idx, len(all_intents), intent_id, trigger[:60])

        checks = {}

        _wrap_check(checks, "registry",     "registry",     _check_registry,     intent_id, registry)
        _wrap_check(checks, "dispatch",     "dispatch",     _check_dispatch,     intent_id, executor)
        _wrap_check(checks, "schema",       "schema",       _check_schema,       intent_id, registry)
        _wrap_check(checks, "dependencies", "dependencies", _check_dependencies, intent_id, registry)
        _wrap_check(checks, "dry_run",      "dry_run",      _check_dry_run,      intent_id, registry, executor)
        _wrap_check(checks, "cli_dry_run",  "cli_dry_run",  _check_cli_dry_run,  intent_id, registry, cli)

        if not skip_router:
            _wrap_check(checks, "router", "router", _check_router, intent_id, trigger, registry, router, registry_ids)
        else:
            checks["router"] = _make_result("skipped", "IntentRouter 跳过")

        item.checks = checks

        all_ok = all(
            c["status"] in ("pass", "skipped")
            for c in checks.values()
        )
        item.overall_pass = all_ok

        if all_ok:
            report.passed += 1
            logger.info("  └─ 🟢 全部通过")
        else:
            failed_checks = [k for k, v in checks.items() if v["status"] not in ("pass", "skipped")]
            item.error_summary = ", ".join(failed_checks)
            report.failed += 1
            logger.warning("  └─ 🔴 失败项: %s", item.error_summary)

        report.items.append(item)
        print()

    report.total = len(report.items)
    report.skipped = report.total - report.passed - report.failed

    print("=" * 72)
    print(f"  结果: ✅ {report.passed} 通过  ❌ {report.failed} 失败  ⏭️ {report.skipped} 跳过")
    print(f"  Router: {'未启用' if report.router_skipped else '已启用'}")
    print("=" * 72)
    print()

    return report


def save_report(report: VerifyReport, path: str):
    if not path or not path.strip():
        raise ValueError("save_report: path 不能为空")
    out_path = Path(path)
    if out_path.suffix != ".json":
        logger.warning("路径后缀非 .json: %s", path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    items_data = []
    for item in report.items:
        items_data.append({
            "intent_id": item.intent_id,
            "trigger": item.trigger,
            "overall_pass": item.overall_pass,
            "error_summary": item.error_summary,
            "checks": item.checks,
        })

    payload = {
        "schema_version": report.schema_version,
        "verified_at": report.verified_at,
        "summary": {
            "total": report.total,
            "passed": report.passed,
            "failed": report.failed,
            "skipped": report.skipped,
            "router_skipped": report.router_skipped,
        },
        "items": items_data,
    }

    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(out_path)
    logger.info("📝 验证报告已保存: %s", out_path)


VERSION = "1.1.0"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="_verify_10_intents.py",
        description="10 intent 逐个端到端干跑 + 日志验证 (Phase 1e)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            示例:
              %(prog)s                          # 全量验证（含 IntentRouter）
              %(prog)s --skip-router            # 跳过 IntentRouter（无需 API Key）
              %(prog)s --local                  # 本地模式
              %(prog)s --json-only -o report.json  # 仅 JSON 输出
        """),
    )
    p.add_argument("--version", "-V", action="version",
                   version=f"ControlSci Verify 10 Intents v{VERSION}")
    p.add_argument("--skip-router", action="store_true",
                   help="跳过 IntentRouter LLM 调用验证")
    p.add_argument("--local", "-l", action="store_true",
                   help="本地模式（Ollama + 本地 MinerU）")
    p.add_argument("--json-only", action="store_true",
                   help="仅输出 JSON 报告（不打印控制台日志）")
    p.add_argument("--output", "-o",
                   default=f"benchmark/agent/logs/verify_10_intents-{datetime.now().strftime('%Y%m%d-%H%M%S')}.json",
                   help="验证报告 JSON 输出路径")
    return p


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.json_only:
        logging.getLogger().setLevel(logging.ERROR)

    report = run(
        all_intents=INTENT_TRIGGERS,
        skip_router=args.skip_router,
        local_mode=args.local,
    )

    save_report(report, args.output)

    if report.failed > 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
