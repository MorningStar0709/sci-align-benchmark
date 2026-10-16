"""Evaluate IntentRouter robustness across natural-language task variants."""

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List

# ROOT = Path(__file__).resolve().parents[2]  # Phase 5: sys.path hack removed, use mineru_benchmark.agent.utils.ROOT instead

from mineru_agent.agent_cli import Executor, IntentRegistry, IntentRouter


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("RouterRobustnessEval")


@dataclass
class RouterCase:
    case_id: str
    expected_family: str
    input: str
    acceptable_intents: List[str]
    expected_first_intents: List[str]


@dataclass
class RouterRecord:
    case_id: str
    input: str
    expected_family: str
    acceptable_intents: List[str]
    expected_first_intents: List[str]
    predicted_intents: List[str] = field(default_factory=list)
    predicted_descriptions: List[str] = field(default_factory=list)
    dry_run_status: str = "not_run"
    dependency_repair: bool = False
    matched_acceptable: bool = False
    first_intent_match: bool = False
    intent_chain_match: bool = False
    single_intent_degradation: bool = False
    status: str = "pending"
    duration_ms: int = 0
    error: str = ""


CASES: List[RouterCase] = [
    RouterCase("literature_01", "literature_search", "检索 2025 年之后关于 control barrier function 和 safe reinforcement learning 的前沿论文。", ["arxiv_search", "sciverse_search"], ["arxiv_search", "sciverse_search"]),
    RouterCase("literature_02", "literature_search", "帮我找几篇最新的模型预测控制综述，优先覆盖正式出版物。", ["sciverse_search", "arxiv_search"], ["sciverse_search", "arxiv_search"]),
    RouterCase("literature_03", "literature_search", "搜索 differentiable MPC 方向的 arXiv 预印本，并准备后续加入评测基准。", ["arxiv_search", "mineru_parse", "corpus_parse", "benchmark_build"], ["arxiv_search"]),
    RouterCase("literature_04", "literature_search", "查找卡尔曼滤波在工业过程控制中的最新文献证据。", ["sciverse_search", "arxiv_search"], ["sciverse_search", "arxiv_search"]),
    RouterCase("literature_05", "literature_search", "给我发现一批控制科学新论文，后面要进入数据飞轮。", ["arxiv_search", "sciverse_search", "mineru_parse", "corpus_parse"], ["arxiv_search", "sciverse_search"]),
    RouterCase("audit_01", "cross_document_audit", "审计这批论文中的图片和公式是否语义一致，并输出质量报告。", ["corpus_parse", "cross_modal_audit", "corpus_quality_report"], ["corpus_parse", "cross_modal_audit"]),
    RouterCase("audit_02", "cross_document_audit", "检查语料库里公式、图像和 chunk 统计之间有没有明显错配。", ["cross_modal_audit", "corpus_quality_report"], ["cross_modal_audit", "corpus_quality_report"]),
    RouterCase("audit_03", "cross_document_audit", "对已有 arXiv 语料做跨模态对齐审计，重点看图表说明和公式引用。", ["cross_modal_audit"], ["cross_modal_audit"]),
    RouterCase("audit_04", "cross_document_audit", "生成一份语料全维度质量诊断，包含 chunk 完整度、公式抽取率和图片匹配情况。", ["corpus_quality_report", "cross_modal_audit"], ["corpus_quality_report", "cross_modal_audit"]),
    RouterCase("audit_05", "cross_document_audit", "我想知道当前语料是否适合继续训练和出题，请先做质量审计。", ["corpus_quality_report", "cross_modal_audit"], ["corpus_quality_report", "cross_modal_audit"]),
    RouterCase("parse_01", "multi_format_parse", "把这批 PDF 和 PPT 解析成结构化 Markdown，保留公式、表格和图片。", ["multi_format_parse", "mineru_parse", "corpus_parse"], ["multi_format_parse", "mineru_parse", "corpus_parse"]),
    RouterCase("parse_02", "multi_format_parse", "调用 MinerU 解析下载的论文，并导出图片资源。", ["mineru_parse", "corpus_parse"], ["mineru_parse", "corpus_parse"]),
    RouterCase("parse_03", "multi_format_parse", "把企业技术报告和表格文档统一转成可检索的语料。", ["multi_format_parse", "corpus_parse", "mineru_parse"], ["multi_format_parse", "corpus_parse", "mineru_parse"]),
    RouterCase("parse_04", "multi_format_parse", "解析这 23 本控制科学教材为结构化语料。", ["corpus_parse", "mineru_parse"], ["corpus_parse", "mineru_parse"]),
    RouterCase("parse_05", "multi_format_parse", "处理一批 Word、PPTX、XLSX 和扫描图片，统一进入科学文档 schema。", ["multi_format_parse", "corpus_parse"], ["multi_format_parse", "corpus_parse"]),
    RouterCase("eval_01", "model_evaluation", "在 ControlSci benchmark 上评测 deepseek-v4-flash 模型。", ["model_evaluate", "quality_arbitrate", "leaderboard_viz"], ["model_evaluate", "quality_arbitrate"]),
    RouterCase("eval_02", "model_evaluation", "对比 API Judge 和本地 Ollama Judge 的评分差异。", ["multi_judge_compare", "model_evaluate"], ["multi_judge_compare", "model_evaluate"]),
    RouterCase("eval_03", "model_evaluation", "基于现有语料生成 500 道评测题，并跑全模型排行榜。", ["benchmark_build", "quality_arbitrate", "model_evaluate", "leaderboard_viz"], ["benchmark_build"]),
    RouterCase("eval_04", "model_evaluation", "评估 qwen3.5:9b 在控制理论题上的表现，并更新排行榜。", ["model_evaluate", "leaderboard_viz"], ["model_evaluate", "quality_arbitrate"]),
    RouterCase("eval_05", "model_evaluation", "对新生成的问题做质量仲裁，再用三个模型跑一遍评测。", ["quality_arbitrate", "model_evaluate"], ["quality_arbitrate"]),
    RouterCase("sciverse_01", "sciverse_search", "用 Sciverse 搜索最新的模型预测控制论文。", ["sciverse_search"], ["sciverse_search"]),
    RouterCase("sciverse_02", "sciverse_search", "检索闭环胰岛素临床试验证据，优先正式出版医学文献。", ["sciverse_search", "medical_rag"], ["sciverse_search", "medical_rag"]),
    RouterCase("sciverse_03", "sciverse_search", "找 2024 年后的卡尔曼滤波前沿文献，不局限 arXiv。", ["sciverse_search", "arxiv_search"], ["sciverse_search"]),
    RouterCase("sciverse_04", "sciverse_search", "从 Sciverse 里找跨学科控制科学论文，返回 top 5。", ["sciverse_search"], ["sciverse_search"]),
    RouterCase("sciverse_05", "sciverse_search", "搜索正式出版物里关于 differentiable MPC 的最新结果。", ["sciverse_search"], ["sciverse_search"]),
]


def _expand_plan_dependencies(plan, registry: IntentRegistry):
    expanded = []
    seen = set()

    def add_with_deps(step):
        for dep_id in step.depends_on:
            if dep_id in seen:
                continue
            dep = registry.get(dep_id)
            if not dep:
                continue
            dep_step = type(step)(
                intent_id=dep_id,
                intent_name=dep.get("name", dep_id),
                description=f"自动补齐依赖: {dep.get('name', dep_id)}",
                tool=", ".join(dep.get("toolchain", [])),
                resource_type=dep.get("resource_type", "script"),
                depends_on=dep.get("depends_on", []),
                parameters={},
                rank=len(expanded) + 1,
            )
            add_with_deps(dep_step)
        if step.intent_id not in seen:
            seen.add(step.intent_id)
            step.rank = len(expanded) + 1
            expanded.append(step)

    for step in plan:
        add_with_deps(step)
    return expanded


def _evaluate_case(case: RouterCase, router: IntentRouter, registry: IntentRegistry, executor: Executor) -> RouterRecord:
    t0 = time.time()
    record = RouterRecord(
        case_id=case.case_id,
        input=case.input,
        expected_family=case.expected_family,
        acceptable_intents=case.acceptable_intents,
        expected_first_intents=case.expected_first_intents,
    )
    try:
        raw_plan = router.plan(case.input)
        expanded_plan = _expand_plan_dependencies(raw_plan, registry)
        raw_ids = [step.intent_id for step in raw_plan]
        expanded_ids = [step.intent_id for step in expanded_plan]
        record.predicted_intents = expanded_ids
        record.predicted_descriptions = [step.description for step in expanded_plan]
        record.dependency_repair = raw_ids != expanded_ids
        record.matched_acceptable = any(intent_id in case.acceptable_intents for intent_id in expanded_ids)
        record.first_intent_match = bool(expanded_ids) and expanded_ids[0] in case.expected_first_intents
        record.intent_chain_match = all(intent_id in expanded_ids for intent_id in case.acceptable_intents if intent_id in raw_ids or intent_id in expanded_ids)
        record.single_intent_degradation = len(expanded_ids) == 1 and len(case.acceptable_intents) > 1
        dry_statuses = []
        for step in expanded_plan:
            log_step, _, error = executor.execute(step, dry_run=True)
            dry_statuses.append(log_step.status if not error else "failed")
        record.dry_run_status = "success" if dry_statuses and all(s == "skipped" for s in dry_statuses) else "failed"
        record.status = "success" if record.matched_acceptable and record.dry_run_status == "success" else "mismatch"
    except Exception as exc:
        record.status = "error"
        record.dry_run_status = "not_run"
        record.error = f"{type(exc).__name__}: {exc}"
    record.duration_ms = int((time.time() - t0) * 1000)
    return record


def _safe_rate(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return round(numerator / denominator, 4)


def _summarize(records: List[RouterRecord]) -> Dict[str, object]:
    total = len(records)
    success = sum(1 for r in records if r.status == "success")
    errors = sum(1 for r in records if r.status == "error")
    dry_success = sum(1 for r in records if r.dry_run_status == "success")
    first_match = sum(1 for r in records if r.first_intent_match)
    acceptable_match = sum(1 for r in records if r.matched_acceptable)
    dependency_repair = sum(1 for r in records if r.dependency_repair)
    single_degradation = sum(1 for r in records if r.single_intent_degradation)
    by_family = {}
    for record in records:
        family = by_family.setdefault(record.expected_family, {"total": 0, "matched": 0, "first_match": 0, "errors": 0})
        family["total"] += 1
        family["matched"] += int(record.matched_acceptable)
        family["first_match"] += int(record.first_intent_match)
        family["errors"] += int(record.status == "error")
    for family in by_family.values():
        family["match_rate"] = _safe_rate(family["matched"], family["total"])
        family["first_intent_accuracy"] = _safe_rate(family["first_match"], family["total"])
    return {
        "total_cases": total,
        "success_cases": success,
        "error_cases": errors,
        "intent_chain_consistency": _safe_rate(acceptable_match, total),
        "first_intent_accuracy": _safe_rate(first_match, total),
        "dry_run_success_rate": _safe_rate(dry_success, total),
        "dependency_repair_rate": _safe_rate(dependency_repair, total),
        "single_intent_degradation_rate": _safe_rate(single_degradation, total),
        "call_count": total,
        "by_family": by_family,
    }


def run(output: str) -> Dict[str, object]:
    registry = IntentRegistry()
    router = IntentRouter(registry)
    executor = Executor(registry=registry)
    records = []
    for idx, case in enumerate(CASES, 1):
        logger.info("[%02d/%02d] %s — %s", idx, len(CASES), case.case_id, case.input[:60])
        record = _evaluate_case(case, router, registry, executor)
        logger.info("  status=%s intents=%s duration=%dms", record.status, record.predicted_intents, record.duration_ms)
        records.append(record)
    payload = {
        "schema_version": "1.0",
        "experiment": "track2_router_robustness_baseline",
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "router_model": "deepseek-v4-flash",
        "prompt_variant": "baseline_existing_agent_cli",
        "api_key_present": bool(os.environ.get("OPENAI_API_KEY") or os.environ.get("DEEPSEEK_API_KEY")),
        "summary": _summarize(records),
        "records": [asdict(record) for record in records],
    }
    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(out_path)
    logger.info("Router robustness report saved: %s", out_path)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate IntentRouter robustness across natural-language variants.")
    parser.add_argument("--output", default="benchmark/eval/results/agent_router_robustness.json")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    payload = run(args.output)
    summary = payload["summary"]
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["error_cases"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
