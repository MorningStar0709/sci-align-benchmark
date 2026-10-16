"""A/B evaluation for multi-source selection: fixed rules vs Agent Router vs Oracle."""

import argparse
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List

# ROOT = Path(__file__).resolve().parents[2]  # Phase 5: sys.path hack removed, use mineru_benchmark.agent.utils.ROOT instead

from mineru_agent.agent_cli import IntentRouter


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("SourceSelectionAblation")


SOURCE_FROM_INTENT = {
    "arxiv_search": "arXiv",
    "sciverse_search": "Sciverse",
    "medical_rag": "PMC_local",
    "mineru_parse": "LocalCorpus",
    "corpus_parse": "LocalCorpus",
    "multi_format_parse": "LocalCorpus",
    "benchmark_build": "BenchmarkPipeline",
    "quality_arbitrate": "BenchmarkPipeline",
    "model_evaluate": "BenchmarkPipeline",
    "leaderboard_viz": "BenchmarkPipeline",
    "multi_judge_compare": "BenchmarkPipeline",
}


@dataclass
class QueryCase:
    case_id: str
    category: str
    query: str
    oracle_source: str
    oracle_reason: str
    privacy_sensitive: bool = False


@dataclass
class StrategyDecision:
    source: str
    reason: str
    selected_intents: List[str] = field(default_factory=list)
    latency_ms: int = 0
    error: str = ""


@dataclass
class AblationRecord:
    case_id: str
    category: str
    query: str
    oracle_source: str
    oracle_reason: str
    privacy_sensitive: bool
    fixed_rule: StrategyDecision
    agent_router: StrategyDecision
    oracle: StrategyDecision
    fixed_rule_match: bool
    agent_match: bool
    fixed_rule_privacy_violation: bool
    agent_privacy_violation: bool


CASES = [
    QueryCase("preprint_01", "frontier_preprint", "检索 2026 年 control barrier function safety filter 的最新预印本", "arXiv", "明确要求最新预印本，优先 arXiv。"),
    QueryCase("preprint_02", "frontier_preprint", "搜索 differentiable MPC 2025 arXiv papers", "arXiv", "查询直接指定 arXiv 与前沿年份。"),
    QueryCase("preprint_03", "frontier_preprint", "找 safe reinforcement learning control 2025 未发表论文", "arXiv", "未发表前沿论文更适合预印本源。"),
    QueryCase("preprint_04", "frontier_preprint", "发现 2024 之后控制科学新论文，后续加入评测基准", "arXiv", "数据飞轮的前沿发现入口优先 arXiv。"),
    QueryCase("published_01", "published_crossdisciplinary", "查找 Lyapunov stability theorem 的教材和正式出版综述", "Sciverse", "教材、手册和正式出版综述优先 Sciverse。"),
    QueryCase("published_02", "published_crossdisciplinary", "搜索 Kalman filter navigation GPS 的跨学科正式出版物", "Sciverse", "跨学科正式出版物覆盖面优先 Sciverse。"),
    QueryCase("published_03", "published_crossdisciplinary", "找 AI 控制交叉领域的会议和期刊结果", "Sciverse", "会议、期刊与跨学科覆盖需要 Sciverse。"),
    QueryCase("published_04", "published_crossdisciplinary", "检索 model predictive control industrial applications review", "Sciverse", "工业应用综述与正式出版文献优先 Sciverse。"),
    QueryCase("medical_01", "medical_clinical", "检索闭环胰岛素临床试验证据", "Sciverse", "公开临床证据检索优先正式出版医学文献。"),
    QueryCase("medical_02", "medical_clinical", "closed loop artificial pancreas insulin glucose control clinical trial", "Sciverse", "公开英文临床试验查询适合 Sciverse。"),
    QueryCase("medical_03", "medical_clinical", "在本地医疗文献库里合成闭环胰岛素证据，不要上传原文", "PMC_local", "本地医疗文献与原文边界要求触发本地医疗 RAG。", True),
    QueryCase("medical_04", "medical_clinical", "用私有医院资料做 glucose control 文献问答", "PMC_local", "私有医疗资料必须本地处理。", True),
    QueryCase("private_01", "private_local", "解析企业内部控制系统报告，不能上传到云端", "LocalCorpus", "私有企业原文应走本地解析。", True),
    QueryCase("private_02", "private_local", "把本地 Word、PPTX、XLSX 统一转成科学文档 schema", "LocalCorpus", "本地多格式文档解析属于 LocalCorpus。", True),
    QueryCase("private_03", "private_local", "审计这批私有 PDF 的公式和图片对应关系", "LocalCorpus", "私有 PDF 原文与图像审计应保持本地语料边界。", True),
    QueryCase("private_04", "private_local", "把公司技术手册解析成 Markdown 并生成质量报告", "LocalCorpus", "公司手册原文优先本地解析和本地派生统计。", True),
    QueryCase("benchmark_01", "benchmark_pipeline", "基于现有语料生成 500 道评测题并跑模型评测", "BenchmarkPipeline", "出题、仲裁、评测属于 benchmark pipeline。"),
    QueryCase("benchmark_02", "benchmark_pipeline", "对新生成的问题做质量仲裁，再评测三个模型", "BenchmarkPipeline", "质量仲裁与模型评测不是外部检索源。"),
    QueryCase("benchmark_03", "benchmark_pipeline", "更新控制科学排行榜并生成可视化", "BenchmarkPipeline", "排行榜和可视化属于本地评测结果聚合。"),
    QueryCase("benchmark_04", "benchmark_pipeline", "对比 API Judge 和本地 Ollama Judge 的评分差异", "BenchmarkPipeline", "Judge 对比属于评测执行链路。"),
]


_FIXED_RULES: list[tuple[list[str], str, str]] = [
    (["arxiv", "预印本", "未发表"], "arXiv", "关键词规则命中 arXiv / 预印本 / 未发表。"),
    (["医院", "私有", "内部", "不能上传", "本地"], "LocalCorpus", "关键词规则命中私有 / 内部 / 本地 / 不能上传。"),
    (["clinical", "trial", "医疗", "胰岛素"], "Sciverse", "关键词规则命中医学或临床，选择正式出版文献。"),
    (["评测", "排行榜", "judge", "仲裁"], "BenchmarkPipeline", "关键词规则命中评测 / 排行榜 / Judge / 仲裁。"),
]



def fixed_rule_select(query: str) -> StrategyDecision:
    q = query.lower()
    for tokens, source, reason in _FIXED_RULES:
        if any(token in q for token in tokens):
            return StrategyDecision(source, reason)
    return StrategyDecision("Sciverse", "默认选择覆盖面更广的 Sciverse。")


def agent_select(query: str, router: IntentRouter) -> StrategyDecision:
    t0 = time.time()
    try:
        plan = router.plan(query)
        intents = [step.intent_id for step in plan]
        for intent_id in intents:
            source = SOURCE_FROM_INTENT.get(intent_id)
            if source:
                return StrategyDecision(source, f"Router 首个可映射 intent: {intent_id}", intents, int((time.time() - t0) * 1000))
        return StrategyDecision("Unknown", "Router 计划中无可映射 source intent。", intents, int((time.time() - t0) * 1000))
    except Exception as exc:
        return StrategyDecision("Error", "Router 调用失败。", [], int((time.time() - t0) * 1000), f"{type(exc).__name__}: {exc}")


def is_privacy_violation(case: QueryCase, source: str) -> bool:
    return case.privacy_sensitive and source in {"arXiv", "Sciverse"}


def evaluate_case(case: QueryCase, router: IntentRouter) -> AblationRecord:
    fixed = fixed_rule_select(case.query)
    agent = agent_select(case.query, router)
    oracle = StrategyDecision(case.oracle_source, case.oracle_reason)
    return AblationRecord(
        case_id=case.case_id,
        category=case.category,
        query=case.query,
        oracle_source=case.oracle_source,
        oracle_reason=case.oracle_reason,
        privacy_sensitive=case.privacy_sensitive,
        fixed_rule=fixed,
        agent_router=agent,
        oracle=oracle,
        fixed_rule_match=fixed.source == case.oracle_source,
        agent_match=agent.source == case.oracle_source,
        fixed_rule_privacy_violation=is_privacy_violation(case, fixed.source),
        agent_privacy_violation=is_privacy_violation(case, agent.source),
    )


def _rate(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0


def summarize(records: List[AblationRecord]) -> Dict[str, object]:
    total = len(records)
    fixed_matches = sum(1 for r in records if r.fixed_rule_match)
    agent_matches = sum(1 for r in records if r.agent_match)
    fixed_privacy = sum(1 for r in records if r.fixed_rule_privacy_violation)
    agent_privacy = sum(1 for r in records if r.agent_privacy_violation)
    agent_errors = sum(1 for r in records if r.agent_router.source == "Error")
    by_category = {}
    for record in records:
        item = by_category.setdefault(record.category, {"total": 0, "fixed_matches": 0, "agent_matches": 0, "fixed_privacy_violations": 0, "agent_privacy_violations": 0})
        item["total"] += 1
        item["fixed_matches"] += int(record.fixed_rule_match)
        item["agent_matches"] += int(record.agent_match)
        item["fixed_privacy_violations"] += int(record.fixed_rule_privacy_violation)
        item["agent_privacy_violations"] += int(record.agent_privacy_violation)
    for item in by_category.values():
        item["fixed_rule_accuracy"] = _rate(item["fixed_matches"], item["total"])
        item["agent_accuracy"] = _rate(item["agent_matches"], item["total"])
    latencies = [r.agent_router.latency_ms for r in records if r.agent_router.latency_ms]
    return {
        "total_queries": total,
        "fixed_rule_accuracy": _rate(fixed_matches, total),
        "agent_accuracy": _rate(agent_matches, total),
        "agent_minus_fixed_accuracy": round(_rate(agent_matches, total) - _rate(fixed_matches, total), 4),
        "fixed_rule_privacy_violation_count": fixed_privacy,
        "agent_privacy_violation_count": agent_privacy,
        "agent_error_count": agent_errors,
        "agent_avg_latency_ms": round(sum(latencies) / len(latencies), 2) if latencies else 0.0,
        "by_category": by_category,
    }


def run(output: str) -> Dict[str, object]:
    router = IntentRouter()
    records = []
    for idx, case in enumerate(CASES, 1):
        logger.info("[%02d/%02d] %s — %s", idx, len(CASES), case.case_id, case.query[:60])
        record = evaluate_case(case, router)
        logger.info("  fixed=%s agent=%s oracle=%s", record.fixed_rule.source, record.agent_router.source, record.oracle_source)
        records.append(record)
    payload = {
        "schema_version": "1.0",
        "experiment": "track2_source_selection_ablation",
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "oracle_policy": {
            "priority_1": "隐私与原文边界优先，私有/本地/不可上传任务不得外发。",
            "priority_2": "明确前沿预印本或 arXiv 需求优先 arXiv。",
            "priority_3": "正式出版、教材、手册、跨学科覆盖优先 Sciverse。",
            "priority_4": "出题、仲裁、模型评测、排行榜进入 BenchmarkPipeline。",
        },
        "summary": summarize(records),
        "records": [asdict(record) for record in records],
    }
    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(out_path)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare fixed rules, Agent Router, and Oracle for source selection.")
    parser.add_argument("--output", default="benchmark/eval/results/agent_source_selection_ablation.json")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    payload = run(args.output)
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    return 0 if payload["summary"]["agent_error_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
