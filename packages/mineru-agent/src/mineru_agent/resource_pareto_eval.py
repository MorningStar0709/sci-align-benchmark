"""Resource scheduling Pareto evaluation for Track 2 Agent evidence."""

import argparse
import json
import logging
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# ROOT = Path(__file__).resolve().parents[2]  # Phase 5: sys.path hack removed, use mineru_benchmark.agent.utils.ROOT instead

from mineru_agent.resource_scheduler import ProviderStatus, ResourceScheduler
from mineru_agent.utils import ROOT


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ResourceParetoEval")


@dataclass
class ParetoCase:
    case_id: str
    task_family: str
    intent_id: str
    decision_goal: str
    candidate_modes: List[str]


@dataclass
class ParetoRecord:
    case_id: str
    task_family: str
    intent_id: str
    decision_goal: str
    mode: str
    provider: str
    model: str
    client_type: str
    provider_type: str
    endpoint_kind: str
    status: str
    scheduler_latency_ms: int
    health_latency_ms: Optional[int]
    call_count: int
    cost_audit: str
    reproducibility_level: str
    data_boundary: str
    cloud_allowed: object
    fallback_provider: str = ""
    failure_boundary: str = ""
    error: str = ""


CASES = [
    ParetoCase(
        "router_planning_01",
        "router_planning",
        "corpus_quality_report",
        "在公开或脱敏语料统计上生成质量诊断，比较 API、local 与 replay 的调度取舍。",
        ["auto", "local", "replay"],
    ),
    ParetoCase(
        "judge_arbitration_01",
        "judge_arbitration",
        "quality_arbitrate",
        "对生成题目做质量仲裁，比较远端 Judge、本地 Judge 与 replay 基线。",
        ["auto", "local", "replay"],
    ),
    ParetoCase(
        "vision_audit_01",
        "vision_audit",
        "cross_modal_audit",
        "审计图片与公式语义一致性，比较 MiMo Vision、本地 Ollama 与 replay 基线。",
        ["auto", "local", "replay"],
    ),
    ParetoCase(
        "formula_recognition_01",
        "formula_recognition",
        "multi_format_parse",
        "处理多格式文档与公式资源，比较 MinerU 本地 API 与 replay 基线。",
        ["auto", "local", "replay"],
    ),
]


PROVIDER_HEALTH_KEY = {
    "deepseek": "deepseek",
    "api": "deepseek",
    "mimo": "mimo_vision",
    "local": "local_chat",
    "mineru": "mineru",
    "script": "script",
    "replay": "replay",
}


PROVIDER_TYPE = {
    "deepseek": "remote_api",
    "api": "remote_api",
    "mimo": "remote_api",
    "minimax": "remote_api",
    "local": "local_api",
    "mineru": "local_api",
    "script": "local_script",
    "replay": "audited_replay",
}


REPRODUCIBILITY = {
    "remote_api": "medium_external_service_dependent",
    "local_api": "high_if_service_available",
    "local_script": "high_local_deterministic",
    "audited_replay": "highest_frozen_artifact",
}


def _provider_type(provider: str) -> str:
    return PROVIDER_TYPE.get(provider, "unknown")


def _reproducibility(provider_type: str) -> str:
    return REPRODUCIBILITY.get(provider_type, "unknown")


def _rate(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0


def _percentile(values: List[int], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * percentile))))
    return round(float(ordered[index]), 2)


def _record_replay(case: ParetoCase) -> ParetoRecord:
    return ParetoRecord(
        case_id=case.case_id,
        task_family=case.task_family,
        intent_id=case.intent_id,
        decision_goal=case.decision_goal,
        mode="replay",
        provider="replay",
        model="frozen_artifact",
        client_type="json_replay",
        provider_type="audited_replay",
        endpoint_kind="replay_baseline",
        status="success",
        scheduler_latency_ms=0,
        health_latency_ms=0,
        call_count=0,
        cost_audit="not_applicable_replay",
        reproducibility_level=_reproducibility("audited_replay"),
        data_boundary="frozen_artifact_only",
        cloud_allowed=False,
        failure_boundary="只能证明既有 trace 可复核，不能代表实时 provider 当前可用性。",
    )


def _record_resolved(case: ParetoCase, mode: str, scheduler: ResourceScheduler, status_cache) -> ParetoRecord:
    t0 = time.time()
    local_mode = mode == "local"
    try:
        resolved = scheduler.resolve(case.intent_id, local_mode=local_mode)
        scheduler_latency_ms = int((time.time() - t0) * 1000)
        health_key = PROVIDER_HEALTH_KEY.get(resolved.provider, resolved.provider)
        health = status_cache.providers.get(health_key)
        health_latency_ms = health.latency_ms if health else None
        if resolved.provider == "script":
            status = "success"
        elif health and health.status in (ProviderStatus.AVAILABLE, ProviderStatus.DEGRADED):
            status = "success"
        else:
            status = "unavailable"
        fallback = scheduler.get_fallback(resolved)
        provider_type = _provider_type(resolved.provider)
        data_policy = resolved.extra_config.get("data_policy", {})
        return ParetoRecord(
            case_id=case.case_id,
            task_family=case.task_family,
            intent_id=case.intent_id,
            decision_goal=case.decision_goal,
            mode=mode,
            provider=resolved.provider,
            model=resolved.model,
            client_type=resolved.client_type,
            provider_type=provider_type,
            endpoint_kind="scheduler_resolve_plus_health_check",
            status=status,
            scheduler_latency_ms=scheduler_latency_ms,
            health_latency_ms=health_latency_ms,
            call_count=1,
            cost_audit="not_available_no_token_or_billing_audit",
            reproducibility_level=_reproducibility(provider_type),
            data_boundary=data_policy.get("boundary", "unknown"),
            cloud_allowed=data_policy.get("cloud_allowed", "unknown"),
            fallback_provider=fallback.provider if fallback else "",
            failure_boundary="本实验只做调度解析和 health probe，不伪装真实模型推理质量。",
        )
    except Exception as exc:
        scheduler_latency_ms = int((time.time() - t0) * 1000)
        return ParetoRecord(
            case_id=case.case_id,
            task_family=case.task_family,
            intent_id=case.intent_id,
            decision_goal=case.decision_goal,
            mode=mode,
            provider="unknown",
            model="",
            client_type="",
            provider_type="unknown",
            endpoint_kind="scheduler_resolve_plus_health_check",
            status="error",
            scheduler_latency_ms=scheduler_latency_ms,
            health_latency_ms=None,
            call_count=1,
            cost_audit="not_available_due_to_error",
            reproducibility_level="unknown",
            data_boundary="unknown",
            cloud_allowed="unknown",
            error=f"{type(exc).__name__}: {exc}",
        )


def evaluate_case(case: ParetoCase, scheduler: ResourceScheduler, status_cache) -> List[ParetoRecord]:
    # Public API
    records = []
    for mode in case.candidate_modes:
        if mode == "replay":
            records.append(_record_replay(case))
        else:
            records.append(_record_resolved(case, mode, scheduler, status_cache))
    return records


def summarize(records: List[ParetoRecord]) -> Dict[str, object]:
    # Public API
    total = len(records)
    successes = sum(1 for record in records if record.status == "success")
    unavailable = sum(1 for record in records if record.status == "unavailable")
    errors = sum(1 for record in records if record.status == "error")
    latencies = [record.health_latency_ms for record in records if isinstance(record.health_latency_ms, int) and record.health_latency_ms > 0]
    by_provider = {}
    by_task_family = {}
    for record in records:
        provider = by_provider.setdefault(record.provider, {"total": 0, "success": 0, "unavailable": 0, "error": 0, "call_count": 0, "latencies_ms": []})
        provider["total"] += 1
        provider["success"] += int(record.status == "success")
        provider["unavailable"] += int(record.status == "unavailable")
        provider["error"] += int(record.status == "error")
        provider["call_count"] += record.call_count
        if isinstance(record.health_latency_ms, int) and record.health_latency_ms > 0:
            provider["latencies_ms"].append(record.health_latency_ms)
        task = by_task_family.setdefault(record.task_family, {"total": 0, "success": 0, "providers": []})
        task["total"] += 1
        task["success"] += int(record.status == "success")
        if record.provider not in task["providers"]:
            task["providers"].append(record.provider)
    for provider in by_provider.values():
        provider["success_rate"] = _rate(provider["success"], provider["total"])
        provider["avg_latency_ms"] = round(sum(provider["latencies_ms"]) / len(provider["latencies_ms"]), 2) if provider["latencies_ms"] else 0.0
        provider["p95_latency_ms"] = _percentile(provider["latencies_ms"], 0.95)
        del provider["latencies_ms"]
    for task in by_task_family.values():
        task["success_rate"] = _rate(task["success"], task["total"])
    return {
        "total_records": total,
        "task_families": len(by_task_family),
        "success_records": successes,
        "unavailable_records": unavailable,
        "error_records": errors,
        "success_rate": _rate(successes, total),
        "total_probe_call_count": sum(record.call_count for record in records),
        "avg_health_latency_ms": round(sum(latencies) / len(latencies), 2) if latencies else 0.0,
        "p95_health_latency_ms": _percentile(latencies, 0.95),
        "cost_amount_audited": False,
        "cost_audit_note": "未发现可交叉审计 token 或账单来源，因此只记录 probe call count、latency、本地/远端分类与可复现性。",
        "by_provider": by_provider,
        "by_task_family": by_task_family,
    }


def run(output: str) -> Dict[str, object]:
    # Public API
    scheduler = ResourceScheduler()
    status_cache = scheduler.health_check(force=True)
    records = []
    for case in CASES:
        logger.info("%s — %s", case.case_id, case.decision_goal)
        case_records = evaluate_case(case, scheduler, status_cache)
        for record in case_records:
            logger.info("  mode=%s provider=%s status=%s", record.mode, record.provider, record.status)
        records.extend(case_records)
    payload = {
        "schema_version": "1.0",
        "experiment": "track2_resource_pareto_eval",
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "evaluation_mode": "non_destructive_scheduler_health_and_replay",
        "scope": {
            "included": ["scheduler resolve", "provider health probe", "fallback availability", "replay baseline"],
            "excluded": ["真实模型质量评测", "不可审计成本金额", "破坏性 provider 故障注入"],
        },
        "resource_status": {
            "checked_at": status_cache.checked_at,
            "available_providers": status_cache.available_providers,
            "degraded_providers": status_cache.degraded_providers,
            "unavailable_providers": status_cache.unavailable_providers,
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
    # Public API
    parser = argparse.ArgumentParser(description="Evaluate resource scheduling Pareto trade-offs across provider modes.")
    parser.add_argument("--output", default=str(ROOT / "eval" / "results" / "agent_resource_pareto.json"))
    return parser


def main() -> int:
    args = build_parser().parse_args()
    payload = run(args.output)
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    return 0 if payload["summary"]["error_records"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
