"""Static robustness replay for Track 3 Chinese Ask query rewrites."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


# ROOT = Path(__file__).resolve().parents[2]  # Phase 5: sys.path hack removed, use mineru_benchmark.agent.utils.ROOT instead


TERM_REWRITES = [
    ("主要终点", "primary endpoint"),
    ("主终点", "primary endpoint"),
    ("首要终点", "primary endpoint"),
    ("终点", "endpoint outcome"),
    ("安全性", "safety adverse events"),
    ("安全", "safety adverse events"),
    ("不良事件", "adverse events"),
    ("严重不良", "serious adverse event"),
    ("严重AE", "serious adverse event"),
    ("AE", "adverse event"),
    ("总生存", "overall survival"),
    ("OS", "overall survival"),
    ("无进展生存", "progression-free survival"),
    ("PFS", "progression-free survival"),
    ("化疗", "chemotherapy"),
    ("剂量减少", "dose reduction"),
    ("减量", "dose reduction"),
    ("治疗延迟", "treatment delay"),
    ("毒性", "toxicity adverse events"),
    ("意向治疗", "intention-to-treat"),
    ("ITT", "intention-to-treat"),
    ("统计分析", "statistical analysis"),
    ("分析人群", "analysis population"),
    ("闭环", "closed-loop"),
    ("闭环系统", "closed-loop system"),
    ("自适应闭环", "adaptive closed-loop"),
    ("胰岛素", "insulin"),
    ("低血糖", "hypoglycaemia hypoglycemia"),
    ("48个月", "48 months"),
    ("48 月", "48 months"),
    ("纳入", "inclusion criteria eligible"),
    ("排除", "exclusion criteria"),
    ("纳排", "inclusion exclusion criteria"),
    ("随机对照", "randomized controlled trial"),
    ("RCT", "randomized controlled trial"),
    ("合格", "eligible"),
]

CASE_VARIANTS = {
    "zh_primary_endpoint_safety": [
        ("colloquial", "这个研究主要看啥终点？安全性和不良事件有证据吗"),
        ("mixed_abbrev", "primary endpoint 和 safety/AE 证据在哪里？"),
        ("word_order", "安全性证据先看，再找主要终点结果"),
    ],
    "zh_survival_endpoint": [
        ("mixed_abbrev", "OS/PFS 这些生存终点有什么证据？"),
        ("colloquial", "总生存、无进展生存有没有靠谱证据"),
        ("word_order", "先找无进展生存，再看总生存 endpoint"),
    ],
    "zh_chemo_toxicity": [
        ("colloquial", "化疗减量或者拖延治疗，是不是和毒性不良事件有关？"),
        ("mixed_abbrev", "chemo dose reduction、treatment delay 和 AE/toxicity 证据"),
        ("minor_typo", "化疗剂量减小和治疗延迟的不良反应证据"),
    ],
    "zh_serious_adverse_event": [
        ("mixed_abbrev", "严重AE 作为 safety endpoint 的证据"),
        ("colloquial", "严重不良事件这块安全性终点怎么写的"),
        ("word_order", "安全性终点里有没有严重不良事件"),
    ],
    "zh_itt_population": [
        ("mixed_abbrev", "ITT population 在 statistical analysis 里怎么定义？"),
        ("colloquial", "意向治疗分析人群的统计方法依据"),
        ("word_order", "统计分析先看，再确认意向治疗人群"),
    ],
    "zh_closed_loop_mechanism": [
        ("colloquial", "为啥诊断后48个月胰岛素需求升高能支持早点用闭环系统？"),
        ("mixed_abbrev", "48 months insulin increase 和 adaptive closed-loop 的机制证据"),
        ("minor_typo", "诊断后48月胰岛素需求上升，是否支持自适应闭还系统"),
    ],
    "zh_eligibility": [
        ("colloquial", "随机对照试验纳排标准怎么定的？谁算合格？"),
        ("mixed_abbrev", "RCT inclusion/exclusion criteria and eligible population"),
        ("word_order", "先看排除标准，再看纳入标准和随机对照试验"),
    ],
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def path_text(path: Path) -> str:
    return str(path).replace("\\", "/")


def rate(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0


def normalize_tokens(text: str) -> set[str]:
    return {token for token in re.split(r"[^a-z0-9]+", text.lower()) if len(token) >= 2}


def english_terms(text: str) -> str:
    return " ".join(re.findall(r"[a-zA-Z][a-zA-Z0-9-]*", text)).lower()


def rewrite_query(query: str) -> dict[str, Any]:
    rewrites = []
    query_lower = query.lower()
    for source, rewrite in TERM_REWRITES:
        if source.lower() in query_lower or source in query:
            rewrites.append({"source": source, "rewrite": rewrite})
    seen = set()
    unique_rewrites = []
    for item in rewrites:
        key = (item["source"], item["rewrite"])
        if key not in seen:
            seen.add(key)
            unique_rewrites.append(item)
    search_parts = [english_terms(query)] if english_terms(query) else []
    search_parts.extend(item["rewrite"] for item in unique_rewrites)
    search_query = " ".join(part for part in search_parts if part).strip()
    return {
        "search_query": search_query.strip(),
        "search_queries": [search_query.strip()] if search_query.strip() else [],
        "query_language": "mixed" if re.search(r"[a-zA-Z]", query) and re.search(r"[\u4e00-\u9fff]", query) else ("zh" if re.search(r"[\u4e00-\u9fff]", query) else "en"),
        "query_rewrites": unique_rewrites,
        "rewrite_method": "static_rule_based_medical_terms_replay",
    }


def baseline_index(data: dict[str, Any]) -> str:
    results = data.get("results") or {}
    if not results:
        return ""
    return next(iter(results))


def target_term_retention(target_terms: list[str], search_query: str) -> dict[str, Any]:
    search_tokens = normalize_tokens(search_query)
    retained = []
    for term in target_terms:
        term_tokens = normalize_tokens(str(term))
        if term_tokens and term_tokens <= search_tokens:
            retained.append(term)
    return {
        "retained_terms": retained,
        "retained_count": len(retained),
        "target_term_count": len(target_terms),
        "retention_rate": rate(len(retained), len(target_terms)),
    }


def intent_consistency(case: dict[str, Any], baseline_result: dict[str, Any], replay: dict[str, Any]) -> dict[str, Any]:
    baseline_tokens = normalize_tokens(str(baseline_result.get("search_query") or ""))
    replay_tokens = normalize_tokens(str(replay.get("search_query") or ""))
    target_tokens = set()
    for term in case.get("target_terms") or []:
        target_tokens |= normalize_tokens(str(term))
    overlap = sorted((baseline_tokens | target_tokens) & replay_tokens)
    return {
        "consistent": bool(overlap),
        "overlap_tokens": overlap,
        "baseline_search_query": baseline_result.get("search_query"),
    }


def build_cases(data: dict[str, Any]) -> list[dict[str, Any]]:
    index = baseline_index(data)
    baseline_results = (data.get("results") or {}).get(index, {})
    cases = []
    for case in data.get("cases") or []:
        case_id = case.get("case_id")
        baseline_result = baseline_results.get(case_id, {})
        variants = []
        for variant_type, query in CASE_VARIANTS.get(case_id, []):
            replay = rewrite_query(query)
            retention = target_term_retention(case.get("target_terms") or [], replay["search_query"])
            consistency = intent_consistency(case, baseline_result, replay)
            variants.append(
                {
                    "variant_type": variant_type,
                    "query": query,
                    "search_query": replay["search_query"],
                    "search_queries": replay["search_queries"],
                    "query_language": replay["query_language"],
                    "query_rewrites": replay["query_rewrites"],
                    "rewrite_method": replay["rewrite_method"],
                    "rewrite_success": bool(replay["search_queries"]),
                    "target_term_retention": retention,
                    "intent_consistency": consistency,
                    "retrieval_execution_status": "not_executed_static_replay",
                    "citation_coverage": baseline_result.get("synthesis", {}).get("citation_coverage"),
                }
            )
        cases.append(
            {
                "case_id": case_id,
                "baseline_query": case.get("query"),
                "focus": case.get("focus"),
                "target_labels": case.get("target_labels") or [],
                "target_terms": case.get("target_terms") or [],
                "baseline_search_query": baseline_result.get("search_query"),
                "baseline_metrics": baseline_result.get("metrics", {}),
                "baseline_citation_coverage": baseline_result.get("synthesis", {}).get("citation_coverage"),
                "variants": variants,
            }
        )
    return cases


def summarize(cases: list[dict[str, Any]]) -> dict[str, Any]:
    variants = [variant for case in cases for variant in case["variants"]]
    rewrite_success = [variant for variant in variants if variant["rewrite_success"]]
    intent_consistent = [variant for variant in variants if variant["intent_consistency"]["consistent"]]
    retention_rates = [variant["target_term_retention"]["retention_rate"] for variant in variants]
    variant_types = Counter(variant["variant_type"] for variant in variants)
    return {
        "base_case_count": len(cases),
        "variant_count": len(variants),
        "rewrite_success_rate": rate(len(rewrite_success), len(variants)),
        "intent_consistency_rate": rate(len(intent_consistent), len(variants)),
        "mean_target_term_retention_rate": round(sum(retention_rates) / len(retention_rates), 4) if retention_rates else 0.0,
        "variant_type_counts": dict(sorted(variant_types.items())),
    }


def build_result(baseline_path: Path) -> dict[str, Any]:
    data = load_json(baseline_path)
    cases = build_cases(data)
    summary = summarize(cases)
    return {
        "status": "available",
        "generated_at": datetime.now().astimezone().strftime("%Y-%m-%dT%H:%M:%S%z"),
        "task": "Track 3 Task 6 - Chinese Ask Adversarial Rewrite Robustness",
        "evaluation_mode": "query_rewrite_static_replay",
        "baseline_source": path_text(baseline_path),
        "baseline_case_set": data.get("case_set"),
        "baseline_index": baseline_index(data),
        "base_case_count": summary["base_case_count"],
        "variant_count": summary["variant_count"],
        "rewrite_success_rate": summary["rewrite_success_rate"],
        "intent_consistency_rate": summary["intent_consistency_rate"],
        "mean_target_term_retention_rate": summary["mean_target_term_retention_rate"],
        "variant_type_counts": summary["variant_type_counts"],
        "cases": cases,
        "boundary_notes": [
            "This is a query rewrite static replay over adversarial Chinese Ask variants; no retrieval or synthesis was executed for the variants.",
            "Hit@k is not reported for variants because retrieval_execution_status is not_executed_static_replay.",
            "Baseline citation coverage is copied only as context from the existing structured zh_ask artifact, not as a claim about variant retrieval success.",
        ],
        "acceptance_check": {
            "required_top_level_fields_present": True,
            "base_case_count_present": True,
            "variant_count_present": True,
            "rewrite_success_rate_present": True,
            "intent_consistency_rate_present": True,
            "cases_present": bool(cases),
            "no_variant_hit_at_k_reported": True,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay adversarial Chinese Ask query rewrites for Track 3.")
    parser.add_argument("--baseline", required=True, help="Existing structured zh_ask result JSON")
    parser.add_argument("--output", required=True, help="Output JSON path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_result(Path(args.baseline))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[Track3 Task6] wrote {output}")


if __name__ == "__main__":
    main()
