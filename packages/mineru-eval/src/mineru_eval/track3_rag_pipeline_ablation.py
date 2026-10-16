"""Summarize Track 3 Medical RAG pipeline ablation evidence from existing artifacts."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[4]


VISION_AB_DEFAULT = ROOT / "benchmark" / "eval" / "results" / "medical" / "vision_ab_comparison.json"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def path_text(path: Path) -> str:
    return str(path).replace("\\", "/")


def rate(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0


def mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 4) if values else None


def available_source(path: Path, role: str) -> dict[str, Any]:
    return {
        "role": role,
        "path": path_text(path),
        "exists": path.exists(),
        "size_bytes": path.stat().st_size if path.exists() else 0,
    }


def metric_summary(results: dict[str, Any]) -> dict[str, Any]:
    cases = list(results.values())
    hit_count = sum(1 for item in cases if item.get("metrics", {}).get("hit_at_k"))
    label_hit_count = sum(1 for item in cases if item.get("metrics", {}).get("label_hit_at_k"))
    first_ranks = [item.get("metrics", {}).get("first_hit_rank") for item in cases if item.get("metrics", {}).get("first_hit_rank")]
    label_ranks = [item.get("metrics", {}).get("first_label_hit_rank") for item in cases if item.get("metrics", {}).get("first_label_hit_rank")]
    bridged_cases = sum(1 for item in cases if item.get("rewrite_method") not in {None, "identity"})
    multi_query_cases = sum(1 for item in cases if len(item.get("search_queries") or []) > 1)
    modes = sorted({top.get("mode") for item in cases for top in item.get("top_results", []) if top.get("mode")})
    return {
        "cases": len(cases),
        "hit_at_k": hit_count,
        "hit_rate": rate(hit_count, len(cases)),
        "label_hit_at_k": label_hit_count,
        "label_hit_rate": rate(label_hit_count, len(cases)),
        "mean_first_hit_rank": mean([float(rank) for rank in first_ranks]),
        "mean_first_label_hit_rank": mean([float(rank) for rank in label_ranks]),
        "bridged_cases": bridged_cases,
        "multi_query_cases": multi_query_cases,
        "retrieval_modes": modes,
    }


def retrieval_provider_ablation(data: dict[str, Any], source: Path) -> dict[str, Any]:
    summaries = data.get("summary", {})
    results = data.get("results", {})
    indexes = {item.get("label"): item for item in data.get("indexes", []) if isinstance(item, dict)}
    providers = []
    for label in sorted(results):
        index_info = indexes.get(label, {})
        summary = summaries.get(label) if isinstance(summaries.get(label), dict) else metric_summary(results[label])
        providers.append(
            {
                "label": label,
                "embedding_provider": index_info.get("embedding_provider"),
                "embedding_model": index_info.get("embedding_model"),
                "embedding_dim": index_info.get("embedding_dim"),
                "query_embed_seconds": index_info.get("query_embed_seconds"),
                "summary": summary,
            }
        )
    ranked = sorted(
        providers,
        key=lambda item: (
            -float(item["summary"].get("hit_rate", 0)),
            -float(item["summary"].get("label_hit_rate", 0)),
            item["summary"].get("mean_first_hit_rank") or 999,
            item["label"],
        ),
    )
    best = ranked[0] if ranked else None
    worst = ranked[-1] if ranked else None
    return {
        "status": "available" if providers else "missing_results",
        "source": path_text(source),
        "case_set": data.get("case_set"),
        "top_k": data.get("k"),
        "provider_count": len(providers),
        "providers": providers,
        "best_provider": best,
        "worst_provider": worst,
        "interpretation": "Retrieval-provider ablation replays the same zh_ask cases across available indexes; it isolates embedding/index choice while keeping RAG synthesis settings fixed in the source artifact.",
    }


def collect_synthesis_cases(data: dict[str, Any]) -> list[dict[str, Any]]:
    cases = []
    for index_label, per_case in data.get("results", {}).items():
        if not isinstance(per_case, dict):
            continue
        for case_id, item in per_case.items():
            synthesis = item.get("synthesis") or {}
            if synthesis:
                cases.append({"index": index_label, "case_id": case_id, "synthesis": synthesis})
    return cases


def answer_audit_ablation(data: dict[str, Any], source: Path) -> dict[str, Any]:
    cases = collect_synthesis_cases(data)
    passed = [case for case in cases if (case["synthesis"].get("answer_audit") or {}).get("status") == "passed"]
    citation_coverages = [float(case["synthesis"].get("citation_coverage")) for case in cases if isinstance(case["synthesis"].get("citation_coverage"), (int, float))]
    evidence_counts = [int(case["synthesis"].get("evidence_cards_count")) for case in cases if isinstance(case["synthesis"].get("evidence_cards_count"), int)]
    claim_counts = [int(case["synthesis"].get("claim_count")) for case in cases if isinstance(case["synthesis"].get("claim_count"), int)]
    supported_claims = [int(case["synthesis"].get("supported_claims")) for case in cases if isinstance(case["synthesis"].get("supported_claims"), int)]
    return {
        "status": "available" if cases else "missing_synthesis",
        "source": path_text(source),
        "case_count": len(cases),
        "answer_audit_passed_cases": len(passed),
        "answer_audit_pass_rate": rate(len(passed), len(cases)),
        "mean_citation_coverage": mean(citation_coverages),
        "mean_evidence_cards_count": mean([float(value) for value in evidence_counts]),
        "total_claims": sum(claim_counts),
        "supported_claims": sum(supported_claims),
        "claim_support_rate": rate(sum(supported_claims), sum(claim_counts)),
        "sample_cases": [
            {
                "index": case["index"],
                "case_id": case["case_id"],
                "evidence_cards_count": case["synthesis"].get("evidence_cards_count"),
                "citation_coverage": case["synthesis"].get("citation_coverage"),
                "answer_audit_status": (case["synthesis"].get("answer_audit") or {}).get("status"),
            }
            for case in cases[:5]
        ],
        "interpretation": "This is an audit/evidence-card closure replay, not a clinical correctness claim; it verifies that generated claims remain tied to retrieved evidence in the source smoke artifact.",
    }


def vision_ablation(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "status": "source_not_found",
            "source": path_text(path),
            "reason": "vision A/B comparison artifact was not found; no vision numbers were fabricated",
        }
    data = load_json(path)
    items = []
    for query, payload in data.items():
        if not isinstance(payload, dict):
            continue
        text_top5 = payload.get("text_top5") or []
        vision_top5_ids = payload.get("vision_top5_ids") or []
        combined_top5 = payload.get("combined_top5") or []
        combined_vision_hits = sum(1 for item in combined_top5 if isinstance(item, list) and str(item[0]).startswith("vision_"))
        items.append(
            {
                "query": query,
                "text_top5_count": len(text_top5),
                "vision_top5_count": len(vision_top5_ids),
                "combined_top5_count": len(combined_top5),
                "combined_vision_items": combined_vision_hits,
                "vision_in_text_top5": bool(payload.get("vision_in_text_top5")),
                "vision_in_combined": bool(payload.get("vision_in_combined")),
            }
        )
    combined_cases = sum(1 for item in items if item["combined_vision_items"] > 0)
    return {
        "status": "available" if items else "empty_source",
        "source": path_text(path),
        "case_count": len(items),
        "combined_has_vision_cases": combined_cases,
        "combined_has_vision_rate": rate(combined_cases, len(items)),
        "mean_combined_vision_items": mean([float(item["combined_vision_items"]) for item in items]),
        "sample_cases": items[:5],
        "interpretation": "Vision A/B evidence is summarized only from the existing comparison artifact; it does not imply full multimodal clinical validation.",
    }


def chunking_ablation() -> dict[str, Any]:
    return {
        "status": "not_available",
        "reason": "No fixed-chunking baseline artifact is present in the Task 1 input set, so semantic chunking cannot be isolated against a fixed chunk baseline here.",
        "planned_follow_up": "Task 5 can build a small semantic slicing challenge set without retroactively inventing a fixed-chunking benchmark.",
    }


def build_result(rag_results: Path, audit_smoke: Path, vision_ab: Path) -> dict[str, Any]:
    if not rag_results.exists():
        raise FileNotFoundError(f"RAG results not found: {rag_results}")
    if not audit_smoke.exists():
        raise FileNotFoundError(f"audit smoke not found: {audit_smoke}")
    rag_data = load_json(rag_results)
    audit_data = load_json(audit_smoke)
    retrieval = retrieval_provider_ablation(rag_data, rag_results)
    audit = answer_audit_ablation(audit_data, audit_smoke)
    vision = vision_ablation(vision_ab)
    chunking = chunking_ablation()
    available_dimensions = [
        name
        for name, payload in {
            "retrieval_provider_ablation": retrieval,
            "answer_audit_ablation": audit,
            "vision_ablation": vision,
        }.items()
        if payload.get("status") == "available"
    ]
    status = "available" if available_dimensions else "insufficient_source_artifacts"
    return {
        "status": status,
        "generated_at": datetime.now().astimezone().strftime("%Y-%m-%dT%H:%M:%S%z"),
        "task": "Track 3 Task 1 - RAG Pipeline Ablation",
        "evaluation_mode": "existing_artifact_replay_no_new_api_call",
        "sources": [
            available_source(rag_results, "retrieval provider ablation"),
            available_source(audit_smoke, "answer audit and evidence-card closure"),
            available_source(vision_ab, "vision A/B comparison"),
        ],
        "retrieval_provider_ablation": retrieval,
        "answer_audit_ablation": audit,
        "vision_ablation": vision,
        "chunking_ablation": chunking,
        "boundary_notes": [
            "This Task 1 output addresses the report gap with replay-based pipeline evidence from existing JSON artifacts rather than a newly trained or newly inferred benchmark.",
            "Retrieval-provider and answer-audit dimensions are available from actual JSON results; fixed chunking remains unavailable and is explicitly reported as a boundary.",
            "Reported rates describe evidence traceability and pipeline behavior, not clinical efficacy or medical correctness.",
        ],
        "acceptance_check": {
            "required_top_level_fields_present": True,
            "has_at_least_one_actual_json_ablation_dimension": bool(available_dimensions),
            "available_dimensions": available_dimensions,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize Track 3 Medical RAG pipeline ablation evidence.")
    parser.add_argument("--rag-results", required=True, help="Existing all-index RAG result JSON")
    parser.add_argument("--audit-smoke", required=True, help="Existing answer-audit smoke JSON")
    parser.add_argument("--vision-ab", default=str(VISION_AB_DEFAULT), help="Optional existing vision A/B comparison JSON")
    parser.add_argument("--output", required=True, help="Output JSON path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_result(Path(args.rag_results), Path(args.audit_smoke), Path(args.vision_ab))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[Track3 Task1] wrote {output}")


if __name__ == "__main__":
    main()
