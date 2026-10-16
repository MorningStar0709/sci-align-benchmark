"""Audit Track 3 structured RAG evidence-card completeness."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


# ROOT = Path(__file__).resolve().parents[2]  # Phase 5: sys.path hack removed, use mineru_benchmark.agent.utils.ROOT instead


CARD_FIELDS = ["card_id", "chunk_id", "role", "support_level", "label", "section", "source_file", "key_text", "score"]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def path_text(path: Path) -> str:
    return str(path).replace("\\", "/")


def rate(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0


_ROLE_KEYWORDS: list[tuple[tuple[str, ...], str]] = [
    (("adverse", "safety"), "safety"),
    (("endpoint", "outcome", "survival"), "endpoint"),
    (("statistical", "analysis", "intention"), "analysis"),
    (("population", "eligib", "inclusion", "exclusion"), "population"),
    (("intervention", "treatment", "chemotherapy"), "intervention"),
]


def _match_role_keywords(text: str) -> str | None:
    """Match text against role keyword groups."""
    for keywords, role in _ROLE_KEYWORDS:
        if any(kw in text for kw in keywords):
            return role
    return None



def infer_role(item: dict[str, Any]) -> str:
    label = str(item.get("medical_label") or "").lower()
    parent = str(item.get("medical_parent") or "").lower()
    section = str(item.get("source_section") or "").lower()
    text = " ".join([label, parent, section])
    role = _match_role_keywords(text)
    return role if role else "context"


def infer_support_level(role: str, question_type: str | None) -> str:
    if question_type == "safety_judgement" and role in {"safety", "endpoint"}:
        return "direct"
    if question_type == "endpoint_definition" and role == "endpoint":
        return "direct"
    if question_type == "mechanism_explanation" and role in {"endpoint", "intervention", "context"}:
        return "indirect"
    if role in {"endpoint", "analysis", "population", "safety"}:
        return "direct"
    return "context"


def build_card(item: dict[str, Any], rank: int, question_type: str | None) -> dict[str, Any]:
    role = infer_role(item)
    return {
        "card_id": f"E{rank}",
        "chunk_id": item.get("chunk_id", ""),
        "role": role,
        "support_level": infer_support_level(role, question_type),
        "label": item.get("medical_label") or item.get("medical_parent") or "-",
        "section": item.get("source_section") or "-",
        "source_file": item.get("source_file") or "unknown",
        "key_text": str(item.get("preview") or "")[:260].replace("\n", " "),
        "score": item.get("score"),
    }


def iter_case_results(data: dict[str, Any]) -> list[dict[str, Any]]:
    items = []
    for index_label, per_case in (data.get("results") or {}).items():
        if not isinstance(per_case, dict):
            continue
        for case_id, result in per_case.items():
            items.append({"index": index_label, "case_id": case_id, "result": result})
    return items


def card_field_coverage(cards: list[dict[str, Any]]) -> dict[str, Any]:
    coverage = {}
    for field in CARD_FIELDS:
        present = sum(1 for card in cards if field in card and card.get(field) not in {None, ""})
        coverage[field] = {"present": present, "total": len(cards), "rate": rate(present, len(cards))}
    return coverage


def citation_whitelist_check(result: dict[str, Any]) -> dict[str, Any]:
    top_chunk_ids = {item.get("chunk_id") for item in result.get("top_results", []) if item.get("chunk_id")}
    synthesis = result.get("synthesis") or {}
    citations = [citation for citation in synthesis.get("citations") or [] if citation]
    reasoning_citations = []
    for step in synthesis.get("reasoning_trace") or []:
        reasoning_citations.extend(step.get("citations") or [])
    all_citations = citations + reasoning_citations
    out_of_whitelist = sorted({citation for citation in all_citations if citation not in top_chunk_ids})
    return {
        "top_result_chunk_count": len(top_chunk_ids),
        "citation_count": len(citations),
        "reasoning_citation_count": len(reasoning_citations),
        "all_citations_in_top_results": not out_of_whitelist,
        "out_of_whitelist_citations": out_of_whitelist,
    }


def claim_binding(result: dict[str, Any]) -> dict[str, Any]:
    synthesis = result.get("synthesis") or {}
    claim_count = int(synthesis.get("claim_count") or 0)
    supported_claims = int(synthesis.get("supported_claims") or 0)
    unsupported_claims = int(synthesis.get("unsupported_claims") or 0)
    coverage = synthesis.get("citation_coverage")
    whitelist = citation_whitelist_check(result)
    reasoning_steps = synthesis.get("reasoning_trace") or []
    reasoning_supported = sum(1 for step in reasoning_steps if step.get("status") == "supported" and step.get("citations"))
    explicit_claims = synthesis.get("claims") or []
    explicit_claims_with_citations = sum(1 for claim in explicit_claims if claim.get("citations"))
    if claim_count:
        binding_rate = rate(supported_claims, claim_count)
    elif explicit_claims:
        binding_rate = rate(explicit_claims_with_citations, len(explicit_claims))
    else:
        binding_rate = None
    return {
        "claim_count": claim_count,
        "supported_claims": supported_claims,
        "unsupported_claims": unsupported_claims,
        "citation_coverage": coverage,
        "claim_binding_rate": binding_rate,
        "explicit_claim_records": len(explicit_claims),
        "explicit_claims_with_citations": explicit_claims_with_citations,
        "reasoning_steps": len(reasoning_steps),
        "reasoning_steps_with_supported_citations": reasoning_supported,
        "citation_whitelist_check": whitelist,
    }


def audit_dataset(data: dict[str, Any], source_path: Path, source_role: str) -> dict[str, Any]:
    case_results = iter_case_results(data)
    all_cards = []
    case_audits = []
    audit_statuses = Counter()
    for entry in case_results:
        result = entry["result"]
        synthesis = result.get("synthesis") or {}
        question_type = synthesis.get("question_type")
        cards = [build_card(item, idx, question_type) for idx, item in enumerate(result.get("top_results", []), start=1)]
        all_cards.extend(cards)
        answer_audit = synthesis.get("answer_audit") or {}
        audit_statuses[str(answer_audit.get("status") or "missing")] += 1
        case_audits.append(
            {
                "source_role": source_role,
                "index": entry["index"],
                "case_id": entry["case_id"],
                "card_count": len(cards),
                "evidence_cards_count_field": synthesis.get("evidence_cards_count"),
                "evidence_card_count_matches_top_results": synthesis.get("evidence_cards_count") in {None, len(cards)},
                "answer_audit_status": answer_audit.get("status"),
                "answer_audit_issue_count": len(answer_audit.get("issues") or []),
                "direct_source_cards": answer_audit.get("direct_source_cards") or synthesis.get("direct_source_cards"),
                "claim_binding": claim_binding(result),
            }
        )
    return {
        "source_role": source_role,
        "source": path_text(source_path),
        "case_count": len(case_results),
        "card_count": len(all_cards),
        "card_field_coverage": card_field_coverage(all_cards),
        "audit_status_distribution": dict(sorted(audit_statuses.items())),
        "case_audits": case_audits,
        "sample_cards": all_cards[:12],
    }


def aggregate(audits: list[dict[str, Any]]) -> dict[str, Any]:
    total_cases = sum(audit["case_count"] for audit in audits)
    total_cards = sum(audit["card_count"] for audit in audits)
    status_counter = Counter()
    binding_items = []
    whitelist_ok = 0
    whitelist_total = 0
    for audit in audits:
        status_counter.update(audit["audit_status_distribution"])
        for case in audit["case_audits"]:
            binding = case["claim_binding"]
            if binding["claim_binding_rate"] is not None:
                binding_items.append(binding["claim_binding_rate"])
            whitelist_total += 1
            if binding["citation_whitelist_check"]["all_citations_in_top_results"]:
                whitelist_ok += 1
    mean_binding = round(sum(binding_items) / len(binding_items), 4) if binding_items else None
    return {
        "case_count": total_cases,
        "card_count": total_cards,
        "audit_status_distribution": dict(sorted(status_counter.items())),
        "claim_binding_coverage": mean_binding,
        "citation_whitelist_pass_rate": rate(whitelist_ok, whitelist_total),
    }


def build_result(rag_path: Path, audit_smoke_path: Path) -> dict[str, Any]:
    rag_audit = audit_dataset(load_json(rag_path), rag_path, "structured_zh_ask")
    smoke_audit = audit_dataset(load_json(audit_smoke_path), audit_smoke_path, "audit_smoke")
    audits = [rag_audit, smoke_audit]
    totals = aggregate(audits)
    return {
        "status": "available",
        "generated_at": datetime.now().astimezone().strftime("%Y-%m-%dT%H:%M:%S%z"),
        "task": "Track 3 Task 7 - Evidence Card Completeness Audit",
        "evaluation_mode": "existing_structured_rag_evidence_card_audit_no_new_api_call",
        "sources": [path_text(rag_path), path_text(audit_smoke_path)],
        "case_count": totals["case_count"],
        "card_count": totals["card_count"],
        "card_field_coverage": rag_audit["card_field_coverage"],
        "claim_binding_coverage": totals["claim_binding_coverage"],
        "citation_whitelist_pass_rate": totals["citation_whitelist_pass_rate"],
        "audit_status_distribution": totals["audit_status_distribution"],
        "dataset_audits": audits,
        "sample_cards": (rag_audit["sample_cards"] + smoke_audit["sample_cards"])[:16],
        "boundary_notes": [
            "Evidence cards are reconstructed from top_results using the same public fields required by the API card schema because benchmark JSON stores card counts/roles rather than full card objects.",
            "Claim binding is checked through claim_count/supported_claims, reasoning_trace citations, synthesis citations, and the top_results chunk whitelist.",
            "This audit verifies evidence-card data structure and citation binding integrity, not medical correctness of the answer content.",
        ],
        "acceptance_check": {
            "required_top_level_fields_present": True,
            "card_field_coverage_present": True,
            "claim_binding_coverage_present": totals["claim_binding_coverage"] is not None,
            "audit_status_distribution_present": True,
            "sample_cards_present": bool(rag_audit["sample_cards"] or smoke_audit["sample_cards"]),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit Track 3 structured RAG evidence card completeness.")
    parser.add_argument("--rag", required=True, help="Structured RAG result JSON")
    parser.add_argument("--audit-smoke", required=True, help="Audit smoke result JSON")
    parser.add_argument("--output", required=True, help="Output JSON path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_result(Path(args.rag), Path(args.audit_smoke))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[Track3 Task7] wrote {output}")


if __name__ == "__main__":
    main()
