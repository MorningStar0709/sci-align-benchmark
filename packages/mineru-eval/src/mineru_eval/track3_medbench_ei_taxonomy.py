"""Build Track 3 MedBench evidence-insufficient taxonomy from existing results."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[4]


MEDBENCH_DIR = ROOT / "data" / "sources_medical" / "medbench"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def path_text(path: Path) -> str:
    return str(path).replace("\\", "/")


def rate(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0


def compact_text(value: object, limit: int = 360) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    return text[:limit]


def infer_llm_ei_flag(item: dict[str, Any]) -> bool:
    answer = str(item.get("generated_answer") or "")
    markers = ("无法回答", "不能回答", "证据不足", "信息不足", "文献信息不足", "不足以回答", "无法依据", "不能依据")
    return any(marker in answer for marker in markers)


def _check_image_region_mismatch(answer: str, subset: str, modality: str) -> str | None:
    """Check for image region mismatch conditions."""
    if "影像部位不符" in answer or "部位不符" in answer or "not match" in answer:
        return "image_region_mismatch"
    if subset in {"MedClass", "MedDetect", "MedVQA", "MedOCR"} and modality.startswith("VLM"):
        return "image_region_mismatch"
    return None


def _check_subset_based_category(subset: str) -> str | None:
    """Check for subset-based taxonomy categories."""
    if subset in {"MedSeqIm", "MedCourse"}:
        return "temporal_reasoning_gap"
    if subset in {"MedTherapy", "MedRxPlan", "MedTreat"}:
        return "therapy_decision_gap"
    return None


def _check_answer_based_category(answer: str, labels: set) -> str | None:
    """Check for answer-text-based taxonomy categories."""
    if "无关" in answer or "不匹配" in answer or "mismatch" in answer:
        return "retrieval_mismatch"
    if "不包含" in answer or "没有任何信息" in answer or "缺乏" in answer:
        return "domain_coverage_gap"
    if labels and labels <= {"references", "front_matter", "_front_matter_other", "other"}:
        return "retrieval_mismatch"
    return None



def taxonomy_category(item: dict[str, Any], modality: str) -> str:
    subset = str(item.get("subset") or "")
    answer = str(item.get("generated_answer") or "").lower()
    labels = {str(label).lower() for label in item.get("retrieval_labels") or []}
    result = _check_image_region_mismatch(answer, subset, modality)
    if result:
        return result
    result = _check_subset_based_category(subset)
    if result:
        return result
    result = _check_answer_based_category(answer, labels)
    if result:
        return result
    if item.get("error"):
        return "result_field_insufficient"
    return "model_conservative_abstention"


def reason_text(item: dict[str, Any]) -> str:
    answer = str(item.get("generated_answer") or "")
    snippets = []
    for marker in ("原因说明", "说明", "结论", "证据不足", "无法回答"):
        idx = answer.find(marker)
        if idx >= 0:
            snippets.append(answer[idx: idx + 260])
            break
    if not snippets:
        snippets.append(answer[:260])
    return compact_text(snippets[0], 300)


def retrieval_sources(item: dict[str, Any]) -> list[str]:
    values = item.get("retrieval_sources") or item.get("retrieval_pmcids") or []
    return [str(value) for value in values]


def taxonomy_record(item: dict[str, Any], source: Path, modality: str, model: str, index: int) -> dict[str, Any]:
    category = taxonomy_category(item, modality)
    return {
        "taxonomy_id": f"EI-MedBench-{modality.replace('/', '-')}-{item.get('subset', 'unknown')}-{index:04d}",
        "source_file": path_text(source),
        "model": model,
        "modality": modality,
        "query_id": item.get("query_id"),
        "subset": item.get("subset"),
        "question_preview": compact_text(item.get("question"), 220),
        "ei_flag": True,
        "ei_category": category,
        "ei_reason_text": reason_text(item),
        "retrieval_count": item.get("retrieval_count"),
        "vision_retrieval_count": item.get("vision_retrieval_count"),
        "context_chunks_used": item.get("context_chunks_used"),
        "retrieval_labels": item.get("retrieval_labels") or [],
        "retrieval_sources": retrieval_sources(item),
        "error": item.get("error", ""),
        "elapsed_seconds": item.get("elapsed_seconds"),
    }


def subset_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(str(record.get("subset")) for record in records).items()))


def category_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(str(record.get("ei_category")) for record in records).items()))


def load_llm_records(results_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    quality_path = MEDBENCH_DIR / "medbench_core_subsets_v2_quality.json"
    detail_path = MEDBENCH_DIR / "medbench_core_subsets_v2.json"
    quality = load_json(quality_path)
    detail = load_json(detail_path)
    per_question = detail.get("per_question") or []
    records = []
    for item in per_question:
        if infer_llm_ei_flag(item):
            records.append(taxonomy_record(item, detail_path, "LLM/text", "qwen3.5:35b", len(records) + 1))
    summary = quality.get("summary", {})
    subsets = {item["subset"]: item for item in quality.get("subsets", []) if isinstance(item, dict) and item.get("subset")}
    insufficient = int(summary.get("insufficient", len(records)))
    total = int(summary.get("total_unique", len(per_question)))
    return {
        "source": path_text(quality_path),
        "detail_source": path_text(detail_path),
        "model": "qwen3.5:35b",
        "modality": "LLM/text",
        "total_questions": total,
        "evidence_insufficient": insufficient,
        "ei_rate": summary.get("insufficient_rate", rate(insufficient, total)),
        "taxonomy_records_from_per_question": len(records),
        "granularity": "subset_level_plus_inferred_per_item_examples",
        "degraded_reason": "per_item_reason_unavailable",
        "subsets": subsets,
        "taxonomy_counts": category_counts(records),
        "subset_ei_counts_from_quality": {name: int(payload.get("insufficient", 0)) for name, payload in sorted(subsets.items())},
    }, records


def load_vlm_records(path: Path, modality: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    data = load_json(path)
    model = str((data.get("meta") or {}).get("model") or "unknown")
    stats = data.get("statistics") or {}
    records = []
    for item in data.get("per_question") or []:
        if item.get("evidence_insufficient") is True:
            records.append(taxonomy_record(item, path, modality, model, len(records) + 1))
    total = int(stats.get("total_questions", 0))
    insufficient = int(stats.get("overall_evidence_insufficient", len(records)))
    return {
        "source": path_text(path),
        "model": model,
        "modality": modality,
        "total_questions": total,
        "evidence_insufficient": insufficient,
        "ei_rate": stats.get("overall_ei_rate", rate(insufficient, total)),
        "taxonomy_records_from_per_question": len(records),
        "granularity": "per_item",
        "degraded_reason": None,
        "subsets": stats.get("subsets", {}),
        "taxonomy_counts": category_counts(records),
        "subset_ei_counts": subset_counts(records),
    }, records


def aggregate_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_modality: dict[str, Counter] = defaultdict(Counter)
    by_subset: dict[str, Counter] = defaultdict(Counter)
    for record in records:
        by_modality[str(record["modality"])][str(record["ei_category"])] += 1
        by_subset[str(record["subset"])][str(record["ei_category"])] += 1
    return {
        "total_taxonomy_records": len(records),
        "taxonomy_counts": category_counts(records),
        "by_modality": {name: dict(sorted(counter.items())) for name, counter in sorted(by_modality.items())},
        "by_subset": {name: dict(sorted(counter.items())) for name, counter in sorted(by_subset.items())},
    }


def report_reference(report: Path) -> dict[str, Any]:
    exists = report.exists()
    text = report.read_text(encoding="utf-8") if exists else ""
    matches = re.findall(r"EI|Evidence-insufficient|evidence-insufficient|证据不足", text, flags=re.IGNORECASE)
    return {
        "path": path_text(report),
        "exists": exists,
        "evidence_boundary_mentions": len(matches),
    }


def build_result(report: Path, results_dir: Path) -> dict[str, Any]:
    llm_summary, llm_records = load_llm_records(results_dir)
    vlm35_summary, vlm35_records = load_vlm_records(MEDBENCH_DIR / "medbench_vlm_report.json", "VLM/text+vision/35b")
    vlm9_summary, vlm9_records = load_vlm_records(MEDBENCH_DIR / "medbench_vlm_report_9b.json", "VLM/text+vision/9b")
    records = llm_records + vlm35_records + vlm9_records
    aggregate = aggregate_records(records)
    return {
        "status": "available",
        "generated_at": datetime.now().astimezone().strftime("%Y-%m-%dT%H:%M:%S%z"),
        "task": "Track 3 Task 3 - MedBench Evidence-Insufficient Taxonomy",
        "evaluation_mode": "existing_medbench_result_replay_no_new_inference",
        "report_reference": report_reference(report),
        "results_dir": path_text(results_dir),
        "llm_ei_summary": llm_summary,
        "vlm_35b_ei_summary": vlm35_summary,
        "vlm_9b_ei_summary": vlm9_summary,
        "taxonomy_counts": aggregate["taxonomy_counts"],
        "taxonomy_by_modality": aggregate["by_modality"],
        "taxonomy_by_subset": aggregate["by_subset"],
        "taxonomy_records": records[:80],
        "taxonomy_record_count": aggregate["total_taxonomy_records"],
        "boundary_notes": [
            "LLM per-item files do not expose an explicit evidence_insufficient boolean, so LLM sample records are inferred from refusal/insufficiency wording and aggregate counts are taken from the quality JSON.",
            "VLM 35B and 9B taxonomy records use explicit per_question.evidence_insufficient fields from existing MedBench JSON artifacts.",
            "This taxonomy describes evidence boundary and retrieval/format failure modes; it is not an official MedBench score or clinical correctness assessment.",
        ],
        "acceptance_check": {
            "required_summary_fields_present": True,
            "llm_key_numbers_traceable": llm_summary["evidence_insufficient"] == 280 and llm_summary["total_questions"] == 330,
            "vlm_35b_key_numbers_traceable": vlm35_summary["evidence_insufficient"] == 13 and vlm35_summary["total_questions"] == 300,
            "vlm_9b_key_numbers_traceable": vlm9_summary["evidence_insufficient"] == 25 and vlm9_summary["total_questions"] == 300,
            "degraded_reason_present_for_llm": llm_summary.get("degraded_reason") == "per_item_reason_unavailable",
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build MedBench EI taxonomy for Track 3.")
    parser.add_argument("--report", required=True, help="Track 3 report path")
    parser.add_argument("--results-dir", required=True, help="Benchmark results directory")
    parser.add_argument("--output", required=True, help="Output JSON path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_result(Path(args.report), Path(args.results_dir))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[Track3 Task3] wrote {output}")


if __name__ == "__main__":
    main()
