"""Integrity and traceability audit for the Track 1 Sci-Align core dataset."""

import argparse
import json
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


# ROOT = Path(__file__).resolve().parents[2]  # Phase 5: sys.path hack removed, use mineru_benchmark.agent.utils.ROOT instead


REQUIRED_FIELDS = [
    "id",
    "question",
    "answer",
    "dimension",
    "difficulty_level",
    "control_category",
    "reasoning_steps",
    "source_ref",
    "consistency_status",
]

VALID_DIMENSIONS = {"A", "B", "C", "D"}


@dataclass
class Issue:
    sample_id: str
    issue_type: str
    field: str
    detail: str


@dataclass
class LinkageRecord:
    sample_id: str
    source_ref: str
    resolved: bool
    resolution_method: str
    chunk_path: str = ""
    image_count: int = 0
    formula_count: int = 0
    cooccurrence_type: str = ""


def load_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def questions(data: Dict[str, object]) -> List[Dict[str, object]]:
    items = data.get("questions", [])
    if not isinstance(items, list):
        raise ValueError("dataset.questions must be a list")
    return items


def is_present(value: object) -> bool:
    return value not in (None, "", [])


def rate(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0


def build_index(multimodal: Dict[str, object]) -> Dict[str, Dict[str, Dict[str, object]]]:
    entries = multimodal.get("entries", [])
    by_id = {}
    by_ref = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        qid = entry.get("question_id")
        ref = entry.get("source_ref")
        if qid:
            by_id[str(qid)] = entry
        if ref:
            by_ref[str(ref)] = entry
    return {"by_id": by_id, "by_ref": by_ref}


def required_field_coverage(items: List[Dict[str, object]], issues: List[Issue]) -> Dict[str, object]:
    total = len(items)
    by_field = {}
    complete = 0
    for item in items:
        sample_id = str(item.get("id", "<missing-id>"))
        has_all = True
        for field in REQUIRED_FIELDS:
            ok = is_present(item.get(field))
            by_field.setdefault(field, 0)
            by_field[field] += int(ok)
            has_all = has_all and ok
            if not ok:
                issues.append(Issue(sample_id, "missing_required_field", field, "required field is empty or missing"))
        complete += int(has_all)
    return {
        "complete_records": complete,
        "complete_rate": rate(complete, total),
        "by_field": {field: {"present": count, "coverage": rate(count, total)} for field, count in by_field.items()},
    }


def unique_id_rate(items: List[Dict[str, object]], issues: List[Issue]) -> Dict[str, object]:
    ids = [str(item.get("id", "")) for item in items]
    counts = Counter(ids)
    duplicate_ids = sorted(sample_id for sample_id, count in counts.items() if count > 1 and sample_id)
    for sample_id in duplicate_ids:
        issues.append(Issue(sample_id, "duplicate_id", "id", f"appears {counts[sample_id]} times"))
    unique_count = sum(1 for sample_id, count in counts.items() if sample_id and count == 1)
    return {"unique_ids": unique_count, "duplicate_ids": duplicate_ids, "rate": rate(unique_count, len(items))}


def dimension_validity_rate(items: List[Dict[str, object]], issues: List[Issue]) -> Dict[str, object]:
    valid = 0
    distribution = Counter()
    for item in items:
        sample_id = str(item.get("id", "<missing-id>"))
        dimension = item.get("dimension")
        distribution[str(dimension)] += 1
        if dimension in VALID_DIMENSIONS:
            valid += 1
        else:
            issues.append(Issue(sample_id, "invalid_dimension", "dimension", f"dimension={dimension}"))
    return {"valid": valid, "rate": rate(valid, len(items)), "distribution": dict(sorted(distribution.items()))}


def reasoning_steps_coverage(items: List[Dict[str, object]], issues: List[Issue]) -> Dict[str, object]:
    count = 0
    for item in items:
        sample_id = str(item.get("id", "<missing-id>"))
        value = item.get("reasoning_steps")
        ok = isinstance(value, list) and len(value) > 0 and all(isinstance(step, str) and step.strip() for step in value)
        count += int(ok)
        if not ok:
            issues.append(Issue(sample_id, "invalid_reasoning_steps", "reasoning_steps", "reasoning_steps must be a non-empty list of strings"))
    return {"present": count, "coverage": rate(count, len(items))}


def find_entry(item: Dict[str, object], index: Dict[str, Dict[str, Dict[str, object]]]) -> tuple[Optional[Dict[str, object]], str]:
    sample_id = str(item.get("id", ""))
    source_ref = str(item.get("source_ref", ""))
    if sample_id and sample_id in index["by_id"]:
        return index["by_id"][sample_id], "question_id"
    if source_ref and source_ref in index["by_ref"]:
        return index["by_ref"][source_ref], "source_ref"
    return None, "unresolved"


def source_ref_audit(items: List[Dict[str, object]], index: Dict[str, Dict[str, Dict[str, object]]], issues: List[Issue]) -> Dict[str, object]:
    records = []
    resolved = 0
    image_linked = 0
    formula_linked = 0
    image_formula_linked = 0
    cooccurrence = Counter()
    unresolved = []
    mismatched_ref = []
    for item in items:
        sample_id = str(item.get("id", "<missing-id>"))
        source_ref = str(item.get("source_ref", ""))
        entry, method = find_entry(item, index)
        if entry is None:
            unresolved.append(sample_id)
            issues.append(Issue(sample_id, "unresolved_source_ref", "source_ref", source_ref))
            records.append(LinkageRecord(sample_id, source_ref, False, method))
            continue
        resolved += 1
        entry_ref = str(entry.get("source_ref", ""))
        if source_ref and entry_ref and source_ref != entry_ref:
            mismatched_ref.append(sample_id)
            issues.append(Issue(sample_id, "source_ref_mismatch", "source_ref", f"dataset={source_ref}; index={entry_ref}"))
        image_count = int(entry.get("image_count", 0) or 0)
        formula_count = int(entry.get("formula_count", 0) or 0)
        co_type = str(entry.get("cooccurrence_type", ""))
        image_linked += int(image_count > 0)
        formula_linked += int(formula_count > 0)
        image_formula_linked += int(image_count > 0 and formula_count > 0)
        cooccurrence[co_type or "unknown"] += 1
        records.append(
            LinkageRecord(
                sample_id=sample_id,
                source_ref=source_ref,
                resolved=True,
                resolution_method=method,
                chunk_path=str(entry.get("chunk_path", "")),
                image_count=image_count,
                formula_count=formula_count,
                cooccurrence_type=co_type,
            )
        )
    return {
        "resolved": resolved,
        "unresolved": len(unresolved),
        "source_ref_resolved_rate": rate(resolved, len(items)),
        "unresolved_source_ref": unresolved,
        "source_ref_mismatch_count": len(mismatched_ref),
        "source_ref_mismatch_ids": mismatched_ref,
        "multimodal_linkage_summary": {
            "image_linked_count": image_linked,
            "formula_linked_count": formula_linked,
            "image_formula_linked_count": image_formula_linked,
            "cooccurrence_distribution": dict(sorted(cooccurrence.items())),
        },
        "linkage_records_sample": [asdict(record) for record in records[:20]],
    }


def run(dataset: str, multimodal_index: str, output: str) -> Dict[str, object]:
    dataset_path = Path(dataset)
    index_path = Path(multimodal_index)
    data = load_json(dataset_path)
    multimodal = load_json(index_path)
    items = questions(data)
    issues: List[Issue] = []
    index = build_index(multimodal)
    field_cov = required_field_coverage(items, issues)
    id_stats = unique_id_rate(items, issues)
    dimension_stats = dimension_validity_rate(items, issues)
    reasoning_stats = reasoning_steps_coverage(items, issues)
    source_stats = source_ref_audit(items, index, issues)
    payload = {
        "schema_version": "1.0",
        "experiment": "track1_sci_align_integrity_audit",
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "evaluation_mode": "local_dataset_integrity_and_traceability_audit",
        "inputs": {
            "dataset": str(dataset_path).replace("\\", "/"),
            "multimodal_index": str(index_path).replace("\\", "/"),
        },
        "total_questions": len(items),
        "unique_id_rate": id_stats["rate"],
        "unique_id_summary": id_stats,
        "required_field_coverage": field_cov,
        "dimension_validity_rate": dimension_stats["rate"],
        "dimension_summary": dimension_stats,
        "reasoning_steps_coverage": reasoning_stats,
        "source_ref_resolved_rate": source_stats["source_ref_resolved_rate"],
        "source_ref_summary": {key: value for key, value in source_stats.items() if key not in {"multimodal_linkage_summary", "linkage_records_sample"}},
        "multimodal_linkage_summary": source_stats["multimodal_linkage_summary"],
        "linkage_records_sample": source_stats["linkage_records_sample"],
        "issues": [asdict(issue) for issue in issues],
        "issue_summary": dict(sorted(Counter(issue.issue_type for issue in issues).items())),
        "boundary_notes": [
            "source_ref_resolved_rate only claims resolvability against the local multimodal_index.json.",
            "image/formula linkage counts are derived from multimodal_index entries, not from fresh document parsing.",
            "If issues is non-empty, report wording must preserve the unresolved or mismatch boundary.",
        ],
    }
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(out)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit Track 1 Sci-Align dataset integrity and source traceability.")
    parser.add_argument("--dataset", default="benchmark/dataset/core.json")
    parser.add_argument("--multimodal-index", default="benchmark/dataset/multimodal_index.json")
    parser.add_argument("--output", default="benchmark/eval/results/track1_sci_align_integrity_audit.json")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    payload = run(args.dataset, args.multimodal_index, args.output)
    summary = {
        "total_questions": payload["total_questions"],
        "unique_id_rate": payload["unique_id_rate"],
        "required_field_complete_rate": payload["required_field_coverage"]["complete_rate"],
        "dimension_validity_rate": payload["dimension_validity_rate"],
        "source_ref_resolved_rate": payload["source_ref_resolved_rate"],
        "reasoning_steps_coverage": payload["reasoning_steps_coverage"]["coverage"],
        "multimodal_linkage_summary": payload["multimodal_linkage_summary"],
        "issue_count": len(payload["issues"]),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    ok = (
        payload["total_questions"] > 0
        and payload["unique_id_rate"] == 1.0
        and payload["required_field_coverage"]["complete_rate"] == 1.0
        and payload["dimension_validity_rate"] == 1.0
        and payload["source_ref_resolved_rate"] == 1.0
        and payload["reasoning_steps_coverage"]["coverage"] == 1.0
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
