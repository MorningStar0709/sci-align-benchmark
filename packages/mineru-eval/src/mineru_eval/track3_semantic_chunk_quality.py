"""Audit Track 3 semantic medical chunk quality challenge samples."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[4]


MANIFEST_REQUIRED_FIELDS = [
    "chunk_id",
    "filepath",
    "source_section",
    "source_line_start",
    "source_line_end",
    "medical_label",
    "medical_parent",
]

TARGET_SECTIONS = {
    "primary_outcome": {
        "labels": {"primary_outcome", "_primary_outcome_other"},
        "patterns": [r"primary outcomes?", r"primary endpoint", r"primary outcome measures?"],
    },
    "secondary_outcome": {
        "labels": {"secondary_outcome", "_secondary_outcome_other"},
        "patterns": [r"secondary outcomes?", r"secondary endpoint", r"secondary outcome measures?"],
    },
    "subgroup_analysis": {
        "labels": {"subgroup_analysis", "_subgroup_analysis_other"},
        "patterns": [r"subgroup analysis", r"subgroup"],
    },
    "sensitivity_analysis": {
        "labels": {"sensitivity_analysis", "_sensitivity_analysis_other"},
        "patterns": [r"sensitivity analysis", r"sensitivity"],
    },
    "adverse_events": {
        "labels": {"adverse_events", "_adverse_events_other", "safety_outcomes"},
        "patterns": [r"adverse events?", r"serious adverse", r"safety"],
    },
    "study_design": {
        "labels": {"study_design", "_study_design_other"},
        "patterns": [r"study design", r"design and participants", r"study design and"],
    },
    "randomization": {
        "labels": {"randomization", "_randomization_other"},
        "patterns": [r"randomization", r"randomisation", r"randomized", r"randomised"],
    },
    "blinding": {
        "labels": {"blinding", "_blinding_other"},
        "patterns": [r"blinding", r"masking", r"masked"],
    },
    "statistical_analysis": {
        "labels": {"statistical_analysis", "_statistical_analysis_other"},
        "patterns": [r"statistical analysis", r"statistical analysis plan", r"statistics"],
    },
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def path_text(path: Path) -> str:
    return str(path).replace("\\", "/")


def rate(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0


def read_preview(path: Path, limit: int = 420) -> str:
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace")
    return re.sub(r"\s+", " ", text).strip()[:limit]


def resolve_chunk_path(root: Path, filepath: str) -> Path:
    path = Path(filepath)
    if path.is_absolute():
        return path
    return ROOT / path


def classify_chunk(chunk: dict[str, Any]) -> list[str]:
    label = str(chunk.get("medical_label") or "").lower()
    parent = str(chunk.get("medical_parent") or "").lower()
    section = str(chunk.get("source_section") or "").lower()
    categories = []
    for name, spec in TARGET_SECTIONS.items():
        if label in spec["labels"] or parent in spec["labels"]:
            categories.append(name)
            continue
        if any(re.search(pattern, section, flags=re.IGNORECASE) for pattern in spec["patterns"]):
            categories.append(name)
    return categories


def field_coverage(chunks: list[dict[str, Any]]) -> dict[str, Any]:
    coverage = {}
    for field in MANIFEST_REQUIRED_FIELDS:
        present = sum(1 for chunk in chunks if field in chunk and chunk.get(field) not in {None, ""})
        coverage[field] = {"present": present, "total": len(chunks), "rate": rate(present, len(chunks))}
    filepath_present = sum(1 for chunk in chunks if chunk.get("filepath"))
    coverage["source_file"] = {"present": filepath_present, "total": len(chunks), "rate": rate(filepath_present, len(chunks)), "derived_from": "filepath"}
    coverage["text_preview"] = {"present": None, "total": len(chunks), "rate": None, "derived_from": "chunk markdown file preview"}
    return coverage


def sample_records(chunks: list[dict[str, Any]], chunks_root: Path, max_per_category: int = 3) -> list[dict[str, Any]]:
    records = []
    selected_by_category: dict[str, int] = defaultdict(int)
    for chunk in chunks:
        categories = classify_chunk(chunk)
        if not categories:
            continue
        chunk_path = resolve_chunk_path(chunks_root, str(chunk.get("filepath") or ""))
        for category in categories:
            if selected_by_category[category] >= max_per_category:
                continue
            records.append(
                {
                    "challenge_category": category,
                    "chunk_id": chunk.get("chunk_id"),
                    "medical_label": chunk.get("medical_label"),
                    "medical_parent": chunk.get("medical_parent"),
                    "source_section": chunk.get("source_section"),
                    "source_file": chunk.get("filepath"),
                    "set": chunk.get("set"),
                    "estimated_tokens": chunk.get("estimated_tokens"),
                    "source_line_start": chunk.get("source_line_start"),
                    "source_line_end": chunk.get("source_line_end"),
                    "text_preview": read_preview(chunk_path),
                    "file_exists": chunk_path.exists(),
                }
            )
            selected_by_category[category] += 1
    return records


def _check_missing_manifest_fields(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Check for chunks missing required manifest fields."""
    flags = []
    missing_by_field = {field: [chunk.get("chunk_id") for chunk in chunks if field not in chunk or chunk.get(field) in {None, ""}] for field in MANIFEST_REQUIRED_FIELDS}
    for field, ids in missing_by_field.items():
        if ids:
            flags.append({"risk": "missing_manifest_field", "field": field, "count": len(ids), "sample_chunk_ids": ids[:8]})
    return flags


def _check_missing_files(chunks: list[dict[str, Any]], chunks_root: Path) -> list[dict[str, Any]]:
    """Check for chunks whose chunk markdown files are missing."""
    missing_files = []
    for chunk in chunks:
        chunk_path = resolve_chunk_path(chunks_root, str(chunk.get("filepath") or ""))
        if not chunk_path.exists():
            missing_files.append(chunk.get("chunk_id"))
    if missing_files:
        return [{"risk": "chunk_markdown_missing", "count": len(missing_files), "sample_chunk_ids": missing_files[:8]}]
    return []


def _check_token_bounds(chunks: list[dict[str, Any]], manifest: dict[str, Any]) -> list[dict[str, Any]]:
    """Check for chunks below min_tokens or above max_tokens."""
    flags = []
    min_tokens = int(manifest.get("min_tokens") or 0)
    max_tokens = int(manifest.get("max_tokens") or 0)
    tiny = [chunk.get("chunk_id") for chunk in chunks if isinstance(chunk.get("estimated_tokens"), int) and min_tokens and chunk["estimated_tokens"] < min_tokens]
    oversized = [chunk.get("chunk_id") for chunk in chunks if isinstance(chunk.get("estimated_tokens"), int) and max_tokens and chunk["estimated_tokens"] > max_tokens]
    if tiny:
        flags.append({"risk": "below_configured_min_tokens", "count": len(tiny), "sample_chunk_ids": tiny[:8], "note": "heading-only or short structured sections are expected in semantic slicing but should be explicit"})
    if oversized:
        flags.append({"risk": "above_configured_max_tokens", "count": len(oversized), "sample_chunk_ids": oversized[:8]})
    return flags


def _check_label_quality(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Check for generic/other labels and chunks not in target sections."""
    flags = []
    generic_labels = [chunk.get("chunk_id") for chunk in chunks if chunk.get("medical_label") in {"other", "_front_matter_other"}]
    flags.append({"risk": "generic_or_other_label", "count": len(generic_labels), "rate": rate(len(generic_labels), len(chunks)), "sample_chunk_ids": generic_labels[:8]})
    targetless = [chunk.get("chunk_id") for chunk in chunks if not classify_chunk(chunk)]
    flags.append({"risk": "not_in_challenge_target_sections", "count": len(targetless), "rate": rate(len(targetless), len(chunks)), "sample_chunk_ids": targetless[:8]})
    return flags



def risk_flags(chunks: list[dict[str, Any]], chunks_root: Path, manifest: dict[str, Any]) -> list[dict[str, Any]]:
    flags = []
    flags.extend(_check_missing_manifest_fields(chunks))
    flags.extend(_check_missing_files(chunks, chunks_root))
    flags.extend(_check_token_bounds(chunks, manifest))
    flags.extend(_check_label_quality(chunks))
    return flags


def target_counts(chunks: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter()
    label_counts: dict[str, Counter] = defaultdict(Counter)
    set_counts: dict[str, Counter] = defaultdict(Counter)
    for chunk in chunks:
        categories = classify_chunk(chunk)
        for category in categories:
            counts[category] += 1
            label_counts[category][str(chunk.get("medical_label"))] += 1
            set_counts[category][str(chunk.get("set"))] += 1
    return {
        category: {
            "count": counts.get(category, 0),
            "label_counts": dict(sorted(label_counts[category].items())),
            "set_counts": dict(sorted(set_counts[category].items())),
        }
        for category in TARGET_SECTIONS
    }


def manifest_summary(manifest: dict[str, Any], chunks: list[dict[str, Any]]) -> dict[str, Any]:
    labels = Counter(str(chunk.get("medical_label")) for chunk in chunks)
    parents = Counter(str(chunk.get("medical_parent")) for chunk in chunks if chunk.get("medical_parent") is not None)
    sets = Counter(str(chunk.get("set")) for chunk in chunks)
    return {
        "total_chunks": manifest.get("total_chunks", len(chunks)),
        "actual_chunk_records": len(chunks),
        "train_chunks": manifest.get("train_chunks"),
        "eval_chunks": manifest.get("eval_chunks"),
        "target_tokens": manifest.get("target_tokens"),
        "min_tokens": manifest.get("min_tokens"),
        "max_tokens": manifest.get("max_tokens"),
        "set_counts": dict(sorted(sets.items())),
        "top_medical_labels": dict(labels.most_common(16)),
        "top_medical_parents": dict(parents.most_common(16)),
    }


def build_result(chunks_root: Path) -> dict[str, Any]:
    manifest_path = chunks_root / "chunks_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"chunks manifest not found: {manifest_path}")
    manifest = load_json(manifest_path)
    chunks = manifest.get("chunks") or []
    records = sample_records(chunks, chunks_root)
    counts = target_counts(chunks)
    covered_targets = [name for name, payload in counts.items() if payload["count"] > 0]
    return {
        "status": "available",
        "generated_at": datetime.now().astimezone().strftime("%Y-%m-%dT%H:%M:%S%z"),
        "task": "Track 3 Task 5 - Semantic Chunk Quality Challenge Set",
        "evaluation_mode": "semantic_chunk_manifest_and_markdown_audit_no_gold_accuracy_claim",
        "chunks_root": path_text(chunks_root),
        "manifest_path": path_text(manifest_path),
        "manifest_summary": manifest_summary(manifest, chunks),
        "target_section_counts": counts,
        "covered_target_sections": covered_targets,
        "covered_target_section_count": len(covered_targets),
        "field_coverage": field_coverage(chunks),
        "sample_records": records,
        "sample_record_count": len(records),
        "risk_flags": risk_flags(chunks, chunks_root, manifest),
        "schema_gap": None if chunks else "manifest contains no chunks array records",
        "boundary_notes": [
            "This challenge set audits semantic section structure, metadata completeness, and sample chunk readability; it does not claim human gold-label chunking accuracy.",
            "A target section can be matched by medical_label, medical_parent, or source_section text because existing manifest labels mix L1/L2 semantic labels and section headings.",
            "Short heading-only chunks are counted as risk flags for review, not automatic errors, because semantic slicing intentionally preserves some structural headings.",
        ],
        "acceptance_check": {
            "required_top_level_fields_present": True,
            "target_section_counts_present": True,
            "field_coverage_present": True,
            "sample_records_present": bool(records),
            "risk_flags_present": True,
            "covered_at_least_four_target_sections": len(covered_targets) >= 4,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit semantic medical chunks for Track 3 challenge sections.")
    parser.add_argument("--chunks", required=True, help="Medical chunks directory containing chunks_manifest.json")
    parser.add_argument("--output", required=True, help="Output JSON path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_result(Path(args.chunks))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[Track3 Task5] wrote {output}")


if __name__ == "__main__":
    main()
