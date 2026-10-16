"""Replay cross-modal alignment weakness taxonomy for Track 1 Sci-Align."""

import argparse
import json
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List


# ROOT = Path(__file__).resolve().parents[2]  # Phase 5: sys.path hack removed, use mineru_benchmark.agent.utils.ROOT instead


REASON_RULES = [
    ("chunk_boundary_mismatch", ["unrelated image", "no specific mapping", "vs unrelated", "different physical subsystem"]),
    ("multi_role_figure", ["dynamic network", "node labels mismatch", "numbers vs letters", "generic note icon"]),
    ("formula_context_missing", ["Laplace transform", "generic controllability proof", "proof", "calculation formulas"]),
    ("image_semantic_mismatch", ["mass-spring-damper vs RLC", "mechanical vs electrical", "circuit diagram vs"]),
    ("model_or_label_ambiguity", ["degenerate Jordan", "stable node", "classification"]),
]


@dataclass
class TaxonomySample:
    sample_id: str
    source_bucket: str
    original_label: str
    reason_category: str
    description: str
    evidence_source: str


def load_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_label(label: str) -> str:
    label = label.lower().strip()
    if label in {"consistent", "c"}:
        return "consistent"
    if label in {"partially_consistent", "partial", "p"}:
        return "partial"
    if label in {"inconsistent", "i"}:
        return "inconsistent"
    if label in {"uncertain", "unknown"}:
        return "uncertain"
    return label or "unknown"


def classify_reason(text: str, label: str) -> str:
    lowered = text.lower()
    if normalize_label(label) == "consistent":
        return "aligned_semantic_match"
    for category, keywords in REASON_RULES:
        if any(keyword.lower() in lowered for keyword in keywords):
            return category
    if normalize_label(label) == "partial":
        return "model_or_label_ambiguity"
    if normalize_label(label) == "inconsistent":
        return "unknown_alignment_gap"
    return "unknown"


def samples_from_summary(summary: Dict[str, object]) -> List[TaxonomySample]:
    detailed = summary.get("detailed_results", {})
    records = []
    for bucket, items in detailed.items():
        label = normalize_label(bucket)
        for idx, item in enumerate(items, 1):
            description = str(item)
            records.append(
                TaxonomySample(
                    sample_id=f"summary_{label}_{idx:02d}",
                    source_bucket=bucket,
                    original_label=label,
                    reason_category=classify_reason(description, label),
                    description=description,
                    evidence_source="cross_modal_audit_summary.detailed_results",
                )
            )
    return records


def samples_from_sciverse(sciverse: Dict[str, object], max_per_label: int) -> List[TaxonomySample]:
    records = []
    per_label = Counter()
    per_paper = sciverse.get("per_paper", {})
    for subfield, paper in per_paper.items():
        for idx, item in enumerate(paper.get("formula_audit", []) or [], 1):
            label = normalize_label(str(item.get("alignment", "unknown")))
            if per_label[label] >= max_per_label:
                continue
            description = f"{subfield}: {item.get('latex', '')} | {item.get('context_preview', '')}"
            records.append(
                TaxonomySample(
                    sample_id=f"sciverse_{label}_{per_label[label] + 1:02d}",
                    source_bucket="sciverse_formula_audit",
                    original_label=label,
                    reason_category=classify_reason(description, label),
                    description=re.sub(r"\s+", " ", description).strip()[:260],
                    evidence_source="sciverse_cross_modal_audit.per_paper.formula_audit",
                )
            )
            per_label[label] += 1
    return records


def limited_samples(samples: List[TaxonomySample], limit_per_label: int) -> List[TaxonomySample]:
    selected = []
    counts = Counter()
    for sample in samples:
        label = normalize_label(sample.original_label)
        if counts[label] >= limit_per_label:
            continue
        selected.append(sample)
        counts[label] += 1
    return selected


def summarize_cooccurrence(cooccurrence: Dict[str, object]) -> Dict[str, object]:
    keys = [
        "total_chunks_manifest",
        "chunks_scanned",
        "read_errors",
        "has_both",
        "has_both_pct",
        "image_only",
        "image_only_pct",
        "formula_only",
        "formula_only_pct",
        "neither",
        "neither_pct",
        "subdomain_count",
    ]
    return {key: cooccurrence.get(key) for key in keys if key in cooccurrence}


def build_taxonomy(samples: List[TaxonomySample]) -> Dict[str, object]:
    by_label = Counter(normalize_label(sample.original_label) for sample in samples)
    by_reason = Counter(sample.reason_category for sample in samples)
    label_reason = defaultdict(Counter)
    for sample in samples:
        label_reason[normalize_label(sample.original_label)][sample.reason_category] += 1
    return {
        "by_label": dict(sorted(by_label.items())),
        "by_reason": dict(sorted(by_reason.items())),
        "label_reason_matrix": {label: dict(sorted(counter.items())) for label, counter in sorted(label_reason.items())},
        "covered_reason_categories": sorted(by_reason.keys()),
    }


def run(cooccurrence: str, audit_summary: str, output: str, sciverse: str = "benchmark/eval/results/sciverse_cross_modal_audit.json") -> Dict[str, object]:
    co_path = Path(cooccurrence)
    summary_path = Path(audit_summary)
    sciverse_path = Path(sciverse)
    co_data = load_json(co_path)
    summary_data = load_json(summary_path)
    samples = samples_from_summary(summary_data)
    if sciverse_path.exists():
        samples.extend(samples_from_sciverse(load_json(sciverse_path), max_per_label=10))
    selected = limited_samples(samples, limit_per_label=10)
    taxonomy = build_taxonomy(selected)
    summary_stats = summary_data.get("stats", {})
    payload = {
        "schema_version": "1.0",
        "experiment": "track1_multimodal_error_taxonomy",
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "evaluation_mode": "replay_existing_cross_modal_audit_no_new_api_calls",
        "inputs": {
            "cooccurrence": str(co_path).replace("\\", "/"),
            "audit_summary": str(summary_path).replace("\\", "/"),
            "sciverse_optional": str(sciverse_path).replace("\\", "/") if sciverse_path.exists() else "not_available",
        },
        "source_audit_summary": {
            "total_judged": summary_stats.get("total_judged"),
            "consistent": summary_stats.get("consistent"),
            "partially_consistent": summary_stats.get("partially_consistent"),
            "inconsistent": summary_stats.get("inconsistent"),
            "file_not_found": summary_stats.get("file_not_found"),
            "alignment_quality": summary_stats.get("alignment_quality"),
        },
        "cooccurrence_summary": summarize_cooccurrence(co_data),
        "sample_count": len(selected),
        "taxonomy_counts": taxonomy,
        "sample_records": [asdict(sample) for sample in selected],
        "boundary_notes": [
            "This taxonomy replays existing cross-modal audit artifacts and does not call a new vision model.",
            "Reason categories are rule-based labels over existing audit descriptions, intended for boundary analysis rather than metric improvement claims.",
            "If a label has fewer than 10 samples, the matrix reports available evidence without fabricating additional cases.",
        ],
    }
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(out)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build Track 1 multimodal alignment weakness taxonomy from audited artifacts.")
    parser.add_argument("--cooccurrence", default="benchmark/eval/results/image_formula_cooccurrence.json")
    parser.add_argument("--audit-summary", default="benchmark/eval/results/cross_modal_audit_summary.json")
    parser.add_argument("--sciverse", default="benchmark/eval/results/sciverse_cross_modal_audit.json")
    parser.add_argument("--output", default="benchmark/eval/results/track1_multimodal_error_taxonomy.json")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    payload = run(args.cooccurrence, args.audit_summary, args.output, args.sciverse)
    summary = {
        "sample_count": payload["sample_count"],
        "labels": payload["taxonomy_counts"]["by_label"],
        "reason_categories": payload["taxonomy_counts"]["by_reason"],
        "covered_reason_category_count": len(payload["taxonomy_counts"]["covered_reason_categories"]),
        "evaluation_mode": payload["evaluation_mode"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    ok = payload["sample_count"] > 0 and len(payload["taxonomy_counts"]["covered_reason_categories"]) >= 3
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
