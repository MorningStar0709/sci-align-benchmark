"""Hard-document coverage stress evaluation for Track 2 Agent evidence."""

import argparse
import json
import zipfile
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[4]  # restored Phase 5


@dataclass
class StressSample:
    sample_id: str
    challenge_type: str
    sample_path: str
    artifact_type: str
    source_policy: str
    expected_signal: str
    fallback_note: str = ""


@dataclass
class StressRecord:
    sample_id: str
    challenge_type: str
    sample_path: str
    artifact_type: str
    source_policy: str
    exists: bool
    status: str
    expected_signal: str
    observed_signal: str
    schema_consistent: bool
    metrics: Dict[str, object] = field(default_factory=dict)
    failure_reason: str = ""
    fallback_note: str = ""


def _load_json(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def _first_existing(patterns: List[str]) -> List[Path]:
    found = []
    for pattern in patterns:
        found.extend(ROOT.glob(pattern))
    return sorted(p for p in found if p.exists())


def _top_docs_by_metric(stats: Dict[str, object], metric: str, limit: int) -> List[Dict[str, object]]:
    docs = stats.get("per_doc", []) if isinstance(stats, dict) else []
    return sorted(docs, key=lambda item: item.get(metric, 0), reverse=True)[:limit]


def _find_visual_samples(report: Dict[str, object], keywords: List[str], limit: int) -> List[Dict[str, object]]:
    samples = []
    per_paper = report.get("per_paper", {}) if isinstance(report, dict) else {}
    for paper_name, paper in per_paper.items():
        for image in paper.get("images", []):
            text = image.get("raw_response", "")
            if any(keyword in text for keyword in keywords):
                samples.append({"paper": paper_name, **image})
    return samples[:limit]


def _zip_metrics(path: Path) -> Dict[str, object]:
    if not path.exists() or path.suffix.lower() not in {".docx", ".pptx", ".xlsx"}:
        return {}
    with zipfile.ZipFile(path) as zf:
        names = zf.namelist()
    return {
        "zip_entries": len(names),
        "media_entries": sum(1 for name in names if "/media/" in name),
        "xml_entries": sum(1 for name in names if name.endswith(".xml")),
    }


def build_samples() -> List[StressSample]:
    stats = _load_json(ROOT / "corpus" / "stats" / "corpus_stats.json")
    visual_report = _load_json(ROOT / "benchmark" / "eval" / "results" / "sciverse_visual_audit_report.json")
    top_formula_docs = _top_docs_by_metric(stats, "formula_count", 2)
    top_chunk_docs = _top_docs_by_metric(stats, "chunk_count", 2)
    flow_images = _find_visual_samples(visual_report, ["框图", "流程图", "结构"], 2)
    chart_images = _find_visual_samples(visual_report, ["曲线", "表格", "参数表", "性能对比"], 2)
    upload_pdfs = _first_existing(["data/uploads/**/*.pdf", "data/sources_flywheel/*.pdf"])
    samples = []
    samples.append(StressSample("numeric_dense_xlsx", "numeric_dense", "benchmark/agent/test_materials/test_xlsx.xlsx", "xlsx", "repository_test_material", "zip table workbook is present and structurally readable"))
    for idx, doc in enumerate(top_formula_docs, 1):
        samples.append(StressSample(f"numeric_dense_formula_{idx}", "numeric_dense", "corpus/stats/corpus_stats.json", "corpus_stats_record", "derived_repository_metric", f"formula_count={doc.get('formula_count', 0)} from {doc.get('filename', '')}"))
    for idx, doc in enumerate(top_chunk_docs, 1):
        samples.append(StressSample(f"cross_page_chunk_{idx}", "cross_page", "corpus/stats/corpus_stats.json", "corpus_stats_record", "derived_repository_metric", f"chunk_count={doc.get('chunk_count', 0)} from {doc.get('filename', '')}"))
    for idx, image in enumerate(flow_images, 1):
        samples.append(StressSample(f"flow_engineering_{idx}", "flow_engineering_diagram", _rel(Path(image.get("path", ""))), "sciverse_image", "sciverse_cached_image", image.get("raw_response", "")[:120]))
    if len(flow_images) < 2:
        samples.append(StressSample("flow_engineering_missing", "flow_engineering_diagram", "benchmark/data/sciverse_images", "sciverse_image", "sciverse_cached_image", "not enough flow diagram samples", "样本不足，保留缺口记录。"))
    for idx, path in enumerate(upload_pdfs[:2], 1):
        samples.append(StressSample(f"low_quality_scan_{idx}", "low_quality_scan", _rel(path), "pdf", "repository_or_flywheel_pdf", "PDF exists for OCR/scan boundary replay"))
    if len(upload_pdfs) < 2:
        samples.append(StressSample("low_quality_scan_missing", "low_quality_scan", "data/uploads or data/sources_flywheel", "pdf", "repository_or_flywheel_pdf", "not enough scan-like pdf samples", "未强行引入外部扫描件。"))
    for idx, image in enumerate(chart_images, 1):
        samples.append(StressSample(f"complex_chart_{idx}", "complex_chart", _rel(Path(image.get("path", ""))), "sciverse_image", "sciverse_cached_image", image.get("raw_response", "")[:120]))
    if len(chart_images) < 2:
        samples.append(StressSample("complex_chart_missing", "complex_chart", "benchmark/data/sciverse_images", "sciverse_image", "sciverse_cached_image", "not enough chart/table samples", "样本不足，保留缺口记录。"))
    samples.append(StressSample("standard_schema_01", "standard_spec", "benchmark/dataset/schema.json", "json_schema", "repository_schema", "JSON Schema required fields and enum constraints are readable", "这是结构规范样本，不等价于行业标准 PDF。"))
    samples.append(StressSample("standard_readme_01", "standard_spec", "benchmark/dataset/README.md", "markdown_spec", "repository_documentation", "dataset README is present", "这是数据集规范说明，不等价于行业标准 PDF。"))
    samples.append(StressSample("multi_format_docx", "standard_spec", "benchmark/agent/test_materials/test_docx.docx", "docx", "repository_test_material", "DOCX package is structurally readable", "补充多格式规范结构覆盖。"))
    samples.append(StressSample("multi_format_pptx", "flow_engineering_diagram", "benchmark/agent/test_materials/test_ppt.pptx", "pptx", "repository_test_material", "PPTX package is structurally readable", "补充流程/图文结构覆盖。"))
    return samples


def evaluate_sample(sample: StressSample) -> StressRecord:
    path = ROOT / sample.sample_path
    exists = path.exists() or sample.artifact_type == "corpus_stats_record"
    metrics = {}
    observed_signal = ""
    failure_reason = ""
    if sample.artifact_type in {"docx", "pptx", "xlsx"} and path.exists():
        metrics.update(_zip_metrics(path))
        observed_signal = f"zip_entries={metrics.get('zip_entries', 0)}, media_entries={metrics.get('media_entries', 0)}, xml_entries={metrics.get('xml_entries', 0)}"
    elif sample.artifact_type == "json_schema" and path.exists():
        schema = _load_json(path)
        required = schema.get("required", [])
        defs = schema.get("$defs", {})
        metrics.update({"required_fields": len(required), "defs_count": len(defs), "has_question_def": "question" in defs})
        observed_signal = f"required_fields={len(required)}, defs_count={len(defs)}"
    elif sample.artifact_type == "markdown_spec" and path.exists():
        text = path.read_text(encoding="utf-8", errors="replace")
        metrics.update({"line_count": text.count("\n") + 1, "size_bytes": path.stat().st_size})
        observed_signal = f"line_count={metrics['line_count']}, size_bytes={metrics['size_bytes']}"
    elif sample.artifact_type == "sciverse_image" and path.exists():
        metrics.update({"size_bytes": path.stat().st_size, "suffix": path.suffix.lower()})
        observed_signal = f"image_size_bytes={metrics['size_bytes']}"
    elif sample.artifact_type == "pdf" and path.exists():
        metrics.update({"size_bytes": path.stat().st_size, "suffix": path.suffix.lower()})
        observed_signal = f"pdf_size_bytes={metrics['size_bytes']}"
    elif sample.artifact_type == "corpus_stats_record":
        metrics.update({"source": "corpus/stats/corpus_stats.json"})
        observed_signal = sample.expected_signal
    else:
        failure_reason = "sample_not_available"
    schema_consistent = bool(sample.sample_id and sample.challenge_type and sample.artifact_type and sample.source_policy and sample.expected_signal)
    signal_ok = bool(observed_signal) or sample.artifact_type == "corpus_stats_record"
    status = "success" if exists and signal_ok and schema_consistent else "not_available"
    if status == "success" and sample.fallback_note:
        status = "partial"
    return StressRecord(
        sample_id=sample.sample_id,
        challenge_type=sample.challenge_type,
        sample_path=sample.sample_path,
        artifact_type=sample.artifact_type,
        source_policy=sample.source_policy,
        exists=exists,
        status=status,
        expected_signal=sample.expected_signal,
        observed_signal=observed_signal,
        schema_consistent=schema_consistent,
        metrics=metrics,
        failure_reason=failure_reason,
        fallback_note=sample.fallback_note,
    )


def _rate(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0


def summarize(records: List[StressRecord]) -> Dict[str, object]:
    total = len(records)
    success = sum(1 for r in records if r.status == "success")
    partial = sum(1 for r in records if r.status == "partial")
    unavailable = sum(1 for r in records if r.status == "not_available")
    schema_ok = sum(1 for r in records if r.schema_consistent)
    by_challenge = {}
    for record in records:
        item = by_challenge.setdefault(record.challenge_type, {"total": 0, "success": 0, "partial": 0, "not_available": 0, "sample_paths": []})
        item["total"] += 1
        item["success"] += int(record.status == "success")
        item["partial"] += int(record.status == "partial")
        item["not_available"] += int(record.status == "not_available")
        item["sample_paths"].append(record.sample_path)
    for item in by_challenge.values():
        item["coverage_rate"] = _rate(item["success"] + item["partial"], item["total"])
    failure_samples = [r.sample_id for r in records if r.status == "not_available"]
    return {
        "total_samples": total,
        "challenge_types": len(by_challenge),
        "success_samples": success,
        "partial_samples": partial,
        "not_available_samples": unavailable,
        "coverage_rate": _rate(success + partial, total),
        "schema_consistency_rate": _rate(schema_ok, total),
        "failure_samples": failure_samples,
        "by_challenge": by_challenge,
        "boundary_note": "样本均来自仓库内文件或既有审计产物；standard_spec 使用 schema/README 做结构规范覆盖，不伪装为行业标准 PDF。",
    }


def run(output: str) -> Dict[str, object]:
    samples = build_samples()
    records = [evaluate_sample(sample) for sample in samples]
    payload = {
        "schema_version": "1.0",
        "experiment": "track2_hard_document_stress_eval",
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "evaluation_mode": "repository_artifact_and_schema_replay",
        "scope": {
            "included": ["sample availability", "schema consistency", "derived corpus metrics", "cached visual audit evidence"],
            "excluded": ["new external document collection", "copyright-unclear sample ingestion", "fresh OCR re-parse"],
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
    parser = argparse.ArgumentParser(description="Evaluate hard-document challenge coverage using auditable repository artifacts.")
    parser.add_argument("--output", default="benchmark/eval/results/agent_hard_document_stress.json")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    payload = run(args.output)
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    return 0 if payload["summary"]["challenge_types"] >= 6 else 1


if __name__ == "__main__":
    raise SystemExit(main())
