"""External benchmark format compatibility audit for Track 1 Sci-Align."""

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List


# ROOT = Path(__file__).resolve().parents[2]  # Phase 5: sys.path hack removed, use mineru_benchmark.agent.utils.ROOT instead


REQUIRED_PRESERVED_FIELDS = ["id", "dimension", "question", "answer", "reasoning_steps", "source_ref"]
DIMENSIONS = ["A", "B", "C", "D"]


@dataclass
class FormatResult:
    format_name: str
    status: str
    sample_count: int
    preserved_fields: List[str]
    missing_fields: List[str]
    field_preservation_rate: float
    example_keys: List[str]
    error_type: str = ""
    error: str = ""


def load_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def questions(data: Dict[str, object]) -> List[Dict[str, object]]:
    items = data.get("questions", [])
    if not isinstance(items, list):
        raise ValueError("dataset.questions must be a list")
    return items


def rate(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0


def path_text(path: Path) -> str:
    return str(path).replace("\\", "/")


def stratified_sample(items: List[Dict[str, object]], sample_size: int) -> List[Dict[str, object]]:
    buckets: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for item in items:
        buckets[str(item.get("dimension", "unknown"))].append(item)
    selected = []
    base = sample_size // len(DIMENSIONS)
    remainder = sample_size % len(DIMENSIONS)
    for idx, dimension in enumerate(DIMENSIONS):
        quota = base + int(idx < remainder)
        selected.extend(buckets.get(dimension, [])[:quota])
    if len(selected) < sample_size:
        selected_ids = {item.get("id") for item in selected}
        for item in items:
            if item.get("id") not in selected_ids:
                selected.append(item)
            if len(selected) >= sample_size:
                break
    return selected[:sample_size]


def distribution(items: List[Dict[str, object]], field: str) -> Dict[str, int]:
    return dict(sorted(Counter(str(item.get(field, "unknown")) for item in items).items()))


def openai_eval_style(item: Dict[str, object]) -> Dict[str, object]:
    return {
        "id": item.get("id"),
        "input": [
            {"role": "system", "content": "You are a control science benchmark solver."},
            {"role": "user", "content": item.get("question")},
        ],
        "ideal": item.get("answer"),
        "metadata": {
            "dimension": item.get("dimension"),
            "difficulty_level": item.get("difficulty_level"),
            "reasoning_steps": item.get("reasoning_steps"),
            "source_ref": item.get("source_ref"),
        },
    }


def hf_row(item: Dict[str, object]) -> Dict[str, object]:
    return {
        "id": item.get("id"),
        "question": item.get("question"),
        "answer": item.get("answer"),
        "dimension": item.get("dimension"),
        "difficulty_level": item.get("difficulty_level"),
        "reasoning_steps": item.get("reasoning_steps"),
        "source_ref": item.get("source_ref"),
        "control_category": item.get("control_category"),
    }


def simple_qa_jsonl(item: Dict[str, object]) -> Dict[str, object]:
    return {
        "id": item.get("id"),
        "question": item.get("question"),
        "answer": item.get("answer"),
        "dimension": item.get("dimension"),
        "reasoning_steps": item.get("reasoning_steps"),
        "source_ref": item.get("source_ref"),
    }


def chatml(item: Dict[str, object]) -> Dict[str, object]:
    return {
        "id": item.get("id"),
        "messages": [
            {"role": "system", "content": "You are a control science reasoning assistant."},
            {"role": "user", "content": str(item.get("question", ""))},
            {"role": "assistant", "content": str(item.get("answer", ""))},
        ],
        "metadata": {
            "dimension": item.get("dimension"),
            "reasoning_steps": item.get("reasoning_steps"),
            "source_ref": item.get("source_ref"),
        },
    }


def extracted_fields(format_name: str, record: Dict[str, object]) -> Dict[str, object]:
    if format_name == "openai_eval_style":
        metadata = record.get("metadata", {}) if isinstance(record.get("metadata"), dict) else {}
        user_messages = [message for message in record.get("input", []) if isinstance(message, dict) and message.get("role") == "user"]
        return {
            "id": record.get("id"),
            "dimension": metadata.get("dimension"),
            "question": user_messages[0].get("content") if user_messages else None,
            "answer": record.get("ideal"),
            "reasoning_steps": metadata.get("reasoning_steps"),
            "source_ref": metadata.get("source_ref"),
        }
    if format_name == "chatml":
        metadata = record.get("metadata", {}) if isinstance(record.get("metadata"), dict) else {}
        messages = record.get("messages", [])
        user_messages = [message for message in messages if isinstance(message, dict) and message.get("role") == "user"]
        assistant_messages = [message for message in messages if isinstance(message, dict) and message.get("role") == "assistant"]
        return {
            "id": record.get("id"),
            "dimension": metadata.get("dimension"),
            "question": user_messages[0].get("content") if user_messages else None,
            "answer": assistant_messages[0].get("content") if assistant_messages else None,
            "reasoning_steps": metadata.get("reasoning_steps"),
            "source_ref": metadata.get("source_ref"),
        }
    return {field: record.get(field) for field in REQUIRED_PRESERVED_FIELDS}


def present(value: object) -> bool:
    return value not in (None, "", [])


def audit_format(format_name: str, converter: Callable[[Dict[str, object]], Dict[str, object]], sample: List[Dict[str, object]]) -> tuple[Dict[str, object], List[Dict[str, object]]]:
    try:
        converted = [converter(item) for item in sample]
        preserved = []
        missing = []
        for field in REQUIRED_PRESERVED_FIELDS:
            ok = all(present(extracted_fields(format_name, record).get(field)) for record in converted)
            if ok:
                preserved.append(field)
            else:
                missing.append(field)
        result = FormatResult(
            format_name=format_name,
            status="success" if len(preserved) >= 5 else "partial",
            sample_count=len(converted),
            preserved_fields=preserved,
            missing_fields=missing,
            field_preservation_rate=rate(len(preserved), len(REQUIRED_PRESERVED_FIELDS)),
            example_keys=list(converted[0].keys()) if converted else [],
        )
        return asdict(result), converted
    except Exception as exc:
        result = FormatResult(
            format_name=format_name,
            status="error",
            sample_count=0,
            preserved_fields=[],
            missing_fields=REQUIRED_PRESERVED_FIELDS,
            field_preservation_rate=0.0,
            example_keys=[],
            error_type=type(exc).__name__,
            error=str(exc),
        )
        return asdict(result), []


def field_preservation_summary(format_results: Dict[str, Dict[str, object]]) -> Dict[str, object]:
    by_field = {}
    for field in REQUIRED_PRESERVED_FIELDS:
        preserved_by = [name for name, result in format_results.items() if field in result.get("preserved_fields", [])]
        by_field[field] = {
            "preserved_by_formats": preserved_by,
            "format_count": len(preserved_by),
            "rate": rate(len(preserved_by), len(format_results)),
        }
    successful = [name for name, result in format_results.items() if result.get("status") == "success"]
    return {
        "required_fields": REQUIRED_PRESERVED_FIELDS,
        "successful_format_count": len(successful),
        "successful_formats": successful,
        "by_field": by_field,
    }


def run(dataset: str, sample_size: int, output: str) -> Dict[str, object]:
    dataset_path = Path(dataset)
    data = load_json(dataset_path)
    items = questions(data)
    sample = stratified_sample(items, min(sample_size, len(items)))
    converters = {
        "openai_eval_style": openai_eval_style,
        "hf_row": hf_row,
        "simple_qa_jsonl": simple_qa_jsonl,
        "chatml": chatml,
    }
    formats = {}
    converted_by_format = {}
    for format_name, converter in converters.items():
        formats[format_name], converted_by_format[format_name] = audit_format(format_name, converter, sample)
    examples = {format_name: records[:1] for format_name, records in converted_by_format.items() if records}
    payload = {
        "schema_version": "1.0",
        "experiment": "track1_export_format_compatibility",
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "status": "ok" if sum(1 for result in formats.values() if result["status"] == "success") >= 3 else "degraded_format_errors",
        "evaluation_mode": "local_stratified_sample_format_conversion_no_external_api",
        "inputs": {"dataset": path_text(dataset_path), "sample_size_requested": sample_size},
        "dataset_question_count": len(items),
        "sample_size": len(sample),
        "dimension_sample_counts": distribution(sample, "dimension"),
        "difficulty_sample_counts": distribution(sample, "difficulty_level"),
        "formats": formats,
        "field_preservation_rate": field_preservation_summary(formats),
        "examples": examples,
        "boundary_notes": [
            "This compatibility audit converts a stratified local sample and does not upload data to external benchmark platforms.",
            "Examples are limited to one record per successful format to avoid duplicating the full dataset in the evidence artifact.",
            "Compatibility means structural preservation of key fields, not official certification by OpenAI, HuggingFace, or other benchmark hosts.",
            "External narrative should describe ecosystem technical usability, not community adoption or impact.",
        ],
    }
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(out)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit Track 1 external benchmark format compatibility.")
    parser.add_argument("--dataset", default="benchmark/dataset/core.json")
    parser.add_argument("--sample-size", type=int, default=50)
    parser.add_argument("--output", default="benchmark/eval/results/track1_export_format_compatibility.json")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    payload = run(args.dataset, args.sample_size, args.output)
    summary = {
        "status": payload["status"],
        "sample_size": payload["sample_size"],
        "dimension_sample_counts": payload["dimension_sample_counts"],
        "format_status": {name: result["status"] for name, result in payload["formats"].items()},
        "successful_format_count": payload["field_preservation_rate"]["successful_format_count"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if payload["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
