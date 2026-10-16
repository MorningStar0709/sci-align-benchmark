"""AI-ready smoke test for the Track 1 Sci-Align dataset."""

import argparse
import importlib.util
import json
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple


# ROOT = Path(__file__).resolve().parents[2]  # Phase 5: sys.path hack removed, use mineru_benchmark.agent.utils.ROOT instead


REQUIRED_FIELDS = [
    "id",
    "dimension",
    "difficulty_level",
    "control_category",
    "question",
    "answer",
    "reasoning_steps",
    "source_ref",
    "consistency_status",
]


@dataclass
class FormatExport:
    format_name: str
    status: str
    sample_count: int
    preserved_fields: List[str]
    example_keys: List[str]
    error: str = ""


def load_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def questions(data: Dict[str, object]) -> List[Dict[str, object]]:
    items = data.get("questions", [])
    if not isinstance(items, list):
        raise ValueError("dataset.questions must be a list")
    return items


def schema_required_question_fields(schema: Dict[str, object]) -> List[str]:
    defs = schema.get("$defs", {})
    question_schema = defs.get("question", {}) if isinstance(defs, dict) else {}
    required = question_schema.get("required", REQUIRED_FIELDS)
    return [field for field in required if isinstance(field, str)]


def coverage(items: List[Dict[str, object]], fields: List[str]) -> Dict[str, object]:
    total = len(items)
    by_field = {}
    complete = 0
    for item in items:
        has_all = True
        for field in fields:
            value = item.get(field)
            ok = value not in (None, "", [])
            by_field.setdefault(field, 0)
            by_field[field] += int(ok)
            has_all = has_all and ok
        complete += int(has_all)
    return {
        "complete_records": complete,
        "complete_rate": round(complete / total, 4) if total else 0.0,
        "by_field": {field: {"present": count, "coverage": round(count / total, 4) if total else 0.0} for field, count in by_field.items()},
    }


def non_empty_list_coverage(items: List[Dict[str, object]], field: str) -> Dict[str, object]:
    total = len(items)
    count = sum(1 for item in items if isinstance(item.get(field), list) and len(item[field]) > 0)
    return {"present": count, "coverage": round(count / total, 4) if total else 0.0}


def non_empty_string_coverage(items: List[Dict[str, object]], field: str) -> Dict[str, object]:
    total = len(items)
    count = sum(1 for item in items if isinstance(item.get(field), str) and bool(item[field].strip()))
    return {"present": count, "coverage": round(count / total, 4) if total else 0.0}


def distribution(items: List[Dict[str, object]], field: str) -> Dict[str, int]:
    return dict(sorted(Counter(item.get(field, "<missing>") for item in items).items()))


def instruction_response(item: Dict[str, object]) -> Dict[str, object]:
    return {
        "id": item.get("id"),
        "instruction": item.get("question"),
        "response": item.get("answer"),
        "metadata": {
            "dimension": item.get("dimension"),
            "difficulty_level": item.get("difficulty_level"),
            "source_ref": item.get("source_ref"),
            "reasoning_steps": item.get("reasoning_steps", []),
        },
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
            "source_ref": item.get("source_ref"),
        },
    }


def simple_qa_jsonl(item: Dict[str, object]) -> Dict[str, object]:
    return {
        "id": item.get("id"),
        "question": item.get("question"),
        "answer": item.get("answer"),
        "dimension": item.get("dimension"),
        "source_ref": item.get("source_ref"),
    }


def export_formats(items: List[Dict[str, object]], sample_size: int = 10) -> Tuple[Dict[str, object], Dict[str, object]]:
    sample = items[:sample_size]
    converters = {
        "instruction_response": instruction_response,
        "chatml": chatml,
        "simple_qa_jsonl": simple_qa_jsonl,
    }
    exports = {}
    examples = {}
    for name, converter in converters.items():
        try:
            converted = [converter(item) for item in sample]
            record = FormatExport(
                format_name=name,
                status="success",
                sample_count=len(converted),
                preserved_fields=["id", "question", "answer", "dimension", "source_ref"],
                example_keys=list(converted[0].keys()) if converted else [],
            )
            exports[name] = asdict(record)
            examples[name] = converted[:2]
        except Exception as exc:
            exports[name] = asdict(FormatExport(name, "error", 0, [], [], f"{type(exc).__name__}: {exc}"))
            examples[name] = []
    return exports, examples


def datasets_loader_check(core_path: Path) -> Dict[str, object]:
    if importlib.util.find_spec("datasets") is None:
        return {
            "status": "not_available",
            "library": "datasets",
            "scope": "local_json_optional_check",
            "note": "datasets package is not installed in the current environment; no package was installed during smoke test.",
        }
    try:
        from datasets import load_dataset

        loaded = load_dataset("json", data_files=str(core_path), split="train")
        return {
            "status": "success",
            "library": "datasets",
            "scope": "local_json_optional_check",
            "row_count": len(loaded),
            "features": list(loaded.features.keys()),
        }
    except Exception as exc:
        return {
            "status": "error",
            "library": "datasets",
            "scope": "local_json_optional_check",
            "error": f"{type(exc).__name__}: {exc}",
        }


def summarize_dataset(path: Path, required_fields: List[str]) -> Dict[str, object]:
    data = load_json(path)
    items = questions(data)
    meta = data.get("meta", {})
    return {
        "path": str(path).replace("\\", "/"),
        "meta_total_questions": meta.get("total_questions") if isinstance(meta, dict) else None,
        "question_count": len(items),
        "required_field_coverage": coverage(items, required_fields),
        "dimension_distribution": distribution(items, "dimension"),
        "difficulty_distribution": distribution(items, "difficulty_level"),
        "reasoning_steps_coverage": non_empty_list_coverage(items, "reasoning_steps"),
        "source_ref_coverage": non_empty_string_coverage(items, "source_ref"),
    }


def run(core: str, full: str, schema: str, output: str) -> Dict[str, object]:
    core_path = Path(core)
    full_path = Path(full)
    schema_path = Path(schema)
    schema_data = load_json(schema_path)
    required_fields = schema_required_question_fields(schema_data)
    core_data = load_json(core_path)
    core_items = questions(core_data)
    format_exports, export_examples = export_formats(core_items)
    payload = {
        "schema_version": "1.0",
        "experiment": "track1_ai_ready_smoke",
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "evaluation_mode": "local_json_schema_and_format_export_smoke",
        "scope": {
            "included": ["local JSON loading", "schema required field coverage", "dimension and difficulty distributions", "format conversion smoke"],
            "excluded": ["remote HuggingFace availability claim", "model training", "external API judge"],
        },
        "core_count": len(core_items),
        "full_count": len(questions(load_json(full_path))),
        "schema_required_question_fields": required_fields,
        "core": summarize_dataset(core_path, required_fields),
        "full": summarize_dataset(full_path, required_fields),
        "required_field_coverage": summarize_dataset(core_path, required_fields)["required_field_coverage"],
        "dimension_distribution": distribution(core_items, "dimension"),
        "difficulty_distribution": distribution(core_items, "difficulty_level"),
        "reasoning_steps_coverage": non_empty_list_coverage(core_items, "reasoning_steps"),
        "source_ref_coverage": non_empty_string_coverage(core_items, "source_ref"),
        "format_exports": format_exports,
        "export_examples": export_examples,
        "datasets_loader_status": datasets_loader_check(core_path),
        "boundary_notes": [
            "This smoke test proves local AI-ready consumption and format conversion only.",
            "Remote HuggingFace availability is not claimed unless checked by a separate network-enabled audit.",
            "Format examples are sampled from the first 10 local core records and do not duplicate the full dataset.",
        ],
    }
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(out)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Track 1 AI-ready dataset smoke checks.")
    parser.add_argument("--core", default="benchmark/dataset/core.json")
    parser.add_argument("--full", default="benchmark/dataset/full.json")
    parser.add_argument("--schema", default="benchmark/dataset/schema.json")
    parser.add_argument("--output", default="benchmark/eval/results/track1_ai_ready_smoke.json")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    payload = run(args.core, args.full, args.schema, args.output)
    summary = {
        "core_count": payload["core_count"],
        "full_count": payload["full_count"],
        "required_field_complete_rate": payload["required_field_coverage"]["complete_rate"],
        "reasoning_steps_coverage": payload["reasoning_steps_coverage"]["coverage"],
        "source_ref_coverage": payload["source_ref_coverage"]["coverage"],
        "format_export_status": {key: value["status"] for key, value in payload["format_exports"].items()},
        "datasets_loader_status": payload["datasets_loader_status"]["status"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    ok = (
        payload["core_count"] > 0
        and payload["required_field_coverage"]["complete_rate"] == 1.0
        and all(item["status"] == "success" for item in payload["format_exports"].values())
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
