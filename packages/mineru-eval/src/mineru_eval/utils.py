"""Shared helpers for ControlSci evaluation scripts."""

import json
import uuid
from pathlib import Path


__all__ = [
    "append_jsonl",
    "iter_report_files",
    "load_json",
    "write_json_atomic",
]


def load_json(path):
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json_atomic(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + f".{uuid.uuid4().hex}.tmp")
    tmp_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.replace(path)


def append_jsonl(path, item):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")


def iter_report_files(results_dir):
    results_dir = Path(results_dir)
    patterns = ("*_report.json", "report_*.json", "output_*.json", "eval_*.json")
    seen = set()
    for pattern in patterns:
        for path in sorted(results_dir.glob(pattern)):
            if path.name in {"leaderboard.json", "summary.json"}:
                continue
            if path.name.endswith((".status.json", ".progress.jsonl")):
                continue
            if path in seen:
                continue
            seen.add(path)
            yield path
