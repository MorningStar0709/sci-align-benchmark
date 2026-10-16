"""Shared helpers — JSON I/O, file iteration, atomic writes.

Migrated from ``benchmark.eval.utils``.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path


__all__ = [
    "append_jsonl",
    "iter_report_files",
    "load_json",
    "write_json_atomic",
]


def load_json(path: str | Path) -> dict | list:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json_atomic(path: str | Path, data: dict | list) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
    tmp_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.replace(path)


def append_jsonl(path: str | Path, item: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")


def iter_report_files(results_dir: str | Path):
    results_dir = Path(results_dir)
    patterns = ("*_report.json", "report_*.json", "output_*.json", "eval_*.json")
    seen: set[Path] = set()
    for pattern in patterns:
        for path in sorted(results_dir.glob(pattern)):
            stem = path.stem
            if any(part in {"leaderboard", "summary"} for part in stem.split("_")):
                continue
            if stem.endswith((".status", ".progress")):
                continue
            if path in seen:
                continue
            seen.add(path)
            yield path
