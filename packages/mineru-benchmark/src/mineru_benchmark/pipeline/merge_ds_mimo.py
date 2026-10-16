"""Merge DeepSeek + MiMo generated questions into core_v3.json.

Usage:
    python -m mineru_benchmark.pipeline.merge_ds_mimo
    python -m mineru_benchmark.pipeline.merge_ds_mimo --ds benchmark/dataset/gen_deepseek.json --mm benchmark/dataset/gen_mimo.json --output benchmark/dataset/core_v3.json
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
DEFAULT_DS = ROOT / "benchmark" / "dataset" / "gen_deepseek.json"
DEFAULT_MM = ROOT / "benchmark" / "dataset" / "gen_mimo.json"
DEFAULT_OUT = ROOT / "benchmark" / "dataset" / "core_v3.json"


def load_questions(path: Path) -> list[dict]:
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"[ERROR] Failed to load {path}: {e}", file=sys.stderr, flush=True)
        raise SystemExit(1) from e
    result = data.get("questions", data if isinstance(data, list) else None)
    if result is None:
        print(f"[WARNING] {path}: expected dict with 'questions' key or a list, got {type(data).__name__}", flush=True)
        return []
    return result


def write_json_atomic(path: Path, data: dict) -> None:
    tmp = path.with_suffix(f"{path.suffix}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def merge(ds_path: Path, mm_path: Path, output_path: Path, dedup: str = "keep-first") -> dict:
    ds_questions = load_questions(ds_path)
    mm_questions = load_questions(mm_path)

    print(f"DeepSeek: {len(ds_questions)} questions", flush=True)
    print(f"MiMo:     {len(mm_questions)} questions", flush=True)

    # Check for missing 'id' fields
    ds_ids = {q["id"] for q in ds_questions if "id" in q}
    mm_ids = {q["id"] for q in mm_questions if "id" in q}
    missing_ds = sum(1 for q in ds_questions if "id" not in q)
    missing_mm = sum(1 for q in mm_questions if "id" not in q)
    if missing_ds or missing_mm:
        print(f"WARNING: {missing_ds} DeepSeek + {missing_mm} MiMo questions missing 'id' field", flush=True)

    # ID overlap detection with dedup strategy
    overlap = ds_ids & mm_ids
    if overlap:
        print(f"WARNING: {len(overlap)} ID collisions: {sorted(overlap)[:5]}...", flush=True)
        if dedup == "keep-first":
            mm_questions = [q for q in mm_questions if q.get("id") not in overlap]
            print(f"  Dedup ({dedup}): kept {len(ds_questions)} DS, filtered {len(overlap)} MM duplicates", flush=True)
        elif dedup == "raise":
            raise SystemExit(f"{len(overlap)} ID collisions detected and dedup=raise. Resolve before merging.")
        # "keep-both" — do nothing, keep all

    merged = ds_questions + mm_questions
    total = len(merged)
    print(f"Merged:   {total} total", flush=True)

    # Stats
    dim_counts = Counter(q.get("dimension", "?") for q in merged)
    diff_counts = Counter(q.get("difficulty_level", "?") for q in merged)
    status_counts = Counter(q.get("consistency_status", "?") for q in merged)
    model_counts = Counter(q.get("model_source", "?") for q in merged)

    print(f"\nDimensions: {dict(sorted(dim_counts.items()))}", flush=True)
    print(f"Difficulties: {dict(sorted(diff_counts.items()))}", flush=True)
    print(f"Status: {dict(sorted(status_counts.items()))}", flush=True)
    print(f"Models: {dict(sorted(model_counts.items()))}", flush=True)

    # Build output
    from datetime import date
    meta = {
        "project": "ControlSci — Sci-Align v3 benchmark (paper-anchored, cross-audited)",
        "version": "3.0",
        "updated": str(date.today()),
        "total_questions": total,
        "dimensions": dict(sorted(dim_counts.items())),
        "source": f"DeepSeek ({len(ds_questions)}) + MiMo ({len(mm_questions)}) dual-blind generation",
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(output_path, {"meta": meta, "questions": merged})
    print(f"\nSaved: {output_path}", flush=True)
    return {"meta": meta, "questions": merged}


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Merge DeepSeek + MiMo outputs into core_v3.json")
    parser.add_argument("--ds", type=Path, default=DEFAULT_DS, help="DeepSeek output")
    parser.add_argument("--mm", type=Path, default=DEFAULT_MM, help="MiMo output")
    parser.add_argument("--output", "-o", type=Path, default=DEFAULT_OUT, help="Output path")
    parser.add_argument("--dedup", choices=["keep-first", "keep-both", "raise"], default="keep-first",
                        help="ID dedup strategy: keep-first (DS priority), keep-both, or raise")
    args = parser.parse_args()

    if not args.ds.exists():
        print(f"[ERROR] DeepSeek file not found: {args.ds}", flush=True)
        return 1
    if not args.mm.exists():
        print(f"[ERROR] MiMo file not found: {args.mm}", flush=True)
        return 1

    merge(args.ds, args.mm, args.output, dedup=args.dedup)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
