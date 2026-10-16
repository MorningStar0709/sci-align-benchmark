"""Split arbitrated candidate pool into core_500 (balanced) and full sets."""
import argparse
import json
import random
from collections import Counter
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]  # restored Phase 5
DEFAULT_INPUT = ROOT / "benchmark" / "dataset" / "merged.json"
DEFAULT_CORE = ROOT / "benchmark" / "dataset" / "core.json"
DEFAULT_FULL = ROOT / "benchmark" / "dataset" / "full.json"

ACCEPTABLE_STATUS = {"auto_passed", "reviewed_kept"}
CORE_PER_DIM = 125
CORE_TOTAL = 500

DIM_ORDER = ["A", "B", "C", "D"]

DIFFICULTY_TARGET = {
    "A": {"L1": 40, "L2": 30, "L3": 30, "L4": 25},
    "B": {"L1":  5, "L2": 30, "L3": 45, "L4": 45},
    "C": {"L1":  0, "L2": 50, "L3": 40, "L4": 35},
    "D": {"L1":  0, "L2": 40, "L3": 45, "L4": 40},
}

DIM_LABELS = {
    "A": "概念定义与数学表达",
    "B": "多步推理与计算求解",
    "C": "敏感性分析与方案对比",
    "D": "完整控制方案设计与综合评估",
}


def load_json(path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    try:
        tmp_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp_path.replace(path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def filter_acceptable(questions):
    return [q for q in questions if isinstance(q, dict) and q.get("consistency_status") in ACCEPTABLE_STATUS]


def _diff_weight(difficulty, dim, diff_counts, per_dim):
    target = DIFFICULTY_TARGET.get(dim, {}).get(difficulty, 0)
    scaled_target = target * per_dim / CORE_PER_DIM if CORE_PER_DIM > 0 else 0
    ratio_target = scaled_target / per_dim if per_dim > 0 else 0
    current_count = diff_counts.get(difficulty, 0)
    ratio_current = current_count / max(1, sum(diff_counts.values()))
    return ratio_target - ratio_current


def select_core(questions):
    by_dim = {d: [] for d in DIM_ORDER}
    for q in questions:
        dim = q.get("dimension")
        if dim in by_dim:
            by_dim[dim].append(q)

    for d in DIM_ORDER:
        if len(by_dim[d]) < CORE_PER_DIM:
            print(f"  [warn] 维度 {d} 仅有 {len(by_dim[d])} 题可用，目标 {CORE_PER_DIM} 题", flush=True)

    selected = []
    for d in DIM_ORDER:
        pool = list(by_dim[d])
        random.shuffle(pool)
        n_select = min(CORE_PER_DIM, len(pool))

        dim_diff_counts = Counter()
        dim_selected = []

        for _ in range(n_select):
            best_idx = max(range(len(pool)), key=lambda i: _diff_weight(
                pool[i].get("difficulty_level", "L1"), d, dim_diff_counts, n_select
            ))
            chosen = pool.pop(best_idx)
            dim_selected.append(chosen)
            dim_diff_counts[chosen.get("difficulty_level", "L1")] += 1

        selected.extend(dim_selected)
        dcounts = {lvl: dim_diff_counts.get(lvl, 0) for lvl in sorted(dim_diff_counts)}
        print(f"  [core] {d}: selected {len(dim_selected)}/{CORE_PER_DIM}  diff={dcounts}", flush=True)

    return selected


def build_meta(total_questions, questions, label, extra=None):
    dimensions = {d: 0 for d in DIM_ORDER}
    for q in questions:
        d = q.get("dimension")
        if d in dimensions:
            dimensions[d] += 1

    status_counts = Counter(q.get("consistency_status") for q in questions)

    meta = {
        "project": "ControlSci — 控制科学结构化语料库与 Sci-Align 跨模态对齐评测基准",
        "version": "1.0",
        "updated": str(date.today()),
        "total_questions": len(questions),
        "dimensions": dimensions,
        "source": f"Split from arbitrated candidate pool ({total_questions} total) — {label}",
        "consistency_status": dict(sorted(status_counts.items())),
    }
    if extra:
        meta.update(extra)
    return meta


def main():
    parser = argparse.ArgumentParser(description="Split arbitrated candidate pool into core_500 and full sets.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Arbitrated merged benchmark JSON.")
    parser.add_argument("--core-output", default=str(DEFAULT_CORE), help="Output core_500 benchmark JSON.")
    parser.add_argument("--full-output", default=str(DEFAULT_FULL), help="Output full benchmark JSON.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for core selection (default: 42).")
    args = parser.parse_args()

    random.seed(args.seed)

    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path}")

    data = load_json(input_path)
    all_questions = data.get("questions", [])
    print(f"[input] {len(all_questions)} total questions", flush=True)

    acceptable = filter_acceptable(all_questions)
    if not acceptable:
        raise SystemExit(
            "No acceptable questions found (auto_passed + reviewed_kept). "
            f"Total questions: {len(all_questions)}. Check consistency_status distribution."
        )
    print(f"[filter] {len(acceptable)} acceptable (auto_passed + reviewed_kept)", flush=True)

    by_status = Counter(q.get("consistency_status") for q in acceptable)
    print(f"  status: {dict(by_status)}", flush=True)

    core_questions = select_core(acceptable)

    core_data = {
        "meta": build_meta(len(all_questions), core_questions, f"core_{CORE_TOTAL}",
                           extra={"dimension_target": {d: CORE_PER_DIM for d in DIM_ORDER},
                                  "dimension_labels": DIM_LABELS,
                                  "selection_seed": args.seed}),
        "questions": core_questions,
    }
    write_json(Path(args.core_output), core_data)
    print(f"[output] core: {args.core_output} ({len(core_questions)} questions)", flush=True)

    full_meta = build_meta(len(all_questions), acceptable, "full")
    full_meta["core_extracted"] = len(core_questions)
    full_data = {"meta": full_meta, "questions": acceptable}
    write_json(Path(args.full_output), full_data)
    print(f"[output] full: {args.full_output} ({len(acceptable)} questions)", flush=True)

    print(f"\nDone. core={len(core_questions)}/{CORE_TOTAL}  full={len(acceptable)}", flush=True)


if __name__ == "__main__":
    main()
