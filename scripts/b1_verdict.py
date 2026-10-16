"""B1 转正判据评估 — sft_baseline(4B) vs raw_9b 逐题配对 delta（n=3 三对）。

背景（docs/plans/paper-gap-supplement-plan.md Task 3）:
    B1 候选发现"训练后 4B 反超未训练 9B（新 epoch）"当前 n=2，
    margin +0.0116 < 2σg 最小口径 0.0132，暂不可转正。
    补 gen4 凑齐 n=3 后按预注册判据评估，定案"转正 or 表格数字"。

判据（docs/reports/strategy-eval-fix-paper-assets.md §3.5 预注册，
2026-08-05 固化，开跑后冻结）:
    - 3 次独立生成全部同向（4B > 9B）且均值 margin > 2σg → promoted
    - 否则 table_number（预注册分流，非失败）
    - 双口径披露: 4B 口径 2σg=0.0132（σg=0.0066）/ 9B 口径 2σg=0.0332（σg=0.0166）
      verdict 以 4B 口径为主判据（B1 主体为 4B 模型）

配对方式: gen{2,3,4} 两两组合共三对 (2,3)(2,4)(3,4)，每对取两模型
同题均分差（先逐题求两 gen 均值，再做 4B−9B 差），三对 margin 计数方向。

用法:
    .venv/bin/python scripts/b1_verdict.py            # 评估并落盘 verdict
    .venv/bin/python scripts/b1_verdict.py --json     # 仅打印 JSON

产物: benchmark/eval/n3_regen/b1_verdict.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
N3_DIR = ROOT / "benchmark" / "eval" / "n3_regen"
OUT_PATH = N3_DIR / "b1_verdict.json"

MODEL_A = "sft_baseline"   # 4B（B1 主张方）
MODEL_B = "raw_9b"         # 9B（未训练基线）
GENS = [2, 3, 4]
N_REQUIRED = 3             # 预注册要求 n=3（三对全部有效）
TWO_SIGMA_4B = 0.0132      # 4B 口径（σg=0.0066，反事实标定段实测）
TWO_SIGMA_9B = 0.0332      # 9B 口径（σg=0.0166，同段实测）


def _load_scores_by_id(path: Path) -> dict[str, float]:
    """加载评测产物逐题分数 {id: score}；缺失/损坏返回空 dict。"""
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    if not isinstance(data, dict) or not data.get("complete"):
        return {}  # 未收官产物（complete=False，超时 kill 残留）不参与预注册裁决
    return {r["id"]: r["score"] for r in data.get("records", [])
            if isinstance(r, dict) and "id" in r
            and isinstance(r.get("score"), (int, float))}


def _pair_margin(scores_a: dict[int, dict[str, float]],
                 scores_b: dict[int, dict[str, float]],
                 g1: int, g2: int) -> float | None:
    """单个 gen 对的 margin：同题两 gen 均值的 (4B − 9B) 差。

    任一模型该对 gen 数据缺失或无公共题时返回 None。
    """
    a1, a2 = scores_a.get(g1), scores_a.get(g2)
    b1, b2 = scores_b.get(g1), scores_b.get(g2)
    if not all((a1, a2, b1, b2)):
        return None
    common = set(a1) & set(a2) & set(b1) & set(b2)
    if not common:
        return None
    deltas = [((a1[i] + a2[i]) - (b1[i] + b2[i])) / 2 for i in common]
    return sum(deltas) / len(deltas)


def _gen_pair_margins(scores_a: dict[int, dict[str, float]],
                      scores_b: dict[int, dict[str, float]],
                      gens: list[int]) -> dict[str, float | None]:
    """全部 gen 两两组合的 margin（如 gen2/3/4 → 三对）。"""
    return {f"gen{g1}_gen{g2}": _pair_margin(scores_a, scores_b, g1, g2)
            for g1, g2 in combinations(gens, 2)}


def _decide(pair_margins: dict[str, float | None],
            threshold_4b: float, threshold_9b: float) -> dict:
    """预注册分流：三对全同向且 mean margin > 2σg → promoted，否则 table_number。"""
    valid = {k: v for k, v in pair_margins.items() if v is not None}
    result = {
        "pair_margins": {k: round(v, 4) if v is not None else None
                         for k, v in pair_margins.items()},
        "n_valid_pairs": len(valid),
        "threshold_2sigma_4b": threshold_4b,
        "threshold_2sigma_9b": threshold_9b,
    }
    if len(valid) < N_REQUIRED:
        result["verdict"] = "insufficient_data"
        result["reason"] = (f"有效 gen 对仅 {len(valid)}/{N_REQUIRED}"
                            "（补 gen 未完成或产物缺失）")
        return result
    mean_margin = sum(valid.values()) / len(valid)
    result["mean_margin"] = round(mean_margin, 4)
    pos = sum(1 for v in valid.values() if v > 0)
    neg = sum(1 for v in valid.values() if v < 0)
    result["direction_count"] = {"4b_ahead": pos, "9b_ahead": neg,
                                 "tie": len(valid) - pos - neg}
    if pos == len(valid):
        # 主判据 4B 口径；9B 口径同步披露（不改变 verdict）
        exceeds = {
            "4b_caliber": mean_margin > threshold_4b,
            "9b_caliber": mean_margin > threshold_9b,
        }
        result["exceeds_threshold"] = exceeds
        if exceeds["4b_caliber"]:
            result["verdict"] = "promoted"
            result["reason"] = (f"三对全同向（4B>9B）且 mean margin "
                                f"{mean_margin:.4f} > 2σg(4B)={threshold_4b}")
        else:
            result["verdict"] = "table_number"
            result["reason"] = (f"三对全同向但 mean margin {mean_margin:.4f} "
                                f"≤ 2σg(4B)={threshold_4b}（预注册分流）")
    else:
        result["verdict"] = "table_number"
        result["reason"] = (f"方向不一致（4B 领先 {pos}/{len(valid)} 对），"
                            "不满足'3 次同向'条件（预注册分流）")
    return result


def _evaluate() -> dict:
    """读盘评估：加载 gen{2,3,4} 产物 → 三对 margin → 预注册分流。"""
    scores: dict[str, dict[int, dict[str, float]]] = {}
    missing: list[str] = []
    for key in (MODEL_A, MODEL_B):
        scores[key] = {}
        for g in GENS:
            path = N3_DIR / f"{key}_gen{g}.json"
            s = _load_scores_by_id(path)
            if s:
                scores[key][g] = s
            else:
                missing.append(path.name)
    result = _decide(_gen_pair_margins(scores[MODEL_A], scores[MODEL_B], GENS),
                     TWO_SIGMA_4B, TWO_SIGMA_9B)
    return {
        "model_a": MODEL_A,
        "model_b": MODEL_B,
        "gens": GENS,
        "criteria_source": "strategy-eval-fix-paper-assets.md §3.5（2026-08-05 冻结）"
                           " + plan Task 3 拍板",
        "deviation_note": "裁决对象按 plan Task 3 改为 sft_baseline vs raw_9b（§3.5 原"
                          "候选 RLVR-C v2 4B vs raw_9b）；方向判据用 gen 两两配对"
                          " margin（plan Step 2 口径，弱于逐代字面条件）",
        "n_questions_per_gen": {
            f"{k}_gen{g}": len(scores[k].get(g, {}))
            for k in (MODEL_A, MODEL_B) for g in GENS},
        "missing_products": missing,
        **result,
    }


def _atomic_write(out_path: Path, payload: dict) -> None:
    """原子写入（pid+uuid tmp → os.replace）。"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name(f"{out_path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, out_path)


def main() -> int:
    parser = argparse.ArgumentParser(description="B1 转正判据评估（n=3 三对配对）")
    parser.add_argument("--json", action="store_true", help="仅打印 JSON，不落盘")
    args = parser.parse_args()

    result = _evaluate()
    if not args.json:
        _atomic_write(OUT_PATH, result)
        print(f"[SAVED] {OUT_PATH.relative_to(ROOT)}")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\n[VERDICT] {result['verdict']} — {result.get('reason', '')}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as e:  # noqa: BLE001 — 顶层兜底，防裸栈崩溃
        print(f"[FATAL] {type(e).__name__}: {e}", file=sys.stderr)
        raise SystemExit(1)
