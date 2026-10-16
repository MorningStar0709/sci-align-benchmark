"""n=3 聚合 + 方差分解报告（25 模型 × 2 域）— 写作硬前置收敛点。

背景（docs/plans/paper-gap-supplement-plan.md Task 5）:
    - 25 模型 mean±std over 3 gens 总表（gen1=run1' 替换口径，gen2/3 独立生成）
    - 方差分解: 生成方差（gen 间 overall 方差）vs judge 方差（gen1 记录内
      judge1/judge2 双判差）；争议率（|judge1−judge2| > DISPUTE_TH 题占比）披露
    - 9B 排序对调支点裁决（预注册 §3.5，2026-08-05 冻结）:
      MGPO-EG 9B vs RLVR-C v2 9B，n=3 均值 margin > 2σg(9B)=0.0332 →
      "排序对调"主张；否则退守"弹性梯度差异"叙事
    - 医疗域为单判口径（medical external_score_fn），分表披露不与 control 混算

用法:
    .venv/bin/python scripts/aggregate_n3_final.py
产物:
    docs/reports/n3-aggregation-variance-decomposition.md
    benchmark/eval/n3_regen/n3_aggregation_summary.json
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import uuid
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_n3_regen import (  # noqa: E402
    MODELS,
    OUTPUT_DIR as N3_DIR,
    _detect_run1,
    _gen_path,
)

REPORT_PATH = ROOT / "docs" / "reports" / "n3-aggregation-variance-decomposition.md"
SUMMARY_PATH = N3_DIR / "n3_aggregation_summary.json"

AGG_GENS = [2, 3]          # gen1 由 _detect_run1 提供（run1' 优先）
DISPUTE_TH = 0.25          # 双判分歧争议阈值（披露口径）
PIVOT_A = "mgpo_eg_9b"     # 支点: MGPO-EG 9B（§3.5 旧 epoch 0.4083 为旧数字，此处用新 epoch 复测）
PIVOT_B = "rlvr_c_v2_9b"   # 支点: RLVR-C v2 9B
TWO_SIGMA_9B = 0.0332      # 9B 口径 2σg（σg=0.0166 实测）

# 控制域曲面坐标：key → (柱, 臂)。显式映射，禁止从 key 名字猜柱子。
CONTROL_SURFACE = {
    "raw_qwen": ("4B", "raw"), "sft_baseline": ("4B", "sft"),
    "rlvr_c_v2": ("4B", "rlvr"), "mgpo_eg": ("4B", "mgpo_eg"),
    "raw_9b": ("9B", "raw"), "sft_9b": ("9B", "sft"),
    "rlvr_c_v2_9b": ("9B", "rlvr"), "mgpo_eg_9b": ("9B", "mgpo_eg"),
    "mgpo_fixed_9b": ("9B", "mgpo_fixed"),
    "raw_gemma4": ("E4B-it", "raw"), "sft_gemma4": ("E4B-it", "sft"),
    "rlvr_c_v2_gemma4": ("E4B-it", "rlvr"), "mgpo_eg_gemma4": ("E4B-it", "mgpo_eg"),
}

# 0.8B / 2B / E2B 三柱此前无任何登记聚合（MODELS 只覆盖 4B/9B/E4B-it）。
# 产物在盘、同窗同池，逐件证据见
# temp/adapter-provenance-2026-09-29/surface-window-matrix.md（T4 矩阵）。
SURFACE_EXTRA = [
    {"key": "raw_08b", "label": "Raw Qwen3.5-0.8B", "column": "0.8B", "arm": "raw",
     "adapter": None, "batch": "0.8b",
     "base_model_id": "principled-intelligence/Qwen3.5-0.8B-text-only",
     "dir": "benchmark/eval/probe_local1b", "stem": "raw_0_8b_control", "gens": [0]},
    {"key": "sft_08b", "label": "SFT Baseline 0.8B", "column": "0.8B", "arm": "sft",
     "adapter": "models/v3_08b_sft/final", "batch": "0.8b", "base_model_id": None,
     "dir": "benchmark/eval/tier2_small", "stem": "sft_08b_gen", "gens": [1, 2, 3]},
    {"key": "rlvr_08b", "label": "RLVR-C v2 0.8B", "column": "0.8B", "arm": "rlvr",
     "adapter": "models/v3_08b_rlvr_c_v2/final", "batch": "0.8b", "base_model_id": None,
     "dir": "benchmark/eval/tier2_small", "stem": "rlvr_c_v2_08b_gen", "gens": [1, 2, 3]},
    {"key": "mgpo_eg_08b", "label": "MGPO-EG 0.8B", "column": "0.8B", "arm": "mgpo_eg",
     "adapter": "models/v3_08b_mgpo_eg/final", "batch": "0.8b", "base_model_id": None,
     "dir": "benchmark/eval/tier2_small", "stem": "mgpo_eg_08b_gen", "gens": [1, 2, 3]},
    {"key": "raw_2b", "label": "Raw Qwen3.5-2B", "column": "2B", "arm": "raw",
     "adapter": None, "batch": "2b",
     "base_model_id": "principled-intelligence/Qwen3.5-2B-text-only",
     "dir": "benchmark/eval/probe_local1b", "stem": "raw_2b_control", "gens": [0]},
    {"key": "sft_2b", "label": "SFT Baseline 2B", "column": "2B", "arm": "sft",
     "adapter": "models/v3_2b_sft/final", "batch": "2b", "base_model_id": None,
     "dir": "benchmark/eval/tier2_small", "stem": "sft_2b_gen", "gens": [1, 2, 3]},
    {"key": "rlvr_2b", "label": "RLVR-C v2 2B", "column": "2B", "arm": "rlvr",
     "adapter": "models/v3_2b_rlvr_c_v2/final", "batch": "2b", "base_model_id": None,
     "dir": "benchmark/eval/tier2_small", "stem": "rlvr_c_v2_2b_gen", "gens": [1, 2, 3]},
    {"key": "mgpo_eg_2b", "label": "MGPO-EG 2B", "column": "2B", "arm": "mgpo_eg",
     "adapter": "models/v3_2b_mgpo_eg/final", "batch": "2b", "base_model_id": None,
     "dir": "benchmark/eval/tier2_small", "stem": "mgpo_eg_2b_gen", "gens": [1, 2, 3]},
    {"key": "raw_e2b", "label": "Raw Gemma-4-E2B-it", "column": "E2B", "arm": "raw",
     "adapter": None, "batch": "e2b",
     "base_model_id": "google/gemma-4-E2B-it",
     "dir": "benchmark/eval/gemma4_e2b_300q", "stem": "raw_gemma4_run", "gens": [1, 2, 3]},
    {"key": "sft_e2b", "label": "SFT Baseline E2B", "column": "E2B", "arm": "sft",
     "adapter": "models/gemma4_e2b_sft/final", "batch": "e2b", "base_model_id": None,
     "dir": "benchmark/eval/gemma4_e2b_300q", "stem": "sft_gemma4_run", "gens": [1, 2, 3]},
    {"key": "rlvr_e2b", "label": "RLVR-C v2 E2B", "column": "E2B", "arm": "rlvr",
     "adapter": "models/gemma4_e2b_rlvr_c_v2/final", "batch": "e2b", "base_model_id": None,
     "dir": "benchmark/eval/gemma4_e2b_300q", "stem": "rlvr_c_v2_gemma4_run",
     "gens": [1, 2, 3]},
    {"key": "mgpo_eg_e2b", "label": "MGPO-EG E2B", "column": "E2B", "arm": "mgpo_eg",
     "adapter": "models/gemma4_e2b_mgpo_eg/final", "batch": "e2b", "base_model_id": None,
     "dir": "benchmark/eval/gemma4_e2b_300q", "stem": "mgpo_eg_gemma4_run", "gens": [1, 2, 3]},
]


def _load_product(path: Path) -> dict | None:
    """加载评测产物；缺失/损坏/非 dict/未收官（complete≠True）返回 None。

    complete 门禁: 超时 kill 残留的半截产物不得计入 n=3 聚合与支点裁决。
    """
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if not isinstance(data, dict) or not data.get("complete"):
        return None
    return data


def _overall(data: dict) -> float | None:
    """产物 overall 分（优先 overall_score 字段，缺失退化为 records 均值）。"""
    if isinstance(data.get("overall_score"), (int, float)):
        return data["overall_score"]
    recs = data.get("records", [])
    scores = [r["score"] for r in recs
              if isinstance(r, dict) and isinstance(r.get("score"), (int, float))]
    return sum(scores) / len(scores) if scores else None


def _judge_stats(data: dict) -> dict | None:
    """gen1 记录双判差统计: mean|j1−j2| 与争议率（|diff|>DISPUTE_TH）。"""
    diffs = []
    for r in data.get("records", []):
        if not isinstance(r, dict):
            continue
        j1, j2 = r.get("judge1_score"), r.get("judge2_score")
        if isinstance(j1, (int, float)) and isinstance(j2, (int, float)):
            diffs.append(abs(j1 - j2))
    if not diffs:
        return None
    return {
        "n_dual_records": len(diffs),
        "mean_abs_judge_diff": round(sum(diffs) / len(diffs), 4),
        "dispute_rate": round(sum(1 for d in diffs if d > DISPUTE_TH) / len(diffs), 4),
    }


def _collect_model(m: dict) -> dict:
    """单模型聚合: gen1(run1' 优先) + gen2/3 → mean±std + 方差分解素材。"""
    row: dict = {"key": m["key"], "label": m["label"], "domain": m["domain"],
                 "batch": m["batch"]}
    gen_scores: dict[str, float] = {}
    run1 = _detect_run1(m)
    if run1:
        data = _load_product(Path(run1["path"]))
        if data:
            ov = _overall(data)
            if ov is not None:
                gen_scores[f"gen1({run1['source']})"] = ov
                row["judge_stats"] = _judge_stats(data)
    for g in AGG_GENS:
        data = _load_product(_gen_path(m["key"], g))
        if data:
            ov = _overall(data)
            if ov is not None:
                gen_scores[f"gen{g}"] = ov
    row["gen_scores"] = gen_scores
    return _summarize(row, gen_scores)


def _summarize(row: dict, gen_scores: dict) -> dict:
    """六柱共用的均值/波动口径——同一段代码，禁止另立一套。

    宁缺不造：只有 1 个 gen 的臂不给 std 与 gen_variance（0.8B/2B 的 raw 即此况）。
    """
    column, arm = CONTROL_SURFACE.get(row["key"], (row.get("column"), row.get("arm")))
    row["column"], row["arm"] = column, arm
    rounded = {k: round(v, 4) for k, v in gen_scores.items()}
    row["gen_scores"] = rounded
    vals = list(rounded.values())
    row["n_gens"] = len(vals)
    if vals:
        row["mean"] = round(sum(vals) / len(vals), 4)
        row["std"] = round(statistics.stdev(vals), 4) if len(vals) >= 2 else None
        row["gen_variance"] = round(statistics.variance(vals), 6) if len(vals) >= 2 else None
    return row


def _collect_extra(spec: dict) -> dict:
    """此前未登记的三柱（0.8B/2B/E2B），汇入同一段 `_summarize`。

    它们的产物散在 tier2_small / probe_local1b / gemma4_e2b_300q，命名既有
    `_gen{N}` 也有 `_run{N}`，故按注册表显式给 dir+stem+gens，不靠猜文件名。
    """
    row = {"key": spec["key"], "label": spec["label"], "domain": "control",
           "batch": spec["batch"], "column": spec["column"], "arm": spec["arm"],
           "adapter": spec["adapter"], "base_model_id": spec["base_model_id"]}
    gen_scores: dict[str, float] = {}
    for g in spec["gens"]:
        name = f"{spec['stem']}{g}.json" if g else f"{spec['stem']}.json"
        data = _load_product(ROOT / spec["dir"] / name)
        if not data:
            continue
        ov = _overall(data)
        if ov is not None:
            gen_scores[f"gen{g}" if g else "single"] = ov
            row["judge_stats"] = _judge_stats(data)
    return _summarize(row, gen_scores)


def _pivot_verdict(rows: list[dict]) -> dict:
    """9B 排序对调支点裁决（预注册 §3.5）: margin > 2σg(9B) → 排序对调。"""
    by_key = {r["key"]: r for r in rows}
    a, b = by_key.get(PIVOT_A), by_key.get(PIVOT_B)
    result = {"pivot": f"{PIVOT_A} vs {PIVOT_B}", "threshold_2sigma_9b": TWO_SIGMA_9B}
    if not (a and b and a.get("n_gens", 0) >= 3 and b.get("n_gens", 0) >= 3):
        result["verdict"] = "insufficient_data"
        result["reason"] = "支点模型 n=3 未齐备（聚合前置任务未收官）"
        return result
    margin = round(a["mean"] - b["mean"], 4)
    result["mean_a"], result["mean_b"] = a["mean"], b["mean"]
    result["margin"] = margin
    if margin > TWO_SIGMA_9B:
        result["verdict"] = "ranking_swap"
        result["claim"] = "排序对调主张成立（MGPO-EG 9B 显著高于 RLVR-C v2 9B）"
    else:
        result["verdict"] = "elastic_gradient"
        result["claim"] = "退守'弹性梯度差异'叙事，废掉'完整对调'表述（预注册分流）"
    return result


def _render_md(rows: list[dict], pivot: dict) -> str:
    """渲染方差分解报告 markdown（control / medical 分表）。"""
    lines = [
        "# n=3 聚合与方差分解报告",
        "",
        f"> **生成时间**: {datetime.now():%Y-%m-%d %H:%M} · "
        "工具: `scripts/aggregate_n3_final.py`",
        "> **口径**: gen1=run1'（新 judge epoch 重判，优先）/ 旧 run1（回退）；"
        "gen2/3 独立生成；mean±std 为样本口径",
        f"> **争议率阈值**: |judge1−judge2| > {DISPUTE_TH}（披露口径）",
        "",
    ]
    for domain, note in (("control", "dual-judge 全维度"),
                         ("medical", "单判口径（medical external_score_fn），"
                                     "judge 方差列不适用")):
        d_rows = [r for r in rows if r["domain"] == domain]
        lines += [f"## {domain} 域（{note}）", "",
                  "| 模型 | 柱 | 臂 | batch | n | mean±std | gen 方差 | judge 均差 | 争议率 |",
                  "|:-----|:--:|:--:|:-----:|:-:|:--------:|:--------:|:----------:|:------:|"]

        def _cell(value) -> str:
            """None 一律显示 n/a——单 gen 行没有波动值，不得渲染成 0 或留空。"""
            return "n/a" if value is None else f"{value}"

        for r in d_rows:
            mean_s = (f"{r['mean']:.4f}±{r['std']:.4f}" if r.get("std") is not None
                      else (f"{r['mean']:.4f}±n/a" if r.get("mean") is not None else "n/a"))
            js = r.get("judge_stats") or {}
            lines.append(
                f"| {r['label']} | {_cell(r.get('column'))} | {_cell(r.get('arm'))} | "
                f"{r['batch']} | {r['n_gens']} | {mean_s} | "
                f"{_cell(r.get('gen_variance'))} | "
                f"{_cell(js.get('mean_abs_judge_diff'))} | "
                f"{_cell(js.get('dispute_rate'))} |")
        lines.append("")
    n_incomplete = [r["key"] for r in rows if r["n_gens"] < 3]
    lines += ["## 9B 排序对调支点裁决（预注册 §3.5）", ""]
    if pivot["verdict"] == "insufficient_data":
        lines += [f"**状态**: {pivot['reason']}", ""]
    else:
        lines += [
            f"- MGPO-EG 9B mean = {pivot['mean_a']:.4f}，"
            f"RLVR-C v2 9B mean = {pivot['mean_b']:.4f}",
            f"- margin = {pivot['margin']:+.4f}（阈值 2σg = {TWO_SIGMA_9B}）",
            f"- **verdict = {pivot['verdict']}** — {pivot['claim']}", ""]
    if n_incomplete:
        lines += [f"> ⚠️ n<3 模型（数据未齐，不参与定稿）: {', '.join(n_incomplete)}",
                  ""]
    return "\n".join(lines)


def _atomic_write(out_path: Path, text: str) -> None:
    """原子写入（pid+uuid tmp → os.replace）；text 为已序列化内容。"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name(f"{out_path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, out_path)


def main() -> int:
    parser = argparse.ArgumentParser(description="n=3 聚合 + 方差分解报告")
    parser.add_argument("--json-only", action="store_true",
                        help="仅落盘 JSON summary，不渲染 markdown")
    args = parser.parse_args()

    rows = [_collect_model(m) for m in MODELS]
    rows += [_collect_extra(s) for s in SURFACE_EXTRA]
    pivot = _pivot_verdict(rows)
    summary = {"generated_at": datetime.now().isoformat(timespec="seconds"),
               "agg_gens": AGG_GENS, "dispute_threshold": DISPUTE_TH,
               "rows": rows, "pivot": pivot}
    _atomic_write(SUMMARY_PATH, json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"[SAVED] {SUMMARY_PATH.relative_to(ROOT)}")

    if not args.json_only:
        _atomic_write(REPORT_PATH, _render_md(rows, pivot))
        print(f"[SAVED] {REPORT_PATH.relative_to(ROOT)}")

    n_full = sum(1 for r in rows if r["n_gens"] >= 3)
    print(f"[DONE] {n_full}/{len(rows)} 模型 n≥3；pivot verdict={pivot['verdict']}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as e:  # noqa: BLE001 — 顶层兜底，防裸栈崩溃
        print(f"[FATAL] {type(e).__name__}: {e}", file=sys.stderr)
        raise SystemExit(1)
