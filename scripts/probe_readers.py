"""控制域探针读数器（E10 位置偏差 / 开卷 A 维）——矩阵装载件的登记读法。

为什么单开一件：这两条主张此前的留桩理由是「实物只在 temp」「无登记算子」，
09-30 独立复查否证了字面理由——两案的实物都在 `benchmark/` 与 `release/` 登记位置，
缺的只是算子。本件把"缺的这一块"补上：**只读实物、不判定、不改任何阈值或分档规则**。

约定：
- 闭卷对照臂一律走 `release/adapter_manifest.json` 的 `n3_state`，不在本件里猜评测目录名。
- 所有返回值都是**描述量**（计数、均值、配对差），没有 2σg 转正阈值语义。
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

POSITION_BIAS_PRODUCT = "benchmark/eval/e10_position_bias/mgpo_eg_gen3__swapped.json"
OPENBOOK_PRODUCT = "benchmark/eval/openbook_control/student_sft_baseline_openbook.json"
CLOSED_ARM_KEY = "sft_baseline"


READS: set[str] = set()      # 本件实际读过哪些实物，供装载件并入 provenance 的 READ_LOG


def _load(root: Path, rel: str) -> dict:
    path = root / rel
    if not path.is_file():
        raise FileNotFoundError(f"缺实物: {rel}")
    READS.add(rel)
    return json.loads(path.read_text(encoding="utf-8"))


def _by_id(data: dict, field: str = "score") -> dict[str, float]:
    out: dict[str, float] = {}
    for rec in data.get("records") or []:
        if isinstance(rec, dict) and rec.get("id") and isinstance(rec.get(field), (int, float)):
            out[rec["id"]] = float(rec[field])
    return out


def read_position_bias(root: Path) -> dict:
    """E10：判序交换前后逐题终分对。

    件里 `arm_a_records` 是冻结臂（未新调判官），`records` 是交换判官指派后那一臂。
    返回终分不变的题数、变化题号、mean(A−B) 与两种标准差。
    """
    data = _load(root, POSITION_BIAS_PRODUCT)
    arm_a = _by_id({"records": data.get("arm_a_records") or []})
    arm_b = _by_id(data)                       # 顶层 records = 交换判官指派后那一臂
    shared = sorted(set(arm_a) & set(arm_b))
    if not shared:
        raise ValueError(f"{POSITION_BIAS_PRODUCT} 两臂无公共题号，配对不成立")
    diffs = [arm_a[i] - arm_b[i] for i in shared]
    changed = [i for i in shared if abs(arm_a[i] - arm_b[i]) > 1e-12]
    return {
        "product": POSITION_BIAS_PRODUCT,
        "source_artifact": data.get("source_artifact"),
        "arm_b_def": data.get("arm_b"),
        "n_items": len(shared),
        "n_unchanged": len(shared) - len(changed),
        "changed_ids": changed,
        "mean_delta": statistics.fmean(diffs),
        "sd_sample": statistics.stdev(diffs) if len(diffs) > 1 else None,
        "sd_population": statistics.pstdev(diffs),
        "mean_a": data.get("mean_score_a"),
        "mean_b": data.get("mean_score_b"),
    }


def closed_arm_products(root: Path) -> list[str]:
    """闭卷同基线臂的全部登记产物（按 manifest `n3_state`，不猜目录）。"""
    manifest = _load(root, "release/adapter_manifest.json")
    entries = manifest.get("n3_state", {}).get(CLOSED_ARM_KEY, {})
    return sorted(e["path"] for e in entries.values()
                  if isinstance(e, dict) and e.get("path") and (root / e["path"]).is_file())


def read_openbook_a(root: Path) -> dict:
    """开卷 A 维 vs 闭卷同题号 A 维：逐题号配对的差。

    开卷件只跑 75 题子集、只打分维度 A；闭卷件是 300 题四维，故按题号取交集配对，
    并逐个闭卷代次给出差值（不给单一代表值——代次之间本身有跨度）。
    """
    openbook = _load(root, OPENBOOK_PRODUCT)
    open_a = _by_id(openbook)
    per_closed: list[dict] = []
    for rel in closed_arm_products(root):
        closed = _by_id(_load(root, rel))
        shared = sorted(set(open_a) & set(closed))
        if not shared:
            continue
        delta = statistics.fmean(open_a[i] - closed[i] for i in shared)
        per_closed.append({"product": rel, "n_items": len(shared),
                           "closed_a_mean": statistics.fmean(closed[i] for i in shared),
                           "delta": delta})
    if not per_closed:
        raise ValueError("开卷题号与任何闭卷登记产物都不相交，配对不成立")
    deltas = [row["delta"] for row in per_closed]
    return {
        "product": OPENBOOK_PRODUCT,
        "open_a_mean": statistics.fmean(open_a.values()),
        "n_items_open": len(open_a),
        "closed_products": per_closed,
        "delta_min": min(deltas),
        "delta_max": max(deltas),
        "all_positive": all(d > 0 for d in deltas),
        "item_level_test": None,
    }


E11_PRODUCT = "benchmark/eval/e11_judge_ppl/e11_judge_ppl_result.json"


def read_judge_ppl_correlation(root: Path) -> dict:
    """E11：PPL × 判官行为分的秩相关——从入仓件逐行**重算**，不照抄件内 rho。

    件内自带 `spearman_all` / `spearman_finetuned_only`，本函数重算并与之判等，
    为的是让「ρ=−0.405」这个印进 A 篇 §intro 的数有一条仓内可复算路径
    （此前它唯一的实物 `temp/e11_judge_ppl_result.json` 在指向 /mnt/rdata 的 symlink 下，
    仓外、未跟踪 ⇒ 公开仓面上不可复算）。09-30 由作者令入仓。

    **样本定义固定成一种并写明含不含 raw**（CALIBERS §2#12 的在册要求）：
    正文引的是 `n_all`＝13 行、含 3 个未训练 base，A 篇 §intro 那句已写
    "thirteen control-domain evaluations, untrained bases included"。
    """
    from scipy.stats import spearmanr

    data = _load(root, E11_PRODUCT)
    rows = data.get("rows") or []
    if not rows:
        raise ValueError(f"{E11_PRODUCT} 没有 rows，无从重算")
    pairs_all = [(float(r["chatml_ppl"]), float(r["judge_overall"])) for r in rows]
    finetuned = [r for r in rows if "raw" not in str(r.get("model", ""))]
    pairs_ft = [(float(r["chatml_ppl"]), float(r["judge_overall"])) for r in finetuned]

    def rho(pairs):
        stat, p = spearmanr([a for a, _ in pairs], [b for _, b in pairs])
        return round(float(stat), 4), float(p)

    n_all, n_ft = rho(pairs_all), rho(pairs_ft)
    recorded_all = data["spearman_all"]
    recorded_ft = data["spearman_finetuned_only"]
    return {
        "product": E11_PRODUCT,
        "n_rows": len(rows),
        "rho": n_all[0], "p_value": n_all[1],
        "rho_finetuned_only": n_ft[0], "p_finetuned_only": n_ft[1],
        "recomputed_matches_file": (abs(n_all[0] - float(recorded_all["rho"])) < 1e-4
                                    and abs(n_ft[0] - float(recorded_ft["rho"])) < 1e-4),
        "raw_models_included": [r["model"] for r in rows if "raw" in str(r.get("model", ""))],
        "caveat": data.get("caveat"),
    }


MEDICAL_RUNS = ("run1", "run1p", "run2", "run3")


def read_medical_gain_vs_base(root: Path, arm_stem: str, base_stem: str) -> dict:
    """医疗域增益：`medical_dual_rejudge` 四 run 池化均值相减（口径＝vs base）。

    为什么必须有这一支：医疗主张在册的两个增益数（4B +0.1306 / 9B +0.1363）出自这个池，
    而矩阵此前取的是 `n3_aggregation_summary` 的 3-gen 臂均值（另一道题、另一套代集合），
    两套池平均不出彼此，才被误记成「在册陈述与现算冲突」。
    """
    def pooled(stem: str) -> dict:
        vals, per = [], {}
        for run in MEDICAL_RUNS:
            rel = f"benchmark/eval/medical_dual_rejudge/{stem}_{run}.json"
            data = _load(root, rel)
            score = float(data["overall_score"])
            vals.append(score)
            per[run] = {"score": score, "n": int(data["total_questions"])}
        return {"scores": per, "mean": statistics.fmean(vals)}

    arm, base = pooled(arm_stem), pooled(base_stem)
    if arm["scores"]["run1"]["n"] != base["scores"]["run1"]["n"]:
        raise ValueError("两臂题池规模不同，增益不成立")
    return {"pool": "medical_dual_rejudge 4-run（vs base）",
            "n_items": arm["scores"]["run1"]["n"],
            "arm": arm_stem, "base": base_stem,
            "arm_mean": arm["mean"], "base_mean": base["mean"],
            "gain_vs_base": arm["mean"] - base["mean"],
            "arm_runs": arm["scores"], "base_runs": base["scores"]}


E12_E4_PRODUCT = "benchmark/eval/e12_e4_medical/e12_e4_medical_result.json"
# 在册值（B 篇台账 E4 / `tab:verdicts-medical`）——读到的声明值须与之判等。
MEDICAL_SIGMA_REGISTERED = 0.0023
MEDICAL_2SIGMA_REGISTERED = 0.0045


def read_medical_sigma(root: Path) -> dict:
    """医疗阈值 σg/2σg：从**入仓件**读声明值，并判等**在册常量**。

    这是**登记一致性检查，不是重算**——必须说清楚：件内 σg 是回显常量
    （同对照域的 `analysis.json#judgement.sigma`），件内只给 `per_model_run_std`
    逐模型标准差，`sqrt(mean(var))` = 0.0039 ≠ 声明 0.0023 ⇒ 由件内数据**重算不出**
    该常量；真重算需标定段逐 run 原始分，那在 `medical_dual_rejudge/` 题级件里，
    属另一件事。所以本函数只保证「正文那个 0.0023 出自哪个仓内件、且件内声明与在册值一致」。

    为什么需要它：该件此前唯一实物在 `temp/`（指向 /mnt/rdata 的 symlink、仓外未跟踪），
    `claim_evidence._load_medical_scale_axis` 的 docstring 自陈「真缺口是那一条的复算源在
    temp/……进不了调用链」。10-01 入仓后本函数把它接回调用链。
    """
    data = _load(root, E12_E4_PRODUCT)
    e4 = data["e4"]
    sigma = float(e4["sigma_g_medical"])
    two = float(e4["threshold_2sigma"])
    return {
        "product": E12_E4_PRODUCT,
        "sigma_g": sigma,
        "threshold_2sigma": two,
        "matches_registered": (abs(sigma - MEDICAL_SIGMA_REGISTERED) < 1e-9
                               and abs(two - MEDICAL_2SIGMA_REGISTERED) < 1e-9),
        "declared_formula": e4.get("sigma_g_formula"),
        "n_verdicts": len(e4.get("verdicts") or []),
        "verdict_margins": {v["pair"]: float(v["mean_margin"])
                            for v in (e4.get("verdicts") or [])},
    }
