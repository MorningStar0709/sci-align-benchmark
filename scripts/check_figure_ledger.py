#!/usr/bin/env python3
"""图内数字 ↔ number_audit.md ↔ 源 JSON 三方对账器（只读，零网络）。

背景：论文出图脚本里凡内嵌台账常量者（fig3_drift 的 LEDGER_MEAN/LEDGER_SD、
fig3_ablation 的 EXPECTED_MEANS/EXPECTED_CF/THRESHOLDS），一旦源产物或正文口径回写
就会与图脱节；此前只有跑图时的脚本内建 guard 能拦。本脚本把三方对账固化为独立检查：

  源侧（唯一真值）  从 benchmark/eval/** 产物现场复算，records 级、未取整。
  台账侧            papers/<paper>/number_audit.md 是否登记该值。
  图侧              draft/figures/*.pdf 文本层里的数字 ＋ 出图脚本内嵌常量（AST 读）。

canonical 一律由**全精度源值**舍入到展示位，绝不用已舍入的中间量再减，因此天然规避
F4 的双舍坑（2B：全精度 0.002167→+0.0022；若用已登记的臂均值再减 0.3274−0.3253 会得
+0.0021，与台账/图上的 +0.0022 冲突）。

腿的语义：某腿传 None 表示"该量在这一侧本就不出现"（如 delta 不是脚本常量、臂均值
不进图内文本），跳过；传了文本却没匹配到，才算不一致。

诚实边界：台账腿与 PDF 腿都是 token 存在性检查，不校验挂在第几行——逐行回链绑定是
audit_data_trace.py 的职责，两项检查互补。出图脚本一律 AST 读取，绝不 exec/import（它会
matplotlib 出图并写盘）；唯一 import 例外是 surface_stats（纯统计模块，无 matplotlib、
无模块级写盘），补充腿复用它以保增益算法单一源。

补充腿（见 figure_ledger_extra.py）：A fig1/fig4 十八格增益、A fig5 九条
医疗 margin、B fig3 逐模型漂移 23 格、B fig4 逐维中位数 20 格——四组此前只靠出图脚本
内建 guard（出口时间检查），现由本件独立复核。

用法：.venv/bin/python scripts/check_figure_ledger.py
退出码：0 = 全部一致；1 = 有不一致清单；2 = 产物/图件缺失或不可解析。
"""
from __future__ import annotations

import ast
import json
import re
import statistics
import sys
from pathlib import Path

import pymupdf

RUN1_DIRS = {"4b": "v3_phase4_300", "9b": "v3_9b_300q", "gemma": "gemma4_300q"}
# control 域 11 模型（排除 rlvr_c_v2 4B 与 mgpo_eg_9b，其 gen1 不保留），剔任一侧假 0。
# 口径出处 docs/CALIBERS.md §2#10 与 analysis-judge-drift-run1-rejudge.md §二；不剔假 0
# 会得 +0.0645（11 模型口径）/ +0.0634（13 对口径），均与正文 +0.0606 冲突。
CONTROL_MODELS = (
    ("raw_qwen", "4b"), ("sft_baseline", "4b"), ("mgpo_eg", "4b"),
    ("raw_9b", "9b"), ("sft_9b", "9b"), ("mgpo_fixed_9b", "9b"), ("rlvr_c_v2_9b", "9b"),
    ("raw_gemma4", "gemma"), ("sft_gemma4", "gemma"),
    ("mgpo_eg_gemma4", "gemma"), ("rlvr_c_v2_gemma4", "gemma"),
)
ABLATION_GENS = (("0.8B", "tier2_small", [1, 2, 3]), ("2B", "tier2_small", [1, 2, 3]),
                 ("9B", "n3_regen", [2, 3]))


def repo_root() -> Path:
    """cwd 优先、再退到 __file__：脚本可能被任意目录调用。"""
    for base in (Path.cwd().resolve(), Path(__file__).resolve().parent):
        for cand in (base, *base.parents):
            if (cand / "papers").is_dir() and (cand / "benchmark" / "eval").is_dir():
                return cand
    raise SystemExit(f"repo root not found (cwd={Path.cwd()})")


def _json(path: Path):
    if not path.is_file():
        raise SystemExit(f"[GATE ERROR] missing product: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


# ---- 源侧复算（未取整）----

def _is_fake_zero(record: dict) -> bool:
    if record.get("score") != 0:
        return False
    return any("failure" in str(x).lower() or "error" in str(x).lower()
               for x in (record.get("issues") or []))


def _records(path: Path) -> dict[str, dict]:
    return {r["id"]: r for r in _json(path).get("records", []) if r.get("id")}


def recompute_drift(root: Path, exclude_fake_zero: bool = True):
    """control 域逐模型 run1 vs run1' 配对 delta → (mean, sd, 配对题数, 同向数, 模型数)。"""
    eval_dir = root / "benchmark" / "eval"
    deltas: list[float] = []
    paired_total = 0
    for key, batch in CONTROL_MODELS:
        old = _records(eval_dir / RUN1_DIRS[batch] / f"{key}_run1.json")
        new = _records(eval_dir / "run1_rejudge" / f"{key}_run1p.json")
        shared = sorted(set(old) & set(new))
        if not shared:
            raise SystemExit(f"[GATE ERROR] {key}: no shared item ids between epochs")
        paired = ([q for q in shared if not (_is_fake_zero(old[q]) or _is_fake_zero(new[q]))]
                  if exclude_fake_zero else shared)
        paired_total += len(paired)
        deltas.append(statistics.fmean(new[q]["score"] for q in paired)
                      - statistics.fmean(old[q]["score"] for q in paired))
    return (statistics.fmean(deltas), statistics.stdev(deltas), paired_total,
            sum(1 for d in deltas if d > 0), len(deltas))


def _mean_over(directory: Path, stem: str, gens: list[int]) -> float:
    return statistics.fmean(float(_json(directory / f"{stem}_gen{g}.json")["overall_score"])
                            for g in gens)


def recompute_ablation(root: Path) -> dict[str, dict[str, float]]:
    """EG−Fixed 三档：返回全精度 fixed/eg 臂均值与 delta（9B 只有 gen2+gen3）。"""
    out: dict[str, dict[str, float]] = {}
    for scale, sub, gens in ABLATION_GENS:
        directory = root / "benchmark" / "eval" / sub
        suffix = {"0.8B": "_08b", "2B": "_2b", "9B": "_9b"}[scale]
        fixed = _mean_over(directory, f"mgpo_fixed{suffix}", gens)
        eg = _mean_over(directory, f"mgpo_eg{suffix}", gens)
        out[scale] = {"fixed": fixed, "eg": eg, "delta": eg - fixed}
    return out


def recompute_cf_margins(root: Path) -> dict[str, float]:
    d = _json(root / "benchmark" / "eval" / "counterfactual" / "analysis.json")
    return {scale: float(d[key]["statistics"]["judgement"]["mean_margin"])
            for key, scale in (("4b", "4B"), ("9b", "9B"))}


def recompute_b1(root: Path) -> dict[str, float]:
    b1 = _json(root / "benchmark" / "eval" / "n3_regen" / "b1_verdict.json")
    return {"mean_margin": float(b1["mean_margin"]),
            "threshold_4b": float(b1["threshold_2sigma_4b"]),
            "threshold_9b": float(b1["threshold_2sigma_9b"])}


def recompute_anchor(root: Path) -> dict[str, int]:
    """锚点档：从 triplets 逐题重算栅格统计（与 fig4_anchor 的算法独立实现）。"""
    tri = _json(root / "benchmark" / "eval" / "judge_audit" / "anchor_report.json")["triplets"]
    if not tri:
        raise SystemExit("[GATE ERROR] anchor_report.json has no triplets")
    good = [float(t["good"]) for t in tri]
    bad = [float(t["bad"]) for t in tri]
    empty = [float(t["empty"]) for t in tri]
    ties = sum(1 for g, b in zip(good, bad) if g == b)
    inverted = sum(1 for g, b in zip(good, bad) if g < b)
    return {"n": len(tri), "ties": ties, "inverted": inverted,
            "faithful_ahead": len(tri) - ties - inverted,
            "fully_ordered": sum(1 for g, b, e in zip(good, bad, empty) if g > b > e),
            "evasive_at_zero": sum(1 for e in empty if e == 0)}


# ---- 图侧读取 ----

def pdf_text(path: Path) -> str:
    if not path.is_file():
        raise SystemExit(f"[GATE ERROR] missing figure pdf: {path}")
    return pymupdf.open(str(path))[0].get_text()


def normalize_signs(text: str) -> str:
    """U+2212 / en-dash 统一成 ASCII 连字符，否则台账里的「−0.0072」永远匹配不上。"""
    return text.replace("\u2212", "-").replace("\u2013", "-")


def parse_pdf_number(text: str, anchor: str) -> float:
    """取 anchor 之后第一个带符号小数——证明图内数字确实可机器读。"""
    norm = normalize_signs(text)
    idx = norm.find(anchor)
    if idx < 0:
        raise SystemExit(f"[GATE ERROR] anchor not in figure text: {anchor!r}")
    m = re.search(r"[+-]?\d+\.\d+", norm[idx + len(anchor):])
    if not m:
        raise SystemExit(f"[GATE ERROR] no number after anchor {anchor!r}")
    return float(m.group())


def script_constants(path: Path) -> dict:
    """AST 抽出图脚本模块级字面量常量；绝不 import（模块顶层会 matplotlib 出图写盘）。

    按 AST 结构分型处理（实测五支脚本共 4 种形态）：
      Name←Constant / Name←Dict / Name←Tuple  → 直接 literal_eval；
      Tuple←Tuple（`LEDGER_MEAN, LEDGER_SD = 0.0606, 0.0058`）→ 按位 zip 展开；
      Name←BinOp（`CHANCE_BASELINE = 1 / 6`）→ 非字面量，跳过不报错。
    """
    if not path.is_file():
        raise SystemExit(f"[GATE ERROR] missing figure script: {path}")
    consts: dict = {}
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if not isinstance(node, ast.Assign):
            continue
        try:
            value = ast.literal_eval(node.value)
        except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
            continue
        target = node.targets[0]
        if isinstance(target, ast.Name):
            consts[target.id] = value
        elif isinstance(target, ast.Tuple) and isinstance(value, tuple) \
                and len(target.elts) == len(value):
            for elt, item in zip(target.elts, value):
                if isinstance(elt, ast.Name):
                    consts[elt.id] = item
    return consts


# ---- 三方对账 ----

def _has(blob: str, token: str) -> bool:
    return token in normalize_signs(blob)


def reconcile_quantity(*, name: str, source_value: float, decimals: int,
                       ledger_text: str | None, pdf_text_layer: str | None,
                       script_constant: float | None) -> dict | None:
    """单个登记量的三方对账；一致返回 None，否则返回不一致清单。

    传 None 的腿视为"该侧不出现此量"⇒ 跳过；传了却没匹配到 ⇒ 该腿不一致。
    source_value 必须是全精度源算值（canonical 由它现舍，杜绝双舍）。"""
    canonical = f"{source_value:.{decimals}f}"
    failed: list[str] = []
    if ledger_text is not None and not _has(ledger_text, canonical):
        failed.append("ledger")
    if pdf_text_layer is not None and not _has(pdf_text_layer, canonical):
        failed.append("pdf")
    if script_constant is not None \
            and round(float(script_constant), decimals) != round(source_value, decimals):
        failed.append("script")
    if not failed:
        return None
    return {"name": name, "canonical": canonical, "failed_legs": failed}


def reconcile_phrase(*, name: str, phrase: str, ledger_text: str | None,
                     pdf_text_layer: str | None) -> dict | None:
    """分数式表述（如 ties=21/23）的存在性对账：源侧已逐题重算出分子分母，这里只核两侧写法。"""
    failed = [leg for leg, blob in (("ledger", ledger_text), ("pdf", pdf_text_layer))
              if blob is not None and not _has(blob, phrase)]
    return None if not failed else {"name": name, "canonical": phrase, "failed_legs": failed}


def build_checks(root: Path) -> list[dict | None]:
    """全部登记量 → 三方对账结果（None 表示一致）。"""
    audit_a = (root / "papers/paper-a/number_audit.md").read_text(encoding="utf-8")
    audit_b = (root / "papers/paper-b/number_audit.md").read_text(encoding="utf-8")
    calibers = (root / "docs/CALIBERS.md").read_text(encoding="utf-8")
    fig_b = root / "papers/paper-b/draft/figures"
    fig_a = root / "papers/paper-a/draft/figures"

    drift_mean, drift_sd, paired, _, _ = recompute_drift(root)
    fig3 = script_constants(root / "papers/paper-b/figures/scripts/fig3_drift.py")
    fig4a = script_constants(root / "papers/paper-a/figures/scripts/fig3_ablation.py")
    ablation = recompute_ablation(root)
    cf = recompute_cf_margins(root)
    b1 = recompute_b1(root)
    anchor = recompute_anchor(root)

    drift_pdf = pdf_text(fig_b / "fig3_drift.pdf")
    abl_pdf = pdf_text(fig_a / "fig3_ablation.pdf")
    b1_pdf = pdf_text(fig_a / "fig2_b1.pdf")
    var_pdf = pdf_text(fig_b / "fig2_variance.pdf")
    anc_pdf = None  # fig4_anchor 已退役（2026-10-06 换 fig4_ladder/fig5_stability）；anchor 数字仍留台账
    lad_pdf = None  # 62.5% 已移出图片：补充材料改用表格（2026-10-06），PDF 文本层无此数字
    sta_pdf = pdf_text(fig_b / "fig5_stability.pdf")

    checks: list[dict | None] = [
        reconcile_quantity(name="F3 drift mean", source_value=drift_mean, decimals=4,
                           ledger_text=audit_b, pdf_text_layer=drift_pdf,
                           script_constant=fig3.get("LEDGER_MEAN")),
        reconcile_quantity(name="F3 drift sd", source_value=drift_sd, decimals=4,
                           ledger_text=audit_b, pdf_text_layer=drift_pdf,
                           script_constant=fig3.get("LEDGER_SD")),
        # 配对题数只登记在 CALIBERS T15（number_audit 未记这一列）
        reconcile_phrase(name="F3 paired items", phrase=f"{paired}",
                         ledger_text=calibers, pdf_text_layer=None),
        reconcile_quantity(name="B F2 b1 mean margin", source_value=b1["mean_margin"], decimals=4,
                           ledger_text=audit_b, pdf_text_layer=b1_pdf, script_constant=None),
        reconcile_quantity(name="B F2 2σg threshold 4B", source_value=b1["threshold_4b"],
                           decimals=4, ledger_text=audit_b, pdf_text_layer=None,
                           script_constant=fig4a["THRESHOLDS"]["4B"]),
        reconcile_quantity(name="B F2 2σg threshold 9B", source_value=b1["threshold_9b"],
                           decimals=4, ledger_text=audit_b, pdf_text_layer=None,
                           script_constant=fig4a["THRESHOLDS"]["9B"]),
    ]
    for scale in ("0.8B", "2B", "9B"):
        registered = fig4a["EXPECTED_MEANS"][scale]
        for arm, idx in (("fixed", 0), ("eg", 1)):
            checks.append(reconcile_quantity(
                name=f"A F3 ablation {arm} mean {scale}（9B 为 gen2+gen3，非三代均值）",
                source_value=ablation[scale][arm], decimals=4, ledger_text=audit_a,
                pdf_text_layer=None, script_constant=registered[idx]))
        checks.append(reconcile_quantity(
            name=f"A F3 ablation EG-Fixed delta {scale}", source_value=ablation[scale]["delta"],
            decimals=4, ledger_text=audit_a, pdf_text_layer=abl_pdf, script_constant=None))
    for scale in ("4B", "9B"):
        checks.append(reconcile_quantity(
            name=f"A/B F3 cf margin {scale}", source_value=cf[scale], decimals=4,
            ledger_text=audit_a, pdf_text_layer=abl_pdf,
            script_constant=fig4a["EXPECTED_CF"].get(scale)))
    for label, value in (("ties", anchor["ties"]), ("fully ordered", anchor["fully_ordered"])):
        checks.append(reconcile_phrase(name=f"B F4 anchor {label}",
                                       phrase=f"{value}/{anchor['n']}",
                                       ledger_text=audit_b, pdf_text_layer=anc_pdf))
    tau_min = float(_json(root / "benchmark" / "eval" / "judge_qa"
                          / "rank_report.json")["R3"]["tau_min"])
    checks.extend([
        reconcile_quantity(name="B F5 stability tau_min", source_value=tau_min, decimals=3,
                           ledger_text=audit_b, pdf_text_layer=sta_pdf,
                           script_constant=None),
        reconcile_phrase(name="B F4 ladder monotone full rate", phrase="62.5%",
                         ledger_text=audit_b, pdf_text_layer=lad_pdf),
    ])
    # ---- 补充腿（figure_ledger_extra：fig1/fig4 增益、fig5 医疗、drift 逐格、ladder 逐格）----
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from scripts.figure_ledger_extra import build_extra_checks  # noqa: PLC0415
    checks.extend(build_extra_checks(root, pdf_text=pdf_text,
                                     reconcile_quantity=reconcile_quantity,
                                     script_constants=script_constants))
    return checks


def unregistered_figure_claims(root: Path) -> list[str]:
    """图内渲染、但台账未作为**该主张**登记的分数式统计量（自愈式：登记即从清单消失）。

    刻意**不计入退出码**：登记要改 papers/**（不随本包分发），本件无权改，
    也不能因此把 CI 常红。

    `faithful ahead` 为何不放进硬对账：它的值是 0/23，而台账里 `0/23` 是
    `full_order_rate` 那一行 —— token 相同、主张不同。放进硬对账会得到一个**因巧合而绿**
    的检查（假绿），比不查更坏，故只在此处如实报为待登记。"""
    audit_b = (root / "papers/paper-b/number_audit.md").read_text(encoding="utf-8")
    anchor = recompute_anchor(root)
    candidates = {
        "inverted": f"{anchor['inverted']}/{anchor['n']}",
        "evasive at 0": f"{anchor['evasive_at_zero']}/{anchor['n']}",
    }
    return sorted(f"{label} = {phrase}" for label, phrase in candidates.items()
                  if not _has(audit_b, phrase))


# token 存在性检查的**已知盲区**：同一分数可属不同主张，机器分不开，只能显式声明并人工
# 报给作者（登记需改 papers/**）。写成常量而不是注释，是为了让它出现在闸的输出里，
# 不至于日后被忘掉 —— 静默的盲区比没有检查更危险。
LEDGER_TOKEN_BLIND_SPOTS: tuple[tuple[str, str, str], ...] = (
    ("faithful ahead", "0/23",
     "S8 已把 faithful ahead 0/23 带主张名登记，但它与 full_order_rate 的 0/23 同分数；token 式存在性检查仍分不开 ⇒ 只能靠带名登记 + 本条声明兜住"),
)


def run_gate(root: Path) -> list[dict]:
    return [p for p in build_checks(root) if p]


def main() -> int:
    root = repo_root()
    drift_mean, drift_sd, paired, upward, n = recompute_drift(root)
    print("== 图侧台账三方对账（源 JSON ↔ number_audit ↔ 图脚本/PDF）==")
    print(f"  F3 源侧 records 级：mean={drift_mean:+.6f} sd={drift_sd:.6f} "
          f"配对题={paired} 同向={upward}/{n}")
    problems = run_gate(root)
    for gap in unregistered_figure_claims(root):
        print(f"  [待登记] 图内渲染但 number_audit 未记（改 papers/** 须作者裁定，"
              f"故不计入退出码）-> {gap}")
    for label, phrase, why in LEDGER_TOKEN_BLIND_SPOTS:
        print(f"  [盲区] token 检查分不开：{label} = {phrase} —— {why}")
    if problems:
        print(f"\n  不一致 {len(problems)} 条：")
        for p in problems:
            print(f"    [MISS] {p['name']} 源侧应为 {p['canonical']}；不一致腿={p['failed_legs']}")
        print("\n  [GATE FAIL] 图侧台账存在未对齐，见上表。")
        return 1
    print("  ✅ 全部登记量三方一致")
    return 0


if __name__ == "__main__":
    sys.exit(main())
