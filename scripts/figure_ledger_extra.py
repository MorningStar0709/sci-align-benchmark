#!/usr/bin/env python3
"""check_figure_ledger 的补充腿：四组图数据的独立复核（此前只有出图脚本内建 guard）。

覆盖（均从 benchmark/eval 产物现算，未取整；canonical 由全精度源值现舍）：
  A fig1/fig4  十八格增益（%）：复用 surface_stats.gains()——图/表/本闸同一算法源；
               REGISTERED 常量由 AST 读取（出图脚本绝不 import，防出图写盘）。
  A fig5       九条医疗 margin：四 run（run1/run1p/run2/run3）顶层 overall_score 均值差，
               同 fig5_medical_flip.py 口径；EXPECTED_MARGINS 由 AST 读取。
  B fig3       逐模型漂移 23 格（control 11 + medical 12）：run1 vs run1' 配对、剔假 0，
               同 fig3_drift.py 口径；逐格值只登记在图内，故 ledger 腿=None。
  B fig4       逐维×档中位数 20 格：judge_qa/mono_state.json（40 题×5 档）现算 median，
               对图内 2dp 显示逐格核；台账只登记部分格，故 ledger 腿=None。

导入边界：本模块 import surface_stats（纯统计模块：无 matplotlib、无模块级写盘）以保
增益算法单一源；fig*.py 出图脚本一律 AST，绝不 import。
腿语义与 canonical 规则同 check_figure_ledger.reconcile_quantity（由调用方注入）。
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

_EVAL = ("benchmark", "eval")

MEDICAL_RUNS = ("run1", "run1p", "run2", "run3")
MEDICAL_BASES = (
    ("4B", "raw_qwen_med", (("SFT", "medical_sft"), ("RLVR-C v2", "medical_rlvr_c_v2"),
                            ("MGPO-EG", "medical_mgpo_eg"))),
    ("9B", "raw_9b_med", (("SFT", "medical_sft_9b"), ("RLVR-C v2", "medical_rl_9b"),
                          ("MGPO-EG", "medical_mgpo_9b"))),
    ("Gemma-4", "raw_gemma4_med", (("SFT", "medical_sft_gemma4"), ("RLVR-C v2", "medical_rl_gemma4"),
                                   ("MGPO-EG", "medical_mgpo_gemma4"))),
)
# control 11 模型与 RUN1 批目录：同 fig3_drift.py / check_figure_ledger.CONTROL_MODELS
DRIFT_RUN1 = {"4b": "v3_phase4_300", "9b": "v3_9b_300q", "gemma": "gemma4_300q"}
DRIFT_CONTROL = (
    ("raw_qwen", "4b"), ("sft_baseline", "4b"), ("mgpo_eg", "4b"),
    ("raw_9b", "9b"), ("sft_9b", "9b"), ("mgpo_fixed_9b", "9b"), ("rlvr_c_v2_9b", "9b"),
    ("raw_gemma4", "gemma"), ("sft_gemma4", "gemma"),
    ("mgpo_eg_gemma4", "gemma"), ("rlvr_c_v2_gemma4", "gemma"),
)
DRIFT_MEDICAL = ("raw_qwen_med", "medical_sft", "medical_mgpo_eg", "medical_rlvr_c_v2",
                 "raw_9b_med", "medical_sft_9b", "medical_mgpo_9b", "medical_rl_9b",
                 "raw_gemma4_med", "medical_sft_gemma4", "medical_mgpo_gemma4", "medical_rl_gemma4")
LADDER_DIMS = ("A", "B", "C", "D")


def _json(path: Path):
    if not path.is_file():
        raise SystemExit(f"[GATE ERROR] missing product: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _overall(path: Path) -> float:
    return float(_json(path)["overall_score"])


def gains_18(root: Path) -> dict[str, dict[str, float]]:
    """十八格增益（全精度 %）。委托 surface_stats.gains()——图、表、闸共用一份算法。"""
    sys.path.insert(0, str(root / "papers" / "paper-a" / "figures" / "scripts"))
    import surface_stats  # noqa: PLC0415

    g = surface_stats.gains()
    return {c: {m: g[c][m] for m in ("SFT", "RLVR", "MGPO")} for c in g}


def medical_margins_9(root: Path) -> dict[str, float]:
    """九条医疗 margin：四 run 均值差（arm − base）。"""
    med = root.joinpath(*_EVAL) / "medical_dual_rejudge"
    out: dict[str, float] = {}
    for group, base_stem, arms in MEDICAL_BASES:
        base = statistics.fmean(_overall(med / f"{base_stem}_{r}.json") for r in MEDICAL_RUNS)
        for name, stem in arms:
            arm = statistics.fmean(_overall(med / f"{stem}_{r}.json") for r in MEDICAL_RUNS)
            out[f"{group} · {name}"] = arm - base
    return out


def _drift_delta(old: Path, new: Path) -> float:
    def _rec(p: Path) -> dict:
        return {r["id"]: r for r in _json(p).get("records", []) if r.get("id")}

    def _fake0(r: dict) -> bool:
        return (r.get("score") == 0 and
                any("failure" in str(x).lower() or "error" in str(x).lower()
                    for x in (r.get("issues") or [])))

    o, n = _rec(old), _rec(new)
    shared = sorted(set(o) & set(n))
    paired = [q for q in shared if not (_fake0(o[q]) or _fake0(n[q]))]
    return (statistics.fmean(n[q]["score"] for q in paired)
            - statistics.fmean(o[q]["score"] for q in paired))


def drift_cells(root: Path) -> dict[str, float]:
    """23 格逐模型漂移——control 11 走 run1_rejudge，medical 12 走 medical_dual_rejudge。"""
    ev = root.joinpath(*_EVAL)
    out: dict[str, float] = {}
    for key, batch in DRIFT_CONTROL:
        out[key] = _drift_delta(ev / DRIFT_RUN1[batch] / f"{key}_run1.json",
                                ev / "run1_rejudge" / f"{key}_run1p.json")
    for key in DRIFT_MEDICAL:
        out[key] = _drift_delta(ev / "medical_dual_rejudge" / f"{key}_run1.json",
                                ev / "medical_dual_rejudge" / f"{key}_run1p.json")
    return out


def ladder_cells(root: Path) -> dict[str, float]:
    """逐维×档中位数 20 格（mono_state.json：40 题 × 5 档）。"""
    rows = _json(root.joinpath(*_EVAL) / "judge_qa" / "mono_state.json")
    acc: dict[tuple[str, int], list[float]] = {}
    for r in rows:
        acc.setdefault((r["dim"], int(r["tier"])), []).append(float(r["score"]))
    return {f"{d}·t{t}": statistics.median(acc[(d, t)])
            for d in LADDER_DIMS for t in range(5)}


def build_extra_checks(root: Path, *, pdf_text, reconcile_quantity, script_constants) -> list:
    """四组补充腿的 reconcile 调用清单（None=一致；腿函数由主闸注入，防同式两份）。"""
    fig_a = root / "papers" / "paper-a" / "draft" / "figures"
    fig_b = root / "papers" / "paper-b" / "draft" / "figures"
    fig1_pdf = pdf_text(fig_a / "fig1_surface.pdf")
    fig4a_pdf = pdf_text(fig_a / "fig4_cross_family.pdf")
    fig5_pdf = pdf_text(fig_a / "fig5_medical_flip.pdf")
    drift_pdf = pdf_text(fig_b / "fig3_drift.pdf")
    lad_pdf = pdf_text(fig_b / "fig4_ladder.pdf")
    audit_a = (root / "papers/paper-a/number_audit.md").read_text(encoding="utf-8")

    checks: list = []
    # A fig1/fig4：18 格增益（Qwen 四柱核 fig1、Gemma 两柱核 fig4；另 12 格复验 fig4 同值）
    reg = script_constants(root / "papers/paper-a/figures/scripts/fig1_surface.py")["REGISTERED"]
    gains = gains_18(root)
    for col, cells in gains.items():
        for method, value in cells.items():
            checks.append(reconcile_quantity(
                name=f"A F1/F4 gain {col} {method}", source_value=value, decimals=1,
                ledger_text=audit_a,
                pdf_text_layer=fig4a_pdf if col in ("E2B", "E4B-it") else fig1_pdf,
                script_constant=reg[col][method]))
    for col in ("0.8B", "2B", "4B", "9B"):
        for method, value in gains[col].items():
            checks.append(reconcile_quantity(
                name=f"A F4 gain {col} {method}（fig4 同值复验）", source_value=value, decimals=1,
                ledger_text=None, pdf_text_layer=fig4a_pdf, script_constant=None))
    # A fig5：9 条医疗 margin（脚本常量腿 = EXPECTED_MARGINS，AST 字符串转 float）
    exp5 = script_constants(
        root / "papers/paper-a/figures/scripts/fig5_medical_flip.py")["EXPECTED_MARGINS"]
    for label, value in medical_margins_9(root).items():
        checks.append(reconcile_quantity(
            name=f"A F5 medical margin {label}", source_value=value, decimals=4,
            ledger_text=audit_a, pdf_text_layer=fig5_pdf,
            script_constant=float(exp5[label])))
    # B fig3：23 格逐模型漂移（逐格值只登记在图内 ⇒ ledger 腿=None）
    for key, value in drift_cells(root).items():
        checks.append(reconcile_quantity(
            name=f"B F3 drift cell {key}", source_value=value, decimals=3,
            ledger_text=None, pdf_text_layer=drift_pdf, script_constant=None))
    # B fig4：20 格逐维中位数（台账只登记部分格 ⇒ ledger 腿=None）
    for cell, value in ladder_cells(root).items():
        checks.append(reconcile_quantity(
            name=f"B F4 ladder cell {cell}", source_value=value, decimals=2,
            ledger_text=None, pdf_text_layer=lad_pdf, script_constant=None))
    return checks
