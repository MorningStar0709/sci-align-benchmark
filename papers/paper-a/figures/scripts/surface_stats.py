"""表 1 的统计线：冻结 2σg 带判定 + 题级 Wilcoxon/Holm（形态甲的口径源）。

为什么要独立成模块：显著性判据必须与表值同源同一次计算，写在 tab1 里就会
和绘图脚本各算一套，正是本仓防的口径分裂。

数据入口只有两个登记件——`n3_aggregation_summary.json`（均值与 gen 数）与
`adapter_manifest.json`（逐题分数所在的产物路径）。禁止在本文件里出现
评测目录名：那等于绕过登记直接猜实物。
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "scripts"))

from aggregate_n3_final import CONTROL_SURFACE, SURFACE_EXTRA, SUMMARY_PATH  # noqa: E402
from holm_wilcoxon import holm_adjust, wilcoxon_p  # noqa: E402

MANIFEST_PATH = ROOT / "release" / "adapter_manifest.json"
# 冻结判据带 2σg(4B)=0.0132 / 2σg(9B)=0.0332，登记于 DATA-TRACE §3.7 与 §4.2，
# 口径条目见 docs/CALIBERS.md 的 T9 与 T13（按内容定位，不写行号——行号会腐，V9）。
# 其余四柱无冻结带，不得事后造带。
FROZEN_2SIGMA = {"4B": 0.0132, "9B": 0.0332}
EXTRA_PATHS = {s["key"]: s for s in SURFACE_EXTRA}
COLUMN_ORDER = ("0.8B", "2B", "4B", "9B", "E2B", "E4B-it")


def _key_of(column: str, arm: str) -> str | None:
    """(柱, 臂) → 登记 key。双向查：老柱在 CONTROL_SURFACE，新柱在 SURFACE_EXTRA。"""
    for key, (col, a) in CONTROL_SURFACE.items():
        if (col, a) == (column, arm):
            return key
    for key, spec in EXTRA_PATHS.items():
        if (spec["column"], spec["arm"]) == (column, arm):
            return key
    return None


def _product_paths(key: str) -> list[Path]:
    """该臂的全部 gen 产物路径；老柱走 manifest n3_state，新柱走注册表定位。"""
    spec = EXTRA_PATHS.get(key)
    if spec:
        out = []
        for g in spec["gens"]:
            name = f"{spec['stem']}{g}.json" if g else f"{spec['stem']}.json"
            path = ROOT / spec["dir"] / name
            if path.is_file():
                out.append(path)
        return out
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    out = []
    for meta in manifest["n3_state"].get(key, {}).values():
        if isinstance(meta, dict) and meta.get("path"):
            path = ROOT / meta["path"]
            if path.is_file():
                out.append(path)
    return out


def item_means(key: str) -> dict[str, float]:
    """按题号取该臂各 gen 分数的均值（配对单元）。"""
    acc: dict[str, list[float]] = defaultdict(list)
    for path in _product_paths(key):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError, UnicodeDecodeError):
            print(f"[WARN] 产物不可读，跳题级配对: {path.relative_to(ROOT)}", file=sys.stderr)
            continue
        for rec in data.get("records") or []:
            if isinstance(rec, dict) and rec.get("id") and isinstance(rec.get("score"), (int, float)):
                acc[rec["id"]].append(rec["score"])
    return {k: sum(v) / len(v) for k, v in acc.items()}


def load_means() -> dict[str, dict]:
    """登记均值与 gen 数，按 (柱, 臂) 索引。"""
    data = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    out: dict[str, dict] = {}
    for row in data.get("rows", []):
        if row.get("domain") != "control" or not row.get("column"):
            continue
        out.setdefault(row["column"], {})[row["arm"]] = {
            "mean": row["mean"], "n_gens": row["n_gens"], "key": row["key"]}
    return out


def gains(columns: dict[str, dict] | None = None) -> dict[str, dict[str, float]]:
    """{柱: {SFT, RLVR, MGPO}} 的链式增益百分比——表与图共用这一个函数。

    分母口径按 §4.1：SFT 行对 raw，两条强化行对 SFT 基线。图与表各算一遍
    是这个项目出现过「图与表数字不一致」的物理来源，所以这里只留一份。
    """
    columns = columns if columns is not None else load_means()
    out = {}
    for column in COLUMN_ORDER:
        arms = columns.get(column, {})
        if not all(a in arms for a in ("raw", "sft", "rlvr", "mgpo_eg")):
            continue
        base, sft = arms["raw"]["mean"], arms["sft"]["mean"]
        out[column] = {
            "SFT": (arms["sft"]["mean"] / base - 1.0) * 100.0,
            "RLVR": (arms["rlvr"]["mean"] / sft - 1.0) * 100.0,
            "MGPO": (arms["mgpo_eg"]["mean"] / sft - 1.0) * 100.0,
            "gens": "/".join(str(arms[a]["n_gens"]) for a in ("raw", "sft", "rlvr", "mgpo_eg")),
        }
    return out


def contrasts(columns: dict[str, dict] | None = None) -> list[dict]:
    """每个三对比：margin（登记均值差）+ 冻结带判定 + 题级 Wilcoxon p（Holm 前）。"""
    pairs = (("sft", "raw"), ("rlvr", "sft"), ("mgpo_eg", "sft"))
    out = []
    for column in COLUMN_ORDER:
        arms = columns.get(column, {})
        for low, high in pairs:
            if low not in arms or high not in arms:
                continue
            margin = round(arms[low]["mean"] - arms[high]["mean"], 4)
            band = FROZEN_2SIGMA.get(column)
            lo_items, hi_items = item_means(arms[low]["key"]), item_means(arms[high]["key"])
            shared = sorted(set(lo_items) & set(hi_items))
            p = wilcoxon_p([lo_items[i] for i in shared], [hi_items[i] for i in shared]) \
                if len(shared) > 10 else None
            out.append({
                "column": column, "low": low, "high": high, "margin": margin,
                "band": band, "min_reachable_p": 2 ** (-min(arms[low]["n_gens"], arms[high]["n_gens"])),
                "n_items": len(shared), "p": p,
            })
    for c in out:
        c["holm_pass"] = None
        c["outside_band"] = None if c["band"] is None else abs(c["margin"]) > c["band"]
    with_p = [i for i, c in enumerate(out) if c["p"] is not None]
    for i, ok in zip(with_p, holm_adjust([out[i]["p"] for i in with_p], alpha=0.05)):
        out[i]["holm_pass"] = ok
    return out


if __name__ == "__main__":
    cols = load_means()
    for c in contrasts(cols):
        band = "未标定" if c["band"] is None else f"{c['band']:.4f}"
        verdict = "n/a" if c["outside_band"] is None else ("带外" if c["outside_band"] else "带内")
        p = "n/a" if c["p"] is None else f"{c['p']:.3e}"
        holm = "n/a" if c["holm_pass"] is None else ("通过" if c["holm_pass"] else "不通过")
        print(f"{c['column']:7s} {c['low']:8s} vs {c['high']:8s} margin={c['margin']:+.4f} "
              f"带={band} {verdict:4s} 最小可达p={c['min_reachable_p']:.4f} "
              f"题级p={p} Holm={holm} 配对题数={c['n_items']}")
