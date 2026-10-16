"""逐代同向读数件：从登记聚合件读「每一代 high−low」的符号一致性。

为什么要有这一件：转正判据是**合取式**——「n 代全部同向 **且** mean margin > 2σg」
（docs/CALIBERS.md「转正判据」行）。但盘上机器记录同向位的只有两件：
`b1_verdict.json`（direction_count / pair_margins）与 `counterfactual/analysis.json`
（all_same_direction）。弹性面、9B 支点、EG−Fixed 消融这三条的在册件只存**聚合均值**，
逐代差分此前活在一次性扫描件 `temp/_e5_sigma_sensitivity.py` + `temp/e5_sigma_sensitivity_result.json`
里，而 AGENTS 明令 `temp/` 文件不得进任何调用链。09-30 接矩阵判据列时这四条就卡在
「同向无机器件」上（见 docs/reports/matrix-wiring-cf-and-blockers-2026-09-30.md §二·冲突 1），
按批准把这一层落成常驻登记算子。

口径不是自造的：排除 `gen1(run1p)` 一类的**复用答案重判件**，与
`tests/test_sigma_definition.py::_gens(drop_reused=True)` 和 CALIBERS T16
「排除 run1p 复用答案重判件、取全部独立生成代」是同一条规矩。
方向与阈值的分工也照在册：`number_audit`（B 篇）写的是「阈值比较用登记 margin，
逐代差分**仅作方向证据**」——所以本件只回答同向与否，不产出比较用的 margin。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from aggregate_n3_final import AGG_GENS, SUMMARY_PATH, SURFACE_EXTRA  # noqa: E402
from check_figure_ledger import ABLATION_GENS  # noqa: E402
from run_n3_regen import MODELS, _detect_run1, _gen_path  # noqa: E402

# 复用答案的重判代次标记：聚合件把这类代记成 "gen1(run1p)"。
REUSED_GENERATION_MARK = "run1"


def load_rows(summary_path: Path = SUMMARY_PATH) -> dict[str, dict]:
    """登记聚合件 rows → {key: row}。只读，不重算任何均值。"""
    data = json.loads(Path(summary_path).read_text(encoding="utf-8"))
    return {r["key"]: r for r in data.get("rows", []) if r.get("key")}


def independent_gen_scores(row: dict) -> dict[str, float]:
    """该臂的独立生成代总分（剔掉复用答案的重判代）。"""
    return {gen: float(score) for gen, score in (row.get("gen_scores") or {}).items()
            if REUSED_GENERATION_MARK not in gen}


def gen_direction(low_key: str, high_key: str,
                  rows: dict[str, dict] | None = None) -> dict:
    """逐代差分与同向判定：margin(gen) = high(gen) − low(gen)，只在两臂都有的代上算。

    读的是登记聚合件 `n3_aggregation_summary.json` 的 `rows[].gen_scores`。
    返回 {"margins": {...}, "same_direction": bool, "n_gens": int}。
    `n_gens == 0` 时 same_direction 为 False——没有代就没有「全部同向」可言，
    不拿空集冒充成立。
    """
    rows = rows if rows is not None else load_rows()
    for key in (low_key, high_key):
        if key not in rows:
            raise KeyError(f"登记聚合件里没有这一臂: {key}")
    low, high = independent_gen_scores(rows[low_key]), independent_gen_scores(rows[high_key])
    shared = sorted(set(low) & set(high))
    margins = {gen: high[gen] - low[gen] for gen in shared}
    return _verdict(margins)


def gen_direction_from_products(root: Path, directory: str, low_stem: str, high_stem: str,
                                gens: list[int]) -> dict:
    """同一判据的产物侧读法，供聚合件里没有登记的臂使用（EG−Fixed 的 0.8B / 2B）。

    代集一律由调用方从 `check_figure_ledger.ABLATION_GENS` 传入，不在这里另列一套；
    `complete is not True` 的代一律剔除，与 `aggregate_n3_final` / `tests/test_sigma_definition.py`
    同一条门禁。
    """
    margins: dict[str, float] = {}
    for gen in gens:
        low = _overall_if_complete(root / directory / f"{low_stem}_gen{gen}.json")
        high = _overall_if_complete(root / directory / f"{high_stem}_gen{gen}.json")
        if low is None or high is None:
            continue
        margins[f"gen{gen}"] = high - low
    return _verdict(margins)


def _overall_if_complete(path: Path) -> float | None:
    """读产物 overall_score；不完整或不存在的代返回 None（由调用方剔掉，不猜）。"""
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("complete") is not True:
        return None
    score = data.get("overall_score")
    return float(score) if isinstance(score, (int, float)) else None


def _verdict(margins: dict[str, float]) -> dict:
    """由逐代差分得出同向判定。0 差（两臂该代同分）算不过同向，不装成「同号」。"""
    signs = {1 if v > 0 else -1 if v < 0 else 0 for v in margins.values()}
    return {
        "margins": margins,
        "same_direction": bool(margins) and len(signs) == 1 and 0 not in signs,
        "n_gens": len(margins),
    }



def ablation_pairs() -> list[tuple[str, str, str, str, list[int]]]:
    """EG−Fixed 三档的 (scale, directory, fixed_stem, eg_stem, gens)。

    代集一律来自 `check_figure_ledger.ABLATION_GENS`（9B 只有 gen2+gen3 是登记事实，
    不是这里挑的），避免「同向用一套代、delta 用另一套代」的口径分裂。
    """
    suffix = {"0.8B": "_08b", "2B": "_2b", "9B": "_9b"}
    return [(scale, f"benchmark/eval/{sub}", f"mgpo_fixed{suffix[scale]}",
             f"mgpo_eg{suffix[scale]}", gens) for scale, sub, gens in ABLATION_GENS]


# ---- 判据前件之二：epoch 洁净度 ----
#
# `Evidence.epoch_clean` 此前在五条判据型装载函数里都是常量 `True`（09-30 独立复查
# B 批 #7 抓到）。K3 把它的语义收窄成「**本行 margin 所在登记代集合内**各臂同判官代次」，
# 与题级 p 的池（`release/adapter_manifest.json` 全 roster，含 2026-07 旧判官件，
# 见 CALIBERS T22/T23）脱钩。本层就是那条收窄语义的实读件。

def relativize_product(path: str, root: Path | None = None) -> str:
    """实物路径转仓库相对。

    登记件里存的既有 Linux 绝对路径也有 Windows 绝对路径（`D:\\...`，Linux 化尾巴，
    CALIBERS 另有登记），形态不一律就地把两种都折成仓库相对。认不出 `SciMonitor/`
    分隔符的一律**拒绝**，不猜——猜错就会把别的臂的头部当成 epoch 证据。

    第三种形态（本仓根之下的绝对路径）是后补的：`_gen_path`／`_detect_run1` 会按**当前仓根**
    拼出绝对路径，于是「路径里有 SciMonitor/」只在仓目录正好叫 SciMonitor 时成立——那是个巧合，
    不是判据。仓目录一换名（发布仓即如此），上面两条都认不出，裸 ValueError 炸穿调用链。
    携带 root 时这种形态无歧义，故收进来；**不认识的一律仍然拒绝**。
    """
    norm = path.replace("\\", "/")
    # 此处字面量是**历史绝对路径匹配器**：冻结登记件里存的就是
    # `D:/WorkPlace/AI/SciMonitor/…` 形态，改名会让归一化失效——不可「清理」。
    if "SciMonitor/" in norm:
        return norm.split("SciMonitor/", 1)[1]
    if norm.startswith("benchmark/"):
        return norm
    base = (root or Path(__file__).resolve().parents[1]).as_posix().rstrip("/")
    if base and norm.startswith(base + "/"):
        return norm[len(base) + 1:]
    raise ValueError(f"实物路径形态不可识别，拒绝当作 epoch 证据: {path}")


def epoch_report(root: Path, products: list[str]) -> dict:
    """读一组实物的判官模型与批次月份，回答「同判官同批次否」。

    批次取 `updated_at` 的年-月（在册判官代次切换按整月记，8/6 之后为新 epoch）。
    读不到字段**一律算不洁净并点名**——不许把"读不到"过拟合成"没问题"。
    """
    judges: set[str] = set()
    months: set[str] = set()
    unreadable: list[str] = []
    for rel in products:
        path = root / rel
        if not path.is_file():
            unreadable.append(f"{rel}（缺件）")
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError, UnicodeDecodeError):
            unreadable.append(f"{rel}（不可读）")
            continue
        judge = data.get("judge_model")
        stamp = (data.get("updated_at") or "")[:7]
        if not judge or not stamp:
            unreadable.append(f"{rel}（缺 judge_model/updated_at）")
            continue
        judges.add(judge)
        months.add(stamp)
    clean = bool(products) and not unreadable and len(judges) == 1 and len(months) == 1
    return {"products": list(products), "judges": sorted(judges),
            "months": sorted(months), "unreadable": unreadable, "clean": clean}


def registered_n3_products(root: Path, key: str) -> list[str]:
    """该臂「登记均值实际用的那几代」实物路径。

    刻意复用聚合件的同一对 locator（`run_n3_regen._detect_run1` 优先取 run1'、
    `_gen_path` 取 gen2/gen3，`SURFACE_EXTRA` 给新柱的 dir+stem+gens），
    不另造命名规则——否则会变成「同向用一套代、epoch 用另一套代」，
    正是本仓 09-30 反复登记的那种口径分裂。
    """
    for spec in SURFACE_EXTRA:
        if spec["key"] == key:
            return [f"{spec['dir']}/{spec['stem']}{g}.json" if g
                    else f"{spec['dir']}/{spec['stem']}.json" for g in spec["gens"]]
    model = next((m for m in MODELS if m["key"] == key), None)
    if model is None:
        raise KeyError(f"{key} 不在 n3 注册表里，epoch 无处可读（不许默认洁净）")
    out = []
    run1 = _detect_run1(model)
    if run1:
        out.append(relativize_product(run1["path"], root))
    out.extend(relativize_product(str(_gen_path(key, g)), root) for g in AGG_GENS)
    return out


def arm_epoch(root: Path, *keys: str) -> dict:
    """若干臂的登记代集合合起来看 epoch：跨臂同判官同批次才算 clean。"""
    products: list[str] = []
    for key in keys:
        products.extend(registered_n3_products(root, key))
    return epoch_report(root, products)


if __name__ == "__main__":
    pairs = (("rlvr_c_v2_9b", "mgpo_eg_9b", "9B 支点 mgpo_eg vs rlvr"),
             ("sft_9b", "mgpo_eg_9b", "9B MGPO-EG vs SFT"),
             ("sft_baseline", "mgpo_eg", "4B MGPO-EG vs SFT"))
    rows = load_rows()
    for low, high, label in pairs:
        d = gen_direction(low, high, rows)
        series = " / ".join(f"{g} {v:+.4f}" for g, v in sorted(d["margins"].items()))
        print(f"{label:30s} n={d['n_gens']} 同向={d['same_direction']}  {series}")

    root = Path(__file__).resolve().parents[1]
    for scale, directory, fixed_stem, eg_stem, gens in ablation_pairs():
        d = gen_direction_from_products(root, directory, fixed_stem, eg_stem, gens)
        series = " / ".join(f"{g} {v:+.4f}" for g, v in sorted(d["margins"].items()))
        print(f"EG−Fixed {scale:6s}（代集 {gens}）  n={d['n_gens']} 同向={d['same_direction']}  {series}")
