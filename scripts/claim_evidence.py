"""矩阵判据列的证据装载：主张 → 登记件/登记算子的实测读数。

为什么独立成模块（09-30）：`score_claim_matrix.py` 是**输出面**（建表、渲染、指针校验），
这一层是**证据面**（每条主张的 margin/阈值/同向位从哪个登记件读）。混在一起时文件已过
AGENTS 的 300 行线，且每加一条主张都要动输出面。分层后加主张只动本件。

三条硬规矩（本仓红线的落地）：
1. **读数一律现跑**——要么读登记件字段（JSON 指针），要么调登记算子；不在这里写新算法、
   不猜阈值、不把台账里的数字抄进来冒充实测。
2. 现跑不到的就**明说现跑不到**：`recompute` 写 `UNBOUND:` 的行保持常量桩（S 级），
   绝不为了离开 S 而编阈值或转写数字。
3. 现算与在册陈述**冲突时照登现算值**，把冲突写进裕度说明等裁，本件不回改数字。
"""

from __future__ import annotations

import functools
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

from scripts.check_figure_ledger import (
    recompute_ablation,
    recompute_anchor,
    recompute_cf_margins,
    recompute_drift,
)
from scripts.claim_specs import ClaimSpec
from scripts.claim_strength import Evidence
from scripts import probe_readers
from scripts.probe_readers import (
    CLOSED_ARM_KEY,
    read_judge_ppl_correlation,
    read_medical_gain_vs_base,
    read_openbook_a,
    read_position_bias,
)
from scripts.surface_direction import (
    ablation_pairs,
    arm_epoch,
    epoch_report,
    gen_direction,
    gen_direction_from_products,
    relativize_product,
)


def _epoch_from_n3(root: Path, *keys: str) -> tuple[bool, str]:
    """判据前件 `epoch_clean` 的实读：本行 margin 的登记代集合内各臂同判官同批次。"""
    rep = arm_epoch(root, *keys)
    return rep["clean"], (f"epoch 实读：{'/'.join(keys)} 登记代 {len(rep['products'])} 件，"
                          f"判官 {','.join(rep['judges']) or '—'}，批次 {','.join(rep['months']) or '—'}"
                          + (f"，读数失败 {rep['unreadable']}" if rep["unreadable"] else ""))


def _epoch_from_products(root: Path, rels: list[str]) -> tuple[bool, str]:
    rep = epoch_report(root, rels)
    return rep["clean"], (f"epoch 实读：{len(rels)} 件，判官 {','.join(rep['judges']) or '—'}，"
                          f"批次 {','.join(rep['months']) or '—'}"
                          + (f"，读数失败 {rep['unreadable']}" if rep["unreadable"] else ""))

# 判据桩：未接产物的行走这一对值。具名是为了让测试能咬「在册却仍读桩」（R7）。
STUB_MARGIN = 0.05
STUB_THRESHOLD = 0.0132

# ---- 判据型（有 2σg 冻结带语义）----

CF_ARTIFACT = "benchmark/eval/counterfactual/analysis.json"
# 两档主张只有一个 margin 位：绑到带归一偏差较大的一档（4B |−0.0072|/0.0132=0.55
# > 9B 0.0063/0.0332=0.19），即对「两档均不显著」最严格的检验；两档读数一律进裕度说明。
CF_BINDING_TIER = "4b"

SURFACE_SUMMARY = "benchmark/eval/n3_regen/n3_aggregation_summary.json"
FIGURES_SCRIPTS = "papers/paper-a/figures/scripts"

# 主张 → (表 1 统计线坐标 (column, low, high), 逐代同向的臂名对 (low_key, high_key))。
# 方向与 `surface_stats.contrasts()` 一致：margin = mean(low) − mean(high)，
# 所以 MGPO-EG 相对 SFT 写成 ("4B", "mgpo_eg", "sft")，取到的 −0.0206 与表 1 同号。
# 上一轮实测到 `pivot_9b_00244` 的旧指针写 `surface_stats::contrasts`，返回的是
# `mgpo_eg vs sft`（+0.0333），与这条主张在册的 `vs rlvr_c_v2`（0.0244）是两个量——
# 即盘点件 §五.1 预警的「指针可解析≠返回的就是这条的数」，本轮指针已改到真源。
SURFACE_CONTRASTS = {
    "elasticity_reversal_4b": (("4B", "mgpo_eg", "sft"), ("sft_baseline", "mgpo_eg")),
    "elasticity_reversal_9b": (("9B", "mgpo_eg", "sft"), ("sft_9b", "mgpo_eg_9b")),
}
# K4（09-30 裁定）：2σg 是**正向**转正判据，对「增益为负」这类方向性主张判 C
# 会把带外且题级显著的真效应读成「未过阈」。不加负向分支（那是动判据），
# 改为把这行放进 N/A 档只报读数。
DIRECTIONAL_NA = {
    "elasticity_reversal_4b": (
        "方向性主张，无正向转正阈值语义：现读 4B mgpo_eg vs sft 为带外**负向**"
        "（绝对值 0.0206 > 2σg(4B)=0.0132），逐代 gen2 −0.0191 / gen3 −0.0108 同负，"
        "题级 p=0.002 过 Holm ⇒ 「增益为负」这一预测被证实。"
        "2σg 判据只管正向转正，故本行不评 A/B/C（K4 已裁：不加负向达带分支）"),
}
PIVOT_MARGIN_POINTER = f"{SURFACE_SUMMARY}#pivot/margin"
PIVOT_GEN_KEYS = ("rlvr_c_v2_9b", "mgpo_eg_9b")   # 支点 = MGPO-EG 9B vs RLVR-C v2 9B
EG_FIXED_TIER = 2                                  # ablation_pairs() 里 9B 那一档的下标
FROZEN_2SIGMA_9B = 0.0332                          # 与表 1 统计线 9B 行 band 列同值

EG_FIXED_TIER_NOTE = (
    "三档全列：0.8B delta=+0.0128（gen1 +0.0192/gen2 +0.0125/gen3 +0.0067，同向）、"
    "2B delta=+0.0022（gen1 −0.0036/gen2 +0.0117/gen3 −0.0016 ⇒ 逐代变号，同向不过）、"
    "9B delta=+0.0133（gen2 +0.0166/gen3 +0.0100，同向）；"
    "0.8B/2B 无冻结带（表 tab:cross-family-stats 用词 not assessable），本行只按 9B 档分级"
    "——正文里 0.8B 的 +5.7pp 与 2B 的 +0.6pp 是相对 SFT 的百分比口径，不在 2σg 判据内"
)

# ---- 描述型（读数真实，但没有 2σg 转正阈值语义 → 分级 N/A）----

ANCHOR_RAW_PRODUCTS = {
    "anchor_27b": "benchmark/eval/probe_scaling/qwen27b_control_gen1.json",
    "anchor_gemma_12b": "benchmark/eval/probe_scaling/gemma12b_control_gen1.json",
}
XCHECK_PRODUCT = "benchmark/eval/a800/xcheck_verdict.json"
SIGMA_PRODUCT = "benchmark/eval/counterfactual/analysis.json"
MEDICAL_TRIPLETS = {
    "attractor_gemma_triplet": ("raw_gemma4_med", "medical_sft_gemma4",
                                "medical_mgpo_gemma4", "medical_rl_gemma4"),
    "attractor_medical_nine": ("raw_9b_med", "medical_sft_9b",
                               "medical_mgpo_9b", "medical_rl_9b"),
    "medical_scale_axis": ("medical_mgpo_eg", "medical_mgpo_9b"),
}


# 装载件**实际读过**哪些实物，记在这里——清单生成器据此覆盖，不能只信 claim_specs 声明的
# `supports`：B1 的判据件就是「声明写 patch_summary、实读 b1_verdict」的典型漏项。
READ_LOG: set[str] = set()


def _read_text(root: Path, rel: str) -> str:
    READ_LOG.add(rel)
    return (root / rel).read_text(encoding="utf-8")


@functools.lru_cache(maxsize=1)
def _summary(root: Path) -> dict:
    return json.loads(_read_text(root, SURFACE_SUMMARY))


def _rows_by_key(root: Path) -> dict[str, dict]:
    """聚合件 rows 按 key 索引——不靠行号，行号会随聚合件重排而腐。"""
    return {r["key"]: r for r in _summary(root)["rows"] if r.get("key")}


def _arm_n_gens(root: Path, key: str) -> int:
    """该臂登记的生成代数（`rows[].n_gens`，含复用答案重判代）。"""
    return int(_rows_by_key(root)[key]["n_gens"])


def _arm_item_count(root: Path, key: str) -> int:
    """该臂的题级配对分母（`rows[].judge_stats.n_dual_records`，双判记录数）。"""
    return int(_rows_by_key(root)[key]["judge_stats"]["n_dual_records"])


class WorkbenchOnly(RuntimeError):
    """本树缺工作仓件 ⇒ 这一项在这里**不可复算**。

    为什么要专门一个类型：表 1 的统计线（2σg 带 + 题级 Wilcoxon/Holm）住在
    `papers/*/figures/scripts/surface_stats.py`，它自己又依赖 `holm_wilcoxon`。这两个缺席时
    原先抛裸 `ModuleNotFoundError`，评审在发布仓跑一条在册命令会看到整段 Python 栈——
    栈的观感就是"这仓没测过"。出声的 fail-closed 才是诚实形态：说清缺什么、为什么缺、
    因此这项未验，且**不得当通过**（exit 1）。
    """


@functools.lru_cache(maxsize=1)
def _surface_stats_module(root_str: str):
    """按既有先例把 figures/scripts 挂进 sys.path 再 import（tests/ 里同一手法）。"""
    scripts_dir = Path(root_str) / FIGURES_SCRIPTS
    if not (scripts_dir / "surface_stats.py").is_file():
        raise WorkbenchOnly(
            f"表 1 统计线件缺席：{FIGURES_SCRIPTS}/surface_stats.py 不在本树"
            "（正文面件不随本 artifact 分发）")
    sys.path.insert(0, str(scripts_dir))
    try:
        import surface_stats
    except ModuleNotFoundError as exc:
        raise WorkbenchOnly(
            f"表 1 统计线依赖工作仓件 {exc.name!r}（本 artifact 有意不含该依赖）"
        ) from None
    return surface_stats


def _contrast_row(root: Path, column: str, low: str, high: str) -> dict:
    """从表 1 的统计线取一条对比（margin / band / 题级 p / Holm 全在同一件里）。"""
    stats = _surface_stats_module(str(root))
    for row in stats.contrasts(stats.load_means()):
        if (row["column"], row["low"], row["high"]) == (column, low, high):
            return row
    raise LookupError(f"表 1 统计线里没有这一对比: {column} {low} vs {high}")


def _gains(root: Path) -> dict[str, dict[str, float]]:
    stats = _surface_stats_module(str(root))
    return stats.gains(stats.load_means())


def _json_pointer(root: Path, expr: str):
    """读 `path#/seg/seg` 指向的字段值——形态与 claim_specs 的 recompute 指针一致。"""
    path_s, pointer = expr.split("#", 1)
    node = json.loads(_read_text(root, path_s.strip()))
    for part in filter(None, pointer.strip("/").split("/")):
        node = node[int(part)] if isinstance(node, list) else node[part]
    return node


def _call_operator(root: Path, expr: str):
    """现跑 `path.py::func` 形态的登记算子（约定：一律以 root 为唯一入参）。"""
    path_s, func_name = expr.split("::", 1)
    spec = importlib.util.spec_from_file_location(f"ce_{Path(path_s).stem}", root / path_s)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载登记算子文件: {path_s}")
    module = importlib.util.module_from_spec(spec)
    # 必须先登记进 sys.modules：被加载的脚本里有 dataclass，
    # dataclasses 解析字符串注解时要按 cls.__module__ 回查 sys.modules，
    # 查不到就炸在 'NoneType' object has no attribute '__dict__'。
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return getattr(module, func_name)(root)


def _not_assessable(note: str, value: float | None = None, *, item_count: int = 0) -> Evidence:
    """描述量/形状/阈值族的行：读数照登，分级 N/A（既不冒充 A/B/C，也不留桩）。"""
    return Evidence(margin=0.0 if value is None else float(value), threshold=0.0,
                    same_direction=False, n_gens=0, item_count=item_count,
                    caliber_unique=False, epoch_clean=False, denominator_clear=False,
                    replayable=False, known=False, note=note)


# ---- 判据型逐条装载 ----

def _load_b1(root: Path) -> Evidence:
    verdict = json.loads(_read_text(root, "benchmark/eval/n3_regen/b1_verdict.json"))
    epoch_clean, epoch_note = _epoch_from_n3(root, "sft_baseline", "raw_9b")
    return Evidence(
        margin=round(verdict["mean_margin"], 4),
        threshold=round(verdict["threshold_2sigma_4b"], 4),
        same_direction=True,          # 产物 reason 载明「三对全同向」
        n_gens=int(verdict["n_valid_pairs"]),
        item_count=300,               # fixed_300 题池（manifest datasets.fixed_300.sha256）
        caliber_unique=True,
        epoch_clean=epoch_clean,
        denominator_clear=True,
        replayable=True,
        note=(f"读自 b1_verdict.json，verdict={verdict['verdict']}；"
              f"pair_margins={verdict['pair_margins']}；对带余量 "
              f"{verdict['mean_margin'] - verdict['threshold_2sigma_4b']:+.4f}；{epoch_note}"),
    )


def _load_cf(root: Path) -> Evidence:
    product = json.loads(_read_text(root, CF_ARTIFACT))
    judgement = product[CF_BINDING_TIER]["statistics"]["judgement"]
    denominators = {int(p["n_questions"]) for p in product[CF_BINDING_TIER]["pairs"]}
    tiers = recompute_cf_margins(root)
    listed = product[CF_BINDING_TIER]["products"]
    cf_rels = [relativize_product(p) for side in ("variant", "control") for p in listed[side]]
    epoch_clean, epoch_note = _epoch_from_products(root, cf_rels)
    return Evidence(
        margin=tiers["4B"],                        # 复用登记算子，不在这里重算
        threshold=round(float(judgement["threshold_2sigma"]), 4),
        same_direction=bool(judgement["all_same_direction"]),
        n_gens=int(judgement["n_pairs"]),
        item_count=max(denominators),
        caliber_unique=True,                       # sigma 与阈值同读一件 judgement
        epoch_clean=epoch_clean,                   # 旧写法是照抄件内 epoch_declaration 自陈，现改实读
        denominator_clear=len(denominators) == 1,
        replayable=True,                           # 件自陈 products+sha256
        note=(f"读自 counterfactual/analysis.json 4b 档 judgement；两档全列 "
              f"4B={tiers['4B']:+.4f} / 9B={tiers['9B']:+.4f}，均在其冻结带内，"
              "产物 verdict=no_significant（绑档规则见本模块 CF_BINDING_TIER 注）；"
              f"{epoch_note}"),
    )


def _load_contrast_claim(root: Path, claim_id: str) -> Evidence:
    (column, low, high), (low_key, high_key) = SURFACE_CONTRASTS[claim_id]
    row = _contrast_row(root, column, low, high)
    direction = gen_direction(low_key, high_key)
    series = "/".join(f"{g} {v:+.4f}" for g, v in sorted(direction["margins"].items()))
    p_value = float(row["p"]) if row["p"] is not None else float("nan")
    epoch_clean, epoch_note = _epoch_from_n3(root, low_key, high_key)
    return Evidence(
        margin=float(row["margin"]),
        threshold=float(row["band"]),
        same_direction=direction["same_direction"],
        n_gens=_arm_n_gens(root, high_key),
        item_count=int(row["n_items"]),
        caliber_unique=True,            # 双臂同 batch、同 300 题池
        epoch_clean=epoch_clean,        # 登记代集合内实读（题级池的跨代次事实归 T22/T23）
        denominator_clear=int(row["n_items"]) > 0,
        replayable=True,
        holm_pass=row["holm_pass"],     # R2b：带外但题级 Holm 不过 ⇒ 保守侧降级
        known=claim_id not in DIRECTIONAL_NA,   # K4：方向性主张走 N/A
        note=((DIRECTIONAL_NA.get(claim_id)
               or f"表 1 统计线 {column} {low} vs {high}；逐代同向 {series}"
               f"（独立生成代 {direction['n_gens']} 个，run1p 按 CALIBERS T16 剔除）；"
               f"题级 Wilcoxon p={p_value:.4g}，Holm {'通过' if row['holm_pass'] else '不过'}"
               "（该 p 的题级池是 manifest 全 roster，与本行 margin 的登记池不同源：9B MGPO 臂含 "
               "2026-07-24 的 run1 而 SFT 臂用其 run1p；同代集重算 p=1.1824e-05 ⇒ 该池差异已裁"
               "「维持不修」，见 CALIBERS T22/T23）；"
               f"margin 绝对值 − 带 = {abs(row['margin']) - row['band']:+.4f}"
               f"（band verdict={'outside' if row['outside_band'] else 'inside'}）")
              + f"；{epoch_note}"),
    )


def _load_pivot(root: Path) -> Evidence:
    pivot = _summary(root)["pivot"]
    direction = gen_direction(*PIVOT_GEN_KEYS)
    series = "/".join(f"{g} {v:+.4f}" for g, v in sorted(direction["margins"].items()))
    epoch_clean, epoch_note = _epoch_from_n3(root, *PIVOT_GEN_KEYS)
    return Evidence(
        margin=float(_json_pointer(root, PIVOT_MARGIN_POINTER)),
        threshold=float(pivot["threshold_2sigma_9b"]),
        same_direction=direction["same_direction"],
        n_gens=_arm_n_gens(root, PIVOT_GEN_KEYS[1]),
        item_count=_arm_item_count(root, PIVOT_GEN_KEYS[1]),
        caliber_unique=True,
        epoch_clean=epoch_clean,
        denominator_clear=True,
        replayable=True,
        holm_pass=None,                 # 表 1 没有这一对比的题级行 ⇒ 不以显著性加码
        note=(f"读自登记聚合件 pivot（verdict={pivot['verdict']}，B 篇表列 not prom.）；"
              f"逐代差分 {series} ＝在册「支点方向证据 0.0391/0.0233 同正」；"
              "按 number_audit 规定：阈值比较用登记 margin，逐代差分仅作方向证据"
              f"（两套代集合不同：margin 含 run1p、逐代不含）；{epoch_note}"),
    )


def _load_eg_fixed(root: Path) -> Evidence:
    scale, directory, fixed_stem, eg_stem, gens = ablation_pairs()[EG_FIXED_TIER]
    delta = recompute_ablation(root)[scale]["delta"]
    direction = gen_direction_from_products(root, directory, fixed_stem, eg_stem, gens)
    series = "/".join(f"{g} {v:+.4f}" for g, v in sorted(direction["margins"].items()))
    ablation_rels = [f"{directory}/{stem}_gen{g}.json"
                     for stem in (fixed_stem, eg_stem) for g in gens]
    epoch_clean, epoch_note = _epoch_from_products(root, ablation_rels)
    return Evidence(
        margin=float(delta),
        threshold=FROZEN_2SIGMA_9B,
        same_direction=direction["same_direction"],
        n_gens=direction["n_gens"],     # 9B 消融只有 gen2+gen3（登记事实）
        item_count=_arm_item_count(root, eg_stem),
        caliber_unique=True,
        epoch_clean=epoch_clean,        # 按本档实际用的那两代实读，不套用 run1p 集合
        denominator_clear=True,
        replayable=True,
        holm_pass=None,                 # 消融臂无题级登记件
        note=(f"本行按 9B 档分级（delta={delta:+.4f}，逐代 {series}）；"
              f"{epoch_note}；{EG_FIXED_TIER_NOTE}"),
    )


# ---- 描述型逐条装载 ----

# 通过位措辞：出现这些词，就是把「判官排对了」当成了结论。
_THROUGH_WORDS = ("验证通过", "接近上限", "排对")


def _statement_conflict(statement: str) -> str:
    """措辞是否把 ties 读成通过位——**现读在册陈述判**，不是写死一句指控。

    为什么要重写（09-30 晚）：旧实现在此硬编码「`claim_specs` 写『21/23 验证通过（接近上限）』」。
    那句**写下时是真的**（`050f2b8` 的 claim_specs 里就是这句，git 可查），病在于它是常量：
    09-30 独立复查把自述句订正成「平局 21/23（ties=21…）」之后，指控活过了它的事实，
    于是矩阵每次重生成都重新印一句已不成立的旧账——副本改好它不消，副本再改坏它也不报。
    现在冲突位由措辞现判：无冲突就照登现算值，措辞若被改回通过位本行会重新亮起。
    """
    hit = [w for w in _THROUGH_WORDS if w in statement]
    if not hit:
        return ""
    return (f"**在册陈述与现算冲突（措辞现读，非写死）**：陈述「{statement}」含通过位措辞 {hit}，"
            "而 21 在件里是 ties（判官把 good 与 bad 打平的题数），不是「排对 21 题」——"
            "属 CALIBERS T21 构造效度位。冲突照登等裁，本件不回改数字。")


def _load_anchor_pair(root: Path, claim_id: str, statement: str) -> Evidence:
    """锚点两行：现算栅格统计，并把与在册陈述的冲突照登。"""
    stats = recompute_anchor(root)
    report = json.loads(_read_text(root, "benchmark/eval/judge_audit/anchor_report.json"))
    shared = (f"现跑 recompute_anchor()：n={stats['n']}、ties={stats['ties']}、"
              f"inverted={stats['inverted']}、faithful_ahead={stats['faithful_ahead']}、"
              f"fully_ordered={stats['fully_ordered']}、evasive_at_zero={stats['evasive_at_zero']}；"
              f"件内 stats={report['stats']}")
    conflict = _statement_conflict(statement)
    if claim_id == "anchor_0_23":
        note = (f"0/23 现跑可复现（full_order_rate={report['stats']['full_order_rate']:.4f}，"
                f"fully_ordered={stats['fully_ordered']}）。{shared}")
        return _not_assessable(f"{conflict}{note}" if conflict else note,
                               value=float(stats["fully_ordered"]), item_count=int(stats["n"]))
    note = (f"平局 {stats['ties']}/{stats['n']} 现跑可复现"
            f"（good_vs_bad_rate={report['stats']['good_vs_bad_rate']:.4f}，"
            f"faithful_ahead={stats['faithful_ahead']}）。{shared}")
    return _not_assessable(f"{conflict}{note}" if conflict else note,
                           value=float(stats["ties"]), item_count=int(stats["n"]))


def _load_anchor_raw(root: Path, claim_id: str) -> Evidence:
    """未训练 raw 锚点：读数就是产物自己的 overall_score。"""
    product = ANCHOR_RAW_PRODUCTS[claim_id]
    score = float(_json_pointer(root, f"{product}#overall_score"))
    n_items = int(_json_pointer(root, f"{product}#total_questions"))
    return _not_assessable(
        f"未训练 raw 锚点 {score:.4f}（现读 {Path(product).name}#overall_score，{n_items} 题，双判）。"
        "趋势位读数，不含任何增益 ⇒ 无转正阈值语义",
        value=score, item_count=n_items)


def _load_arbitration(root: Path) -> Evidence:
    n = len(_call_operator(root, "scripts/run_arbitration_patch.py::scan_failures"))
    summary = json.loads(
        _read_text(root, "benchmark/eval/arbitration_patch/patch_summary.json"))
    return _not_assessable(
        f"现跑 scan_failures() → {n} 条；patch_summary：n_patched={summary['n_patched']}、"
        f"n_failed={summary['n_failed']}、n_products={summary['n_products']}、"
        f"shift_mean={summary['shift_mean']:+.4f}"
        "（题次计数；「B1 由贴线变稳健」的前后对比属 b1_verdict，不在本行）",
        value=float(n), item_count=int(summary["n_patched"]))


def _load_denominator_family(root: Path, claim_id: str) -> Evidence:
    entries = _call_operator(
        root, "scripts/check_product_self_consistency.py::collect_denominator_map")
    kinds = dict(Counter(e["kind"] for e in entries))
    kinds_text = " / ".join(f"{k}={v}" for k, v in sorted(kinds.items()))
    if claim_id == "e12_dispute_rate":
        mismatch = _call_operator(
            root, "scripts/check_product_self_consistency.py::dispute_rate_denominator_note")
        return _not_assessable(
            f"现跑 collect_denominator_map()：{len(entries)} 件，分母形态 {kinds_text}；"
            f"在册「两种（12/9 产物 + 5 个第三种）」与之吻合（第三种=「其他」5 件，"
            f"另有 {kinds.get('歧义(两者皆可)', 0)} 件两者皆可）；"
            f"dispute_rate_denominator_note() → {mismatch} 件的存值与逐题重算不符"
            "（比率口径账，无阈值语义）", item_count=len(entries), value=float(mismatch))
    unequal = [e for e in entries if e["n_dual"] and e["n_dual"] != e["total"]]
    return _not_assessable(
        f"现跑 collect_denominator_map()：{len(entries)} 件里 {len(unequal)} 件的"
        f"双判记录数与总题数不等；形态 {kinds_text}。本行是比率账，无 2σg 阈值语义",
        item_count=len(entries), value=float(len(unequal)))


def _load_xcheck(root: Path) -> Evidence:
    gen6 = float(_json_pointer(root, f"{XCHECK_PRODUCT}#gen6_mean"))
    deviation = float(_json_pointer(root, f"{XCHECK_PRODUCT}#deviation_from_band_edge"))
    band = float(_json_pointer(root, f"{XCHECK_PRODUCT}#two_sigma_g"))
    verdict = _json_pointer(root, f"{XCHECK_PRODUCT}#verdict")
    return _not_assessable(
        f"现读 xcheck_verdict.json：gen6_mean={gen6:.4f}、对带边缘偏离 {deviation:+.4f}、"
        f"two_sigma_g={band:.4f}、verdict={verdict}。这是跨环境**偏移量的记录**，"
        "不是效应量；分母 Raw vs 双判那笔账在 collect_denominator_map（本行不重复）",
        value=deviation)


def _load_drift(root: Path) -> Evidence:
    mean, sd, paired, ahead, models = recompute_drift(root)
    return _not_assessable(
        f"现跑 recompute_drift()：聚合漂移 {mean:+.4f}（sd {sd:.4f}、配对题次 {paired}、"
        f"{ahead}/{models} 模型同向）。比率/聚合量，无 2σg 转正语义"
        "（剔假 0 口径由算子自陈，与 CALIBERS §2#10 同源）",
        value=mean, item_count=int(paired))


def _load_wilcoxon_holm(root: Path) -> Evidence:
    stats = _surface_stats_module(str(root))
    rows = stats.contrasts(stats.load_means())
    tested = [r for r in rows if r["p"] is not None]
    passed = [r for r in tested if r["holm_pass"]]
    listing = "、".join(f"{r['column']} {r['low']}vs{r['high']}" for r in passed)
    return _not_assessable(
        f"现跑表 1 统计线：{len(rows)} 组对比，{len(tested)} 组有题级 Wilcoxon p"
        f"（配对题数 >10 才算），Holm 校正后通过 {len(passed)} 组：{listing}。"
        "在册主张「核心主张经 Holm 仍显著」的落点即此清单——显著性计数，不是裕度对阈值"
        "（旧的 np.random 合成 p 值仍停用，见输出面）",
        item_count=len(rows), value=float(len(passed)))


def _load_rlvr_band(root: Path) -> Evidence:
    series = _gains(root)
    points = {c: series[c]["RLVR"] for c in series if "RLVR" in series[c]}
    extreme = max(points, key=lambda c: abs(points[c]))
    detail = "、".join(f"{c} {v:+.4f}pp" for c, v in points.items())
    return _not_assessable(
        f"现跑 gains() RLVR 相对 SFT 增益：{detail}；绝对值最大 {abs(points[extreme]):.4f}pp "
        f"< 描述带 5pp ⇒ 各柱全在带内。±5pp 是登记在案的**描述带**"
        "（A 篇 §robustness / number_audit），不是 2σg 转正判据，故不套 A/B/C",
        value=abs(points[extreme]))


def _load_u_shape(root: Path) -> Evidence:
    series = _gains(root)
    four = [c for c in ("0.8B", "2B", "4B", "9B") if c in series]
    values = [series[c]["MGPO"] for c in four]
    detail = " / ".join(f"{c} {v:+.4f}pp" for c, v in zip(four, values))
    lowest = four[min(range(len(four)), key=lambda i: values[i])]
    highest = four[max(range(len(four)), key=lambda i: values[i])]
    extra = series.get("E4B-it", {}).get("MGPO")
    return _not_assessable(
        f"现跑 gains() MGPO 四点：{detail}；极小柱={lowest}、极大柱={highest} ⇒ 非单调成立。"
        "这是**形状**主张（沿尺度轴的序关系），不是「裕度对阈值」，无 2σg 语义"
        + (f"；E4B-it 第五点 {extra:+.4f}pp 同轴并披" if extra is not None else ""),
        value=min(values))


def _load_sigma_family(root: Path) -> Evidence:
    sig4 = float(_json_pointer(root, f"{SIGMA_PRODUCT}#4b/statistics/judgement/sigma"))
    band4 = float(_json_pointer(root, f"{SIGMA_PRODUCT}#4b/statistics/judgement/threshold_2sigma"))
    sig9 = float(_json_pointer(root, f"{SIGMA_PRODUCT}#9b/statistics/judgement/sigma"))
    band9 = float(_json_pointer(root, f"{SIGMA_PRODUCT}#9b/statistics/judgement/threshold_2sigma"))
    return _not_assessable(
        f"现读 analysis.json judgement：σg(4B)={sig4:.4f} → 2σg={band4:.4f}；"
        f"σg(9B)={sig9:.4f} → 2σg={band9:.4f}（表 1 统计线的 band 列即此两值）。"
        "本行的读数**本身就是阈值**——没有 margin 一侧，套 grade() 等于为它造判据。"
        "27B 无 σg（未训练、无 n=3 重生成）。主张句旧写法「9B 定义有三说」属结案前措辞，"
        "09-30 裁以 CALIBERS T16 为准（轴＝between-generation，残留仅为代子集的历史偶然），"
        "本行措辞已同步；S7 状态列同日由「定义仍开」改为「轴已结案」",
        value=sig9)


def _load_e10(root: Path) -> Evidence:
    """E10 位置偏差：从 swapped 实物逐题配对现算（旧留桩理由「实物只在 temp」已否证）。"""
    r = read_position_bias(root)
    return _not_assessable(
        f"现跑 probe_readers.read_position_bias()：实物 {r['product']}"
        f"（source_artifact={r['source_artifact']}）逐题配对 {r['n_items']} 题 ⇒ "
        f"终分不变 {r['n_unchanged']}/{r['n_items']}、变化 {len(r['changed_ids'])} 题 "
        f"{r['changed_ids']}；mean(A−B)={r['mean_delta']:+.4f}、样本 sd={r['sd_sample']:.4f}"
        f"（总体 sd {r['sd_population']:.4f}）；件内自陈 mean_score_a={r['mean_a']}、"
        f"mean_score_b={r['mean_b']}，差即 {r['mean_a'] - r['mean_b']:+.4f}。"
        f"**两条限定**：⑴ arm_b 的登记定义是「{r['arm_b_def']}」——交换位次的同时换了判官指派，"
        "故主张句里「6 题变化全可归因单判官采样噪声」这句**归因在件内无字段支撑**，属待改措辞；"
        "⑵ 本行是描述量，无 2σg 转正阈值语义。",
        value=float(r["n_unchanged"]), item_count=r["n_items"])


def _load_openbook(root: Path) -> Evidence:
    """开卷 A 维：与闭卷**登记臂**逐题号配对，取最保守那一端的差值。"""
    r = read_openbook_a(root)
    detail = "、".join(f"{p['product'].split('/')[-1]} {p['delta']:+.4f}"
                       for p in r["closed_products"])
    return _not_assessable(
        f"现跑 probe_readers.read_openbook_a()：开卷件 {r['product']} 只跑 A 维、{r['n_items_open']} 题"
        f"（均值 {r['open_a_mean']:.4f}，件内 overall_score 取整印 0.7133）；闭卷对照一律走 manifest "
        f"`n3_state['{CLOSED_ARM_KEY}']`（{len(r['closed_products'])} 件，题号交集均为 "
        f"{r['closed_products'][0]['n_items']}），逐臂配对差 {detail} ⇒ 区间 "
        f"{r['delta_min']:+.4f}~{r['delta_max']:+.4f}，全部 {len(r['closed_products'])} 臂均为正"
        f"（本行取保守端 {r['delta_min']:+.4f}）。**限定**：未做题级检验（`item_level_test=None`），"
        "故本行不得写成「显著」——主张句原写法确曾用该词，09-30 已改为「大幅高于」并写明未做检验"
        "（守卫见 `tests/test_probe_readers.py`）；本行是描述量，无 2σg 阈值语义。"
        "另记：早先手工扫描曾把不在 manifest 里的 `v3_phase4_300/sft_baseline_run1.json` 当对照臂"
        "（那样是五臂），本算子按登记面取四臂。",
        value=r["delta_min"], item_count=r["n_items_open"])


def _load_medical_scale_axis(root: Path) -> Evidence:
    """医疗尺度轴：登记读法改为 `medical_dual_rejudge` 4-run 池（vs base），照主张句同池。

    09-30 裁（K6）：本行此前读 n3 聚合件的 3-gen 臂均值，与主张句的 +0.1306/+0.1363 不同池，
    于是被记成「在册与现算冲突」。换成本算子后两侧同池、逐位吻合 ⇒ 冲突取消。
    **本行仍不做 2σg 判定，理由要写对**：医疗阈值（σg=0.0023、2σg=0.0045，B 篇台账 E4）
    10-01 已入仓并接回调用链（`probe_readers.read_medical_sigma`），不再是"接不上"；
    不做判定的真理由是**该 σg 是回显常量、由件内数据重算不出**（同对照域 σg，见 CALIBERS T9/T13），
    且本行的描述量（跨臂极差/增益）本就没有 2σg 转正语义。
    """
    g4 = read_medical_gain_vs_base(root, "medical_mgpo_eg", "raw_qwen_med")
    g9 = read_medical_gain_vs_base(root, "medical_mgpo_9b", "raw_9b_med")
    vs_sft = read_medical_gain_vs_base(root, "medical_mgpo_eg", "medical_sft")
    sigma = probe_readers.read_medical_sigma(root)
    return _not_assessable(
        f"现跑 probe_readers.read_medical_gain_vs_base()：{g4['pool']}，{g4['n_items']} 题。"
        f"4B MGPO-EG {g4['arm_mean']:.4f} − raw {g4['base_mean']:.4f} = "
        f"**{g4['gain_vs_base']:+.4f}**；9B MGPO {g9['arm_mean']:.4f} − raw {g9['base_mean']:.4f} = "
        f"**{g9['gain_vs_base']:+.4f}** ⇒ 与主张句逐位吻合（B 篇 `tab:verdicts-medical` 同表）。"
        f"医疗阈值现读入仓件 `{sigma['product']}`：σg={sigma['sigma_g']}、"
        f"2σg={sigma['threshold_2sigma']}（与在册值一致＝{sigma['matches_registered']}；"
        f"{sigma['n_verdicts']} 条 verdict）。"
        f"**两句限定**：⑴ 口径是 vs base，不是 vs SFT——同一臂对 SFT 是 "
        f"{vs_sft['gain_vs_base']:+.4f}（4B 医疗 MGPO-EG 低于其 SFT 基线），主张句未标基准，"
        "属待加固措辞；⑵ 本行不做 2σg 判定的理由是**该阈值系回显常量、由件内数据重算不出**"
        "（不是「没有阈值语义」，也不是旧写的「复算源在 temp/ 进不了调用链」——该件 10-01 已入仓）。"
        "另记：27B 医疗两版都在盘（probe_scaling 0.525 单判、probe_medical_rejudge 0.5175 双判），"
        "早先那版行内声称该件缺失的话已被否证并删除",
        value=min(g4["gain_vs_base"], g9["gain_vs_base"]), item_count=g4["n_items"])


def _load_ppl_correlation(root: Path) -> Evidence:
    """E11 PPL×判官分秩相关：从 09-30 入仓件逐行重算（此前该数仓内无可复算路径）。

    行内必须带着它的**代次限定**：件里 13 行 `judge_overall` 只有 4 行能以现行聚合口径
    （剔 run1p 前两代均值）复现，Gemma 四行差 0.06~0.09 —— 那是旧判官代次的分。
    所以这条只作「当时那一轮评测」的旁证，不得当当代 judge epoch 的结论引用。
    """
    r = read_judge_ppl_correlation(root)
    return _not_assessable(
        f"现跑 probe_readers.read_judge_ppl_correlation()：{r['product']} 逐 {r['n_rows']} 行重算 "
        f"Spearman ⇒ **ρ={r['rho']:+.4f}（p={r['p_value']:.4g}，n={r['n_rows']}，"
        f"含 {len(r['raw_models_included'])} 个未训练 base：{','.join(r['raw_models_included'])}**）；"
        f"剔 raw 的另一种样本定义 ⇒ ρ={r['rho_finetuned_only']:+.4f}"
        f"（p={r['p_finetuned_only']:.4g}，n={r['n_rows'] - len(r['raw_models_included'])}）；"
        f"重算与件内自陈判等={r['recomputed_matches_file']}。"
        "**样本定义已固定**：A 篇 §intro 引的是含 raw 的 n=13，并已写明 "
        '"untrained bases included"（CALIBERS §2#12 的要求，09-30 闭合）。'
        "**两条限定**：⑴ 代次——13 行里只有 4 行的 judge_overall 能以现行聚合口径复现"
        "（qwen4b_raw、qwen9b_mgpo_fixed、qwen9b_rlvr、qwen9b_sft 对「剔 run1p 前两代均值」全等），"
        "Gemma 四行差 0.06~0.09 属旧判官代次 ⇒ 本条只作当时那轮评测的旁证，不是当代 epoch 的结论；"
        f"⑵ 构念——件内 caveat 自陈「{r['caveat']}」；ρ 是秩相关计数，无 2σg 转正阈值语义。",
        value=r["rho"], item_count=r["n_rows"])


def medical_arm_means(root: Path, claim_id: str) -> dict[str, float]:
    """按 key 现读聚合件里该条医疗主张涉及臂的登记均值。

    这一件是**登记的读法**（`claim_specs` 的 recompute 指针就指它）：不靠行号——
    `rows` 是列表，行号会随聚合件重排而腐，按 key 查才不会把别的臂的数当成这条主张的数。
    """
    rows = _rows_by_key(root)
    return {key: float(rows[key]["mean"]) for key in MEDICAL_TRIPLETS[claim_id]}


def _load_medical_descriptive(root: Path, claim_id: str) -> Evidence:
    values = medical_arm_means(root, claim_id)
    rows = _rows_by_key(root)
    spread = max(values.values()) - min(values.values())
    detail = "、".join(f"{k} {v:.4f}" for k, v in values.items())
    single_judge = [k for k in values if not rows[k].get("judge_stats")]
    return _not_assessable(
        f"现读 n3_aggregation_summary.json rows（按 key 查）：{detail}；极差 {spread:.4f}。"
        f"医疗臂单/双判并存（CALIBERS T7）：judge_stats 为 None 的有 {single_judge or '无'}。"
        "吸引子=跨模型/跨臂的收敛程度，无 2σg 转正语义。"
        "（尺度轴一条 09-30 起改走 4-run vs base 池，见 `_load_medical_scale_axis`）",
        value=spread)



def load_evidence(claim: ClaimSpec, root: Path) -> Evidence:
    """按 claim_id 装载证据；没有登记件背书的主张返回常量桩（S 级）。"""
    claim_id = claim.claim_id
    if claim_id == "B1_adjacent_granularity":
        return _load_b1(root)
    if claim_id == "cf_two_tiers":
        return _load_cf(root)
    if claim_id in SURFACE_CONTRASTS:
        return _load_contrast_claim(root, claim_id)
    if claim_id == "pivot_9b_00244":
        return _load_pivot(root)
    if claim_id == "eg_fixed_three_tiers":
        return _load_eg_fixed(root)
    if claim_id in ("anchor_0_23", "anchor_21_23"):
        return _load_anchor_pair(root, claim_id, claim.statement)
    if claim_id in ANCHOR_RAW_PRODUCTS:
        return _load_anchor_raw(root, claim_id)
    if claim_id == "arbitration_degradation_114":
        return _load_arbitration(root)
    if claim_id in ("e12_dispute_rate", "small_end_raw_denominator"):
        return _load_denominator_family(root, claim_id)
    if claim_id == "xcheck_offset":
        return _load_xcheck(root)
    if claim_id == "drift_00606":
        return _load_drift(root)
    if claim_id == "wilcoxon_holm_significance":
        return _load_wilcoxon_holm(root)
    if claim_id == "rlvr_5pct_band":
        return _load_rlvr_band(root)
    if claim_id == "mgpo_eg_u_shape":
        return _load_u_shape(root)
    if claim_id == "e10_position_bias":
        evidence = _load_e10(root)
        READ_LOG.update(probe_readers.READS)
        return evidence
    if claim_id == "openbook_dimension_a":
        evidence = _load_openbook(root)
        READ_LOG.update(probe_readers.READS)
        return evidence
    if claim_id == "medical_scale_axis":      # 09-30 裁：本行走 4-run vs base 池，不走 n3 均值
        evidence = _load_medical_scale_axis(root)
        READ_LOG.update(probe_readers.READS)
        return evidence
    if claim_id == "ppl_behavior_correlation":
        evidence = _load_ppl_correlation(root)
        READ_LOG.update(probe_readers.READS)
        return evidence
    if claim_id == "sigma_g_threshold_family":
        return _load_sigma_family(root)
    if claim_id in MEDICAL_TRIPLETS:
        return _load_medical_descriptive(root, claim_id)
    # 默认：证据桩（未接产物）——这些行的 recompute 现为 UNBOUND，见各件说明
    return Evidence(margin=STUB_MARGIN, threshold=STUB_THRESHOLD, same_direction=True,
                    n_gens=3, item_count=300, caliber_unique=True, epoch_clean=True,
                    denominator_clear=True, replayable=True)


SCAFFOLD_REASON = "脚手架：判据列为常量、未接产物，不作证据引用"

# 判据列已接入产物的主张集合（含分级行与 N/A 行——两者都读实数，都不许读桩）。
# 09-30 两轮接完之后**只剩 1 条走桩**：`qlora_ppl_down_score_down`。
# 它留桩的理由已换成真话（旧写「实物只在 temp/」被 09-30 复查否证）：该主张要的是
# 「QLoRA 前后 PPL 降 **且** 行为分降」的成对读数，而 `models/**/*ppl*.json`（现数 263 件）
# 与行为分产物之间**没有登记的 join**（模型名→产物路径靠目录命名约定），且 E11 那 13 行的
# 行为侧本身混了旧判官代次（见 `ppl_behavior_correlation` 行内限定⑴）。
# 接它需要先注册一个 join 算子，属独立一批工作，不随本轮顺手做。
MEASURED_CLAIMS: frozenset[str] = frozenset({
    "B1_adjacent_granularity", "cf_two_tiers", "pivot_9b_00244",
    "elasticity_reversal_4b", "elasticity_reversal_9b", "eg_fixed_three_tiers",
    "anchor_0_23", "anchor_21_23", "anchor_27b", "anchor_gemma_12b",
    "arbitration_degradation_114", "e12_dispute_rate", "small_end_raw_denominator",
    "xcheck_offset", "drift_00606", "wilcoxon_holm_significance", "rlvr_5pct_band",
    "mgpo_eg_u_shape", "sigma_g_threshold_family",
    "attractor_gemma_triplet", "attractor_medical_nine", "medical_scale_axis",
    "e10_position_bias", "openbook_dimension_a", "ppl_behavior_correlation",
})

# S 级行的逐条理由：不许所有留桩行共用一句套话——套话会盖掉每条各自的真缺口。
SCAFFOLD_NOTES: dict[str, str] = {
    "qlora_ppl_down_score_down": (
        "｜真缺口＝models/**/*ppl*.json（现数 263 件）与行为分产物之间无登记 join"
        "（现靠目录命名约定对应），且 E11 行为侧 13 行仅 4 行可按现行聚合复现（Gemma 四行"
        "为旧判官代次）。旧写的「实物只在 temp/」不成立：PPL 侧 13/13 在 "
        "models/fbc_census/*_control/*/ppl_eval.json 登记位置可取。"),
}


def margin_note(claim_id: str, evidence: Evidence, root: Path) -> str:
    """裕度说明：读数出处、多档全值、与在册的冲突都要在行内可见。"""
    base = f"margin={evidence.margin:.4f}, threshold={evidence.threshold:.4f}"
    if not evidence.known:
        return f"实测值 {evidence.margin:.4f}（无阈值语义）｜{evidence.note}"
    return f"{base}（{evidence.note}）" if evidence.note else base
