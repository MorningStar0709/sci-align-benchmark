"""主张矩阵生成器：分级引擎的输出面。

分工（09-30 拆分，AGENTS 的 300 行线到了）：
  `scripts/claim_evidence.py`  —— 证据面：每条主张读哪个登记件的哪个字段、调哪个登记算子
  `scripts/claim_strength.py`  —— 分级规则：A/B/C/N/A/VETO
  `scripts/surface_direction.py` —— 逐代同向读数（转正判据的合取前件）
  本件 —— 输出面：指针可达性校验、建表、渲染
"""

import argparse
import ast
import json
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

# 添加仓库根到 sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.claim_evidence import (  # noqa: E402
    MEASURED_CLAIMS,
    SCAFFOLD_NOTES,
    SCAFFOLD_REASON,
    load_evidence,
    margin_note,
)
from scripts.claim_specs import ClaimSpec, all_claims  # noqa: E402
from scripts.claim_strength import Evidence, grade  # noqa: E402


@dataclass
class MatrixRow:
    """矩阵行"""

    claim_id: str
    statement: str
    domain: str
    section: str
    grade: str  # A / B / C / N/A / VETO / S（脚手架）
    vetoed: bool
    reason: str
    margin_note: str  # 裕度或实测说明（含读数出处）
    effect_size: float | None  # 效应量／实测读数（单列呈现）
    wilcoxon_p: float | None  # Wilcoxon p 值（合成 p 已停用，恒为 None）
    holm_rejected: bool | None  # Holm 校正后是否拒绝（同上）
    recompute_expr: str  # 可复算表达式
    supports: list[str]
    caliber_ids: list[str]


POINTER_UNRESOLVED = "⚠ 指针未解析（该表达式在盘上取不到值，不得当可复算依据引用）"


def resolve_pointer(root: Path, expr: str) -> tuple[bool, str]:
    """判一条 `recompute` 表达式是否真能解析。

    为什么必须有这条：09-30 盘点发现 26 条主张里有 25 条写着
    `某脚本.py::compute_某东西`，而那个函数**从来没被实现过**——矩阵表因此印着一列
    跑不通的"可复算表达式"。空头指针比空白更坏：它读起来像证据。
    支持两种形态：`path::func`（AST 查顶层 def，不 import ⇒ 无副作用）
    与 `path#json/pointer`（文件存在 + 指针可达）。
    """
    if "::" in expr:
        path_s, func = expr.split("::", 1)
        func = func.split("(")[0].strip()
        f = root / path_s.strip()
        if not f.is_file():
            return False, "文件不存在"
        try:
            tree = ast.parse(f.read_text(encoding="utf-8"))
        except SyntaxError as exc:
            return False, f"语法错误 {exc.msg}"
        defined = {n.name for n in ast.walk(tree)
                   if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}
        return (True, "") if func in defined else (False, f"无函数 {func}()")
    if "#" in expr:
        path_s, pointer = expr.split("#", 1)
        f = root / path_s.strip()
        if not f.is_file():
            return False, "产物文件不存在"
        node = json.loads(f.read_text(encoding="utf-8"))
        for part in filter(None, pointer.strip("/").split("/")):
            if isinstance(node, list):
                if not part.isdigit() or int(part) >= len(node):
                    return False, f"数组下标越界 {part}"
                node = node[int(part)]
            elif isinstance(node, dict) and part in node:
                node = node[part]
            else:
                return False, f"键不存在 {part}"
        return True, ""
    return False, "表达式形态未知"


def _load_evidence(claim: ClaimSpec, root: Path) -> Evidence:
    """转调证据装载层（保留这个名字，既有测试与调用面不破）。"""
    return load_evidence(claim, root)


def build_matrix(root: Path) -> list[MatrixRow]:
    """构建主张矩阵"""
    rows = []
    for claim in all_claims():
        evidence = load_evidence(claim, root)
        # σg 一律由阈值反解（threshold/2）。旧写法对 B1 硬塞 σg=0.006
        # 以使「裕度 ≥ 2σg」成立——那是反推判据，已删（V11）。
        # N/A 行在 grade() 里先于反解返回，所以不会被塞进任何 σg 比值（R3 的坑）。
        result = grade(evidence, sigma_g=evidence.threshold / 2)

        # Wilcoxon+Holm：**已停用**。此前对 4 条核心主张用 np.random.seed(42) 造
        # 合成配对样本算 p 值——那是把随机数当证据（value-version-audit V11）。
        # 真实的题级 Wilcoxon + Holm 在 surface_stats.contrasts() 里做，
        # 结论落在表 tab:cross-family-stats；矩阵只由 `wilcoxon_holm_significance`
        # 一行现跑引用其计数，不再自己造 p 值。
        measured = claim.claim_id in MEASURED_CLAIMS

        row = MatrixRow(
            claim_id=claim.claim_id,
            statement=claim.statement,
            domain=claim.domain,
            section=claim.section,
            grade=result.grade if measured else "S",
            vetoed=result.vetoed if measured else False,
            reason=result.reason if measured else SCAFFOLD_REASON,
            margin_note=margin_note(claim.claim_id, evidence, root) if measured
            else "未实测（证据桩常量）" + SCAFFOLD_NOTES.get(claim.claim_id, ""),
            effect_size=evidence.margin if measured else None,
            wilcoxon_p=None,
            holm_rejected=None,
            recompute_expr=(claim.recompute if resolve_pointer(root, claim.recompute)[0]
                            else f"{claim.recompute} {POINTER_UNRESOLVED}"),
            supports=claim.supports,
            caliber_ids=claim.caliber_ids,
        )
        rows.append(row)
    return rows


def _grade_counts(rows: list[MatrixRow]) -> dict[str, int]:
    counts = {"A": 0, "B": 0, "C": 0, "N/A": 0, "VETO": 0, "S": 0}
    for row in rows:
        counts[row.grade] = counts.get(row.grade, 0) + 1
    return counts


def _to_json(rows: list[MatrixRow]) -> str:
    """生成 JSON 输出"""
    return json.dumps(
        [
            {
                "claim_id": r.claim_id,
                "statement": r.statement,
                "domain": r.domain,
                "section": r.section,
                "grade": r.grade,
                "vetoed": r.vetoed,
                "reason": r.reason,
                "margin_note": r.margin_note,
                "effect_size": r.effect_size,
                "wilcoxon_p": r.wilcoxon_p,
                "holm_rejected": r.holm_rejected,
                "recompute_expr": r.recompute_expr,
                "supports": r.supports,
                "caliber_ids": r.caliber_ids,
            }
            for r in rows
        ],
        ensure_ascii=False,
        indent=2,
    )


def _to_markdown(rows: list[MatrixRow]) -> str:
    """生成 Markdown 输出"""
    from datetime import datetime

    counts = _grade_counts(rows)
    lines = [
        "# 主张-证据矩阵",
        "",
        f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "## 分级统计",
        "",
        "| 分级 | 数量 | 含义 |",
        "|:--|--:|:--|",
        "| A 级 | {} | 达 2σg 冻结带、逐代同向、题级 Holm 未被判不过"
        "（`holm_pass=None` 表示该对比没有题级检验件，按\"未判不过\"处理，不等于\"判过\"）、可复演 |".format(counts["A"]),
        "| B 级 | {} | 达带但降级（Holm 不过／可复演有缺口），或 0 < margin < 带 |".format(counts["B"]),
        "| C 级 | {} | 未过阈（margin ≤ 0） |".format(counts["C"]),
        "| N/A | {} | 读数已实测，但该行没有可用的 2σg 转正阈值语义"
        "（无该轴阈值，或有阈值而其复算登记源进不了调用链——后者在行内写明） |".format(counts["N/A"]),
        "| VETO | {} | 一票否决（同向／口径／epoch／分母不过） |".format(counts["VETO"]),
        "| S | {} | 脚手架：判据列为常量、未接产物，不作证据引用 |".format(counts["S"]),
        "",
        "> **「否决＝否」这一列不是四道前件都跑过。** `grade()` 的前件里 `same_direction`、",
        "> `denominator_clear` 与 **`epoch_clean`（09-30 K3 起改为实读，见",
        "> `scripts/surface_direction.py::epoch_report`——按本行 margin 所在登记代集合读各臂",
        "> `judge_model` 与 `updated_at` 批次，读不到即判不洁净并点名）** 是读出来的；",
        "> `caliber_unique` 与 `replayable` 在判据型行仍是常量 `True`（逐条具名见",
        "> `scripts/claim_evidence.py` 的各 `_load_*`）。注意 `epoch_clean` 的语义**只覆盖登记代",
        "> 集合**，不覆盖题级 p 的 manifest 全 roster（那一池的跨判官代次事实归 CALIBERS T22/T23，",
        "> 已裁维持不修）。引用本表的「否决＝否」时按此折扣。（本段 09-30 独立复查后加，",
        "> 非分级引擎的输出。）",
        "",
        "> **离开 S 的条数 = 接通的条数。** 读数一律现跑：JSON 指针读登记件字段，"
        "或调登记算子；逐条读的是哪个文件哪个字段写在 `scripts/claim_evidence.py`，"
        "行内「裕度/实测说明」列同时给出处。",
        "",
        "## Wilcoxon+Holm 统计检验（已停用）",
        "",
        "> 本节曾对 4 条核心主张输出 p 值，其配对样本由 `np.random.seed(42)` 合成，"
        "**不是**题级实测（value-version-audit V11）。已停用。",
        "> 真实的题级 Wilcoxon + Holm（18 对比、300 题配对）见 "
        "`figures/scripts/surface_stats.py` 与表 `tab:cross-family-stats`；"
        "矩阵里由 `wilcoxon_holm_significance` 一行现跑引用其计数。",
        "",
        "| 主张 ID | Wilcoxon p | Holm 校正后拒绝 |",
        "|:--|--:|:--:|",
    ]

    emitted = [r for r in rows if r.wilcoxon_p is not None]
    for r in emitted:
        lines.append(f"| {r.claim_id} | {r.wilcoxon_p:.4f} | {'是' if r.holm_rejected else '否'} |")
    if not emitted:
        lines.append("| （无——合成 p 值已停用） | — | — |")

    lines.extend([
        "",
        "## 矩阵详表",
        "",
        "> 分级 `S` = 判据列为常量、未接产物；`N/A` = 读数是真的但没有可用的转正阈值语义"
        "（描述量／描述带／形状主张／阈值族本身，或有阈值而登记源接不上）。"
        "**引用本表的 A 级结论前，"
        "先确认该行不是 S 或 N/A。**",
        "",
        "| 主张 ID | 陈述 | 域 | 章节 | 分级 | 否决 | 理由 | 裕度/实测说明 | 实测读数（量纲见行内） | 可复算表达式 |",
        "|:--|:--|:--|:--|:--:|:--:|:--|:--|--:|:--|",
    ])

    for r in rows:
        effect_size_str = f"{r.effect_size:.4f}" if r.effect_size is not None else "—"
        lines.append(
            f"| {r.claim_id} | {r.statement} | {r.domain} | {r.section} | {r.grade} | "
            f"{'是' if r.vetoed else '否'} | {r.reason} | {r.margin_note} | {effect_size_str} | {r.recompute_expr} |"
        )

    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="主张矩阵生成器")
    parser.add_argument("--dry-run", action="store_true", help="只打印计数，不写盘")
    parser.add_argument("--emit", type=Path, help="输出到指定文件（.md 或 .json）")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    rows = build_matrix(root)
    counts = _grade_counts(rows)

    print(f"矩阵行数: {len(rows)}")
    for name in ("A", "B", "C", "N/A", "VETO", "S"):
        print(f"{name} 级: {counts[name]}")
    bad = [r.claim_id for r in rows if POINTER_UNRESOLVED in r.recompute_expr]
    print(f"recompute 指针可解析: {len(rows) - len(bad)}/{len(rows)}"
          + (f"；未解析 {len(bad)} 条：{bad}" if bad else ""))
    unbound = [r.claim_id for r in rows if r.recompute_expr.startswith("UNBOUND")]
    print(f"显式 UNBOUND: {len(unbound)} 条：{unbound}")

    if args.dry_run:
        return 0

    if args.emit:
        emit_path = args.emit.resolve()
        # 落点守卫：限定 docs/reports/
        if "docs/reports" not in str(emit_path):
            print(f"错误：--emit 必须落在 docs/reports/ 下，当前: {emit_path}")
            return 1

        # 原子写：临时名必须不可猜（`<out>.tmp` 会吞掉用户按提示另存的备份），再 replace
        fd, tmp_name = tempfile.mkstemp(dir=str(emit_path.parent),
                                        prefix=f".{emit_path.name}.", suffix=".tmp")
        content = _to_json(rows) if emit_path.suffix == ".json" else _to_markdown(rows)
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(content)
        Path(tmp_name).replace(emit_path)
        print(f"已生成: {emit_path}")

    return 0


if __name__ == "__main__":
    from scripts.claim_evidence import WorkbenchOnly  # noqa: E402

    try:
        sys.exit(main())
    except WorkbenchOnly as exc:
        print(f"[WORKBENCH-ONLY] {exc}")
        print("[UNCHECKED] 这一项在本树不可复算 ⇒ 记为未验，不得当通过")
        sys.exit(1)
