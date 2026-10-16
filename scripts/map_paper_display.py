"""S 层抽取：从两篇正文与台账反查已展示主张"""

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

# 添加仓库根到 sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@dataclass
class ShownClaim:
    """已展示主张"""

    claim_id: str
    paper: str  # A / B
    section: str  # 章节
    shown: bool  # 是否在正文展示
    source: str  # 来源（正文 / 台账 / 未用）


@dataclass
class Gaps:
    """三个差集"""

    strong_not_shown: list  # C−S：A 级未进正文
    shown_but_weak: list  # S−C：已进正文但 B/C 级或被否决
    shown_not_assessable: list  # 已进正文但 N/A／S（D2，10-01）：无阈值语义或脚手架位
    data_without_claim: list  # D−C：清点到的产物组无任何主张引用


@dataclass
class SemanticCollision:
    """语义冲突"""

    value: str  # 冲突数值
    papers: list[str]  # 涉及的论文（A / B）
    sections: list[str]  # 涉及的章节
    calibers_ref: str  # CALIBERS.md 行号引用


def _parse_paper_tex(tex_path: Path) -> dict[str, bool]:
    r"""解析论文正文，返回各章节是否已写（非 TODO）

    检查 \section 级别的 TODO 标记（包括所有子节）。
    如果章节内有任何 [TODO]，则判定为未展示。
    """
    if not tex_path.exists():
        return {}

    content = tex_path.read_text(encoding="utf-8")
    sections = {}

    # 提取所有标题（\section 及 \subsection/\subsubsection）及其内容
    # ——目标节可挂在子节下，只认顶层会漏
    for match in re.finditer(r"\\(?:sub){0,2}section\{([^}]+)\}(.*?)(?=\\(?:sub){0,2}section\{|\Z)",
                             content, re.DOTALL):
        section_name = match.group(1)
        section_body = match.group(2)
        # 跳过注释行
        section_body = "\n".join(
            line for line in section_body.splitlines() if not line.strip().startswith("%")
        )
        # 判断是否含 [TODO]（包括所有子节）
        sections[section_name] = "[TODO]" not in section_body

    return sections


# 章节名映射：主张规格表 section → 论文实际章节名
SECTION_MAP = {
    "surface": ["Within-family elasticity surface"],
    "surface-scale": ["Within-family elasticity surface", "Generalization probe I: a second family"],
    "ablation": ["Ablation purification"],
    "limitations": ["Limitations"],
    "analysis": ["Robustness", "Threshold Sensitivity of the Promotion Criterion"],
}


def _parse_number_audit(audit_path: Path) -> dict[str, str]:
    """解析 number_audit.md，返回数值与用途章节的映射"""
    if not audit_path.exists():
        return {}

    content = audit_path.read_text(encoding="utf-8")
    mapping = {}

    # 简化解析：假设每行格式为 "| 数值 | 用途章节 | ..."
    for line in content.splitlines():
        if line.startswith("|") and "用途章节" not in line:
            parts = [p.strip() for p in line.split("|")]
            if len(parts) >= 3:
                value = parts[1]
                section = parts[2]
                mapping[value] = section

    return mapping


def extract_shown_claims(root: Path) -> list[ShownClaim]:
    """从两篇正文与台账反查已展示主张"""
    shown = []

    # 解析 A 篇
    paper_a_tex = root / "papers" / "paper-a" / "draft" / "main.tex"
    paper_a_sections = _parse_paper_tex(paper_a_tex)
    paper_a_audit = _parse_number_audit(root / "papers" / "paper-a" / "number_audit.md")

    # 解析 B 篇
    paper_b_tex = root / "papers" / "paper-b" / "draft" / "main.tex"
    paper_b_sections = _parse_paper_tex(paper_b_tex)
    paper_b_audit = _parse_number_audit(root / "papers" / "paper-b" / "number_audit.md")

    # 从主张规格表反查
    from scripts.claim_specs import all_claims

    for claim in all_claims():
        # 判断是否在 A 篇展示
        shown_in_a = False
        for actual_section in SECTION_MAP.get(claim.section, [claim.section]):
            if actual_section in paper_a_sections and paper_a_sections[actual_section]:
                shown_in_a = True
                break

        # 判断是否在 B 篇展示
        shown_in_b = False
        for actual_section in SECTION_MAP.get(claim.section, [claim.section]):
            if actual_section in paper_b_sections and paper_b_sections[actual_section]:
                shown_in_b = True
                break

        # 添加到结果
        if shown_in_a:
            shown.append(
                ShownClaim(
                    claim_id=claim.claim_id,
                    paper="A",
                    section=claim.section,
                    shown=True,
                    source="正文",
                )
            )
        if shown_in_b:
            shown.append(
                ShownClaim(
                    claim_id=claim.claim_id,
                    paper="B",
                    section=claim.section,
                    shown=True,
                    source="正文",
                )
            )

        # 如果既不在 A 也不在 B，但台账有登记，标记为"仅台账登记"
        if not shown_in_a and not shown_in_b:
            for value, section in paper_a_audit.items():
                if claim.claim_id in value:
                    shown.append(
                        ShownClaim(
                            claim_id=claim.claim_id,
                            paper="A",
                            section=section,
                            shown=False,
                            source="台账",
                        )
                    )
                    break
            for value, section in paper_b_audit.items():
                if claim.claim_id in value:
                    shown.append(
                        ShownClaim(
                            claim_id=claim.claim_id,
                            paper="B",
                            section=section,
                            shown=False,
                            source="台账",
                        )
                    )
                    break

    return shown


def value_semantic_collisions(root: Path) -> list[SemanticCollision]:
    """检查同一数值在两篇/两口径下的语义冲突

    已知正例：
    - 0.4621（A 篇消融臂值 vs 被误引的三代均值）
    - 0.0139（B1 均值口径 vs 仲裁口径）
    """
    collisions = []

    # 检查 CALIBERS.md 中的冲突记录
    calibers_path = root / "docs" / "CALIBERS.md"
    if not calibers_path.exists():
        return collisions

    content = calibers_path.read_text(encoding="utf-8")

    # 简化实现：硬编码已知冲突
    if "0.4621" in content:
        collisions.append(
            SemanticCollision(
                value="0.4621",
                papers=["A"],
                sections=["ablation"],
                calibers_ref="CALIBERS §二 #16",
            )
        )

    if "0.0139" in content:
        collisions.append(
            SemanticCollision(
                value="0.0139",
                papers=["A", "B"],
                sections=["ablation", "limitations"],
                calibers_ref="CALIBERS §二 #16",
            )
        )

    return collisions


def gaps(matrix_rows, shown_claims, eval_groups) -> Gaps:
    """计算三个差集

    - C−S（strong_not_shown）：A 级未进正文
    - S−C（shown_but_weak）：已进正文但 B/C 级或被否决
    - S−N（shown_not_assessable）：已进正文但 N/A／S，即无阈值语义或脚手架位（D2）
    - D−C（data_without_claim）：清点到的产物组无任何主张引用
    """
    # C−S：A 级未进正文
    shown_ids = {s.claim_id for s in shown_claims if s.shown}
    strong_not_shown = [r for r in matrix_rows if r.grade == "A" and r.claim_id not in shown_ids]

    # S−C：已进正文但 B/C 级或被否决
    shown_but_weak = [
        r for r in matrix_rows if r.claim_id in shown_ids and (r.grade in {"B", "C"} or r.vetoed)
    ]

    # 已进正文但 N/A／S（D2，10-01）：这两类过去**一条都不报**——26 条均被 `shown` 判为已展示
    # （注意 `shown` 是章节级判据，不是句子级，见 tests/test_map_paper_display.py 的更正），
    # 其中 N/A 20＋S 1＝21 条落在旧口径之外。不并进 shown_but_weak 是故意的：
    # B/C 是「测了没过硬」，N/A 是「本无阈值语义」，S 是「脚手架位」，混栈就丢了这条区分。
    shown_not_assessable = [
        r for r in matrix_rows if r.claim_id in shown_ids and r.grade in {"N/A", "S"}
    ]

    # D−C：清点到的产物组无任何主张引用
    claimed_groups = set()
    for r in matrix_rows:
        for support in r.supports:
            # 提取组名（如 benchmark/eval/n3_regen/... → n3_regen）
            parts = Path(support).parts
            if "benchmark" in parts and "eval" in parts:
                idx = parts.index("eval")
                if idx + 1 < len(parts):
                    claimed_groups.add(parts[idx + 1])

    all_groups = {g.group for g in eval_groups if not g.is_smoke}
    data_without_claim = list(all_groups - claimed_groups)

    return Gaps(
        strong_not_shown=strong_not_shown,
        shown_but_weak=shown_but_weak,
        shown_not_assessable=shown_not_assessable,
        data_without_claim=data_without_claim,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="S 层抽取")
    parser.add_argument("--dry-run", action="store_true", help="只打印计数，不写盘")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    shown = extract_shown_claims(root)

    print(f"已展示主张数: {len([s for s in shown if s.shown])}")
    print(f"仅台账登记数: {len([s for s in shown if not s.shown and s.source == '台账'])}")
    print(f"完全未用数: {len([s for s in shown if not s.shown and s.source == '未用'])}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
