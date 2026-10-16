"""强度评分纯函数：A/B/C/N/A/VETO 分级规则"""

from dataclasses import dataclass


@dataclass
class Evidence:
    """证据输入"""

    margin: float  # 裕度（实际值 - 阈值）
    threshold: float  # 阈值
    same_direction: bool  # 同向性
    n_gens: int  # 代次数
    item_count: int  # 题次
    caliber_unique: bool  # 口径唯一性
    # epoch 洁净度：指**本行 margin 所在登记代集合内**各臂同判官代次。
    # 不含题级 p 的池——那一池按 manifest 全 roster，9B MGPO-EG 臂含 2026-07-24 的旧判官件，
    # 该不一致登记在 CALIBERS T22/T23，是否落成实读属 K3 待裁（现仍为本窗判定）。
    epoch_clean: bool
    denominator_clear: bool  # 分母口径清晰
    replayable: bool  # 可复演性
    # 09-30 接线时补的三个位（按批准 R2b / R2a）：
    known: bool = True  # False ⇒ 该行没有 2σg 转正阈值语义，分级为 N/A（只报实测值与出处）
    holm_pass: bool | None = None  # 题级 Wilcoxon+Holm 是否通过；None = 该主张没有题级检验件
    note: str = ""  # N/A 行的实测陈述（值 + 出处），不得留空


@dataclass
class GradeResult:
    """分级结果"""

    grade: str  # A / B / C / N/A / VETO
    vetoed: bool  # 是否被否决
    reason: str  # 分级理由


def grade(evidence: Evidence, sigma_g: float = 0.0132) -> GradeResult:
    """分级函数

    规则：
    - N/A：`known=False` 的行——读数是真的，但没有 2σg 转正阈值语义（描述量、
      描述带、形状主张、阈值族本身）。只报实测值与出处，不评级、不冒充 A/B/C。
    - 一票否决（VETO）：同向性不过、口径不唯一、epoch 混杂、分母口径不可辨
    - A 级：过阈且裕度 ≥ 2×σg，n_gens 与题次均显式，混杂已逐段披露，可复演，
      且题级 Wilcoxon+Holm 没有被判不过（09-30 R2b）
    - B 级：过阈但贴线（0 < 裕度 < 2×σg）或可复演性有缺口，或带外但题级 Holm 不过
    - C 级：描述性、未过阈、或依赖未裁定口径
    """
    if not evidence.known:
        return GradeResult(grade="N/A", vetoed=False,
                           reason=evidence.note or "N/A：读数已实测，但该行没有阈值语义")

    # 一票否决项
    if not evidence.same_direction:
        return GradeResult(grade="VETO", vetoed=True, reason="same_direction=False")
    if not evidence.caliber_unique:
        return GradeResult(grade="VETO", vetoed=True, reason="caliber_unique=False")
    if not evidence.epoch_clean:
        return GradeResult(grade="VETO", vetoed=True, reason="epoch_clean=False")
    if not evidence.denominator_clear:
        return GradeResult(grade="VETO", vetoed=True, reason="denominator_clear=False")

    # 过阈判定
    if evidence.margin <= 0:
        return GradeResult(grade="C", vetoed=False, reason="margin <= 0（未过阈）")

    # 裕度判定
    margin_ratio = evidence.margin / sigma_g if sigma_g > 0 else 0
    if margin_ratio >= 2.0:
        # A 级候选：先查题级显著性（09-30 R2b）
        if evidence.holm_pass is False:
            return GradeResult(
                grade="B",
                vetoed=False,
                reason=(f"裕度 {margin_ratio:.2f}×σg ≥ 2×σg 但题级 Holm 不过"
                        "（两判据不一致，按 CALIBERS T16 保守侧降级）"),
            )
        if not evidence.replayable:
            return GradeResult(
                grade="B",
                vetoed=False,
                reason=f"裕度 {margin_ratio:.2f}×σg ≥ 2×σg 但 replayable=False（降级）",
            )
        return GradeResult(
            grade="A",
            vetoed=False,
            reason=f"裕度 {margin_ratio:.2f}×σg ≥ 2×σg，可复演",
        )
    else:
        # B 级：贴线
        return GradeResult(
            grade="B",
            vetoed=False,
            reason=f"裕度 {margin_ratio:.2f}×σg < 2×σg（贴线）",
        )
