"""强度评分纯函数测试（TDD）"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_grade_a():
    """A 级：过阈且裕度 ≥ 2×σg"""
    from scripts.claim_strength import Evidence, grade

    def ev(**kw):
        return Evidence(
            margin=0.05,
            threshold=0.0132,
            same_direction=True,
            n_gens=3,
            item_count=300,
            caliber_unique=True,
            epoch_clean=True,
            denominator_clear=True,
            replayable=True,
            **kw,
        )

    assert grade(ev()).grade == "A"


def test_grade_b_borderline():
    """B 级：过阈但贴线（0 < 裕度 < 2×σg）"""
    from scripts.claim_strength import Evidence, grade

    def ev(**kw):
        return Evidence(
            margin=0.0139,  # B1 均值替代口径裕度 0.0007 < 2σg
            threshold=0.0132,
            same_direction=True,
            n_gens=3,
            item_count=300,
            caliber_unique=True,
            epoch_clean=True,
            denominator_clear=True,
            replayable=True,
            **kw,
        )

    assert grade(ev()).grade == "B"


def test_veto_same_direction():
    """一票否决：同向性不过"""
    from scripts.claim_strength import Evidence, grade

    def ev(**kw):
        return Evidence(
            margin=0.05,
            threshold=0.0132,
            same_direction=False,  # 否决项
            n_gens=3,
            item_count=300,
            caliber_unique=True,
            epoch_clean=True,
            denominator_clear=True,
            replayable=True,
            **kw,
        )

    result = grade(ev())
    assert result.vetoed is True
    assert "same_direction" in result.reason


def test_veto_caliber_unique():
    """一票否决：口径不唯一"""
    from scripts.claim_strength import Evidence, grade

    def ev(**kw):
        return Evidence(
            margin=0.05,
            threshold=0.0132,
            same_direction=True,
            n_gens=3,
            item_count=300,
            caliber_unique=False,  # 否决项
            epoch_clean=True,
            denominator_clear=True,
            replayable=True,
            **kw,
        )

    result = grade(ev())
    assert result.vetoed is True
    assert "caliber_unique" in result.reason


def test_veto_epoch_clean():
    """一票否决：epoch 混杂"""
    from scripts.claim_strength import Evidence, grade

    def ev(**kw):
        return Evidence(
            margin=0.05,
            threshold=0.0132,
            same_direction=True,
            n_gens=3,
            item_count=300,
            caliber_unique=True,
            epoch_clean=False,  # 否决项
            denominator_clear=True,
            replayable=True,
            **kw,
        )

    result = grade(ev())
    assert result.vetoed is True
    assert "epoch_clean" in result.reason


def test_replayable_gap_only_downgrade():
    """可复演性缺口只降级不否决（σg T13 型）"""
    from scripts.claim_strength import Evidence, grade

    def ev(**kw):
        return Evidence(
            margin=0.05,
            threshold=0.0132,
            same_direction=True,
            n_gens=3,
            item_count=300,
            caliber_unique=True,
            epoch_clean=True,
            denominator_clear=True,
            replayable=False,  # 降级项，不否决
            **kw,
        )

    result = grade(ev())
    assert result.vetoed is False
    assert result.grade == "B"


def test_na_short_circuits_before_sigma_ratio():
    """N/A 行必须在反解 σg 之前返回——否则 ±5pp 这类描述带会被印成「σg=2.5pp」。

    09-30 R2a/R3 的机器形态：`known=False` 时不看 margin/threshold/否决位，直接 N/A。
    这里故意把 threshold 设成描述带的量级（5.0），若实现走反解就会算出 σg=2.5 并印进理由。
    """
    from scripts.claim_strength import Evidence, grade

    evidence = Evidence(margin=4.7044, threshold=5.0, same_direction=False, n_gens=0,
                        item_count=300, caliber_unique=False, epoch_clean=False,
                        denominator_clear=False, replayable=False,
                        known=False, note="现跑 gains()：各柱全在 ±5pp 描述带内")
    result = grade(evidence, sigma_g=evidence.threshold / 2)
    assert result.grade == "N/A"
    assert result.vetoed is False
    assert "σg" not in result.reason, f"N/A 的理由里不该出现 σg 比值: {result.reason}"
    assert "5.0" not in result.reason


def test_na_known_true_falls_back_but_must_say_so():
    """`known=True` 却带 N/A 语义的行不该悄悄拿到分级——margin=0 走 C，不装 A/B。"""
    from scripts.claim_strength import Evidence, grade

    evidence = Evidence(margin=0.0, threshold=0.0132, same_direction=True, n_gens=3,
                        item_count=300, caliber_unique=True, epoch_clean=True,
                        denominator_clear=True, replayable=True)
    result = grade(evidence, sigma_g=evidence.threshold / 2)
    assert result.grade == "C"


def test_holm_pass_false_downgrades_a_to_b():
    """R2b：达带但题级 Holm 不过 ⇒ 保守侧降 B，而不是印 A。

    实物来源就是 9B 的 MGPO-EG vs SFT：margin +0.0333 对带 0.0332（超 1e-4），
    题级 p=0.038 不过 Holm，A 篇 number_audit 把该格定为「表内最薄支持格」。
    """
    from scripts.claim_strength import Evidence, grade

    def ev(**kw):
        return Evidence(margin=0.0333, threshold=0.0332, same_direction=True, n_gens=3,
                        item_count=300, caliber_unique=True, epoch_clean=True,
                        denominator_clear=True, replayable=True, **kw)

    assert grade(ev(), sigma_g=0.0332 / 2).grade == "A"          # 无 Holm 件：不动既有分级
    assert grade(ev(holm_pass=None), sigma_g=0.0332 / 2).grade == "A"
    downgraded = grade(ev(holm_pass=False), sigma_g=0.0332 / 2)
    assert downgraded.grade == "B"
    assert "Holm" in downgraded.reason
