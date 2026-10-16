"""Wilcoxon + Holm 统计量测试（TDD）"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_holm_adjust():
    """Holm 校正：已知小例"""
    from scripts.holm_wilcoxon import holm_adjust

    assert holm_adjust([0.01, 0.04, 0.03], alpha=0.05) == [True, False, False]


def test_wilcoxon_p():
    """Wilcoxon 检验：已知手算例"""
    from scripts.holm_wilcoxon import wilcoxon_p

    assert abs(wilcoxon_p([1, 1, 1, 1], [0, 0, 0, 0]) - 0.125) < 1e-9
