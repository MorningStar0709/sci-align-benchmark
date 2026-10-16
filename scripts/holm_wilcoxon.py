"""Wilcoxon + Holm 统计量（只用 scipy，禁止 statsmodels）"""

from scipy.stats import wilcoxon


def holm_adjust(p_values: list[float], alpha: float = 0.05) -> list[bool]:
    """Holm-Bonferroni 校正

    规则：
    1. 将 p 值从小到大排序
    2. 第 i 个 p 值的校正阈值为 alpha / (n - i)
    3. 如果 p_i <= alpha / (n - i)，则拒绝原假设
    4. 一旦遇到不拒绝的，后续全部不拒绝
    """
    n = len(p_values)
    sorted_p = sorted(enumerate(p_values), key=lambda x: x[1])

    rejected = [False] * n
    for rank, (idx, p) in enumerate(sorted_p):
        threshold = alpha / (n - rank)
        if p <= threshold:
            rejected[idx] = True
        else:
            break

    return rejected


def wilcoxon_p(x: list[float], y: list[float]) -> float:
    """Wilcoxon 符号秩检验的 p 值"""
    stat, p = wilcoxon(x, y, alternative="two-sided")
    return p
