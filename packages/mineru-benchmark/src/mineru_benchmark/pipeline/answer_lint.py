"""Answer Lint rules for 4-dimension benchmark quality checks.

Implements §四 of sci-align-generation-spec.md — heuristic checks that run
before audit. Non-passing questions are flagged as needs_review.

Usage:
    from mineru_benchmark.pipeline.answer_lint import lint_question
    result = lint_question(question)
    if not result["pass"]:
        print(f"Lint failed: {result['issues']}")
"""
from __future__ import annotations

import re
from typing import Any

# ---------------------------------------------------------------------------
# Dimension-specific lint rules
# ---------------------------------------------------------------------------

# A-dim: at least one of three signals
_A_LATEX_RE = re.compile(
    r'\\frac|\\sum|\\dot|\\int|\\partial|\\infty|\\sqrt|'
    r'G\s*\(\s*s\s*\)|[=＝].*[a-zA-Z]|[a-zA-Z].*[=＝]',
)

_A_CONDITION_RE = re.compile(
    r'(如果|当|若|假如|假设|设).*(则|那么|则|便|即)|'
    r'(if|when|suppose|assuming|given|let).*(then|thus)',
    re.IGNORECASE,
)

_A_TERMS = {
    '稳定性', '极点', '传递函数', '状态空间', '控制器', '反馈',
    '前馈', '观测器', '鲁棒', '稳定', '镇定', '镇定器',
    '控制律', '控制率', '控制输入', '闭环', '开环', '根轨迹',
    '频率响应', '增益', '相位', '裕度', '带宽', '阻尼',
    '自然频率', '特征值', '特征根', 'Lyapunov', '李雅普诺夫',
    'Routh', 'Hurwitz', 'Nyquist', 'Bode', 'stability',
    'pole', 'eigenvalue', 'feedback', 'observer', 'robust',
    'stabiliz', 'controller', 'transfer function', 'state space',
}

# B-dim: derivation markers
_B_DERIVATION_RE = re.compile(
    r'(解得|得到|因此|故|所以|从而|于是|推导|计算得|求得|'
    r'由此|可见|显然|即|意味着)',
)

# C-dim: multi-direction comparison
_C_INCREASE_RE = re.compile(
    r'(升[高大了]|增[大加]|提高|变大|变快|改善|优化|增强)',
)
_C_DECREASE_RE = re.compile(
    r'(降[低小]|减[小少]|变小|变慢|恶化|退化|削弱)',
)
_C_NUM_COMPARE_RE = re.compile(
    r'(\d+(?:\.\d+)?)\s*(?:[到至~\-—])\s*(\d+(?:\.\d+)?)|'
    r'(?:从|由)\s*(\d+(?:\.\d+)?)\s*(?:变[为到]|增[加至]|减[少至])',
)

# D-dim: design elements
_D_DESIGN_ELEMENTS = {
    '建模': 'modeling',
    '模型': 'modeling',
    '传递函数': 'modeling',
    '状态空间': 'modeling',
    '选型': 'selection',
    '选择': 'selection',
    '采用': 'selection',
    '控制器': 'selection',
    '参数': 'tuning',
    '整定': 'tuning',
    '调节': 'tuning',
    '配置': 'tuning',
    '验证': 'validation',
    '仿真': 'validation',
    '性能': 'validation',
    '指标': 'validation',
    '工程': 'engineering',
    '实施': 'engineering',
    '约束': 'engineering',
    '实际': 'engineering',
}


def _lint_a(answer: str, reasoning_steps: list[str]) -> tuple[bool, list[str]]:
    """A-dim: must satisfy at least 1 of 3 quality signals."""
    issues = []
    has_latex = bool(_A_LATEX_RE.search(answer))
    has_condition = bool(_A_CONDITION_RE.search(answer))

    term_count = 0
    answer_lower = answer.lower()
    for term in _A_TERMS:
        if term.lower() in answer_lower:
            term_count += 1
            if term_count >= 3:
                break
    has_terms = term_count >= 3

    if has_latex or has_condition or has_terms:
        return True, []

    issues.append("A维答案过于模糊：缺少 LaTeX 公式、条件-结论结构、或 ≥3 个控制科学术语")
    if not has_latex:
        issues.append("  - 未检测到 LaTeX 公式标记")
    if not has_condition:
        issues.append("  - 未检测到条件-结论结构")
    if not has_terms:
        issues.append(f"  - 控制科学术语不足（仅 {term_count}/3）")
    return False, issues


def _lint_b(answer: str, reasoning_steps: list[str]) -> tuple[bool, list[str]]:
    """B-dim: reasoning_steps >= 3 AND answer has derivation result markers."""
    issues = []
    step_count = len([s for s in reasoning_steps if isinstance(s, str) and s.strip()])

    if step_count < 3:
        issues.append(f"B维推理步骤不足：{step_count}/3（需要 ≥3 步）")
        return False, issues

    has_derivation = bool(_B_DERIVATION_RE.search(answer))
    if not has_derivation:
        issues.append("B维答案缺少推导结论标记（未找到\"解得\"\"得到\"\"因此\"等结果标记）")
        return False, issues

    return True, []


def _lint_c(answer: str) -> tuple[bool, list[str]]:
    """C-dim: >= 2 parameter change directions AND >= 1 numerical comparison."""
    issues = []
    has_increase = bool(_C_INCREASE_RE.search(answer))
    has_decrease = bool(_C_DECREASE_RE.search(answer))

    if not has_increase and not has_decrease:
        issues.append("C维缺少参数变化方向分析（未找到升高/降低描述）")
        return False, issues

    if not (has_increase and has_decrease):
        direction = "升高" if has_increase else "降低"
        issues.append(f"C维只有单一方向分析（仅有\"{direction}\"方向，缺少多方向对比）")
        return False, issues

    has_num_compare = bool(_C_NUM_COMPARE_RE.search(answer))
    if not has_num_compare:
        issues.append("C维缺少数值量化对比（如\"从 2.5 变为 3.8\"）")
        return False, issues

    return True, []


def _lint_d(answer: str) -> tuple[bool, list[str]]:
    """D-dim: answer >= 200 chars AND >= 2 design element categories."""
    issues = []
    if len(answer) < 200:
        issues.append(f"D维答案过短：{len(answer)}/200 字符（需要 ≥200）")
        return False, issues

    categories = set()
    answer_lower = answer.lower()
    for keyword, category in _D_DESIGN_ELEMENTS.items():
        if keyword.lower() in answer_lower:
            categories.add(category)

    if len(categories) < 2:
        found = ', '.join(sorted(categories)) if categories else '无'
        issues.append(f"D维缺少关键设计要素（已覆盖: {found}，需要 ≥2 个类别）")
        return False, issues

    return True, []



# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


_LINTERS = {
    "A": _lint_a,
    "B": _lint_b,
    "C": _lint_c,
    "D": _lint_d,
}


def lint_question(question: dict[str, Any]) -> dict[str, Any]:
    """Run dimension-specific lint on a single control science question.

    Returns:
        {"pass": bool, "issues": [str, ...], "dimension": str}
    """
    dim = question.get("dimension", "")

    if dim not in _LINTERS:
        return {"pass": True, "issues": [], "dimension": dim}

    answer = question.get("answer") or ""
    reasoning_steps = question.get("reasoning_steps") or []

    if dim in ("A", "B"):
        passed, issues = _LINTERS[dim](answer, reasoning_steps)
    else:
        passed, issues = _LINTERS[dim](answer)

    return {"pass": passed, "issues": issues, "dimension": dim}


def lint_batch(questions: list[dict[str, Any]]) -> dict[str, Any]:
    """Run lint on all questions, return summary + flagged indices."""
    results = []
    passed = 0
    failed = 0
    flagged_indices = []

    for i, q in enumerate(questions):
        result = lint_question(q)
        results.append(result)
        if result["pass"]:
            passed += 1
        else:
            failed += 1
            flagged_indices.append(i)

    return {
        "total": len(questions),
        "passed": passed,
        "failed": failed,
        "pass_rate": round(passed / max(len(questions), 1) * 100, 1),
        "flagged_indices": flagged_indices,
        "details": results,
    }
