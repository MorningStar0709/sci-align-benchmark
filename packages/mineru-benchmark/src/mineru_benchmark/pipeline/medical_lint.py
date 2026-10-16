"""Medical cross-domain 3-dimension x 4-difficulty lint rules.

Contains identify/model/transfer dimension quality checks for medical domain
benchmark questions, with difficulty-aware thresholds.
"""
from __future__ import annotations

import re
from typing import Any


# ---------------------------------------------------------------------------
# Medical cross-domain concept/element dictionaries
# ---------------------------------------------------------------------------

# identify: control theory concept names (Chinese + English)
_MEDICAL_CONTROL_CONCEPTS = {
    '反馈', 'feedback', '稳定性', 'stability', '最优控制', 'optimal control',
    '自适应', 'adaptive', '鲁棒', 'robust', 'PID', '模型预测', 'MPC',
    '状态空间', 'state space', '传递函数', 'transfer function', '控制器',
    'controller', '闭环', 'closed-loop', '开环', 'open-loop', '增益', 'gain',
    '观测器', 'observer', '镇定', 'stabiliz', '收敛', 'convergence',
    '动态', 'dynamics', '平衡', 'equilibrium', '调节', 'regulation',
}

# model: control model elements
_MEDICAL_MODEL_ELEMENTS = {
    '状态变量', 'state variable', '控制输入', 'control input', '观测输出',
    'observation output', '输出变量', 'output variable', '输入变量',
    'input variable', '状态方程', '状态空间方程', '数学模型', '模型',
    '传递函数', 'transfer function', '框图', 'block diagram',
}

# transfer: predictive statement markers
_MEDICAL_PREDICT_RE = re.compile(
    r'(预期|预计|将会|会导致|可能|有望|能够|可以|改善|提高|降低|减少|'
    r'expected|predicted|likely|would|could|improve|reduce|increase)',
    re.IGNORECASE,
)

# transfer: quantitative markers
_MEDICAL_QUANT_RE = re.compile(
    r'(\d+\s*%|增加|减少|上升|下降|超调|稳态误差|时间常数|截止频率|'
    r'\d+\s*(?:ms|s|min|h|°|rad)|增益.*\d|K\s*[=<>]|\d+\s*倍)',
    re.IGNORECASE,
)

# transfer: constraint/clinical discussion markers
_MEDICAL_CONSTRAINT_RE = re.compile(
    r'(约束|限制|挑战|困难|实际|临床|安全|可靠|可行|实施|部署|'
    r'constraint|challenge|clinical|safety|feasib|implement|deploy)',
    re.IGNORECASE,
)

# identify: multi-concept / cross-branch markers
_MEDICAL_CROSS_BRANCH_RE = re.compile(
    r'(此外|同时|另外|并且|而且|以及|对应|映射|关联|结合|'
    r'also|additionally|furthermore|moreover|correspond|map|relate)',
    re.IGNORECASE,
)

# model: mathematical expression markers
_MEDICAL_MATH_RE = re.compile(
    r'([=<>≤≥]|方程|公式|传递函数|状态空间|微分|积分|拉普拉斯|'
    r'equation|formula|transfer function|state space|differential|integral|laplace)',
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Difficulty-aware thresholds
# ---------------------------------------------------------------------------

# identify: minimum concept count by difficulty
_IDENTIFY_MIN_CONCEPTS = {"L1": 1, "L2": 1, "L3": 2, "L4": 2}
# identify: whether cross-branch mapping is required
_IDENTIFY_NEEDS_CROSS_BRANCH = {"L1": False, "L2": False, "L3": True, "L4": True}

# model: minimum model element count by difficulty
_MODEL_MIN_ELEMENTS = {"L1": 1, "L2": 2, "L3": 3, "L4": 3}
# model: whether math expression is required
_MODEL_NEEDS_MATH = {"L1": False, "L2": False, "L3": True, "L4": True}
# model: whether multi-scheme comparison is required
_MODEL_NEEDS_COMPARISON = {"L1": False, "L2": False, "L3": False, "L4": True}

# transfer: minimum answer length by difficulty
_TRANSFER_MIN_LENGTH = {"L1": 100, "L2": 150, "L3": 200, "L4": 250}
# transfer: whether quantitative analysis is required
_TRANSFER_NEEDS_QUANT = {"L1": False, "L2": False, "L3": True, "L4": True}
# transfer: whether constraint/clinical discussion is required
_TRANSFER_NEEDS_CONSTRAINT = {"L1": False, "L2": False, "L3": False, "L4": True}


# ---------------------------------------------------------------------------
# Lint functions (dimension + difficulty aware)
# ---------------------------------------------------------------------------


def _lint_medical_identify(answer: str, difficulty: str) -> tuple[bool, list[str]]:
    """identify: answer must contain control theory concepts, threshold by difficulty."""
    issues = []
    answer_lower = answer.lower()
    found_concepts = [c for c in _MEDICAL_CONTROL_CONCEPTS if c.lower() in answer_lower]
    min_concepts = _IDENTIFY_MIN_CONCEPTS.get(difficulty, 1)

    if len(found_concepts) < min_concepts:
        issues.append(
            f"identify({difficulty})答案控制概念不足（检测到 {len(found_concepts)}/{min_concepts} 个）"
        )
        return False, issues

    # L3/L4: check for cross-branch mapping markers
    if _IDENTIFY_NEEDS_CROSS_BRANCH.get(difficulty, False):
        if not _MEDICAL_CROSS_BRANCH_RE.search(answer):
            issues.append(
                f"identify({difficulty})答案缺少跨分支映射标记（未检测到此外/同时/对应/映射等词）"
            )
            return False, issues

    return True, []


def _lint_medical_model(answer: str, difficulty: str) -> tuple[bool, list[str]]:
    """model: answer must contain model elements, threshold by difficulty."""
    issues = []
    answer_lower = answer.lower()
    found_elements = [e for e in _MEDICAL_MODEL_ELEMENTS if e.lower() in answer_lower]
    min_elements = _MODEL_MIN_ELEMENTS.get(difficulty, 2)

    if len(found_elements) < min_elements:
        issues.append(
            f"model({difficulty})答案控制要素不足（检测到 {len(found_elements)}/{min_elements} 个，"
            f"需要包含状态变量/控制输入/观测输出等至少 {min_elements} 个要素）"
        )
        return False, issues

    # L3/L4: check for mathematical expressions
    if _MODEL_NEEDS_MATH.get(difficulty, False):
        if not _MEDICAL_MATH_RE.search(answer):
            issues.append(
                f"model({difficulty})答案缺少数学表达（未检测到方程/公式/传递函数/状态空间等）"
            )
            return False, issues

    # L4: check for multi-scheme comparison markers
    if _MODEL_NEEDS_COMPARISON.get(difficulty, False):
        comparison_markers = re.compile(
            r'(对比|比较|方案.*[二2两]|替代|优于|劣于|选择.*理由|'
            r'compar|versus|alternative|superior|choose|scheme)',
            re.IGNORECASE,
        )
        if not comparison_markers.search(answer):
            issues.append(
                f"model({difficulty})答案缺少多方案对比标记（未检测到对比/比较/方案/替代等词）"
            )
            return False, issues

    return True, []


def _lint_medical_transfer(answer: str, difficulty: str) -> tuple[bool, list[str]]:
    """transfer: answer length + predictive statements, threshold by difficulty."""
    issues = []
    min_length = _TRANSFER_MIN_LENGTH.get(difficulty, 150)

    if len(answer) < min_length:
        issues.append(
            f"transfer({difficulty})答案过短：{len(answer)}/{min_length} 字符（需要 >={min_length}）"
        )
        return False, issues

    has_predict = bool(_MEDICAL_PREDICT_RE.search(answer))
    if not has_predict:
        issues.append(
            f"transfer({difficulty})答案缺少预测性表述（未找到预期/预计/将会/可能等预测标记）"
        )
        return False, issues

    # L3/L4: check for quantitative markers
    if _TRANSFER_NEEDS_QUANT.get(difficulty, False):
        if not _MEDICAL_QUANT_RE.search(answer):
            issues.append(
                f"transfer({difficulty})答案缺少量化分析标记（未检测到百分比/数值/增益等量化表述）"
            )
            return False, issues

    # L4: check for constraint/clinical discussion
    if _TRANSFER_NEEDS_CONSTRAINT.get(difficulty, False):
        if not _MEDICAL_CONSTRAINT_RE.search(answer):
            issues.append(
                f"transfer({difficulty})答案缺少约束/临床讨论（未检测到约束/限制/临床/安全等词）"
            )
            return False, issues

    return True, []


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


_LINTERS = {
    "identify": _lint_medical_identify,
    "model": _lint_medical_model,
    "transfer": _lint_medical_transfer,
}


def lint_medical_question(question: dict[str, Any]) -> dict[str, Any]:
    """Run dimension+difficulty-specific lint on a single medical question.

    Returns:
        {"pass": bool, "issues": [str, ...], "dimension": str, "difficulty": str}
    """
    dim = question.get("dimension", "") or question.get("medical_dimension", "")
    diff = question.get("difficulty_level", "L2")

    if dim not in _LINTERS:
        return {"pass": True, "issues": [], "dimension": dim, "difficulty": diff}

    answer = question.get("answer") or ""
    passed, issues = _LINTERS[dim](answer, diff)
    return {"pass": passed, "issues": issues, "dimension": dim, "difficulty": diff}


def lint_medical_batch(questions: list[dict[str, Any]]) -> dict[str, Any]:
    """Run lint on all medical questions, return summary + flagged indices."""
    results = []
    passed = 0
    failed = 0
    flagged_indices = []

    for i, q in enumerate(questions):
        result = lint_medical_question(q)
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
