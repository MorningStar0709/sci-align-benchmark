"""GRPO reward functions for control science CoT training.

Provides three complementary reward functions for GRPOTrainer:
- accuracy_reward: answer correctness via keyword/formula matching
- format_reward: <think>...</think> format compliance (granular scoring)
- reasoning_depth_reward: reasoning chain quality (depth, formulas, logic)

Usage with GRPOTrainer::

    from mineru_eval.finetune.rewards import get_reward_funcs

    reward_funcs, reward_weights = get_reward_funcs()
    trainer = GRPOTrainer(
        reward_funcs=reward_funcs,
        args=GRPOConfig(reward_weights=reward_weights, ...),
        ...
    )
"""

from __future__ import annotations

import re
from typing import Callable


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LOGICAL_CONNECTORS_ZH = (
    "因为", "所以", "因此", "由于", "可得", "由此", "从而",
    "代入", "推导", "解得", "化简", "变换", "利用", "根据",
)

LOGICAL_CONNECTORS_EN = (
    "therefore", "thus", "hence", "because", "since",
    "substituting", "yields", "implies",
)

# LaTeX / math operators that indicate formula content.
# Note: bare "=" is excluded — it appears in nearly all math text and
# provides no discriminative signal.  Use structural indicators instead.
FORMULA_INDICATORS = (
    "\\frac", "\\sum", "\\int", "\\dot", "\\ddot",
    "\\partial", "\\infty", "\\leq", "\\geq", "\\approx",
    "\\alpha", "\\beta", "\\gamma", "\\delta", "\\lambda",
    "\\omega", "\\sigma", "\\theta", "\\epsilon",
    "\\Rightarrow", "\\Leftrightarrow",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_content(completion: list[dict[str, str]]) -> str:
    """Extract text content from a single completion."""
    if not completion:
        return ""
    return completion[0].get("content", "")


def _extract_final_answer(content: str) -> str:
    """Extract text after the closing </think> tag.

    Returns the stripped text after </think>, or the full content
    if no </think> tag is found.
    """
    match = re.search(r"</think>\s*(.*)", content, re.DOTALL)
    if match:
        return match.group(1).strip()
    return content.strip()


def _extract_think_content(content: str) -> str:
    """Extract text inside the first <think>...</think> block.

    Always extracts the *first* <think>...</think> block from the
    beginning of the content.  Returns empty string if tags are
    missing or malformed.
    """
    match = re.search(r"\s*<think>(.*?)</think>", content, re.DOTALL)
    return match.group(1) if match else ""


def _tokenize_answer(text: str) -> set[str]:
    """Split answer text into meaningful tokens for comparison.

    Produces a set of lowercase tokens after removing LaTeX commands,
    extra whitespace, and common punctuation.  This enables robust
    keyword-level matching for control science answers that contain
    a mix of Chinese text, math formulas, and variable names.
    """
    # Strip LaTeX commands: \frac, \sum, \dot{x} -> keep content
    cleaned = re.sub(r"\\[a-zA-Z]+", " ", text)
    # Remove remaining backslashes
    cleaned = cleaned.replace("\\", " ")
    # Remove common punctuation
    cleaned = re.sub(r"[，。；：、,.:;!?！？()（）\[\]{}]", " ", cleaned)
    # Lowercase and split
    return {t for t in cleaned.lower().split() if len(t) >= 2}


def _numerical_similarity(pred: str, sol: str, tolerance: float = 0.05) -> float | None:
    """Check if both texts contain the same numerical values (+-tolerance).

    Returns 1.0 if all numbers match within tolerance, 0.0 if any mismatch,
    or None if no numbers are found in the solution.
    """
    # Preserve original order — sorting would destroy variable-value
    # correspondence (e.g. "Kp=1, Ki=2" vs "Kp=2, Ki=1").
    num_pattern = re.compile(r"-?\d+\.?\d*")
    sol_nums = [float(m) for m in num_pattern.findall(sol)]
    pred_nums = [float(m) for m in num_pattern.findall(pred)]

    if not sol_nums:
        return None
    if len(sol_nums) != len(pred_nums):
        return 0.0

    for s, p in zip(sol_nums, pred_nums):
        if s == 0:
            if abs(p) > tolerance:
                return 0.0
        elif abs(s - p) / max(abs(s), 1e-9) > tolerance:
            return 0.0
    return 1.0


def _count_reasoning_steps(think_text: str) -> int:
    """Estimate the number of reasoning steps in think content.

    Uses a combination of explicit step markers and sentence boundaries
    to produce a robust step count estimate.
    """
    if not think_text.strip():
        return 0

    # Explicit numbered steps: "1.", "步骤1", "第一步", "第十一步"
    explicit = re.findall(
        r"(?:^|\n)\s*(?:\d+[.、)）]|步骤\s*\d+|第[一二三四五六七八九十百千\d]+\s*步)",
        think_text,
    )
    if len(explicit) >= 2:
        return len(explicit)

    # Sentence-level heuristic: split by Chinese/English sentence endings
    # that typically separate reasoning steps
    sentences = re.split(r"[。！？\n]+", think_text)
    # Filter out very short fragments (connectors, labels)
    meaningful = [s for s in sentences if len(s.strip()) >= 8]
    return len(meaningful)


# ---------------------------------------------------------------------------
# Reward Functions
# ---------------------------------------------------------------------------


def accuracy_reward(
    completions: list[list[dict[str, str]]],
    solution: list[str],
    **kwargs,
) -> list[float]:
    """Check answer correctness against reference solutions.

    Extracts the final answer (text after </think>) and compares
    it with the reference solution using a multi-strategy approach:

    1. Keyword overlap -- token-level Jaccard similarity
    2. Formula matching -- key mathematical symbol overlap
    3. Numerical tolerance -- +-5% for numeric answers

    Designed for control science problems where answers include
    formulas, numerical values, and concept descriptions.

    .. note::
        Returns 0.0 for empty solutions (previously ``None``).
        This change ensures GRPOTrainer compatibility, since
        ``None`` rewards cause training to abort.

    Args:
        completions: Model-generated completions from GRPOTrainer.
        solution: Reference answers from the dataset ``solution`` column.

    Returns:
        Scores in [0.0, 1.0]. Returns 0.0 if the solution is empty
        or the prediction cannot be extracted.

    Raises:
        ValueError: If *completions* and *solution* have different lengths.
    """
    if len(completions) != len(solution):
        raise ValueError(
            f"completions ({len(completions)}) and solution "
            f"({len(solution)}) length mismatch"
        )

    rewards: list[float] = []

    for completion, sol in zip(completions, solution):
        if not sol or not sol.strip():
            rewards.append(0.0)
            continue

        content = _extract_content(completion)
        pred = _extract_final_answer(content)

        if not pred:
            rewards.append(0.0)
            continue

        # Strategy 1: keyword token overlap (Jaccard-like)
        pred_tokens = _tokenize_answer(pred)
        sol_tokens = _tokenize_answer(sol)
        if sol_tokens:
            overlap = len(pred_tokens & sol_tokens) / len(sol_tokens)
        else:
            overlap = 0.0

        # Strategy 2: key formula indicator overlap
        pred_lower = pred.lower()
        sol_lower = sol.lower()
        sol_indicators = [ind for ind in FORMULA_INDICATORS if ind.lower() in sol_lower]
        if sol_indicators:
            formula_score = sum(
                1 for ind in sol_indicators if ind.lower() in pred_lower
            ) / len(sol_indicators)
        else:
            formula_score = 0.0

        # Strategy 3: numerical value matching (+-5%)
        num_score = _numerical_similarity(pred, sol)
        if num_score is None:
            num_score = 0.0  # no numbers to compare

        # Weighted combination -- keyword overlap is primary signal
        score = 0.5 * overlap + 0.2 * formula_score + 0.3 * num_score
        rewards.append(min(1.0, score))

    return rewards


def format_reward(
    completions: list[list[dict[str, str]]],
    **kwargs,
) -> list[float]:
    """Check <think>...</think> format compliance with granular scoring.

    Scoring breakdown (total 1.0):

    +-------+---------------------------------------------------+
    | Score | Condition                                         |
    +=======+===================================================+
    |  0.5  | Has opening <think> + closing </think>            |
    |  0.3  | Think content is non-trivial (>=50 characters)    |
    |  0.2  | Think content begins within first 20 characters   |
    +-------+---------------------------------------------------+

    This is intentionally more granular than trl's built-in
    ``think_format_reward`` (which is binary 0/1), providing a
    smoother learning signal for GRPO.

    Args:
        completions: Model-generated completions from GRPOTrainer.

    Returns:
        Scores in [0.0, 1.0] for each completion.
    """
    rewards: list[float] = []

    for completion in completions:
        content = _extract_content(completion)
        score = 0.0

        # 1. Tag completeness (0.5)
        has_open = "<think>" in content
        has_close = "</think>" in content
        if has_open and has_close:
            score += 0.5

            think_text = _extract_think_content(content)

            # 2. Non-trivial content -- at least 50 characters (0.3)
            if len(think_text.strip()) >= 50:
                score += 0.3

            # 3. Think before answer — <think> must be near the start (0.2)
            after_think = _extract_final_answer(content)
            think_start = content.find("<think>")
            text_before_think = content[:think_start].strip() if think_start > 0 else ""
            if (
                len(after_think.strip()) >= 10
                and think_start >= 0
                and len(text_before_think) < 20  # no substantial text before <think>
            ):
                score += 0.2

        rewards.append(score)

    return rewards


def reasoning_depth_reward(
    completions: list[list[dict[str, str]]],
    **kwargs,
) -> list[float]:
    """Evaluate reasoning chain depth and quality.

    Scoring breakdown (total 1.0):

    +-------+---------------------------------------------------+
    | Score | Condition                                         |
    +=======+===================================================+
    |  0.3  | Reasoning steps >= 3                              |
    |  0.3  | Contains formulas / equations                     |
    |  0.2  | Contains logical connectors                       |
    |  0.2  | No excessive repetition (dedup ratio > 0.8)      |
    +-------+---------------------------------------------------+

    Args:
        completions: Model-generated completions from GRPOTrainer.

    Returns:
        Scores in [0.0, 1.0] for each completion.
    """
    rewards: list[float] = []

    for completion in completions:
        content = _extract_content(completion)
        think_text = _extract_think_content(content)

        if not think_text.strip():
            rewards.append(0.0)
            continue

        score = 0.0

        # 1. Reasoning step count >= 3 (0.3)
        steps = _count_reasoning_steps(think_text)
        if steps >= 3:
            score += 0.3
        elif steps >= 2:
            score += 0.15

        # 2. Contains formulas / equations (0.3)
        think_lower = think_text.lower()
        n_indicators = sum(1 for ind in FORMULA_INDICATORS if ind.lower() in think_lower)
        if n_indicators >= 2:
            score += 0.3
        elif n_indicators >= 1:
            score += 0.15

        # 3. Logical connectors (0.2)
        has_connector = any(
            conn in think_text for conn in LOGICAL_CONNECTORS_ZH
        ) or any(
            conn in think_lower for conn in LOGICAL_CONNECTORS_EN
        )
        if has_connector:
            score += 0.2

        # 4. No excessive repetition (0.2)
        #    Split into ~20-char chunks; check uniqueness ratio
        chunk_size = 20
        chunks = [
            think_text[i : i + chunk_size]
            for i in range(0, max(1, len(think_text) - chunk_size + 1), chunk_size)
        ]
        if chunks:
            unique_ratio = len(set(chunks)) / len(chunks)
            if unique_ratio > 0.8:
                score += 0.2
            elif unique_ratio > 0.6:
                score += 0.1

        rewards.append(min(1.0, score))

    return rewards


# ---------------------------------------------------------------------------
# Convenience aggregator
# ---------------------------------------------------------------------------


def get_reward_funcs(
    weights: tuple[float, float, float] = (1.0, 0.1, 0.3),
) -> tuple[list[Callable], list[float]]:
    """Return the default reward functions and weights for v3 GRPO training.

    Default weights: accuracy=1.0, format=0.1, reasoning_depth=0.3.

    Args:
        weights: (accuracy_w, format_w, depth_w) override.

    Returns:
        (reward_funcs, reward_weights) ready for GRPOTrainer.
    """
    funcs: list[Callable] = [accuracy_reward, format_reward, reasoning_depth_reward]
    return funcs, list(weights)


def accuracy_reward_c(
    completions: list[list[dict[str, str]]],
    solution: list[str],
    **kwargs,
) -> list[float]:
    """C‑dimension accuracy with relaxed numerical matching.

    Unlike :func:`accuracy_reward`, uses :func:`_number_coverage`
    (fraction of solution numbers present in answer) instead of
    strict count‑and‑order matching.  This avoids the zero‑signal
    problem where models produce answers with different numerical
    counts than the reference.

    Scoring breakdown:
    - 0.5 × keyword token overlap (same as accuracy_reward)
    - 0.2 × formula indicator overlap (same as accuracy_reward)
    - 0.3 × number coverage (relaxed: fraction‑based, not strict)
    """
    if len(completions) != len(solution):
        raise ValueError(
            f"completions ({len(completions)}) and solution "
            f"({len(solution)}) length mismatch"
        )

    rewards: list[float] = []

    for completion, sol in zip(completions, solution):
        if not sol or not sol.strip():
            rewards.append(0.0)
            continue

        content = _extract_content(completion)
        pred = _extract_final_answer(content)

        if not pred:
            rewards.append(0.0)
            continue

        # Strategy 1: keyword token overlap (Jaccard‑like)
        pred_tokens = _tokenize_answer(pred)
        sol_tokens = _tokenize_answer(sol)
        if sol_tokens:
            overlap = len(pred_tokens & sol_tokens) / len(sol_tokens)
        else:
            overlap = 0.0

        # Strategy 2: key formula indicator overlap
        pred_lower = pred.lower()
        sol_lower = sol.lower()
        sol_indicators = [ind for ind in FORMULA_INDICATORS if ind.lower() in sol_lower]
        if sol_indicators:
            formula_score = sum(
                1 for ind in sol_indicators if ind.lower() in pred_lower
            ) / len(sol_indicators)
        else:
            formula_score = 0.0

        # Strategy 3: relaxed numerical coverage (fraction‑based, not strict)
        num_score = _number_coverage(pred, sol)

        # Weighted combination — keyword overlap is primary signal
        score = 0.5 * overlap + 0.2 * formula_score + 0.3 * num_score
        rewards.append(min(1.0, score))

    return rewards


# ---------------------------------------------------------------------------
# C‑dimension constraint‑following reward (Phase 4 RLVR)
# ---------------------------------------------------------------------------

# Pattern: numbered analysis points like "1.", "2.", "3.", "1.", "2."
NUMBERED_POINT_PATTERN = re.compile(r"(?:^|\n)\s*\d+[.、)）]\s*(?:\*{0,2})")

COMPARATIVE_MARKERS = (
    "对比", "比较", "相比之下", "相比较",
    "增大", "减小", "增加", "减少", "上升", "下降",
    "降低", "提高", "增强", "减弱", "扩大", "缩小",
    "原系统", "原模型", "原参数", "基准",
    "变化", "改变", "变更", "新模型", "新参数",
)

CAUSAL_MARKERS = (
    "导致", "使得", "从而", "意味着",
    "因此", "所以", "由于",
)


def _extract_key_numbers(text: str) -> list[float]:
    """Extract non‑zero numeric values from *text*, preserving order."""
    return [
        float(m)
        for m in re.findall(r"-?\d+\.?\d*(?:[eE][+-]?\d+)?", text)
        if abs(float(m)) > 1e-9
    ]


def _number_coverage(answer: str, solution: str, tolerance: float = 0.05) -> float:
    """Fraction of key numbers in *solution* also present in *answer* (±5 %)."""
    sol_nums = _extract_key_numbers(solution)
    if not sol_nums:
        return 0.0

    ans_nums = _extract_key_numbers(answer)
    if not ans_nums:
        return 0.0

    matched = 0
    for sn in sol_nums:
        for an in ans_nums:
            if abs(sn - an) / max(abs(sn), 1e-9) <= tolerance:
                matched += 1
                break
    return matched / len(sol_nums)


def constraint_reward(
    completions: list[list[dict[str, str]]],
    solution: list[str],
    **kwargs,
) -> list[float]:
    """Score C‑dimension constraint‑following for parameter‑change questions.

    Designed for C‑dimension questions that ask: "if parameter *X* changes
    from *Y* to *Z*, what happens?"  Based on analysis of model answers
    vs. actual judge scores, the reward uses three empirical signals:

    +-------+---------------------------------------------------------+
    | Score | Aspect                                                  |
    +=======+=========================================================+
    |  0.35 | **Structured analysis** — numbered points (1., 2., 3.), |
    |       | not just narrative flow.                                |
    |  0.35 | **Technical depth** — formula indicators + parameter    |
    |       | values from the question appear in the answer.          |
    |  0.30 | **Comprehensiveness** — adequate length (≥300 chars),   |
    |       | comparative markers, and causal connectors.             |
    +-------+---------------------------------------------------------+

    Args:
        completions: Model‑generated completions from GRPOTrainer.
        solution: Reference answers (used to extract key parameters).

    Returns:
        Scores in [0.0, 1.0] for each completion.
    """
    rewards: list[float] = []

    if len(completions) != len(solution):
        raise ValueError(
            f"completions ({len(completions)}) and solution "
            f"({len(solution)}) length mismatch"
        )

    for completion, sol in zip(completions, solution):
        content = _extract_content(completion)
        answer = _extract_final_answer(content)
        think = _extract_think_content(content)
        combined = think + " " + answer

        if not combined.strip():
            rewards.append(0.0)
            continue

        combined_lower = combined.lower()
        score = 0.0

        # 1. Structured analysis (0.35): numbered points
        points = NUMBERED_POINT_PATTERN.findall(combined)
        n_points = len(points)
        if n_points >= 3:
            score += 0.35
        elif n_points == 2:
            score += 0.25
        elif n_points == 1:
            score += 0.15

        # 2. Technical depth (0.35): formulas + parameter coverage
        tech_score = 0.0

        # 2a. Formula indicators
        formula_count = sum(1 for ind in FORMULA_INDICATORS if ind.lower() in combined_lower)
        if formula_count >= 2:
            tech_score += 0.15
        elif formula_count == 1:
            tech_score += 0.08

        # 2b. Parameter coverage (numbers from solution)
        if sol:
            tech_score += 0.20 * _number_coverage(combined, sol)

        score += min(0.35, tech_score)

        # 3. Comprehensiveness (0.30)
        comp_score = 0.0

        # 3a. Adequate length
        if len(combined) >= 300:
            comp_score += 0.10
        elif len(combined) >= 200:
            comp_score += 0.05

        # 3b. Comparative markers
        if any(m in combined for m in COMPARATIVE_MARKERS):
            comp_score += 0.10

        # 3c. Causal / logical connectors
        if any(m in combined for m in CAUSAL_MARKERS) or \
           any(c in combined for c in LOGICAL_CONNECTORS_ZH) or \
           any(c in combined_lower for c in LOGICAL_CONNECTORS_EN):
            comp_score += 0.10

        score += min(0.30, comp_score)
        rewards.append(min(1.0, score))

    return rewards


def get_reward_funcs_c(
    weights: tuple[float, float, float, float] = (1.0, 0.1, 0.3, 0.5),
) -> tuple[list[Callable], list[float]]:
    """Return reward functions for C‑dimension RLVR training.

    Adds :func:`constraint_reward` on top of the default three.  Default
    weights reflect that the main learning signal comes from the constraint
    reward, with accuracy as a secondary anchor.

    Args:
        weights: (accuracy, format, depth, constraint) override.

    Returns:
        (reward_funcs, reward_weights) ready for GRPOTrainer.
    """
    funcs: list[Callable] = [
        accuracy_reward,
        format_reward,
        reasoning_depth_reward,
        constraint_reward,
    ]
    return funcs, list(weights)


def get_reward_funcs_c_v2(
    weights: tuple[float, float] = (0.5, 1.0),
) -> tuple[list[Callable], list[float]]:
    """Return reward functions for C‑dimension RLVR v2 training.

    Two‑component architecture:
    - accuracy_reward_c (0.5): relaxed numerical matching
    - constraint_reward  (1.0): structured analysis + technical depth

    Removes format_reward and reasoning_depth_reward which produce
    zero signal when the model doesn't generate <think> tags.

    Args:
        weights: (accuracy_c, constraint) override.

    Returns:
        (reward_funcs, reward_weights) ready for GRPOTrainer.
    """
    funcs: list[Callable] = [accuracy_reward_c, constraint_reward]
    return funcs, list(weights)


# ---------------------------------------------------------------------------
# Medical domain reward (RLVR-C v2 medical)
# ---------------------------------------------------------------------------

# Medical + control-science cross-domain key terms for coverage scoring.
MEDICAL_KEY_TERMS_ZH = (
    "诊断", "治疗", "病理", "药理", "剂量", "禁忌", "适应证",
    "不良反应", "代谢", "药代动力学", "受体", "靶点", "信号通路",
    "反馈", "稳态", "调节", "抑制", "激活", "表达", "突变",
    "闭环", "开环", "控制器", "被控对象", "传递函数", "状态空间",
    "观测器", "鲁棒", "最优", "自适应", "PID", "增益",
)

MEDICAL_KEY_TERMS_EN = (
    "diagnosis", "treatment", "pathology", "pharmacology", "dosage",
    "contraindication", "indication", "adverse", "metabolism",
    "pharmacokinetics", "receptor", "target", "signaling", "feedback",
    "homeostasis", "regulation", "inhibition", "activation", "expression",
    "mutation", "closed-loop", "open-loop", "controller", "plant",
    "transfer function", "state-space", "observer", "robust", "optimal",
    "adaptive", "pid", "gain",
)


def _medical_term_coverage(answer: str, solution: str) -> float:
    """Compute medical/control term coverage of *solution* in *answer*.

    Extracts key terms present in the reference solution and checks
    what fraction also appears in the model answer.  Returns 0.0 if
    the solution contains no recognised key terms.
    """
    sol_lower = solution.lower()
    ans_lower = answer.lower()

    # Collect terms that appear in the reference solution
    relevant_terms = [
        t for t in MEDICAL_KEY_TERMS_ZH if t in sol_lower
    ] + [
        t for t in MEDICAL_KEY_TERMS_EN if t in sol_lower
    ]

    if not relevant_terms:
        return 0.0

    matched = sum(1 for t in relevant_terms if t in ans_lower)
    return matched / len(relevant_terms)


def accuracy_reward_medical(
    completions: list[list[dict[str, str]]],
    solution: list[str],
    **kwargs,
) -> list[float]:
    """Medical-domain accuracy reward via keyword coverage + structure bonus.

    Scoring breakdown:
    - 0.5 × token overlap (same as accuracy_reward)
    - 0.3 × medical/control term coverage
    - 0.2 × structure bonus (numbered points or headings in final answer)

    Designed for open-ended medical QA where exact numerical matching
    is less relevant than conceptual coverage.

    Args:
        completions: Model-generated completions from GRPOTrainer.
        solution: Reference answers from the dataset ``solution`` column.

    Returns:
        Scores in [0.0, 1.0].
    """
    if len(completions) != len(solution):
        raise ValueError(
            f"completions ({len(completions)}) and solution "
            f"({len(solution)}) length mismatch"
        )

    rewards: list[float] = []

    for completion, sol in zip(completions, solution):
        if not sol or not sol.strip():
            rewards.append(0.0)
            continue

        content = _extract_content(completion)
        pred = _extract_final_answer(content)

        if not pred:
            rewards.append(0.0)
            continue

        # Strategy 1: token overlap (Jaccard-like, same as control-sci)
        pred_tokens = _tokenize_answer(pred)
        sol_tokens = _tokenize_answer(sol)
        if sol_tokens:
            overlap = len(pred_tokens & sol_tokens) / len(sol_tokens)
        else:
            overlap = 0.0

        # Strategy 2: medical/control term coverage
        term_score = _medical_term_coverage(pred, sol)

        # Strategy 3: structure bonus (numbered points / headings)
        struct_score = 0.0
        points = NUMBERED_POINT_PATTERN.findall(pred)
        if len(points) >= 3:
            struct_score = 1.0
        elif len(points) == 2:
            struct_score = 0.7
        elif len(points) == 1:
            struct_score = 0.4
        # Markdown headings also count
        headings = re.findall(r"(?:^|\n)#{1,4}\s+\S", pred)
        if len(headings) >= 2 and struct_score < 1.0:
            struct_score = max(struct_score, 0.8)

        score = 0.5 * overlap + 0.3 * term_score + 0.2 * struct_score
        rewards.append(min(1.0, score))

    return rewards


def get_reward_funcs_medical(
    weights: tuple[float, float] = (1.0, 0.2),
) -> tuple[list[Callable], list[float]]:
    """Return reward functions for medical-domain RLVR-C v2 training.

    Two-component architecture:
    - accuracy_reward_medical (1.0): keyword + term coverage + structure
    - format_reward           (0.2): <think>...</think> compliance

    Args:
        weights: (accuracy_medical, format) override.

    Returns:
        (reward_funcs, reward_weights) ready for GRPOTrainer.
    """
    funcs: list[Callable] = [accuracy_reward_medical, format_reward]
    return funcs, list(weights)
