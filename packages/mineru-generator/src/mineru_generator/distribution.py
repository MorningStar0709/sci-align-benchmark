"""Dimension and difficulty distribution control for benchmark generation."""

CATEGORY_MAP = {
    "控制理论": "classical",
    "控制科学": "modern",
    "控制系统": "classical",
    "经典控制": "classical",
    "现代控制": "modern",
    "非线性控制": "nonlinear",
    "鲁棒控制": "robust",
    "最优控制": "optimal",
    "自适应控制": "adaptive",
    "数字控制": "digital",
    "智能控制": "intelligent",
    "模型预测控制": "mpc",
    "多智能体系统": "multi_agent",
}

DIMENSION_CYCLE = ["A", "B", "C", "D"]

DIFFICULTY_BY_DIMENSION = {
    "A": ["L1", "L2", "L3", "L4"],
    "B": ["L2", "L3", "L4", "L1"],
    "C": ["L2", "L3", "L4"],
    "D": ["L2", "L3", "L4"],
}

DIM_TARGET = {
    "A": 300, "B": 420, "C": 550, "D": 730,
}

DIFFICULTY_TARGET = {
    "A": {"L1": 100, "L2": 100, "L3": 80, "L4": 20},
    "B": {"L1": 40, "L2": 160, "L3": 150, "L4": 70},
    # C/D dimensions have no L1 (basic) questions — sensitivity analysis and
    # open-ended design are inherently intermediate-to-expert level tasks.
    "C": {"L2": 180, "L3": 230, "L4": 135},
    "D": {"L2": 100, "L3": 250, "L4": 380},
}



def normalize_categories(raw_categories):
    categories = []
    for raw in raw_categories or []:
        mapped = CATEGORY_MAP.get(raw, raw)
        if mapped not in categories:
            categories.append(mapped)
    return categories[:3] or ["classical"]


def pick_next_dim_and_diff(questions, limit):
    if not questions:
        return "A", "L1"
    counts = {"A": 0, "B": 0, "C": 0, "D": 0}
    for q in questions:
        dimension = q.get("dimension")
        if dimension in counts:
            counts[dimension] += 1
    total_target = sum(DIM_TARGET.values())
    target_ratio = {d: DIM_TARGET[d] / total_target for d in DIM_TARGET}
    current_ratio = {d: counts[d] / max(len(questions), 1) for d in counts}
    dim = max(DIM_TARGET, key=lambda d: target_ratio[d] - current_ratio[d])
    if DIM_TARGET[dim] > 0 and counts[dim] >= DIM_TARGET[dim]:
        remaining = [d for d in DIM_TARGET if counts[d] < DIM_TARGET[d]]
        if remaining:
            dim = max(remaining, key=lambda d: target_ratio[d] - current_ratio[d])

    diff_target = DIFFICULTY_TARGET[dim]
    diff_counts = {d: 0 for d in diff_target}
    for q in questions:
        if q.get("dimension") == dim:
            difficulty = q.get("difficulty_level")
            if difficulty in diff_counts:
                diff_counts[difficulty] += 1
    total_diff_target = sum(diff_target.values())
    diff_target_ratio = {d: diff_target[d] / total_diff_target for d in diff_target}
    diff_current_ratio = {d: diff_counts.get(d, 0) / max(sum(diff_counts.values()), 1) for d in diff_target}
    difficulty = max(diff_target, key=lambda d: diff_target_ratio.get(d, 0) - diff_current_ratio.get(d, 0))
    return dim, difficulty


# ---------------------------------------------------------------------------
# Offline dimension plan precomputation (for concurrent generation)
# ---------------------------------------------------------------------------


def precompute_dim_plan(total_limit: int) -> list[tuple[str, str]]:
    """Precompute a sequence of (dimension, difficulty) pairs for offline assignment.

    Uses local counters to simulate the serial pick_next_dim_and_diff behavior,
    eliminating the need for a shared mutable questions list during concurrent
    generation.  Returns a list of ``(dim, difficulty)`` tuples of length
    ``total_limit``.
    """
    plan = []
    dim_counts: dict[str, int] = {"A": 0, "B": 0, "C": 0, "D": 0}
    # Per-dimension difficulty counters: {dim: {L1: n, L2: n, ...}}
    diff_counts: dict[str, dict[str, int]] = {
        d: {lv: 0 for lv in DIFFICULTY_TARGET[d]} for d in DIM_TARGET
    }

    for _ in range(total_limit):
        dim, diff = _pick_next_dim_offline(dim_counts, diff_counts)
        plan.append((dim, diff))
        dim_counts[dim] += 1
        diff_counts[dim][diff] += 1
    return plan


def _pick_next_dim_offline(
    dim_counts: dict[str, int],
    diff_counts: dict[str, dict[str, int]],
) -> tuple[str, str]:
    """Select dimension and difficulty from offline counters only.

    Counterpart of ``pick_next_dim_and_diff`` that operates on local state
    instead of scanning a list of question dicts.  Produces identical results
    when fed with the same cumulative counts.
    """
    if sum(dim_counts.values()) == 0:
        return "A", "L1"

    total_target = sum(DIM_TARGET.values())
    target_ratio = {d: DIM_TARGET[d] / total_target for d in DIM_TARGET}
    current_total = max(sum(dim_counts.values()), 1)
    current_ratio = {d: dim_counts[d] / current_total for d in dim_counts}

    dim = max(DIM_TARGET, key=lambda d: target_ratio[d] - current_ratio[d])
    if DIM_TARGET[dim] > 0 and dim_counts[dim] >= DIM_TARGET[dim]:
        remaining = [d for d in DIM_TARGET if dim_counts[d] < DIM_TARGET[d]]
        if remaining:
            dim = max(remaining, key=lambda d: target_ratio[d] - current_ratio[d])

    diff_target = DIFFICULTY_TARGET[dim]
    total_diff_target = sum(diff_target.values())
    diff_target_ratio = {d: diff_target[d] / total_diff_target for d in diff_target}
    local_diff = diff_counts[dim]
    diff_total = max(sum(local_diff.values()), 1)
    diff_current_ratio = {d: local_diff[d] / diff_total for d in diff_target}
    difficulty = max(
        diff_target,
        key=lambda d: diff_target_ratio.get(d, 0) - diff_current_ratio.get(d, 0),
    )
    return dim, difficulty

