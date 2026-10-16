"""Medical cross-domain 3-dimension x 4-difficulty distribution control.

Provides dimension (identify/model/transfer) and difficulty (L1-L4) joint
allocation, mirroring the control science domain's non-uniform distribution
ratios for controlled cross-domain comparison experiments.

Total target: 720 questions (3 dims x 4 difficulties = 12 cells).
"""

# ---------------------------------------------------------------------------
# Target distributions
# ---------------------------------------------------------------------------

MEDICAL_DIMENSION_CYCLE = ["identify", "model", "transfer"]

# Dimension targets (equal split: 240 each)
MEDICAL_DIM_TARGET = {
    "identify": 240,
    "model": 240,
    "transfer": 240,
}

# Difficulty targets (mirrors control domain ratios: L1~14%, L2~35%, L3~35%, L4~16%)
MEDICAL_DIFFICULTY_TARGET = {
    "L1": 100,
    "L2": 250,
    "L3": 250,
    "L4": 120,
}

# All medical dimensions support all difficulty levels
MEDICAL_DIFFICULTY_BY_DIMENSION = {
    "identify": ["L1", "L2", "L3", "L4"],
    "model":    ["L1", "L2", "L3", "L4"],
    "transfer": ["L1", "L2", "L3", "L4"],
}


# ---------------------------------------------------------------------------
# Online picker (serial generation)
# ---------------------------------------------------------------------------


def pick_next_medical_dim_and_diff(questions: list[dict], limit: int) -> tuple[str, str]:
    """Pick next (dimension, difficulty) based on target distribution.

    Mirrors distribution.py.pick_next_dim_and_diff: first selects dimension
    by ratio deficit, then selects difficulty within that dimension.

    Args:
        questions: List of already-generated questions.
        limit: Total target question count.

    Returns:
        (dimension, difficulty) tuple.
    """
    if not questions:
        return "identify", "L1"

    # --- Step 1: pick dimension ---
    counts = {"identify": 0, "model": 0, "transfer": 0}
    for q in questions:
        dim = q.get("dimension") or q.get("medical_dimension")
        if dim in counts:
            counts[dim] += 1

    total_target = sum(MEDICAL_DIM_TARGET.values())
    target_ratio = {d: MEDICAL_DIM_TARGET[d] / total_target for d in MEDICAL_DIM_TARGET}
    current_ratio = {d: counts[d] / max(len(questions), 1) for d in counts}

    dim = max(MEDICAL_DIM_TARGET, key=lambda d: target_ratio[d] - current_ratio[d])
    if MEDICAL_DIM_TARGET[dim] > 0 and counts[dim] >= MEDICAL_DIM_TARGET[dim]:
        remaining = [d for d in MEDICAL_DIM_TARGET if counts[d] < MEDICAL_DIM_TARGET[d]]
        if remaining:
            dim = max(remaining, key=lambda d: target_ratio[d] - current_ratio[d])

    # --- Step 2: pick difficulty within dimension ---
    diff_target = MEDICAL_DIFFICULTY_TARGET
    diff_counts = {lv: 0 for lv in diff_target}
    for q in questions:
        q_dim = q.get("dimension") or q.get("medical_dimension")
        if q_dim == dim:
            diff = q.get("difficulty_level")
            if diff in diff_counts:
                diff_counts[diff] += 1

    total_diff_target = sum(diff_target.values())
    diff_target_ratio = {d: diff_target[d] / total_diff_target for d in diff_target}
    diff_total = max(sum(diff_counts.values()), 1)
    diff_current_ratio = {d: diff_counts[d] / diff_total for d in diff_target}

    difficulty = max(
        diff_target,
        key=lambda d: diff_target_ratio.get(d, 0) - diff_current_ratio.get(d, 0),
    )

    return dim, difficulty


# Backward-compatible alias
def pick_next_medical_dim(questions: list[dict], limit: int) -> str:
    """Pick the next medical dimension (dimension only, for backward compat)."""
    dim, _ = pick_next_medical_dim_and_diff(questions, limit)
    return dim


# ---------------------------------------------------------------------------
# Offline plan precomputation (concurrent generation)
# ---------------------------------------------------------------------------


def precompute_medical_dim_plan(total_limit: int) -> list[tuple[str, str]]:
    """Precompute a sequence of (dimension, difficulty) pairs.

    Uses local counters to simulate serial pick_next_medical_dim_and_diff
    behavior, eliminating shared mutable state during concurrent generation.
    """
    plan = []
    dim_counts: dict[str, int] = {"identify": 0, "model": 0, "transfer": 0}
    diff_counts: dict[str, dict[str, int]] = {
        d: {lv: 0 for lv in MEDICAL_DIFFICULTY_TARGET}
        for d in MEDICAL_DIM_TARGET
    }

    for _ in range(total_limit):
        dim, diff = _pick_next_medical_offline(dim_counts, diff_counts)
        plan.append((dim, diff))
        dim_counts[dim] += 1
        diff_counts[dim][diff] += 1

    return plan


def _pick_next_medical_offline(
    dim_counts: dict[str, int],
    diff_counts: dict[str, dict[str, int]],
) -> tuple[str, str]:
    """Select next (dimension, difficulty) from offline counters only."""
    if sum(dim_counts.values()) == 0:
        return "identify", "L1"

    # --- Dimension ---
    total_target = sum(MEDICAL_DIM_TARGET.values())
    target_ratio = {d: MEDICAL_DIM_TARGET[d] / total_target for d in MEDICAL_DIM_TARGET}
    current_total = max(sum(dim_counts.values()), 1)
    current_ratio = {d: dim_counts[d] / current_total for d in dim_counts}

    dim = max(MEDICAL_DIM_TARGET, key=lambda d: target_ratio[d] - current_ratio[d])
    if MEDICAL_DIM_TARGET[dim] > 0 and dim_counts[dim] >= MEDICAL_DIM_TARGET[dim]:
        remaining = [d for d in MEDICAL_DIM_TARGET if dim_counts[d] < MEDICAL_DIM_TARGET[d]]
        if remaining:
            dim = max(remaining, key=lambda d: target_ratio[d] - current_ratio[d])

    # --- Difficulty within dimension ---
    diff_target = MEDICAL_DIFFICULTY_TARGET
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
