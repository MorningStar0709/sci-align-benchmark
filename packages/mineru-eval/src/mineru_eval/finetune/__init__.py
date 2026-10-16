"""Finetune subpackage: FR-LoRA SFT training and GRPO reward functions."""

# Lazy import to avoid blocking the entire subpackage if rewards.py has issues
# (e.g. missing optional dependency). Use getattr pattern for on-demand loading.
def __getattr__(name):
    if name in ("accuracy_reward", "format_reward", "get_reward_funcs",
                "reasoning_depth_reward", "constraint_reward", "get_reward_funcs_c"):
        from mineru_eval.finetune import rewards as _rewards
        return getattr(_rewards, name)
    if name == "__all__":
        return [
            "accuracy_reward",
            "format_reward",
            "reasoning_depth_reward",
            "constraint_reward",
            "get_reward_funcs",
            "get_reward_funcs_c",
        ]
    raise AttributeError(f"module 'mineru_eval.finetune' has no attribute {name!r}")
