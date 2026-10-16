"""Shared constants for MinerU packages — model aliases, version, labels."""

from __future__ import annotations

VERSION = "0.1.0"
"""Current MinerU workspace version."""

# ---------------------------------------------------------------------------
# Model aliases used across benchmark eval modules
# ---------------------------------------------------------------------------
CORE_9: list[str] = [
    "mimo-v2-flash", "deepseek-v4-flash", "qwen3.5:9b",
    "deepseek-v4-pro", "minimax-m2.5", "minimax-m2.7",
    "mimo-v2.5-pro", "mimo-v2-pro", "mimo-v2.5",
]

MODEL_SHORT: dict[str, str] = {
    "mimo-v2-flash": "MiMo-v2-flash",
    "deepseek-v4-flash": "DS-v4-flash",
    "qwen3.5:9b": "Qwen3.5:9b",
    "deepseek-v4-pro": "DS-v4-pro",
    "minimax-m2.5": "MM-m2.5",
    "minimax-m2.7": "MM-m2.7",
    "mimo-v2.5-pro": "MiMo-v2.5-pro",
    "mimo-v2-pro": "MiMo-v2-pro",
    "mimo-v2.5": "MiMo-v2.5",
}

MODEL_FAMILY: dict[str, str] = {
    "mimo-v2-flash": "MiMo",
    "deepseek-v4-flash": "DeepSeek",
    "qwen3.5:9b": "Qwen",
    "deepseek-v4-pro": "DeepSeek",
    "minimax-m2.5": "MiniMax",
    "minimax-m2.7": "MiniMax",
    "mimo-v2.5-pro": "MiMo",
    "mimo-v2-pro": "MiMo",
    "mimo-v2.5": "MiMo",
}

FAMILY_COLORS: dict[str, str] = {
    "MiMo": "#E74C3C",
    "DeepSeek": "#3498DB",
    "Qwen": "#2ECC71",
    "MiniMax": "#9B59B6",
}

# ---------------------------------------------------------------------------
# Standard classification labels for control science corpus
# ---------------------------------------------------------------------------
STANDARD_CATEGORIES: list[str] = [
    "PID控制", "估计与定位", "导航制导与控制", "控制理论", "数字控制",
    "智能控制", "最优控制", "现代控制", "线性系统", "经典控制",
    "自抗扰控制", "自适应控制", "非线性控制", "鲁棒控制",
]
