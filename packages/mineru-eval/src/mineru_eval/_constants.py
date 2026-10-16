"""
Shared constants for T2/T3 deep analysis modules.

Centralized to avoid duplication between eval/t2_deep_analysis/ and
eval/t3_deep_analysis/.  If a constant needs to change, change it here.
"""

CORE_9 = [
    "mimo-v2-flash",
    "deepseek-v4-flash",
    "qwen3.5:9b",
    "deepseek-v4-pro",
    "qwen3.7-plus",
    "qwen3.5-plus",
    "mimo-v2.5-pro",
    "mimo-v2-pro",
    "mimo-v2.5",
]

_MODEL_ALIASES = {
    "Qwen3.7-plus": "qwen3.7-plus",
    "Qwen3.5-plus": "qwen3.5-plus",
}

MODEL_SHORT = {
    "mimo-v2-flash": "MiMo-v2-flash",
    "deepseek-v4-flash": "DS-v4-flash",
    "qwen3.5:9b": "Qwen3.5:9b",
    "deepseek-v4-pro": "DS-v4-pro",
    "qwen3.7-plus": "Qwen3.7-plus",
    "qwen3.5-plus": "Qwen3.5-plus",
    "mimo-v2.5-pro": "MiMo-v2.5-pro",
    "mimo-v2-pro": "MiMo-v2-pro",
    "mimo-v2.5": "MiMo-v2.5",
}

MODEL_FAMILY = {
    "mimo-v2-flash": "MiMo",
    "deepseek-v4-flash": "DeepSeek",
    "qwen3.5:9b": "Qwen",
    "deepseek-v4-pro": "DeepSeek",
    "qwen3.7-plus": "Qwen",
    "qwen3.5-plus": "Qwen",
    "mimo-v2.5-pro": "MiMo",
    "mimo-v2-pro": "MiMo",
    "mimo-v2.5": "MiMo",
}

FAMILY_COLORS = {
    "MiMo": "#1f77b4",
    "DeepSeek": "#ff7f0e",
    "Qwen": "#2ca02c",
    "MiniMax": "#d62728",
}

DIMENSIONS = ["A", "B", "C", "D"]
DIFFICULTIES = ["L1", "L2", "L3", "L4"]
