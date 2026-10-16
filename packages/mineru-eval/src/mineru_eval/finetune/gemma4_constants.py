"""Gemma4 加载常量（纯常量模块，禁止引入 torch/transformers/peft 重依赖）。

供 adapter_loader（包层）与 scripts/_gemma4_common（脚本层）共用，
使编排器 run_gemma4_pipeline 的 dry-run/preflight 路径不被传染 ML 堆栈。

GEMMA4_KEY_MAPPING: Gemma4 multimodal checkpoint 的权重键带
model.language_model.* 前缀，Gemma4ForCausalLM 期望 model.*。不重映射时
文本塔全部 MISSING 并被静默随机初始化（transformers 5.14 实测，2026-07-26）。
"""
from __future__ import annotations

GEMMA4_KEY_MAPPING = {r"^model\.language_model\.": "model."}
