"""Medical cross-domain 3-dimension x 4-difficulty prompt templates.

Contains identify/model/transfer dimension descriptions and difficulty-aware
prompt builders for the medical domain. Difficulty levels mirror the control
science domain's L1-L4 hierarchy.
"""

# ---------------------------------------------------------------------------
# Dimension descriptions (difficulty-agnostic cognitive ability)
# ---------------------------------------------------------------------------

_MEDICAL_DIM_DESCRIPTIONS = {
    "identify": (
        "维度：识别（Identify）——考察被测试者能否从医学文本中识别隐含的控制科学概念。\n"
        "题目要求：从提供的医学论文内容出发，识别其中隐含的控制理论概念或机制，"
        "并说明其与哪个控制理论分支对应。"
    ),
    "model": (
        "维度：建模（Model）——考察被测试者能否将医学场景抽象为控制科学模型。\n"
        "题目要求：从提供的医学论文内容出发，将其中描述的医学过程或系统"
        "建模为控制科学模型（状态空间方程、传递函数、框图等）。"
    ),
    "transfer": (
        "维度：迁移（Transfer）——考察被测试者能否预测控制方法在医学场景中的适用性。\n"
        "题目要求：从提供的医学论文内容出发，分析如果将某种控制方法"
        "应用于该医学场景，预期会产生什么效果或面临什么挑战。"
    ),
}

# ---------------------------------------------------------------------------
# Difficulty-specific answer requirements (dimension x difficulty = 12 cells)
# ---------------------------------------------------------------------------

_MEDICAL_DIFFICULTY_REQUIREMENTS = {
    # --- identify dimension ---
    ("identify", "L1"): (
        "答案要求（L1-基础）：\n"
        "- 识别医学文本中**显式**提到的控制理论概念\n"
        "- 给出该概念的标准定义\n"
        "- 指出其在医学文本中的对应描述\n"
        "- 精简控制在 200 字以内"
    ),
    ("identify", "L2"): (
        "答案要求（L2-标准）：\n"
        "- 识别医学文本中**隐含**的控制理论概念\n"
        "- 建立控制概念与医学场景的精确对应关系\n"
        "- 引用文本证据支持映射\n"
        "- 精简控制在 250 字以内"
    ),
    ("identify", "L3"): (
        "答案要求（L3-进阶）：\n"
        "- 识别多个隐含控制概念并完成跨分支映射\n"
        "- 给出完整的理论解释和概念间关联\n"
        "- 引用文本证据链\n"
        "- 精简控制在 300 字以内"
    ),
    ("identify", "L4"): (
        "答案要求（L4-高级）：\n"
        "- 识别深层隐含概念，发现反直觉映射关系\n"
        "- 给出完整理论链证明\n"
        "- 讨论映射的局限性和边界条件\n"
        "- 精简控制在 350 字以内"
    ),
    # --- model dimension ---
    ("model", "L1"): (
        "答案要求（L1-基础）：\n"
        "- 给出模型名称/类型（如 SISO/MIMO、线性/非线性）\n"
        "- 指出基本医学对应关系\n"
        "- 精简控制在 200 字以内"
    ),
    ("model", "L2"): (
        "答案要求（L2-标准）：\n"
        "- 提供完整模型框架（状态方程/传递函数/框图）\n"
        "- 参数有物理意义 + 医学对应\n"
        "- 精简控制在 250 字以内"
    ),
    ("model", "L3"): (
        "答案要求（L3-进阶）：\n"
        "- 完整建模过程（状态变量、控制输入、观测输出）\n"
        "- 参数估计 + 数学表达式 + 医学对应\n"
        "- 讨论模型假设与适用范围\n"
        "- 精简控制在 300 字以内"
    ),
    ("model", "L4"): (
        "答案要求（L4-高级）：\n"
        "- 提出多种建模方案并对比优劣\n"
        "- 论证最优选择的临床约束合理性\n"
        "- 讨论模型复杂度与临床可行性的权衡\n"
        "- 精简控制在 350 字以内"
    ),
    # --- transfer dimension ---
    ("transfer", "L1"): (
        "答案要求（L1-基础）：\n"
        "- 给出 1 个合理的定性预测\n"
        "- 逻辑清晰，说明控制方法与医学场景的关联\n"
        "- 精简控制在 200 字以内"
    ),
    ("transfer", "L2"): (
        "答案要求（L2-标准）：\n"
        "- 给出合理定性预测\n"
        "- 清晰的逻辑推导链（因果关系明确）\n"
        "- 引用医学文本中的支持证据\n"
        "- 精简控制在 250 字以内"
    ),
    ("transfer", "L3"): (
        "答案要求（L3-进阶）：\n"
        "- 提供量化预测（参数变化对系统行为的影响）\n"
        "- 讨论临床意义和预期效果\n"
        "- 精简控制在 300 字以内"
    ),
    ("transfer", "L4"): (
        "答案要求（L4-高级）：\n"
        "- 量化预测 + 多因素交互分析\n"
        "- 讨论临床实施约束（安全性、实时性、个体差异）\n"
        "- 给出可行建议和优化方向\n"
        "- 精简控制在 350 字以内"
    ),
}

# Difficulty label for prompt injection
_DIFFICULTY_LABELS = {
    "L1": "基础",
    "L2": "标准",
    "L3": "进阶",
    "L4": "高级",
}

# Dimension label
_DIM_LABELS = {
    "identify": "识别",
    "model": "建模",
    "transfer": "迁移",
}


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------


def build_medical_system_prompt():
    """Build system prompt for medical cross-domain question generation."""
    return (
        "你是一个控制科学×医学跨域评测题目生成专家。你的任务是基于提供的医学论文内容，"
        "生成控制理论与医学场景交叉的跨域推理评测题目。\n\n"
        "你必须严格按以下JSON格式输出，不要包含任何其他文字或代码块标记：\n"
        '{\n'
        '  "question": "题目正文（中文，可直接阅读）",\n'
        '  "answer": "标准答案（必须完整、自包含，包含所有必要的分析或公式）",\n'
        '  "reasoning_steps": ["第一步（单一可验证断言）", "第二步（依赖前一步）", ...],\n'
        '  "medical_dimension": "identify 或 model 或 transfer",\n'
        '  "difficulty_level": "L1 或 L2 或 L3 或 L4",\n'
        '  "control_concept": "对应的控制理论概念（如 feedback, stability, optimal control 等）",\n'
        '  "scoring_reason": "该题考察...满分标准为...",\n'
        '  "scoring_issues": ["可能扣分点1", "可能扣分点2", ...]\n'
        "}\n\n"
        "## 废弃规则（触犯任一条则整道题被丢弃）\n"
        "- question 包含多个子问题 → 丢弃\n"
        "- answer 不完整或缺少关键分析 → 丢弃\n"
        "- medical_dimension 不在枚举值内 → 丢弃\n"
        "- difficulty_level 不在 L1/L2/L3/L4 内 → 丢弃\n\n"
        "## 输出约束\n"
        "- question：一次只问一个点，必须与提供的医学论文内容直接相关\n"
        "- answer：按难度等级要求的字数限制输出\n"
        "- reasoning_steps：每条必须是可独立验证的单一断言\n"
        "- control_concept：必须是控制理论标准术语的英文\n"
    )


def build_medical_user_prompt(chunk_text, chunk_id, dimension, difficulty="L2",
                              override_desc=None):
    """Build user prompt for medical cross-domain question generation.

    Args:
        chunk_text: Medical paper text content.
        chunk_id: Paper identifier.
        dimension: "identify", "model", or "transfer".
        difficulty: "L1", "L2", "L3", or "L4" (default "L2").
        override_desc: Optional override for dimension description.
    """
    desc = override_desc or _MEDICAL_DIM_DESCRIPTIONS.get(
        dimension, _MEDICAL_DIM_DESCRIPTIONS["identify"]
    )
    diff_req = _MEDICAL_DIFFICULTY_REQUIREMENTS.get(
        (dimension, difficulty), _MEDICAL_DIFFICULTY_REQUIREMENTS.get((dimension, "L2"), "")
    )
    dim_label = _DIM_LABELS.get(dimension, dimension)
    diff_label = _DIFFICULTY_LABELS.get(difficulty, "标准")

    return (
        f"基于以下医学论文内容：\n\n"
        f"```\n{chunk_text}\n```\n\n"
        f"请基于此内容生成一道{dim_label}维度、{diff_label}难度（{difficulty}）的跨域推理评测题。\n\n"
        f"{desc}\n\n"
        f"{diff_req}\n\n"
        f"注意：题目必须与提供的医学论文内容直接相关，不要脱离内容编造。\n"
        f"答案必须包含完整的分析或公式。\n\n"
        f"⚠️ 严格约束（若违反则整道题被废弃）：\n"
        f"1. question 必须是单一、聚焦的问题——一次只问一个点。\n"
        f"2. answer 必须能在一段话内完整回答问题，不依赖后续补充。\n"
        f"3. 任何时候都不要将两个逻辑上独立的问题合并到一道题中。\n"
        f"4. difficulty_level 必须与指定难度（{difficulty}）匹配。"
    )
