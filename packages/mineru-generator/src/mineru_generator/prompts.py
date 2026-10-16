"""Prompt templates for benchmark question generation."""

_DIM_DESCRIPTIONS = {
    "A": (
        "A维度（概念回溯）：考察被测试者对核心控制概念的定义、数学表达式或基本判据的精确回忆。\n"
        "题目必须是单一、聚焦的问题（一次只问一个概念或一个公式），不得包含多个子问题。\n"
        "答案要求（对齐 AScorer 满分标准——答案本身在 Judge 手里必须拿满分）：\n"
        "- 概念定义型：必须包含精确的条件陈述（\"如果...则...\"\"当...时\"\"若...那么\"结构）\n"
        "- 公式表达型：必须包含完整的数学表达式，并说明各参数含义\n"
        "- 对于含多个条件/子句的定义，列出全部条件，不可省略\n"
        "- 精简控制在 200 字以内\n"
        "示例：什么是PID控制器的传递函数？请写出标准形式并说明各参数的作用。"
    ),
    "B": (
        "B维度（多步推理链）：考察被测试者从系统描述出发，经过多步推导得出控制结论的能力。\n"
        "题目必须是单一、聚焦的问题，一次只给一个系统参数组和一个推导目标。\n"
        "题目应给出具体的系统参数或结构（如传递函数、状态空间矩阵、增益值），需要多步计算或代数推导。\n"
        "答案要求（对齐 BScorer 满分标准）：\n"
        "- 必须包含完整推导过程，reasoning_steps ≥ 3\n"
        "- 每步必须是可独立验证的单一断言，步骤之间构成逻辑链\n"
        "- 必须引用推导出的最终计算结果（数值或结论标记如\"解得\"\"得到\"）\n"
        "- 中间步骤不能跳跃\n"
        "示例：给定开环传递函数G(s)=K/[s(s+2)(s+5)]，用Routh-Hurwitz判据求闭环稳定的K范围。"
    ),
    "C": (
        "C维度（条件敏感性）：考察被测试者评估参数/环境/约束变化时，原控制结论是否保持。★\n"
        "题目必须是单一、聚焦的问题，一次只分析一个变化因素。\n"
        "题目须明确给出基准条件、变化条件、以及变化后需要判断的结论。\n\n"
        "答案要求（对齐 CScorer 满分标准——你的答案将被以下 5 级量规评分）：\n"
        "你的答案必须按以下四段结构组织：\n"
        "1. **基准条件复述**：原系统的参数、环境、约束（含具体数值）\n"
        "2. **变化条件分析**：量化变化前后的差异（至少列出数值对比）\n"
        "3. **多方案对比**：至少比较 2 种不同的参数变化方向（如\"增益增大则...增益减小则...\"）\n"
        "4. **工程结论**：给出明确的敏感性判定和工程建议\n\n"
        "Judge 满分标准：完整的敏感性分析 + 多方案对比 + 清晰结论。\n"
        "如果你的答案缺少任一部分，在 Judge 评分中最多只能拿到 0.5 分。\n\n"
        '输出JSON额外包含 "sensitivity_dimension" 字段，枚举值只能是 "parameter" 或 "environment"。'
    ),
    "D": (
        "D维度（开放设计）：考察被测试者设计完整控制方案的能力，包括建模、控制器设计、验证指标。★\n"
        "题目必须是单一、聚焦的问题，一次只设计一个控制目标。\n"
        "题目须给出控制目标和工程约束（如响应时间、稳态误差、鲁棒性指标）。\n\n"
        "答案要求（对齐 DScorer 满分标准——你的答案将被以下 5 级量规评分）：\n"
        "你的答案必须按以下五段结构组织：\n"
        "1. **系统建模**：建立数学模型或传递函数\n"
        "2. **控制器选型**：选择控制方法并说明选择理由\n"
        "3. **参数整定**：给出具体参数数值和整定过程\n"
        "4. **性能验证**：分析关键性能指标（稳定裕度、响应时间、稳态误差）\n"
        "5. **工程考量**：讨论实际工程实施的约束和注意事项\n\n"
        "Judge 满分标准：完整方案 + 参数整定 + 性能分析 + 工程考量。\n"
        "如果你的答案缺少任一部分，在 Judge 评分中最多只能拿到 0.5 分。\n\n"
        '输出JSON额外包含 "rubric" 字段（若无则跳过）：\n'
        '"rubric": {\n'
        '  "feasibility": {"max_score": 1, "weight": 0.2, "description": "..."},\n'
        '  "method_choice": {"max_score": 1, "weight": 0.2, "description": "..."},\n'
        '  "completeness": {"max_score": 1, "weight": 0.2, "description": "..."},\n'
        '  "innovation": {"max_score": 1, "weight": 0.2, "description": "..."},\n'
        '  "clarity": {"max_score": 1, "weight": 0.2, "description": "..."}\n'
        "}"
    ),
}


def build_system_prompt():
    """Build system prompt for control science question generation."""
    return (
        "你是一个控制科学评测题目生成专家。你的任务是基于提供的论文内容，"
        "生成符合指定维度和难度的控制科学评测题目。\n\n"
        "你必须严格按以下JSON格式输出，不要包含任何其他文字或代码块标记：\n"
        '{\n'
        '  "question": "题目正文（中文，可直接阅读）",\n'
        '  "answer": "标准答案（必须完整、自包含，包含所有必要的数学表达式或公式，不可省略任何条件）",\n'
        '  "reasoning_steps": ["第一步（单一可验证断言）", "第二步（依赖前一步）", ...],\n'
        '  "control_category": ["控制分支英文枚举值"],\n'
        '  "scoring_reason": "该题考察...满分标准为...",\n'
        '  "scoring_issues": ["可能扣分点1", "可能扣分点2", ...]\n'
        "}\n\n"
        "## 废弃规则（触犯任一条则整道题被丢弃）\n"
        "- question 包含多个子问题（如\"请写出X并说明Y\"\"说明A和B\"\"比较C与D然后分析E\"）→ 丢弃\n"
        "- answer 不完整（定义/判据缺少必要条件）→ 丢弃\n"
        "- control_category 使用枚举值之外的字符串 → 丢弃\n\n"
        "## 输出约束\n"
        "- question：一次只问一个点。\n"
        "- answer：对于含多个条件/子句的定义，必须列出全部条件，不可遗漏。\n"
        "- answer：精简准确，控制在300字以内，不要冗余展开。\n"
        "- reasoning_steps：每条必须是可独立验证的单一断言，步骤之间构成逻辑链，不得合并为一段话。\n"
        "- scoring_reason：一句话说明该题考查目标和Judge满分标准。\n"
        "- scoring_issues：列出可能扣分的点，2-4 条。\n"
        "- control_category：基于chunk内容的控制分支，枚举值仅限以下之一或最多3个：\n"
        "  classical, modern, nonlinear, robust, optimal, adaptive, digital, intelligent, mpc, multi_agent\n"
    )


def build_user_prompt(chunk_text, chunk_id, dimension, difficulty, override_desc=None):
    """Build user prompt for control science question generation."""
    desc = override_desc or _DIM_DESCRIPTIONS.get(dimension, _DIM_DESCRIPTIONS["A"])
    difficulty_map = {
        "L1": "基础题（单一概念/简单计算）",
        "L2": "中等题（多步骤/中等复杂度）",
        "L3": "较难题（综合应用/深层推理）",
        "L4": "挑战题（跨概念/高级分析）",
    }
    diff_desc = difficulty_map.get(difficulty, "中等题")
    return (
        f"基于以下控制科学教材内容：\n\n"
        f"```\n{chunk_text}\n```\n\n"
        f"请基于此内容生成一道{dimension}维度的评测题，难度为{diff_desc}。\n\n"
        f"{desc}\n\n"
        f"注意：题目必须与提供的教材内容直接相关，不要脱离内容编造。\n"
        f"答案必须包含准确的数学表达式或公式，精简控制在300字以内。\n\n"
        f"⚠️ 严格约束（若违反则整道题被废弃）：\n"
        f"1. question 必须是单一、聚焦的问题——一次只问一个点。\n"
        f"   禁止：\"请写出X并分析Y\"\"说明A和B\"\"比较C与D并给出结论\"等含有两个子问句的题目。\n"
        f"   允许：\"写出X的表达式\"\"分析Y的稳定性\"\"比较C与D的异同\"——一次只问一件。\n"
        f"2. answer 必须能在一段话内完整回答问题，不依赖后续补充。\n"
        f"   对于定义/判据类问题，所有条件必须全部列出，不得省略。\n"
        f"3. 任何时候都不要将两个逻辑上独立的问题合并到一道题中。"
    )


ARBITER_SYSTEM_PROMPT = """你是一个控制科学答案评分员。你的任务是判断两个版本的答案哪个更完整、更正确。

判断规则：
- "round_1"：第1个答案更完整或更正确
- "round_2"：第2个答案更完整或更正确
- "both_equivalent"：两个答案核心内容一致，只是表达方式不同。选此项。
- "both_wrong"：两个答案都不正确或都明显不完整。

倾向宽松——只要核心结论和公式正确即视为等价，选 both_equivalent。

输出严格JSON：{"choice": "<选项>", "reason": "简要理由"}
choice 只能是 round_1 / round_2 / both_equivalent / both_wrong 之一。"""
