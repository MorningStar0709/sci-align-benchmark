"""Intent Router — LLM few-shot natural language to intent plan.

Migrated from ``benchmark/agent/agent_cli.py``.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import List, Optional

from mineru_agent.intent_registry import IntentRegistry, PlanStep
from mineru_agent.utils import _extract_json_block
from mineru_core.settings import PROVIDERS

logger = logging.getLogger(__name__)

FEW_SHOT_EXAMPLES = """\
示例1:
用户: "帮我解析 data/sources/ 下这 23 本控制科学教材"
计划:
```json
{
  "plan": [
    {
      "step": 1,
      "intent_id": "corpus_parse",
      "description": "使用 MinerU 批量解析 23 本控制科学教材为结构化 Markdown 语料",
      "parameters": {"source": "data/sources/", "recursive": true}
    }
  ]
}
```

示例2:
用户: "审计这批 arXiv 论文的公式-图片跨模态对齐质量"
计划:
```json
{
  "plan": [
    {
      "step": 1,
      "intent_id": "cross_modal_audit",
      "description": "使用 MiMo-V2.5 原生视觉模型全量扫描语料 chunk，审计图片-公式语义对齐",
      "parameters": {"source": "corpus/chunks/", "concurrency": 5, "resume": true}
    }
  ]
}
```

示例3:
用户: "从零生成控制科学评测数据集 500 题并跑全模型评测"
计划:
```json
{
  "plan": [
    {
      "step": 1,
      "intent_id": "benchmark_build",
      "description": "基于已解析语料 chunk 使用三 Provider 生成 500 题评测数据集",
      "parameters": {"source": "corpus/", "target_size": 500, "output": "benchmark/dataset/core.json"}
    },
    {
      "step": 2,
      "intent_id": "quality_arbitrate",
      "description": "对生成题目做双层 LLM 仲裁确保质量",
      "parameters": {"source": "benchmark/dataset/core.json", "consistency_filter": true}
    },
    {
      "step": 3,
      "intent_id": "model_evaluate",
      "description": "评测 deepseek-v4-flash 模型",
      "parameters": {"model": "deepseek-v4-flash", "mode": "model"}
    },
    {
      "step": 4,
      "intent_id": "leaderboard_viz",
      "description": "生成排行榜 JSON 和可视化 HTML",
      "parameters": {"source": "benchmark/eval/results/", "formats": ["json", "html"]}
    }
  ]
}
```

示例4:
用户: "对比 API Judge 和本地 Ollama Judge 的评分差异"
计划:
```json
{
  "plan": [
    {
      "step": 1,
      "intent_id": "multi_judge_compare",
      "description": "使用 DeepSeek API Judge 和 Ollama 本地 Judge 双路径评测，生成一致性对比",
      "parameters": {"models": ["deepseek-v4-flash", "mimo-v2-flash", "qwen3.5:9b"], "api_judge": "deepseek-v4-flash", "local_judge": "qwen3.5:9b"}
    }
  ]
}
```

示例5:
用户: "QLoRA 微调 qwen3.5 模型并验证 Perplexity 改善"
计划:
```json
{
  "plan": [
    {
      "step": 1,
      "intent_id": "local_finetune",
      "description": "使用 QLoRA 在控制科学语料上微调 qwen3.5:9b，训练后自动运行 Perplexity 探针",
      "parameters": {"dataset": "finetune/data/", "base_model": "qwen3.5:9b", "epochs": 3, "perplexity_validation": true}
    }
  ]
}
```

示例6:
用户: "生成语料全维度质量报告"
计划:
```json
{
  "plan": [
    {
      "step": 1,
      "intent_id": "corpus_quality_report",
      "description": "基于 chunk 统计和跨模态审计结果，使用 DeepSeek 分析生成全维度质量报告",
      "parameters": {"source": "corpus/stats/", "granularity": "full", "output_format": "both"}
    }
  ]
}
```

示例7:
用户: "全量复现整个评测流程"
计划:
```json
{
  "plan": [
    {
      "step": 1,
      "intent_id": "reproduce_all",
      "description": "一键执行全链路从语料解析到排行榜生成",
      "parameters": {"skip_existing": true}
    }
  ]
}
```

示例8:
用户: "检索 2025-2026 控制科学前沿论文，把它们加入评测基准，跑全模型评测"
计划:
```json
{
  "plan": [
    {
      "step": 1,
      "intent_id": "arxiv_search",
      "description": "使用自研 searching-arxiv-papers Skill 检索 2025-2026 控制科学新论文并下载 PDF",
      "parameters": {"keywords": ["control barrier function", "differentiable MPC", "safe RL"], "max_papers": 5}
    },
    {
      "step": 2,
      "intent_id": "mineru_parse",
      "description": "调用 MinerU API 解析下载的 PDF 为结构化 Markdown，含公式提取和图片导出",
      "parameters": {"source": "data/sources_flywheel/", "output-dir": "data/sources_flywheel/md/"}
    },
    {
      "step": 3,
      "intent_id": "corpus_parse",
      "description": "加载已解析的 flywheel 语料库，验证 chunk 结构和元数据完整性",
      "parameters": {"source": "data/sources_flywheel/md/", "recursive": true, "max_files": 5}
    },
    {
      "step": 4,
      "intent_id": "benchmark_build",
      "description": "从新 chunk 生成 15-20 道评测题",
      "parameters": {"source": "corpus/chunks/", "target_size": 15}
    },
    {
      "step": 5,
      "intent_id": "quality_arbitrate",
      "description": "LLM 双层仲裁过滤低质题目",
      "parameters": {"consistency_filter": true}
    },
    {
      "step": 6,
      "intent_id": "model_evaluate",
      "description": "评测 deepseek-v4-flash 在新题上的表现",
      "parameters": {"model": "deepseek-v4-flash", "mode": "model"}
    },
    {
      "step": 7,
      "intent_id": "leaderboard_viz",
      "description": "更新排行榜展示 500→512 题扩张",
      "parameters": {"source": "benchmark/eval/results/", "formats": ["json", "html"]}
    }
  ]
}
```
"""

INTENT_ROUTER_SYSTEM = """你是 ControlSci Data Agent 的 Intent Router。你的任务是将用户的自然语言请求拆解为可执行的能力步骤序列。

## 可用能力（intent_id）

{capabilities}

## 规则

1. 只使用上述列出的 intent_id，不允许编造
2. 每个 step 必须包含: intent_id, description, parameters（如需）
3. 考虑依赖关系：如果 intent X 依赖 intent Y，Y 必须排在 X 前面
4. parameters 中填写执行所需的关键参数（source, model, output 等），使用默认值即可
5. 输出严格 JSON，包裹在 ```json ``` 代码块中
6. 如果用户的请求中没有明确提到某个参数，使用合理的默认值

## 输出格式

```json
{{"plan": [{{"step": 1, "intent_id": "...", "description": "...", "parameters": {{...}}}}]}}
```
"""


class IntentRouter:
    """Layer 1 — LLM few-shot natural language to intent plan."""

    def __init__(self, registry: IntentRegistry = None):
        self._registry = registry or IntentRegistry()

    def _build_system_prompt(self) -> str:
        return INTENT_ROUTER_SYSTEM.format(
            capabilities=self._registry.to_compact_description()
        )

    def _call_llm(self, user_input: str) -> str:
        from openai import OpenAI, DefaultHttpxClient
        from mineru_eval.client_factory import wrap_client

        api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("DEEPSEEK_API_KEY", "")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY 或 DEEPSEEK_API_KEY 环境变量未设置")

        client = wrap_client(OpenAI(
            api_key=api_key,
            base_url=PROVIDERS["deepseek"].base_url,
            timeout=120,
            max_retries=1,
            http_client=DefaultHttpxClient(proxy=None, trust_env=False),
        ))

        system = self._build_system_prompt()
        safe_input = json.dumps(user_input, ensure_ascii=False)
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": FEW_SHOT_EXAMPLES},
            {"role": "user", "content": f"用户: {safe_input}\n计划:"},
        ]

        response = client.chat.completions.create(
            model="deepseek-v4-flash",
            messages=messages,
            temperature=0.1,
            max_tokens=4096,
        )
        return response.choices[0].message.content.strip()

    def plan(self, user_input: str) -> List[PlanStep]:
        raw = self._call_llm(user_input)
        json_str = _extract_json_block(raw)

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            logger.warning("LLM 返回 JSON 解析失败，尝试修复: %s", raw[:200])
            data = self._attempt_repair(json_str)
            if data is None:
                raise ValueError(f"Intent Router 无法解析 LLM 输出: {raw[:500]}")

        plan_raw = data.get("plan", [])
        if not plan_raw:
            raise ValueError(f"Intent Router 返回空计划: {raw[:500]}")

        plan = []
        valid_ids = set(self._registry.list_ids())
        for s in plan_raw:
            pid = s.get("intent_id", "")
            if pid not in valid_ids:
                logger.warning("跳过未知 intent_id: %s", pid)
                continue
            intent_info = self._registry.get(pid) or {}
            plan.append(PlanStep(
                intent_id=pid,
                intent_name=intent_info.get("name", pid),
                description=s.get("description", ""),
                tool=", ".join(intent_info.get("toolchain", [])),
                resource_type=intent_info.get("resource_type", "script"),
                depends_on=intent_info.get("depends_on", []),
                parameters=s.get("parameters", {}),
                rank=s.get("step", len(plan) + 1),
            ))

        if not plan:
            raise ValueError(f"Intent Router 产生的计划无有效 intent: {raw[:500]}")

        return plan

    @staticmethod
    def _attempt_repair(json_str: str) -> Optional[dict]:
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass
        cleaned = re.sub(r",\s*}", "}", json_str)
        cleaned = re.sub(r",\s*]", "]", cleaned)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            return None
