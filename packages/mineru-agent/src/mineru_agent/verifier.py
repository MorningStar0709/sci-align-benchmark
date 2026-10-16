"""Verifier — LLM-based step execution verification with heuristic fallback.

Migrated from ``benchmark/agent/agent_cli.py``.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Optional

from mineru_agent.intent_registry import PlanStep, VerifyResult
from mineru_agent.utils import _extract_json_block
from mineru_core.settings import PROVIDERS

logger = logging.getLogger(__name__)

VERIFY_SYSTEM = """你是 ControlSci Data Agent 的执行验证器。评估每一步执行是否成功。

## 评分规则
- score: 0.0 (完全失败) / 0.5 (部分成功) / 1.0 (完全成功)
- issues: 发现的问题列表（空列表表示无问题）
- reason: 简短的中文评估说明
- correction_hint: 如果 score < 1.0，给出修正建议（否则为空字符串）

## 输出格式
```json
{"score": 1.0, "reason": "...", "issues": [], "correction_hint": ""}
```
"""


class Verifier:
    """Layer 3 — LLM-based step execution verification with heuristic fallback."""

    def __init__(self):
        self._system_prompt = VERIFY_SYSTEM

    def verify(self, step: PlanStep, exec_output: str, error: Optional[str] = None) -> VerifyResult:
        if error:
            return VerifyResult(
                passed=False, score=0.0,
                reason=f"执行报错: {error[:200]}",
                issues=[error[:300]],
                correction_hint="检查错误信息并重试或切换 provider",
            )

        if not exec_output or len(exec_output.strip()) < 10:
            return VerifyResult(
                passed=False, score=0.0,
                reason="执行输出为空或过短",
                issues=["empty output"],
                correction_hint="检查子进程是否正常启动并输出内容",
            )

        return self._llm_verify(step, exec_output)

    def _llm_verify(self, step: PlanStep, exec_output: str) -> VerifyResult:
        from openai import OpenAI, DefaultHttpxClient
        from mineru_eval.client_factory import wrap_client

        api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("DEEPSEEK_API_KEY", "")
        if not api_key:
            logger.warning("无法调用 LLM 验证（无 API Key），回退到启发式验证")
            return self._heuristic_verify(exec_output)

        try:
            client = wrap_client(OpenAI(
                api_key=api_key,
                base_url=PROVIDERS["deepseek"].base_url,
                timeout=60,
                max_retries=0,
                http_client=DefaultHttpxClient(proxy=None, trust_env=False),
            ))

            user_prompt = (
                f"Intent: {step.intent_id} ({step.intent_name})\n"
                f"描述: {step.description}\n\n"
                f"执行输出（前 1500 字符）:\n{exec_output[:1500]}"
            )

            response = client.chat.completions.create(
                model="deepseek-v4-flash",
                messages=[
                    {"role": "system", "content": self._system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.1,
                max_tokens=2048,
            )
            raw = response.choices[0].message.content.strip()
            json_str = _extract_json_block(raw)
            data = json.loads(json_str)
        except Exception as e:
            logger.warning("LLM 验证调用失败: %s，回退到启发式验证", e)
            return self._heuristic_verify(exec_output)

        score = float(data.get("score", 0.5))
        return VerifyResult(
            passed=score >= 0.5,
            score=score,
            reason=data.get("reason", ""),
            issues=data.get("issues", []) if isinstance(data.get("issues"), list) else [],
            correction_hint=data.get("correction_hint", ""),
        )

    @staticmethod
    def _heuristic_verify(exec_output: str) -> VerifyResult:
        lower = exec_output.lower()
        fail_keywords = ["error", "exception", "traceback", "failed", "失败", "错误", "超时"]
        success_keywords = ["success", "completed", "完成", "ok", "passed", "通过"]

        fail_count = sum(1 for kw in fail_keywords if kw in lower)
        success_count = sum(1 for kw in success_keywords if kw in lower)

        if fail_count > success_count:
            return VerifyResult(
                passed=False, score=0.0,
                reason=f"输出含 {fail_count} 个错误关键词",
                issues=[f"heuristic: {fail_count} errors vs {success_count} successes"],
                correction_hint="检查子进程输出中的错误信息",
            )
        if success_count > 0:
            return VerifyResult(
                passed=True, score=1.0,
                reason=f"输出含 {success_count} 个成功关键词",
                issues=[],
                correction_hint="",
            )
        return VerifyResult(
            passed=True, score=0.5,
            reason="输出无明显错误或成功标识",
            issues=[],
            correction_hint="",
        )
