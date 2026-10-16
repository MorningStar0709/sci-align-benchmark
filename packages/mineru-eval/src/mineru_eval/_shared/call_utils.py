"""Shared LLM call utilities — domain-agnostic model invocation and retry logic.

Extracted from evaluate.py to break cross-domain dependencies.
"""

import time
from datetime import datetime, timedelta

# Default system prompt for target model (control science domain).
# Other domains should pass their own system_prompt parameter.
TARGET_SYSTEM_PROMPT = """你是一个控制科学领域的专家。请回答以下控制科学问题。
如果问题涉及公式推导，请逐步写出推理过程。
如果问题涉及概念解释，请给出精确定义和必要的数学表达式。
回答要准确、专业、完整。"""


def call_with_retries(label, func, retries=3, base_sleep=2.0):
    """Call a function with exponential-backoff retry on failure.

    Args:
        label: Human-readable label for error messages.
        func: Zero-argument callable to invoke.
        retries: Maximum number of attempts.
        base_sleep: Initial sleep duration in seconds (doubles each retry).

    Raises:
        The last exception if all retries are exhausted.
    """
    last_exc = None
    for attempt in range(1, retries + 1):
        try:
            return func()
        except Exception as exc:
            last_exc = exc
            if attempt >= retries:
                break
            sleep_s = base_sleep * (2 ** (attempt - 1))
            print(f"  {label} failed attempt {attempt}/{retries}: {exc}; retrying in {sleep_s:.1f}s", flush=True)
            time.sleep(sleep_s)
    raise last_exc


def format_eta(minutes):
    """Format an ETA as an absolute timestamp string."""
    if minutes is None:
        return ""
    return (datetime.now() + timedelta(minutes=minutes)).strftime("%Y-%m-%d %H:%M:%S")


def call_target_model(question, client, model_name, provider_kind="openai", system_prompt=None,
                      max_tokens: int = 4000):
    """Call the target model to generate an answer for a given question.

    Args:
        question: 问题文本。
        client: Provider 客户端实例。
        model_name: 模型名称。
        provider_kind: Provider 类型标识 ("local" | "anthropic" | "openai")。
        system_prompt: 自定义系统提示词（默认使用控制科学域）。
        max_tokens: 输出 token 预算（默认 4000，medical 域历史硬编码值；
            补充实验预算覆盖入口，决策报告 §3.4 映射）。
    """
    sys_p = system_prompt or TARGET_SYSTEM_PROMPT
    if provider_kind in ("local", "ollama"):
        resp = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": sys_p},
                {"role": "user", "content": question},
            ],
            temperature=0.3,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content.strip()

    if provider_kind == "anthropic":
        resp = client.messages.create(
            model=model_name,
            max_tokens=max_tokens,
            system=sys_p,
            messages=[{"role": "user", "content": [{"type": "text", "text": question}]}],
        )
        text = ""
        for block in resp.content:
            if hasattr(block, 'text') and block.text:
                text += block.text
        return text.strip()

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": sys_p},
            {"role": "user", "content": question},
        ],
        temperature=0.3,
        max_tokens=max_tokens,
    )
    message = response.choices[0].message
    content = (message.content or "").strip()
    if content:
        return content
    reasoning_content = getattr(message, "reasoning_content", None)
    return (reasoning_content or "").strip()
