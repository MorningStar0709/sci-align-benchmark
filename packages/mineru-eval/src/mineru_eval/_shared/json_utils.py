"""Shared JSON parsing utilities for LLM-as-Judge result handling.

Extracted from judge.py to break cross-domain dependencies.
"""

import json
import re

VALID_SCORES = (0.0, 0.25, 0.5, 0.75, 1.0)


def _extract_json_block(text, *, greedy=False):
    """Extract JSON string from markdown code blocks or raw text.

    Args:
        text: Raw LLM response text.
        greedy: When True, use DOTALL regex to match the LAST closing ``` —
            handles embedded backticks inside JSON values (e.g. reason fields
            referencing code snippets).  Default False uses the original
            index-based approach (first closing ``` wins).
    """
    if greedy:
        m = re.search(r'```json\s*\n(.*)\n\s*```', text, re.DOTALL)
        if m:
            return m.group(1).strip()
        m = re.search(r'```\s*\n(.*)\n\s*```', text, re.DOTALL)
        if m:
            return m.group(1).strip()
        return text.strip()
    # Non-greedy (original) path
    if "```json" in text:
        start = text.index("```json") + 7
        end = text.index("```", start) if "```" in text[start:] else len(text)
        return text[start:end].strip()
    if "```" in text:
        start = text.index("```") + 3
        end = text.index("```", start) if "```" in text[start:] else len(text)
        return text[start:end].strip()
    return text.strip()


def _coerce_float(value, default=0.0):
    """Coerce a value to float, returning default on failure."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _parse_judge_result(raw, valid_scores, *, greedy=False):
    """解析 LLM 返回的 JSON 字符串为标准 dict。

    Args:
        raw: Raw LLM response text.
        valid_scores: Set/list of valid score values.
        greedy: Forwarded to ``_extract_json_block`` — enables DOTALL matching
            for embedded backticks in JSON values.
    """
    json_str = _extract_json_block(raw, greedy=greedy)

    try:
        result = json.loads(json_str)
    except (json.JSONDecodeError, ValueError):
        return {
            "score": 0.0,
            "reason": f"LLM 返回格式异常，无法解析: {raw[:200]}",
            "issues": ["LLM judge response parse failure"],
        }

    score = _coerce_float(result.get("score", 0.0), default=0.0)
    if score not in valid_scores:
        closest = min(valid_scores, key=lambda v: abs(v - score))
        score = closest

    issues = result.get("issues", [])
    if isinstance(issues, str):
        issues = [issues]
    elif not isinstance(issues, list):
        issues = []

    output = {
        "score": score,
        "reason": result.get("reason", ""),
        "issues": issues,
    }
    if "dimensions" in result and isinstance(result["dimensions"], dict):
        output["dimensions"] = result["dimensions"]
    return output
