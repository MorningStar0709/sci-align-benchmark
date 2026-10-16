"""Shared scoring base — common scoring infrastructure for all domains.

Extracted from judge_strategies.py to break cross-domain dependencies.
Domain-specific prompts are imported from control/ or medical/ subpackages.
"""

import json
import time

UNIFIED_VALID_SCORES = (0.0, 0.25, 0.5, 0.75, 1.0)


def _snap_to_valid(value, valid_scores=UNIFIED_VALID_SCORES):
    """将浮点平均值吸附到最近的合法评分刻度。"""
    return min(valid_scores, key=lambda v: (abs(v - value), -v))


def _score_via_anthropic(client, model_name, question, expected, model_answer, dimension,
                         rubric=None, *, sys_prompt=None, usr_template=None,
                         parse_fn=None, **extra_format_kwargs):
    """通过 Anthropic 兼容客户端执行评分（供仲裁 Judge 使用）。

    By default, lazy-imports control-domain prompts.  Callers from other domains
    should pass *sys_prompt* and *usr_template* explicitly to avoid the cross-domain
    import.

    Args:
        client: Anthropic-compatible client (``client.messages.create``).
        model_name: Model identifier.
        question: Question text.
        expected: Expected answer text.
        model_answer: Model-generated answer text.
        dimension: Scoring dimension key (used to select prompt when not explicit).
        rubric: Optional rubric dict appended to user prompt (control domain D).
        sys_prompt: Override system prompt (skips lazy import of control prompts).
        usr_template: Override user prompt template (``str.format()`` with
            ``question``, ``expected_answer``, ``model_answer`` keys).
        parse_fn: Custom response parser.  Default: ``_parse_judge_result``.
        **extra_format_kwargs: Extra keys forwarded to ``usr_template.format()``
            (e.g. ``difficulty``).
    """
    from mineru_eval._shared.json_utils import _parse_judge_result

    if sys_prompt is None or usr_template is None:
        # Lazy import: control domain prompts (backward-compatible path)
        from mineru_eval.control.prompts import (
            A_SYSTEM_PROMPT, A_USER_PROMPT_TEMPLATE,
            B_SYSTEM_PROMPT, B_USER_PROMPT_TEMPLATE,
            C_SYSTEM_PROMPT, C_USER_PROMPT_TEMPLATE,
            D_SYSTEM_PROMPT, D_USER_PROMPT_TEMPLATE,
        )
        prompt_map = {
            "A": (A_SYSTEM_PROMPT, A_USER_PROMPT_TEMPLATE),
            "B": (B_SYSTEM_PROMPT, B_USER_PROMPT_TEMPLATE),
            "C": (C_SYSTEM_PROMPT, C_USER_PROMPT_TEMPLATE),
            "D": (D_SYSTEM_PROMPT, D_USER_PROMPT_TEMPLATE),
        }
        default_sys, default_tpl = prompt_map.get(dimension, prompt_map["A"])
    else:
        default_sys, default_tpl = sys_prompt, usr_template

    user_prompt = default_tpl.format(
        question=question, expected_answer=expected, model_answer=model_answer,
        **extra_format_kwargs,
    )
    # 当有 rubric 时，追加评分量规信息（与 RubricScorer 评判标准对齐）
    if rubric and dimension == "D":
        user_prompt += f"\n\n## 评分量规\n{json.dumps(rubric, ensure_ascii=False)}"
    resp = client.messages.create(
        model=model_name, max_tokens=3000,
        system=default_sys, messages=[{"role": "user", "content": user_prompt}],
    )
    text = "".join(b.text for b in resp.content if hasattr(b, "text") and b.text).strip()
    if parse_fn is not None:
        return parse_fn(text)
    return _parse_judge_result(text, valid_scores=set(UNIFIED_VALID_SCORES))


def _score_via_openai(client, model_name, question, expected, model_answer, dimension,
                      rubric=None, *, sys_prompt=None, usr_template=None,
                      parse_fn=None, disable_thinking=False, **extra_format_kwargs):
    """通过 OpenAI 兼容客户端执行仲裁评分（供 Qwen 等 OpenAI API 端点使用）。

    与 _score_via_anthropic 功能等价，但使用 chat.completions.create() 接口。
    """
    from mineru_eval._shared.json_utils import _parse_judge_result

    if sys_prompt is None or usr_template is None:
        from mineru_eval.control.prompts import (
            A_SYSTEM_PROMPT, A_USER_PROMPT_TEMPLATE,
            B_SYSTEM_PROMPT, B_USER_PROMPT_TEMPLATE,
            C_SYSTEM_PROMPT, C_USER_PROMPT_TEMPLATE,
            D_SYSTEM_PROMPT, D_USER_PROMPT_TEMPLATE,
        )
        prompt_map = {
            "A": (A_SYSTEM_PROMPT, A_USER_PROMPT_TEMPLATE),
            "B": (B_SYSTEM_PROMPT, B_USER_PROMPT_TEMPLATE),
            "C": (C_SYSTEM_PROMPT, C_USER_PROMPT_TEMPLATE),
            "D": (D_SYSTEM_PROMPT, D_USER_PROMPT_TEMPLATE),
        }
        default_sys, default_tpl = prompt_map.get(dimension, prompt_map["A"])
    else:
        default_sys, default_tpl = sys_prompt, usr_template

    user_prompt = default_tpl.format(
        question=question, expected_answer=expected, model_answer=model_answer,
        **extra_format_kwargs,
    )
    if rubric and dimension == "D":
        user_prompt += f"\n\n## 评分量规\n{json.dumps(rubric, ensure_ascii=False)}"

    extra_body = {}
    if disable_thinking:
        extra_body["chat_template_kwargs"] = {"enable_thinking": False}

    resp = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": default_sys},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
        max_tokens=3000,
        **({"extra_body": extra_body} if extra_body else {}),
    )
    text = (resp.choices[0].message.content or "").strip()
    if parse_fn is not None:
        return parse_fn(text)
    return _parse_judge_result(text, valid_scores=set(UNIFIED_VALID_SCORES))

def _dual_judge_score(question, expected, model_answer, dimension,
                      judge1_client, judge1_model,
                      judge2_client, judge2_model,
                      arb_client=None, arb_model=None,
                      disable_thinking=False,
                      *, score_fn=None, arb_fn=None,
                      arb_retries: int = 3, arb_backoff_base: float = 1.0,
                      **score_kwargs):
    """双 Judge 并行评分 + 争议仲裁（参数化通用版）。

    Args:
        score_fn: ``callable(question, expected, model_answer, dimension,
            client, model, **score_kwargs) -> result_dict``.
            Default: lazy-imports control domain ``score_model_answer``.
        arb_fn: ``callable(client, model_name, question, expected, model_answer,
            dimension, **score_kwargs) -> result_dict``.
            Default: ``_score_via_anthropic`` (control domain Anthropic arbiter).
        **score_kwargs: Extra keyword arguments forwarded to both *score_fn*
            and *arb_fn* (e.g. ``rubric``, ``difficulty``).

    Returns:
        dict: 包含 score, reason, issues, 以及双 Judge 元数据字段
              (judge1_score, judge2_score, avg_score, score_diff, disputed, arbitrated)
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    if score_fn is None:
        # Avoid circular import — score_model_answer is in control.evaluate
        from mineru_eval.control.evaluate import score_model_answer
        score_fn = score_model_answer

    if arb_fn is None:
        arb_fn = _score_via_anthropic

    def _call(client, model):
        return score_fn(
            question, expected, model_answer, dimension,
            client, model, disable_thinking=disable_thinking,
            **score_kwargs,
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {
            pool.submit(_call, judge1_client, judge1_model): "judge1",
            pool.submit(_call, judge2_client, judge2_model): "judge2",
        }
        results = {}
        errors = {}
        for fut in as_completed(futures):
            name = futures[fut]
            try:
                results[name] = fut.result()
            except Exception as exc:
                errors[name] = exc

    # 降级处理：单个 Judge 失败时使用另一个的结果
    if len(results) < 2:
        if len(results) == 1:
            sole_name, sole_result = next(iter(results.items()))
            sole_score = sole_result["score"]
            failed_names = list(errors.keys())
            return {
                "score": sole_score,
                "reason": f"Degraded ({failed_names} failed): {sole_result.get('reason', '')}",
                "issues": sole_result.get("issues", []) + [f"{failed_names} failed"],
                "judge1_score": sole_score if sole_name == "judge1" else None,
                "judge2_score": sole_score if sole_name == "judge2" else None,
                "avg_score": sole_score,
                "score_diff": 0.0,
                "disputed": False,
                "arbitrated": False,
            }
        # 两个 Judge 都失败，抛出第一个异常
        raise list(errors.values())[0]

    s1 = results["judge1"]["score"]
    s2 = results["judge2"]["score"]
    diff = abs(s1 - s2)
    avg = _snap_to_valid((s1 + s2) / 2.0)
    disputed = diff > 0.5
    arbitrated = False

    final_score = avg
    final_reason = (
        f"Judge1={s1} Judge2={s2} avg={avg} diff={diff}"
        f" | J1:{results['judge1'].get('reason', '')[:80]}"
        f" | J2:{results['judge2'].get('reason', '')[:80]}"
    )
    issues = (results["judge1"].get("issues") or []) + (results["judge2"].get("issues") or [])

    if disputed and arb_client is not None:
        # 仲裁器零重试曾是 65 件均值入分的根因（一次网络抖动即落均值）；
        # 失败路径的最终行为保持不变：score 仍为均值、arbitrated=False、留同一行痕迹。
        delay = arb_backoff_base
        last_exc: Exception | None = None
        for attempt in range(arb_retries + 1):
            try:
                arb_result = arb_fn(
                    arb_client, arb_model, question, expected, model_answer, dimension,
                    **score_kwargs,
                )
                final_score = arb_result["score"]
                final_reason = f"ARBITRATED: {arb_result.get('reason', '')} (J1={s1}, J2={s2})"
                issues += arb_result.get("issues", [])
                arbitrated = True
                last_exc = None
                break
            except Exception as exc:  # noqa: BLE001 - 网络/解析失败都要退避后重试
                last_exc = exc
                if attempt < arb_retries:
                    time.sleep(delay)
                    delay *= 2
        if last_exc is not None:
            final_reason += f" | arbitration_failed: {last_exc}"

    return {
        "score": final_score,
        "reason": final_reason,
        "issues": issues,
        "judge1_score": s1,
        "judge2_score": s2,
        "avg_score": avg,
        "score_diff": diff,
        "disputed": disputed,
        "arbitrated": arbitrated,
    }


# ---------------------------------------------------------------------------
# Rubric scoring helpers
# ---------------------------------------------------------------------------

RUBRIC_BATCH_SYSTEM_PROMPT = """你是一个控制科学开放设计评分员。你需要对模型的回答按多个评分项逐一评分。

每个评分项需要给出 0.0、0.3、0.6、1.0 之一的分值：
- 1.0: 完全覆盖，内容充分且准确
- 0.6: 部分覆盖，有相关内容但不够完整
- 0.3: 轻微提及，只有少量相关内容
- 0.0: 未覆盖，完全没有涉及该评分项

请严格按照以下 JSON 格式输出所有评分项的结果：
{"items": [{"name": "...", "score": 0.0, "reason": "..."}, ...]}"""


def _rubric_judge_batch(question, rubric_items, model_answer, client, model_name,
                       disable_thinking: bool = False):
    """批量 rubric 评分：一次 API 调用评估所有子项。"""
    from mineru_eval._shared.json_utils import _coerce_float, _extract_json_block
    items_desc = []
    for item in rubric_items:
        items_desc.append(f"- {item['name']} (weight {item['weight']}): {item['description']}")

    user_prompt = f"""## 问题
{question}

## 评分项列表
{chr(10).join(items_desc)}

## 模型回答
{model_answer}

请对每个评分项逐一评分。"""

    kwargs = dict(
        model=model_name,
        messages=[
            {"role": "system", "content": RUBRIC_BATCH_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
        max_tokens=2000,
    )

    response = client.chat.completions.create(**kwargs)
    raw = response.choices[0].message.content.strip()
    json_str = _extract_json_block(raw)
    try:
        result = json.loads(json_str)
    except (json.JSONDecodeError, ValueError):
        return None

    items = result.get("items", [])
    if not items:
        return None

    valid = {0.0, 0.3, 0.6, 1.0}
    detail_map = {}
    for item in items:
        name = item.get("name", "")
        score = _coerce_float(item.get("score", 0.0), default=0.0)
        if score not in valid:
            score = min(valid, key=lambda v: abs(v - score))
        detail_map[name] = {"score": score, "reason": item.get("reason", "")}
    return detail_map


def _format_rubric_item(item):
    name = item.get("name", "")
    desc = item.get("description", "")
    weight = item.get("weight", 1)
    return f"{name} (weight {weight}): {desc}"


def _rubric_judge_single(question, rubric_item, model_answer, client, model_name,
                         disable_thinking: bool = False):
    """对单个 rubric 子项调用 LLM-as-Judge 评分（降级方案）。"""
    from mineru_eval._shared.json_utils import _coerce_float, _extract_json_block
    system_prompt = """你是一个控制科学开放设计评分员。你需要判断模型的回答在给定的评分项上是否充分覆盖。
评分规则：
- 1.0: 完全覆盖，内容充分且准确
- 0.6: 部分覆盖，有相关内容但不够完整
- 0.3: 轻微提及，只有少量相关内容
- 0.0: 未覆盖，完全没有涉及该评分项
请严格按照 JSON 格式输出：{"score": <分数>, "reason": "<评分理由>"}"""

    user_prompt = f"""## 问题
{question}

## 评分项
{_format_rubric_item(rubric_item)}

## 模型回答
{model_answer}

请判断模型回答是否充分覆盖了该评分项。"""

    kwargs = dict(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
        max_tokens=1000,
    )

    response = client.chat.completions.create(**kwargs)
    raw = response.choices[0].message.content.strip()
    json_str = _extract_json_block(raw)
    try:
        result = json.loads(json_str)
    except (json.JSONDecodeError, ValueError):
        return {"score": 0.0, "reason": "Parse error", "weight": rubric_item.get("weight", 1)}
    score = _coerce_float(result.get("score", 0.0), default=0.0)
    valid = {0.0, 0.3, 0.6, 1.0}
    if score not in valid:
        score = min(valid, key=lambda v: abs(v - score))
    return {"score": score, "reason": result.get("reason", ""), "weight": rubric_item.get("weight", 1)}
