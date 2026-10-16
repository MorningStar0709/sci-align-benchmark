"""Shared judge base — domain-agnostic LLM-as-Judge API call.

Extracted from control/scorers.py and medical/scorers.py to eliminate
duplicated API call boilerplate across all scorer classes.
"""


def _base_judge_call(system_prompt, user_template, question, expected, model_answer,
                     client, model_name, *, parse_fn=None, max_tokens=3000, temperature=0):
    """Generic LLM-as-Judge API call — domain-agnostic.

    1. Format user_prompt from *user_template* with question/expected/model_answer.
    2. Call ``client.chat.completions.create`` with the given system + user prompts.
    3. Parse the raw response via *parse_fn* (default: ``_parse_judge_result``).

    Args:
        system_prompt: System prompt string.
        user_template: User prompt template — ``str.format()`` with
            ``question``, ``expected_answer``, ``model_answer`` keys.
        question: Question text.
        expected: Expected answer text.
        model_answer: Model-generated answer text.
        client: OpenAI-compatible chat client.
        model_name: Model identifier.
        parse_fn: ``callable(raw_text) -> dict``.  Defaults to
            ``_parse_judge_result(raw, valid_scores=UNIFIED_VALID_SCORES)``.
        max_tokens: Max tokens for the completion request.
        temperature: Sampling temperature.

    Returns:
        dict with at least ``score``, ``reason``, ``issues``.
    """
    from mineru_eval._shared.json_utils import _parse_judge_result
    from mineru_eval._shared.scoring_base import UNIFIED_VALID_SCORES

    user_prompt = user_template.format(
        question=question,
        expected_answer=expected,
        model_answer=model_answer,
    )
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    raw = response.choices[0].message.content.strip()
    if parse_fn is not None:
        return parse_fn(raw)
    return _parse_judge_result(raw, valid_scores=set(UNIFIED_VALID_SCORES))
