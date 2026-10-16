"""Unified LLM API call layer with retry, JSON extraction, proxy isolation, and diagnostic logging."""
import hashlib
import json
import re
import threading
import time
from pathlib import Path

import mineru_core  # noqa: F401 — trigger _load_dotenv()
from mineru_core.settings import PROVIDERS as _PROVIDERS
from mineru_eval.client_factory import wrap_client

PROVIDER_CONFIG = {
    "deepseek": {
        "base_url": _PROVIDERS["deepseek"].base_url,
        "model": _PROVIDERS["deepseek"].default_model,
        "models": _PROVIDERS["deepseek"].models or {},
        "sdk": "openai",
    },
    "qwen": {
        "base_url": _PROVIDERS["qwen"].base_url,
        "model": _PROVIDERS["qwen"].default_model,
        "models": _PROVIDERS["qwen"].models or {},
        "sdk": "openai",
    },
    "mimo": {
        "base_url": _PROVIDERS["mimo"].base_url,
        "model": _PROVIDERS["mimo"].default_model,
        "models": _PROVIDERS["mimo"].models or {},
        "sdk": "openai",
    },
}

PROVIDER_ENV_MAP = {
    "deepseek": _PROVIDERS["deepseek"].api_key_env,
    "qwen": _PROVIDERS["qwen"].api_key_env,
    "mimo": _PROVIDERS["mimo"].api_key_env,
}


def resolve_model(provider: str, alias: str | None = None) -> str:
    """Resolve a model name from provider + optional alias.

    Resolution order:
      1. ``alias`` matches a key in ``PROVIDER_CONFIG[provider]["models"]`` → full ID
      2. ``alias`` is a non-empty string not in the map → used as-is (full ID pass-through)
      3. ``alias`` is None/empty → ``PROVIDER_CONFIG[provider]["model"]`` (default)

    Examples:
      >>> resolve_model("deepseek", "ds-pro")    # → "deepseek-v4-pro"
      >>> resolve_model("deepseek", "ds-flash")  # → "deepseek-v4-flash"
      >>> resolve_model("deepseek")              # → "deepseek-v4-flash" (default)
      >>> resolve_model("mimo", "mm-2.5-pro")    # → "mimo-v2.5-pro"
      >>> resolve_model("deepseek", "deepseek-v4-pro")  # → "deepseek-v4-pro" (pass-through)
    """
    config = PROVIDER_CONFIG[provider]
    models = config.get("models", {})
    if alias:
        return models.get(alias, alias)
    return config["model"]


def _try_fix_json(text):
    """修复常见 JSON 格式问题后重试解析。"""
    try:
        decoder = json.JSONDecoder()
        obj, _ = decoder.raw_decode(text)
        return obj
    except json.JSONDecodeError:
        pass
    try:
        return json.loads(text.replace("'", '"'))
    except json.JSONDecodeError:
        pass
    try:
        fixed = re.sub(r',\s*}', '}', text)
        fixed = re.sub(r',\s*]', ']', fixed)
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass
    return None


def parse_json_response(text):
    """Multi-layer JSON extraction from LLM output with format repair."""
    if not text:
        return None
    text = text.strip()
    pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
    matches = re.findall(pattern, text, re.DOTALL)
    for match in matches:
        try:
            return json.loads(match.strip())
        except json.JSONDecodeError:
            continue
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start : end + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            result = _try_fix_json(candidate)
            if result is not None:
                return result
    return _try_fix_json(text)


def validate_api_question(q):
    """Validate LLM-generated question has required fields."""
    if not isinstance(q, dict):
        return False
    for field in ["question", "answer", "reasoning_steps"]:
        if field not in q or not q[field]:
            return False
    if not isinstance(q.get("reasoning_steps"), list) or len(q["reasoning_steps"]) == 0:
        return False
    return True


def validate_arbiter_response(q):
    """Validate arbitration response shape."""
    if not isinstance(q, dict):
        return False
    return q.get("choice") in {"round_1", "round_2", "both_equivalent", "both_wrong"}


def validate_diag_payload(parsed, diag_ctx=None):
    if diag_ctx and diag_ctx.get("schema") == "arbiter":
        return validate_arbiter_response(parsed)
    return validate_api_question(parsed)


def _write_diag_log(diag_path, record):
    if not diag_path:
        return
    Path(diag_path).parent.mkdir(parents=True, exist_ok=True)
    with open(diag_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _call_with_retry(call_fn, model_name, max_retries=3):
    last_error = None
    for attempt in range(max_retries):
        try:
            result = call_fn()
            if result is not None:
                return result
            last_error = RuntimeError(f"call_fn returned None (attempt {attempt+1}/{max_retries})")
        except Exception as e:
            last_error = e
            status = getattr(e, 'status_code', None) or getattr(e, 'code', None)
            if status == 429:
                print(f"[API] 429 rate limited ({model_name}), retry {attempt+1}/{max_retries}", flush=True)
            else:
                print(f"[API] Error ({model_name}, attempt {attempt+1}/{max_retries}): {type(e).__name__}: {e}", flush=True)
        if attempt < max_retries - 1:
            time.sleep(2 ** attempt)
    print(f"[API] All {max_retries} attempts failed ({model_name}). Last error: {last_error}", flush=True)
    return None


def _build_diag_record(resp, content, elapsed_ms, parsed, parse_success, validated, diag_ctx, model, sp_hash, _usage_acc):
    """Build diagnostic log record shared between OpenAI and Anthropic callers."""
    record = {
        "q_index": diag_ctx.get("q_index"),
        "provider": diag_ctx.get("provider"),
        "model": model,
        "round": diag_ctx.get("round"),
        "elapsed_ms": elapsed_ms,
        "content_length": len(content),
        "parse_success": parse_success,
        "validation_passed": validated,
        "system_prompt_hash": sp_hash,
    }
    if _usage_acc is not None and hasattr(resp, 'usage') and resp.usage:
        if hasattr(resp.usage, 'prompt_tokens'):
            record["prompt_tokens"] = resp.usage.prompt_tokens or 0
            record["completion_tokens"] = resp.usage.completion_tokens or 0
        else:
            record["prompt_tokens"] = getattr(resp.usage, 'input_tokens', 0) or 0
            record["completion_tokens"] = getattr(resp.usage, 'output_tokens', 0) or 0
    return record


def _track_usage(resp, _usage_acc):
    """Accumulate token usage from response."""
    if _usage_acc is not None and hasattr(resp, 'usage') and resp.usage:
        if hasattr(resp.usage, 'prompt_tokens'):
            _usage_acc[0] += resp.usage.prompt_tokens or 0
            _usage_acc[1] += resp.usage.completion_tokens or 0
        else:
            _usage_acc[0] += getattr(resp.usage, 'input_tokens', 0) or 0
            _usage_acc[1] += getattr(resp.usage, 'output_tokens', 0) or 0


# Cache for OpenAI clients to avoid per-call TCP/TLS handshake overhead.
# Each unique (api_key_prefix, base_url, timeout) reuses a single httpx connection pool.
_client_cache: dict[str, object] = {}
_client_lock = threading.Lock()


def _get_cached_client(api_key: str | None, base_url: str, timeout: int):
    """Return a cached OpenAI client, creating one on first access (thread-safe)."""
    cache_key = f"{(api_key or '')[:12]}@{base_url}@{timeout}"
    with _client_lock:
        if cache_key not in _client_cache:
            from openai import OpenAI, DefaultHttpxClient
            _client_cache[cache_key] = wrap_client(OpenAI(
                api_key=api_key,
                base_url=base_url,
                timeout=timeout,
                max_retries=0,
                http_client=DefaultHttpxClient(proxy=None, trust_env=False),
            ))
    return _client_cache[cache_key]


def _call_openai(config, api_key, system_prompt, user_prompt, timeout=120, max_retries=3, _usage_acc=None, diag_path=None, diag_ctx=None, max_tokens=8192, thinking=False):
    client = _get_cached_client(api_key, config["base_url"], timeout)
    model = config["model"]
    kwargs = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.1,
        "max_tokens": max_tokens,
    }
    if config["sdk"] == "openai":
        kwargs["response_format"] = {"type": "json_object"}
    # Thinking mode: only enable explicitly; proxy layer handles disable automatically
    if thinking:
        if model.startswith(("mimo", "deepseek")):
            kwargs["extra_body"] = {"thinking": {"type": "enabled"}}
        elif model.startswith("qwen"):
            kwargs["extra_body"] = {"enable_thinking": True}

    sp_hash = hashlib.md5(system_prompt.encode("utf-8")).hexdigest()[:8]

    def _attempt():
        t_call_start = time.time()
        resp = client.chat.completions.create(**kwargs)
        content = resp.choices[0].message.content
        if content is None:
            raise RuntimeError("API returned content=None (likely content filtered)")
        content = content.strip()
        elapsed_ms = int((time.time() - t_call_start) * 1000)
        parsed = parse_json_response(content)
        parse_success = parsed is not None
        validated = parse_success and validate_diag_payload(parsed, diag_ctx)

        if diag_path and diag_ctx:
            record = _build_diag_record(resp, content, elapsed_ms, parsed, parse_success, validated, diag_ctx, model, sp_hash, _usage_acc)
            _write_diag_log(diag_path, record)

        if parsed:
            _track_usage(resp, _usage_acc)
            return parsed
        return None

    return _call_with_retry(_attempt, model, max_retries)


def _call_anthropic(config, api_key, system_prompt, user_prompt, timeout=120, max_retries=3, _usage_acc=None, diag_path=None, diag_ctx=None, max_tokens=8192):
    import httpx
    from anthropic import Anthropic

    client = Anthropic(
        api_key=api_key,
        base_url=config["base_url"],
        timeout=timeout,
        http_client=httpx.Client(proxy=None, trust_env=False),
    )
    model = config["model"]
    sp_hash = hashlib.md5(system_prompt.encode("utf-8")).hexdigest()[:8]

    def _attempt():
        t_call_start = time.time()
        resp = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": [{"type": "text", "text": user_prompt}]}],
        )
        text = ""
        for block in resp.content:
            if hasattr(block, 'text') and block.text:
                text += block.text
        if not text:
            return None
        elapsed_ms = int((time.time() - t_call_start) * 1000)
        parsed = parse_json_response(text)
        parse_success = parsed is not None
        validated = parse_success and validate_diag_payload(parsed, diag_ctx)

        if diag_path and diag_ctx:
            record = _build_diag_record(resp, text, elapsed_ms, parsed, parse_success, validated, diag_ctx, model, sp_hash, _usage_acc)
            _write_diag_log(diag_path, record)

        if parsed:
            _track_usage(resp, _usage_acc)
            return parsed
        return None

    return _call_with_retry(_attempt, model, max_retries)


def call_openai_direct(api_key, system_prompt, user_prompt, timeout=120, max_retries=3, _usage_acc=None, diag_path=None, diag_ctx=None, max_tokens=8192, thinking=False):
    """Public wrapper: call DeepSeek via OpenAI SDK with retry + proxy isolation."""
    return _call_openai(
        PROVIDER_CONFIG["deepseek"],
        api_key,
        system_prompt,
        user_prompt,
        timeout,
        max_retries,
        _usage_acc=_usage_acc,
        diag_path=diag_path,
        diag_ctx=diag_ctx,
        max_tokens=max_tokens,
        thinking=thinking,
    )


def call_llm(provider, api_key, system_prompt, user_prompt, timeout=120, max_retries=3, _usage_acc=None, diag_path=None, diag_ctx=None, max_tokens=8192, model=None, thinking=False):
    config = PROVIDER_CONFIG[provider]
    if model:
        config = {**config, "model": model}
    if config["sdk"] == "openai":
        return _call_openai(config, api_key, system_prompt, user_prompt, timeout, max_retries, _usage_acc=_usage_acc, diag_path=diag_path, diag_ctx=diag_ctx, max_tokens=max_tokens, thinking=thinking)
    elif config["sdk"] == "anthropic":
        return _call_anthropic(config, api_key, system_prompt, user_prompt, timeout, max_retries, _usage_acc=_usage_acc, diag_path=diag_path, diag_ctx=diag_ctx, max_tokens=max_tokens)
    raise ValueError(f"Unknown SDK type: {config['sdk']}")
