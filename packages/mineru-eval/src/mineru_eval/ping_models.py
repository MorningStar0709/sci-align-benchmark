"""Lightweight target-model connectivity checks.

This script sends a tiny prompt to target models only. It does not call the
judge model and does not consume benchmark questions.
"""

import argparse
import json
import os
import time


# ROOT = Path(__file__).resolve().parents[2]  # Phase 5: sys.path hack removed, use mineru_benchmark.agent.utils.ROOT instead

from mineru_core.settings import PROVIDERS
from mineru_eval.client_factory import create_client, resolve_api_key
from mineru_eval.evaluate import call_target_model
from mineru_eval.utils import write_json_atomic


MODEL_PRESETS = {
    "deepseek-v4-flash": ("deepseek-v4-flash", PROVIDERS["deepseek"].base_url),
    "deepseek-v4-pro": ("deepseek-v4-pro", PROVIDERS["deepseek"].base_url),
    "mimo-v2.5-pro": ("mimo-v2.5-pro", PROVIDERS["mimo"].base_url),
    "mimo-v2.5": ("mimo-v2.5", PROVIDERS["mimo"].base_url),
    "mimo-v2-pro": ("mimo-v2-pro", PROVIDERS["mimo"].base_url),
    "mimo-v2-flash": ("mimo-v2-flash", PROVIDERS["mimo"].base_url),
    "qwen3.5-9b": ("qwen3.5-9b", f"http://localhost:{os.environ.get('LLAMA_CHAT_PORT', '8080')}/v1"),
    "qwen3.7-plus": ("qwen3.7-plus", PROVIDERS["qwen"].base_url),
    "qwen3.5-plus": ("qwen3.5-plus", PROVIDERS["qwen"].base_url),
}


def ping(name, timeout):
    model, base_url = MODEL_PRESETS[name]
    api_key = resolve_api_key(base_url)
    start = time.time()
    try:
        client, model_name, provider_kind = create_client(api_key, base_url, model)
        answer = call_target_model("请只回答 OK。", client, model_name, provider_kind=provider_kind)
        elapsed = time.time() - start
        return {
            "name": name,
            "model": model,
            "base_url": base_url,
            "ok": True,
            "seconds": round(elapsed, 2),
            "answer_preview": answer[:80],
        }
    except Exception as exc:
        elapsed = time.time() - start
        return {
            "name": name,
            "model": model,
            "base_url": base_url,
            "ok": False,
            "seconds": round(elapsed, 2),
            "error": str(exc)[:500],
        }


def main():
    parser = argparse.ArgumentParser(description="Ping target models without judge calls.")
    parser.add_argument("--models", default=",".join(MODEL_PRESETS), help="Comma-separated preset names")
    parser.add_argument("--output", default="", help="Optional JSON output path")
    parser.add_argument("--timeout", type=float, default=30.0, help="Reserved for reporting; clients use configured SDK timeouts")
    args = parser.parse_args()

    names = [item.strip() for item in args.models.split(",") if item.strip()]
    results = []
    for name in names:
        if name not in MODEL_PRESETS:
            results.append({"name": name, "ok": False, "seconds": 0, "error": "unknown preset"})
            continue
        print(f"PING {name}...", flush=True)
        result = ping(name, args.timeout)
        results.append(result)
        status = "OK" if result["ok"] else "FAIL"
        print(f"{status} {name} {result['seconds']}s", flush=True)

    payload = {
        "results": results,
        "ok_count": sum(1 for item in results if item.get("ok")),
        "total": len(results),
    }
    if args.output:
        write_json_atomic(args.output, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if payload["ok_count"] != payload["total"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
