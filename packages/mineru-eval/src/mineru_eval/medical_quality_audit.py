"""Thin backward-compatible shell — medical quality audit entry point.

Delegates to control.quality_audit infrastructure with medical-specific defaults.
The medical quality audit follows the same 4-dimension pattern as control.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]

DEFAULT_INPUT = ROOT / "benchmark" / "dataset" / "medical_benchmark.json"
DEFAULT_OUTPUT = ROOT / "benchmark" / "eval" / "reports" / "medical_quality_audit.json"


def main():
    """Medical quality audit — reuses control audit infrastructure with medical prompt."""
    import argparse

    parser = argparse.ArgumentParser(
        description="医学域四维质量审计 — DS/MM 双 Provider 独立全量审计"
    )
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Medical benchmark JSON")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Audit output JSON")
    parser.add_argument("--provider", choices=["ds", "mm"], default="ds")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--sample", type=int, default=0, help="Sample N questions (0=all)")
    args = parser.parse_args()

    from mineru_eval.control.quality_audit import (
        AUDIT_SYSTEM_PROMPT,
        _get_deepseek_key,
        build_audit_prompt,
        compute_summary,
    )
    from mineru_eval._shared.json_utils import _extract_json_block
    from mineru_eval.client_factory import create_client
    from mineru_eval.utils import load_json, write_json_atomic

    print(f"[medical-quality-audit] provider={args.provider} input={args.input}")

    data = load_json(args.input)
    questions = data.get("questions", [])
    if args.sample > 0:
        import random
        random.seed(42)
        questions = random.sample(questions, min(args.sample, len(questions)))

    print(f"[medical-quality-audit] Auditing {len(questions)} questions")

    api_key = _get_deepseek_key()
    client, model_name, _ = create_client(api_key=api_key)

    import json

    MEDICAL_AUDIT_PROMPT = (
        "你是医学与控制科学跨领域资深命题审核专家。"
        "请对以下医学场景中控制理论题目的质量进行四维审计。\n\n"
        + AUDIT_SYSTEM_PROMPT.split("## 评分维度", 1)[-1]  # keep all scoring dims + format
    )

    results = []
    for i, q in enumerate(questions):
        qid = q.get("id", f"med-{i}")
        print(f"  [{i+1}/{len(questions)}] {qid}...", end=" ", flush=True)
        prompt = build_audit_prompt(q)  # single-arg, same as control
        try:
            resp = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": MEDICAL_AUDIT_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=1000,
            )
            raw = resp.choices[0].message.content.strip()
            json_str = _extract_json_block(raw)
            try:
                parsed = json.loads(json_str)
                results.append({"id": qid, **parsed})
                print("OK", flush=True)
            except (json.JSONDecodeError, TypeError):
                results.append({"id": qid, "error": "parse_failed"})
                print("PARSE_FAIL", flush=True)
        except Exception as e:
            results.append({"id": qid, "error": str(e)[:200]})
            print(f"ERROR: {e}", flush=True)

    report = {
        "domain": "medical",
        "provider": args.provider,
        "total_audited": len(results),
        "results": results,
        "summary": compute_summary(results),
    }
    write_json_atomic(args.output, report)
    print(f"\n[medical-quality-audit] Done. {len(results)} audited → {args.output}")


if __name__ == "__main__":
    main()
