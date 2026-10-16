"""Qwen arbitration for benchmark verification.

Extracted from cross_audit.py — resolves disagreements between two auditors
(DS vs MM) using a senior expert arbitrator (Qwen).

Usage:
    .venv/bin/python -m mineru_benchmark.pipeline.cross_arbitrate \\
        --questions benchmark/dataset/core_v3.json \\
        --audit-by benchmark/dataset/audit_by_ds.json benchmark/dataset/audit_by_mimo.json \\
        --output benchmark/dataset/audit_arbitration.json
"""
from __future__ import annotations

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from mineru_benchmark.pipeline._audit_common import (
    _get_api_client, _sanitize_json_escapes,
    load_paper_registry, _load_paper_text,
    write_json_atomic, save_checkpoint, load_checkpoint, _resolve_path,
)

# ---------------------------------------------------------------------------
# Arbitration prompt (Qwen resolves DS↔MM disagreements)
# ---------------------------------------------------------------------------

ARBITRATION_SYSTEM_PROMPT = """You are a senior control science expert arbitrator. Two independent auditors disagree about a Q&A pair. Your job is to make the final judgment.

## Your Role
- Review both auditor verdicts carefully
- Check the answer against the source paper
- Make the final pass/reject decision

## Judgment Criteria
- **pass=true**: The answer is generally consistent with the paper. At least one auditor supports it with reasonable evidence. No clear contradiction with the paper.
- **pass=false**: The answer directly contradicts the paper or contains clear factual errors. One auditor correctly identified the problem.

## Key Principles
- Domain knowledge beyond the paper excerpt is acceptable
- For design questions (Dimension D): creative synthesis is expected
- When in doubt, lean toward pass — reject only for clear errors

Output strict JSON (no other text):
{
  "pass": true,
  "confidence": 0.85,
  "evidence": "Paper Section 3 states '...'. Auditor A correctly noted ...",
  "issues": ["Minor: could mention the stability condition"]
}
"""


def build_arbitration_user_prompt(
    question: str, answer: str, paper_text: str,
    audit_a: dict, audit_b: dict, auditor_a: str, auditor_b: str,
) -> str:
    """Build arbitration user prompt showing both auditor verdicts."""
    a_verdict = "PASS" if audit_a.get("pass") else "REJECT"
    b_verdict = "PASS" if audit_b.get("pass") else "REJECT"
    return f"""## Source Paper
{paper_text}

## Question
{question}

## Answer
{answer}

## Auditor A ({auditor_a}): {a_verdict} (confidence: {audit_a.get('confidence', 0):.2f})
Evidence: {audit_a.get('evidence', 'N/A')}
Issues: {audit_a.get('issues', [])}

## Auditor B ({auditor_b}): {b_verdict} (confidence: {audit_b.get('confidence', 0):.2f})
Evidence: {audit_b.get('evidence', 'N/A')}
Issues: {audit_b.get('issues', [])}

Please make the final judgment."""


# ---------------------------------------------------------------------------
# Core arbitration
# ---------------------------------------------------------------------------


def arbitrate_one(
    question: dict,
    paper_text: str,
    audit_a: dict,
    audit_b: dict,
    auditor_a: str,
    auditor_b: str,
    client,
    model: str,
) -> dict:
    """Arbitrate a disputed question using Qwen."""
    user_prompt = build_arbitration_user_prompt(
        question.get("question", ""),
        question.get("answer", ""),
        paper_text,
        audit_a, audit_b,
        auditor_a, auditor_b,
    )

    for attempt in range(3):
        try:
            kwargs = {
                "model": model,
                "messages": [
                    {"role": "system", "content": ARBITRATION_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": 0.1,
                "max_tokens": 1024,
                "response_format": {"type": "json_object"},
            }
            resp = client.chat.completions.create(**kwargs)
            content = resp.choices[0].message.content.strip()

            json_match = re.search(r'\{[\s\S]*"pass"[\s\S]*\}', content)
            if json_match:
                try:
                    result = json.loads(_sanitize_json_escapes(json_match.group(0)))
                except json.JSONDecodeError:
                    json_match = re.search(r'\{[^{}]*"pass"[^{}]*\}', content, re.DOTALL)
                    if json_match:
                        result = json.loads(_sanitize_json_escapes(json_match.group(0)))
                    else:
                        raise
            else:
                generic_match = re.search(r'\{[\s\S]+\}', content)
                if generic_match:
                    result = json.loads(_sanitize_json_escapes(generic_match.group(0)))
                else:
                    result = json.loads(_sanitize_json_escapes(content))
            return {
                "id": question.get("id", ""),
                "pass": result.get("pass", False),
                "confidence": float(result.get("confidence", 0.5)),
                "evidence": result.get("evidence", ""),
                "issues": result.get("issues", []),
                "raw_response": content,
            }
        except (json.JSONDecodeError, KeyError) as e:
            if attempt < 2:
                time.sleep(2 ** attempt)
                continue
            return {
                "id": question.get("id", ""),
                "pass": None, "confidence": 0, "evidence": "",
                "issues": [f"Parse error after 3 attempts: {e}"],
                "raw_response": locals().get('content', ''),
            }
        except Exception as e:
            if attempt < 2:
                time.sleep(3 ** attempt)
                continue
            return {
                "id": question.get("id", ""),
                "pass": None, "confidence": 0, "evidence": "",
                "issues": [f"API error: {e}"],
                "raw_response": "",
            }

    return {"id": question.get("id", ""), "pass": None, "confidence": 0, "evidence": "", "issues": ["Unknown error"]}


# ---------------------------------------------------------------------------
# Disputed ID loading
# ---------------------------------------------------------------------------


def load_disputed_ids(path: Path, audit_a: dict, audit_b: dict) -> list[str]:
    """Load disputed IDs from file, or compute from two audit results.

    Returns sorted list for deterministic iteration order (important for resume).
    """
    if path and path.exists():
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        ids = data.get("disputed_ids", [])
    else:
        # Compute: questions where one auditor passed, other rejected
        all_ids = set(audit_a.keys()) | set(audit_b.keys())
        ids = []
        for qid in all_ids:
            a = audit_a.get(qid, {})
            b = audit_b.get(qid, {})
            if a.get("pass") is True and b.get("pass") is False:
                ids.append(qid)
            elif a.get("pass") is False and b.get("pass") is True:
                ids.append(qid)
    return sorted(ids)


# ---------------------------------------------------------------------------
# Arbitration pipeline
# ---------------------------------------------------------------------------


def _arbitrate_load_inputs(args):
    """Load and validate all inputs for arbitration. Returns dict or None on error."""
    ds_audit_path = _resolve_path(args.audit_by[0])
    mm_audit_path = _resolve_path(args.audit_by[1])
    for p, label in [(ds_audit_path, "DS audit"), (mm_audit_path, "MM audit")]:
        if not p.exists():
            print(f"[ERROR] {label} file not found: {p}", flush=True)
            raise SystemExit(1)
    with open(ds_audit_path, encoding="utf-8") as f:
        ds_data = json.load(f)
    ds_results = {r["id"]: r for r in ds_data.get("results", []) if "id" in r}
    with open(mm_audit_path, encoding="utf-8") as f:
        mm_data = json.load(f)
    mm_results = {r["id"]: r for r in mm_data.get("results", []) if "id" in r}
    disputed_ids = load_disputed_ids(
        Path(args.disputed_ids) if args.disputed_ids else None,
        ds_results, mm_results,
    )
    print(f"[arbitrate] DS: {len(ds_results)} | MM: {len(mm_results)} | Disputed: {len(disputed_ids)}", flush=True)
    if not disputed_ids:
        print("[arbitrate] No disputed questions — nothing to arbitrate.", flush=True)
        return None
    questions_path = _resolve_path(args.questions)
    with open(questions_path, encoding="utf-8") as f:
        qdata = json.load(f)
    questions = qdata.get("questions", qdata if isinstance(qdata, list) else [])
    q_map = {q["id"]: q for q in questions if "id" in q}
    disputed_qs = [q_map[qid] for qid in disputed_ids if qid in q_map]
    print(f"[arbitrate] Loaded {len(disputed_qs)} disputed questions for arbitration", flush=True)
    registry_path = _resolve_path(args.registry)
    corpus_dir = Path(args.corpus)
    paper_registry = {}
    if registry_path.exists():
        paper_registry = load_paper_registry(registry_path)
    output_path = _resolve_path(args.output)
    results = []
    if args.resume:
        cp_path = output_path.with_suffix(".arb.checkpoint.json")
        results = load_checkpoint(cp_path)
        print(f"[arbitrate] Resumed {len(results)} completed arbitrations", flush=True)
    return {
        "ds_results": ds_results, "mm_results": mm_results,
        "disputed_qs": disputed_qs, "paper_registry": paper_registry,
        "corpus_dir": corpus_dir, "output_path": output_path,
        "results": results, "cp_path": output_path.with_suffix(".arb.checkpoint.json"),
    }


def run_arbitrate(args) -> int:
    """Run Qwen arbitration on disputed questions."""
    inputs = _arbitrate_load_inputs(args)
    if inputs is None:
        return 0  # no disputed questions

    disputed_qs = inputs["disputed_qs"]
    ds_results = inputs["ds_results"]
    mm_results = inputs["mm_results"]
    paper_registry = inputs["paper_registry"]
    corpus_dir = inputs["corpus_dir"]
    output_path = inputs["output_path"]
    results = inputs["results"]
    cp_path = inputs["cp_path"]
    max_q = args.max_questions if args.max_questions > 0 else len(disputed_qs)
    start_time = time.time()

    # ---- Concurrent arbitration path ----
    if args.workers > 1:
        _arbitrate_concurrent(args, disputed_qs, ds_results, mm_results, paper_registry, corpus_dir, output_path, results, max_q, start_time, cp_path)
    else:
        _arbitrate_serial(args, disputed_qs, ds_results, mm_results, paper_registry, corpus_dir, output_path, results, max_q, start_time, cp_path)

    return _arbitrate_write_output(output_path, results, start_time, cp_path)


def _arbitrate_concurrent(args, disputed_qs, ds_results, mm_results, paper_registry, corpus_dir,
                          output_path, results, max_q, start_time, cp_path):
    """Concurrent arbitration path with worker pool."""
    client, model = _get_api_client("qwen", args.model)
    if args.delay > 0:
        print(f"[arbitrate] WARNING: --delay={args.delay} ignored in concurrent mode", flush=True)
    print(f"[arbitrate] Concurrent mode: {args.workers} workers", flush=True)
    tasks: list[tuple[int, dict, str]] = []
    for i, q in enumerate(disputed_qs[:max_q]):
        if i < len(results):
            continue
        tasks.append((i, q, _load_paper_text(q, corpus_dir, paper_registry)))
    _ds_results = ds_results
    _mm_results = mm_results

    def _arb_task(task):
        i, q, paper_text = task
        qid = q.get("id", f"q{i}")
        ds_r = _ds_results.get(qid, {})
        mm_r = _mm_results.get(qid, {})
        if not paper_text:
            print(f"[arbitrate] q{i} ({qid}): NO PAPER TEXT", flush=True)
            return {"id": qid, "pass": None, "confidence": 0, "evidence": "",
                    "issues": ["Paper full_text.md not found"], "raw_response": ""}
        result = arbitrate_one(q, paper_text, ds_r, mm_r, "deepseek", "mimo", client, model)
        status = "\u2713" if result["pass"] else ("\u2717" if result["pass"] is False else "?")
        print(f"[arbitrate] q{i} ({qid}): DS={ds_r.get('pass')} MM={mm_r.get('pass')} \u2192 {status} conf={result['confidence']:.2f}", flush=True)
        return result

    from concurrent.futures import as_completed
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(_arb_task, t): t for t in tasks}
        batch: list[dict] = []
        for completed in as_completed(futures):
            try:
                batch.append(completed.result())
            except Exception as e:
                import traceback
                t = futures[completed]
                print(f"[arbitrate] Task q{t[0]} crashed: {e}", flush=True)
                traceback.print_exc()
                batch.append({"id": t[1].get("id", f"q{t[0]}"), "pass": None, "confidence": 0,
                              "evidence": "", "issues": [f"Worker crash: {e}"], "raw_response": ""})
            if len(batch) >= args.checkpoint_interval:
                results.extend(batch)
                batch = []
                save_checkpoint(cp_path, results, start_time)
                elapsed = time.time() - start_time
                rate = len(results) / elapsed * 60 if elapsed > 0 else 0
                remaining = (max_q - len(results)) / rate * 60 if rate > 0 else 0
                print(f"  [checkpoint] {len(results)}/{max_q} | {elapsed:.0f}s | ETA {remaining:.0f}s", flush=True)
    results.extend(batch)


def _arbitrate_serial(args, disputed_qs, ds_results, mm_results, paper_registry, corpus_dir,
                      output_path, results, max_q, start_time, cp_path):
    """Serial arbitration path."""
    client, model = _get_api_client("qwen", args.model)
    print("[arbitrate] Serial mode (use --workers N for concurrent)", flush=True)
    for i, q in enumerate(disputed_qs):
        if i < len(results):
            continue
        if i >= max_q:
            break
        qid = q.get("id", f"q{i}")
        ds_r = ds_results.get(qid, {})
        mm_r = mm_results.get(qid, {})
        paper_text = _load_paper_text(q, corpus_dir, paper_registry)
        if not paper_text:
            print(f"[arbitrate] q{i} ({qid}): NO PAPER TEXT \u2014 skipping", flush=True)
            results.append({
                "id": qid, "pass": None, "confidence": 0, "evidence": "",
                "issues": ["Paper full_text.md not found"], "raw_response": "",
            })
            continue
        print(f"[arbitrate] q{i} ({qid}): DS={ds_r.get('pass')} MM={mm_r.get('pass')}...", end=" ", flush=True)
        result = arbitrate_one(q, paper_text, ds_r, mm_r, "deepseek", "mimo", client, model)
        results.append(result)
        status = "\u2713" if result["pass"] else ("\u2717" if result["pass"] is False else "?")
        print(f"{status} conf={result['confidence']:.2f}", flush=True)
        if (i + 1) % args.checkpoint_interval == 0 or (i + 1) == max_q:
            save_checkpoint(cp_path, results, start_time)
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed * 60 if elapsed > 0 else 0
            remaining = (max_q - i - 1) / rate * 60 if rate > 0 else 0
            print(f"  [checkpoint] {i+1}/{max_q} | {elapsed:.0f}s | ETA {remaining:.0f}s", flush=True)
        if args.delay > 0:
            time.sleep(args.delay)


def _arbitrate_write_output(output_path, results, start_time, cp_path) -> int:
    """Write final arbitration output and clean up."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    elapsed = time.time() - start_time
    passed = sum(1 for r in results if r["pass"] is True)
    failed = sum(1 for r in results if r["pass"] is False)
    unknown = sum(1 for r in results if r["pass"] is None)
    write_json_atomic(output_path, {
        "meta": {
            "arbitrator": "qwen",
            "total_disputed": len(results),
            "passed": passed,
            "rejected": failed,
            "unknown": unknown,
            "elapsed_sec": round(elapsed, 0),
        },
        "results": results,
    })
    if cp_path.exists():
        cp_path.unlink()
    print(f"\n[arbitrate] Complete: {len(results)} arbitrated | \u2713{passed} \u2717{failed} ?{unknown}", flush=True)
    print(f"[arbitrate] Output: {output_path}", flush=True)
    return 0
