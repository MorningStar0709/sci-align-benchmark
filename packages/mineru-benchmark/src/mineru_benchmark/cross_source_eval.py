"""D12: Cross-Source Evaluation — Sciverse vs arXiv Homogeneity Test.

Generates 50 questions from each source (arXiv corpus + Sciverse papers) using the
same prompt/distribution, then runs a 4-way judge matrix:
  - Ollama 3-model consensus (qwen3.5:4b / 9b / 35b → median)
  - DeepSeek v4-flash cloud baseline

Usage:
  # Step 2 only (question generation):
  python -m benchmark.sciverse_cross_source_eval --step generate --n-questions 50

  # Step 3 only (judge matrix):
  python -m benchmark.sciverse_cross_source_eval --step judge

  # Full run:
  python -m benchmark.sciverse_cross_source_eval --step all --n-questions 50
"""
import argparse
import asyncio
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]  # restored Phase 5

from mineru_benchmark.pipeline.build_benchmark import load_corpus, select_chunks, generate_api_questions
from mineru_eval.client_factory import create_client
from mineru_eval.judge import _parse_judge_result
from mineru_core.settings import PROVIDERS

OUTPUT_DIR = ROOT / "benchmark" / "eval" / "results"
SCIVERSE_QUESTIONS = OUTPUT_DIR / "cross_source_sciverse_questions.json"
ARXIV_QUESTIONS = OUTPUT_DIR / "cross_source_arxiv_questions.json"
CROSS_SOURCE_REPORT = OUTPUT_DIR / "sciverse_cross_source_eval.json"

AUDIT_JSON = ROOT / "benchmark" / "eval" / "results" / "sciverse_cross_modal_audit.json"

_VALID_SCORES = {0.0, 0.25, 0.5, 0.75, 1.0}
CONSENSUS_MODELS = ["qwen3.5:4b", "qwen3.5:9b", "qwen3.5:35b"]
BASELINE_JUDGE = {"base_url": PROVIDERS["deepseek"].base_url, "model": PROVIDERS["deepseek"].default_model}

FM_BLOCK = re.compile(r'\$\$(.+?)\$\$', re.DOTALL)
FM_INLINE = re.compile(r'(?<!\$)\$(?!\$)([^$]+?)(?<!\$)\$(?!\$)')


def _score_filter_paragraphs(text, max_chars):
    """Split, clean, and score paragraphs by formula density."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    scored = []
    for p in paragraphs:
        p = re.sub(r'^#+\s+.*$', '', p, flags=re.MULTILINE).strip()
        p = re.sub(r'\[.*?\]\(.*?\)', '', p)
        p = re.sub(r'!\[.*?\]\(.*?\)', '', p)
        p = re.sub(r'\|.*?\|', '', p)
        p = p.strip()
        if len(p) < 80:
            continue
        block_count = len(FM_BLOCK.findall(p))
        inline_count = len(FM_INLINE.findall(p))
        score = block_count * 5 + inline_count * 2
        scored.append((score, p[:max_chars]))
    scored.sort(key=lambda x: -x[0])
    return scored


def _build_chunk_entries(scored, doc_id, title, label, max_frags):
    """Build chunk dicts from scored paragraphs."""
    return [
        {
            "chunk_id": f"sciverse:{doc_id}:frag{i}",
            "text": text,
            "doc_id": doc_id,
            "source_label": label,
            "title": title,
            "estimated_tokens": len(text) // 3,
            "_source": "sciverse",
        }
        for i, (_, text) in enumerate(scored[:max_frags])
    ]


async def _pull_paper_text(client, doc_id):
    """Pull full text for a single Sciverse paper via paginated API."""
    all_text = ""
    offset = 0
    for _ in range(4):
        for attempt in range(4):
            try:
                content = await client.read_content(doc_id=doc_id, offset=offset, limit=6000)
                break
            except Exception as e:
                if "429" in str(e) and attempt < 3:
                    await asyncio.sleep((attempt + 1) * 3)
                    continue
                raise
        all_text += content.get("text", "")
        if not content.get("more", False) or offset >= 24000:
            break
        offset = content.get("next_offset", offset + 6000)
    return all_text


async def _pull_sciverse_chunks(client, audit, max_frags_per_paper=8, max_chars=2000):
    """Async core: iterate papers, pull text, score, build chunks."""
    chunks = []
    paper_list = list(audit["per_paper"].items())
    for pi, (label, paper) in enumerate(paper_list):
        doc_id = paper["doc_id"]
        title = paper.get("title", "")
        print(f"  Pulling text: {label} ({pi+1}/{len(paper_list)})...", end=" ", flush=True)

        try:
            all_text = await _pull_paper_text(client, doc_id)
            print(f"{len(all_text)} chars")
        except Exception as e:
            print(f"FAILED: {e}")
            continue

        if pi < len(paper_list) - 1:
            await asyncio.sleep(2)

        scored = _score_filter_paragraphs(all_text, max_chars)
        chunks.extend(_build_chunk_entries(scored, doc_id, title, label, max_frags_per_paper))
    return chunks



def extract_sciverse_chunks(max_frags_per_paper=8, max_chars=2000):
    import asyncio
    from sciverse import AgentToolsClient

    client = AgentToolsClient(token=os.environ.get("SCIVERSE_API_KEY", ""))
    with open(AUDIT_JSON, encoding="utf-8") as f:
        audit = json.load(f)

    return asyncio.run(_pull_sciverse_chunks(client, audit, max_frags_per_paper, max_chars))


def prepare_arxiv_chunks(n_questions=50):
    corpus_dir = ROOT / "corpus"
    docs, all_chunks = load_corpus(corpus_dir)
    selected = select_chunks(corpus_dir, docs, all_chunks, min_tokens=80, seed=42)
    print(f"[ArXiv] {len(selected)} candidate chunks from {len(all_chunks)} total")
    return selected, corpus_dir


def apply_fallback_questions(questions):
    for q in questions:
        if not q.get("question") and q.get("raw_response"):
            try:
                raw = q["raw_response"]
                json_start = raw.index("{")
                json_end = raw.rindex("}") + 1
                parsed = json.loads(raw[json_start:json_end])
                q["question"] = parsed.get("question", "")
                q["answer"] = parsed.get("answer", "")
            except (ValueError, KeyError, json.JSONDecodeError):
                pass
        q.pop("raw_response", None)
    return [q for q in questions if q.get("question") and q.get("answer")]


def step_generate(args):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    n = args.n_questions
    api_key = os.environ.get("DEEPSEEK_API_KEY", "")

    print("=" * 60)
    print(f"D12 Step 2: Question Generation ({n}x2)")
    print("  Provider: deepseek")
    print("=" * 60)

    # --- Sciverse side ---
    print(f"\n[Sciverse] Extracting chunks from {AUDIT_JSON}...")
    sv_chunks = extract_sciverse_chunks()
    print(f"[Sciverse] {len(sv_chunks)} formula-dense chunks")

    print(f"\n[Sciverse] Generating {n} questions via DeepSeek API...")
    sv_questions, sv_review = generate_api_questions(
        sv_chunks, limit=n, api_key=api_key, model="deepseek-v4-flash",
        corpus_dir=ROOT / "corpus", provider="deepseek",
        output_path=SCIVERSE_QUESTIONS, review_path=SCIVERSE_QUESTIONS.with_suffix(".review.json"),
        reset_every=15,
    )
    sv_questions = apply_fallback_questions(sv_questions)
    SCIVERSE_QUESTIONS.write_text(json.dumps(sv_questions, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[Sciverse] {len(sv_questions)} valid questions saved to {SCIVERSE_QUESTIONS}")

    # --- arXiv side ---
    print("\n[ArXiv] Sampling chunks from corpus...")
    arxiv_chunks, corpus_dir = prepare_arxiv_chunks(n_questions=n)

    print(f"\n[ArXiv] Generating {n} questions via DeepSeek API...")
    arxiv_questions, arxiv_review = generate_api_questions(
        arxiv_chunks, limit=n, api_key=api_key, model="deepseek-v4-flash",
        corpus_dir=corpus_dir, provider="deepseek",
        output_path=ARXIV_QUESTIONS, review_path=ARXIV_QUESTIONS.with_suffix(".review.json"),
        reset_every=15,
    )
    arxiv_questions = apply_fallback_questions(arxiv_questions)
    ARXIV_QUESTIONS.write_text(json.dumps(arxiv_questions, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ArXiv] {len(arxiv_questions)} valid questions saved to {ARXIV_QUESTIONS}")

    return sv_questions, arxiv_questions


def run_judge_single(question_text, answer_text, client, model_name, provider="openai"):
    JUDGE_SYSTEM = """你是一位控制科学评分专家。请对以下回答进行评分。

## 评分维度（总分 0-1）
- 准确性: 答案是否在数学和逻辑上正确
- 完整性: 是否覆盖问题要求的全部内容
- 清晰性: 表达是否清晰、无歧义

## 评分规则
- 1.0: 完全正确，无遗漏，表达清晰
- 0.5: 部分正确，存在小错误或遗漏
- 0.0: 完全错误或无关

请输出 JSON: {"score": <0.0/0.25/0.5/0.75/1.0>, "reason": "<简短理由>"}"""

    user_prompt = f"## 问题\n{question_text[:2000]}\n\n## 回答\n{answer_text[:2000]}"
    messages = [
        {"role": "system", "content": JUDGE_SYSTEM},
        {"role": "user", "content": user_prompt},
    ]

    try:
        if provider == "anthropic":
            response = client.messages.create(model=model_name, system=JUDGE_SYSTEM,
                                              messages=[{"role": "user", "content": user_prompt}],
                                              temperature=0.0, max_tokens=256)
            raw = ""
            for block in response.content:
                if hasattr(block, "text"):
                    raw += block.text
            if not raw.strip():
                return {"score": 0.0, "reason": "empty response"}
        else:
            response = client.chat.completions.create(
                model=model_name, messages=messages, temperature=0.0, max_tokens=256)
            raw = response.choices[0].message.content
    except Exception as e:
        return {"score": 0.0, "reason": f"api_error: {str(e)[:100]}"}

    return _parse_judge_result(raw, _VALID_SCORES)


def _phase_generate_answers(all_questions, gen_client, gen_model_name):
    """Phase A: Generate all answers via Ollama 9B (serial)."""
    print(f"\n[Answer Gen] Generating {len(all_questions)} answers via Ollama qwen3.5:9b...")
    answer_map = {}
    for qi, (source, q) in enumerate(all_questions):
        question_text = q.get("question", "")
        if not question_text:
            answer_map[qi] = ""
            continue
        try:
            resp = gen_client.chat.completions.create(
                model=gen_model_name,
                messages=[
                    {"role": "system", "content": "你是控制科学专家。请准确回答以下问题，给出最终答案。"},
                    {"role": "user", "content": question_text},
                ],
                temperature=0.0, max_tokens=512,
            )
            answer_map[qi] = resp.choices[0].message.content
        except Exception as e:
            answer_map[qi] = f"[gen_error: {str(e)[:80]}]"
        if (qi + 1) % 10 == 0:
            print(f"  [{qi+1}/{len(all_questions)}] generated")
    return answer_map


def _phase_deepseek_judge(all_questions, ds_client, ds_model_name, ds_provider):
    """Phase B: DeepSeek baseline judge on gold answers (4-way concurrent)."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    print("\n[DeepSeek Judge] Concurrent evaluation (4 workers)...")
    ds_results = [None] * len(all_questions)

    def _judge_ds(idx):
        source, q = all_questions[idx]
        question_text = q.get("question", "")
        answer_text = q.get("answer", "")
        if not question_text or not answer_text:
            return idx, {"score": 0.0, "reason": "empty"}
        return idx, run_judge_single(question_text, answer_text,
                                     ds_client, ds_model_name, ds_provider)

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(_judge_ds, i): i for i in range(len(all_questions))}
        for future in as_completed(futures):
            idx, result = future.result()
            ds_results[idx] = result
            done = sum(1 for r in ds_results if r is not None)
            if done % 10 == 0:
                print(f"  [DS] {done}/{len(all_questions)} ({done*100//len(all_questions)}%)")
    return ds_results


def _phase_ollama_consensus(all_questions, answer_map):
    """Phase C: Ollama 3-model consensus — batch by model, not interleaved."""
    print("\n[Ollama Consensus] Batched by model (1 model × 60 questions, then switch)...")
    all_ollama = {m: [None] * len(all_questions) for m in CONSENSUS_MODELS}
    for mi, ollama_model in enumerate(CONSENSUS_MODELS):
        oc, om, op = create_client(base_url=f"http://localhost:{os.environ.get('LLAMA_CHAT_PORT', '8080')}/v1", model=ollama_model)
        print(f"  [{ollama_model}] Running {len(all_questions)} questions...")
        for qi in range(len(all_questions)):
            question_text = all_questions[qi][1].get("question", "")
            gen_answer = answer_map.get(qi, "")
            if not question_text:
                continue
            try:
                res = run_judge_single(question_text, gen_answer, oc, om, op)
                all_ollama[ollama_model][qi] = res.get("score", 0.0)
            except Exception:
                all_ollama[ollama_model][qi] = None
            if (qi + 1) % 20 == 0:
                print(f"    [{ollama_model}] {qi+1}/{len(all_questions)}")
        print(f"    [{ollama_model}] done ({mi+1}/{len(CONSENSUS_MODELS)})")
    return all_ollama


def _build_judge_report(all_questions, answer_map, ds_results, all_ollama, sv_questions, arxiv_questions, t_start):
    """Compute consensus, build results, cross-source comparison, and write report."""
    import threading
    results = []
    results_lock = threading.Lock()

    # Compute per-question median and build results
    print("\n[Consensus] Computing median per question...")
    for qi in range(len(all_questions)):
        source, q = all_questions[qi]
        qid = q.get("id", f"{source}_{qi}")
        question_text = q.get("question", "")
        answer_text = q.get("answer", "")
        gen_answer = answer_map.get(qi, "")
        if not question_text:
            continue

        ollama_scores = {}
        for m in CONSENSUS_MODELS:
            ollama_scores[m] = all_ollama[m][qi]
        valid_scores = [s for s in ollama_scores.values() if s is not None]
        consensus = float(np.median(valid_scores)) if valid_scores else 0.0

        entry = {
            "source": source,
            "question_id": qid,
            "question": question_text[:300],
            "gold_answer": answer_text[:300],
            "generated_answer": gen_answer[:300],
            "deepseek_baseline": ds_results[qi],
            "ollama_consensus": {"scores": ollama_scores, "median": round(consensus, 4)},
        }
        with results_lock:
            results.append(entry)

    # --- Cross-source comparison ---
    sv_entries = [r for r in results if r["source"] == "sciverse"]
    arxiv_entries = [r for r in results if r["source"] == "arxiv"]

    def stats(entries, key="deepseek_baseline"):
        scores = []
        for e in entries:
            v = e.get(key, {})
            if isinstance(v, dict):
                scores.append(v.get("score", 0))
            else:
                scores.append(0)
        if not scores:
            return {"mean": 0, "median": 0, "n": 0}
        return {"mean": round(float(np.mean(scores)), 4),
                "median": round(float(np.median(scores)), 4), "n": len(scores)}

    sv_ds = stats(sv_entries)
    arxiv_ds = stats(arxiv_entries)
    sv_cons = stats(sv_entries, key="ollama_consensus")
    arxiv_cons = stats(arxiv_entries, key="ollama_consensus")
    delta_ds = round(abs(sv_ds["mean"] - arxiv_ds["mean"]), 4) if sv_ds["n"] and arxiv_ds["n"] else 0

    report = {
        "analysis": "cross_source_evaluation",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "config": {"n_questions_target": len(sv_questions)},
        "summary": {
            "sciverse": {"n_questions": len(sv_questions), "deepseek_baseline": sv_ds, "ollama_consensus": sv_cons},
            "arxiv": {"n_questions": len(arxiv_questions), "deepseek_baseline": arxiv_ds, "ollama_consensus": arxiv_cons},
            "delta_deepseek_mean": delta_ds,
            "delta_pct": round(delta_ds / max(arxiv_ds["mean"], 0.001) * 100, 1),
            "comparable": delta_ds <= 0.05,
        },
        "per_question": results,
    }

    CROSS_SOURCE_REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n{'=' * 60}")
    print(f"D12 COMPLETE ({time.time()-t_start:.0f}s)")
    print(f"  Sciverse DS: {sv_ds['mean']:.3f} (n={sv_ds['n']})")
    print(f"  ArXiv DS:    {arxiv_ds['mean']:.3f} (n={arxiv_ds['n']})")
    print(f"  Delta:       {delta_ds:.3f} ({'<=5% comparable' if delta_ds <= 0.05 else '>5% drift'})")
    print(f"  Report:      {CROSS_SOURCE_REPORT}")
    print(f"{'=' * 60}")

    return report


def step_judge(args):
    """D12 Step 3: 4-way Judge matrix."""
    print("=" * 60)
    print("D12 Step 3: 4-Way Judge Matrix")
    print("  Answer Gen: qwen3.5:9b (Ollama API)")
    print("  Consensus:  qwen3.5:4b / 9b / 35b (Ollama API)")
    print("  Baseline:   deepseek-v4-flash (OpenAI SDK, 4-way concurrent)")
    print("=" * 60)

    if not SCIVERSE_QUESTIONS.exists() or not ARXIV_QUESTIONS.exists():
        print("ERROR: Question files not found. Run --step generate first.")
        sys.exit(1)

    sv_questions = json.loads(SCIVERSE_QUESTIONS.read_text(encoding="utf-8"))
    arxiv_questions = json.loads(ARXIV_QUESTIONS.read_text(encoding="utf-8"))
    print(f"\n[Load] Sciverse: {len(sv_questions)} questions, ArXiv: {len(arxiv_questions)} questions")

    all_questions = [("sciverse", q) for q in sv_questions] + [("arxiv", q) for q in arxiv_questions]

    gen_client, gen_model_name, _ = create_client(
        base_url=f"http://localhost:{os.environ.get('LLAMA_CHAT_PORT', '8080')}/v1", model="qwen3.5-9b")
    ds_client, ds_model_name, ds_provider = create_client(
        base_url=PROVIDERS["deepseek"].base_url, model=PROVIDERS["deepseek"].default_model)

    t_start = time.time()

    answer_map = _phase_generate_answers(all_questions, gen_client, gen_model_name)
    ds_results = _phase_deepseek_judge(all_questions, ds_client, ds_model_name, ds_provider)
    all_ollama = _phase_ollama_consensus(all_questions, answer_map)
    report = _build_judge_report(all_questions, answer_map, ds_results, all_ollama,
                                  sv_questions, arxiv_questions, t_start)
    return report


def main():
    parser = argparse.ArgumentParser(description="D12: Cross-Source Evaluation")
    parser.add_argument("--step", choices=["generate", "judge", "all"], default="all")
    parser.add_argument("--n-questions", type=int, default=50)
    args = parser.parse_args()

    if args.step in ("generate", "all"):
        step_generate(args)
    if args.step in ("judge", "all"):
        step_judge(args)


if __name__ == "__main__":
    main()
