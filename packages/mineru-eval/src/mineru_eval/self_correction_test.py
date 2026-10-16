"""
Self-Correction Test: Phase 5B — API 修正作答 + Judge 重评

从 5A 产出的低分候选题目中，构造 feedback prompt 引导模型修正答案，
再用同一 Judge 对修正答案评分，记录原始/修正分数对比与 Δ。

Pipeline:
  1. 加载 self_correction_candidates.json（20 条，A/B/C/D 各 5 题）
  2. 构造 feedback prompt → 调用 DS API 获取修正答案
  3. 对修正答案调用同维度 Judge 评分
  4. 输出 self_correction_results.json

并发: ThreadPoolExecutor(max_workers=4)，两轮 API 调用均为 DeepSeek-v4-flash。
"""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]

import mineru_core  # noqa: F401 — trigger _load_dotenv()
from mineru_core.settings import PROVIDERS

from openai import OpenAI, DefaultHttpxClient

from mineru_eval.utils import load_json, write_json_atomic

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEEPSEEK_KEY = os.getenv("DEEPSEEK_API_KEY", "")
BASE_URL = PROVIDERS["deepseek"].base_url
MODEL = PROVIDERS["deepseek"].default_model

CANDIDATES_PATH = ROOT / "benchmark" / "eval" / "results" / "self_correction_candidates.json"
RESULTS_PATH = ROOT / "benchmark" / "eval" / "results" / "self_correction_results.json"

MAX_WORKERS = 4
MAX_TOKENS_CORRECT = 4000

# ---------------------------------------------------------------------------
# Feedback prompt template
# ---------------------------------------------------------------------------
FEEDBACK_SYSTEM_PROMPT = """你是一个控制科学领域的专家。请根据反馈意见修正你之前的回答。
仔细阅读扣分理由，针对性地改进你的答案。回答要准确、专业、完整。
如果问题涉及公式推导，请逐步写出推理过程。
如果问题涉及概念解释，请给出精确定义和必要的数学表达式。"""

FEEDBACK_USER_TEMPLATE = """你之前的回答得分 {score}，扣分理由：{reason}

请基于此反馈重新作答以下问题：

{question}"""

# ---------------------------------------------------------------------------
# D dimension default rubric (mirrors build_benchmark.py)
# ---------------------------------------------------------------------------
def default_rubric():
    return {
        "feasibility": {"max_score": 1, "weight": 0.2, "description": "控制方案是否具有工程可实施性"},
        "method_choice": {"max_score": 1, "weight": 0.2, "description": "控制方法选择是否适合该问题"},
        "completeness": {"max_score": 1, "weight": 0.2, "description": "是否覆盖建模、设计、分析与验证流程"},
        "innovation": {"max_score": 1, "weight": 0.2, "description": "是否包含有价值的设计改进或工程权衡"},
        "clarity": {"max_score": 1, "weight": 0.2, "description": "数学表达和文字说明是否清晰准确"},
    }


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------
def _make_client():
    from mineru_eval.client_factory import wrap_client
    return wrap_client(OpenAI(
        api_key=DEEPSEEK_KEY,
        base_url=BASE_URL,
        timeout=120,
        max_retries=0,
        http_client=DefaultHttpxClient(proxy=None, trust_env=False),
    ))


def call_correction(question, judge_score, judge_reason):
    user_prompt = FEEDBACK_USER_TEMPLATE.format(
        score=judge_score,
        reason=judge_reason,
        question=question,
    )
    client = _make_client()
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": FEEDBACK_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.3,
        max_tokens=MAX_TOKENS_CORRECT,
    )
    return (response.choices[0].message.content or "").strip()


def call_judge(question, expected_answer, model_answer, dimension, rubric):
    from mineru_eval.judge import AScorer, BScorer, GenericScorer, RubricScorer

    client = _make_client()
    if dimension == "A":
        return AScorer.judge(question, expected_answer, model_answer, client, MODEL)
    if dimension == "B":
        return BScorer.judge(question, expected_answer, model_answer, client, MODEL)
    if dimension == "D" and rubric:
        return RubricScorer.judge(question, expected_answer, model_answer, client, MODEL, rubric=rubric)
    if dimension in ("C", "D"):
        return GenericScorer.judge(question, expected_answer, model_answer, client, MODEL)
    return {"score": 0.0, "reason": "Unknown dimension", "issues": ["unknown dimension"]}


# ---------------------------------------------------------------------------
# Per-item worker (runs in thread pool)
# ---------------------------------------------------------------------------
def process_one(candidate, rubric):
    qid = candidate["qid"]
    dimension = candidate["dimension"]
    question_text = candidate["question_text"]
    expected_answer = candidate["expected_answer"]
    original_score = candidate["judge_score"]
    judge_reason = candidate["judge_reason"]
    original_answer = candidate["original_answer"]

    # Step 1: correction
    t0 = time.time()
    try:
        corrected_answer = call_correction(question_text, original_score, judge_reason)
    except Exception as exc:
        return {
            "qid": qid,
            "dimension": dimension,
            "original_score": original_score,
            "original_answer": original_answer,
            "corrected_answer": "",
            "corrected_score": 0.0,
            "delta": -original_score,
            "error": f"correction_failed: {exc}",
            "complete": False,
        }
    t_correct = time.time() - t0

    # Step 2: judge corrected answer
    try:
        result = call_judge(question_text, expected_answer, corrected_answer, dimension, rubric)
    except Exception as exc:
        return {
            "qid": qid,
            "dimension": dimension,
            "original_score": original_score,
            "original_answer": original_answer,
            "corrected_answer": corrected_answer,
            "corrected_score": 0.0,
            "delta": -original_score,
            "error": f"judge_failed: {exc}",
            "complete": False,
        }

    corrected_score = result.get("score", 0.0)
    delta = round(corrected_score - original_score, 4)

    print(
        f"[{qid}] ({dimension}) "
        f"{original_score:.2f} → {corrected_score:.2f} "
        f"(Δ={delta:+.2f}) "
        f"t_correct={t_correct:.1f}s",
        flush=True,
    )

    return {
        "qid": qid,
        "dimension": dimension,
        "original_score": original_score,
        "original_answer": original_answer,
        "corrected_answer": corrected_answer,
        "corrected_score": corrected_score,
        "delta": delta,
        "judge_reason_original": judge_reason,
        "judge_reason_corrected": result.get("reason", ""),
        "difficulty_level": candidate.get("difficulty_level", "?"),
        "complete": True,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    candidates_data = load_json(CANDIDATES_PATH)
    candidates = candidates_data.get("candidates", [])
    if not candidates:
        print("No candidates found.", flush=True)
        return

    rubric = default_rubric()

    print(f"Loaded {len(candidates)} candidates from {CANDIDATES_PATH}", flush=True)
    dims = {}
    for c in candidates:
        dims[c["dimension"]] = dims.get(c["dimension"], 0) + 1
    print(f"  Distribution: {dims}", flush=True)
    print(f"  Workers: {MAX_WORKERS} threads", flush=True)

    t_start = time.time()
    results = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_map = {
            executor.submit(process_one, c, rubric if c["dimension"] == "D" else None): c["qid"]
            for c in candidates
        }
        for future in as_completed(future_map):
            results.append(future.result())

    elapsed = time.time() - t_start

    n_complete = sum(1 for r in results if r.get("complete"))
    n_failed = sum(1 for r in results if not r.get("complete"))
    delta_pos = sum(1 for r in results if r.get("delta", 0) > 0)
    delta_neg = sum(1 for r in results if r.get("delta", 0) <= 0)

    output = {
        "meta": {
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "source_candidates": str(CANDIDATES_PATH),
            "correction_model": MODEL,
            "judge_model": MODEL,
            "max_workers": MAX_WORKERS,
            "elapsed_sec": round(elapsed, 2),
            "total": len(results),
            "complete": n_complete,
            "failed": n_failed,
            "delta_positive": delta_pos,
            "delta_non_positive": delta_neg,
        },
        "results": results,
    }

    write_json_atomic(RESULTS_PATH, output)
    print(f"\n{'='*60}", flush=True)
    print(f"Done: {n_complete}/{len(results)} complete, {n_failed} failed", flush=True)
    print(f"  Δ>0: {delta_pos}, Δ≤0: {delta_neg}", flush=True)
    print(f"  Elapsed: {elapsed:.1f}s", flush=True)
    print(f"  Output: {RESULTS_PATH}", flush=True)


if __name__ == "__main__":
    main()
