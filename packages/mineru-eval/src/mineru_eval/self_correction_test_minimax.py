"""
Self-Correction Test: Task 5D — MiniMax-M2.5 跨模型自修正

跨模型验证三号通道：用 MiniMax-M2.5 (Anthropic API) 修正低分题，
DeepSeek-v4-flash Judge 保持评分一致性。

Pipeline:
  1. 加载 self_correction_candidates_minimax.json（20 题，A/B/C/D 各 5）
  2. 构造 feedback prompt → 调用 MiniMax Anthropic API 获取修正答案
  3. 对修正答案调用 DeepSeek Judge 评分
  4. 输出 self_correction_minimax_results.json + trajectory.json

并发: ThreadPoolExecutor(max_workers=4)，修正为 MiniMax Anthropic，Judge 为 DeepSeek API。
"""

import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]

import mineru_core  # noqa: F401 — trigger _load_dotenv() to load .env API keys

from mineru_core.settings import PROVIDERS

import httpx
from anthropic import Anthropic
from openai import OpenAI, DefaultHttpxClient

from mineru_eval.utils import load_json, write_json_atomic

# ---------------------------------------------------------------------------
# Constants — MiniMax Anthropic for correction
# ---------------------------------------------------------------------------
MINIMAX_API_KEY = os.environ.get("MINIMAX_API_KEY", "")
MINIMAX_BASE_URL = PROVIDERS["minimax"].base_url
MINIMAX_MODEL = "MiniMax-M2.5-highspeed"

DEEPSEEK_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = PROVIDERS["deepseek"].base_url
DEEPSEEK_MODEL = "deepseek-v4-flash"

CANDIDATES_PATH = ROOT / "benchmark" / "eval" / "results" / "self_correction_candidates_minimax.json"
RESULTS_PATH = ROOT / "benchmark" / "eval" / "results" / "self_correction_minimax_results.json"
TRAJECTORY_PATH = ROOT / "benchmark" / "eval" / "results" / "self_correction_minimax_trajectory.json"

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
# D dimension default rubric
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
# API clients
# ---------------------------------------------------------------------------
def _make_anthropic_client():
    return Anthropic(
        api_key=MINIMAX_API_KEY,
        base_url=MINIMAX_BASE_URL,
        timeout=120,
        http_client=httpx.Client(proxy=None, trust_env=False),
    )


def _make_ds_client():
    from mineru_eval.client_factory import wrap_client
    return wrap_client(OpenAI(
        api_key=DEEPSEEK_KEY,
        base_url=DEEPSEEK_BASE_URL,
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
    client = _make_anthropic_client()
    resp = client.messages.create(
        model=MINIMAX_MODEL,
        max_tokens=MAX_TOKENS_CORRECT,
        system=FEEDBACK_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": [{"type": "text", "text": user_prompt}]}],
    )
    text = ""
    for block in resp.content:
        if hasattr(block, 'text') and block.text:
            text += block.text
    return text.strip()


def call_judge(question, expected_answer, model_answer, dimension, rubric):
    from mineru_eval.judge import AScorer, BScorer, GenericScorer, RubricScorer

    client = _make_ds_client()
    if dimension == "A":
        return AScorer.judge(question, expected_answer, model_answer, client, DEEPSEEK_MODEL)
    if dimension == "B":
        return BScorer.judge(question, expected_answer, model_answer, client, DEEPSEEK_MODEL)
    if dimension == "D" and rubric:
        return RubricScorer.judge(question, expected_answer, model_answer, client, DEEPSEEK_MODEL, rubric=rubric)
    if dimension in ("C", "D"):
        return GenericScorer.judge(question, expected_answer, model_answer, client, DEEPSEEK_MODEL)
    return {"score": 0.0, "reason": "Unknown dimension", "issues": ["unknown dimension"]}


# ---------------------------------------------------------------------------
# Per-item worker
# ---------------------------------------------------------------------------
def process_one(candidate, rubric):
    qid = candidate["qid"]
    dimension = candidate["dimension"]
    question_text = candidate["question_text"]
    expected_answer = candidate["expected_answer"]
    original_score = candidate["judge_score"]
    judge_reason = candidate["judge_reason"]
    original_answer = candidate["original_answer"]

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
        f"{original_score:.2f} -> {corrected_score:.2f} "
        f"(Δ={delta:+.2f}) "
        f"t_correct={t_correct:.1f}s",
        flush=True,
    )

    judge_reason_corrected = result.get("reason", "")
    trajectory_record = {
        "qid": qid,
        "dimension": dimension,
        "difficulty_level": candidate.get("difficulty_level", "?"),
        "original_score": original_score,
        "corrected_score": corrected_score,
        "delta": delta,
        "judge_reason_original": judge_reason,
        "judge_reason_corrected": judge_reason_corrected,
        "original_answer_length": len(original_answer),
        "corrected_answer_length": len(corrected_answer),
    }

    return {
        "result": {
            "qid": qid,
            "dimension": dimension,
            "original_score": original_score,
            "original_answer": original_answer,
            "corrected_answer": corrected_answer,
            "corrected_score": corrected_score,
            "delta": delta,
            "judge_reason_original": judge_reason,
            "judge_reason_corrected": judge_reason_corrected,
            "difficulty_level": candidate.get("difficulty_level", "?"),
            "complete": True,
        },
        "trajectory": trajectory_record,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    if not MINIMAX_API_KEY:
        print("ERROR: MINIMAX_API_KEY environment variable not set.", flush=True)
        return

    candidates_data = load_json(CANDIDATES_PATH)
    candidates = candidates_data.get("candidates", [])
    if not candidates:
        print(f"No candidates found in {CANDIDATES_PATH}", flush=True)
        return

    rubric = default_rubric()

    print(f"Loaded {len(candidates)} candidates from {CANDIDATES_PATH}", flush=True)
    dims = {}
    for c in candidates:
        dims[c["dimension"]] = dims.get(c["dimension"], 0) + 1
    print(f"  Distribution: {dims}", flush=True)
    print(f"  Correction model: {MINIMAX_MODEL} (MiniMax Anthropic)", flush=True)
    print(f"  Judge model: {DEEPSEEK_MODEL} (DeepSeek, consistent)", flush=True)
    print(f"  Workers: {MAX_WORKERS} threads", flush=True)

    t_start = time.time()
    results_list = []
    trajectories = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_map = {
            executor.submit(process_one, c, rubric if c["dimension"] == "D" else None): c["qid"]
            for c in candidates
        }
        for future in as_completed(future_map):
            output = future.result()
            results_list.append(output["result"])
            trajectories.append(output["trajectory"])

    elapsed = time.time() - t_start

    n_complete = sum(1 for r in results_list if r.get("complete"))
    n_failed = sum(1 for r in results_list if not r.get("complete"))
    delta_pos = sum(1 for r in results_list if r.get("delta", 0) > 0)
    delta_neg = sum(1 for r in results_list if r.get("delta", 0) <= 0)

    results_list.sort(key=lambda r: r["qid"])

    output_results = {
        "meta": {
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "source_candidates": str(CANDIDATES_PATH),
            "correction_model": MINIMAX_MODEL,
            "judge_model": DEEPSEEK_MODEL,
            "max_workers": MAX_WORKERS,
            "elapsed_sec": round(elapsed, 2),
            "total": len(results_list),
            "complete": n_complete,
            "failed": n_failed,
            "delta_positive": delta_pos,
            "delta_non_positive": delta_neg,
        },
        "results": results_list,
    }

    write_json_atomic(RESULTS_PATH, output_results)

    trajectories.sort(key=lambda t: t["qid"])
    output_trajectory = {
        "meta": {
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "correction_model": MINIMAX_MODEL,
            "judge_model": DEEPSEEK_MODEL,
            "total": len(trajectories),
        },
        "trajectories": trajectories,
    }
    write_json_atomic(TRAJECTORY_PATH, output_trajectory)

    print(f"\n{'='*60}", flush=True)
    print(f"Done: {n_complete}/{len(results_list)} complete, {n_failed} failed", flush=True)
    print(f"  Δ>0: {delta_pos}, Δ≤0: {delta_neg}", flush=True)
    print(f"  Elapsed: {elapsed:.1f}s", flush=True)
    print(f"  Results: {RESULTS_PATH}", flush=True)
    print(f"  Trajectory: {TRAJECTORY_PATH}", flush=True)

    dim_stats = {}
    for r in results_list:
        d = r["dimension"]
        if d not in dim_stats:
            dim_stats[d] = {"total": 0, "pos": 0, "scores": []}
        dim_stats[d]["total"] += 1
        if r.get("delta", 0) > 0:
            dim_stats[d]["pos"] += 1
        dim_stats[d]["scores"].append((r["original_score"], r["corrected_score"]))

    print("\n  Per-dimension correction rate:", flush=True)
    for d in ["A", "B", "C", "D"]:
        s = dim_stats.get(d, {"total": 0, "pos": 0})
        rate = s["pos"] / s["total"] * 100 if s["total"] else 0
        print(f"    {d}: {s['pos']}/{s['total']} ({rate:.0f}%)", flush=True)


if __name__ == "__main__":
    main()
