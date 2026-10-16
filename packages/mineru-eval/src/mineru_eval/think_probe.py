"""思考模式探针：35B think=true vs think=false Judge 评分对照
复现: & python benchmark/eval/think_probe.py  (with myenv Python)
输出: benchmark/eval/results/think_probe_35b_10queries.json
"""
import httpx
import json
import os
import random
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]

B_SYSTEM_PROMPT = """你是一个控制科学推理步骤评分员。你的任务是判断模型给出的推理步骤与标准推理链在逻辑上是否一致。

## 评分规则
- **1.0 分（全部步骤正确）**：推理步骤完整、逻辑清晰，每一步推理都正确，结论正确。
- **0.7 分（步骤完整但结论有错）**：推理链条完整，步骤基本正确，但最终结论出现错误。
- **0.5 分（部分步骤正确，≥50%）**：至少一半以上的推理步骤正确，但存在部分步骤错误或缺失。
- **0.2 分（部分步骤正确，<50%）**：有少量正确步骤，但大部分推理错误或缺失。
- **0.0 分（无有效步骤）**：没有有效的推理步骤，或回答完全无关。

## 评判原则
- 关注推理逻辑的一致性，而非措辞的精确匹配
- 如果模型使用了与标准答案不同的推理路径但逻辑正确，应视为有效步骤
- 宽松评判：只要推理方向正确、关键步骤存在，即视为正确步骤

请严格按照以下 JSON 格式输出，不要输出任何其他内容：
{"score": <分数>, "reason": "<评分理由>", "issues": ["<问题1>", ...]}
分数只能是 0.0、0.2、0.5、0.7、1.0 之一。如果没有问题，issues 为空列表 []。"""

B_USER_PROMPT_TEMPLATE = """## 问题
{question}

## 标准推理步骤
{expected_answer}

## 模型回答的推理步骤
{model_answer}

请根据评分规则对模型回答的推理步骤进行评分。"""

D_SYSTEM_PROMPT = """你是一个控制科学答案评分员。判断模型回答与标准答案是否对应同一结论。
评分规则：
- 1.0: 完全正确，核心结论一致
- 0.75: 基本正确，有少量表述差异
- 0.5: 部分正确，存在明显遗漏
- 0.25: 有正确元素但整体不完整
- 0.0: 错误或空白

输出格式：{"score": <分数>, "reason": "<理由>", "issues": ["<问题>"]}
分数只能是 0.0, 0.25, 0.5, 0.75, 1.0 之一。"""

NUM_PREDICT = 16384


def _extract_json_block(text):
    if "```json" in text:
        start = text.index("```json") + 7
        end = text.index("```", start) if "```" in text[start:] else len(text)
        return text[start:end].strip()
    if "```" in text:
        start = text.index("```") + 3
        end = text.index("```", start) if "```" in text[start:] else len(text)
        return text[start:end].strip()
    return text.strip()


def _recover_score_from_raw(raw_content, valid_scores):
    """从非标准格式或截断JSON中恢复分数 (fallback parser)."""
    if not raw_content:
        return None
    for suffix in ['"}', '"]}', '", "issues": []}']:
        try:
            r = json.loads(raw_content + suffix)
            s = float(r.get("score", -1))
            if s in valid_scores:
                return s
        except (json.JSONDecodeError, ValueError):
            continue
    import re
    patterns = [
        r'"score"\s*:\s*([\d.]+)',
        r'\*\*评分[：:]\*\*\s*([\d.]+)',
        r'Score[：:]\s*([\d.]+)',
        r'score[：:]\s*([\d.]+)',
        r'评分[：:]\s*([\d.]+)',
    ]
    for pat in patterns:
        m = re.search(pat, raw_content)
        if m:
            s = float(m.group(1))
            if s in valid_scores:
                return s
    return None


def extract_score(content, valid_scores):
    json_str = _extract_json_block(content)
    try:
        result = json.loads(json_str)
    except (json.JSONDecodeError, ValueError):
        recovered = _recover_score_from_raw(content, valid_scores)
        if recovered is not None:
            return {
                "score": recovered,
                "reason": f"Score {recovered} recovered from non-JSON response: {content[:200]}",
                "issues": ["score recovered from non-standard format"],
                "raw_content": content,
            }
        return {
            "score": 0.0,
            "reason": f"Parse failure: {content[:200]}",
            "issues": ["parse failure"],
            "raw_content": content,
        }
    score = float(result.get("score", 0.0))
    if score not in valid_scores:
        score = min(valid_scores, key=lambda v: abs(v - score))
    issues = result.get("issues", [])
    if isinstance(issues, str):
        issues = [issues]
    elif not isinstance(issues, list):
        issues = []
    return {"score": score, "reason": result.get("reason", ""), "issues": issues}


def select_questions(baseline_records, answer_by_id):
    random.seed(42)
    candidates = []
    for r in baseline_records:
        dim = r["dimension"]
        lvl = r["difficulty_level"]
        if dim in ("B", "D") and lvl in ("L2", "L3", "L4"):
            if r["id"] in answer_by_id:
                candidates.append(r)

    b_pool = sorted(
        [r for r in candidates if r["dimension"] == "B"],
        key=lambda r: (r["difficulty_level"], r["id"]),
    )
    d_pool = sorted(
        [r for r in candidates if r["dimension"] == "D"],
        key=lambda r: (r["difficulty_level"], r["id"]),
    )

    def stratified_pick(records, n):
        by_level = {}
        for r in records:
            by_level.setdefault(r["difficulty_level"], []).append(r)
        picked = []
        for lvl in ["L2", "L3", "L4"]:
            if lvl in by_level and by_level[lvl]:
                picked.append(by_level[lvl].pop(0))
        remaining = [r for recs in by_level.values() for r in recs]
        random.shuffle(remaining)
        while len(picked) < n and remaining:
            picked.append(remaining.pop(0))
        return picked[:n]

    selected = stratified_pick(b_pool, 5) + stratified_pick(d_pool, 5)
    random.shuffle(selected)
    return selected


def main():
    local_llm_url = f"http://localhost:{os.environ.get('LLAMA_CHAT_PORT', '8080')}/v1/chat/completions"

    core = json.loads(
        (ROOT / "benchmark/dataset/core.json").read_text(encoding="utf-8")
    )
    baseline = json.loads(
        (ROOT / "benchmark/eval/results/judge_matrix/qwen3.5_35b_sample_judge.json").read_text(
            encoding="utf-8"
        )
    )
    model_answers = json.loads(
        (ROOT / "benchmark/eval/results/judge_matrix/deepseek_v4_flash_judge.json").read_text(
            encoding="utf-8"
        )
    )

    question_by_id = {q["id"]: q for q in core["questions"]}
    answer_by_id = {r["id"]: r["model_answer"] for r in model_answers["records"]}
    baseline_by_id = {r["id"]: r for r in baseline["records"]}

    selected = select_questions(baseline["records"], answer_by_id)
    print(
        f"Selected {len(selected)} questions: "
        f"{[r['id'] + ' (' + r['dimension'] + '/' + r['difficulty_level'] + ')' for r in selected]}"
    )
    print(f"num_predict={NUM_PREDICT} | 预计总耗时 ~{len(selected)*90 // 60}min")

    client = httpx.Client(timeout=600, trust_env=False)
    treatment_records = []
    total_elapsed = 0.0

    for i, rec in enumerate(selected):
        qid = rec["id"]
        dim = rec["dimension"]
        q = question_by_id[qid]
        model_answer = answer_by_id[qid]

        if dim == "B":
            system_prompt = B_SYSTEM_PROMPT
            user_prompt = B_USER_PROMPT_TEMPLATE.format(
                question=q["question"],
                expected_answer=q["answer"],
                model_answer=model_answer,
            )
            valid_scores = {0.0, 0.2, 0.5, 0.7, 1.0}
        else:
            system_prompt = D_SYSTEM_PROMPT
            user_prompt = (
                f"## 问题\n{q['question']}\n\n"
                f"## 标准答案\n{q['answer']}\n\n"
                f"## 模型回答\n{model_answer}\n\n"
                f"请评分。"
            )
            valid_scores = {0.0, 0.25, 0.5, 0.75, 1.0}

        bl_score = rec["score"]
        print(
            f"\n[{i + 1}/{len(selected)}] {qid} ({dim}/{rec['difficulty_level']}) "
            f"baseline={bl_score} | calling 35B think=true (num_predict={NUM_PREDICT})..."
        )
        t0 = time.time()
        try:
            resp = client.post(
                local_llm_url,
                json={
                    "model": "qwen3.5:35b",
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "stream": False,
                    "options": {"temperature": 0, "num_predict": NUM_PREDICT},
                },
            )
            resp.raise_for_status()
            data = resp.json()
            msg = data.get("message", {})
            content = msg.get("content", "")
            thinking_text = msg.get("thinking", "")
            elapsed = time.time() - t0
            total_elapsed += elapsed

            if content:
                result = extract_score(content, valid_scores)
            else:
                result = {
                    "score": 0.0,
                    "reason": f"Empty content (thinking consumed all {NUM_PREDICT} tokens, thinking_len={len(thinking_text)})",
                    "issues": ["empty content - insufficient num_predict"],
                    "raw_thinking": thinking_text[:500],
                }

            result["id"] = qid
            result["dimension"] = dim
            result["difficulty_level"] = rec["difficulty_level"]
            result["think_mode"] = True
            result["elapsed_s"] = round(elapsed, 1)
            result["thinking_len"] = len(thinking_text)
            result["content_len"] = len(content)
            treatment_records.append(result)

            delta = result["score"] - bl_score
            print(f"  score={result['score']} (Δ={delta:+.2f}) elapsed={elapsed:.0f}s think={len(thinking_text)}c content={len(content)}c")
            print(f"  reason: {result['reason'][:200]}")
        except Exception as e:
            elapsed = time.time() - t0
            total_elapsed += elapsed
            print(f"  ERROR: {e}")
            treatment_records.append(
                {
                    "id": qid,
                    "dimension": dim,
                    "difficulty_level": rec["difficulty_level"],
                    "score": 0.0,
                    "reason": f"Error: {e}",
                    "issues": ["probe error"],
                    "think_mode": True,
                    "elapsed_s": round(elapsed, 1),
                }
            )

    client.close()

    deltas = []
    for tr in treatment_records:
        bl = baseline_by_id[tr["id"]]
        delta = tr["score"] - bl["score"]
        deltas.append(delta)
        tr["baseline_score"] = bl["score"]
        tr["delta"] = round(delta, 4)

    delta_mean = round(sum(deltas) / len(deltas), 4) if deltas else 0.0

    if delta_mean > 0.05:
        verdict = "假说成立：开思考后35B评分更宽松（Δ>0.05）"
    elif delta_mean < -0.05:
        verdict = "假说反向：开思考后35B评分更严苛（Δ<-0.05）"
    else:
        verdict = "无显著影响：思考模式在本次采样中未观测到显著差异（|Δ|<=0.05）"

    output = {
        "experiment": "think_probe_35b",
        "hypothesis": "35B即使关了显式思考仍可能在隐层执行轻量推理→评分更严苛",
        "design": "10 questions (5B + 5D, L2-L4), same judge prompt, think=true vs think=false",
        "baseline_source": "benchmark/eval/results/judge_matrix/qwen3.5_35b_sample_judge.json",
        "treatment_model": "qwen3.5:35b",
        "think_mode": "true (default, no think=false in request)",
        "num_predict": NUM_PREDICT,
        "total_questions": len(treatment_records),
        "total_elapsed_s": round(total_elapsed, 1),
        "summary": {
            "delta_mean": delta_mean,
            "delta_min": round(min(deltas), 4) if deltas else None,
            "delta_max": round(max(deltas), 4) if deltas else None,
            "verdict": verdict,
        },
        "records": treatment_records,
    }

    out_path = ROOT / "benchmark" / "eval" / "results" / "think_probe_35b_10queries.json"
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n{'=' * 60}")
    print(f"RESULTS: delta_mean={delta_mean}  total_elapsed={total_elapsed:.0f}s ({total_elapsed/60:.1f}min)")
    print(f"Deltas: {[round(d, 2) for d in deltas]}")
    print(f"Verdict: {verdict}")
    print(f"Output: {out_path}")


if __name__ == "__main__":
    main()
