"""方案 B：孤儿单答案 QA Judge — 维度感知的自动回收。
对 185 条 needs_review 按维度逐条调用 DeepSeek API 判断是否合格。
"""
import json
import os
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
BENCHMARK = ROOT / "benchmark" / "dataset"

# ── Load data ──
merged = json.loads((BENCHMARK / "merged.json").read_text(encoding="utf-8"))
orphans = [q for q in merged["questions"] if q.get("consistency_status") == "needs_review"]
print(f"[orphan_judge] {len(orphans)} orphans to judge")
random.Random(42).shuffle(orphans)

# ── API setup ──
from openai import OpenAI, DefaultHttpxClient
from mineru_core.settings import PROVIDERS
from mineru_eval.client_factory import wrap_client
api_key = os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("OPENAI_API_KEY")
if not api_key:
    raise SystemExit("DEEPSEEK_API_KEY/OPENAI_API_KEY is required for orphan_judge.py")
client = wrap_client(OpenAI(
    api_key=api_key,
    base_url=PROVIDERS["deepseek"].base_url,
    timeout=60,
    max_retries=1,
    http_client=DefaultHttpxClient(proxy=None, trust_env=False),
))

def build_prompt(dim, question, answer):
    rules = {
        "A": "判断该答案是否准确写出了控制科学概念的精确定义、数学表达式和核心条件。评分规则：1.0=完全正确（定义+公式+条件全部正确）；0.6=核心正确但有轻微遗漏；0.3=部分正确但有明显错误；0.0=错误或空白。",
        "B": "判断该答案的推理步骤是否逻辑自洽、关键推导是否正确、结论是否合理。评分规则：1.0=全部步骤正确且逻辑自洽；0.6=核心推导正确但有中间步骤遗漏；0.3=部分正确但关键步骤有误；0.0=无有效推理或完全错误。",
        "C": "判断该答案对条件变化的分析是否合理、结论是否有依据。评分规则：1.0=分析完整且结论合理；0.6=核心分析正确但不完整；0.3=部分合理但存在误解；0.0=无有效分析。",
        "D": "判断该设计方案是否可行、结构是否完整、是否覆盖了关键环节。评分规则：1.0=方案完整且可行；0.6=核心思路正确但细节不完整；0.3=有合理元素但整体不完整；0.0=不可行或空白。",
    }
    rule = rules.get(dim, rules["A"])
    return f"问题：{question}\n\n答案：{answer}\n\n{rule}\n\n输出JSON：{{\"score\": <0.0|0.3|0.6|1.0>, \"reason\": \"...\"}}"

SYSTEM = "你是一个控制科学答案评分员。严格按照 JSON 格式输出。分数只能是 0.0、0.3、0.6、1.0 之一。"

passed = 0
failed = 0
results = []

for i, q in enumerate(orphans):
    qid = q["id"]
    dim = q.get("dimension", "A")
    prompt = build_prompt(dim, q.get("question", ""), q.get("answer", ""))

    try:
        resp = client.chat.completions.create(
            model="deepseek-v4-flash",
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            max_tokens=512,
            response_format={"type": "json_object"},
        )
        raw = resp.choices[0].message.content.strip()
        obj = json.loads(raw)
        score = float(obj.get("score", 0.0))
    except Exception as e:
        print(f"  [{i+1}/{len(orphans)}] {qid} ERROR: {e}", flush=True)
        score = 0.0

    accepted = score >= 0.4
    if accepted:
        q["consistency_status"] = "reviewed_kept"
        passed += 1
    else:
        q["consistency_status"] = "reviewed_discarded"
        failed += 1

    results.append({"id": qid, "score": score, "accepted": accepted, "dim": dim})
    print(f"  [{i+1}/{len(orphans)}] {qid} ({dim}) score={score:.1f} {'OK' if accepted else 'NO'}", flush=True)

    if (i + 1) % 25 == 0:
        cp = {"done": i+1, "passed": passed, "failed": failed, "results": results}
        (BENCHMARK / "orphan_judge.checkpoint.json").write_text(json.dumps(cp), encoding="utf-8")
        print(f"  [checkpoint] {i+1} done", flush=True)

# ── Update merged.json ──
merged["meta"]["dimensions"] = {}
for q in merged["questions"]:
    d = q.get("dimension")
    if d:
        merged["meta"]["dimensions"][d] = merged["meta"]["dimensions"].get(d, 0) + 1

(BENCHMARK / "merged.json").write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
(BENCHMARK / "orphan_judge_diag.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in results), encoding="utf-8")

print(f"\n[orphan_judge] Done. passed={passed}, failed={failed}")
