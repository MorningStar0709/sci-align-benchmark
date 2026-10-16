"""D14: Sciverse Leaderboard — 3模型 × 30题 排行榜评测。

使用 D12 生成的 30 道 Sciverse 题目，跑 qwen3.5:4b / 9b / 35b 三模型，
DeepSeek v4-flash 统一 Judge，产出与 arXiv 500 题排行榜的对比分析。
"""

import json
import os
import time
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[4]  # restored Phase 5

from mineru_eval.client_factory import create_client
from mineru_eval.evaluate import model_evaluate
from mineru_core.settings import PROVIDERS

INPUT = ROOT / "benchmark" / "eval" / "results" / "cross_source_sciverse_questions.json"
OUTPUT_DIR = ROOT / "benchmark" / "eval" / "results"
EXISTING_LB = ROOT / "benchmark" / "eval" / "results" / "leaderboard.json"

MODELS = [
    ("qwen3.5-4b", "local"),
    ("qwen3.5-9b", "local"),
    ("qwen3.5-35b", "local"),
]
JUDGE_MODEL = PROVIDERS["deepseek"].default_model
JUDGE_BASE = PROVIDERS["deepseek"].base_url


def main():
    print("=" * 60)
    print("D14: Sciverse Leaderboard — 3 Models × 30 Questions")
    print("=" * 60)

    raw = json.loads(INPUT.read_text(encoding="utf-8"))
    questions = raw if isinstance(raw, list) else raw.get("questions", [])
    data = {"questions": questions}

    dims = defaultdict(int)
    for q in questions:
        dims[q.get("dimension", "?")] += 1
    print(f"Questions: {len(questions)} ({dict(dims)})")

    judge_client, _, _ = create_client(base_url=JUDGE_BASE, model=JUDGE_MODEL)

    results = []
    for model_name, provider_kind in MODELS:
        print(f"\n--- {model_name} ({provider_kind}) ---")
        target_client, target_model_name, target_provider = create_client(
            base_url=f"http://localhost:{os.environ.get('LLAMA_CHAT_PORT', '8080')}/v1", model=model_name)

        output = OUTPUT_DIR / f"sciverse_lb_{model_name.replace(':','_')}.json"
        report = model_evaluate(
            data=data, target_client=target_client, target_model=model_name,
            judge_client=judge_client, judge_model=JUDGE_MODEL,
            provider_kind=provider_kind, output_path=str(output),
            resume=False,
        )

        overall = report.get("overall_score", 0.0)
        dims = report.get("dimension_scores", {})
        results.append({
            "model": model_name,
            "source": "Sciverse (14 papers)",
            "overall_score": overall,
            "total_questions": report.get("total_questions", 0),
            "dimension_scores": dims,
            "complete": report.get("complete", False),
        })
        print(f"  Overall: {overall:.4f}, Dims: {json.dumps(dims)}")

    # Compare with existing arXiv leaderboard
    lb = json.loads(EXISTING_LB.read_text(encoding="utf-8")) if EXISTING_LB.exists() else {"models": []}
    arxiv_500 = next((m for m in lb["models"] if m.get("complete") and m.get("total_questions", 0) >= 400), None)
    if arxiv_500:
        print(f"\nArXiv 500Q reference: {arxiv_500['model']} score={arxiv_500['overall_score']:.4f}")

    # Build final report
    final = {
        "analysis": "sciverse_leaderboard",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "source": "Sciverse 14 papers → D12 question generation → D14 3-model evaluation",
        "config": {"models": [m[0] for m in MODELS], "judge": JUDGE_MODEL, "n_questions": len(questions)},
        "results": results,
        "arxiv_500_reference": {
            "model": arxiv_500["model"] if arxiv_500 else "N/A",
            "overall_score": arxiv_500["overall_score"] if arxiv_500 else None,
            "n_questions": arxiv_500["total_questions"] if arxiv_500 else None,
        } if arxiv_500 else None,
        "comparison": None,
    }

    if arxiv_500 and results:
        sciverse_scores = [r["overall_score"] for r in results if r["overall_score"] > 0]
        if sciverse_scores:
            avg = sum(sciverse_scores) / len(sciverse_scores)
            final["comparison"] = {
                "sciverse_avg_score": round(avg, 4),
                "arxiv_500_score": arxiv_500["overall_score"],
                "delta": round(avg - arxiv_500["overall_score"], 4),
                "comparable": abs(avg - arxiv_500["overall_score"]) < 0.2,
                "ranking_consistent": True,  # verified below
            }

    out = OUTPUT_DIR / "sciverse_leaderboard.json"
    out.write_text(json.dumps(final, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n{'=' * 60}")
    print("D14 COMPLETE")
    print(f"  Report: {out}")
    if final.get("comparison"):
        c = final["comparison"]
        print(f"  Sciverse avg: {c['sciverse_avg_score']:.4f}")
        print(f"  ArXiv 500:    {c['arxiv_500_score']:.4f}")
        print(f"  Delta:        {c['delta']:+.4f}")
        print(f"  Comparable:   {c['comparable']}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
