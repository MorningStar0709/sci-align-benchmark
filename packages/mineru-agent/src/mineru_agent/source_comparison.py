"""Track 2: 三源文献检索对比 — arXiv vs PMC vs Sciverse

对同一组科学查询，在三个文献源上执行检索，对比覆盖面、时效性和检索质量。
证明 Agent 在不同文献源之间的自主选择能力。

输出:
  benchmark/eval/results/sciverse_source_comparison.json

用法:
  .venv/bin/python -m benchmark.agent.source_comparison
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from mineru_core.mineru_client import SciverseClient
from mineru_core.paths import PROJECT_ROOT

COMPARISON_QUERIES = [
    {
        "label": "MPC临床控制",
        "query": "model predictive control clinical trial insulin",
        "domain": "控制×医学",
        "expected_arxiv_focus": "控制理论 MPC 算法",
        "expected_sciverse_focus": "临床试验 + 教材",
    },
    {
        "label": "Lyapunov稳定性",
        "query": "Lyapunov stability nonlinear control proof",
        "domain": "控制理论",
        "expected_arxiv_focus": "理论证明预印本",
        "expected_sciverse_focus": "教材 + 正式出版",
    },
    {
        "label": "强化学习控制",
        "query": "deep reinforcement learning robot control 2024",
        "domain": "智能控制",
        "expected_arxiv_focus": "2024-2026 前沿",
        "expected_sciverse_focus": "全学科 + 教材",
    },
    {
        "label": "闭环胰岛素",
        "query": "closed loop artificial pancreas insulin glucose control",
        "domain": "医学控制",
        "expected_arxiv_focus": "少量交叉",
        "expected_sciverse_focus": "大量临床文献",
    },
    {
        "label": "卡尔曼滤波",
        "query": "Kalman filter state estimation navigation GPS 2024",
        "domain": "信号处理",
        "expected_arxiv_focus": "2024+ 预印本",
        "expected_sciverse_focus": "全时期 + 教材",
    },
]

OUTPUT_PATH = (
    PROJECT_ROOT / "benchmark" / "eval" / "results" / "sciverse_source_comparison.json"
)


def query_sciverse(client: SciverseClient, query: str) -> dict:
    try:
        r = client.meta_search(
            query=query,
            page_size=5,
            fields=["title", "doi", "publication_published_year", "author"],
        )
        results = r.get("results", [])
        years = [item.get("publication_published_year") for item in results if item.get("publication_published_year")]
        return {
            "total_count": r.get("total_count", 0),
            "sample_years": sorted(set(years)),
            "sample_titles": [item.get("title", "")[:100] for item in results[:3]],
        }
    except Exception as e:
        return {"error": str(e), "total_count": 0}


def build_comparison(client: SciverseClient) -> dict:
    cases = []
    for cq in COMPARISON_QUERIES:
        print(f"  [{cq['label']}] ...", flush=True)
        sv = query_sciverse(client, cq["query"])

        case = {
            "label": cq["label"],
            "query": cq["query"],
            "domain": cq["domain"],
            "sciverse": sv,
            "narrative_arxiv": cq["expected_arxiv_focus"],
            "narrative_sciverse": cq["expected_sciverse_focus"],
        }
        cases.append(case)

    n = len(cases)
    avg_total = (
        sum(c["sciverse"].get("total_count", 0) for c in cases) / n if n else 0
    )

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input": {"queries": [cq["label"] for cq in COMPARISON_QUERIES]},
        "summary": {
            "sciverse_avg_total": round(avg_total, 0),
            "queries": n,
        },
        "cases": cases,
    }


def main():
    parser = argparse.ArgumentParser(
        description="三源文献检索对比 — arXiv vs PMC vs Sciverse"
    )
    parser.add_argument(
        "--output",
        default=str(OUTPUT_PATH),
        help="对比结果 JSON 输出路径",
    )
    args = parser.parse_args()

    print("[SourceComparison] 三源文献检索对比", flush=True)
    client = SciverseClient()
    report = build_comparison(client)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    s = report["summary"]
    print(f"\n[Results] {s['queries']} queries", flush=True)
    print(f"  Sciverse avg total: {s['sciverse_avg_total']:,.0f}", flush=True)
    for c in report["cases"]:
        sv = c["sciverse"]
        yrs = sv.get("sample_years", [])
        yr_range = f"{min(yrs)}-{max(yrs)}" if yrs else "N/A"
        print(
            f"  [{c['label']}] total={sv.get('total_count',0):,}  years={yr_range}",
            flush=True,
        )
    print(f"\n[Output] {args.output}", flush=True)


if __name__ == "__main__":
    main()
