"""Track 1: Sciverse 14 子领域覆盖统计 + 跨模态对齐抽查

在控制科学 14 个子领域上调用 Sciverse meta-search 统计文献覆盖量，并通过 agentic-search
抽查跨模态对齐质量。结果纳入 DATA-TRACE 溯源体系。

输出:
  benchmark/eval/results/sciverse_14_subfield_coverage.json
  benchmark/eval/results/sciverse_alignment_spot.json

用法:
  .venv/bin/python -m benchmark.sciverse_coverage
"""

import argparse
import json
from datetime import datetime, timezone

from mineru_core.mineru_client import SciverseClient
from mineru_core.paths import PROJECT_ROOT

SUBFIELDS: dict[str, str] = {
    "PID控制": "PID control stability",
    "估计与定位": "Kalman filter state estimation",
    "导航制导与控制": "guidance navigation control GPS",
    "控制理论": "control theory Lyapunov",
    "数字控制": "digital control discrete time",
    "智能控制": "intelligent control neural network",
    "最优控制": "optimal control Pontryagin",
    "现代控制": "modern control state space",
    "线性系统": "linear system theory",
    "经典控制": "classical control root locus",
    "自抗扰控制": "active disturbance rejection control",
    "自适应控制": "adaptive control nonlinear",
    "非线性控制": "nonlinear control backstepping",
    "鲁棒控制": "robust control H-infinity",
}

ALIGNMENT_QUERIES: list[dict] = [
    {"label": "PID稳定性边界", "query": "PID controller stability margin analysis root locus", "subfield": "PID控制"},
    {"label": "Kalman状态估计", "query": "Kalman filter state estimation covariance convergence", "subfield": "估计与定位"},
    {"label": "GPS导航控制", "query": "GPS navigation Kalman filter attitude determination", "subfield": "导航制导与控制"},
    {"label": "Lyapunov稳定性", "query": "Lyapunov stability theorem proof nonlinear system", "subfield": "控制理论"},
    {"label": "数字离散控制", "query": "digital control discrete time Z-transform stability", "subfield": "数字控制"},
    {"label": "神经网络控制", "query": "neural network adaptive control backpropagation", "subfield": "智能控制"},
    {"label": "Pontryagin最优", "query": "Pontryagin minimum principle optimal control Hamiltonian", "subfield": "最优控制"},
    {"label": "状态空间现代", "query": "state space representation controllability observability", "subfield": "现代控制"},
    {"label": "线性系统理论", "query": "linear system theory transfer function pole zero", "subfield": "线性系统"},
    {"label": "H-infinity鲁棒", "query": "H-infinity robust control synthesis uncertainty", "subfield": "鲁棒控制"},
]

COVERAGE_OUTPUT = (
    PROJECT_ROOT / "benchmark" / "eval" / "results" / "sciverse_14_subfield_coverage.json"
)
ALIGNMENT_OUTPUT = (
    PROJECT_ROOT / "benchmark" / "eval" / "results" / "sciverse_alignment_spot.json"
)


def classify_alignment(hit: dict, query_keywords: list[str]) -> str:
    if not hit.get("chunk"):
        return "inconsistent"

    chunk_lower = hit["chunk"].lower()
    title_lower = hit.get("title", "").lower()
    combined = chunk_lower + " " + title_lower

    match_count = sum(1 for kw in query_keywords if kw.lower() in combined)
    kw_count = len(query_keywords)

    if match_count >= kw_count * 0.6:
        return "consistent"
    if match_count >= kw_count * 0.3:
        return "partial"
    return "inconsistent"


def run_coverage(client: SciverseClient) -> dict:
    fields = ["title", "doi", "publication_published_year"]
    results = {}

    for zh_name, en_query in SUBFIELDS.items():
        r = client.meta_search(query=en_query, fields=fields, page_size=3)
        total = r.get("total_count", 0)
        items = r.get("results", [])
        top_item = items[0] if items else {}

        results[zh_name] = {
            "en_query": en_query,
            "total_count": total,
            "top_title": top_item.get("title", ""),
            "top_year": top_item.get("publication_published_year"),
            "top_doi": top_item.get("doi", ""),
        }

    return results


def run_alignment_spots(client: SciverseClient) -> list[dict]:
    spots = []

    for aq in ALIGNMENT_QUERIES:
        r = client.agentic_search(query=aq["query"], top_k=1)
        hits = r.get("hits", [])
        keywords = aq["query"].lower().split()

        if hits:
            h = hits[0]
            alignment = classify_alignment(h, keywords)
            spots.append(
                {
                    "label": aq["label"],
                    "query": aq["query"],
                    "subfield": aq["subfield"],
                    "alignment": alignment,
                    "score": h.get("score"),
                    "title": h.get("title", ""),
                    "chunk_preview": h.get("chunk", "")[:200],
                    "page_no": h.get("page_no"),
                    "doc_id": h.get("doc_id", ""),
                }
            )
        else:
            spots.append(
                {
                    "label": aq["label"],
                    "query": aq["query"],
                    "subfield": aq["subfield"],
                    "alignment": "inconsistent",
                    "score": 0,
                    "title": "",
                    "chunk_preview": "",
                }
            )

    return spots


def main():
    parser = argparse.ArgumentParser(
        description="Sciverse 14 子领域覆盖统计 + 跨模态对齐抽查"
    )
    parser.add_argument(
        "--skip-coverage",
        action="store_true",
        help="跳过覆盖统计，仅做对齐抽查",
    )
    parser.add_argument(
        "--skip-alignment",
        action="store_true",
        help="跳过对齐抽查，仅做覆盖统计",
    )
    args = parser.parse_args()

    client = SciverseClient()
    timestamp = datetime.now(timezone.utc).isoformat()

    if not args.skip_coverage:
        print("[Coverage] 统计 14 子领域 Sciverse 文献量...", flush=True)
        coverage = run_coverage(client)
        coverage_report = {
            "generated_at": timestamp,
            "input": {"subfields": list(SUBFIELDS.keys())},
            "summary": {
                "total_subfields": len(coverage),
                "subfields_above_100k": sum(
                    1 for v in coverage.values() if v["total_count"] > 100000
                ),
                "min_total": min(v["total_count"] for v in coverage.values()),
                "max_total": max(v["total_count"] for v in coverage.values()),
            },
            "subfields": coverage,
        }
        COVERAGE_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        with open(COVERAGE_OUTPUT, "w", encoding="utf-8") as f:
            json.dump(coverage_report, f, ensure_ascii=False, indent=2)
        print(
            f"[Coverage] {coverage_report['summary']['subfields_above_100k']}/"
            f"{len(coverage)} subfields above 100k"
        )
        for zh, v in coverage.items():
            print(f"  {zh}: {v['total_count']:,}")
        print(f"[Coverage] → {COVERAGE_OUTPUT}", flush=True)

    if not args.skip_alignment:
        print("\n[Alignment] 10 条跨模态对齐抽查...", flush=True)
        spots = run_alignment_spots(client)
        consistent = sum(1 for s in spots if s["alignment"] == "consistent")
        partial = sum(1 for s in spots if s["alignment"] == "partial")
        alignment_report = {
            "generated_at": timestamp,
            "input": {"queries": [aq["label"] for aq in ALIGNMENT_QUERIES]},
            "summary": {
                "total": len(spots),
                "consistent": consistent,
                "partial": partial,
                "inconsistent": len(spots) - consistent - partial,
            },
            "spots": spots,
        }
        ALIGNMENT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        with open(ALIGNMENT_OUTPUT, "w", encoding="utf-8") as f:
            json.dump(alignment_report, f, ensure_ascii=False, indent=2)
        for s in spots:
            print(
                f"  [{s['alignment']:12s}] {s['label']} | score={s.get('score',0):.3f} "
                f"| {s.get('title','')[:60]}"
            )
        print(
            f"[Alignment] {consistent}/{len(spots)} consistent, {partial}/{len(spots)} partial"
        )
        print(f"[Alignment] → {ALIGNMENT_OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
