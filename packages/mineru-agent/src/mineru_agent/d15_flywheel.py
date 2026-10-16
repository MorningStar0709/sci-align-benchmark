"""D15: Agent Sciverse Flywheel — 真实执行 + structured trace。

通过 sciverse_search_handler 执行 3 条控制科学查询，
记录 ExecutionLog 格式 trace，证明 Agent 可自主消费 Sciverse API。
"""

import json
import time
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[4]  # restored Phase 5

from mineru_agent.search_handler import handle_sciverse_search

OUT_DIR = ROOT / "benchmark" / "agent" / "logs"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT = OUT_DIR / "d15_sciverse_flywheel_trace.json"

QUERIES = [
    "robust model predictive control safety constraints for autonomous systems",
    "reinforcement learning model-free adaptive control convergence guarantees",
    "distributed optimization consensus control multi-agent network topology",
]

steps = []
all_hits = {}
started = time.time()

for qi, query in enumerate(QUERIES):
    print(f"[D15] Step {qi+1}/3: {query[:70]}...", end=" ", flush=True)
    t0 = time.time()
    try:
        result = handle_sciverse_search({"query": query, "top_k": 5})
        n = result.get("total_hits", 0)
        elapsed = int((time.time() - t0) * 1000)
        titles = [h.get("title", "")[:80] for h in result.get("hits", [])[:3]]
        steps.append({
            "step_id": qi + 1, "step_name": f"sciverse_search #{qi+1}",
            "tool": "Sciverse API (api.sciverse.space)", "status": "success",
            "duration_ms": elapsed,
            "input_summary": f"query={query[:60]}",
            "output_summary": f"total_hits={n}, top={titles[0] if titles else 'N/A'}",
            "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        })
        all_hits[f"query_{qi+1}"] = {
            "query": query, "total_hits": n,
            "sample_titles": titles,
            "sample_hits": result.get("hits", [])[:2],
        }
        print(f"{n} hits ({elapsed}ms)")
    except Exception as e:
        elapsed = int((time.time() - t0) * 1000)
        steps.append({
            "step_id": qi + 1, "step_name": f"sciverse_search #{qi+1}",
            "tool": "Sciverse API", "status": "failed",
            "duration_ms": elapsed, "input_summary": f"query={query[:60]}",
            "output_summary": f"error={str(e)[:80]}",
            "error": str(e)[:200], "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        })
        print(f"FAIL: {str(e)[:60]}")

total_ms = int((time.time() - started) * 1000)
n_success = sum(1 for s in steps if s["status"] == "success")

report = {
    "schema_version": "1.0",
    "task_id": f"d15-sciverse-flywheel-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
    "task_name": "Agent Sciverse Flywheel: 3 queries to structured trace",
    "steps": steps,
    "total_duration_ms": total_ms,
    "final_status": "completed" if n_success == len(QUERIES) else "partial",
    "created_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
    "all_hits": all_hits,
    "analysis": "Agent sciverse_search intent executed via handle_sciverse_search; "
                "3 queries covering robust MPC, RL control convergence, distributed optimization.",
    "n_queries": len(QUERIES),
    "n_successful": n_success,
    "conclusion": "Agent consumed Sciverse API autonomously — "
                   f"{n_success}/{len(QUERIES)} queries returned hits.",
}

OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

print(f"\n[D15] COMPLETE: {n_success}/{len(QUERIES)} queries successful")
print(f"  Total hits: {sum(h['total_hits'] for h in all_hits.values())}")
print(f"  Trace: {OUT}")
