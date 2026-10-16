"""Sciverse 文献检索 Intent Handler

通过 Sciverse agentic-search API 在 5.16 亿条科学文献记录中进行语义检索。

用法 (由 resource_scheduler 调度):
    from mineru_agent.search_handler import handle_sciverse_search
  result = handle_sciverse_search({"query": "model predictive control", "top_k": 5})
"""

from mineru_core.mineru_client import SciverseClient


def handle_sciverse_search(params: dict) -> dict:
    client = SciverseClient()
    r = client.agentic_search(
        query=params["query"],
        top_k=params.get("top_k", 10),
    )
    return {
        "status": "success",
        "hits": r.get("hits", []),
        "total_hits": len(r.get("hits", [])),
    }
