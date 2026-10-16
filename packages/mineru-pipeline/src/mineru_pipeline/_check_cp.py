"""Debug script: inspect checkpoint file. Not importable as library code."""
import json
from pathlib import Path

if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parents[4]
    cp_path = ROOT / "pipeline" / "complete_metadata.checkpoint.json"
    with open(cp_path, encoding="utf-8") as f:
        cp = json.load(f)
    print(f"arXiv enriched: {len(cp.get('arxiv_enriched_ids', []))}")
    print(f"enriched: {cp.get('arxiv_enriched_ids', [])}")
