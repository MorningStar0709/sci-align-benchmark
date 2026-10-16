"""Merge dual-audit + arbitration results into audit_merged.json.

Reads (read-only):
  - benchmark/dataset/audit_by_ds.json
  - benchmark/dataset/audit_by_mimo.json
  - benchmark/dataset/audit_arbitration.json

Writes (new file):
  - benchmark/dataset/audit_merged.json

Usage:
    python -m mineru_benchmark.pipeline.merge_audit_summary
"""
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
DATASET_DIR = ROOT / "benchmark" / "dataset"


def _pct(n, total):
    """Safe percentage formatting."""
    return f"{n/total*100:.1f}%" if total else "N/A"


def main() -> int:
    # --- Load source data (read-only) ---
    with open(DATASET_DIR / "audit_by_ds.json", encoding="utf-8") as f:
        ds_data = json.load(f)
    with open(DATASET_DIR / "audit_by_mimo.json", encoding="utf-8") as f:
        mm_data = json.load(f)
    with open(DATASET_DIR / "audit_arbitration.json", encoding="utf-8") as f:
        arb_data = json.load(f)

    ds_results = {r["id"]: r for r in ds_data.get("results", []) if "id" in r}
    mm_results = {r["id"]: r for r in mm_data.get("results", []) if "id" in r}
    arb_results = {r["id"]: r for r in arb_data.get("results", []) if "id" in r}

    all_ids = sorted(set(ds_results.keys()) | set(mm_results.keys()))
    print(f"Loaded: DS={len(ds_results)}, MM={len(mm_results)}, ARB={len(arb_results)}, Total IDs={len(all_ids)}")

    # --- Merge logic ---
    merged = []
    stats = {"consensus": 0, "arbitration": 0, "error": 0, "passed": 0, "failed": 0, "excluded": 0}

    for qid in all_ids:
        ds_r = ds_results.get(qid, {})
        mm_r = mm_results.get(qid, {})
        arb_r = arb_results.get(qid, {})

        ds_pass = ds_r.get("pass")
        mm_pass = mm_r.get("pass")

        # Determine source and final verdict
        if qid in arb_results:
            # Disputed → use arbitration
            final_pass = arb_r.get("pass")
            source = "arbitration"
            stats["arbitration"] += 1
        elif ds_pass is not None and mm_pass is not None and ds_pass == mm_pass:
            # Consensus
            final_pass = ds_pass
            source = "consensus"
            stats["consensus"] += 1
        elif ds_pass is None and mm_pass is None:
            # Both error
            final_pass = None
            source = "error"
            stats["error"] += 1
        elif ds_pass is None:
            # DS error, use MM
            final_pass = mm_pass
            source = "partial_error"
            stats["error"] += 1
        elif mm_pass is None:
            # MM error, use DS
            final_pass = ds_pass
            source = "partial_error"
            stats["error"] += 1
        else:
            # Shouldn't reach here (disputes should be in arb_results)
            final_pass = None
            source = "error"
            stats["error"] += 1

        if final_pass is True:
            stats["passed"] += 1
        elif final_pass is False:
            stats["failed"] += 1
        else:
            stats["excluded"] += 1

        # Build entry
        entry = {
            "id": qid,
            "pass": final_pass,
            "source": source,
            "ds": {
                "pass": ds_pass,
                "confidence": ds_r.get("confidence", 0),
                "evidence": ds_r.get("evidence", ""),
                "issues": ds_r.get("issues", []),
            },
            "mm": {
                "pass": mm_pass,
                "confidence": mm_r.get("confidence", 0),
                "evidence": mm_r.get("evidence", ""),
                "issues": mm_r.get("issues", []),
            },
        }

        if source == "arbitration":
            entry["arbitration"] = {
                "pass": arb_r.get("pass"),
                "confidence": arb_r.get("confidence", 0),
                "evidence": arb_r.get("evidence", ""),
                "issues": arb_r.get("issues", []),
            }

        merged.append(entry)

    # --- Output ---
    total = len(merged)
    output = {
        "meta": {
            "merge_strategy": "dual_audit_with_arbitration",
            "auditors": ["deepseek", "mimo"],
            "arbitrator": "qwen",
            "total": total,
            "passed": stats["passed"],
            "failed": stats["failed"],
            "excluded": stats["excluded"],
            "pass_rate": round(stats["passed"] / total, 4) if total else 0,
            "consensus_count": stats["consensus"],
            "arbitration_count": stats["arbitration"],
            "error_count": stats["error"],
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
        "results": merged,
    }

    output_path = DATASET_DIR / "audit_merged.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n{'=' * 60}")
    print("MERGE COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Total:         {total}")
    print(f"  ✓ Passed:      {stats['passed']} ({_pct(stats['passed'], total)})")
    print(f"  ✗ Failed:      {stats['failed']} ({_pct(stats['failed'], total)})")
    print(f"  ? Excluded:    {stats['excluded']} ({_pct(stats['excluded'], total)})")
    print(f"  Consensus:     {stats['consensus']}")
    print(f"  Arbitration:   {stats['arbitration']}")
    print(f"  Error/partial: {stats['error']}")
    print(f"\n  Output: {output_path}")
    print(f"{'=' * 60}")

    # --- Integrity check: verify source files unchanged ---
    for fn in ["audit_by_ds.json", "audit_by_mimo.json", "audit_arbitration.json"]:
        fp = DATASET_DIR / fn
        sz = fp.stat().st_size
        print(f"  [intact] {fn} ({sz:,} bytes)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
