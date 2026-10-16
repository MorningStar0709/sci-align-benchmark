import json
from pathlib import Path

# Phase 5: restored from new package location
ROOT = Path(__file__).resolve().parents[4]  # packages/mineru-pipeline/src/mineru_pipeline/ → project root
CHECKPOINT_DIR = ROOT / "pipeline"
METADATA_PATH = ROOT / "corpus" / "metadata.json"
OUTPUT_CHECKPOINT = CHECKPOINT_DIR / "complete_metadata.checkpoint.json"

CHUNKS = 4

def _load_existing_preserved() -> tuple[set, dict]:
    """Load preserved arxiv IDs and cross results from existing checkpoint."""
    arxiv_ids: set = set()
    cross_results: dict = {}
    if OUTPUT_CHECKPOINT.exists():
        with open(OUTPUT_CHECKPOINT, encoding="utf-8") as f:
            existing_cp = json.load(f)
        arxiv_ids.update(existing_cp.get("arxiv_enriched_ids", []))
        cross_results.update(existing_cp.get("cross_results", {}))
    return arxiv_ids, cross_results


def _merge_chunks() -> tuple[set, dict, set]:
    """Merge all chunk checkpoints."""
    merged_classified: set = set()
    merged_results: dict = {}
    preserved_arxiv_ids: set = set()
    for i in range(CHUNKS):
        cp_path = CHECKPOINT_DIR / f"complete_metadata.checkpoint.chunk_{i}.json"
        if not cp_path.exists():
            print(f"[WARN] chunk {i} checkpoint missing: {cp_path}")
            continue
        with open(cp_path, encoding="utf-8") as f:
            cp = json.load(f)
        ids = cp.get("classified_ids", [])
        results = cp.get("classification_results", {})
        merged_classified.update(ids)
        merged_results.update(results)
        preserved_arxiv_ids.update(cp.get("arxiv_enriched_ids", []))
        print(f"[chunk {i}] {len(ids)} classified, {len(results)} results")
    return merged_classified, merged_results, preserved_arxiv_ids


def _enrich_arxiv_ids(all_docs: list[dict], preserved_arxiv_ids: set) -> set:
    """Add arxiv IDs from metadata that meet quality criteria."""
    for doc in all_docs:
        if (
            doc.get("type") == "arxiv"
            and doc.get("arxiv_id")
            and "Unknown" not in doc.get("author", "")
            and "To be verified" not in doc.get("author", "")
        ):
            preserved_arxiv_ids.add(doc["id"])
    return preserved_arxiv_ids


def _apply_classification(all_docs: list[dict], merged_results: dict) -> int:
    """Apply classification results to docs."""
    applied = 0
    for doc in all_docs:
        doc_id = doc["id"]
        if doc_id in merged_results:
            r = merged_results[doc_id]
            doc["control_category"] = r["categories"]
            if r.get("difficulty"):
                doc["difficulty"] = r["difficulty"]
            applied += 1
    return applied


def _save_checkpoint(merged_classified, merged_results, preserved_arxiv_ids, preserved_cross_results):
    """Write merged checkpoint."""
    merged_cp = {
        "phase": "A_complete",
        "classified_ids": sorted(merged_classified),
        "classification_results": merged_results,
        "arxiv_enriched_ids": sorted(preserved_arxiv_ids),
        "cross_results": preserved_cross_results,
    }
    with open(OUTPUT_CHECKPOINT, "w", encoding="utf-8") as f:
        json.dump(merged_cp, f, ensure_ascii=False, indent=2)
    print(f"[Merge] 合并 checkpoint 写入: {OUTPUT_CHECKPOINT}")


def _validate(all_docs: list[dict]):
    """Validate classification results."""
    empty_cat = sum(1 for d in all_docs if not d.get("control_category"))
    print(f"\n[验证] control_category 空: {empty_cat}/{len(all_docs)}")
    non_std = 0
    from mineru_pipeline.complete_metadata import STANDARD_CATEGORIES
    for d in all_docs:
        for c in d.get("control_category", []):
            if c not in STANDARD_CATEGORIES:
                non_std += 1
    print(f"[验证] 非标准分类: {non_std}")



def main():
    preserved_arxiv_ids, preserved_cross_results = _load_existing_preserved()
    merged_classified, merged_results, chunk_arxiv_ids = _merge_chunks()
    preserved_arxiv_ids.update(chunk_arxiv_ids)

    print(f"\n[Merge] 合计: {len(merged_classified)} classified, {len(merged_results)} results")

    with open(METADATA_PATH, encoding="utf-8") as f:
        metadata = json.load(f)
    all_docs = metadata["docs"]

    preserved_arxiv_ids = _enrich_arxiv_ids(all_docs, preserved_arxiv_ids)
    applied = _apply_classification(all_docs, merged_results)

    metadata["docs"] = all_docs
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"[Merge] metadata.json updated: {applied} docs classified")

    _save_checkpoint(merged_classified, merged_results, preserved_arxiv_ids, preserved_cross_results)
    _validate(all_docs)


if __name__ == "__main__":
    main()
