import argparse
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from mineru_pipeline._constants import CATEGORY_DIFFICULTY_MAP


# Phase 5: restored from new package location
ROOT = Path(__file__).resolve().parents[4]  # packages/mineru-pipeline/src/mineru_pipeline/ → project root
DEFAULT_CORPUS = ROOT / "corpus"

_LEVEL_ORDER = {"L1": 1, "L2": 2, "L3": 3, "L4": 4}


def _clean_md_for_stats(md_content):
    """清洗 Markdown 中不影响统计的噪声内容。"""
    content = md_content
    content = re.sub(r'^---[\s\S]*?---\n', '', content)
    content = re.sub(r'```[\s\S]*?```', '', content)
    content = re.sub(r'<[^>]+>', '', content)
    return content


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_json(path, default):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def generated_at_iso():
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def build_chunk_counts(corpus_root):
    manifest = load_json(corpus_root / "chunks" / "chunks_manifest.json", {"chunks": []})
    counts = Counter()
    tokens = Counter()
    for chunk in manifest.get("chunks", []):
        filename = chunk.get("filename", "")
        counts[filename] += 1
        tokens[filename] += int(chunk.get("estimated_tokens", 0) or 0)
    return manifest, counts, tokens


def _compute_document_quality(doc, filename, formulas, figures, chunk_counts):
    """计算单篇文档的质量检查结果。"""
    block_count = sum(1 for item in formulas if item.get("type") == "block")
    inline_count = sum(1 for item in formulas if item.get("type") == "inline")
    image_with_file_count = sum(1 for item in figures if item.get("image_path"))

    issues = []
    if not doc.get("id") or not doc.get("title") or not doc.get("author"):
        issues.append({"filename": filename, "issue": "metadata_incomplete"})
    if not formulas:
        issues.append({"filename": filename, "issue": "no_formulas"})
    if figures and image_with_file_count == 0:
        issues.append({"filename": filename, "issue": "image_references_without_files", "count": len(figures)})
    if chunk_counts[filename] == 0:
        issues.append({"filename": filename, "issue": "no_chunks"})

    return block_count, inline_count, image_with_file_count, issues


def _compute_document_language(doc_dir, filename, categories):
    """检测文档语言 + 难度等级。"""
    md_file = doc_dir / f"{filename}.md"
    if not md_file.exists():
        return 0, "unknown", 0.0, "L3"

    md_content = md_file.read_text(encoding="utf-8")
    md_clean = _clean_md_for_stats(md_content)
    char_count = len(md_clean)
    chinese_chars = sum(1 for c in md_clean if '\u4e00' <= c <= '\u9fff')
    total_chars = len(md_clean)
    zh_ratio = chinese_chars / total_chars if total_chars > 0 else 0

    if zh_ratio > 0.3:
        language = "zh"
    elif zh_ratio < 0.05:
        language = "en"
    else:
        language = "mixed"

    math_symbols = sum(1 for c in md_clean if c in '=+-*/^_()[]{}<>|\\$%#@!')
    symbol_to_word_ratio = round(math_symbols / total_chars, 4) if total_chars > 0 else 0

    category_levels = [CATEGORY_DIFFICULTY_MAP.get(c, "L3") for c in categories]
    difficulty = min(category_levels, key=lambda x: _LEVEL_ORDER.get(x, 3)) if category_levels else "L3"

    return char_count, language, symbol_to_word_ratio, difficulty


def _build_per_doc_entry(doc, filename, doc_dir, corpus_root, categories, chunk_counts, chunk_tokens):
    formulas = load_json(doc_dir / "formulas.json", [])
    figures = load_json(doc_dir / "figures.json", [])
    quality_issues = []

    block_count, inline_count, image_with_file_count, doc_quality = _compute_document_quality(
        doc, filename, formulas, figures, chunk_counts
    )
    quality_issues.extend(doc_quality)

    char_count, language, symbol_to_word_ratio, difficulty = _compute_document_language(
        doc_dir, filename, categories
    )

    entry = {
        "id": doc.get("id", ""),
        "filename": filename,
        "title": doc.get("title", ""),
        "control_category": categories,
        "formula_count": len(formulas),
        "block_formula_count": block_count,
        "inline_formula_count": inline_count,
        "image_count": len(figures),
        "image_with_file_count": image_with_file_count,
        "file_size_bytes": doc.get("file_size_bytes", 0),
        "chunk_count": chunk_counts[filename],
        "estimated_tokens": chunk_tokens[filename],
        "formula_density": round(len(formulas) / char_count, 6) if char_count > 0 else 0,
        "symbol_to_word_ratio": symbol_to_word_ratio,
        "language": language,
        "difficulty_level": difficulty,
    }
    return entry, quality_issues


def _process_docs(metadata, corpus_root, chunk_counts, chunk_tokens):
    """Process all documents and return per-doc entries, category counts, and quality issues."""
    per_doc = []
    category_counts = Counter()
    quality_issues = []

    for doc in metadata.get("docs", []):
        filename = doc.get("filename", "")
        if not filename:
            quality_issues.append({"filename": "", "issue": "metadata_filename_missing"})
            continue

        doc_dir = corpus_root / "processed" / filename
        if not doc_dir.exists():
            quality_issues.append({"filename": filename, "issue": "corpus_processed_dir_missing"})

        categories = doc.get("control_category", [])
        for category in categories:
            category_counts[category] += 1

        doc_entry, doc_quality = _build_per_doc_entry(doc, filename, doc_dir, corpus_root, categories, chunk_counts, chunk_tokens)
        per_doc.append(doc_entry)
        quality_issues.extend(doc_quality)

    return per_doc, category_counts, quality_issues


def _build_corpus_stats(per_doc, category_counts, chunks_manifest):
    """Build the corpus statistics dict from per-doc data."""
    return {
        "total_docs": len(per_doc),
        "total_formulas": sum(item["formula_count"] for item in per_doc),
        "total_block_formulas": sum(item["block_formula_count"] for item in per_doc),
        "total_inline_formulas": sum(item["inline_formula_count"] for item in per_doc),
        "total_images": sum(item["image_count"] for item in per_doc),
        "total_images_with_file": sum(item["image_with_file_count"] for item in per_doc),
        "total_chunks": chunks_manifest.get("total_chunks", 0),
        "total_tokens_estimated": sum(item["estimated_tokens"] for item in per_doc),
        "classification_distribution": dict(sorted(category_counts.items())),
        "per_doc": per_doc,
        "generated_at": generated_at_iso(),
    }


def _build_quality_report(stats, quality_issues, corpus_root, metadata_path, per_doc):
    """Build the quality report dict from stats and issues."""
    return {
        "generated_at": stats["generated_at"],
        "total_docs": stats["total_docs"],
        "issue_count": len(quality_issues),
        "issues": quality_issues,
        "checks": {
            "metadata_json_exists": metadata_path.exists(),
            "chunks_manifest_exists": (corpus_root / "chunks" / "chunks_manifest.json").exists(),
            "all_docs_have_processed_dirs": all((corpus_root / "processed" / item["filename"]).exists() for item in per_doc),
            "all_docs_have_formulas_json": all((corpus_root / "processed" / item["filename"] / "formulas.json").exists() for item in per_doc),
            "all_docs_have_figures_json": all((corpus_root / "processed" / item["filename"] / "figures.json").exists() for item in per_doc),
            "all_docs_have_chunks": all(item["chunk_count"] > 0 for item in per_doc),
        },
    }


def build_stats(corpus_root):
    metadata_path = corpus_root / "metadata.json"
    if not metadata_path.exists():
        raise SystemExit(f"Missing metadata: {metadata_path}. Run build_corpus.py first.")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    chunks_manifest, chunk_counts, chunk_tokens = build_chunk_counts(corpus_root)

    per_doc, category_counts, quality_issues = _process_docs(metadata, corpus_root, chunk_counts, chunk_tokens)
    stats = _build_corpus_stats(per_doc, category_counts, chunks_manifest)
    quality_report = _build_quality_report(stats, quality_issues, corpus_root, metadata_path, per_doc)

    stats_dir = corpus_root / "stats"
    write_json(stats_dir / "corpus_stats.json", stats)
    write_json(stats_dir / "quality_report.json", quality_report)
    return stats, quality_report


def main():
    parser = argparse.ArgumentParser(description="Build corpus statistics and quality reports.")
    parser.add_argument("-c", "--corpus", default=str(DEFAULT_CORPUS), help="Corpus root directory.")
    args = parser.parse_args()

    corpus_root = Path(args.corpus).resolve()
    if not corpus_root.exists():
        raise SystemExit(f"Corpus not found: {corpus_root}")

    stats, quality = build_stats(corpus_root)
    print(f"Saved: {corpus_root / 'stats' / 'corpus_stats.json'}")
    print(f"Docs: {stats['total_docs']} | Formulas: {stats['total_formulas']} | Chunks: {stats['total_chunks']}")
    print(f"Quality issues: {quality['issue_count']}")


if __name__ == "__main__":
    main()
