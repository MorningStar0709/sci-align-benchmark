import argparse
import json
import re
from pathlib import Path

from mineru_pipeline import markdown_io

# Phase 5: restored from new package location
ROOT = Path(__file__).resolve().parents[4]  # packages/mineru-pipeline/src/mineru_pipeline/ → project root
DEFAULT_INPUT = ROOT / "data" / "processed"
DEFAULT_OUTPUT = ROOT / "corpus" / "stats" / "pipeline_stats.json"
SOURCE_LIST = ROOT / "data" / "sources" / "source_list.json"


def load_source_ids():
    if not SOURCE_LIST.exists():
        return {}

    with SOURCE_LIST.open("r", encoding="utf-8") as f:
        data = json.load(f)

    ids = {}
    for items in data.get("categories", {}).values():
        if isinstance(items, dict):
            items = items.values()
        if not isinstance(items, list):
            continue

        for item in items:
            if not isinstance(item, dict):
                continue
            filename = item.get("filename")
            doc_id = item.get("id")
            if filename and doc_id:
                ids[Path(filename).stem] = doc_id
    return ids


def count_tables(text):
    table_count = 0
    in_table = False

    for line in text.splitlines():
        stripped = line.strip()
        is_table_row = (
            stripped.startswith("|")
            and stripped.endswith("|")
            and stripped.count("|") >= 3
            and not stripped.startswith("|<")
            and not stripped.startswith("| <")
        )

        if is_table_row and not in_table:
            table_count += 1
            in_table = True
        elif not is_table_row:
            in_table = False

    return table_count


def extract_stats_from_text(text, filename, doc_id):
    scan_text = markdown_io.strip_code_fences(text)

    block_formulas = re.findall(r"\$\$(.+?)\$\$", scan_text, flags=re.DOTALL)
    inline_formulas = re.findall(r"\\\((.+?)\\\)", scan_text, flags=re.DOTALL)
    inline_dollar = re.findall(r"(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)", scan_text)
    inline_dollar = [s for s in inline_dollar if re.search(r"[\\\^_{}=&\|]", s)]
    images = re.findall(r"!\[[^\]]*\]\([^)]+\)", scan_text)

    return {
        "id": doc_id,
        "filename": filename,
        "formula_count": len(block_formulas) + len(inline_formulas) + len(inline_dollar),
        "block_formula_count": len(block_formulas),
        "inline_formula_count": len(inline_formulas) + len(inline_dollar),
        "table_count": count_tables(scan_text),
        "image_count": len(images),
        "char_count": len(text),
        "line_count": len(text.splitlines()),
    }


def extract_doc_stats(path, doc_id):
    text = path.read_text(encoding="utf-8")
    return extract_stats_from_text(text, path.stem, doc_id)


def build_stats_report(per_doc, failures=None):
    return {
        "total_docs": len(per_doc),
        "total_formulas": sum(item["formula_count"] for item in per_doc),
        "total_tables": sum(item["table_count"] for item in per_doc),
        "total_images": sum(item["image_count"] for item in per_doc),
        "total_chars": sum(item["char_count"] for item in per_doc),
        "total_lines": sum(item["line_count"] for item in per_doc),
        "failed_docs": len(failures) if failures else 0,
        "failures": failures or [],
        "per_doc": per_doc,
    }


def iter_markdown_files(input_path):
    return markdown_io.iter_markdown_files(input_path)


def build_stats(input_path, output_path):
    source_ids = load_source_ids()
    per_doc = []
    failures = []

    for index, md_path in enumerate(iter_markdown_files(input_path), start=1):
        doc_id = source_ids.get(md_path.stem, f"D{index:02d}")
        try:
            per_doc.append(extract_doc_stats(md_path, doc_id))
        except (OSError, UnicodeDecodeError) as exc:
            failures.append({"file": str(md_path), "error": str(exc)})
            continue

    stats = build_stats_report(per_doc, failures)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    return stats


def main():
    parser = argparse.ArgumentParser(description="Extract formula/table/image statistics from MinerU Markdown output.")
    parser.add_argument("input", nargs="?", default=str(DEFAULT_INPUT), help="Markdown file or directory. Defaults to data/processed.")
    parser.add_argument("-o", "--output", default=str(DEFAULT_OUTPUT), help="Output JSON path. Defaults to corpus/stats/corpus_stats.json.")
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()

    if not input_path.exists():
        raise SystemExit(f"Input not found: {input_path}")

    stats = build_stats(input_path, output_path)
    print(f"Saved: {output_path}")
    print(f"Docs: {stats['total_docs']}")
    print(f"Formulas: {stats['total_formulas']} | Tables: {stats['total_tables']} | Images: {stats['total_images']}")


if __name__ == "__main__":
    main()
