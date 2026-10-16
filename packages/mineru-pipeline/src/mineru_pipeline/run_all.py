import argparse
import json
import traceback
from pathlib import Path

from mineru_pipeline import clean_formulas
from mineru_pipeline import clean_tables
from mineru_pipeline import extract_stats
from mineru_pipeline import markdown_io
from mineru_pipeline import verify_order


# Phase 5: restored from new package location
ROOT = Path(__file__).resolve().parents[4]  # packages/mineru-pipeline/src/mineru_pipeline/ → project root
DEFAULT_INPUT = ROOT / "data" / "processed"
DEFAULT_STATS = ROOT / "corpus" / "stats" / "pipeline_stats.json"
DEFAULT_ORDER = ROOT / "corpus" / "stats" / "reading_order_report.json"


def process_one_file(md_path, doc_id, dry_run=False):
    original = md_path.read_text(encoding="utf-8")

    formula_cleaned = clean_formulas.clean_text(original)
    table_cleaned = clean_tables.clean_text(formula_cleaned)
    changed = table_cleaned != original

    if changed and not dry_run:
        md_path.write_text(table_cleaned, encoding="utf-8", newline="\n")

    stats_entry = extract_stats.extract_stats_from_text(table_cleaned, md_path.stem, doc_id)
    order_entry = verify_order.verify_order_from_text(table_cleaned, md_path.stem)

    return stats_entry, order_entry, changed


def find_duplicate_ids(stats_entries):
    counts = {}
    for entry in stats_entries:
        counts[entry["id"]] = counts.get(entry["id"], 0) + 1
    return sorted(doc_id for doc_id, count in counts.items() if count > 1)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Phase 2 Markdown post-processing: clean formulas, clean tables, extract stats, verify order."
    )
    parser.add_argument(
        "input_path",
        nargs="?",
        help="Markdown file or directory. Defaults to data/processed.",
    )
    parser.add_argument(
        "-i",
        "--input",
        dest="input_option",
        help="Markdown file or directory. Overrides positional input when both are provided.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Preview Markdown changes without writing files.")
    parser.add_argument(
        "--stats-output",
        default=str(DEFAULT_STATS),
        help="Output path for pipeline statistics JSON.",
    )
    parser.add_argument(
        "--order-output",
        default=str(DEFAULT_ORDER),
        help="Output path for reading-order report JSON.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    raw_input = args.input_option or args.input_path or str(DEFAULT_INPUT)
    input_path = Path(raw_input).resolve()
    stats_output = Path(args.stats_output).resolve()
    order_output = Path(args.order_output).resolve()

    if not input_path.exists():
        raise SystemExit(f"Input not found: {input_path}")

    source_ids = extract_stats.load_source_ids()
    md_paths = markdown_io.iter_markdown_files(input_path)
    total = len(md_paths)

    if total == 0:
        raise SystemExit(f"No Markdown files found: {input_path}")

    print(f"Input: {input_path}")
    print(f"Markdown files: {total}")
    print(f"Mode: {'dry-run' if args.dry_run else 'write'}")

    stats_entries = []
    order_entries = []
    failures = []

    for index, md_path in enumerate(md_paths, start=1):
        doc_id = source_ids.get(md_path.stem, f"D{index:02d}")
        print(f"[{index}/{total}] {md_path.stem} (id={doc_id})")

        try:
            stats_entry, order_entry, changed = process_one_file(md_path, doc_id, dry_run=args.dry_run)
        except Exception as exc:
            failure = {"path": str(md_path), "error": str(exc), "traceback": traceback.format_exc()}
            failures.append(failure)
            print(f"  FAILED: {exc}")
            continue

        stats_entries.append(stats_entry)
        order_entries.append(order_entry)

        action = "would update" if args.dry_run and changed else "updated" if changed else "unchanged"
        print(
            f"  {action}; {stats_entry['formula_count']} formulas, "
            f"{stats_entry['table_count']} tables, {stats_entry['image_count']} images, "
            f"{order_entry['issue_count']} order issues"
        )

    duplicate_ids = find_duplicate_ids(stats_entries)
    if duplicate_ids:
        print(f"WARNING: duplicate doc_id(s): {', '.join(duplicate_ids)}")

    stats = extract_stats.build_stats_report(stats_entries, failures)
    order = verify_order.build_order_report(order_entries, failures)

    stats_output.parent.mkdir(parents=True, exist_ok=True)
    order_output.parent.mkdir(parents=True, exist_ok=True)
    stats_output.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    order_output.write_text(json.dumps(order, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Stats saved: {stats_output}")
    print(f"Reading-order report saved: {order_output}")
    print(
        f"Done: {stats['total_docs']} docs, {stats['total_formulas']} formulas, "
        f"{stats['total_tables']} tables, {stats['total_images']} images, "
        f"{order['total_issues']} order issues, {len(failures)} failures"
    )

    if failures:
        print("Failures:")
        for failure in failures:
            print(f"  {failure['path']}: {failure['error']}")


if __name__ == "__main__":
    main()
