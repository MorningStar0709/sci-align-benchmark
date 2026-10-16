import argparse
import re
from pathlib import Path

from mineru_pipeline import markdown_io


ROOT = Path(__file__).resolve().parents[4]  # packages/mineru-pipeline/src/mineru_pipeline/ → project root
DEFAULT_INPUT = ROOT / "data" / "processed"


def is_table_row(line):
    stripped = line.strip()
    return (
        stripped.startswith("|")
        and stripped.endswith("|")
        and stripped.count("|") >= 3
        and not stripped.startswith("|<")
        and not stripped.startswith("| <")
    )


def is_separator_row(line):
    cells = split_cells(line)
    return bool(cells) and all(re.fullmatch(r":?-{3,}:?", cell.strip()) for cell in cells)


def split_cells(line):
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def format_row(cells):
    return "| " + " | ".join(cell.strip() for cell in cells) + " |"


def separator_for(width):
    return "| " + " | ".join("---" for _ in range(width)) + " |"


def normalize_table_block(lines):
    if len(lines) == 1:
        return lines

    rows = [split_cells(line) for line in lines]
    width = max(len(row) for row in rows)
    normalized_rows = [row + [""] * (width - len(row)) for row in rows]

    result = [format_row(normalized_rows[0])]
    if not is_separator_row(lines[1]):
        result.append(separator_for(width))

    start = 1
    if len(normalized_rows) > 1 and is_separator_row(lines[1]):
        result.append(separator_for(width))
        start = 2

    for row in normalized_rows[start:]:
        result.append(format_row(row))

    return result


def clean_text(text):
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    parts = re.split(r"(```.*?```)", normalized, flags=re.DOTALL)

    for index, part in enumerate(parts):
        if part.startswith("```"):
            continue
        parts[index] = clean_tables_outside_code(part)

    return "".join(parts)


def clean_tables_outside_code(text):
    lines = text.split("\n")
    output = []
    block = []

    def flush_block():
        if not block:
            return
        if len(block) >= 2:
            output.extend(normalize_table_block(block))
        else:
            output.extend(block)
        block.clear()

    for line in lines:
        if is_table_row(line):
            block.append(line)
        else:
            flush_block()
            output.append(line)

    flush_block()
    return "\n".join(output)


def iter_markdown_files(input_path):
    return markdown_io.iter_markdown_files(input_path)


def clean_file(path, dry_run=False):
    original = path.read_text(encoding="utf-8")
    cleaned = clean_text(original)
    changed = original != cleaned

    if changed and not dry_run:
        path.write_text(cleaned, encoding="utf-8", newline="\n")

    return changed


def main():
    parser = argparse.ArgumentParser(description="Normalize Markdown pipe tables in MinerU Markdown files.")
    parser.add_argument("input", nargs="?", default=str(DEFAULT_INPUT), help="Markdown file or directory. Defaults to data/processed.")
    parser.add_argument("--dry-run", action="store_true", help="Report changed files without writing them.")
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    if not input_path.exists():
        raise SystemExit(f"Input not found: {input_path}")

    changed_files = []
    for md_path in iter_markdown_files(input_path):
        if clean_file(md_path, dry_run=args.dry_run):
            changed_files.append(str(md_path))

    action = "Would update" if args.dry_run else "Updated"
    print(f"{action} {len(changed_files)} file(s).")
    for path in changed_files:
        print(path)


if __name__ == "__main__":
    main()
