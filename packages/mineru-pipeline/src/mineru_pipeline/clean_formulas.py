import argparse
import re
from pathlib import Path

from mineru_pipeline import markdown_io


ROOT = Path(__file__).resolve().parents[4]  # packages/mineru-pipeline/src/mineru_pipeline/ → project root
DEFAULT_INPUT = ROOT / "data" / "processed"


PRESERVE_NEWLINES_PATTERNS = (
    r"\\begin\{(?:aligned|align|array|matrix|pmatrix|bmatrix|cases|split|gathered)\}",
    r"\\\\",
)
BLOCK_FORMULA_RE = re.compile(r"\$\$(.{1,10000}?)\$\$", flags=re.DOTALL)
INLINE_FORMULA_RE = re.compile(r"\\\((.{1,1000}?)\\\)", flags=re.DOTALL)


def normalize_block_formula(match):
    body = match.group(1).strip()

    preserve_newlines = any(re.search(pattern, body) for pattern in PRESERVE_NEWLINES_PATTERNS)
    if preserve_newlines:
        body = re.sub(r"\n{3,}", "\n\n", body)
        return f"$$\n{body}\n$$"

    body = re.sub(r"\s*\n\s*", " ", body)
    body = re.sub(r"[ \t]{2,}", " ", body).strip()
    return f"$${body}$$"


def normalize_inline_formula(match):
    body = re.sub(r"\s+", " ", match.group(1)).strip()
    return rf"\({body}\)"


def clean_text(text):
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    parts = re.split(r"(```.*?```)", normalized, flags=re.DOTALL)

    for index, part in enumerate(parts):
        if part.startswith("```"):
            continue
        parts[index] = clean_formulas_outside_code(part)

    return "".join(parts)


def clean_formulas_outside_code(text):
    cleaned = BLOCK_FORMULA_RE.sub(normalize_block_formula, text)
    cleaned = INLINE_FORMULA_RE.sub(normalize_inline_formula, cleaned)

    # Remove empty formulas left by parser noise.
    cleaned = re.sub(r"\$\$\s*\$\$", "", cleaned)
    cleaned = re.sub(r"\\\(\s*\\\)", "", cleaned)

    return cleaned


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
    parser = argparse.ArgumentParser(description="Clean common LaTeX formula formatting issues in MinerU Markdown files.")
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
