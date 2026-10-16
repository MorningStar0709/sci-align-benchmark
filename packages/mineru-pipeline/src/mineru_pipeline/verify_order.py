import argparse
import json
import re
from pathlib import Path

from mineru_pipeline import markdown_io


# Phase 5: computed from new package location
ROOT = Path(__file__).resolve().parents[4]  # packages/mineru-pipeline/src/mineru_pipeline/ → project root
DEFAULT_INPUT = ROOT / "data" / "processed"
DEFAULT_OUTPUT = ROOT / "corpus" / "stats" / "reading_order_report.json"


SENTENCE_ENDINGS = tuple(".。!！?？:：;；)]）}\"'")


def paragraph_blocks(text):
    blocks = []
    start_line = None
    current = []
    in_code_fence = False

    def flush_current():
        nonlocal start_line, current
        if current:
            blocks.append((start_line, " ".join(current)))
            current = []
            start_line = None

    for line_no, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if stripped.startswith("```"):
            flush_current()
            in_code_fence = not in_code_fence
            continue

        if in_code_fence:
            continue

        if not stripped:
            flush_current()
            continue

        if stripped.startswith(("![](", "|", "#", "<")):
            flush_current()
            continue

        if start_line is None:
            start_line = line_no
        current.append(stripped)

    flush_current()

    return blocks


def looks_like_order_issue(prev_text, next_text):
    if not prev_text or not next_text:
        return False

    prev_tail = prev_text.rstrip()
    next_head = next_text.lstrip()

    if prev_tail.endswith(("...", "……")):
        return True

    if not prev_tail.endswith(SENTENCE_ENDINGS) and re.match(r"^[A-Z0-9a-z(（]", next_head):
        return True

    if re.match(r"^[a-z,.;:)\]}]", next_head):
        return True

    return False


def verify_order_from_text(text, filename):
    blocks = paragraph_blocks(text)
    issues = []

    for index in range(len(blocks) - 1):
        line_no, paragraph = blocks[index]
        next_line_no, next_paragraph = blocks[index + 1]
        if looks_like_order_issue(paragraph, next_paragraph):
            issues.append(
                {
                    "line": line_no,
                    "next_line": next_line_no,
                    "previous_tail": paragraph[-120:],
                    "next_head": next_paragraph[:120],
                    "reason": "possible paragraph break or reading-order discontinuity",
                }
            )

    return {"filename": filename, "issue_count": len(issues), "issues": issues}


def verify_file(path):
    text = path.read_text(encoding="utf-8")
    return verify_order_from_text(text, path.stem)


def build_order_report(per_doc, failures=None):
    return {
        "total_docs": len(per_doc),
        "total_issues": sum(item["issue_count"] for item in per_doc),
        "failed_docs": len(failures) if failures else 0,
        "failures": failures or [],
        "per_doc": per_doc,
    }


def iter_markdown_files(input_path):
    return markdown_io.iter_markdown_files(input_path)


def build_report(input_path, output_path):
    per_doc = [verify_file(path) for path in iter_markdown_files(input_path)]
    report = build_order_report(per_doc)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description="Detect possible reading-order discontinuities in MinerU Markdown files.")
    parser.add_argument("input", nargs="?", default=str(DEFAULT_INPUT), help="Markdown file or directory. Defaults to data/processed.")
    parser.add_argument("-o", "--output", default=str(DEFAULT_OUTPUT), help="Output JSON report path.")
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()

    if not input_path.exists():
        raise SystemExit(f"Input not found: {input_path}")

    report = build_report(input_path, output_path)
    print(f"Saved: {output_path}")
    print(f"Docs: {report['total_docs']} | Possible issues: {report['total_issues']}")


if __name__ == "__main__":
    main()
