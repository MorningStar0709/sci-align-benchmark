import argparse
import json
import re
import shutil
from datetime import date
from pathlib import Path

from mineru_pipeline import markdown_io
from mineru_pipeline._constants import FALLBACK_METADATA, CATEGORY_DIFFICULTY_MAP

# Phase 5: restored from new package location
ROOT = Path(__file__).resolve().parents[4]  # packages/mineru-pipeline/src/mineru_pipeline/ → project root
DEFAULT_INPUT = ROOT / "data" / "processed"
DEFAULT_OUTPUT = ROOT / "corpus"
SOURCE_LIST = ROOT / "data" / "sources" / "source_list.json"


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def iter_source_items(value):
    if isinstance(value, dict):
        for child in value.values():
            yield from iter_source_items(child)
    elif isinstance(value, list):
        for item in value:
            if isinstance(item, dict) and "filename" in item:
                yield item
            else:
                yield from iter_source_items(item)


def load_source_metadata(path=SOURCE_LIST):
    if not path.exists():
        return {}

    data = json.loads(path.read_text(encoding="utf-8"))
    sources = {}
    for item in iter_source_items(data.get("categories", {})):
        filename = item.get("filename")
        if filename:
            sources[Path(filename).stem] = item
    return sources


def split_categories(value):
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if not value:
        return []
    return [part.strip() for part in re.split(r"[,，]", str(value)) if part.strip()]


def iter_document_markdown(input_path):
    if input_path.is_file():
        return [input_path]

    if input_path.name != "processed" and (input_path / f"{input_path.name}.md").exists():
        return [input_path / f"{input_path.name}.md"]

    docs = []
    for doc_dir in sorted(path for path in input_path.iterdir() if path.is_dir()):
        expected = doc_dir / f"{doc_dir.name}.md"
        if expected.exists():
            docs.append(expected)
            continue

        candidates = sorted(path for path in doc_dir.rglob("*.md") if not path.stem.endswith("_original"))
        if candidates:
            docs.append(candidates[0])
    return docs


def is_single_document_input(input_path):
    if input_path.is_file():
        return True
    return input_path.name != "processed" and (input_path / f"{input_path.name}.md").exists()


def extract_formulas(text):
    scan_text = markdown_io.strip_code_fences(text)
    formulas = []

    for tex in re.findall(r"\$\$(.+?)\$\$", scan_text, flags=re.DOTALL):
        formulas.append({"tex": tex.strip(), "type": "block", "index": len(formulas)})

    for tex in re.findall(r"\\\((.+?)\\\)", scan_text, flags=re.DOTALL):
        formulas.append({"tex": tex.strip(), "type": "inline", "index": len(formulas)})

    for tex in re.findall(r"(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)", scan_text):
        if re.search(r"[\\\^_{}=&\|]", tex):
            formulas.append({"tex": tex.strip(), "type": "inline", "index": len(formulas)})

    return formulas


def line_number_for_offset(line_starts, offset):
    low = 0
    high = len(line_starts) - 1
    while low <= high:
        mid = (low + high) // 2
        if line_starts[mid] <= offset:
            low = mid + 1
        else:
            high = mid - 1
    return max(1, high + 1)


def extract_figures(text, doc_dir):
    lines = text.splitlines()
    line_starts = []
    offset = 0
    for line in lines:
        line_starts.append(offset)
        offset += len(line) + 1

    figures = []
    for match in re.finditer(r"!\[([^\]]*)\]\(([^)]+)\)", text):
        target = match.group(2).strip()
        line_no = line_number_for_offset(line_starts, match.start())
        image_filename = Path(target).name
        candidate = doc_dir / "image" / image_filename
        start = max(0, line_no - 3)
        end = min(len(lines), line_no + 2)
        surrounding = [line for index, line in enumerate(lines[start:end], start=start + 1) if index != line_no]

        figures.append(
            {
                "image_path": str(candidate.resolve()) if candidate.exists() else None,
                "image_filename": image_filename,
                "markdown_ref": match.group(0),
                "line_number": line_no,
                "surrounding_text": "\n".join(surrounding).strip(),
            }
        )

    return figures


def metadata_for_doc(doc_name, md_path, source, doc_id):
    fallback = FALLBACK_METADATA.get(doc_name, {})
    if not source and doc_name not in FALLBACK_METADATA:
        print(f"Warning: no metadata for {doc_name}, using defaults", flush=True)
    cats = split_categories(source.get("control_category") or fallback.get("control_category", ""))
    difficulty = "L3"  # default
    if cats:
        level_vals = [CATEGORY_DIFFICULTY_MAP.get(c, "L3") for c in cats]
        level_order = {"L1": 1, "L2": 2, "L3": 3, "L4": 4}
        min_level = min(level_vals, key=lambda x: level_order.get(x, 3))
        difficulty = min_level
    return {
        "id": doc_id,
        "filename": doc_name,
        "title": source.get("title") or fallback.get("title") or doc_name,
        "author": source.get("author") or fallback.get("author") or "Unknown / To be verified",
        "source": source.get("source") or fallback.get("source") or "",
        "copyright": source.get("copyright") or fallback.get("copyright") or "",
        "control_category": cats,
        "difficulty_level": difficulty,
        "size_mb": source.get("size_mb"),
        "file_size_bytes": md_path.stat().st_size,
    }


def load_existing_metadata(output_root):
    metadata_path = output_root / "metadata.json"
    if not metadata_path.exists():
        return []

    data = json.loads(metadata_path.read_text(encoding="utf-8"))
    return [item for item in data.get("docs", []) if isinstance(item, dict) and item.get("filename")]


def next_dynamic_id(existing_docs):
    max_index = 0
    for doc in existing_docs:
        doc_id = str(doc.get("id", ""))
        if re.fullmatch(r"D\d+", doc_id):
            max_index = max(max_index, int(doc_id[1:]))
    return f"D{max_index + 1:02d}"


def assign_doc_id(doc_name, source, existing_docs, dynamic_index):
    if source.get("id"):
        return str(source["id"]), dynamic_index

    for doc in existing_docs:
        if doc.get("filename") == doc_name and doc.get("id"):
            return str(doc["id"]), dynamic_index

    return f"D{dynamic_index:02d}", dynamic_index + 1


def build_corpus(input_path, output_root, source_list=SOURCE_LIST):
    source_metadata = load_source_metadata(source_list)
    processed_root = output_root / "processed"
    processed_root.mkdir(parents=True, exist_ok=True)

    md_paths = iter_document_markdown(input_path)
    single_doc_mode = is_single_document_input(input_path)
    existing_docs = load_existing_metadata(output_root) if single_doc_mode else []
    existing_by_filename = {doc["filename"]: doc for doc in existing_docs}
    updated_names = set()
    dynamic_index = int(next_dynamic_id(existing_docs)[1:]) if single_doc_mode else 1

    for md_path in md_paths:
        doc_name = md_path.stem
        source = source_metadata.get(doc_name, {})
        doc_id, dynamic_index = assign_doc_id(doc_name, source, existing_docs, dynamic_index)

        text = md_path.read_text(encoding="utf-8")
        formulas = extract_formulas(text)
        figures = extract_figures(text, md_path.parent)

        doc_output = processed_root / doc_name
        if doc_output.exists():
            shutil.rmtree(doc_output)
        doc_output.mkdir(parents=True, exist_ok=True)

        shutil.copyfile(md_path, doc_output / "full_text.md")
        write_json(doc_output / "formulas.json", formulas)
        write_json(doc_output / "figures.json", figures)

        doc_meta = metadata_for_doc(doc_name, md_path, source, str(doc_id))
        doc_meta["image_count"] = len(figures)
        doc_meta["formula_count"] = len(formulas)
        existing_by_filename[doc_name] = doc_meta
        updated_names.add(doc_name)

    if single_doc_mode:
        docs = sorted(existing_by_filename.values(), key=lambda item: item["filename"])
    else:
        docs = sorted((existing_by_filename[name] for name in updated_names), key=lambda item: item["filename"])
        stale_dirs = [path for path in processed_root.iterdir() if path.is_dir() and path.name not in updated_names]
        for stale_dir in stale_dirs:
            shutil.rmtree(stale_dir)

    metadata = {
        "project": "ControlSci — 控制科学结构化语料库",
        "version": "1.0",
        "updated": date.today().isoformat(),
        "total_docs": len(docs),
        "docs": docs,
    }
    write_json(output_root / "metadata.json", metadata)
    return metadata


def main():
    parser = argparse.ArgumentParser(description="Build the structured ControlSci corpus from processed Markdown.")
    parser.add_argument("-i", "--input", default=str(DEFAULT_INPUT), help="Processed Markdown file or directory.")
    parser.add_argument("-o", "--output", default=str(DEFAULT_OUTPUT), help="Corpus output directory.")
    parser.add_argument("--source-list", default=str(SOURCE_LIST), help="Source metadata JSON file.")
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    output_root = Path(args.output).resolve()
    source_list = Path(args.source_list).resolve()

    if not input_path.exists():
        raise SystemExit(f"Input not found: {input_path}")

    metadata = build_corpus(input_path, output_root, source_list)
    print(f"Saved: {output_root / 'metadata.json'}")
    print(f"Docs: {metadata['total_docs']}")


if __name__ == "__main__":
    main()
