import argparse
import json
import re
import shutil
from pathlib import Path


# Phase 5: restored from new package location
ROOT = Path(__file__).resolve().parents[4]  # packages/mineru-pipeline/src/mineru_pipeline/ → project root
DEFAULT_CORPUS = ROOT / "corpus"

_CHAR_CLEAN_RE = re.compile(r"\s+")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.。；;!?！？])\s+")
_PARAGRAPH_SPLIT_RE = re.compile(r"\n\s*\n")
_ATX_CLEAN_RE = re.compile(r"^#{1,6}\s+|\s+#+\s*$")

try:
    from _tools.medical_section_detector import enrich_sections
    HAS_MEDICAL_DETECTOR = True
except ImportError:
    HAS_MEDICAL_DETECTOR = False


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def estimate_tokens(text):
    words = len(text.split())
    cjk_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    ascii_chars = sum(1 for c in text if c.isascii() and not c.isspace())
    return int(max(words * 1.3, cjk_chars * 1.5 + ascii_chars * 0.25))


def line_spans(text):
    spans = []
    offset = 0
    for line_no, line in enumerate(text.splitlines(True), start=1):
        spans.append((offset, offset + len(line), line_no))
        offset += len(line)
    if not spans:
        spans.append((0, 0, 1))
    return spans


def line_for_offset(spans, offset):
    for start, end, line_no in spans:
        if start <= offset < end:
            return line_no
    return spans[-1][2]


def split_sections(text):
    headings = list(re.finditer(r"^#{1,6}\s+.*$", text, flags=re.MULTILINE))
    spans = line_spans(text)

    if not headings:
        return [
            {
                "text": text.strip(),
                "section": "Document",
                "line_start": 1,
                "line_end": len(text.splitlines()) or 1,
            }
        ] if text.strip() else []

    sections = []
    if headings[0].start() > 0 and text[: headings[0].start()].strip():
        prefix = text[: headings[0].start()].strip()
        sections.append(
            {
                "text": prefix,
                "section": "Front Matter",
                "line_start": 1,
                "line_end": line_for_offset(spans, headings[0].start()),
            }
        )

    for index, heading in enumerate(headings):
        start = heading.start()
        end = headings[index + 1].start() if index + 1 < len(headings) else len(text)
        section_text = text[start:end].strip()
        if not section_text:
            continue
        section_clean = _ATX_CLEAN_RE.sub("", heading.group(0)).strip()
        sections.append(
            {
                "text": section_text,
                "section": section_clean,
                "line_start": line_for_offset(spans, start),
                "line_end": line_for_offset(spans, max(start, end - 1)),
            }
        )
    return sections


def split_long_paragraph(paragraph, max_tokens):
    sentences = _SENTENCE_SPLIT_RE.split(paragraph)
    parts = []
    current = []

    for sentence in sentences:
        if not sentence:
            continue
        candidate = "\n".join([*current, sentence]) if current else sentence
        if current and estimate_tokens(candidate) > max_tokens:
            parts.append("\n".join(current).strip())
            current = [sentence]
        else:
            current.append(sentence)

    if current:
        parts.append("\n".join(current).strip())

    return parts or [paragraph]


def split_oversized_section(section, min_tokens, max_tokens):
    paragraphs = [part.strip() for part in _PARAGRAPH_SPLIT_RE.split(section["text"]) if part.strip()]
    chunks = []
    current = []

    def flush():
        if current:
            chunks.append("\n\n".join(current).strip())
            current.clear()

    for paragraph in paragraphs:
        para_tokens = estimate_tokens(paragraph)
        paragraph_parts = (
            split_long_paragraph(paragraph, max_tokens)
            if para_tokens > max_tokens
            else [paragraph]
        )
        for part in paragraph_parts:
            candidate = "\n\n".join([*current, part]) if current else part
            if current and estimate_tokens(candidate) > max_tokens and estimate_tokens("\n\n".join(current)) >= min_tokens:
                flush()
            current.append(part)

    flush()

    merged = []
    for chunk in chunks:
        if merged and estimate_tokens(chunk) < min_tokens:
            merged[-1] = f"{merged[-1]}\n\n{chunk}".strip()
        else:
            merged.append(chunk)

    result = []
    for chunk in merged:
        entry = {
            "text": chunk,
            "section": section["section"],
            "line_start": section["line_start"],
            "line_end": section["line_end"],
        }
        if "medical_label" in section:
            entry["medical_label"] = section["medical_label"]
        if "medical_parent" in section:
            entry["medical_parent"] = section["medical_parent"]
        result.append(entry)
    return result


def chunk_sections(sections, min_tokens, max_tokens):
    chunks = []
    for section in sections:
        if estimate_tokens(section["text"]) <= max_tokens:
            chunks.append(section)
        else:
            chunks.extend(split_oversized_section(section, min_tokens, max_tokens))
    return chunks


def load_metadata(corpus_root):
    metadata_path = corpus_root / "metadata.json"
    if not metadata_path.exists():
        return {}
    try:
        data = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        print(f"Warning: corrupted metadata.json ({e}), proceeding without metadata")
        return {}
    return {item["filename"]: item for item in data.get("docs", [])}


def split_set(index, total, split_ratio):
    train_count = int(total * split_ratio)
    if total > 1:
        train_count = max(1, min(total - 1, train_count))
    else:
        train_count = total
    return "train" if index < train_count else "eval"


def load_existing_manifest(chunks_root):
    manifest_path = chunks_root / "chunks_manifest.json"
    if not manifest_path.exists():
        return {
            "total_chunks": 0,
            "train_chunks": 0,
            "eval_chunks": 0,
            "chunks": [],
        }
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def remove_doc_chunks(chunks_root, manifest_chunks, doc_name):
    remaining = []
    for item in manifest_chunks:
        if item.get("filename") == doc_name:
            path = ROOT / item.get("filepath", "")
            if path.exists():
                path.unlink()
        else:
            remaining.append(item)

    for split_dir in (chunks_root / "train", chunks_root / "eval"):
        if split_dir.exists():
            for path in split_dir.glob(f"{doc_name}_chunk_*.md"):
                path.unlink()

    return remaining


def build_manifest(manifest_chunks, split_ratio, target_tokens, min_tokens, max_tokens):
    manifest_chunks = sorted(manifest_chunks, key=lambda item: (item["filename"], item["chunk_id"]))
    return {
        "total_chunks": len(manifest_chunks),
        "train_chunks": sum(1 for item in manifest_chunks if item["set"] == "train"),
        "eval_chunks": sum(1 for item in manifest_chunks if item["set"] == "eval"),
        "split_ratio": split_ratio,
        "target_tokens": target_tokens,
        "min_tokens": min_tokens,
        "max_tokens": max_tokens,
        "chunks": manifest_chunks,
    }


def _find_text_file(doc_dir, doc_name_hint=None):
    primary = doc_dir / "full_text.md"
    if primary.exists():
        return primary
    if doc_name_hint:
        named = doc_dir / f"{doc_name_hint}.md"
        if named.exists():
            return named
    candidates = sorted(
        p for p in doc_dir.rglob("*.md")
        if not p.stem.endswith("_original")
    )
    return candidates[0] if candidates else None


def _setup_chunk_env(corpus_root, doc_name, chunks_dir):
    metadata = load_metadata(corpus_root)
    if (corpus_root / "processed").is_dir():
        processed_root = corpus_root / "processed"
    else:
        processed_root = corpus_root
    chunks_root = Path(chunks_dir) if chunks_dir else corpus_root / "chunks"

    if chunks_root.exists() and not doc_name:
        shutil.rmtree(chunks_root)
    (chunks_root / "train").mkdir(parents=True, exist_ok=True)
    (chunks_root / "eval").mkdir(parents=True, exist_ok=True)

    docs = sorted(path for path in processed_root.iterdir() if path.is_dir())
    if doc_name:
        docs = [path for path in docs if path.name == doc_name]
        if not docs:
            raise SystemExit(f"Document not found in corpus: {doc_name}")
    return metadata, chunks_root, docs


def _build_chunk_entry(doc_dir, chunk, index, chunks, chunks_root, split_ratio, doc_meta):
    set_name = split_set(index, len(chunks), split_ratio)
    chunk_id = f"{doc_dir.name}_chunk_{index:03d}"
    output_path = chunks_root / set_name / f"{chunk_id}.md"
    output_path.write_text(chunk["text"].strip() + "\n", encoding="utf-8", newline="\n")
    entry = {
        "chunk_id": chunk_id,
        "doc_id": doc_meta.get("id", ""),
        "filename": doc_dir.name,
        "set": set_name,
        "filepath": str(output_path.relative_to(ROOT)).replace("\\", "/"),
        "estimated_tokens": estimate_tokens(chunk["text"]),
        "source_section": chunk["section"],
        "source_line_start": chunk["line_start"],
        "source_line_end": chunk["line_end"],
    }
    if "medical_label" in chunk:
        entry["medical_label"] = chunk["medical_label"]
    if "medical_parent" in chunk:
        entry["medical_parent"] = chunk["medical_parent"]
    return entry



def build_chunks(corpus_root, doc_name=None, target_tokens=768, min_tokens=256, max_tokens=1024,
                 split_ratio=0.8, medical_mode=False, chunks_dir=None):
    metadata, chunks_root, docs = _setup_chunk_env(corpus_root, doc_name, chunks_dir)

    if doc_name:
        existing_manifest = load_existing_manifest(chunks_root)
        manifest_chunks = remove_doc_chunks(chunks_root, existing_manifest.get("chunks", []), doc_name)
    else:
        manifest_chunks = []

    for doc_dir in docs:
        text_path = _find_text_file(doc_dir, doc_name_hint=doc_dir.name)
        if text_path is None:
            print(f"Warning: missing text file for {doc_dir.name}")
            continue
        text = text_path.read_text(encoding="utf-8")
        sections = split_sections(text)
        if medical_mode and HAS_MEDICAL_DETECTOR:
            sections = enrich_sections(sections)
        chunks = chunk_sections(sections, min_tokens=min_tokens, max_tokens=max_tokens)
        if not chunks:
            print(f"Warning: no chunks generated for {text_path}")
            continue
        doc_meta = metadata.get(doc_dir.name, {})

        for index, chunk in enumerate(chunks):
            entry = _build_chunk_entry(doc_dir, chunk, index, chunks, chunks_root, split_ratio, doc_meta)
            manifest_chunks.append(entry)

    manifest = build_manifest(manifest_chunks, split_ratio, target_tokens, min_tokens, max_tokens)
    write_json(chunks_root / "chunks_manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description="Chunk the structured corpus into train/eval Markdown files.")
    parser.add_argument("-c", "--corpus", default=str(DEFAULT_CORPUS), help="Corpus root directory.")
    parser.add_argument("-d", "--doc", help="Only chunk one document by corpus processed directory name.")
    parser.add_argument("--target-tokens", type=int, default=768, help="Preferred token estimate per chunk.")
    parser.add_argument("--min-tokens", type=int, default=256, help="Minimum token estimate before splitting.")
    parser.add_argument("--max-tokens", type=int, default=1024, help="Maximum token estimate per chunk.")
    parser.add_argument("--split-ratio", type=float, default=0.8, help="Train split ratio per document.")
    parser.add_argument("--tokenizer-estimate", default="simple", choices=["simple"], help="Token estimation strategy.")
    parser.add_argument("--medical-mode", action="store_true", help="Enable medical section hierarchy detection (IMRAD + clinical subsections).")
    args = parser.parse_args()

    corpus_root = Path(args.corpus).resolve()
    if not corpus_root.exists():
        raise SystemExit(f"Corpus not found: {corpus_root}")
    if not 0 < args.split_ratio <= 1:
        raise SystemExit("--split-ratio must be within (0, 1]")

    manifest = build_chunks(
        corpus_root=corpus_root,
        doc_name=args.doc,
        target_tokens=args.target_tokens,
        min_tokens=args.min_tokens,
        max_tokens=args.max_tokens,
        split_ratio=args.split_ratio,
        medical_mode=args.medical_mode,
    )
    print(f"Saved: {corpus_root / 'chunks' / 'chunks_manifest.json'}")
    print(f"Chunks: {manifest['total_chunks']} | Train: {manifest['train_chunks']} | Eval: {manifest['eval_chunks']}")


if __name__ == "__main__":
    main()
