"""
轻量级 metadata.json 重建脚本 (方案B)
输入: data/processed/ (362 dirs) + data/sources/ (source_list.json, arxiv_candidates.json) + build_corpus.py FALLBACK_METADATA
输出: corpus/metadata.json  (不复制文件，纯元数据生成)
"""

import json
import os
from pathlib import Path
from datetime import date
from mineru_pipeline._constants import FALLBACK_METADATA, CATEGORY_DIFFICULTY_MAP

# Phase 5: restored from new package location
ROOT = Path(__file__).resolve().parents[4]  # packages/mineru-pipeline/src/mineru_pipeline/ → project root
PROCESSED = ROOT / "data" / "processed"
SOURCE_LIST = ROOT / "data" / "sources" / "source_list.json"
ARXIV_CANDIDATES = ROOT / "data" / "sources" / "arxiv_candidates.json"
OUTPUT = ROOT / "corpus"
METADATA_OUT = OUTPUT / "metadata.json"


def _build_lookup_tables():
    txt_by_dirname = {}
    if SOURCE_LIST.exists():
        with open(SOURCE_LIST, encoding="utf-8") as f:
            sl = json.load(f)
        for t in sl["categories"]["textbooks"]:
            filename = t["filename"].replace(".pdf", "")
            txt_by_dirname[filename] = {
                "title": t["title"],
                "author": t["author"],
                "source": t.get("source", ""),
                "copyright": t.get("copyright", ""),
                "control_category": t.get("control_category", ""),
                "size_mb": t.get("size_mb"),
            }

    txt_fallback = {}
    for dirname, info in FALLBACK_METADATA.items():
        txt_fallback[dirname] = info

    arxiv_lookup = {}
    if ARXIV_CANDIDATES.exists():
        with open(ARXIV_CANDIDATES, encoding="utf-8") as f:
            ac = json.load(f)
        for direction in ac:
            dn = direction.get("direction_cn", direction.get("direction", ""))
            for c in direction["candidates"]:
                arxiv_lookup[c["arxiv_id"]] = {
                    "title": c["title"],
                    "author": ", ".join(c.get("authors", [])),
                    "source": f"arXiv: {c['arxiv_id']}",
                    "copyright": "arXiv Open Access",
                    "control_category": dn,
                    "year": c.get("year"),
                    "arxiv_id": c["arxiv_id"],
                    "direction": direction.get("direction", ""),
                }
    return txt_by_dirname, txt_fallback, arxiv_lookup


def _resolve_meta(dirname, txt_by_dirname, txt_fallback, arxiv_lookup):
    parts = dirname.split("_", 1)
    if parts[0].count(".") >= 1 and any(c.isdigit() for c in parts[0]):
        arxiv_id = parts[0]
        doc_type = "arxiv"
        if arxiv_id in arxiv_lookup:
            m = arxiv_lookup[arxiv_id]
            meta = {
                "arxiv_id": arxiv_id,
                "title": m["title"],
                "author": m["author"],
                "source": m["source"],
                "copyright": m["copyright"],
                "control_category": m["control_category"],
                "year": m.get("year"),
            }
        else:
            title_part = parts[1] if len(parts) > 1 else dirname
            title_clean = title_part[:100] if title_part else dirname
            meta = {
                "arxiv_id": arxiv_id,
                "title": dirname,
                "author": "Unknown / arXiv",
                "source": f"arXiv: {arxiv_id}",
                "copyright": "arXiv Open Access",
                "control_category": "控制理论",
                "year": int(arxiv_id[:2]) + 2000 if arxiv_id[:2].isdigit() else None,
            }
        return meta, doc_type, True
    else:
        doc_type = "textbook"
        if dirname in txt_by_dirname:
            meta = txt_by_dirname[dirname]
        elif dirname in txt_fallback:
            meta = txt_fallback[dirname]
        else:
            meta = {
                "title": dirname,
                "author": "Unknown / To be verified",
                "source": "中文教材 / Chinese Textbook",
                "copyright": "Educational / Copyright holder to be verified",
                "control_category": "控制理论",
            }
        return meta, doc_type, False


def _resolve_difficulty(meta):
    raw = meta.get("control_category", "控制理论")
    if not raw or not isinstance(raw, str):
        return "L3"
    cats = [c.strip() for c in raw.split(",")]
    difficulty_candidates = []
    for cat in cats:
        if cat in CATEGORY_DIFFICULTY_MAP:
            difficulty_candidates.append(CATEGORY_DIFFICULTY_MAP[cat])
        elif cat.lower() in CATEGORY_DIFFICULTY_MAP:
            difficulty_candidates.append(CATEGORY_DIFFICULTY_MAP[cat.lower()])
    difficulty = "L3"
    if difficulty_candidates:
        order = {"L1": 1, "L2": 2, "L3": 3, "L4": 4}
        difficulty = min(difficulty_candidates, key=lambda x: order.get(x, 3))
    return difficulty


def _write_output(docs, textbook_count, arxiv_matched, arxiv_fallback):
    OUTPUT.mkdir(parents=True, exist_ok=True)

    metadata = {
        "project": "ControlSci — 控制科学结构化语料库",
        "version": "1.0-light",
        "updated": str(date.today()),
        "total_docs": len(docs),
        "textbooks": textbook_count,
        "arxiv_papers": len(docs) - textbook_count,
        "categories": {"textbooks": [], "arxiv_papers": []},
        "generation_note": f"方案B轻量重建: {arxiv_matched} arXiv from candidates, "
                          f"{arxiv_fallback} arXiv fallback, {textbook_count} textbooks",
        "docs": docs,
    }

    with open(METADATA_OUT, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print("[OK] corpus/metadata.json 已生成", flush=True)
    print(f"     {len(docs)} 篇文档", flush=True)
    print(f"     {textbook_count} 教材 + {len(docs) - textbook_count} 论文", flush=True)
    print(f"     arXiv: {arxiv_matched} from candidates + {arxiv_fallback} fallback", flush=True)
    print(f"     输出: {METADATA_OUT}  ({METADATA_OUT.stat().st_size / 1024:.1f} KB)", flush=True)

    empty_author = sum(1 for d in docs if not d.get("author") or d["author"] == "Unknown")
    empty_cat = sum(1 for d in docs if not d.get("control_category"))
    print("\n[VALIDATION]", flush=True)
    print(f"     author 空/Unknown: {empty_author}/{len(docs)}", flush=True)
    print(f"     control_category 空: {empty_cat}/{len(docs)}", flush=True)


def main():
    processed_dirs = sorted(d for d in os.listdir(PROCESSED) if os.path.isdir(PROCESSED / d))

    txt_by_dirname, txt_fallback, arxiv_lookup = _build_lookup_tables()

    docs = []
    textbook_count = 0
    arxiv_matched = 0
    arxiv_fallback = 0

    for i, dirname in enumerate(processed_dirs, 1):
        md_path = PROCESSED / dirname
        md_files = list(md_path.glob("*.md"))
        file_size = sum(f.stat().st_size for f in md_files) if md_files else 0

        doc_id = f"D{i:02d}"
        meta, doc_type, is_arxiv = _resolve_meta(dirname, txt_by_dirname, txt_fallback, arxiv_lookup)
        if is_arxiv:
            arxiv_matched += 1 if meta.get("arxiv_id") and meta.get("title") != dirname else 0
            arxiv_fallback += 1 if meta.get("title") == dirname else 0
        else:
            textbook_count += 1

        difficulty = _resolve_difficulty(meta)

        doc = {
            "id": doc_id,
            "type": doc_type,
            "filename": dirname,
            "title": meta.get("title", dirname),
            "author": meta.get("author", "Unknown"),
            "source": meta.get("source", ""),
            "copyright": meta.get("copyright", ""),
            "control_category": [c.strip() for c in meta.get("control_category", "控制理论").split(",")],
            "difficulty_level": difficulty,
            "file_size_bytes": file_size,
        }
        if meta.get("size_mb"):
            doc["size_mb"] = meta["size_mb"]
        if meta.get("arxiv_id"):
            doc["arxiv_id"] = meta["arxiv_id"]
        if meta.get("year"):
            doc["year"] = meta["year"]

        docs.append(doc)

    _write_output(docs, textbook_count, arxiv_matched, arxiv_fallback)


if __name__ == "__main__":
    main()
