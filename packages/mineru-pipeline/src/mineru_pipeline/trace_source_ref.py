"""C2 精确溯源 — 题目↔语料chunk 反向匹配索引

从 core.json 的 source_ref 字段解析题目来源，与 metadata.json 文档对齐，
构建双向索引并统计覆盖率。

source_ref 格式: "{arxiv_id}_{title_slug}_chunk_{NNN}"
  例: "2201.05599_Performance_Analysis_chunk_047"

产出:
  - dataset/source_trace.json  双向索引 + 覆盖率统计
"""

import argparse
import re
from collections import defaultdict
from pathlib import Path

# Phase 5: restored from new package location
ROOT = Path(__file__).resolve().parents[4]  # packages/mineru-pipeline/src/mineru_pipeline/ → project root

from mineru_core.utils import load_json, write_json_atomic

DEFAULT_CORE = ROOT / "benchmark" / "dataset" / "core.json"
DEFAULT_METADATA = ROOT / "corpus" / "metadata.json"
DEFAULT_OUTPUT = ROOT / "dataset" / "source_trace.json"

SOURCE_REF_PATTERN = re.compile(
    r"^(?P<arxiv_id>\d{4}\.\d{4,5})_(?P<title_slug>.+?)_chunk_(?P<chunk>\d{3,})$"
)


def parse_source_ref(raw):
    if not raw:
        return None
    m = SOURCE_REF_PATTERN.match(raw)
    if not m:
        return None
    return {
        "arxiv_id": m.group("arxiv_id"),
        "title_slug": m.group("title_slug"),
        "chunk": m.group("chunk"),
    }


def build_metadata_index(metadata):
    index = {}
    for doc in metadata.get("docs", []):
        aid = doc.get("arxiv_id", "")
        if aid:
            index[aid] = doc
    return index


def trace_sources(core_data, metadata_index):
    questions = core_data.get("questions", [])
    q_to_source = {}
    source_to_q = defaultdict(list)
    unmatched = []
    matched_count = 0

    for q in questions:
        qid = q.get("id", "?")
        raw = q.get("source_ref", "")
        parsed = parse_source_ref(raw)

        if not parsed:
            reason = "empty" if not raw or not raw.strip() else "malformed"
            unmatched.append({"id": qid, "source_ref": raw, "reason": reason})
            continue

        arxiv_id = parsed["arxiv_id"]
        doc = metadata_index.get(arxiv_id)

        entry = {
            "question_id": qid,
            "dimension": q.get("dimension", ""),
            "difficulty": q.get("difficulty_level", ""),
            "source_ref": raw,
            "arxiv_id": arxiv_id,
            "chunk": parsed["chunk"],
        }

        if doc:
            entry["doc_id"] = doc.get("id", "")
            entry["doc_title"] = doc.get("title", "")
            entry["doc_author"] = doc.get("author", "")
            entry["doc_year"] = doc.get("year", 0)
            entry["doc_category"] = doc.get("control_category", [])
            entry["matched"] = True
            matched_count += 1
        else:
            entry["doc_id"] = ""
            entry["doc_title"] = ""
            entry["matched"] = False

        q_to_source[qid] = entry
        source_to_q[arxiv_id].append(entry)

    return q_to_source, dict(source_to_q), unmatched, matched_count


def main():
    parser = argparse.ArgumentParser(description="题目↔语料chunk 反向匹配索引 (C2 精确溯源)")
    parser.add_argument("--core", default=str(DEFAULT_CORE), help="core.json 路径")
    parser.add_argument("--metadata", default=str(DEFAULT_METADATA), help="metadata.json 路径")
    parser.add_argument("--output", "-o", default=str(DEFAULT_OUTPUT), help="输出路径")
    parser.add_argument("--show-unmatched", action="store_true", help="显示未匹配详情")
    args = parser.parse_args()

    core_path = Path(args.core)
    meta_path = Path(args.metadata)
    if not core_path.exists():
        raise SystemExit(f"core.json 不存在: {core_path}")
    if not meta_path.exists():
        raise SystemExit(f"metadata.json 不存在: {meta_path}")

    core_data = load_json(core_path)
    metadata = load_json(meta_path)
    total_q = len(core_data.get("questions", []))

    meta_index = build_metadata_index(metadata)
    print(f"metadata 索引: {len(meta_index)} 篇文档 (含 arxiv_id)")

    q_to_source, source_to_q, unmatched, matched_count = trace_sources(core_data, meta_index)

    report = {
        "meta": {
            "project": "ControlSci C2 精确溯源",
            "core_questions": total_q,
            "source_ref_populated": total_q - len(unmatched),
            "matched_to_metadata": matched_count,
            "unmatched": len(unmatched),
            "coverage": round(matched_count / total_q * 100, 1) if total_q else 0,
            "unique_sources": len(source_to_q),
        },
        "question_index": q_to_source,
        "source_index": source_to_q,
        "unmatched": unmatched,
    }

    write_json_atomic(args.output, report)
    print(f"[OK] 溯源报告已写入: {args.output}")
    print(f"  题目总数: {total_q}")
    print(f"  source_ref 可解析: {total_q - len(unmatched)}")
    print(f"  匹配 metadata: {matched_count}/{total_q} ({report['meta']['coverage']}%)")
    print(f"  唯一来源文档: {len(source_to_q)}")

    if args.show_unmatched and unmatched:
        print(f"\n  未匹配详情 ({len(unmatched)}):")
        for item in unmatched[:20]:
            print(f"    {item['id']}: {item['source_ref'] or '(空)'}")


if __name__ == "__main__":
    main()
