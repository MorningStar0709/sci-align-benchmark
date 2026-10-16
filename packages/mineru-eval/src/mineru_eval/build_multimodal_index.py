"""
build_multimodal_index.py — 跨模态可追溯索引构建器 (P1)

为 core.json 中 500 道评测题建立 source_ref → chunk → 图片/公式计数的全链路可追溯索引。
每个题目可追溯到原始 chunk 文件中的图片数量、公式数量和共现类型。

检测逻辑与 scan_multimodal_chunks.py 保持一致：
  图片: MinerU <details> 块 或 ![]() 引用
  公式: $$ 块级 或 $...$ 行内 LaTeX
"""

import json
import re
from pathlib import Path
from collections import Counter

PROJECT_ROOT = Path(__file__).resolve().parents[4]
CHUNKS_DIR = PROJECT_ROOT / "corpus" / "chunks"
CORE_PATH = PROJECT_ROOT / "benchmark" / "dataset" / "core.json"
MULTIMODAL_SCAN_PATH = PROJECT_ROOT / "benchmark" / "eval" / "results" / "multimodal_chunks.json"

ANY_DETAIL = re.compile(r'<details>\s*<summary>\w+</summary>', re.DOTALL)
MD_IMG = re.compile(r'!\[\]\(')
FM_BLOCK = re.compile(r'\$\$')
FM_INLINE = re.compile(r'\$[^$]+\$')


def count_images(text: str) -> int:
    details = len(ANY_DETAIL.findall(text))
    md_imgs = len(MD_IMG.findall(text))
    return details + md_imgs


def count_formulas(text: str) -> int:
    blocks = len(FM_BLOCK.findall(text)) // 2
    inlines = len(FM_INLINE.findall(text))
    return blocks + inlines


def classify_cooccurrence(image_count: int, formula_count: int) -> str:
    if image_count > 0 and formula_count > 0:
        return "image_formula"
    if image_count > 0:
        return "image_only"
    if formula_count > 0:
        return "formula_only"
    return "text_only"


def build_chunk_lookup():
    """构建 source_ref_stem → (split, relative_path) 的查找表。
    仅索引 core.json 中出现的 source_ref，避免扫描全量 28514 chunks。
    """
    with open(CORE_PATH, "r", encoding="utf-8") as f:
        core = json.load(f)

    source_refs = set()
    for q in core["questions"]:
        source_refs.add(q["source_ref"])

    lookup = {}
    for split in ("eval", "train"):
        split_dir = CHUNKS_DIR / split
        if not split_dir.exists():
            continue
        for md_file in split_dir.glob("*.md"):
            stem = md_file.stem
            if stem in source_refs:
                rel = str(md_file.relative_to(CHUNKS_DIR))
                lookup[stem] = (split, rel)

    return lookup, core


def load_multimodal_metadata():
    """加载 multimodal_chunks.json 中的候选元数据作为补充信息。"""
    if not MULTIMODAL_SCAN_PATH.exists():
        return {}
    with open(MULTIMODAL_SCAN_PATH, "r", encoding="utf-8") as f:
        scan = json.load(f)
    meta = {}
    for c in scan.get("candidates", []):
        stem = Path(c["path"]).stem
        meta[stem] = {
            "doc_id": c.get("doc_id", "?"),
            "subdomains": c.get("subdomains", []),
            "difficulty": c.get("difficulty", ""),
            "detection": c.get("detection", ""),
        }
    return meta


def main():
    print("=" * 64)
    print("  ControlSci 跨模态可追溯索引构建器 (P1)")
    print("  source_ref → chunk → image_count / formula_count / cooccurrence_type")
    print("=" * 64)

    lookup, core = build_chunk_lookup()
    multimodal_meta = load_multimodal_metadata()

    questions = core["questions"]
    total = len(questions)
    found = 0
    not_found_refs = []
    entries = []
    type_dist = Counter()
    split_dist = Counter()

    print(f"\n  题目总数: {total}")
    print(f"  source_ref 去重: {len(lookup)} 个")
    print(f"  multimodal_meta 已加载: {len(multimodal_meta)} 条\n")

    for i, q in enumerate(questions):
        source_ref = q["source_ref"]
        chunk_info = lookup.get(source_ref)

        if chunk_info is None:
            not_found_refs.append(source_ref)
            entries.append({
                "question_id": q["id"],
                "dimension": q["dimension"],
                "control_category": q.get("control_category", []),
                "source_ref": source_ref,
                "chunk_path": None,
                "split": None,
                "image_count": 0,
                "formula_count": 0,
                "cooccurrence_type": "not_found",
            })
            continue

        split, rel_path = chunk_info
        chunk_path = CHUNKS_DIR / rel_path
        content = chunk_path.read_text(encoding="utf-8")

        img_cnt = count_images(content)
        fm_cnt = count_formulas(content)
        cooc_type = classify_cooccurrence(img_cnt, fm_cnt)

        found += 1
        type_dist[cooc_type] += 1
        split_dist[split] += 1

        mm = multimodal_meta.get(source_ref, {})

        entries.append({
            "question_id": q["id"],
            "dimension": q["dimension"],
            "control_category": q.get("control_category", []),
            "source_ref": source_ref,
            "chunk_path": rel_path,
            "split": split,
            "image_count": img_cnt,
            "formula_count": fm_cnt,
            "cooccurrence_type": cooc_type,
            "multimodal_detection": mm.get("detection", ""),
            "subdomains": mm.get("subdomains", []),
        })

        if (i + 1) % 100 == 0:
            print(f"  进度: {i+1}/{total} ...", flush=True)

    match_rate = found / total * 100

    print(f"\n{'=' * 64}")
    print("  匹配统计")
    print(f"{'=' * 64}")
    print(f"  找到 chunk:     {found:>5} / {total}  ({match_rate:.1f}%)")
    print(f"  未找到:         {len(not_found_refs):>5}")
    print("\n  共现类型分布:")
    for t, c in type_dist.most_common():
        pct = c / (found or 1) * 100
        print(f"    {t:<18} {c:>5}  ({pct:.1f}%)")
    print("\n  Split 分布:")
    for s, c in split_dist.most_common():
        print(f"    {s:<18} {c:>5}")
    if not_found_refs:
        print("\n  未匹配 source_ref (前10):")
        for ref in not_found_refs[:10]:
            print(f"    - {ref}")

    dimensions = Counter(e["dimension"] for e in entries)
    print("\n  维度覆盖:")
    for d in ["A", "B", "C", "D"]:
        print(f"    维度 {d}: {dimensions.get(d, 0)}/{total//4}")

    output = {
        "build_time": "2026-05-07",
        "version": "1.0",
        "description": "ControlSci 跨模态可追溯索引 — 500 题 × chunk 文件 × 图片/公式计数",
        "summary": {
            "total_questions": total,
            "chunks_found": found,
            "chunks_not_found": len(not_found_refs),
            "match_rate_pct": round(match_rate, 1),
            "cooccurrence_distribution": dict(type_dist),
            "split_distribution": dict(split_dist),
            "dimension_coverage": dict(dimensions),
        },
        "not_found_refs": not_found_refs,
        "entries": entries,
    }

    out_path = PROJECT_ROOT / "benchmark" / "dataset" / "multimodal_index.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n  输出: {out_path}")
    print(f"  匹配率: {match_rate:.1f}% {'✅ ≥ 90%' if match_rate >= 90 else '❌ < 90%'}")
    print(f"\n{'=' * 64}")


if __name__ == "__main__":
    main()
