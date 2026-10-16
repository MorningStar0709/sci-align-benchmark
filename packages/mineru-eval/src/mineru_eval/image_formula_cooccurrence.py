"""image_formula_cooccurrence.py — 图片-公式共现统计 (Task 1.2b)

扫描 28,475 chunks，统计同时含图片+公式的 chunk 比例与子领域覆盖。
复用 build_corpus.py 的正则模式以保持一致性。
"""

import json
import re
from pathlib import Path
from collections import Counter

PROJECT_ROOT = Path(__file__).resolve().parents[4]
CHUNKS_DIR = PROJECT_ROOT / "corpus" / "chunks"
MANIFEST_PATH = PROJECT_ROOT / "corpus" / "chunks" / "chunks_manifest.json"
METADATA_PATH = PROJECT_ROOT / "corpus" / "metadata.json"


def strip_code_fences(text):
    return re.sub(r"```.*?```", "", text, flags=re.DOTALL)


def has_images(text):
    return bool(re.search(r"!\[([^\]]*)\]\(([^)]+)\)", text))


def has_formulas(text):
    scan = strip_code_fences(text)
    if re.search(r"\$\$(.+?)\$\$", scan, flags=re.DOTALL):
        return True
    if re.search(r"\\\((.+?)\\\)", scan, flags=re.DOTALL):
        return True
    for tex in re.findall(r"(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)", scan):
        if re.search(r"[\\\^_{}=&\|]", tex):
            return True
    return False


def load_doc_categories(metadata_path):
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    mapping = {}
    for doc in metadata.get("docs", []):
        mapping[doc["id"]] = doc.get("control_category", [])
    return mapping


def main():
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    doc_categories = load_doc_categories(METADATA_PATH)

    chunks = manifest["chunks"]
    total = len(chunks)

    has_both = 0
    image_only = 0
    formula_only = 0
    neither = 0
    read_errors = 0

    cooccur_subdomains = Counter()
    cooccur_examples = []

    for i, chunk in enumerate(chunks):
        if (i + 1) % 5000 == 0:
            print(f"  [{i+1}/{total}] scanned...", flush=True)

        filepath = CHUNKS_DIR / chunk["filepath"].replace("corpus/chunks/", "")
        try:
            content = filepath.read_text(encoding="utf-8")
        except (FileNotFoundError, UnicodeDecodeError):
            read_errors += 1
            continue

        img = has_images(content)
        fm = has_formulas(content)

        if img and fm:
            has_both += 1
            doc_id = chunk["doc_id"]
            for cat in doc_categories.get(doc_id, []):
                cooccur_subdomains[cat] += 1
            if len(cooccur_examples) < 5:
                cooccur_examples.append(chunk["chunk_id"])
        elif img and not fm:
            image_only += 1
        elif not img and fm:
            formula_only += 1
        else:
            neither += 1

    processed = total - read_errors
    pct_both = has_both / processed * 100 if processed else 0

    print(f"\n{'='*60}")
    print("  图片-公式共现统计 (image-formula co-occurrence)")
    print(f"{'='*60}")
    print(f"  总 chunks:             {total:>8}")
    print(f"  读取失败:              {read_errors:>8}")
    print("  ─────────────────────────────")
    print(f"  同时含图片+公式:       {has_both:>8}  ({pct_both:.1f}%)")
    print(f"  仅含图片:              {image_only:>8}  ({image_only/processed*100:.1f}%)")
    print(f"  仅含公式:              {formula_only:>8}  ({formula_only/processed*100:.1f}%)")
    print(f"  两者均无:              {neither:>8}  ({neither/processed*100:.1f}%)")
    print(f"{'='*60}")
    print(f"  图片-公式共现覆盖子领域: {len(cooccur_subdomains)} 个")
    print("\n  子领域分布 (top 14):")
    for cat, count in cooccur_subdomains.most_common():
        bar_len = int(count / max(cooccur_subdomains.values()) * 30) if cooccur_subdomains else 0
        print(f"    {cat:<16} {count:>6}  {'█' * bar_len}")
    print("\n  共现 chunk 示例 (前5):")
    for ex in cooccur_examples:
        print(f"    {ex}")

    result = {
        "total_chunks_manifest": total,
        "chunks_scanned": processed,
        "read_errors": read_errors,
        "has_both": has_both,
        "has_both_pct": round(pct_both, 1),
        "image_only": image_only,
        "image_only_pct": round(image_only / processed * 100, 1) if processed else 0,
        "formula_only": formula_only,
        "formula_only_pct": round(formula_only / processed * 100, 1) if processed else 0,
        "neither": neither,
        "neither_pct": round(neither / processed * 100, 1) if processed else 0,
        "subdomain_count": len(cooccur_subdomains),
        "subdomains": dict(cooccur_subdomains.most_common()),
        "summary": (
            f"{pct_both:.1f}% 的语料 chunk 实现了图片-公式共现，"
            f"覆盖 {len(cooccur_subdomains)} 个控制子领域"
        ),
    }

    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "image_formula_cooccurrence.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\n  结果已写入: {output_path}")
    print(f"\n  📊 摘要: {result['summary']}")


if __name__ == "__main__":
    main()
