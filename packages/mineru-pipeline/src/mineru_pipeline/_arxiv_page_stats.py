"""统计 arXiv PDF 页数分布，校验 page/rate 阈值，评估全量解析时间。
Usage: .venv/bin/python pipeline/_arxiv_page_stats.py
"""
import json
import sys
import time
from pathlib import Path

from pypdf import PdfReader

ARXIV_DIR = Path(__file__).resolve().parents[4] / "data" / "pdf" / "arXiv"
SAMPLE_SIZE = 50  # 采样 50 篇做页数统计，避免扫全部 342 篇耗时过长


def get_page_count(path):
    try:
        reader = PdfReader(str(path))
        return len(reader.pages)
    except Exception:
        return None


def main():
    pdfs = sorted(ARXIV_DIR.glob("*.pdf"))
    total = len(pdfs)
    total_mb = sum(p.stat().st_size for p in pdfs) / (1024 * 1024)
    print(f"arXiv PDF count: {total}")
    print(f"Total size: {total_mb:.1f} MB")
    print(f"\nSampling {min(SAMPLE_SIZE, total)} files for page count...")

    sample = pdfs[:SAMPLE_SIZE]
    pages = []
    failures = 0

    t0 = time.time()
    for i, path in enumerate(sample, start=1):
        n = get_page_count(path)
        if n is not None:
            pages.append(n)
        else:
            failures += 1
        sys.stdout.write(f"\r  [{i}/{len(sample)}] {path.stem[:50]}...")
        sys.stdout.flush()

    elapsed = time.time() - t0
    avg_pages = sum(pages) / len(pages) if pages else 0
    min_pages = min(pages) if pages else 0
    max_pages = max(pages) if pages else 0

    # MB/页 比率
    sample_mb = sum(p.stat().st_size for p in sample) / (1024 * 1024)
    sample_total_pages = sum(pages)
    mb_per_page = sample_mb / sample_total_pages if sample_total_pages else 0

    print(f"\n\nSample results ({len(pages)} valid / {failures} failed):")
    print(f"  Avg pages:  {avg_pages:.1f}")
    print(f"  Min pages:  {min_pages}")
    print(f"  Max pages:  {max_pages}")
    print(f"  Total pages (sample): {sample_total_pages}")
    print(f"  Sample size: {sample_mb:.1f} MB")
    print(f"  MB/page:     {mb_per_page:.4f}")
    print(f"  Scan time:   {elapsed:.1f}s ({elapsed/len(sample):.2f}s per file)")

    # 全量估算（假设采样有代表性）
    estimated_total_pages = int(avg_pages * total)
    print("\n=== Full estimate ===")
    print(f"  Estimated total pages: {estimated_total_pages}")

    # 如果 MB/page ≤ 0.04 → 数字 PDF (50-80 页/分钟)
    # 如果 MB/page > 0.04  → 扫描版 (20-40 页/分钟)
    # arXiv 论文预期都是纯数字出生
    rate_optimistic = estimated_total_pages / 80  # 分钟
    rate_normal = estimated_total_pages / 60
    rate_conservative = estimated_total_pages / 50

    print(f"  Type classification: {'Digital' if mb_per_page <= 0.04 else 'Scanner'}")
    print(f"  MB/page = {mb_per_page:.4f}  (threshold=0.04)")
    print("\n  Estimated parse time (GPU only, no overhead):")
    print(f"    Optimistic (80 p/m):  {rate_optimistic:.1f} min ({rate_optimistic/60:.1f} h)")
    print(f"    Normal    (60 p/m):  {rate_normal:.1f} min ({rate_normal/60:.1f} h)")
    print(f"    Conservative (50 p/m): {rate_conservative:.1f} min ({rate_conservative/60:.1f} h)")

    # 加 API 开销（每篇 ~30s 提交+轮询+下载固定开销）
    overhead_total = total * 30 / 60  # 分钟
    print("\n  With API overhead (~30s per doc):")
    print(f"    API overhead: {overhead_total:.0f} min ({overhead_total/60:.1f} h)")
    for label, rate_min in [("Optimistic", 80), ("Normal", 60), ("Conservative", 50)]:
        gpu_min = estimated_total_pages / rate_min
        total_min = gpu_min + overhead_total
        print(f"    {label:15s}: {total_min:.0f} min ({total_min/60:.1f} h)")

    # 输出 JSON 供后续参考
    result = {
        "total_pdfs": total,
        "total_size_mb": round(total_mb, 1),
        "sample_size": len(pages),
        "sample_avg_pages": round(avg_pages, 1),
        "sample_min_pages": min_pages,
        "sample_max_pages": max_pages,
        "sample_total_pages": sample_total_pages,
        "mb_per_page": round(mb_per_page, 4),
        "type": "digital" if mb_per_page <= 0.04 else "scanner",
        "estimated_total_pages": estimated_total_pages,
        "estimate": {
            "optimistic_hours": round(estimated_total_pages / 80 / 60, 1),
            "normal_hours": round(estimated_total_pages / 60 / 60, 1),
            "conservative_hours": round(estimated_total_pages / 50 / 60, 1),
        },
        "with_overhead": {
            "optimistic_hours": round(estimated_total_pages / 80 / 60 + overhead_total / 60, 1),
            "normal_hours": round(estimated_total_pages / 60 / 60 + overhead_total / 60, 1),
            "conservative_hours": round(estimated_total_pages / 50 / 60 + overhead_total / 60, 1),
        },
    }
    result_path = Path(__file__).resolve().parents[4] / "corpus" / "stats" / "arxiv_page_estimate.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nResult saved: {result_path}")


if __name__ == "__main__":
    main()
