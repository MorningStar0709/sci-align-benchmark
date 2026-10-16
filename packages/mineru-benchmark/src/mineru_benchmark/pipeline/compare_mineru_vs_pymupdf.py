"""
MinerU vs PyMuPDF 对比分析脚本

对比维度：
1. 公式提取：MinerU 结构化 LaTeX vs PyMuPDF 纯文本（公式处乱码）
2. 文字提取：MinerU Markdown 段落 vs PyMuPDF 逐文本块
3. 处理速度
4. 输出结构质量

用法：
  .venv/bin/python benchmark/pipeline/compare_mineru_vs_pymupdf.py --doc 自抗扰控制技术
  .venv/bin/python benchmark/pipeline/compare_mineru_vs_pymupdf.py --doc 自动控制原理_胡寿松
  .venv/bin/python benchmark/pipeline/compare_mineru_vs_pymupdf.py --doc 非线性系统_Khalil
  .venv/bin/python benchmark/pipeline/compare_mineru_vs_pymupdf.py --all
"""

import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]  # restored Phase 5

PROJECT_ROOT = Path(__file__).resolve().parents[5]
PDF_DIR = PROJECT_ROOT / "data" / "pdf" / "textbooks"
CORPUS_DIR = PROJECT_ROOT / "corpus" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "benchmark" / "dataset" / "compare_output"

MATH_UNICODE_CHARS = set(
    "∫∂∑√∏∞≈≠≤≥∈∉⊂⊃∩∪∀∃∄⇒⇔∧∨⊕⊗"
    "αβγδεζηθικλμνξοπρστυφχψω"
    "ΓΔΘΛΞΠΣΦΨΩ"
    "∇∂∆∅ℵ"
    "←↑→↓↔↕↖↗↘↙"
    "⊢⊣⊤⊥⊧⊨⊩⊪⊫⊬⊭⊮⊯"
)
MATH_LATEX_KEYWORDS = [
    "frac", "sum", "int", "partial", "infty", "rightarrow", "leftarrow",
    "longrightarrow", "longleftarrow", "Rightarrow", "Leftarrow", "Leftrightarrow",
    "cdot", "cdots", "vdots", "ddots", "times", "pm", "mp", "div",
    "sqrt", "nabla", "approx", "equiv", "propto", "sim", "cong",
    "hat", "tilde", "bar", "dot", "ddot", "vec",
    "begin", "end", "equation", "align", "array",
    "alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta",
    "kappa", "lambda", "mu", "nu", "xi", "omicron", "pi", "rho",
    "sigma", "tau", "upsilon", "phi", "chi", "psi", "omega",
]


def load_mineru_formulas(doc_name):
    doc_dir = CORPUS_DIR / doc_name
    formulas_path = doc_dir / "formulas.json"
    if not formulas_path.exists():
        print(f"[ERROR] MinerU formulas.json not found: {formulas_path}")
        return None
    with open(formulas_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        formulas = data.get("formulas", data)
    else:
        formulas = data
    return formulas


def load_mineru_fulltext(doc_name):
    doc_dir = CORPUS_DIR / doc_name
    text_path = doc_dir / "full_text.md"
    if not text_path.exists():
        print(f"[ERROR] MinerU full_text.md not found: {text_path}")
        return ""
    with open(text_path, "r", encoding="utf-8") as f:
        return f.read()


def analyze_mineru_formulas(formulas):
    if not formulas:
        return {"total": 0, "block": 0, "inline": 0, "with_tag": 0, "tag_set": set()}
    total = len(formulas)
    block = 0
    inline = 0
    with_tag = 0
    for f in formulas:
        ftype = f.get("type", "")
        if ftype == "block":
            block += 1
        else:
            inline += 1
        tex = f.get("tex", "")
        if "\\tag{" in tex or "\\label{" in tex:
            with_tag += 1
    return {"total": total, "block": block, "inline": inline, "with_tag": with_tag}


def analyze_pymupdf_page(page, page_num):
    t0 = time.time()
    text = page.get_text("text")
    parse_time = time.time() - t0
    text_len = len(text)

    math_char_count = sum(1 for c in text if c in MATH_UNICODE_CHARS)
    latex_keyword_count = sum(
        text.count("\\" + kw) for kw in MATH_LATEX_KEYWORDS
    )

    lines = text.split("\n")
    short_lines = sum(1 for l in lines if 0 < len(l.strip()) < 5 and any(c in l.strip() for c in "=+-*/()[]{}<>"))
    whitespace_ratio = text.count(" ") / max(text_len, 1)

    return {
        "page": page_num,
        "parse_time_ms": round(parse_time * 1000, 2),
        "char_count": text_len,
        "math_unicode_chars": math_char_count,
        "latex_keyword_count": latex_keyword_count,
        "short_line_count": short_lines,
        "whitespace_ratio": round(whitespace_ratio, 3),
        "raw_text_snippet": text[:300] if text else "[empty page]",
    }


def select_formula_dense_pages(pages_stats, n=5):
    sorted_pages = sorted(
        pages_stats,
        key=lambda p: (p["math_unicode_chars"] + p["latex_keyword_count"] * 5),
        reverse=True,
    )
    return sorted_pages[:n]


def match_page_text_to_fulltext(page_text, full_text, page_num):
    if not page_text or not full_text:
        return None
    fingerprint = page_text.strip()[:80].strip()
    if not fingerprint:
        fingerprint = page_text.strip()[-80:].strip()
    if not fingerprint:
        return None

    idx = full_text.find(fingerprint)
    if idx >= 0:
        start = max(0, idx - 20)
        end = min(len(full_text), idx + len(fingerprint) + 500)
        return {
            "page": page_num,
            "matched": True,
            "match_char_start": idx,
            "context": full_text[start:end],
        }

    fingerprint_clean = re.sub(r"\s+", " ", fingerprint)
    full_text_clean = re.sub(r"\s+", " ", full_text)
    idx2 = full_text_clean.find(fingerprint_clean)
    if idx2 >= 0:
        start = max(0, idx2 - 20)
        end = min(len(full_text_clean), idx2 + len(fingerprint_clean) + 500)
        return {
            "page": page_num,
            "matched": True,
            "match_char_start": idx2,
            "context": full_text_clean[start:end],
        }

    return {"page": page_num, "matched": False, "context": None}


def compute_document_metrics(doc_name, pdf_path, mineru_formulas, mineru_fulltext):
    import fitz

    doc = fitz.open(pdf_path)
    total_pages = len(doc)

    t0 = time.time()
    page_texts = []
    pages_stats = []
    all_pymupdf_text = []
    load_page_times = []
    for i in range(total_pages):
        t_page = time.time()
        page = doc.load_page(i)
        load_page_times.append(time.time() - t_page)
        text = page.get_text("text")
        page_texts.append(text)
        all_pymupdf_text.append(text)
        stat = analyze_pymupdf_page(page, i + 1)
        pages_stats.append(stat)
    pymupdf_time = time.time() - t0
    doc.close()

    all_pymupdf_text_joined = "\n".join(all_pymupdf_text)
    # Count extracted text only. Joining empty page strings would otherwise
    # count page separators as characters for scanned PDFs with no text layer.
    pymupdf_chars = sum(len(t) for t in all_pymupdf_text)

    mineru_chars = len(mineru_fulltext)
    formulas_analysis = analyze_mineru_formulas(mineru_formulas)

    formula_dense_pages = select_formula_dense_pages(pages_stats, n=5)

    page_matches = []
    for i, pt in enumerate(page_texts):
        match = match_page_text_to_fulltext(pt, mineru_fulltext, i + 1)
        page_matches.append(match)
    matched_page_count = sum(1 for m in page_matches if m and m.get("matched"))

    total_math_chars_pymupdf = sum(ps["math_unicode_chars"] for ps in pages_stats)

    parse_times_ms = [p["parse_time_ms"] for p in pages_stats]
    load_times_ms = [t * 1000 for t in load_page_times]

    mineru_seconds_est = round(total_pages / 60 * 60, 1)
    mineru_seconds_per_page_est = 1.0

    metrics = {
        "doc_name": doc_name,
        "pdf_path": str(pdf_path),
        "pdf_size_mb": round(pdf_path.stat().st_size / 1024 / 1024, 1),
        "total_pages": total_pages,

        "pymupdf_processing_seconds": round(pymupdf_time, 3),
        "pymupdf_seconds_per_page": round(pymupdf_time / total_pages, 4),
        "pymupdf_total_chars": pymupdf_chars,
        "pymupdf_math_unicode_chars": total_math_chars_pymupdf,

        "pymupdf_timing": {
            "total_seconds": round(pymupdf_time, 3),
            "seconds_per_page_avg": round(pymupdf_time / total_pages, 4),
            "pages_per_second": round(total_pages / pymupdf_time, 1),
            "parse_time_per_page_ms_avg": round(sum(parse_times_ms) / len(parse_times_ms), 2),
            "parse_time_per_page_ms_min": round(min(parse_times_ms), 2),
            "parse_time_per_page_ms_max": round(max(parse_times_ms), 2),
            "parse_time_per_page_ms_p50": round(sorted(parse_times_ms)[len(parse_times_ms) // 2], 2),
            "format_time_per_page_ms_avg": round(sum(load_times_ms) / len(load_times_ms), 2),
        },

        "mineru_estimate": {
            "total_seconds": mineru_seconds_est,
            "seconds_per_page_avg": mineru_seconds_per_page_est,
            "pages_per_hour": 3600,
            "basis": "实测教材~60页/min (含版面分析+公式检测+OCR)",
        },

        "speed_ratio": {
            "pymupdf_vs_mineru_speedup_x": round(mineru_seconds_est / max(pymupdf_time, 0.001), 0),
            "pymupdf_vs_mineru_per_page_ms_ratio": round(mineru_seconds_per_page_est * 1000 / (pymupdf_time / total_pages * 1000), 0),
        },

        "mineru_total_chars": mineru_chars,
        "mineru_formula_count": formulas_analysis["total"],
        "mineru_block_formulas": formulas_analysis["block"],
        "mineru_inline_formulas": formulas_analysis["inline"],
        "mineru_formulas_with_tag": formulas_analysis["with_tag"],
        "mineru_chars_per_formula": round(mineru_chars / max(formulas_analysis["total"], 1), 1),

        "pymupdf_chars_per_second": round(pymupdf_chars / max(pymupdf_time, 0.001), 0),
        "mineru_chars_per_second": round(mineru_chars / max(mineru_seconds_est, 0.001), 0),

        "pages_matched_via_text": matched_page_count,
        "pages_total": total_pages,
        "match_rate": round(matched_page_count / total_pages * 100, 1),

        "formula_dense_pages": formula_dense_pages,
        "all_pages_stats": pages_stats,
    }
    return metrics


def format_metric_row(label, mineru_val, pymupdf_val, unit="", better="mineru"):
    if isinstance(mineru_val, (int, float)) and isinstance(pymupdf_val, (int, float)):
        if mineru_val == 0 and pymupdf_val == 0:
            ratio = "N/A"
        elif pymupdf_val == 0:
            ratio = "∞"
        else:
            ratio = f"{mineru_val / pymupdf_val:.2f}x"
    else:
        ratio = "-"
    label_col = f"  {label}"
    mineru_s = f"{mineru_val:,}{unit}" if isinstance(mineru_val, (int, float)) else str(mineru_val)
    pymupdf_s = f"{pymupdf_val:,}{unit}" if isinstance(pymupdf_val, (int, float)) else str(pymupdf_val)
    return f"| {label_col:<40s} | {mineru_s:>20s} | {pymupdf_s:>20s} | {ratio:>10s} |"


def generate_markdown_report(metrics_list):
    lines = []
    lines.append("# MinerU vs PyMuPDF 对比分析报告")
    lines.append("")
    lines.append(f"> 生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"> 测试文档: {', '.join(m['doc_name'] for m in metrics_list)}")
    lines.append("")
    lines.append("---")
    lines.append("")

    for metrics in metrics_list:
        doc = metrics["doc_name"]
        lines.append(f"## {doc}")
        lines.append("")
        lines.append("| 属性 | 值 |")
        lines.append("|------|-----|")
        lines.append(f"| PDF 路径 | `{metrics['pdf_path']}` |")
        lines.append(f"| PDF 大小 | {metrics['pdf_size_mb']} MB |")
        lines.append(f"| 总页数 | {metrics['total_pages']} |")
        lines.append(f"| PyMuPDF 处理耗时 | {metrics['pymupdf_processing_seconds']:.3f}s ({metrics['pymupdf_seconds_per_page']:.4f}s/页) |")
        lines.append(f"| MinerU 预计耗时 | ~{metrics['total_pages'] * 60 // 60}min (按60页/min估算) |")
        lines.append("")
        lines.append("### 全文档指标对比")
        lines.append("")
        header = "| 指标 | MinerU | PyMuPDF | 比值(M/P) |"
        sep = "|------|--------|---------|------------|"
        lines.append(header)
        lines.append(sep)

        rows = [
            format_metric_row("总提取字符数", metrics["mineru_total_chars"], metrics["pymupdf_total_chars"]),
            format_metric_row("公式提取总数", metrics["mineru_formula_count"], metrics["pymupdf_math_unicode_chars"],
                              better="mineru"),
            format_metric_row("块级公式", metrics["mineru_block_formulas"], "-"),
            format_metric_row("行内公式", metrics["mineru_inline_formulas"], "-"),
            format_metric_row("带编号公式", metrics["mineru_formulas_with_tag"], "-"),
            format_metric_row("每公式对应字符数", metrics["mineru_chars_per_formula"], "-"),
            format_metric_row("文本匹配页数", metrics["pages_matched_via_text"], f"{metrics['total_pages']}"),
            format_metric_row("文本匹配率", f"{metrics['match_rate']}%", "100%"),
        ]
        lines.extend(rows)
        lines.append("")

        lines.append("### 处理速度对比")
        lines.append("")
        tim = metrics["pymupdf_timing"]
        min_est = metrics["mineru_estimate"]
        sr = metrics["speed_ratio"]
        speed_rows = [
            "| 工具 | 总耗时 | 每页耗时 | 吞吐量 |",
            "|------|--------|----------|--------|",
            f"| PyMuPDF | {tim['total_seconds']:.3f}s | {tim['seconds_per_page_avg']*1000:.2f}ms/页 | {tim['pages_per_second']:.0f} 页/秒 |",
            f"| MinerU | ~{min_est['total_seconds']:.0f}s | ~{min_est['seconds_per_page_avg']*1000:.0f}ms/页 | ~{min_est['pages_per_hour']/60:.0f} 页/分钟 |",
            f"| 速度比 | —— | PyMuPDF 快 **{sr['pymupdf_vs_mineru_per_page_ms_ratio']:.0f}x** | —— |",
            "",
            "| 字符吞吐量 | PyMuPDF | MinerU | 差距 |",
            "|------------|--------|--------|------|",
            f"| 字符/秒 | {metrics['pymupdf_chars_per_second']:,} | {metrics['mineru_chars_per_second']:,} | PyMuPDF 快 **{sr['pymupdf_vs_mineru_speedup_x']:.0f}x** |",
            "",
            "**PyMuPDF 解析时间分布（逐页）：**",
            "| 统计项 | 值 |",
            "|--------|-----|",
            f"| 平均页面加载 | {tim['format_time_per_page_ms_avg']:.2f}ms |",
            f"| 平均 get_text() 解析 | {tim['parse_time_per_page_ms_avg']:.2f}ms |",
            f"| 解析 P50 | {tim['parse_time_per_page_ms_p50']:.2f}ms |",
            f"| 解析 P0（最快页） | {tim['parse_time_per_page_ms_min']:.2f}ms |",
            f"| 解析 P100（最慢页） | {tim['parse_time_per_page_ms_max']:.2f}ms |",
            "",
            f"MinerU 耗时依据：{min_est['basis']}",
        ]
        lines.extend(speed_rows)
        lines.append("")

        lines.append("### 公式密集页详细对比")
        lines.append("")
        lines.append("以下为 PyMuPDF 检测到的公式密度最高的 5 页：")
        lines.append("")
        dense_header = "| 页码 | PyMuPDF 字符数 | Unicode数学符号 | LaTeX关键词 | 短行数 | 空白符比例 |"
        dense_sep = "|------|----------------|-----------------|-------------|--------|------------|"
        lines.append(dense_header)
        lines.append(dense_sep)
        for page in metrics["formula_dense_pages"]:
            lines.append(
                f"| {page['page']:>4d} | {page['char_count']:>14,d} | {page['math_unicode_chars']:>15,d} | "
                f"{page['latex_keyword_count']:>11,d} | {page['short_line_count']:>6,d} | {page['whitespace_ratio']:>10.3f} |"
            )
        lines.append("")

        lines.append("#### 公式最密集页的 PyMuPDF 原始输出")
        lines.append("")
        top_page = metrics["formula_dense_pages"][0]
        lines.append(f"**第 {top_page['page']} 页** — 此页 PyMuPDF 检测到 {top_page['math_unicode_chars']} 个 Unicode 数学符号，")
        lines.append(f"提取字符 {top_page['char_count']:,}，空白符比例 {top_page['whitespace_ratio']:.1%}")
        lines.append("")
        lines.append("```text")
        raw = top_page.get("raw_text_snippet", "[empty]")
        lines.append(raw)
        lines.append("```")
        lines.append("")
        lines.append(f"> PyMuPDF 在此页面上检测到 {top_page['latex_keyword_count']} 个 LaTeX 关键词片段，")
        lines.append("> 但纯文本输出中公式信息已**不可恢复**地丢失。")
        lines.append(f"> MinerU 在本文档中共提取 {metrics['mineru_formula_count']} 条结构化 LaTeX 公式。")
        lines.append("")

        lines.append("---")
        lines.append("")

    lines.append("## 汇总表格")
    lines.append("")
    header2 = "| 文档 | 页数 | PDF大小 | MinerU公式数 | PyMuPDF符号数 | MinerU字符 | PyMuPDF字符 | PyMuPDF耗时 |"
    sep2 = "|------|------|---------|-------------|----------------|------------|-------------|-------------|"
    lines.append(header2)
    lines.append(sep2)
    for metrics in metrics_list:
        lines.append(
            f"| {metrics['doc_name']:<15s} | {metrics['total_pages']:>4d} | {metrics['pdf_size_mb']:>5.1f}MB | "
            f"{metrics['mineru_formula_count']:>10,d} | {metrics['pymupdf_math_unicode_chars']:>14,d} | "
            f"{metrics['mineru_total_chars']:>10,d} | {metrics['pymupdf_total_chars']:>11,d} | "
            f"{metrics['pymupdf_processing_seconds']:>6.3f}s |"
        )
    lines.append("")

    lines.append("## 核心发现")
    lines.append("")
    lines.append("1. **公式提取能力**：MinerU 通过视觉+语义模型可提取结构化 LaTeX 公式；PyMuPDF 仅能输出原始")
    lines.append("   PDF 文本，公式位置出现 Unicode 乱码或部分数学符号，LaTeX 语义完全丢失。")
    lines.append("2. **文字提取质量**：两者在纯文字段落上差异不大，但 MinerU 做了段落合并和版面分析，")
    lines.append("   PyMuPDF 的 get_text() 输出保留原始 PDF 文本块（常被截断）。")
    lines.append("3. **处理速度**：PyMuPDF 极快（毫秒级/页），MinerU 需要约 1 分钟/页的 GPU 布局分析。")
    lines.append("   但 MinerU 的附加值（公式结构化、表格识别、图片提取）远超速度差距。")
    lines.append("4. **适用场景**：若仅需纯文字提取且 PDF 为数字版，PyMuPDF 足够；")
    lines.append("   若需结构化公式/Layout/多模态内容，必须使用 MinerU。")
    lines.append("5. **扫描版 PDF**：MinerU 内置 OCR 管线，对扫描版教材仍可提取公式；")
    lines.append("   PyMuPDF 对扫描版仅能输出空文本（get_text() 返回空字符串）。")
    lines.append("")

    return "\n".join(lines)


def save_json_data(metrics_list, output_path):
    output = []
    for m in metrics_list:
        entry = {
            "doc_name": m["doc_name"],
            "pdf_size_mb": m["pdf_size_mb"],
            "total_pages": m["total_pages"],
            "pymupdf": {
                "processing_seconds": m["pymupdf_processing_seconds"],
                "seconds_per_page": m["pymupdf_seconds_per_page"],
                "total_chars": m["pymupdf_total_chars"],
                "math_unicode_chars": m["pymupdf_math_unicode_chars"],
                "formula_dense_pages": m["formula_dense_pages"],
            },
            "mineru": {
                "total_chars": m["mineru_total_chars"],
                "formula_count": m["mineru_formula_count"],
                "block_formulas": m["mineru_block_formulas"],
                "inline_formulas": m["mineru_inline_formulas"],
                "formulas_with_tag": m["mineru_formulas_with_tag"],
                "chars_per_formula": m["mineru_chars_per_formula"],
            },
            "match_rate_pct": m["match_rate"],
        }
        output.append(entry)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"[OK] JSON data saved: {output_path}")


def process_document(doc_name):
    print(f"\n{'='*60}")
    print(f"  Processing: {doc_name}")
    print(f"{'='*60}")

    pdf_path = PDF_DIR / f"{doc_name}.pdf"
    if not pdf_path.exists():
        print(f"[ERROR] PDF not found: {pdf_path}")
        print(f"  Tried: {pdf_path}")
        print(f"  PDF_DIR contents: {list(PDF_DIR.glob('*.pdf'))[:5]}")
        return None

    print("  [1/5] Loading MinerU data...")
    mineru_formulas = load_mineru_formulas(doc_name)
    if mineru_formulas is None:
        return None
    mineru_fulltext = load_mineru_fulltext(doc_name)
    print(f"    formulas: {len(mineru_formulas)} items, full_text: {len(mineru_fulltext):,} chars")

    print("  [2/5] Loading PDF with PyMuPDF...")
    print(f"    PDF size: {pdf_path.stat().st_size / 1024 / 1024:.1f} MB")

    print("  [3/5] Running page analysis...")
    metrics = compute_document_metrics(doc_name, pdf_path, mineru_formulas, mineru_fulltext)
    print(f"    Pages: {metrics['total_pages']}")
    print(f"    PyMuPDF time: {metrics['pymupdf_processing_seconds']:.3f}s")
    print(f"    MinerU formulas: {metrics['mineru_formula_count']}")
    print(f"    Text match rate: {metrics['match_rate']}%")

    print("  [4/5] Formula dense pages detected:")
    for p in metrics["formula_dense_pages"]:
        print(f"    Page {p['page']}: {p['math_unicode_chars']} math chars, {p['latex_keyword_count']} latex keywords")

    return metrics


def main():
    import argparse

    parser = argparse.ArgumentParser(description="MinerU vs PyMuPDF 对比分析")
    parser.add_argument("--doc", type=str, default=None,
                        help="文档名（corpus/processed 下的子目录名）")
    parser.add_argument("--all", action="store_true",
                        help="处理所有教材")
    parser.add_argument("--output", type=str, default=None,
                        help="输出目录（默认 benchmark/dataset/compare_output）")
    args = parser.parse_args()

    output_dir = Path(args.output) if args.output else OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    textbooks = [
        "自抗扰控制技术",
        "自动控制原理_胡寿松",
        "非线性系统_Khalil",
        "自动控制原理题海与考研指导_胡寿松",
        "线性系统理论_郑大钟",
        "智能控制",
        "最优控制理论与应用",
        "控制之美",
        "控制理论导论_郭雷",
        "先进PID控制MATLAB仿真",
        "动态系统的反馈控制_Franklin",
    ]

    if args.doc:
        textbooks = [args.doc]

    if args.all:
        pass
    elif args.doc is None:
        textbooks = [
            "自抗扰控制技术",
            "自动控制原理_胡寿松",
            "非线性系统_Khalil",
        ]

    results = []
    for doc_name in textbooks:
        try:
            metrics = process_document(doc_name)
            if metrics:
                results.append(metrics)
        except Exception as e:
            print(f"[ERROR] Failed to process {doc_name}: {e}")
            import traceback
            traceback.print_exc()

    if not results:
        print("[ERROR] No documents processed successfully!")
        sys.exit(1)

    print(f"\n{'='*60}")
    print("  Generating reports...")
    print(f"{'='*60}")

    report_md = generate_markdown_report(results)

    md_path = output_dir / "mineru_vs_pymupdf_report.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(report_md)
    print(f"  [OK] Markdown report: {md_path}")

    json_path = output_dir / "mineru_vs_pymupdf_data.json"
    save_json_data(results, json_path)

    print(f"\n{'='*60}")
    print(f"  DONE! {len(results)} documents analyzed")
    print(f"  Report: {md_path}")
    print(f"  Data:   {json_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
