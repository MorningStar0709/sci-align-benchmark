"""Track 1: Sciverse 全文内容审计

从 14 控制科学子领域各取 1 篇文献全文（Sciverse content 端点），提取 LaTeX 公式
并审计公式-上下文对齐质量。Content 端点返回 MinerU 结构化 Markdown，图片引用为
hash 文件名（可通过 resource 端点下载），跨模态视觉审计链路已验证就绪。

输出:
  benchmark/eval/results/sciverse_cross_modal_audit.json

用法:
  conda activate myenv && python -m benchmark.sciverse_content_audit
  conda activate myenv && python -m benchmark.sciverse_content_audit --max-papers 5
"""

import argparse
import json
import re
import time
from datetime import datetime, timezone

from mineru_core.mineru_client import SciverseClient
from mineru_core.paths import PROJECT_ROOT

SUBFIELDS: dict[str, str] = {
    "PID控制": "PID control stability",
    "估计与定位": "Kalman filter state estimation",
    "导航制导与控制": "guidance navigation control GPS",
    "控制理论": "control theory Lyapunov",
    "数字控制": "digital control discrete time",
    "智能控制": "intelligent control neural network",
    "最优控制": "optimal control Pontryagin",
    "现代控制": "modern control state space",
    "线性系统": "linear system theory",
    "经典控制": "classical control root locus",
    "自抗扰控制": "active disturbance rejection control",
    "自适应控制": "adaptive control nonlinear",
    "非线性控制": "nonlinear control backstepping",
    "鲁棒控制": "robust control H-infinity",
}

OUTPUT = (
    PROJECT_ROOT / "benchmark" / "eval" / "results"
    / "sciverse_cross_modal_audit.json"
)

FM_BLOCK = re.compile(r"\$\$(.+?)\$\$", re.DOTALL)
FM_INLINE = re.compile(r"(?<!\$)\$(?!\$)([^$]+?)(?<!\$)\$(?!\$)")
FM_PAREN = re.compile(r"\\\((.+?)\\\)", re.DOTALL)

MAX_CONTENT_PAGES = 50
CONTENT_CHUNK_SIZE = 700
CONTEXT_WINDOW = 200

API_RATE_DELAY = 1.2
API_RATE_DELAY_CONTENT = 0.4

IMG_HASH_PATTERN = re.compile(r"(?:images?/|imgs?/)?([a-f0-9]{40,64}\.(?:jpg|png|jpeg|webp))", re.IGNORECASE)

CONTROL_KEYWORDS = [
    "control", "stability", "feedback", "Lyapunov", "PID", "Kalman",
    "filter", "state", "robust", "H-infinity", "optimal", "nonlinear",
    "linear", "adaptive", "disturbance", "observer", "controller",
    "regulator", "tracking", "convergence", "pole", "zero", "transfer",
    "function", "frequency", "response", "system", "model", "dynamic",
    "input", "output", "signal", "noise", "uncertainty", "identification",
    "estimation", "prediction", "optimization", "constraint", "trajectory",
    "guidance", "navigation", "backstepping", "sliding", "mode",
]


def extract_formulas(text: str) -> list[dict]:
    scan = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
    formulas: list[dict] = []

    for m in FM_BLOCK.findall(scan):
        latex = m.strip()
        if len(latex) >= 3:
            formulas.append({"type": "block", "latex": latex})

    for m in FM_INLINE.findall(scan):
        fm = m.strip()
        if fm and len(fm) >= 2 and re.search(r"[\\\^_{}=&\|]", fm):
            formulas.append({"type": "inline", "latex": fm})

    for m in FM_PAREN.findall(scan):
        latex = m.strip()
        if len(latex) >= 3:
            formulas.append({"type": "paren", "latex": latex})

    return formulas


def extract_image_refs(text: str) -> list[str]:
    seen: set[str] = set()
    refs: list[str] = []
    for m in IMG_HASH_PATTERN.finditer(text):
        fname = m.group(1)
        if fname not in seen:
            seen.add(fname)
            refs.append(fname)
    return refs


def pull_full_content(client: SciverseClient, doc_id: str) -> tuple[str, int]:
    full = ""
    offset = 0
    pages = 0
    while pages < MAX_CONTENT_PAGES:
        r = client.get_content(doc_id, offset=offset, limit=CONTENT_CHUNK_SIZE)
        txt = r.get("text", "")
        if not txt:
            break
        full += txt
        pages += 1
        if not r.get("more"):
            break
        offset = r.get("next_offset", offset + CONTENT_CHUNK_SIZE)
        if pages % 5 == 0:
            time.sleep(API_RATE_DELAY_CONTENT)
    return full, pages


def classify_formula_context(
    formulas: list[dict], full_text: str
) -> list[dict]:
    results = []
    for fm in formulas:
        latex = fm["latex"]
        context = _extract_context(full_text, latex)
        combined = (context + " " + latex).lower()
        match_count = sum(
            1 for kw in CONTROL_KEYWORDS if kw.lower() in combined
        )
        if match_count >= 5:
            alignment = "consistent"
        elif match_count >= 2:
            alignment = "partial"
        else:
            alignment = "inconsistent"
        results.append({
            "type": fm["type"],
            "latex": latex[:120],
            "context_preview": context[:100],
            "keyword_matches": match_count,
            "alignment": alignment,
        })
    return results


def _extract_context(full_text: str, latex: str) -> str:
    idx = full_text.find(latex)
    if idx < 0:
        return ""
    start = max(0, idx - CONTEXT_WINDOW)
    end = min(len(full_text), idx + len(latex) + CONTEXT_WINDOW)
    return full_text[start:end]


def run_search_phase(client: SciverseClient) -> dict[str, dict]:
    print("[Phase 1/4] agentic-search: 获取 14 子领域 doc_id...", flush=True)
    docs = {}
    items = list(SUBFIELDS.items())
    for i, (zh_name, en_query) in enumerate(items):
        if i > 0:
            time.sleep(API_RATE_DELAY)
        r = client.agentic_search(query=en_query, top_k=1)
        hits = r.get("hits", [])
        if hits:
            h = hits[0]
            docs[zh_name] = {
                "doc_id": h.get("doc_id", ""),
                "title": h.get("title", ""),
                "score": h.get("score"),
                "en_query": en_query,
            }
        else:
            docs[zh_name] = {
                "doc_id": "",
                "title": "",
                "score": 0,
                "en_query": en_query,
            }
        print(f"  {zh_name}: {docs[zh_name]['title'][:60]}...", flush=True)
    return docs


def run_content_phase(client: SciverseClient, docs: dict) -> dict:
    print(f"\n[Phase 2/4] content: 拉取 {len(docs)} 篇全文...", flush=True)
    for idx, (zh_name, d) in enumerate(docs.items()):
        if idx > 0:
            time.sleep(API_RATE_DELAY)
        did = d.get("doc_id")
        if not did:
            d["content_chars"] = 0
            d["content_pages"] = 0
            d["content_ok"] = False
            d["content_error"] = "no doc_id"
            print(f"  {zh_name}: SKIP (no doc_id)", flush=True)
            continue
        try:
            txt, pages = pull_full_content(client, did)
            d["content_chars"] = len(txt)
            d["content_pages"] = pages
            d["content_ok"] = True
            d["full_text"] = txt
            d["image_refs"] = extract_image_refs(txt)
            print(
                f"  {zh_name}: {len(txt)} chars in {pages} pages"
                f" ({len(d['image_refs'])} images)",
                flush=True,
            )
        except Exception as e:
            d["content_chars"] = 0
            d["content_pages"] = 0
            d["content_ok"] = False
            d["content_error"] = str(e)
            print(f"  {zh_name}: ERROR {e}", flush=True)
    return docs


def run_formula_phase(docs: dict) -> dict:
    print("\n[Phase 3/4] formula: 提取公式 + 上下文对齐审计...", flush=True)
    for zh_name, d in docs.items():
        if not d.get("content_ok"):
            d["formula_count"] = 0
            d["formula_audit"] = []
            continue
        full_text = d.get("full_text", "")
        formulas = extract_formulas(full_text)
        audit = classify_formula_context(formulas, full_text)
        d["formula_count"] = len(formulas)
        d["formula_audit"] = audit

        c_count = sum(1 for a in audit if a["alignment"] == "consistent")
        p_count = sum(1 for a in audit if a["alignment"] == "partial")
        i_count = sum(1 for a in audit if a["alignment"] == "inconsistent")
        pct = (c_count / max(len(audit), 1)) * 100
        print(
            f"  {zh_name}: {len(formulas)} formulas "
            f"(c={c_count} p={p_count} i={i_count}) "
            f"consistent={pct:.0f}%",
            flush=True,
        )
    return docs


def build_report(docs: dict) -> dict:
    timestamp = datetime.now(timezone.utc).isoformat()
    ok_count = sum(1 for d in docs.values() if d.get("content_ok"))
    total_fm = sum(d.get("formula_count", 0) for d in docs.values())
    all_audit = []
    for d in docs.values():
        all_audit.extend(d.get("formula_audit", []))

    c_count = sum(1 for a in all_audit if a["alignment"] == "consistent")
    p_count = sum(1 for a in all_audit if a["alignment"] == "partial")
    i_count = sum(1 for a in all_audit if a["alignment"] == "inconsistent")
    total_audited = len(all_audit)
    pct = (c_count / max(total_audited, 1)) * 100
    total_imgs = sum(len(d.get("image_refs", [])) for d in docs.values())
    papers_with_imgs = sum(
        1 for d in docs.values() if len(d.get("image_refs", [])) > 0
    )

    per_paper = {}
    for zh_name, d in docs.items():
        per_paper[zh_name] = {
            "doc_id": d.get("doc_id", ""),
            "title": d.get("title", ""),
            "en_query": d.get("en_query", ""),
            "agentic_score": d.get("score"),
            "content_chars": d.get("content_chars", 0),
            "content_pages": d.get("content_pages", 0),
            "content_ok": d.get("content_ok", False),
            "content_error": d.get("content_error", ""),
            "image_count": len(d.get("image_refs", [])),
            "image_refs_sample": d.get("image_refs", [])[:5],
            "formula_count": d.get("formula_count", 0),
            "formula_audit_summary": _paper_audit_summary(
                d.get("formula_audit", [])
            ),
        }

    return {
        "generated_at": timestamp,
        "config": {
            "subfields": list(SUBFIELDS.keys()),
            "content_chunk_size": CONTENT_CHUNK_SIZE,
            "max_content_pages": MAX_CONTENT_PAGES,
            "context_window": CONTEXT_WINDOW,
            "note": (
                "Sciverse content 端点返回 MinerU 结构化 Markdown。"
                "图片引用为 hash 文件名（可通过 resource 端点下载）。"
                "公式-上下文对齐审计基于关键词匹配。"
            ),
        },
        "summary": {
            "papers_attempted": len(docs),
            "papers_ok": ok_count,
            "total_image_refs": total_imgs,
            "papers_with_images": papers_with_imgs,
            "total_formulas": total_fm,
            "formulas_audited": total_audited,
            "consistent": c_count,
            "partial": p_count,
            "inconsistent": i_count,
            "consistent_pct": round(pct, 1),
        },
        "per_paper": per_paper,
    }


def _paper_audit_summary(audit: list[dict]) -> dict:
    if not audit:
        return {"c": 0, "p": 0, "i": 0, "total": 0, "pct": 0}
    c = sum(1 for a in audit if a["alignment"] == "consistent")
    p = sum(1 for a in audit if a["alignment"] == "partial")
    i = sum(1 for a in audit if a["alignment"] == "inconsistent")
    return {"c": c, "p": p, "i": i, "total": len(audit),
            "pct": round(c / len(audit) * 100, 1)}


def main():
    parser = argparse.ArgumentParser(
        description="Sciverse 全文内容审计 — 拉取全文 + 公式提取 + 上下文对齐"
    )
    parser.add_argument(
        "--max-papers",
        type=int,
        default=14,
        help="最多审计论文数（默认 14）",
    )
    parser.add_argument(
        "--skip-search",
        action="store_true",
        help="跳过 agentic-search，使用已知 doc_id",
    )
    args = parser.parse_args()

    client = SciverseClient()

    if not args.skip_search:
        docs = run_search_phase(client)
    else:
        docs = {}  # would need pre-populated doc_ids

    docs = dict(list(docs.items())[: args.max_papers])
    docs = run_content_phase(client, docs)
    docs = run_formula_phase(docs)

    report = build_report(docs)
    s = report["summary"]

    print("\n[Phase 4/4] 报告...", flush=True)
    print(
        f"  全文成功: {s['papers_ok']}/{s['papers_attempted']}"
    )
    print(
        f"  图片引用: {s['total_image_refs']} ({s['papers_with_images']} 篇含图)"
    )
    print(
        f"  公式总数: {s['total_formulas']}  "
        f"审计: {s['formulas_audited']}"
    )
    print(
        f"  公式-上下文对齐: {s['consistent']} consistent, "
        f"{s['partial']} partial, {s['inconsistent']} inconsistent"
    )
    print(f"  consistent 率: {s['consistent_pct']}%")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"  -> {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
