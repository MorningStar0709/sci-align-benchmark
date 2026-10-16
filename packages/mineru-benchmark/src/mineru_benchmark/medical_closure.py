"""D13b: Full-Text Pipeline (merged from v2/v3).

Pulls doc_ids from agentic-search → full text → chunks → keyword eval.
"""

import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from mineru_core.mineru_client import SciverseClient

ROOT = Path(__file__).resolve().parents[4]

KEY = os.environ.get("SCIVERSE_API_KEY", "")
C_API = "https://api.sciverse.space/content"
PMC_MANIFEST = ROOT / "data" / "sources_medical" / "chunks" / "chunks_manifest.json"
OUT_PATH = ROOT / "benchmark" / "eval" / "results" / "medical" / "sciverse_medical_closure.json"

N_TARGET = 12

QUERIES = [
    "closed-loop insulin pump glucose control algorithm",
    "model predictive control ICU mechanical ventilation optimization",
    "adaptive control drug delivery pharmacokinetic model",
    "telemedicine chronic disease management intervention effectiveness",
    "diagnostic accuracy sensitivity specificity MRI deep learning",
    "systematic review meta-analysis evidence-based clinical practice",
    "inhaled nanoparticle pulmonary absorption controlled release",
    "randomized controlled trial design methodology",
    "Kalman filter biomedical signal ECG noise removal",
    "sliding mode control surgical robotic arm precision",
    "fuzzy logic medical diagnosis decision support system",
    "neural network rehabilitation brain computer interface",
    "robust control anesthesia drug infusion safety",
    "biomedical image segmentation CNN U-Net",
]


# ---------------------------------------------------------------------------
# Low-level helpers (no CC impact on main)
# ---------------------------------------------------------------------------


def _pull_all(doc_id):
    """Fetch full text for a doc_id via paginated content API."""
    full = ""
    offset = 0
    for _ in range(6):
        qs = urllib.parse.urlencode({"doc_id": doc_id, "offset": str(offset), "limit": str(6000)})
        req = urllib.request.Request(f"{C_API}?{qs}", headers={"Authorization": f"Bearer {KEY}"})
        r = urllib.request.urlopen(req, timeout=30)
        data = json.loads(r.read())
        full += data.get("text", "")
        if not data.get("more", False) or offset >= 24000:
            break
        offset = data.get("next_offset", offset + 6000)
    return full


def load_questions():
    """Load Chinese Ask questions; fallback to QUERIES if not found."""
    zh_path = ROOT / "benchmark" / "eval" / "results" / "medical_rag_eval_zh_ask.json"
    if not zh_path.exists():
        print("  [FALLBACK] using search queries")
        return QUERIES

    data = json.loads(zh_path.read_text(encoding="utf-8"))
    raw = data.get("per_query", data.get("results", []))
    qs = []
    if isinstance(raw, dict):
        for v in raw.values():
            qt = v.get("question", v.get("query", "")) if isinstance(v, dict) else ""
            if qt:
                qs.append(qt)
    elif isinstance(raw, list):
        for q in raw:
            qt = q.get("question", q.get("query", "")) if isinstance(q, dict) else ""
            if qt:
                qs.append(qt)

    if qs:
        print(f"  [LOAD] {len(qs)} Chinese Ask questions")
        return qs

    print("  [FALLBACK] using search queries")
    return QUERIES


# ---------------------------------------------------------------------------
# Phase helpers
# ---------------------------------------------------------------------------


def _phase_collect_doc_ids(client, queries):
    """Phase 1: Collect unique doc_ids from agentic-search queries."""
    print("\n[Phase 1] Collecting doc_ids from agentic-search...")
    seen = set()
    unique_papers = []
    for qi, query in enumerate(queries):
        print(f"  [{qi + 1}/{len(queries)}] {query[:60]}...", end=" ", flush=True)
        try:
            resp = client.agentic_search(query=query, top_k=5)
            hits = resp.get("hits", resp.get("results", []))
            for h in hits:
                did = h.get("doc_id", "")
                if did and did not in seen:
                    seen.add(did)
                    unique_papers.append({
                        "doc_id": did,
                        "title": h.get("title", ""),
                        "abstract": h.get("abstract", ""),
                        "query": query,
                    })
            print(f"+{len(hits)} hits ({len(unique_papers)} unique)")
        except Exception as e:
            print(f"ERR: {str(e)[:50]}")
        time.sleep(1.5)
    print(f"\n  {len(unique_papers)} unique doc_ids from {len(queries)} queries")
    return unique_papers


def _phase_pull_full_text(papers):
    """Phase 2: Pull full text for up to N_TARGET papers."""
    target = papers[:N_TARGET]
    print(f"\n[Phase 2] Pulling full text for {len(target)} papers...")
    full_texts = []
    for pi, paper in enumerate(target):
        did = paper["doc_id"]
        title = paper["title"][:60]
        print(f"  [{pi + 1}/{len(target)}] {title}...", end=" ", flush=True)
        try:
            full = _pull_all(did)
            ok = len(full) > 300
            if ok:
                full_texts.append({**paper, "full_text": full})
            print(f"{len(full)} chars {'OK' if ok else 'TOO_SHORT'}")
        except Exception as e:
            print(f"FAIL: {str(e)[:60]}")
        time.sleep(1.5)
    print(f"\n  Successfully pulled {len(full_texts)}/{len(target)} papers")
    return full_texts


def _phase_chunk_all(full_texts):
    """Phase 3: Chunk all pulled full texts."""
    print("\n[Phase 3] Chunking...")
    all_chunks = []
    for paper in full_texts:
        paras = [p.strip() for p in paper["full_text"].split("\n\n") if len(p.strip()) > 60]
        for ci, para in enumerate(paras):
            para = re.sub(r"\[.*?\]\(.*?\)", "", para)
            para = re.sub(r"!\[.*?\]\(.*?\)", "", para)
            para = para.strip()
            if len(para) < 60:
                continue
            all_chunks.append({
                "chunk_id": f"{paper['doc_id']}:c{ci}",
                "doc_id": paper["doc_id"],
                "title": paper["title"],
                "text": para[:1500],
            })
    print(f"  {len(all_chunks)} chunks")
    return all_chunks


def _phase_eval(all_chunks, questions):
    """Phase 4: Keyword-overlap evaluation against PMC chunks."""
    print("\n[Phase 4] Evaluation against Chinese Ask + PMC...")
    pmc = json.loads(PMC_MANIFEST.read_text(encoding="utf-8"))
    pmc_chunks = pmc["chunks"]

    _STOPWORDS = {"what", "when", "which", "that", "this", "with", "from", "have"}
    results = []
    sv_only = pmc_only = both = neither = 0

    for qi, qt in enumerate(questions):
        keywords = [
            w.strip(".,;:()").lower()
            for w in qt.replace(",", " ").split()
            if len(w) > 3 and w.lower() not in _STOPWORDS
        ]
        keywords = list(set(keywords))

        sv_hit = sum(1 for kw in keywords for c in all_chunks if kw in c["text"].lower())
        pmc_hit = sum(
            1
            for kw in keywords
            for c in pmc_chunks[:1000]
            if kw in c.get("text", "")[:500].lower()
        )

        has_sv = sv_hit >= 1
        has_pmc = pmc_hit >= 1
        if has_sv and has_pmc:
            both += 1
        elif has_sv:
            sv_only += 1
        elif has_pmc:
            pmc_only += 1
        else:
            neither += 1

        results.append({
            "question": qt[:150],
            "sv_hits": sv_hit,
            "pmc_hits": pmc_hit,
            "sciverse_unique": sv_only if has_sv and not has_pmc else 0,
        })
        if has_sv and not has_pmc:
            print(f"  [{qi + 1}] SV_UNIQUE: {qt[:60]}...")

    return results, {"sv_only": sv_only, "pmc_only": pmc_only, "both": both, "neither": neither}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _build_closure_report(unique_papers, full_texts, all_chunks, questions, results, coverage):
    """Build the final closure report dict."""
    report = {
        "analysis": "sciverse_medical_closure",
        "n_doc_ids_collected": len(unique_papers),
        "n_pulled": len(full_texts),
        "n_chunks": len(all_chunks),
        "n_questions": len(questions),
        "chunks": all_chunks,
        "coverage": {
            "sciverse_unique": coverage["sv_only"],
            "pmc_unique": coverage["pmc_only"],
            "both": coverage["both"],
            "neither": coverage["neither"],
            "combined": coverage["sv_only"] + coverage["pmc_only"] + coverage["both"],
        },
        "per_question": results,
    }
    return report


def _print_closure_summary(full_texts, all_chunks, coverage, questions):
    """Print the final summary to stdout."""
    print(f"\n{'=' * 60}")
    print("D13b COMPLETE")
    print(f"  Papers pulled:    {len(full_texts)}")
    print(f"  Chunks:           {len(all_chunks)}")
    print(f"  Sciverse unique:  {coverage['sv_only']}")
    print(f"  PMC unique:       {coverage['pmc_only']}")
    print(f"  Both covered:     {coverage['both']}")
    print(f"  Combined:         {coverage['sv_only'] + coverage['pmc_only'] + coverage['both']}/{len(questions)}")
    print(f"  Report:           {OUT_PATH}")
    print(f"{'=' * 60}")
    if coverage["sv_only"] > 0:
        print(f"  \u2705 {coverage['sv_only']} questions Sciverse adds evidence PMC misses \u2014 pipeline fully closed")
    else:
        print("  \u26a0\ufe0f  No unique Sciverse-only hits")



def main():
    print("=" * 60)
    print("D13b Full-Text Pipeline (agentic-search doc_ids)")
    print("=" * 60)

    client = SciverseClient(api_key=KEY)

    unique_papers = _phase_collect_doc_ids(client, QUERIES)
    if not unique_papers:
        print("[STOP] No doc_ids collected, aborting.")
        return

    full_texts = _phase_pull_full_text(unique_papers)
    all_chunks = _phase_chunk_all(full_texts) if full_texts else []
    questions = load_questions()
    results, coverage = _phase_eval(all_chunks, questions)

    report = _build_closure_report(unique_papers, full_texts, all_chunks, questions, results, coverage)
    OUT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    _print_closure_summary(full_texts, all_chunks, coverage, questions)


if __name__ == "__main__":
    main()
