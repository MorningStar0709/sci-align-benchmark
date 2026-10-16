"""Classify 97 PMC medical papers and split into two groups for dual-blind generation.

Adapted from classify_papers.py for the medical cross-domain benchmark.
Key differences:
  - Directory format: PMC{id}/full_text.md (not arxiv_id_Title/)
  - No control category classification (medical domain)
  - Simple random split (no stratified by category)

Usage:
    python -m mineru_benchmark.pipeline.classify_medical_papers
    python -m mineru_benchmark.pipeline.classify_medical_papers --dry-run
    python -m mineru_benchmark.pipeline.classify_medical_papers --corpus ~/WorkPlace/AI/MinerU/data/sources_medical/md
"""
from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Optional

# ---- Paths ----
ROOT = Path(__file__).resolve().parents[5]
CORPUS_DIR = Path.home() / "WorkPlace" / "AI" / "MinerU" / "data" / "sources_medical" / "md"
REGISTRY_PATH = ROOT / "benchmark" / "dataset" / "medical_registry.json"
SPLIT_PATH = ROOT / "benchmark" / "dataset" / "medical_split.json"


# Headings to skip when looking for the real paper title
_SKIP_TITLES = {
    "OPEN ACCESS", "STUDY PROTOCOL", "RESEARCH ARTICLE", "ORIGINAL ARTICLE",
    "REVIEW", "SYSTEMATIC REVIEW", "META-ANALYSIS", "CASE REPORT",
    "CHECK FOR UPDATES", "EDITED BY", "REVIEWED BY",
}


def extract_metadata(full_text_path: Path) -> tuple[str, list[str], Optional[int], str]:
    """Extract title, authors, year, and abstract from full_text.md.

    Medical paper format varies, but generally:
      # Real Title (or # OPEN ACCESS / # STUDY PROTOCOL first)
      Author1, Author2, ...
      Affiliation...
      # Abstract
      ...
    """
    title = ""
    authors = []
    year = None
    abstract = ""

    if not full_text_path.exists():
        return title, authors, year, abstract

    text = full_text_path.read_text(encoding="utf-8", errors="ignore")
    lines = text.split("\n")

    # ---- Title: first meaningful H1 heading ----
    title_line_idx = -1
    for i, line in enumerate(lines[:20]):
        m = re.match(r'^#\s+(.+)$', line.strip())
        if m:
            candidate = m.group(1).strip()
            # Skip boilerplate headings
            if candidate.upper() in _SKIP_TITLES:
                continue
            if len(candidate) < 10:
                continue
            title = candidate
            title_line_idx = i
            break

    # ---- Authors: line(s) after title, before Abstract ----
    if title_line_idx >= 0:
        # Scan lines after title for author-like content
        for j in range(title_line_idx + 1, min(title_line_idx + 15, len(lines))):
            line = lines[j].strip()
            if not line:
                continue
            # Stop at next heading or Abstract
            if re.match(r'^#', line):
                break
            if re.match(r'(?i)^abstract', line):
                break
            # Author line heuristic: contains multiple names separated by , or |
            # or contains superscript markers like 1,2,*
            if re.search(r'[,|]\s*[A-Z][a-z]+', line) or re.search(r'[\d\*]+\s*[,|]', line):
                # Clean up: remove superscript markers like 1*, 2,3
                cleaned = re.sub(r'[\d\*]+', '', line)
                cleaned = re.sub(r'[‑–—]', ',', cleaned)  # normalize dashes
                if '|' in cleaned:
                    authors = [a.strip() for a in cleaned.split('|') if a.strip()]
                elif ',' in cleaned:
                    authors = [a.strip() for a in cleaned.split(',') if a.strip()]
                else:
                    authors = [cleaned.strip()] if cleaned.strip() else []
                # Filter out affiliation-like strings
                authors = [a for a in authors if not re.search(
                    r'@|Department|University|Institute|Laboratory|College|School|Hospital|Center|Centre',
                    a, re.IGNORECASE
                )]
                authors = authors[:5]
                break

    # ---- Abstract ----
    abstract_match = re.search(
        r'(?:#+\s*)?[Aa]bstract\s*\n+(.+?)(?:\n(?:#+\s*)?(?:\d+\s*)?(?:[Ii]ntroduction|[Rr]elated|[Bb]ackground|[Kk]eywords|[Kk]ey\s*[Ww]ords|[Mm]ethods|[Oo]bjectives))',
        text, re.DOTALL,
    )
    if abstract_match:
        abstract = abstract_match.group(1).strip()[:800]
    else:
        # Fallback: use first 500 chars after "Abstract" keyword
        abs_idx = text.lower().find("abstract")
        if abs_idx >= 0:
            after = text[abs_idx + 8:].strip()
            abstract = after[:800]

    # ---- Year ----
    year_match = re.search(r'(?:20(?:1[5-9]|2[0-6]))', text[:2000])
    if year_match:
        year = int(year_match.group(0))

    return title, authors, year, abstract


def estimate_control_density(text: str) -> str:
    """Estimate control-theory density in a medical paper.

    Returns: 'high' | 'medium' | 'low'
    """
    control_keywords = [
        "feedback", "control system", "pid", "regulation", "stability",
        "optimization", "adaptive", "robust", "model predictive",
        "state space", "transfer function", "controller", "actuator",
        "sensor", "closed-loop", "open-loop", "gain", "dynamics",
        "equilibrium", "convergence", "oscillation", "damping",
        "proportional", "integral", "derivative", "setpoint",
        "homeostasis", "regulatory", "dose optimization", "drug delivery",
    ]
    text_lower = text.lower()
    hits = sum(1 for kw in control_keywords if kw in text_lower)
    if hits >= 8:
        return "high"
    elif hits >= 3:
        return "medium"
    return "low"


def write_json_atomic(path: Path, data) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Classify 97 medical papers and split for dual-blind generation")
    parser.add_argument("--corpus", default=str(CORPUS_DIR), help="Medical corpus directory")
    parser.add_argument("--registry", default=str(REGISTRY_PATH), help="Output registry path")
    parser.add_argument("--split", default=str(SPLIT_PATH), help="Output split path")
    parser.add_argument("--dry-run", action="store_true", help="Print plan without writing")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    corpus_dir = Path(args.corpus)
    registry_path = Path(args.registry)
    split_path = Path(args.split)

    if not corpus_dir.exists():
        print(f"[ERROR] Corpus directory not found: {corpus_dir}")
        return 1

    # ---- Discover papers ----
    paper_dirs = sorted([d for d in corpus_dir.iterdir() if d.is_dir() and (d / "full_text.md").exists()])
    print(f"[classify-medical] Found {len(paper_dirs)} paper directories with full_text.md")

    # ---- Process each paper ----
    registry = []
    density_counts = {"high": 0, "medium": 0, "low": 0}

    for paper_dir in paper_dirs:
        pmc_id = paper_dir.name  # e.g., PMC10493474
        full_text_path = paper_dir / "full_text.md"
        title, authors, year, abstract = extract_metadata(full_text_path)

        # Read full text for density estimation
        full_text = full_text_path.read_text(encoding="utf-8", errors="ignore")
        density = estimate_control_density(full_text)
        density_counts[density] += 1

        entry = {
            "pmc_id": pmc_id,
            "title": title,
            "authors": authors,
            "year": year,
            "control_density": density,
            "paper_path": pmc_id,  # relative path within corpus_dir
            "full_text_path": str(full_text_path),
        }
        registry.append(entry)

    # ---- Stats ----
    from collections import Counter
    year_counts = Counter(e["year"] for e in registry if e["year"])

    print(f"\n[classify-medical] Results: {len(registry)} papers")
    print("  Control theory density:")
    for d in ["high", "medium", "low"]:
        print(f"    {d}: {density_counts[d]}")
    print(f"  Year range: {min(year_counts.keys())}-{max(year_counts.keys())}" if year_counts else "  Year range: unknown")
    print(f"  Papers with title: {sum(1 for e in registry if e['title'])}")
    print(f"  Papers with authors: {sum(1 for e in registry if e['authors'])}")

    if args.dry_run:
        print("\n[dry-run] Sample entries:")
        for e in registry[:5]:
            print(f"  {e['pmc_id']}: {e['title'][:60]}... density={e['control_density']}")
        print("\n[dry-run] No files written.")
        return 0

    # ---- Write registry ----
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(registry_path, {
        "total_papers": len(registry),
        "domain": "medical",
        "papers": registry,
    })
    print(f"\n[classify-medical] Registry saved: {registry_path}")

    # ---- Shuffle and split ----
    rng = random.Random(args.seed)
    entries = list(registry)
    rng.shuffle(entries)
    mid = len(entries) // 2
    group_a = entries[:mid]
    group_b = entries[mid:]

    split_path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(split_path, {
        "total_papers": len(registry),
        "domain": "medical",
        "group_a_count": len(group_a),
        "group_b_count": len(group_b),
        "group_a": group_a,
        "group_b": group_b,
    })

    print(f"[classify-medical] Split saved: {split_path}")
    print(f"  Group A: {len(group_a)} papers → DeepSeek")
    print(f"  Group B: {len(group_b)} papers → MiMo")

    # Density balance in split
    den_a = Counter(e["control_density"] for e in group_a)
    den_b = Counter(e["control_density"] for e in group_b)
    print("\n  Split density balance:")
    for d in ["high", "medium", "low"]:
        print(f"    {d}: A={den_a.get(d, 0)}, B={den_b.get(d, 0)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
