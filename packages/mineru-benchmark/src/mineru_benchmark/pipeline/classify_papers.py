"""Classify 362 papers into control categories and split into two groups for dual-blind generation.

Usage:
    python -m mineru_benchmark.pipeline.classify_papers
    python -m mineru_benchmark.pipeline.classify_papers --dry-run  # show classification plan
    python -m mineru_benchmark.pipeline.classify_papers --max-api 5  # limit API calls
"""
from __future__ import annotations

import json
import os
import random
import re
from pathlib import Path
from typing import Optional

# ---- Paths ----
ROOT = Path(__file__).resolve().parents[5]
CORPUS_DIR = Path.home() / "WorkPlace" / "AI" / "MinerU" / "corpus" / "processed"
REGISTRY_PATH = ROOT / "benchmark" / "dataset" / "paper_registry.json"
SPLIT_PATH = ROOT / "benchmark" / "dataset" / "paper_split.json"

CATEGORY_MAP = {
    "控制理论": "classical",
    "控制科学": "modern",
    "控制系统": "classical",
    "经典控制": "classical",
    "现代控制": "modern",
    "非线性控制": "nonlinear",
    "鲁棒控制": "robust",
    "最优控制": "optimal",
    "自适应控制": "adaptive",
    "数字控制": "digital",
    "智能控制": "intelligent",
    "模型预测控制": "mpc",
    "多智能体系统": "multi_agent",
    "reinforcement learning": "intelligent",
    "deep reinforcement": "intelligent",
    "neural network control": "intelligent",
    "adaptive control": "adaptive",
    "robust control": "robust",
    "optimal control": "optimal",
    "nonlinear control": "nonlinear",
    "nonlinear system": "nonlinear",
    "model predictive": "mpc",
    "MPC": "mpc",
    "multi-agent": "multi_agent",
    "multi agent": "multi_agent",
    "consensus": "multi_agent",
    "formation control": "multi_agent",
    "event-triggered": "digital",
    "self-triggered": "digital",
    "sampled-data": "digital",
    "discrete-time": "digital",
    "PID": "classical",
    "backstepping": "nonlinear",
    "sliding mode": "nonlinear",
    "Lyapunov": "modern",
    "linear matrix inequality": "modern",
    "LMI": "modern",
    "H-infinity": "robust",
    "H∞": "robust",
    "uncertainty": "robust",
    "disturbance": "robust",
    "stochastic": "optimal",
    "Kalman": "optimal",
    "LQR": "optimal",
    "LQG": "optimal",
    "RL": "intelligent",
    "fuzzy": "intelligent",
    "learning": "intelligent",
    "neural network": "intelligent",
    "deep learning": "intelligent",
    "predictive control": "mpc",
    "model predictive control": "mpc",
}

CLASSIFICATION_PROMPT = """You are a control science expert. Given a paper title and abstract excerpt, assign it to the most appropriate control category.

## Paper Information
Title: {title}
Abstract: {abstract}

## Available Categories
- classical: Classical control theory (PID, root locus, frequency domain, transfer functions)
- modern: Modern control theory (state space, LQR, LQG, Lyapunov, observers, LMI)
- nonlinear: Nonlinear control (backstepping, sliding mode, feedback linearization, Lyapunov-based)
- robust: Robust control (H-infinity, mu-synthesis, uncertainty, disturbance rejection)
- optimal: Optimal control (LQR, LQG, stochastic control, Kalman filtering, dynamic programming)
- adaptive: Adaptive control (MRAC, self-tuning, parameter estimation, adaptation laws)
- digital: Digital/Sampled-data control (discrete-time, event-triggered, networked control)
- intelligent: Intelligent control (neural networks, fuzzy logic, reinforcement learning, learning-based)
- mpc: Model Predictive Control (MPC, receding horizon, constrained optimization)
- multi_agent: Multi-agent systems (consensus, formation, distributed control, flocking)

Output strict JSON: {"control_category": "<one category>", "confidence": "high|medium|low", "reasoning": "<1 sentence>"}
Category must be exactly one of the 10 values listed above."""


def parse_dir_name(dir_name: str) -> tuple[str, str]:
    """Parse arxiv_id and title from directory name like '2201.05599_Title_Abbreviated'."""
    parts = dir_name.split("_", 1)
    arxiv_id = parts[0] if parts else dir_name
    title = parts[1].replace("_", " ") if len(parts) > 1 else dir_name
    return arxiv_id, title


def extract_metadata(full_text_path: Path) -> tuple[list[str], Optional[int], str]:
    """Extract authors, year, and abstract from full_text.md."""
    authors = []
    year = None
    abstract = ""

    if not full_text_path.exists():
        return authors, year, abstract

    text = full_text_path.read_text(encoding="utf-8", errors="ignore")

    # Extract abstract (text between Abstract and Introduction, or first 500 chars after Abstract)
    abstract_match = re.search(
        r'(?:#+\s*)?[Aa]bstract\s*\n+(.+?)(?:\n(?:#+\s*)?(?:\d+\s*)?(?:[Ii]ntroduction|[Rr]elated|[Bb]ackground))',
        text, re.DOTALL,
    )
    if abstract_match:
        abstract = abstract_match.group(1).strip()[:800]
    else:
        # Fallback: use text after first heading as abstract
        lines = text.split("\n")
        abstract_lines = []
        in_content = False
        for line in lines:
            if re.match(r'^#+\s', line):
                if in_content:
                    break
                in_content = True
                continue
            if in_content and line.strip():
                abstract_lines.append(line.strip())
            if len("\n".join(abstract_lines)) > 800:
                break
        abstract = "\n".join(abstract_lines)

    # Extract authors from lines after "Authors" heading
    in_authors = False
    for line in text.split("\n")[:30]:
        line_stripped = line.strip()
        if re.match(r'^#*\s*Authors?\s*$', line_stripped, re.IGNORECASE):
            in_authors = True
            continue
        if in_authors:
            if re.match(r'^#+\s', line_stripped) or line_stripped == "":
                break
            # Try to match author names (skip affiliation lines)
            if not re.search(r'@|Department|University|Institute|Laboratory|College|School', line_stripped, re.IGNORECASE):
                name_match = re.match(r'^([A-Z][a-z]+(?:\s+[A-Z]\.?)?(?:\s+[A-Z][a-z]+)?)', line_stripped)
                if name_match:
                    authors.append(name_match.group(1).strip())

    # Extract year
    year_match = re.search(r'(?:20(?:1[5-9]|2[0-6]))', text[:1500])
    if year_match:
        year = int(year_match.group(0))

    return authors[:5], year, abstract


def keyword_classify(title: str, abstract: str) -> Optional[str]:
    """Try keyword-based classification. Returns category key or None."""
    combined = (title + " " + abstract).lower()
    scores = {}
    for keyword, category in CATEGORY_MAP.items():
        if keyword.lower() in combined:
            scores[category] = scores.get(category, 0) + 1

    if not scores:
        return None
    return max(scores, key=scores.get)


def api_classify(title: str, abstract: str, api_key: str) -> Optional[str]:
    """Use DeepSeek API to classify a paper."""
    import time
    try:
        from openai import OpenAI
    except ImportError:
        print("[WARN] openai not installed, falling back to keyword-only classification", flush=True)
        return None

    from mineru_generator.api import PROVIDER_CONFIG, resolve_model
    config = PROVIDER_CONFIG["deepseek"]
    model = resolve_model("deepseek")

    client = OpenAI(
        api_key=api_key,
        base_url=config["base_url"],
    )

    prompt = CLASSIFICATION_PROMPT.replace("{title}", title).replace("{abstract}", abstract[:600])

    valid_categories = set(CATEGORY_MAP.values())
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=150,
            )
            content = resp.choices[0].message.content.strip()

            # Extract JSON
            json_match = re.search(r'\{[^}]+\}', content, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group(0))
                category = result.get("control_category", "")
                if category in valid_categories:
                    return category
                # Try keyword-map fallback (API may return Chinese names)
                mapped = CATEGORY_MAP.get(category)
                if mapped:
                    return mapped
            time.sleep(0.5)
        except Exception as e:
            print(f"  API attempt {attempt + 1} failed: {e}", flush=True)
            time.sleep(2 ** attempt)
    return None


def write_json_atomic(path: Path, data: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Classify 362 papers into control categories")
    parser.add_argument("--corpus", default=str(CORPUS_DIR), help="Corpus directory")
    parser.add_argument("--registry", default=str(REGISTRY_PATH), help="Output registry path")
    parser.add_argument("--split", default=str(SPLIT_PATH), help="Output split path")
    parser.add_argument("--dry-run", action="store_true", help="Print plan without writing")
    parser.add_argument("--max-api", type=int, default=20, help="Max API classification calls (default 20)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    corpus_dir = Path(args.corpus)
    registry_path = Path(args.registry)
    split_path = Path(args.split)

    if not corpus_dir.exists():
        print(f"[ERROR] Corpus directory not found: {corpus_dir}")
        return 1

    # ---- Discover papers ----
    paper_dirs = sorted([d for d in corpus_dir.iterdir() if d.is_dir()])
    print(f"[classify] Found {len(paper_dirs)} paper directories")

    # ---- Get API key ----
    api_key = os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("OPENAI_API_KEY") or ""
    if not api_key:
        print("[classify] WARNING: No DEEPSEEK_API_KEY found. Using keyword-only classification.")

    # ---- Process each paper ----
    registry = []
    api_calls = 0
    keyword_hits = 0
    api_hits = 0
    unclassified = 0

    for paper_dir in paper_dirs:
        dir_name = paper_dir.name
        arxiv_id, title = parse_dir_name(dir_name)
        full_text_path = paper_dir / "full_text.md"
        authors, year, abstract = extract_metadata(full_text_path)

        # Classification
        category = keyword_classify(title, abstract)
        method = "keyword"
        if category:
            keyword_hits += 1
        elif api_key and api_calls < args.max_api and abstract:
            category = api_classify(title, abstract, api_key)
            api_calls += 1
            if category:
                api_hits += 1
                method = "api"

        if not category:
            category = "classical"  # fallback
            unclassified += 1
            method = "fallback"

        entry = {
            "arxiv_id": arxiv_id,
            "title": title,
            "authors": authors,
            "year": year,
            "control_category": category,
            "paper_path": dir_name,
            "full_text_path": str(full_text_path),
            "classification_method": method,
        }
        registry.append(entry)

    # ---- Distribution stats ----
    from collections import Counter
    cat_counts = Counter(e["control_category"] for e in registry)

    print(f"\n[classify] Results: {len(registry)} papers")
    print(f"  Keyword hits: {keyword_hits}")
    print(f"  API hits: {api_hits} (of {api_calls} calls)")
    print(f"  Fallback: {unclassified}")
    print("\n  Category distribution:")
    for cat, count in cat_counts.most_common():
        print(f"    {cat}: {count}")

    if args.dry_run:
        print("\n[dry-run] No files written.")
        return 0

    # ---- Write registry ----
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(registry_path, {
        "total_papers": len(registry),
        "papers": registry,
    })
    print(f"\n[classify] Registry saved: {registry_path}")

    # ---- Shuffle and split ----
    rng = random.Random(args.seed)

    # Stratified shuffle: group by control_category, then split each group evenly
    by_category = {}
    for entry in registry:
        by_category.setdefault(entry["control_category"], []).append(entry)

    group_a, group_b = [], []
    for cat, entries in by_category.items():
        rng.shuffle(entries)
        mid = len(entries) // 2
        group_a.extend(entries[:mid])
        group_b.extend(entries[mid:])

    # Shuffle within groups
    rng.shuffle(group_a)
    rng.shuffle(group_b)

    split_path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(split_path, {
        "total_papers": len(registry),
        "group_a_count": len(group_a),
        "group_b_count": len(group_b),
        "group_a": group_a,
        "group_b": group_b,
    })

    print(f"[classify] Split saved: {split_path}")
    print(f"  Group A: {len(group_a)} papers → DeepSeek")
    print(f"  Group B: {len(group_b)} papers → MiMo")

    # Verify category balance in split
    cat_a = Counter(e["control_category"] for e in group_a)
    cat_b = Counter(e["control_category"] for e in group_b)
    print("\n  Split category balance:")
    all_cats = sorted(set(list(cat_a.keys()) + list(cat_b.keys())))
    for cat in all_cats:
        print(f"    {cat}: A={cat_a.get(cat, 0)}, B={cat_b.get(cat, 0)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
