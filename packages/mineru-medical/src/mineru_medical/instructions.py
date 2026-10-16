"""从 97 篇医疗 MD 文件中提取 section → 生成 QLoRA ChatML instruction 数据

前置条件:
  - MinerU 解析产物就绪: data/sources_medical/md/ (97 篇 .md)
  - 元数据: data/sources_medical/source_list.json

输出:
  - finetune/data/medical/train.jsonl
  - finetune/data/medical/val.jsonl

格式: ChatML (messages: system / user / assistant)
"""

import argparse
import hashlib
import json
import os
import random
import re
import sys
import tempfile
from pathlib import Path

from mineru_core.paths import PROJECT_ROOT


ROOT = PROJECT_ROOT

MEDICAL_SYSTEM_PROMPT = "你是一位临床医学研究专家，精通循证医学文献的解读与分析。请基于医学文献内容回答以下问题。"

SECTION_TEMPLATES = {
    "abstract": {
        "query": "请概述这篇医学文献的研究背景、方法、主要结果和结论。",
        "min_chars": 200,
    },
    "introduction": {
        "query": "请描述这项研究的背景、临床意义和研究假设。",
        "min_chars": 300,
    },
    "methods": {
        "query": "请详细说明本研究的研究设计、入排标准、干预措施、终点指标和统计分析方法。",
        "min_chars": 300,
    },
    "results": {
        "query": "请列出本研究的主要结局指标数据、基线特征、亚组分析结果和不良事件。",
        "min_chars": 300,
    },
    "discussion": {
        "query": "请解读本研究的主要发现、临床意义、局限性及与既往文献的比较。",
        "min_chars": 300,
    },
    "conclusion": {
        "query": "请总结本研究的核心结论和未来研究方向。",
        "min_chars": 100,
    },
}

SECTION_PATTERNS = {
    "abstract":    re.compile(r"(?i)^#+\s*(abstract|摘要)\s*$"),
    "introduction": re.compile(r"(?i)^#+\s*(introduction|背景|引言|background)\s*$"),
    "methods":     re.compile(r"(?i)^#+\s*(methods?|方法|study\s*design|研究设计|materials?\s*and\s*methods?)\s*$"),
    "results":     re.compile(r"(?i)^#+\s*(results?|结果|findings)\s*$"),
    "discussion":  re.compile(r"(?i)^#+\s*(discussion|讨论)\s*$"),
    "conclusion":  re.compile(r"(?i)^#+\s*(conclusion|结论|conclusions?)\s*$"),
}

UNWANTED_HEADER_PATTERNS = re.compile(
    r"(?i)^#+\s*(acknowledg?ments?|致谢|references?|参考文献|"
    r"conflict\s*of\s*interest|利益冲突|supplementary|补充材料|"
    r"funding|资助|author\s*contributions?|作者贡献|abbreviations?|缩写|"
    r"doi\s|received:?\s|accepted:?\s|published:?\s|citation\b|copyright\b|open\s*access|"
    r"edited\s+by|reviewed\s+by|correspondence\b|keywords?|check\s*for\s*updates|"
    r"data\s*availability|ethical\s*approval|informed\s*consent|trial\s*registration|"
    r"figure\s*legend|table\s*of\s*contents|appendix|availability\s*of\s*data)\s*$"
)


def classify_section(heading):
    for label, pattern in SECTION_PATTERNS.items():
        if pattern.match(heading):
            return label
    return None


def is_unwanted_section(heading):
    return bool(UNWANTED_HEADER_PATTERNS.match(heading))


def _process_md_line(line, current_heading, current_lines, preamble_lines, sections):
    """Process a single MD line in the section extraction state machine.
    Returns (current_heading, current_lines, preamble_lines) updated state.
    """
    stripped = line.strip()
    if stripped.startswith("#") and not stripped.startswith("##"):
        heading_text = re.sub(r"^#+\s*", "", stripped)
        if is_unwanted_section(heading_text):
            if current_lines and current_heading:
                sections.append((classify_section(current_heading), "\n".join(current_lines)))
            return None, [], preamble_lines
        section_type = classify_section(heading_text)
        if section_type:
            if current_lines and current_heading:
                sections.append((classify_section(current_heading), "\n".join(current_lines)))
            return heading_text, [], preamble_lines
        elif current_heading:
            current_lines.append(line)
        else:
            preamble_lines.append(line)
    else:
        if current_heading:
            current_lines.append(line)
        else:
            preamble_lines.append(line)
    return current_heading, current_lines, preamble_lines


def extract_sections(md_text):
    """\u6309 # \u6807\u9898\u5207\u5206 MD \u6587\u672c\u4e3a (section_type, content) \u5217\u8868"""
    lines = md_text.split("\n")
    sections = []
    current_heading = None
    current_lines = []
    preamble_lines = []

    for line in lines:
        current_heading, current_lines, preamble_lines = _process_md_line(
            line, current_heading, current_lines, preamble_lines, sections
        )

    if current_lines and current_heading:
        sections.append((classify_section(current_heading), "\n".join(current_lines)))

    if preamble_lines and not sections:
        preamble_text = "\n".join(preamble_lines).strip()
        if len(preamble_text) > 500:
            sections.append(("abstract", preamble_text))

    return sections


def clean_section_text(text):
    text = re.sub(r"!\[.*?\]\(image/.*?\)", "", text)
    text = re.sub(r"<details>.*?</details>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.strip()
    return text


def build_instruction_pairs(md_path, pmcid, source_label):
    with open(md_path, "r", encoding="utf-8") as f:
        md_text = f.read()

    sections = extract_sections(md_text)
    pairs = []

    for section_type, content in sections:
        if section_type is None:
            continue
        content = clean_section_text(content)
        template = SECTION_TEMPLATES.get(section_type)
        if template is None or len(content) < template["min_chars"]:
            continue

        pairs.append({
            "messages": [
                {"role": "system", "content": MEDICAL_SYSTEM_PROMPT},
                {"role": "user", "content": template["query"]},
                {"role": "assistant", "content": content},
            ],
            "_meta": {
                "pmcid": pmcid,
                "section": section_type,
                "label": source_label,
                "char_count": len(content),
                "source_file": md_path.name,
            },
        })

    return pairs


def load_source_labels(source_list_path):
    with open(source_list_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    papers = data.get("papers", {})
    return {pmcid: info.get("label", "unknown") for pmcid, info in papers.items()}


def _write_instruction_subsets(train_pairs, val_pairs, output_dir):
    """Write train/val subsets as JSONL files with atomic tmpfile write."""
    for name, subset in [("train", train_pairs), ("val", val_pairs)]:
        if not subset:
            print(f"\n[\u8df3\u8fc7  ] {name}.jsonl: 0 \u6761\uff0c\u4e0d\u5199\u5165")
            continue
        path = output_dir / f"{name}.jsonl"
        fd, tmp_path = tempfile.mkstemp(suffix=".jsonl", prefix=f"{name}_", dir=str(output_dir))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                for pair in subset:
                    entry = {"messages": pair["messages"]}
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            os.replace(tmp_path, str(path))
            print(f"\n[\u5199\u5165  ] {name}.jsonl: {len(subset)} \u6761")
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise


def main():
    parser = argparse.ArgumentParser(description="从 97 篇医疗 MD 文件生成 QLoRA instruction 数据")
    parser.add_argument(
        "--md-dir",
        default=str(ROOT / "data" / "sources_medical" / "md"),
        help="MinerU 解析产物目录",
    )
    parser.add_argument(
        "--source-list",
        default=str(ROOT / "data" / "sources_medical" / "source_list.json"),
        help="source_list.json 元数据路径",
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "finetune" / "data" / "medical"),
        help="输出目录",
    )
    parser.add_argument("--train-ratio", type=float, default=0.85)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-pairs-per-file", type=int, default=6, help="每篇 MD 最多抽取的 instruction 对数")
    args = parser.parse_args()

    md_dir = Path(args.md_dir)
    if not md_dir.exists():
        sys.exit(f"[ERROR] MD 目录不存在: {md_dir}")

    source_labels = {}
    source_list_path = Path(args.source_list)
    if source_list_path.exists():
        source_labels = load_source_labels(source_list_path)
        print(f"[元数据] 加载 {len(source_labels)} 条 paper label")
    else:
        print(f"[WARNING] source_list.json 不存在: {source_list_path}，label 将设为 unknown")

    md_files = sorted(md_dir.rglob("*.md"))
    print(f"[扫描  ] 发现 {len(md_files)} 个 .md 文件")

    all_pairs = []
    skipped_empty = 0

    for md_path in md_files:
        pmcid = md_path.parent.name if md_path.parent.name.startswith("PMC") else md_path.stem
        label = source_labels.get(pmcid, "unknown")

        try:
            pairs = build_instruction_pairs(md_path, pmcid, label)
        except Exception as e:
            print(f"  [SKIP] {md_path.name}: {e}")
            continue

        if not pairs:
            skipped_empty += 1
            continue

        if len(pairs) > args.max_pairs_per_file:
            _seed = int.from_bytes(hashlib.sha256(md_path.name.encode()).digest()[:4], "big") % (2**31 - 1)
            rng = random.Random(_seed)
            pairs = rng.sample(pairs, args.max_pairs_per_file)

        all_pairs.extend(pairs)

        if len(all_pairs) % 100 == 0 or len(all_pairs) <= 5:
            print(f"  [{len(all_pairs):4d}] {md_path.name}: {len(pairs)} pairs (label={label})")

    print(f"\n[汇总  ] 总 pair 数: {len(all_pairs)}")
    print(f"  无有效 section: {skipped_empty} 篇")

    section_dist = {}
    label_dist = {}
    for p in all_pairs:
        s = p["_meta"]["section"]
        section_dist[s] = section_dist.get(s, 0) + 1
        l = p["_meta"]["label"]
        label_dist[l] = label_dist.get(l, 0) + 1
    print(f"  section 分布: {dict(sorted(section_dist.items()))}")
    print(f"  label 分布:   {dict(sorted(label_dist.items()))}")

    random.Random(args.seed).shuffle(all_pairs)

    n_train = max(1, int(len(all_pairs) * args.train_ratio))
    n_train = min(n_train, len(all_pairs) - 1) if len(all_pairs) > 1 else len(all_pairs)
    train_pairs = all_pairs[:n_train]
    val_pairs = all_pairs[n_train:]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_instruction_subsets(train_pairs, val_pairs, output_dir)

    if len(all_pairs) > 0:
        print(f"\n总计: train={len(train_pairs)} ({len(train_pairs)/len(all_pairs):.1%}), "
              f"val={len(val_pairs)} ({len(val_pairs)/len(all_pairs):.1%})")
    else:
        print("\n[WARNING] 未生成任何 instruction pair，请检查 MD 文件内容")


if __name__ == "__main__":
    main()
