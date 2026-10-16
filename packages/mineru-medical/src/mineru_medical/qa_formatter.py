"""跨文献证据合成 + 官方 QA 格式封装 — Medical Data Agent Step 6

功能:
  1. 从 kb_quality_report 中加载评估查询
  2. 对每个临床查询执行 Hybrid 检索，获取多篇文献的 top-k chunks
  3. 调用 LLM 合成跨文献答案 + 一致性判断 + 冲突检测
  4. 封装为官方 QA 格式（id输入→输出→来源→说明）
  5. 输出 data/sources_medical/qa/qa_output.json + qa_overview.md

用法:
  .venv/bin/python -m mineru_medical.qa_formatter
  .venv/bin/python -m mineru_medical.qa_formatter --dry-run
  .venv/bin/python -m mineru_medical.qa_formatter --k 3

依赖:
  - data/sources_medical/chunks/chunks_manifest.json
  - data/sources_medical/index/ (FAISS + BM25)
  - kb_quality 产出的 kb_quality_report.json
  - DeepSeek API (DEEPSEEK_API_KEY 环境变量)
"""

import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

from mineru_eval.client_factory import create_client
from mineru_core.paths import PROJECT_ROOT
from mineru_core.settings import PROVIDERS
from mineru_medical.indexing import load_manifest, search_hybrid

OUTPUT_DIR = PROJECT_ROOT / "data" / "sources_medical" / "qa"
REPORT_PATH = PROJECT_ROOT / "benchmark" / "eval" / "results" / "medical" / "kb_quality_report.json"
MANIFEST_PATH = PROJECT_ROOT / "data" / "sources_medical" / "chunks" / "chunks_manifest.json"
INDEX_DIR = PROJECT_ROOT / "data" / "sources_medical" / "index"

DEFAULT_TOP_K = 5
MIN_CHUNK_CHARS = 80

LIMITATIONS_DEFAULT = (
    "基于 Hybrid 检索的 top-k chunks 合成，不保证覆盖全部相关文献。"
    "检索结果受嵌入模型和 BM25 分词影响，部分专业术语可能存在漏检。"
    "LLM 合成可能引入幻觉，关键临床决策前需验证原始文献。"
)

SYNTHESIS_SYSTEM_PROMPT = """你是一位临床证据合成专家。你的任务是基于多篇医学文献的检索结果，合成一个综合性的临床答案。

## 输出格式（严格 JSON，不可包含 Markdown 代码块包裹）
{
  "synthesis": "基于所有检索到的医学文献chunks，综合回答临床问题。合并多篇文献中的数据和结论，按主题组织。包含具体数值（如疗效数据、置信区间、p值）和引用来源（括号注明chunk来源）。若文献间存在分歧，应明确指出。",
  "consistency": "high/medium/low",
  "consistency_reason": "简要说明一致性判断依据",
  "conflicts": [
    "列出文献间的不一致信息，每项一条。无明显冲突则为空数组。"
  ],
  "key_findings": [
    "各文献共享的关键发现，bullet point 形式"
  ]
}

## 一致性判定标准
- high: 多篇文献的主要结论方向一致，数值范围合理（如 A药 vs B药 疗效差异 ±10% 以内）
- medium: 核心结论方向一致，但具体数值或次要结论存在差异
- low: 文献间存在重要结论冲突（如疗效方向相反、安全性信号矛盾）

## 输出要求
- 仅输出上述 JSON 对象，不要包含其他文字
- 引用来源时使用自然语言标注（如"PMCxxxx 研究显示…"）
- conflicts 字段如实记录不一致信息，无冲突时为空数组
- key_findings 至少列 2 条，每条不超过 80 字"""

SYNTHESIS_USER_TEMPLATE = """## 临床查询
{question}

## 检索到的文献 Chunks（共 {n_sources} 个）
{sources_text}

请基于以上多篇医学文献片段，合成一个综合性临床答案。"""


def _stable_id(text, prefix="QA"):
    return f"{prefix}-{hashlib.md5(text.encode()).hexdigest()[:8].upper()}"


def _load_chunk_text_by_filepath(filepath):
    path = PROJECT_ROOT / filepath
    if path.exists():
        return path.read_text(encoding="utf-8").strip()
    return ""


def _format_sources_for_prompt(results):
    lines = []
    for i, r in enumerate(results):
        chunk_id = r.get("chunk_id", f"chunk_{i}")
        filepath = r.get("filepath", "")
        text = r.get("_chunk_text", "") or _load_chunk_text_by_filepath(filepath)
        if not text or len(text) < MIN_CHUNK_CHARS:
            label = r.get("medical_label", "?")
            parent = r.get("medical_parent", "")
            text = f"[chunk {chunk_id}] label={label} parent={parent} (content too short or empty)"
        else:
            text = text[:2000]

        lines.append(
            f"--- 来源 {i+1}: {chunk_id} (rrf={r.get('_rrf_score','?')}, "
            f"label={r.get('medical_label','?')}) ---\n{text}"
        )
    return "\n\n".join(lines)


def _parse_synthesis_response(text):
    text = text.strip()
    json_match = re.search(r"\{[\s\S]*\}", text)
    if json_match:
        text = json_match.group()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        text = text.replace("'", '"')
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {"synthesis": text[:200], "consistency": "unknown", "conflicts": [], "key_findings": []}


def _is_meaningful_query(question_text):
    query_text = question_text.strip()
    if not query_text:
        return False
    meaningful_patterns = [
        r"\?$", r"\？$",
        r"如何", r"什么", r"哪些", r"是否", r"有何",
        r"effect", r"impact", r"outcome", r"efficacy", r"safety",
        r"compare", r"difference", r"association", r"predict",
        r"治疗", r"疗效", r"安全性", r"影响", r"差异",
    ]
    has_meaningful = any(re.search(p, query_text, re.IGNORECASE) for p in meaningful_patterns)
    has_min_length = len(query_text) >= 8
    return has_meaningful and has_min_length


def _process_single_query(query, index_dir, manifest, k, client, model, api_timeout, dry_run):
    """Process a single query: search, format sources, call LLM synth, build QA pair."""
    question = query.get("question", "").strip()
    qa_id = _stable_id(question)

    try:
        results = search_hybrid(question, str(index_dir), manifest, k=k, texts_override=None)
    except Exception as e:
        print(f"    [\u68c0\u7d22\u5931\u8d25] {e}")
        return None

    if not results:
        print("    [\u7a7a\u7ed3\u679c] \u8df3\u8fc7")
        return None

    for r in results:
        filepath = r.get("filepath", "")
        r["_chunk_text"] = _load_chunk_text_by_filepath(filepath)

    sources_text = _format_sources_for_prompt(results)
    sources = []
    for r in results:
        chunk_id = r.get("chunk_id", "")
        text = r.get("_chunk_text", "")
        pmcid = chunk_id.split("_chunk_")[0] if "_chunk_" in chunk_id else ""
        sources.append({
            "chunk_id": chunk_id,
            "pmcid": pmcid,
            "rrf_score": r.get("_rrf_score"),
            "medical_label": r.get("medical_label", ""),
            "medical_parent": r.get("medical_parent"),
            "filepath": r.get("filepath", ""),
            "snippet": (text[:300] + "...") if text else "",
        })

    if dry_run:
        synthesized = {
            "synthesis": "[DRY RUN] \u5408\u6210\u7ed3\u679c\u5360\u4f4d",
            "consistency": "unknown",
            "consistency_reason": "dry run \u6a21\u5f0f\uff0c\u672a\u8c03\u7528 API",
            "conflicts": [],
            "key_findings": ["[DRY RUN]"],
        }
    else:
        user_prompt = SYNTHESIS_USER_TEMPLATE.format(
            question=question,
            n_sources=len(results),
            sources_text=sources_text,
        )
        for attempt in range(2):
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": SYNTHESIS_SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.3,
                    max_tokens=2048,
                    timeout=api_timeout,
                )
                raw = resp.choices[0].message.content or ""
                synthesized = _parse_synthesis_response(raw)
                if synthesized.get("synthesis"):
                    break
            except Exception as e:
                print(f"    [API \u5931\u8d25 \u5c1d\u8bd5{attempt+1}] {e}", flush=True)
                synthesized = {"synthesis": "", "consistency": "unknown", "conflicts": [], "key_findings": []}
                time.sleep(2)

    qa_pair = {
        "qa_id": qa_id,
        "question": question,
        "source_chunk_id": query.get("source_chunk_id"),
        "source_label": query.get("source_label"),
        "output": synthesized.get("synthesis", ""),
        "consistency": synthesized.get("consistency", "unknown"),
        "consistency_reason": synthesized.get("consistency_reason", ""),
        "conflicts": synthesized.get("conflicts", []),
        "key_findings": synthesized.get("key_findings", []),
        "sources": sources,
        "n_sources": len(sources),
    }
    return qa_pair


def build_qa_pairs(kb_report_path=REPORT_PATH, manifest_path=MANIFEST_PATH,
                   index_dir=INDEX_DIR, k=DEFAULT_TOP_K, dry_run=False,
                   api_timeout=60, limit_queries=0):
    with open(kb_report_path, "r", encoding="utf-8") as f:
        report = json.load(f)

    manifest = load_manifest(manifest_path)
    queries = report.get("queries", [])
    if not queries:
        print("[错误] kb_quality_report 中无 queries 字段")
        return []

    client, model, _ = create_client(
        api_key=os.environ.get("DEEPSEEK_API_KEY", ""),
        base_url=PROVIDERS["deepseek"].base_url,
        model=PROVIDERS["deepseek"].default_model,
    )

    qa_pairs = []
    meaningful_count = 0

    if limit_queries and limit_queries > 0:
        queries = queries[:limit_queries]

    for qi, query in enumerate(queries):
        question = query.get("question", "").strip()
        if not question:
            continue

        if not _is_meaningful_query(question):
            print(f"  [{qi+1}/{len(queries)}] \u8df3\u8fc7\u7b80\u5355\u67e5\u8be2: {question!r}")
            continue

        meaningful_count += 1
        print(f"  [{qi+1}/{len(queries)}] QA#{_stable_id(question)} \u5904\u7406: {question[:60]}...", flush=True)

        qa_pair = _process_single_query(query, index_dir, manifest, k, client, model, api_timeout, dry_run)
        if qa_pair is None:
            continue

        qa_pairs.append(qa_pair)
        print(f"    \u2192 consistency={qa_pair.get('consistency','?')} | "
              f"sources={qa_pair['n_sources']} | conflicts={len(qa_pair.get('conflicts',[]))}", flush=True)

    print(f"[完成] 处理 {meaningful_count}/{len(queries)} 个有意义查询 → {len(qa_pairs)} 个 QA 对")
    return qa_pairs


def save_output(qa_pairs, output_dir=OUTPUT_DIR):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "qa_output.json"
    tmp_path = json_path.with_suffix(".tmp")
    output = {
        "meta": {
            "title": "Medical RAG 跨文献证据合成 — QA 格式化输出",
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "total_qa_pairs": len(qa_pairs),
            "format_version": "1.0",
            "format_note": "P1=问题输入, P2=综合答案, P3=来源, P4=限制声明 | 参考 MinerU 官方示例格式",
            "limitations": LIMITATIONS_DEFAULT,
        },
        "qa_pairs": qa_pairs,
    }
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    tmp_path.replace(json_path)
    print(f"[保存] {json_path}")

    md_path = output_dir / "qa_overview.md"
    lines = [
        "# Medical RAG QA 格式化输出\n",
        f"> 生成时间: {output['meta']['generated_at']} | QA 对总数: {len(qa_pairs)}\n\n",
        "> **提取限制**: " + LIMITATIONS_DEFAULT + "\n\n",
        "| QA ID | 临床问题 | 一致性 | 来源数 | 冲突数 | 关键发现 |\n",
        "|:------|:---------|:------:|:------:|:------:|:----------|\n",
    ]
    for qa in qa_pairs:
        findings = "; ".join(qa.get("key_findings", [])[:2])
        lines.append(
            f"| {qa['qa_id']} | {qa['question'][:60]} | {qa['consistency']} "
            f"| {qa['n_sources']} | {len(qa.get('conflicts',[]))} | {findings[:80]} |\n"
        )

    lines.extend([
        "\n---\n\n",
        "## 详细 QA 对\n",
    ])
    for qa in qa_pairs:
        lines.append(f"\n### QA ID: {qa['qa_id']}\n")
        lines.append(f"**临床问题**: {qa['question']}\n\n")
        lines.append(f"**综合答案**:\n\n{qa['output']}\n\n")
        lines.append(f"**跨文献一致性**: {qa['consistency']}")
        if qa.get("consistency_reason"):
            lines.append(f" ({qa['consistency_reason']})")
        lines.append("\n\n")
        if qa.get("conflicts"):
            lines.append("**文献间冲突**:\n")
            for c in qa["conflicts"]:
                lines.append(f"- {c}\n")
            lines.append("\n")
        if qa.get("key_findings"):
            lines.append("**关键发现**:\n")
            for f_item in qa["key_findings"]:
                lines.append(f"- {f_item}\n")
            lines.append("\n")
        lines.append("**来源文献**:\n")
        for s in qa.get("sources", []):
            snippet = s.get("snippet", "")[:120].replace("\n", " ")
            lines.append(f"- {s['chunk_id']} (rrf={s.get('rrf_score','?')}, label={s.get('medical_label','?')}): {snippet}...\n")
        lines.append("\n---\n")

    with open(md_path, "w", encoding="utf-8") as f:
        f.writelines(lines)
    print(f"[保存] {md_path}")

    consistency_counts = {}
    for qa in qa_pairs:
        c = qa["consistency"]
        consistency_counts[c] = consistency_counts.get(c, 0) + 1
    print(f"[统计] consistency 分布: {dict(sorted(consistency_counts.items()))}")
    print(f"[统计] 总冲突数: {sum(len(qa.get('conflicts',[])) for qa in qa_pairs)}")


def main():
    parser = argparse.ArgumentParser(description="跨文献证据合成 + 官方 QA 格式封装")
    parser.add_argument("--kb-report", default=str(REPORT_PATH), help="kb_quality_report.json 路径")
    parser.add_argument("--manifest", default=str(MANIFEST_PATH), help="chunks manifest 路径")
    parser.add_argument("--index-dir", default=str(INDEX_DIR), help="Hybrid 索引目录")
    parser.add_argument("--output", default=str(OUTPUT_DIR), help="QA 输出目录")
    parser.add_argument("--k", type=int, default=DEFAULT_TOP_K, help="每查询检索 top-k (默认 5)")
    parser.add_argument("--dry-run", action="store_true", help="跳过 API 调用，仅测试流程")
    parser.add_argument("--api-timeout", type=int, default=60, help="API 调用超时秒数 (默认 60)")
    parser.add_argument("--limit-queries", type=int, default=0, help="Limit report queries for smoke tests (0 = all)")
    parser.add_argument("--allow-empty", action="store_true", help="Exit 0 when no QA pairs are produced.")
    args = parser.parse_args()

    qa_pairs = build_qa_pairs(
        kb_report_path=args.kb_report,
        manifest_path=args.manifest,
        index_dir=args.index_dir,
        k=args.k,
        dry_run=args.dry_run,
        api_timeout=args.api_timeout,
        limit_queries=args.limit_queries,
    )

    if qa_pairs:
        save_output(qa_pairs, output_dir=args.output)
    else:
        print("[警告] 无 QA 对产出")
        if not args.allow_empty:
            sys.exit(1)


if __name__ == "__main__":
    main()
