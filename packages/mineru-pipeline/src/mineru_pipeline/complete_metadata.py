"""
T1: 元数据分类补全 — 将 control_category 统一校准为14类标准标签，补全 author/source/copyright
策略:
  1. 用 DeepSeek API 对每篇文档 (title + 首段md) 做14类分类
  2. 对 arXiv 论文用 arXiv API 补全 author/source 元数据
  3. 每10条 checkpoint，支持 resume 续传
  4. 交叉审核：随机10%样本重跑验证一致性

标准14分类:
  PID控制, 估计与定位, 导航制导与控制, 控制理论, 数字控制, 智能控制,
  最优控制, 现代控制, 线性系统, 经典控制, 自抗扰控制, 自适应控制, 非线性控制, 鲁棒控制
"""

import argparse
import json
import random
import time
import xml.etree.ElementTree as ET
from datetime import date
import os
from pathlib import Path
from typing import Tuple

import requests

# Phase 5: restored from new package location
ROOT = Path(__file__).resolve().parents[4]  # packages/mineru-pipeline/src/mineru_pipeline/ → project root
METADATA_PATH = ROOT / "corpus" / "metadata.json"
PROCESSED_DIR = ROOT / "data" / "processed"
CHECKPOINT_PATH = ROOT / "pipeline" / "complete_metadata.checkpoint.json"

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")

STANDARD_CATEGORIES = [
    "PID控制", "估计与定位", "导航制导与控制", "控制理论", "数字控制",
    "智能控制", "最优控制", "现代控制", "线性系统", "经典控制",
    "自抗扰控制", "自适应控制", "非线性控制", "鲁棒控制",
]

CLASSIFY_SYSTEM_PROMPT = """你是控制科学与工程领域的资深专家，擅长对学术论文和教材进行精细分类。
你的任务是根据文档内容，从给定的14个标准分类中选择最匹配的1-2个分类。
你必须只返回JSON，不包含任何其他文字。"""

CLASSIFY_USER_PROMPT_TEMPLATE = """请根据以下文档信息，从列表中选择最匹配的控制分类(选1-2个):

标准分类列表: {categories}

文档标题: {title}

文档内容摘要:
{content}

返回JSON格式（只返回JSON，不要其他内容）:
{{"categories": ["分类1"], "confidence": "high|medium|low", "reasoning": "一句话理由"}}
"""

CROSS_VALIDATE_USER_PROMPT_TEMPLATE = """请重新审视以下文档的分类结果是否正确。
文档已被初步分类为: {existing_cats}

文档标题: {title}

文档内容摘要:
{content}

标准分类列表: {categories}

如果原分类正确，返回: {{"decision": "agree", "categories": {existing_cats_json}, "reasoning": "..."}}
如果需要修改，返回: {{"decision": "revise", "categories": ["新分类1", "新分类2"], "reasoning": "..."}}
只返回JSON。"""

CAT_DIFFICULTY_MAP = {
    "经典控制": "L1", "PID控制": "L1",
    "现代控制": "L2", "线性系统": "L2", "数字控制": "L2", "控制理论": "L2", "估计与定位": "L2",
    "最优控制": "L3", "鲁棒控制": "L3", "自适应控制": "L3",
    "非线性控制": "L3", "智能控制": "L3", "导航制导与控制": "L3", "自抗扰控制": "L3",
}


def read_doc_content(filename):
    """读取文档首段内容 (最多2000字符)，从第一个 # 标题行开始；无标题则取全文"""
    doc_dir = PROCESSED_DIR / filename
    if not doc_dir.exists():
        return ""

    md_files = sorted(doc_dir.glob("*.md"), key=lambda f: f.stat().st_size, reverse=True)
    if not md_files:
        return ""

    for md_file in md_files:
        try:
            text = md_file.read_text(encoding="utf-8", errors="replace")
            if len(text.strip()) < 50:
                continue
            lines = text.split("\n")
            start_idx = 0
            for i, line in enumerate(lines):
                if line.strip().startswith("#"):
                    start_idx = i
                    break
            content_lines = []
            for line in lines[start_idx:]:
                content_lines.append(line)
                if sum(len(l) for l in content_lines) > 2000:
                    break
            result = "\n".join(content_lines)
            if sum(len(l) for l in content_lines) < 100:
                result = text[:2000]
            return result
        except Exception:
            continue
    return ""


def call_deepseek(system_prompt, user_prompt, max_tokens=1024, timeout=120, max_retries=3):
    """调用 DeepSeek API — 使用 mineru_core 的 LLM 客户端"""
    from mineru_core.llm_client import call_openai_direct

    result = call_openai_direct(
        DEEPSEEK_API_KEY, system_prompt, user_prompt,
        timeout=timeout, max_retries=max_retries, max_tokens=max_tokens,
    )
    if result is None:
        print(f"  [API] call_openai_direct returned None after {max_retries} retries", flush=True)
    return result


def classify_doc(doc, content):
    """用 DeepSeek 对一篇文档做14类标准分类"""
    title = doc.get("title", "")
    prompt = CLASSIFY_USER_PROMPT_TEMPLATE.format(
        categories=", ".join(STANDARD_CATEGORIES),
        title=title,
        content=content[:2000],
    )
    result = call_deepseek(CLASSIFY_SYSTEM_PROMPT, prompt, max_tokens=512)
    if result is None:
        return None
    cats = result.get("categories", [])
    # 过滤非标准分类
    valid_cats = [c for c in cats if c in STANDARD_CATEGORIES]
    if not valid_cats:
        # fallback: 保留原分类
        print(f"  [WARN] 分类结果不在标准列表中: {cats}, 保留原分类", flush=True)
        return None
    return {
        "categories": valid_cats,
        "confidence": result.get("confidence", "medium"),
        "reasoning": result.get("reasoning", ""),
    }


def cross_validate_doc(doc, content, existing_cats):
    """交叉审核：重新判断分类是否正确"""
    title = doc.get("title", "")
    prompt = CROSS_VALIDATE_USER_PROMPT_TEMPLATE.format(
        existing_cats=", ".join(existing_cats),
        existing_cats_json=json.dumps(existing_cats, ensure_ascii=False),
        title=title,
        content=content[:2000],
        categories=", ".join(STANDARD_CATEGORIES),
    )
    return call_deepseek(CLASSIFY_SYSTEM_PROMPT, prompt, max_tokens=512)


def _parse_arxiv_xml_response(resp_text: str) -> dict | None:
    """Parse arXiv Atom XML response into title/authors/year dict."""
    ns = {
        "atom": "http://www.w3.org/2005/Atom",
        "arxiv": "http://arxiv.org/schemas/atom",
    }
    root = ET.fromstring(resp_text)
    entry = root.find("atom:entry", ns)
    if entry is None:
        print("  [arXiv] No entry found", flush=True)
        return None

    title_el = entry.find("atom:title", ns)
    title = title_el.text.strip().replace("\n", " ") if title_el is not None else None

    authors = []
    for author_el in entry.findall("atom:author", ns):
        name_el = author_el.find("atom:name", ns)
        if name_el is not None and name_el.text:
            authors.append(name_el.text.strip())

    published_el = entry.find("atom:published", ns)
    year = None
    if published_el is not None and published_el.text:
        try:
            year = int(published_el.text[:4])
        except (ValueError, IndexError):
            pass

    return {"title": title, "authors": authors, "year": year}


def query_arxiv_api(arxiv_id, max_retries=2):
    """\u901a\u8fc7 arXiv API \u83b7\u53d6\u8bba\u6587\u5143\u6570\u636e (\u542b\u9000\u907f\u91cd\u8bd5)"""
    url = f"https://export.arxiv.org/api/query?id_list={arxiv_id}&max_results=1"
    for attempt in range(max_retries + 1):
        try:
            resp = requests.get(url, timeout=30)
            if resp.status_code in (429, 503):
                retry_after = resp.headers.get("Retry-After")
                wait_seconds = int(retry_after) if retry_after and retry_after.isdigit() else 10 * (attempt + 1)
                print(f"  [arXiv] HTTP {resp.status_code} for {arxiv_id}, attempt {attempt+1}/{max_retries+1}", flush=True)
                if attempt < max_retries:
                    time.sleep(wait_seconds)
                    continue
                return None
            if resp.status_code != 200:
                print(f"  [arXiv] HTTP {resp.status_code} for {arxiv_id}", flush=True)
                return None
            return _parse_arxiv_xml_response(resp.text)
        except requests.exceptions.Timeout:
            print(f"  [arXiv] Timeout for {arxiv_id}, attempt {attempt+1}/{max_retries+1}", flush=True)
            if attempt < max_retries:
                time.sleep(2 ** attempt)
                continue
            return None
        except ET.ParseError as e:
            print(f"  [arXiv] XML parse error for {arxiv_id}: {e}", flush=True)
            return None
        except Exception as e:
            print(f"  [arXiv] Error querying {arxiv_id}: {e}", flush=True)
            if attempt < max_retries:
                time.sleep(2 ** attempt)
                continue
            return None


def load_checkpoint():
    """加载断点续传数据"""
    if CHECKPOINT_PATH.exists():
        with open(CHECKPOINT_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {"classified_ids": [], "classification_results": {}, "arxiv_enriched_ids": []}


def save_checkpoint(checkpoint_data):
    """保存断点"""
    checkpoint_data["updated"] = str(date.today())
    with open(CHECKPOINT_PATH, "w", encoding="utf-8") as f:
        json.dump(checkpoint_data, f, ensure_ascii=False, indent=2)


def determine_difficulty(categories):
    """根据分类确定难度等级（L1=基础, L2=中等, L3=前沿）"""
    levels = [CAT_DIFFICULTY_MAP[c] for c in categories if c in CAT_DIFFICULTY_MAP]
    if not levels:
        return "L3"
    order = {"L1": 1, "L2": 2, "L3": 3, "L4": 4}
    return min(levels, key=order.get)


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the T1 metadata pipeline."""
    parser = argparse.ArgumentParser(description="T1: 元数据分类补全")
    parser.add_argument("--limit", type=int, default=0, help="限制处理文档数 (0=全部)")
    parser.add_argument("--offset", type=int, default=0, help="跳过前N篇文档（并行分片用）")
    parser.add_argument("--checkpoint-every", type=int, default=10, help="每N条保存checkpoint")
    parser.add_argument("--checkpoint-path", type=str, default="", help="自定义checkpoint路径（并行分片用）")
    parser.add_argument("--skip-classify", action="store_true", help="跳过分类阶段")
    parser.add_argument("--skip-arxiv", action="store_true", help="跳过arXiv API补全")
    parser.add_argument("--skip-cross", action="store_true", help="跳过交叉审核")
    parser.add_argument("--skip-write-metadata", action="store_true", help="跳过写入metadata.json")
    return parser.parse_args()


def _load_metadata() -> Tuple[dict, list]:
    """Load metadata.json and return (metadata, all_docs)."""
    with open(METADATA_PATH, encoding="utf-8") as f:
        metadata = json.load(f)
    return metadata, metadata["docs"]


def _select_working_docs(all_docs: list, args: argparse.Namespace) -> Tuple[list, set]:
    """Select working subset based on limit/offset args."""
    if args.limit > 0:
        working_docs = all_docs[args.offset:args.offset + args.limit]
    elif args.offset > 0:
        working_docs = all_docs[args.offset:]
    else:
        working_docs = all_docs
    working_ids = {d["id"] for d in working_docs}
    print(f"[T1] 加载 {len(all_docs)} 篇文档, 处理 {len(working_docs)} 篇 "
          f"(offset={args.offset}, limit={args.limit or 'all'})", flush=True)
    return working_docs, working_ids


def _prepare_checkpoint(args) -> dict:
    """Load checkpoint, applying custom path if provided."""
    if args.checkpoint_path:
        global CHECKPOINT_PATH
        CHECKPOINT_PATH = Path(args.checkpoint_path)
    return load_checkpoint()


def _phase_classify(
    args, working_docs, cp, classified_ids, classification_results
) -> Tuple[set, dict]:
    """Phase A: Classify working docs using DeepSeek API."""
    if args.skip_classify:
        print("[T1] Phase A: 跳过 (--skip-classify)", flush=True)
        classified_ids |= {d["id"] for d in working_docs}
        return classified_ids, classification_results

    remaining = [d for d in working_docs if d["id"] not in classified_ids]
    print(f"[T1] Phase A: 待分类 {len(remaining)} 篇", flush=True)

    batch_count = len(classified_ids & {d["id"] for d in working_docs})
    for doc in remaining:
        doc_id = doc["id"]
        filename = doc["filename"]
        print(f"[{doc_id}] 正在分类: {doc.get('title', filename)[:80]}...", flush=True)

        content = read_doc_content(filename)
        if not content:
            print("  [SKIP] 无法读取内容", flush=True)
            classified_ids.add(doc_id)
            batch_count += 1
            continue

        result = classify_doc(doc, content)
        if result is None:
            print("  [FAIL] 分类失败，保留原分类", flush=True)
            classified_ids.add(doc_id)
        else:
            classification_results[doc_id] = result
            classified_ids.add(doc_id)
            print(f"  → {result['categories']} (confidence: {result['confidence']})", flush=True)
        batch_count += 1

        if batch_count % args.checkpoint_every == 0:
            cp["classified_ids"] = sorted(classified_ids)
            cp["classification_results"] = classification_results
            save_checkpoint(cp)
            print(f"  [CHECKPOINT] {len(classified_ids)}/{len(working_docs)} 已分类", flush=True)

    cp["classified_ids"] = sorted(classified_ids)
    cp["classification_results"] = classification_results
    save_checkpoint(cp)
    print(f"\n[Phase A 完成] {len(classified_ids)}/{len(working_docs)} 已分类", flush=True)
    return classified_ids, classification_results


def _apply_classification_results(all_docs, classification_results) -> int:
    """Apply classification results to all_docs and count changes."""
    changes = 0
    for doc in all_docs:
        doc_id = doc["id"]
        if doc_id in classification_results:
            new_cats = classification_results[doc_id]["categories"]
            old_cats = doc.get("control_category", [])
            if set(new_cats) != set(old_cats):
                changes += 1
            doc["control_category"] = new_cats
            doc["difficulty_level"] = determine_difficulty(new_cats)
    print(f"[T1] 分类变更: {changes}/{len(all_docs)} 篇", flush=True)
    return changes


def _enrich_single_arxiv_doc(doc: dict, arxiv_enriched_ids: set) -> bool:
    """Enrich a single arXiv document with API metadata. Returns True if enriched."""
    arxiv_id = doc.get("arxiv_id")
    arxiv_meta = query_arxiv_api(arxiv_id)
    if not arxiv_meta:
        return False
    if arxiv_meta.get("authors"):
        doc["author"] = ", ".join(arxiv_meta["authors"])
    if arxiv_meta.get("title") and (
        "Unknown" in doc.get("title", "") or doc.get("title", "") == doc.get("filename", "")
    ):
        doc["title"] = arxiv_meta["title"]
    if arxiv_meta.get("year"):
        doc["year"] = arxiv_meta["year"]
    doc["source"] = f"arXiv: {arxiv_id}"
    doc["copyright"] = "arXiv Open Access"
    arxiv_enriched_ids.add(doc["id"])
    return True


def _phase_arxiv_enrich(args, working_docs, cp, arxiv_enriched_ids) -> set:
    """Phase B: Enrich arXiv metadata via arXiv API."""
    if args.skip_arxiv:
        print("\n[Phase B] arXiv API: 跳过 (--skip-arxiv)", flush=True)
        return arxiv_enriched_ids

    arxiv_docs = [d for d in working_docs if d["type"] == "arxiv"]
    needs_arxiv = [
        d for d in arxiv_docs
        if ("Unknown" in d.get("author", "") or "To be verified" in d.get("author", ""))
        and d.get("arxiv_id")
        and d["id"] not in arxiv_enriched_ids
    ]
    print(f"\n[Phase B] arXiv API: 待补全 {len(needs_arxiv)} 篇", flush=True)

    arxiv_batch = 0
    for doc in needs_arxiv:
        doc_id = doc["id"]
        print(f"[{doc_id}] arXiv查询: {doc.get('arxiv_id')} ...", flush=True)

        if _enrich_single_arxiv_doc(doc, arxiv_enriched_ids):
            if "author" in doc:
                print(f"  → author: {doc['author'][:80]}", flush=True)
            if doc.get("title") and "Unknown" not in doc.get("title", ""):
                print(f"  → title: {doc['title'][:80]}", flush=True)
            arxiv_batch += 1
        else:
            print("  [FAIL] arXiv API 无数据", flush=True)

        if arxiv_batch % args.checkpoint_every == 0 and arxiv_batch > 0:
            cp["arxiv_enriched_ids"] = sorted(arxiv_enriched_ids)
            save_checkpoint(cp)
            print(f"  [CHECKPOINT] {len(arxiv_enriched_ids)} 已补arXiv", flush=True)

    cp["arxiv_enriched_ids"] = sorted(arxiv_enriched_ids)
    save_checkpoint(cp)
    print(f"[Phase B 完成] {len(arxiv_enriched_ids)} 篇已补arXiv元数据", flush=True)
    return arxiv_enriched_ids


def _setup_cross_sample(classified_ids, working_ids, working_docs, classification_results):
    """Set up a random 10% sample for cross-validation."""
    random.seed(42)
    classified_list = sorted(classified_ids & working_ids)
    n_sample = min(max(10, len(classified_list) // 10), len(classified_list))
    sample_ids = set(random.sample(classified_list, n_sample))
    return [d for d in working_docs if d["id"] in sample_ids and d["id"] in classification_results]


def _process_single_cross_validation(doc, classification_results):
    """Cross-validate a single document and return (decision, detail, changed_cats)."""
    doc_id = doc["id"]
    existing_cats = classification_results[doc_id]["categories"]
    content = read_doc_content(doc["filename"])
    if not content:
        return "failed", {"decision": "failed", "categories": existing_cats, "reasoning": "no content"}, None

    result = cross_validate_doc(doc, content, existing_cats)
    if result is None:
        print(f"  [{doc_id}] 失败: API returned None", flush=True)
        return "failed", {"decision": "failed", "categories": existing_cats, "reasoning": "API returned None"}, None
    elif result.get("decision") == "revise":
        new_cats = [c for c in result.get("categories", []) if c in STANDARD_CATEGORIES]
        detail = {
            "decision": "revise", "categories": new_cats or existing_cats,
            "original_categories": existing_cats, "reasoning": result.get("reasoning", ""),
        }
        changed = new_cats if (new_cats and set(new_cats) != set(existing_cats)) else None
        if changed:
            print(f"  [{doc_id}] 修订: {existing_cats} → {new_cats}", flush=True)
        return "revise", detail, changed
    else:
        detail = {"decision": "agree", "categories": existing_cats, "reasoning": result.get("reasoning", "")}
        return "agree", detail, None


def _phase_cross_validate(
    args, working_docs, working_ids, classification_results, classified_ids, cp, all_docs
) -> dict:
    """Phase C: Cross-validate classification on a random 10% sample."""
    if args.skip_cross:
        print("\n[Phase C] 交叉审核: 跳过 (--skip-cross)", flush=True)
        return {"agree": 0, "revise": 0, "failed": 0}

    sample_docs = _setup_cross_sample(classified_ids, working_ids, working_docs, classification_results)
    print(f"\n[Phase C] 交叉审核: 抽样 {len(sample_docs)} 篇", flush=True)
    cross_results = {"agree": 0, "revise": 0, "failed": 0}
    cross_details = {}
    cross_changes = {}

    for doc in sample_docs:
        doc_id = doc["id"]
        decision, detail, changed = _process_single_cross_validation(doc, classification_results)
        cross_results[decision] = cross_results.get(decision, 0) + 1
        cross_details[doc_id] = detail
        if changed is not None:
            cross_changes[doc_id] = changed

    for doc_id, new_cats in cross_changes.items():
        classification_results[doc_id]["categories"] = new_cats
        classification_results[doc_id]["cross_revised"] = True
        for doc in all_docs:
            if doc["id"] == doc_id:
                doc["control_category"] = new_cats
                doc["difficulty_level"] = determine_difficulty(new_cats)
                break

    cp["cross_results"] = {"summary": cross_results, "sample_ids": [d["id"] for d in sample_docs], "details": cross_details}
    if cross_changes:
        cp["classification_results"] = classification_results
    save_checkpoint(cp)
    print(f"  [CHECKPOINT] 交叉审核结果 {len(cross_details)} 条已保存", flush=True)
    print(f"[Phase C 完成] agree={cross_results['agree']}, revise={cross_results['revise']}, failed={cross_results['failed']}", flush=True)
    return cross_results


def _write_metadata(args, metadata, all_docs, classification_changes, arxiv_enriched_ids, cross_results):
    """Phase D: Write final metadata to disk."""
    if args.skip_write_metadata:
        print("[T1] --skip-write-metadata: 跳过写入metadata.json", flush=True)
        return

    metadata["updated"] = str(date.today())
    metadata["version"] = "1.1-standardized"
    metadata["generation_note"] = (
        f"T1标准化: control_category统一为14类标准标签; "
        f"{classification_changes}篇分类变更; {len(arxiv_enriched_ids)}篇arXiv补全; "
        f"cross_validation: {cross_results}"
    )
    metadata["docs"] = all_docs
    metadata["total_docs"] = len(all_docs)

    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}", flush=True)
    print(f"[T1 完成] metadata.json 已更新 ({METADATA_PATH.stat().st_size / 1024:.1f} KB)", flush=True)


def _print_validation_summary(all_docs):
    """Phase E: Print validation summary statistics."""
    empty_author = sum(1 for d in all_docs if "Unknown" in d.get("author", "") or "To be verified" in d.get("author", ""))
    non_std_cats = 0
    cat_dist = {}
    for d in all_docs:
        for c in d.get("control_category", []):
            if c not in STANDARD_CATEGORIES:
                non_std_cats += 1
            cat_dist[c] = cat_dist.get(c, 0) + 1

    print("\n[验证结果]", flush=True)
    print(f"  author 待补全: {empty_author}/{len(all_docs)}", flush=True)
    print(f"  非标准分类: {non_std_cats}", flush=True)
    print("  分类分布 (14类标准):", flush=True)
    for cat in STANDARD_CATEGORIES:
        count = cat_dist.get(cat, 0)
        bar = "█" * (count // 2)
        print(f"    {cat: <10s} {count:>3d}  {bar}", flush=True)


def main():
    """T1: 元数据分类补全 — orchestrate all phases."""
    args = _parse_args()
    metadata, all_docs = _load_metadata()
    working_docs, working_ids = _select_working_docs(all_docs, args)
    cp = _prepare_checkpoint(args)

    classified_ids = set(cp.get("classified_ids", []))
    classification_results = cp.get("classification_results", {})
    arxiv_enriched_ids = set(cp.get("arxiv_enriched_ids", []))
    print(f"[T1] Checkpoint: {len(classified_ids)} 已分类, {len(arxiv_enriched_ids)} 已补arXiv", flush=True)

    # Phase A: Classification
    classified_ids, classification_results = _phase_classify(
        args, working_docs, cp, classified_ids, classification_results
    )

    # Apply classification results to all docs
    classification_changes = _apply_classification_results(all_docs, classification_results)

    # Phase B: arXiv enrichment
    arxiv_enriched_ids = _phase_arxiv_enrich(args, working_docs, cp, arxiv_enriched_ids)

    # Phase C: Cross-validation
    cross_results = _phase_cross_validate(
        args, working_docs, working_ids, classification_results, classified_ids, cp, all_docs
    )

    # Phase D: Write metadata
    _write_metadata(args, metadata, all_docs, classification_changes, arxiv_enriched_ids, cross_results)

    # Phase E: Print summary
    _print_validation_summary(all_docs)


if __name__ == "__main__":
    main()
