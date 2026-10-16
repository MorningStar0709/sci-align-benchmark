import asyncio
import base64
import json
import os
import re
import time
import threading
from pathlib import Path
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

import httpx
from sciverse import AgentToolsClient

ROOT = Path(__file__).resolve().parents[4]  # restored Phase 5

import mineru_core  # noqa: F401 — trigger _load_dotenv()
from mineru_core.settings import PROVIDERS

API_KEY = os.environ.get("SCIVERSE_API_KEY", "")
MIMO_API_KEY = os.environ.get("MIMO_API_KEY", "")
MIMO_BASE_URL = PROVIDERS["mimo"].base_url
MIMO_MODEL = "mimo-v2.5"

AUDIT_JSON = ROOT / "benchmark" / "eval" / "results" / "sciverse_cross_modal_audit.json"
IMAGE_DIR = ROOT / "benchmark" / "data" / "sciverse_images"
OUTPUT_DIR = ROOT / "benchmark" / "eval" / "results"
REPORT_PATH = OUTPUT_DIR / "sciverse_visual_audit_report.json"
RESULTS_LOG = OUTPUT_DIR / "sciverse_visual_audit_results.jsonl"

IMAGE_DIR.mkdir(parents=True, exist_ok=True)

_thread_local = threading.local()

def _get_mimo_client():
    if not hasattr(_thread_local, 'httpx_client'):
        _thread_local.httpx_client = httpx.Client(proxy=None, timeout=120)
    return _thread_local.httpx_client

VISION_JUDGE_SYSTEM = (
    "你是一个专业的跨模态对齐审计助手。判断科学文档图片与 LaTeX 公式之间的语义相关性。"
    "严格基于图片中的视觉内容做判断，不要臆想图片中不存在的内容。"
)

VISION_JUDGE_USER_TEMPLATE = (
    "你是一个控制科学领域的跨模态对齐评审专家，负责判断PDF中提取的图片与LaTeX公式之间的语义相关性。\n\n"
    "以下是同一论文中提取的：\n"
    "(1) 一张图片（控制工程/科学相关的图表、框图、仿真曲线等）\n"
    "(2) 该论文中的 LaTeX 公式代码\n\n"
    "请判断：这张图片的**核心内容**是否与 LaTeX 公式描述的数学语义相关？\n\n"
    "--- LaTeX 公式 ---\n{formulas}\n\n"
    "--- 文本上下文（前400字） ---\n{text_context}\n\n"
    "请用以下严格格式回答：\n判断: [一致 | 部分一致 | 不一致 | 不确定]\n理由: [10-30字的判断依据]"
)


def call_mimo_judge(image_path, formulas_text, text_context):
    with open(image_path, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode("utf-8")
    user_prompt = VISION_JUDGE_USER_TEMPLATE.format(formulas=formulas_text, text_context=text_context[:400])
    payload = {
        "model": MIMO_MODEL,
        "messages": [
            {"role": "system", "content": VISION_JUDGE_SYSTEM},
            {"role": "user", "content": [{"type": "text", "text": user_prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}", "detail": "high"}}]},
        ],
        "temperature": 1.0, "max_tokens": 512, "thinking": {"type": "disabled"},
    }
    headers = {"api-key": MIMO_API_KEY, "Content-Type": "application/json"}
    client = _get_mimo_client()
    resp = client.post(f"{MIMO_BASE_URL}/chat/completions", json=payload, headers=headers)
    if resp.status_code != 200:
        raise RuntimeError(f"MiMo HTTP {resp.status_code}: {resp.text[:200]}")
    data = resp.json()
    content = data["choices"][0]["message"]["content"]
    usage = data.get("usage", {})
    details = usage.get("prompt_tokens_details", {})
    tokens = {"prompt_tokens": usage.get("prompt_tokens", 0), "completion_tokens": usage.get("completion_tokens", 0),
              "image_tokens": details.get("image_tokens", 0)}
    return content.strip(), tokens


def parse_judgment(raw):
    raw = raw.strip()
    if "一致" in raw: return "consistent", raw
    elif "部分一致" in raw or "部分相关" in raw: return "partial", raw
    elif "不一致" in raw or "不相关" in raw: return "inconsistent", raw
    return "uncertain", raw


FM_BLOCK = re.compile(r'\$\$(.+?)\$\$', re.DOTALL)
FM_INLINE = re.compile(r'(?<!\$)\$(?!\$)([^$]+?)(?<!\$)\$(?!\$)')

def extract_formulas(text):
    scan = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
    formulas = []
    for m in FM_BLOCK.findall(scan): formulas.append(m.strip())
    for m in FM_INLINE.findall(scan): formulas.append(m.strip())
    return formulas


async def read_content_with_retry(client, doc_id, offset, limit, max_retries=5):
    for attempt in range(max_retries):
        try:
            return await client.read_content(doc_id=doc_id, offset=offset, limit=limit)
        except Exception as e:
            if "429" in str(e) and attempt < max_retries - 1:
                wait = (attempt + 1) * 5
                print(f"    rate limited, sleeping {wait}s...")
                await asyncio.sleep(wait)
                continue
            raise


async def get_resource_with_retry(client, file_name, max_retries=5):
    for attempt in range(max_retries):
        try:
            return await client.get_resource(file_name=file_name)
        except Exception as e:
            if "429" in str(e) and attempt < max_retries - 1:
                wait = (attempt + 1) * 3
                await asyncio.sleep(wait)
                continue
            raise


async def download_images_for_paper(client, doc_id, label):
    subfield_dir = IMAGE_DIR / label
    subfield_dir.mkdir(parents=True, exist_ok=True)

    all_image_refs = []
    all_text = ""
    offset = 0
    MAX_SCAN = 80000

    for chunk_i in range(6):
        try:
            content = await read_content_with_retry(client, doc_id, offset, 10000)
        except Exception as e:
            print(f"    content read error at offset {offset}: {str(e)[:100]}")
            break
        text = content.get("text", "")
        all_text += text
        refs = re.findall(r'!\[.*?\]\(([^)]+)\)', text)
        all_image_refs.extend(refs)
        if not content.get("more", False) or offset >= MAX_SCAN:
            break
        offset = content.get("next_offset", offset + 10000)

    all_image_refs = list(dict.fromkeys(all_image_refs))
    print(f"  {len(all_image_refs)} image refs, {len(all_text)} chars")

    formulas = extract_formulas(all_text)

    downloaded = []
    for i, ref in enumerate(all_image_refs):
        hash_name = ref.rsplit("/", 1)[-1] if "/" in ref else ref
        saved_path = subfield_dir / hash_name

        if saved_path.exists():
            downloaded.append({"path": str(saved_path), "file_name": ref, "hash": hash_name, "cached": True})
            continue

        if i > 0 and i % 8 == 0:
            await asyncio.sleep(2)

        try:
            data, mime = await get_resource_with_retry(client, ref)
            saved_path.write_bytes(data)
            downloaded.append({"path": str(saved_path), "file_name": ref, "hash": hash_name,
                               "mime": mime, "size": len(data), "cached": False})
            print(f"    [{i+1}/{len(all_image_refs)}] {hash_name[:20]}... => {mime}, {len(data)}B ✅")
        except Exception as e:
            downloaded.append({"path": None, "file_name": ref, "hash": hash_name, "error": str(e)[:200]})
            print(f"    [{i+1}/{len(all_image_refs)}] {hash_name[:20]}... => ERR {str(e)[:80]}")

    return downloaded, formulas, all_text[:3000]


def audit_single_image(item, formulas_text, text_context, idx, total):
    img_path = item.get("path")
    if not img_path or not os.path.exists(img_path):
        return {**item, "judgment": None, "error": "image_not_found"}
    try:
        raw, tokens = call_mimo_judge(img_path, formulas_text, text_context)
        judgment, full_raw = parse_judgment(raw)
        print(f"    [{idx+1}/{total}] {item['hash'][:20]}... => {judgment}")
        return {**item, "judgment": judgment, "raw_response": full_raw, "token_usage": tokens}
    except Exception as e:
        print(f"    [{idx+1}/{total}] {item['hash'][:20]}... => ERR {str(e)[:60]}")
        return {**item, "judgment": None, "error": str(e)[:200]}


def audit_paper_images(downloaded, formulas_text, text_context, workers=3):
    items = [d for d in downloaded if d.get("path")]
    if not items: return downloaded

    formulas_joined = "\n".join(formulas_text[:50])[:2000]
    total = len(items)
    print(f"  Auditing {total} images with MiMo-V2.5...")

    results = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(audit_single_image, item, formulas_joined, text_context, i, total): i
                   for i, item in enumerate(items)}
        for fut in as_completed(futures):
            results.append(fut.result())

    result_map = {r["hash"]: r for r in results}
    return [result_map.get(d["hash"], d) for d in downloaded]


async def main():
    print("=" * 60)
    print("D2: Sciverse Image Download + MiMo Visual Audit")
    print(f"MIMO_API_KEY: {'SET' if MIMO_API_KEY else 'MISSING'}")
    print(f"SCIVERSE_API_KEY: {'SET' if API_KEY else 'MISSING'}")
    print("=" * 60)
    print()

    with open(AUDIT_JSON, encoding="utf-8") as f:
        audit_data = json.load(f)

    client = AgentToolsClient(token=API_KEY)
    all_results = {}
    total_ok = 0
    total_fail = 0
    total_images = 0
    audit_counts = Counter()

    papers = list(audit_data["per_paper"].items())
    for pi, (label, paper) in enumerate(papers):
        doc_id = paper["doc_id"]
        print(f"\n--- [{pi+1}/{len(papers)}] {label} ---")

        downloaded, formulas, text_ctx = await download_images_for_paper(client, doc_id, label)

        ok = sum(1 for d in downloaded if d.get("path"))
        fail = sum(1 for d in downloaded if d.get("error"))
        total_ok += ok
        total_fail += fail
        total_images += len(downloaded)
        print(f"  downloaded={ok} failed={fail}")

        audited = audit_paper_images(downloaded, formulas, text_ctx)
        for a in audited:
            if a.get("judgment"):
                audit_counts[a["judgment"]] += 1

        all_results[label] = {
            "doc_id": doc_id,
            "title": paper.get("title", ""),
            "image_count_downloaded": ok,
            "image_count_failed": fail,
            "audit_summary": {j: sum(1 for a in audited if a.get("judgment") == j)
                              for j in ["consistent", "partial", "inconsistent", "uncertain"]},
            "images": audited,
        }

        with open(RESULTS_LOG, "a", encoding="utf-8") as f:
            for img in audited:
                f.write(json.dumps({**img, "paper": label}, ensure_ascii=False) + "\n")

        if pi < len(papers) - 1:
            await asyncio.sleep(4)

    audit_total = sum(audit_counts.values())
    consistent_pct = round(audit_counts["consistent"] / audit_total * 100, 1) if audit_total else 0

    report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "papers_total": len(all_results),
        "images_total": total_images,
        "images_downloaded": total_ok,
        "images_failed": total_fail,
        "images_audited": audit_total,
        "audit_summary": {
            "consistent": audit_counts["consistent"], "partial": audit_counts["partial"],
            "inconsistent": audit_counts["inconsistent"], "uncertain": audit_counts["uncertain"],
            "consistent_pct": consistent_pct,
        },
        "per_paper": all_results,
    }

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n{'=' * 60}")
    print("D2 COMPLETE")
    print(f"  Papers: {len(all_results)}")
    print(f"  Images: {total_ok} downloaded / {total_fail} failed / {total_images} total")
    print(f"  MiMo Audit: {audit_total} judged")
    print(f"    consistent: {audit_counts['consistent']} ({consistent_pct}%)")
    print(f"    partial:    {audit_counts['partial']}")
    print(f"    inconsistent: {audit_counts['inconsistent']}")
    print(f"    uncertain:  {audit_counts['uncertain']}")
    print(f"  Report: {REPORT_PATH}")
    print(f"  Log: {RESULTS_LOG}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    asyncio.run(main())
