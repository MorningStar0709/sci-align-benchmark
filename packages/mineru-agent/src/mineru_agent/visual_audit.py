"""跨模态对齐质量审计引擎 — 全量批量版

核心流程：科学文档图片 × LaTeX 公式 → 多模态 LLM 判决 → 输出一致性审计报告

Provider 选择（环境变量 VISION_PROVIDER）：
  - mimo（默认）：MiMo-V2.5 原生全模态 API — 已验证科学图片理解
    - 模型: mimo-v2.5（"原生全模态"）
    - 注意: 必须用 api-key header 认证，不能用 OpenAI SDK
    - mimo-v2.5-pro 不支持图像输入（返回 404）
  - minimax：MiniMax Anthropic API — 已验证兜底方案
  - 两者可交叉验证，形成双 Judge 一致性审计

验证历史（2026-05-08）：
  - MiMo-V2.5 (mimo-v2.5): 4.7s, 77in/317out, 40 图像 token — ✅ 正确识别控制公式
  - MiMo-V2-Omni: 5.1s, 77in/361out, 40 图像 token — ✅ 同 V2.5
  - MiniMax: cross_modal_audit.py 已验证 29 样本，综合对齐质量 79.3%

环境变量：
  MIMO_API_KEY — MiMo API Key（主视觉引擎）
  MINIMAX_API_KEY — MiniMax API Key（兜底）
  VISION_PROVIDER — mimo | minimax（默认 mimo）

用法：
  .venv/bin/python benchmark/agent/visual_audit.py --workers 5 --resume
  .venv/bin/python benchmark/agent/visual_audit.py --test          # 单图测试
  .venv/bin/python benchmark/agent/visual_audit.py --max-items 100 # 限量调试
"""

import argparse
import base64
import json
import os
import re
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import threading

import httpx

ROOT = Path(__file__).resolve().parents[4]  # restored Phase 5

import mineru_core  # noqa: F401 — trigger _load_dotenv()
from mineru_core.settings import PROVIDERS

from mineru_eval.utils import write_json_atomic

MIMO_BASE_URL = PROVIDERS["mimo"].base_url
MIMO_MODEL = "mimo-v2.5"

MINIMAX_BASE_URL = PROVIDERS["minimax"].base_url
MINIMAX_MODEL = "MiniMax-M2.5-highspeed"

_vision_provider = None
_mimo_api_key = None
_minimax_api_key = None


def _get_vision_provider():
    global _vision_provider
    if _vision_provider is None:
        _vision_provider = os.environ.get("VISION_PROVIDER", "mimo")
    return _vision_provider


def _get_mimo_api_key():
    global _mimo_api_key
    if _mimo_api_key is None:
        _mimo_api_key = os.environ.get("MIMO_API_KEY", "")
    return _mimo_api_key


def _get_minimax_api_key():
    global _minimax_api_key
    if _minimax_api_key is None:
        _minimax_api_key = os.environ.get("MINIMAX_API_KEY", "")
    return _minimax_api_key

IMG_PATTERN = re.compile(r'!\[\]\((?:images|image)/([a-f0-9]+\.jpg)\)')
FM_BLOCK = re.compile(r'\$\$(.+?)\$\$', re.DOTALL)
FM_INLINE = re.compile(r'(?<!\$)\$(?!\$)([^$]+?)(?<!\$)\$(?!\$)')

DATA_PROCESSED = ROOT / "data" / "processed"
CHUNKS_DIR = ROOT / "corpus" / "chunks"
MANIFEST_PATH = ROOT / "corpus" / "chunks" / "chunks_manifest.json"
OUTPUT_DIR = ROOT / "benchmark" / "agent" / "results"
CHECKPOINT_FILE = OUTPUT_DIR / "visual_audit_checkpoint.json"
RESULTS_LOG = OUTPUT_DIR / "visual_audit_results.jsonl"
REPORT_FILE = OUTPUT_DIR / "visual_audit_report.json"
CHECKPOINT_INTERVAL = 50

_manifest = None
_thread_local = threading.local()


def _get_shared_client():
    if not hasattr(_thread_local, 'httpx_client'):
        _thread_local.httpx_client = httpx.Client(proxy=None, timeout=120)
    return _thread_local.httpx_client

VISION_JUDGE_SYSTEM = "你是一个专业的跨模态对齐审计助手。判断科学文档图片与 LaTeX 公式之间的语义相关性。严格基于图片中的视觉内容做判断，不要臆想图片中不存在的内容。"

VISION_JUDGE_USER_TEMPLATE = (
    "你是一个控制科学领域的跨模态对齐评审专家，负责判断PDF中提取的图片与LaTeX公式之间的语义相关性。\n\n"
    "以下是同一chunk中提取的：\n"
    "(1) 一张图片（控制工程/科学相关的图表、框图、仿真曲线等）\n"
    "(2) LaTeX 公式代码\n\n"
    "请判断：这张图片的**核心内容**（不是图片中是否有文字形式的公式，"
    "而是图片展示的数学模型/系统行为/控制结构）是否与 LaTeX 公式描述的数学语义相关？\n\n"
    "换句话说：这张图片和这些公式是否在描述同一个控制系统的相关方面？\n\n"
    "--- LaTeX 公式 ---\n{formulas}\n\n"
    "--- 文本上下文（前200字） ---\n{text_context}\n\n"
    "请用以下严格格式回答，不要添加多余内容：\n"
    "判断: [一致 | 部分一致 | 不一致 | 不确定]\n"
    "理由: [10-30字的判断依据]"
)


def image_to_base64(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def extract_images(text):
    return IMG_PATTERN.findall(text)


def extract_formulas(text):
    scan = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
    formulas = []
    for m in FM_BLOCK.findall(scan):
        formulas.append(("block", m.strip()))
    for m in FM_INLINE.findall(scan):
        formulas.append(("inline", m.strip()))
    return formulas


def resolve_image_path(filename, hash_name):
    return DATA_PROCESSED / filename / "image" / hash_name


def call_vision_judge_mimo(image_b64, formulas_text, chunk_text_preview):
    user_prompt = VISION_JUDGE_USER_TEMPLATE.format(
        formulas=formulas_text,
        text_context=chunk_text_preview[:400],
    )
    payload = {
        "model": MIMO_MODEL,
        "messages": [
            {"role": "system", "content": VISION_JUDGE_SYSTEM},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_b64}",
                            "detail": "high",
                        },
                    },
                ],
            },
        ],
        "temperature": 1.0,
        "max_tokens": 512,
        "thinking": {"type": "disabled"},
    }
    headers = {
        "api-key": _get_mimo_api_key(),
        "Content-Type": "application/json",
    }
    client = _get_shared_client()
    resp = client.post(
        f"{MIMO_BASE_URL}/chat/completions",
        json=payload,
        headers=headers,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"MiMo API HTTP {resp.status_code}: {resp.text[:200]}")

    data = resp.json()
    content = data["choices"][0]["message"]["content"]
    if content is None:
        raise RuntimeError("MiMo API returned content=None")

    usage = data.get("usage", {})
    details = usage.get("prompt_tokens_details", {})
    tokens = {
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
        "image_tokens": details.get("image_tokens", 0),
    }
    return content.strip(), tokens


def call_vision_judge_minimax(image_b64, formulas_text, chunk_text_preview):
    from anthropic import Anthropic
    user_prompt = VISION_JUDGE_USER_TEMPLATE.format(
        formulas=formulas_text,
        text_context=chunk_text_preview[:400],
    )
    client = Anthropic(
        api_key=_get_minimax_api_key(),
        base_url=MINIMAX_BASE_URL,
        timeout=120,
        http_client=httpx.Client(proxy=None, trust_env=False),
    )
    resp = client.messages.create(
        model=MINIMAX_MODEL,
        max_tokens=512,
        system=VISION_JUDGE_SYSTEM,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": user_prompt},
                {"type": "image", "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": image_b64,
                }},
            ],
        }],
    )
    text = ""
    for block in resp.content:
        if hasattr(block, 'text') and block.text:
            text += block.text
    tokens = {}
    if hasattr(resp, 'usage') and resp.usage:
        tokens = {
            "input_tokens": getattr(resp.usage, 'input_tokens', 0),
            "output_tokens": getattr(resp.usage, 'output_tokens', 0),
        }
    return text.strip(), tokens


def parse_judgment(response_text):
    if not response_text:
        return "uncertain"
    response_lower = response_text.lower().strip()
    if "判断: 一致" in response_lower or "判断:一致" in response_lower:
        return "consistent"
    elif "判断: 部分一致" in response_lower or "判断:部分一致" in response_lower:
        return "partially_consistent"
    elif "判断: 不一致" in response_lower or "判断:不一致" in response_lower:
        return "inconsistent"
    elif "判断: 不确定" in response_lower or "判断:不确定" in response_lower:
        return "uncertain"
    if "一致" in response_lower and "不一致" not in response_lower:
        return "consistent"
    if "部分一致" in response_lower:
        return "partially_consistent"
    if "不一致" in response_lower:
        return "inconsistent"
    return "uncertain"


def audit_single_image(image_path, formulas_text, chunk_text_preview, provider=None):
    provider = provider or _get_vision_provider()
    image_b64 = image_to_base64(image_path)

    if provider == "mimo":
        if not _get_mimo_api_key():
            raise ValueError("MIMO_API_KEY not set")
        response, token_usage = call_vision_judge_mimo(
            image_b64, formulas_text, chunk_text_preview
        )
    elif provider == "minimax":
        if not _get_minimax_api_key():
            raise ValueError("MINIMAX_API_KEY not set")
        response, token_usage = call_vision_judge_minimax(
            image_b64, formulas_text, chunk_text_preview
        )
    else:
        raise ValueError(f"Unknown provider: {provider}")

    judgment = parse_judgment(response)
    return {
        "provider": provider,
        "model": MIMO_MODEL if provider == "mimo" else MINIMAX_MODEL,
        "judgment": judgment,
        "raw_response": response,
        "token_usage": token_usage,
    }


def audit_from_chunk(chunk_id, filename, image_hash, formulas_text, text_preview,
                     provider=None, doc_id=None):
    image_path = resolve_image_path(filename, image_hash)
    if not image_path.exists():
        return {"chunk_id": chunk_id, "error": f"image not found: {image_path}"}
    result = audit_single_image(image_path, formulas_text, text_preview, provider=provider)
    result["chunk_id"] = chunk_id
    result["filename"] = filename
    result["image_hash"] = image_hash
    result["image_path"] = str(image_path)
    if doc_id:
        result["doc_id"] = doc_id
    return result


def load_manifest():
    global _manifest
    if _manifest is None:
        with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
            _manifest = json.load(f)
    return _manifest


def scan_work_items(max_items=None):
    manifest = load_manifest()
    chunks = manifest["chunks"]
    total = len(chunks)

    print(f"\n  扫描 {total} 个 chunks 中（寻找图片+公式共现项）...")
    t0 = time.time()

    work_items = []
    scan_errors = 0

    for i, chunk in enumerate(chunks):
        if (i + 1) % 2000 == 0:
            elapsed = time.time() - t0
            print(f"    扫描进度: {i+1}/{total} ({elapsed:.0f}s)", flush=True)

        filepath = CHUNKS_DIR / chunk["filepath"].replace("corpus/chunks/", "")
        try:
            content = filepath.read_text(encoding="utf-8")
        except (FileNotFoundError, UnicodeDecodeError):
            scan_errors += 1
            continue

        images = extract_images(content)
        if not images:
            continue

        formulas = extract_formulas(content)
        if not formulas:
            continue

        formulas_text = "\n".join(f"{t}: ${f}$" for t, f in formulas[:10])
        text_preview = re.sub(r'!\[\]\([^)]+\)', '', content)[:400]

        filename = chunk["filename"]
        doc_id = chunk.get("doc_id", "?")

        for img_hash in images:
            work_items.append({
                "chunk_id": chunk["chunk_id"],
                "filename": filename,
                "doc_id": doc_id,
                "image_hash": img_hash,
                "formulas_text": formulas_text,
                "text_preview": text_preview,
            })

            if max_items and len(work_items) >= max_items:
                break

        if max_items and len(work_items) >= max_items:
            break

    scan_time = time.time() - t0
    print(f"  扫描完成: {len(work_items)} 个审计项, 扫描耗时 {scan_time:.0f}s")
    if scan_errors:
        print(f"  扫描读取错误: {scan_errors} 个（已跳过）")

    return work_items


def _save_checkpoint(processed_count, total_count, summary):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = {
        "checkpoint_time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "processed": processed_count,
        "total": total_count,
        "summary": summary,
    }
    write_json_atomic(CHECKPOINT_FILE, data)


def _load_checkpoint():
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def _append_result(result):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")


def _load_results_log():
    if not RESULTS_LOG.exists():
        return [], set()
    results = []
    seen = set()
    with open(RESULTS_LOG, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            results.append(r)
            cid = r.get("chunk_id", "")
            ih = r.get("image_hash", "")
            if cid and ih:
                seen.add((cid, ih))
    return results, seen


def _compute_summary(results):
    judgments = [r["judgment"] for r in results if "judgment" in r]
    errors = [r for r in results if "error" in r]
    if not judgments:
        return {"total_audited": len(results), "error_count": len(errors)}
    dist = Counter(judgments)
    total = len(judgments)
    return {
        "total_audited": total,
        "consistent": dist.get("consistent", 0),
        "consistent_pct": round(dist.get("consistent", 0) / total * 100, 1),
        "partially_consistent": dist.get("partially_consistent", 0),
        "partially_consistent_pct": round(dist.get("partially_consistent", 0) / total * 100, 1),
        "inconsistent": dist.get("inconsistent", 0),
        "inconsistent_pct": round(dist.get("inconsistent", 0) / total * 100, 1),
        "uncertain": dist.get("uncertain", 0),
        "uncertain_pct": round(dist.get("uncertain", 0) / total * 100, 1),
        "error_count": len(errors),
    }


def run_batch_audit(work_items, workers=5, resume=False):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if resume:
        cp = _load_checkpoint()
        existing_results, processed_set = _load_results_log()
        if cp and existing_results:
            remaining = [w for w in work_items if (w["chunk_id"], w["image_hash"]) not in processed_set]
            print(f"\n  断点续跑: 已处理 {cp['processed']} 个, 剩余 {len(remaining)} 个")
            work_items = remaining
            start_index = cp["processed"]
            completed_results = existing_results
        else:
            print("\n  未找到 checkpoint 或结果日志，从头开始")
            completed_results = []
            start_index = 0
    else:
        completed_results = []
        start_index = 0

    total = start_index + len(work_items)
    print(f"\n  开始审计: {len(work_items)} 个图片-公式共现项, {workers} 线程")
    print(f"  Provider: {_get_vision_provider()}")
    print(f"  {'=' * 50}")

    t_start = time.time()
    done_count = start_index

    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_map = {}
        for item in work_items:
            future = executor.submit(
                audit_from_chunk,
                item["chunk_id"],
                item["filename"],
                item["image_hash"],
                item["formulas_text"],
                item["text_preview"],
                None,
                item.get("doc_id"),
            )
            future_map[future] = item

        for future in as_completed(future_map):
            item = future_map[future]
            done_count += 1
            try:
                result = future.result()
            except Exception as exc:
                result = {
                    "chunk_id": item["chunk_id"],
                    "image_hash": item["image_hash"],
                    "error": str(exc),
                }

            completed_results.append(result)
            _append_result(result)

            elapsed = time.time() - t_start
            rate = done_count / (elapsed / 60) if elapsed > 0 else 0
            eta_min = (total - done_count) / rate if rate > 0 else 0

            judgment = result.get("judgment", "ERROR")
            j_emoji = {"consistent": "✅", "partially_consistent": "🟡",
                       "inconsistent": "❌", "uncertain": "⚠️"}.get(judgment, "❓")

            if done_count % 50 == 0 or done_count <= 5 or done_count > total - 5:
                print(
                    f"  [{done_count}/{total}] {j_emoji} {judgment:<20}"
                    f" | {elapsed:.0f}s | {rate:.0f} 项/min"
                    f" | ETA {eta_min:.0f}min"
                    f" | {item['chunk_id'][:40]}...",
                    flush=True,
                )

            if done_count % CHECKPOINT_INTERVAL == 0 or done_count >= total:
                summary = _compute_summary(completed_results)
                _save_checkpoint(done_count, total, summary)

    total_time = time.time() - t_start
    summary = _compute_summary(completed_results)
    print(f"\n{'=' * 64}")
    print("  全量审计完成")
    print(f"  总处理: {total} 项")
    print(f"  总耗时: {total_time:.0f}s ({total_time/60:.1f}min)")
    print(f"  平均:   {total/total_time:.1f} 项/s" if total_time > 0 else "  平均:   N/A")
    if "consistent_pct" in summary:
        print(f"  一致:    {summary['consistent']} ({summary['consistent_pct']}%)")
        print(f"  部分一致: {summary['partially_consistent']} ({summary['partially_consistent_pct']}%)")
        print(f"  不一致:  {summary['inconsistent']} ({summary['inconsistent_pct']}%)")
        print(f"  不确定:  {summary['uncertain']} ({summary['uncertain_pct']}%)")
        print(f"  错误:    {summary['error_count']}")
    print(f"{'=' * 64}")

    report = {
        "report_time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "provider": _get_vision_provider(),
        "model": MIMO_MODEL if _get_vision_provider() == "mimo" else MINIMAX_MODEL,
        "total_work_items": total,
        "total_audited": summary.get("total_audited", 0),
        "elapsed_seconds": total_time,
        "summary": summary,
        "result_log": str(RESULTS_LOG),
    }

    write_json_atomic(REPORT_FILE, report)
    print(f"  报告已保存: {REPORT_FILE}")

    return report


def run_test():
    print("=" * 64)
    print("  跨模态对齐质量审计引擎 — 快速测试模式")
    provider = _get_vision_provider()
    print(f"  Provider: {provider}")
    if provider == "mimo":
        print(f"  MiMo:    {MIMO_MODEL}（原生全模态 ✅）")
    else:
        print(f"  MiniMax: {MINIMAX_MODEL}")
    print("=" * 64)

    if provider == "mimo" and not _get_mimo_api_key():
        print("ERROR: MIMO_API_KEY not set.")
        sys.exit(1)
    if provider == "minimax" and not _get_minimax_api_key():
        print("ERROR: MINIMAX_API_KEY not set.")
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    test_image = None
    for proc_dir in sorted(DATA_PROCESSED.iterdir())[:5]:
        img_dir = proc_dir / "image"
        if img_dir.exists():
            jpgs = sorted(img_dir.glob("*.jpg"))
            if jpgs:
                test_image = jpgs[0]
                break

    if not test_image:
        print("ERROR: No test image found.")
        sys.exit(1)

    test_formulas = (
        "block: $G_{\\mathrm{p}}(s) = \\frac{133}{s^{2} + 25s}$\n"
        "inline: $y_{\\mathrm{d}}(k)=1.0$\n"
        "inline: $k_{p}$\n"
        "inline: $k_{i}$\n"
    )
    test_context = (
        "被控对象为 G_p(s) = 133/(s^2+25s)。"
        "采用模糊PID控制器，输入为误差e(k)和误差变化率ec(k)。"
    )

    print(f"\n测试图片: {test_image.name}")
    print(f"[{_get_vision_provider().upper()}] 跨模态对齐审计测试...")

    t0 = time.time()
    result = audit_single_image(test_image, test_formulas, test_context)
    elapsed = time.time() - t0

    tu = result["token_usage"]
    if _get_vision_provider() == "mimo":
        tok_str = f"{tu.get('prompt_tokens',0)} in / {tu.get('completion_tokens',0)} out (图: {tu.get('image_tokens',0)} tok)"
    else:
        tok_str = f"{tu.get('input_tokens',0)} in / {tu.get('output_tokens',0)} out"

    print(f"  耗时: {elapsed:.1f}s")
    print(f"  Token: {tok_str}")
    print(f"  判决: {result['judgment']}")
    print(f"  理由: {result['raw_response'][:200]}")

    print(f"\n{'=' * 64}")
    print("  审计测试完成")
    print(f"  图片: {test_image.name}")
    print(f"  Provider: {_get_vision_provider()}")
    print(f"  判决: {result['judgment']}")
    print(f"  耗时: {elapsed:.1f}s")
    print(f"{'=' * 64}")


def main():
    global OUTPUT_DIR, CHECKPOINT_FILE, RESULTS_LOG, REPORT_FILE

    parser = argparse.ArgumentParser(
        description="跨模态对齐质量审计引擎 — 科学文档图片 × LaTeX 公式一致性审计"
    )
    parser.add_argument("--test", action="store_true", help="单图片快速测试模式")
    parser.add_argument("--workers", type=int, default=5, help="并发线程数（默认 5）")
    parser.add_argument("--resume", action="store_true", help="从 checkpoint 断点续跑")
    parser.add_argument("--max-items", type=int, default=None, help="最大审计项数（调试用）")
    parser.add_argument("--provider", default=None, help="mimo | minimax（覆盖环境变量 VISION_PROVIDER）")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="审计结果输出目录")
    args = parser.parse_args()

    OUTPUT_DIR = Path(args.output_dir)
    CHECKPOINT_FILE = OUTPUT_DIR / "visual_audit_checkpoint.json"
    RESULTS_LOG = OUTPUT_DIR / "visual_audit_results.jsonl"
    REPORT_FILE = OUTPUT_DIR / "visual_audit_report.json"

    if args.provider:
        os.environ["VISION_PROVIDER"] = args.provider

    provider = _get_vision_provider()

    if args.test:
        run_test()
        return

    if provider == "mimo" and not _get_mimo_api_key():
        print("ERROR: MIMO_API_KEY not set. 请设置环境变量或使用 --test 模式验证 MiMo 视觉可用。")
        sys.exit(1)

    work_items = scan_work_items(max_items=args.max_items)
    if not work_items:
        print("未找到含图片+公式的 chunk，无可审计项。")
        return

    print(f"\n  扫描到 {len(work_items)} 个审计项")
    print(f"  使用 {args.workers} 线程, Provider: {provider}")

    run_batch_audit(work_items, workers=args.workers, resume=args.resume)


if __name__ == "__main__":
    main()
