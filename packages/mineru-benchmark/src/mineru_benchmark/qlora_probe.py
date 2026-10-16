"""D4: Sciverse PPL Probe — base vs QLoRA on Sciverse 14 papers.

Uses existing control-science QLoRA adapter to measure domain transfer:
base model → Sciverse text PPL → QLoRA adapter → Sciverse text PPL → delta.

Data: Sciverse 14 papers fetched via SDK read_content, formula-dense fragments.
"""
import asyncio
import json
import re
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

ROOT = Path(__file__).resolve().parents[4]

from sciverse import AgentToolsClient

SCIVERSE_API_KEY = __import__('os').environ.get("SCIVERSE_API_KEY", "")

AUDIT_JSON = ROOT / "benchmark" / "eval" / "results" / "sciverse_cross_modal_audit.json"
OUTPUT_PATH = ROOT / "benchmark" / "eval" / "results" / "sciverse_qlora_ppl.json"

MODEL_ID = "Qwen/Qwen3.5-9B"
ADAPTER_PATH = ROOT / "finetune" / "output" / "qlora-final"

SYSTEM_PROMPT = "你是控制科学专家。请准确、专业、完整地解读以下控制科学文献片段。"

FM_BLOCK = re.compile(r'\$\$(.+?)\$\$', re.DOTALL)
FM_INLINE = re.compile(r'(?<!\$)\$(?!\$)([^$]+?)(?<!\$)\$(?!\$)')

MIN_FRAGMENT_CHARS = 50
MAX_FRAGMENT_CHARS = 2000
MAX_FRAGMENTS_PER_PAPER = 8
QUANT_DTYPE = torch.bfloat16


def extract_formula_dense_fragments(text, max_frags=MAX_FRAGMENTS_PER_PAPER):
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    scored = []
    for p in paragraphs:
        p = re.sub(r'^#+\s+.*$', '', p, flags=re.MULTILINE).strip()
        p = re.sub(r'\[.*?\]\(.*?\)', '', p)
        p = re.sub(r'!\[.*?\]\(.*?\)', '', p)
        p = re.sub(r'\|.*?\|', '', p)
        p = p.strip()
        if len(p) < MIN_FRAGMENT_CHARS:
            continue
        block_count = len(FM_BLOCK.findall(p))
        inline_count = len(FM_INLINE.findall(p))
        score = block_count * 5 + inline_count * 2
        scored.append((score, p[:MAX_FRAGMENT_CHARS]))

    scored.sort(key=lambda x: -x[0])
    return [s[1] for s in scored[:max_frags]]


def load_model_and_tokenizer(model_id, adapter_path=None):
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_compute_dtype=QUANT_DTYPE,
        bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id, quantization_config=quant_config, device_map="auto",
        trust_remote_code=True, attn_implementation="eager", dtype=QUANT_DTYPE,
    ).eval()

    if adapter_path and adapter_path.exists():
        model = PeftModel.from_pretrained(model, str(adapter_path))
        model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def compute_ppl(model, tokenizer, text):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "请解读以下内容。"},
        {"role": "assistant", "content": text},
    ]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    inputs = tokenizer(formatted, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
    loss = outputs.loss.item()
    ppl = torch.exp(torch.tensor(loss)).item()
    return loss, ppl


async def load_sciverse_texts():
    client = AgentToolsClient(token=SCIVERSE_API_KEY)
    with open(AUDIT_JSON, encoding="utf-8") as f:
        audit = json.load(f)

    papers = {}
    paper_list = list(audit["per_paper"].items())
    for pi, (label, paper) in enumerate(paper_list):
        doc_id = paper["doc_id"]
        title = paper.get("title", "")
        all_text = ""
        offset = 0

        for chunk_i in range(5):
            for attempt in range(4):
                try:
                    content = await client.read_content(doc_id=doc_id, offset=offset, limit=6000)
                    break
                except Exception as e:
                    if "429" in str(e) and attempt < 3:
                        await asyncio.sleep((attempt + 1) * 4)
                        continue
                    raise
            all_text += content.get("text", "")
            if not content.get("more", False) or offset >= 30000:
                break
            offset = content.get("next_offset", offset + 6000)

        fragments = extract_formula_dense_fragments(all_text)
        if fragments:
            papers[label] = {"doc_id": doc_id, "title": title, "fragments": fragments}

        if pi < len(paper_list) - 1:
            await asyncio.sleep(3)
    return papers


def main():
    print("=" * 60)
    print("D4: Sciverse PPL Probe (base vs QLoRA)")
    print(f"  Model: {MODEL_ID}")
    print(f"  Adapter: {ADAPTER_PATH}")
    print("=" * 60)
    print()

    if not torch.cuda.is_available():
        print("[FATAL] CUDA not available")
        sys.exit(1)
    print(f"[GPU] {torch.cuda.get_device_name(0)}")

    print("\n[Data] Loading Sciverse texts...")
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    papers = loop.run_until_complete(load_sciverse_texts())
    loop.close()

    total_frags = sum(len(p["fragments"]) for p in papers.values())
    print(f"[Data] {len(papers)} papers, {total_frags} formula-dense fragments")

    print("\n[Base] Loading base model...")
    t0 = time.time()
    base_model, tokenizer = load_model_and_tokenizer(MODEL_ID)
    print(f"  Loaded in {time.time() - t0:.0f}s")

    base_results = {}
    print("\n[Base] Computing PPL...")
    t_start = time.time()
    for paper_name, paper in papers.items():
        ppls = []
        for fi, frag in enumerate(paper["fragments"]):
            loss, ppl = compute_ppl(base_model, tokenizer, frag)
            ppls.append(ppl)
        avg_ppl = sum(ppls) / len(ppls) if ppls else 0
        base_results[paper_name] = {"ppls": ppls, "avg_ppl": avg_ppl, "n": len(ppls)}
        print(f"  {paper_name}: {avg_ppl:.1f} ({len(ppls)} fragments)")

    base_overall = sum(r["avg_ppl"] for r in base_results.values()) / len(base_results)
    print(f"\n[Base] Overall avg_ppl={base_overall:.1f}, time={time.time()-t_start:.0f}s")

    del base_model
    torch.cuda.empty_cache()

    print("\n[QLoRA] Loading QLoRA adapter...")
    t0 = time.time()
    qlora_model, _ = load_model_and_tokenizer(MODEL_ID, adapter_path=ADAPTER_PATH)
    print(f"  Loaded in {time.time() - t0:.0f}s")

    qlora_results = {}
    print("\n[QLoRA] Computing PPL...")
    t_start = time.time()
    for paper_name, paper in papers.items():
        ppls = []
        for fi, frag in enumerate(paper["fragments"]):
            loss, ppl = compute_ppl(qlora_model, tokenizer, frag)
            ppls.append(ppl)
        avg_ppl = sum(ppls) / len(ppls) if ppls else 0
        qlora_results[paper_name] = {"ppls": ppls, "avg_ppl": avg_ppl, "n": len(ppls)}
        print(f"  {paper_name}: {avg_ppl:.1f} ({len(ppls)} fragments)")

    qlora_overall = sum(r["avg_ppl"] for r in qlora_results.values()) / len(qlora_results)
    print(f"\n[QLoRA] Overall avg_ppl={qlora_overall:.1f}, time={time.time()-t_start:.0f}s")

    delta = qlora_overall - base_overall
    delta_pct = (delta / base_overall * 100) if base_overall else 0

    per_paper = {}
    for name in papers:
        bp = base_results[name]["avg_ppl"]
        qp = qlora_results[name]["avg_ppl"]
        d = qp - bp
        per_paper[name] = {
            "base_ppl": round(bp, 2),
            "qlora_ppl": round(qp, 2),
            "delta_ppl": round(d, 2),
            "delta_pct": round(d / bp * 100, 1) if bp else 0,
            "n_fragments": base_results[name]["n"],
        }

    report = {
        "analysis": "sciverse_qlora_ppl_probe",
        "base_model": MODEL_ID,
        "qlora_adapter": str(ADAPTER_PATH),
        "n_papers": len(papers),
        "total_fragments": total_frags,
        "overall": {
            "base_avg_ppl": round(base_overall, 2),
            "qlora_avg_ppl": round(qlora_overall, 2),
            "delta_ppl": round(delta, 2),
            "delta_percent": round(delta_pct, 1),
        },
        "per_paper": per_paper,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n{'=' * 60}")
    print("D4 COMPLETE")
    print(f"  Base avg_ppl:  {base_overall:.1f}")
    print(f"  QLoRA avg_ppl: {qlora_overall:.1f}")
    print(f"  Δ ppl:         {delta:+.1f} ({delta_pct:+.1f}%)")
    print(f"  {'✓ Domain adaptation' if delta < 0 else '⚠ Base better — unexpected'}")
    print(f"  Report: {OUTPUT_PATH}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
