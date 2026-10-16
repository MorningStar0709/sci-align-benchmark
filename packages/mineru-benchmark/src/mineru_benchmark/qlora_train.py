"""D4b: Sciverse QLoRA Training Pipeline.

1. Extract formula-dense fragments from Sciverse 14 papers (reuse D4a text cache via SDK)
2. Format as ChatML (teacher-forcing: system→user query→fragment as assistant)
3. QLoRA train (r=8, 1 epoch, ~10min GPU)
4. PPL probe: base vs Sciverse-QLoRA

Usage:
    python -m benchmark.sciverse_qlora_train
"""
import asyncio
import argparse
import json
import os
import random
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]  # restored Phase 5

from sciverse import AgentToolsClient

AUDIT_JSON = ROOT / "benchmark" / "eval" / "results" / "sciverse_cross_modal_audit.json"
TRAIN_DIR = ROOT / "finetune" / "data" / "sciverse"
CONFIG_TEMPLATE = ROOT / "finetune" / "config" / "qlora_medical_4b.yaml"
CONFIG_OUT = ROOT / "finetune" / "config" / "qlora_sciverse.yaml"
OUTPUT_DIR = ROOT / "finetune" / "output" / "qlora-sciverse"
PPL_OUTPUT = ROOT / "benchmark" / "eval" / "results" / "sciverse_qlora_train_ppl.json"

SEED = 42
SYSTEM_PROMPT = "你是控制科学专家。请准确、专业、完整地解读以下控制科学文献段落。"
USER_QUERY = "请详细分析以下文献段落，逐条解释其中涉及的数学公式、变量定义、物理含义和控制方法。"
FM_BLOCK = re.compile(r'\$\$(.+?)\$\$', re.DOTALL)
FM_INLINE = re.compile(r'(?<!\$)\$(?!\$)([^$]+?)(?<!\$)\$(?!\$)')
SCIVERSE_API_KEY = os.environ.get("SCIVERSE_API_KEY", "")


def extract_fragments():
    print("[Step 1/4] Extracting Sciverse formula-dense fragments...")
    client = AgentToolsClient(token=SCIVERSE_API_KEY)
    with open(AUDIT_JSON, encoding="utf-8") as f:
        audit = json.load(f)

    async def _pull():
        fragments = []
        paper_list = list(audit["per_paper"].items())
        for pi, (label, paper) in enumerate(paper_list):
            doc_id = paper["doc_id"]
            print(f"  Pulling {label} ({pi+1}/{len(paper_list)})...", end=" ", flush=True)
            all_text = ""
            offset = 0
            try:
                for _ in range(4):
                    for attempt in range(4):
                        try:
                            content = await client.read_content(doc_id=doc_id, offset=offset, limit=6000)
                            break
                        except Exception as e:
                            if "429" in str(e) and attempt < 3:
                                await asyncio.sleep((attempt + 1) * 3)
                                continue
                            raise
                    all_text += content.get("text", "")
                    if not content.get("more", False) or offset >= 24000:
                        break
                    offset = content.get("next_offset", offset + 6000)
                print(f"{len(all_text)} chars")
            except Exception as e:
                print(f"FAILED: {e}")
                continue
            if pi < len(paper_list) - 1:
                await asyncio.sleep(2)

            paragraphs = [p.strip() for p in all_text.split("\n\n") if p.strip()]
            for p in paragraphs:
                p = re.sub(r'^#+\s+.*$', '', p, flags=re.MULTILINE).strip()
                p = re.sub(r'\[.*?\]\(.*?\)', '', p)
                p = re.sub(r'!\[.*?\]\(.*?\)', '', p)
                p = re.sub(r'\|.*?\|', '', p)
                p = p.strip()
                if len(p) < 80:
                    continue
                fragments.append({"source": label, "text": p[:2000]})
        return fragments

    return asyncio.run(_pull())


def generate_chatml(fragments):
    print(f"\n[Step 2/4] Formatting {len(fragments)} fragments as ChatML...")
    random.seed(SEED)
    random.shuffle(fragments)

    items = []
    for fi, frag in enumerate(fragments):
        items.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_QUERY},
                {"role": "assistant", "content": frag["text"]},
            ],
            "_meta": {"source": frag["source"], "char_count": len(frag["text"])},
        })

    n = len(items)
    n_train = int(n * 0.85)
    train = items[:n_train]
    val = items[n_train:]

    TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    with open(TRAIN_DIR / "train.jsonl", "w", encoding="utf-8") as f:
        for item in train:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    with open(TRAIN_DIR / "val.jsonl", "w", encoding="utf-8") as f:
        for item in val:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"  train: {len(train)} items, val: {len(val)} items")
    return n_train, n


def write_config():
    print("\n[Step 3/4] Writing QLoRA config...")
    config = {
        "model": {"name": "principled-intelligence/Qwen3.5-4B-text-only",
                  "trust_remote_code": True, "attn_implementation": "eager"},
        "bitsandbytes": {"load_in_4bit": True, "bnb_4bit_compute_dtype": "bfloat16",
                         "bnb_4bit_quant_type": "nf4", "bnb_4bit_use_double_quant": True},
        "lora": {"r": 8, "lora_alpha": 16, "target_modules": "all-linear",
                 "lora_dropout": 0.05, "bias": "none", "task_type": "CAUSAL_LM",
                 "init_lora_weights": "gaussian"},
        "training": {"output_dir": "finetune/output/qlora-sciverse",
                     "per_device_train_batch_size": 2, "gradient_accumulation_steps": 4,
                     "num_train_epochs": 1, "learning_rate": 2.0e-4,
                     "logging_steps": 10, "save_steps": 50, "eval_steps": 50,
                     "save_total_limit": 2, "warmup_ratio": 0.1,
                     "lr_scheduler_type": "cosine", "bf16": True,
                     "gradient_checkpointing": True, "gradient_checkpointing_kwargs": {"use_reentrant": True},
                     "dataloader_num_workers": 2, "remove_unused_columns": False,
                     "max_grad_norm": 1.0, "optim": "paged_adamw_8bit",
                     "torch_compile": False, "seed": 42},
        "data": {"train_file": "finetune/data/sciverse/train.jsonl",
                 "val_file": "finetune/data/sciverse/val.jsonl",
                 "max_seq_length": 2048},
    }
    CONFIG_OUT.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  Config: {CONFIG_OUT}")


def run_training():
    print("\n[Step 4/4] Running QLoRA training...")
    train_script = ROOT / "finetune" / "scripts" / "train_qlora.py"
    cmd = [sys.executable, "-u", str(train_script), "--config", str(CONFIG_OUT), "--force"]
    print(f"  {' '.join(cmd)}")
    t0 = time.time()
    result = subprocess.run(cmd, cwd=str(ROOT),
                            env={**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"})
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"  TRAINING FAILED (exit code {result.returncode}, {elapsed:.0f}s)")
        sys.exit(1)
    print(f"  Training complete ({elapsed:.0f}s)")


def _compute_ppl(tokenizer, model, user_msg, expected_answer, system_prompt):
    """Compute per-token loss and perplexity for a single example."""
    import torch
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_msg},
        {"role": "assistant", "content": expected_answer},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
    loss = outputs.loss.item()
    return loss, torch.exp(torch.tensor(loss)).item()


def _load_model_and_eval(model_id, quant_config, tokenizer, test_items):
    """Load model and compute PPL on all test items."""
    import torch
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        model_id, quantization_config=quant_config, device_map="auto",
        trust_remote_code=True, attn_implementation="eager", dtype=torch.bfloat16).eval()
    ppls = []
    for item in test_items:
        u_msg = next((m["content"] for m in item["messages"] if m["role"] == "user"), "")
        a_msg = next((m["content"] for m in item["messages"] if m["role"] == "assistant"), "")
        if u_msg and a_msg:
            _, ppl = _compute_ppl(tokenizer, model, u_msg, a_msg, SYSTEM_PROMPT)
            ppls.append(ppl)
    avg = sum(ppls) / len(ppls) if ppls else 0
    print(f"    avg_ppl={avg:.1f}")
    del model
    torch.cuda.empty_cache()
    return ppls, avg



def run_ppl_probe():
    print("\n[PPL Probe] base vs Sciverse-QLoRA...")
    import torch
    from transformers import AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel

    MODEL_ID = "principled-intelligence/Qwen3.5-4B-text-only"

    # Find best checkpoint
    ckpt_dir = OUTPUT_DIR
    if (ckpt_dir / "checkpoint-100").exists():
        ckpt_dir = ckpt_dir / "checkpoint-100"
    for ckpt in sorted(ckpt_dir.glob("checkpoint-*"), key=lambda p: -int(p.name.split("-")[1])):
        if (ckpt / "adapter_config.json").exists():
            ckpt_dir = ckpt
            break
    print(f"  Adapter: {ckpt_dir}")

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    val_path = TRAIN_DIR / "val.jsonl"
    test_items = [json.loads(l) for l in val_path.read_text(encoding="utf-8").strip().split("\n")]

    # Base PPL
    print("  Base model...")
    base_ppls, avg_base = _load_model_and_eval(MODEL_ID, quant_config, tokenizer, test_items)

    # QLoRA PPL
    print("  Sciverse-QLoRA model...")
    import transformers
    qlora_base = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL_ID, quantization_config=quant_config, device_map="auto",
        trust_remote_code=True, attn_implementation="eager", dtype=torch.bfloat16).eval()
    qlora_model = PeftModel.from_pretrained(qlora_base, str(ckpt_dir))
    qlora_model.eval()
    qlora_ppls = []
    for item in test_items:
        u_msg = next((m["content"] for m in item["messages"] if m["role"] == "user"), "")
        a_msg = next((m["content"] for m in item["messages"] if m["role"] == "assistant"), "")
        if u_msg and a_msg:
            _, ppl = _compute_ppl(tokenizer, qlora_model, u_msg, a_msg, SYSTEM_PROMPT)
            qlora_ppls.append(ppl)
    avg_qlora = sum(qlora_ppls) / len(qlora_ppls) if qlora_ppls else 0
    print(f"    avg_ppl={avg_qlora:.1f}")

    delta = avg_qlora - avg_base
    delta_pct = (delta / avg_base * 100) if avg_base else 0

    report = {
        "analysis": "sciverse_qlora_train_ppl",
        "base_model": MODEL_ID,
        "qlora_adapter": str(ckpt_dir),
        "train_items": len(test_items),
        "overall": {"base_avg_ppl": round(avg_base, 2), "qlora_avg_ppl": round(avg_qlora, 2),
                    "delta_ppl": round(delta, 2), "delta_percent": round(delta_pct, 1)},
        "base_ppls": [round(p, 2) for p in base_ppls],
        "qlora_ppls": [round(p, 2) for p in qlora_ppls],
    }
    PPL_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    PPL_OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n{'=' * 60}")
    print("D4b COMPLETE")
    print(f"  Base avg_ppl:  {avg_base:.1f}")
    print(f"  QLoRA avg_ppl: {avg_qlora:.1f}")
    print(f"  Δ ppl:         {delta:+.1f} ({delta_pct:+.1f}%)")
    print(f"  {'✓ Domain adaptation' if delta < 0 else '⚠ Base better'}")
    print(f"  Report: {PPL_OUTPUT}")
    print(f"{'=' * 60}")

    del qlora_model
    torch.cuda.empty_cache()


def main():
    print("=" * 60)
    print("D4b: Sciverse QLoRA Training Pipeline")
    print(f"  System: {SYSTEM_PROMPT[:40]}...")
    print(f"  Data: {AUDIT_JSON}")
    print("=" * 60)
    print()

    if not torch_available():
        return

    fragments = extract_fragments()
    n_train, n_total = generate_chatml(fragments)
    write_config()
    run_training()
    run_ppl_probe()


def torch_available():
    try:
        import torch
        if not torch.cuda.is_available():
            print("[FATAL] CUDA not available")
            return False
        print(f"[GPU] {torch.cuda.get_device_name(0)}")
        return True
    except ImportError:
        print("[FATAL] torch not installed")
        return False



def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QLoRA 训练（本仓仅保留历史实现，训练权威入口在 Kiln）")
    parser.add_argument("--config", default=None, help="训练配置 YAML 路径")
    parser.add_argument("--dry-run", action="store_true", help="只解析配置与数据，不启动训练")
    return parser.parse_args()


if __name__ == "__main__":
    _parse_args()
    main()
